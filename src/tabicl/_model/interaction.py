from __future__ import annotations

from typing import Optional
from functools import partial
from collections import OrderedDict

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint

from .encoders import Encoder
from .inference import InferenceManager
from .inference_config import MgrConfig, InferenceConfig


class RowInteraction(nn.Module):
    """Context-aware row-wise interaction.

    This module captures interactions between features within each row using a transformer
    encoder with rotary positional encoding. It prepends learnable class tokens to the
    learned feature embeddings and uses these tokens to aggregate information.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.

    num_blocks : int
        Number of blocks used in the encoder.

    nhead : int
        Number of attention heads of the encoder.

    dim_feedforward : int
        Dimension of the feedforward network of the encoder.

    num_cls : int, default=4
        Number of learnable CLS tokens to prepend to the feature embeddings. The outputs
        of these CLS tokens are concatenated for the final representation per row.

    rope_base : float, default=100000
        Base scaling factor for rotary position encoding.

    rope_interleaved : bool, default=True
        If True, uses interleaved rotation where dimension pairs are (0,1), (2,3), etc.
        If False, uses non-interleaved rotation where the embedding is split into
        first half [0:d//2] and second half [d//2:d].

    identity_mode : {"rope", "temporary", "none"}, default="rope"
        Feature identity signal used by the row transformer. ``"rope"`` keeps
        the input feature order and applies RoPE. ``"temporary"`` applies a
        fresh, table-wise random permutation to feature tokens before RoPE;
        the permutation is shared by every row in a table, so it supplies
        within-table identity without a stable ordinal meaning. ``"none"``
        disables RoPE entirely.

    dropout : float, default=0.0
        Dropout probability used in the encoder.

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable.

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward).

    bias_free_ln : bool, default=False
        If True, removes bias from all LayerNorm layers.

    recompute : bool, default=False
        If True, uses gradient checkpointing to save memory at the cost of additional computation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        num_cls: int = 4,
        rope_base: float = 100000,
        rope_interleaved: bool = True,
        identity_mode: str = "rope",
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        zero_init: bool = True,
        recompute: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.num_cls = num_cls
        self.norm_first = norm_first
        self.recompute = recompute

        if identity_mode not in {"rope", "temporary", "none"}:
            raise ValueError(
                f"identity_mode must be one of 'rope', 'temporary', or 'none', got {identity_mode!r}"
            )
        self.identity_mode = identity_mode

        self.tf_row = Encoder(
            num_blocks=num_blocks,
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            use_rope=identity_mode != "none",
            rope_base=rope_base,
            rope_interleaved=rope_interleaved,
            zero_init=zero_init,
            recompute=recompute,
        )

        self.cls_tokens = nn.Parameter(torch.empty(num_cls, embed_dim))
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)

        self.out_ln = nn.LayerNorm(embed_dim, bias=not bias_free_ln) if norm_first else nn.Identity()
        self.inference_mgr = InferenceManager(enc_name="tf_row", out_dim=embed_dim * self.num_cls, out_no_seq=True)
        # This fallback exists for direct model use outside Trainer. Formal
        # training injects permutations from TrainerIdentityRNG explicitly.
        self._identity_generator = torch.Generator(device="cpu")
        self._identity_generator.manual_seed(torch.initial_seed())

    def sample_row_identity_permutation(
        self, *, batch_size: int, num_features: int, device: torch.device | str
    ) -> Optional[Tensor]:
        """Sample using the module-local fallback generator, never a global RNG."""
        if self.identity_mode != "temporary":
            return None
        if num_features < 1:
            raise ValueError("num_features must be positive")
        return torch.stack(
            [
                torch.randperm(num_features, generator=self._identity_generator)
                for _ in range(batch_size)
            ]
        ).to(device=device)

    def _apply_temporary_feature_identity(
        self,
        embeddings: Tensor,
        key_mask: Optional[Tensor] = None,
        row_identity_permutation: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Randomly assign RoPE positions to features for this forward pass.

        A single permutation is sampled per table and shared across all rows.
        CLS tokens stay fixed. When a padding mask is present it is permuted
        together with the feature tokens.
        """
        if self.identity_mode != "temporary":
            return embeddings, key_mask

        batch_size, num_rows, total_tokens, embed_dim = embeddings.shape
        num_features = total_tokens - self.num_cls
        if num_features <= 1:
            return embeddings, key_mask

        if row_identity_permutation is None:
            permutations = self.sample_row_identity_permutation(
                batch_size=batch_size,
                num_features=num_features,
                device=embeddings.device,
            )
        else:
            if row_identity_permutation.shape != (batch_size, num_features):
                raise ValueError(
                    "row_identity_permutation must have shape "
                    f"{(batch_size, num_features)}, got {tuple(row_identity_permutation.shape)}"
                )
            permutations = row_identity_permutation.to(device=embeddings.device, dtype=torch.long)
            expected = torch.arange(num_features, device=embeddings.device).expand(batch_size, -1)
            if not torch.equal(permutations.sort(dim=-1).values, expected):
                raise ValueError("row_identity_permutation rows must each be a feature permutation")
        feature_index = permutations[:, None, :, None].expand(batch_size, num_rows, num_features, embed_dim)
        permuted_features = embeddings[:, :, self.num_cls :].gather(2, feature_index)
        embeddings = torch.cat((embeddings[:, :, : self.num_cls], permuted_features), dim=2)

        if key_mask is not None:
            mask_index = permutations[:, None, :].expand(batch_size, num_rows, num_features)
            permuted_mask = key_mask[:, :, self.num_cls :].gather(2, mask_index)
            key_mask = torch.cat((key_mask[:, :, : self.num_cls], permuted_mask), dim=2)

        return embeddings, key_mask

    def _aggregate_embeddings(self, embeddings: Tensor, key_mask: Optional[Tensor] = None) -> Tensor:
        """Process a batch of rows through a transformer encoder.

        This method:

        1. Processes embeddings through the transformer
        2. Extracts only the class token representations and applies normalization if pre-norm
        3. Concatenates the class tokens into a single vector per row

        Parameters
        ----------
        embeddings : Tensor
            Feature embeddings of shape (B, T, H+C, E) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features
             - C is the number of class tokens
             - E is the embedding dimension

        key_mask : Optional[Tensor], default=None
            Boolean mask of shape (B, T, H+C) where True indicates positions
            to ignore during attention (empty feature slots).

        Returns
        -------
        Tensor
            Flattened class token outputs of shape (B*T, C*E).
        """
        rope = self.tf_row.rope

        # Process all blocks except the last
        if self.recompute:
            for block in self.tf_row.blocks[:-1]:
                embeddings = checkpoint(
                    partial(block, key_padding_mask=key_mask, rope=rope), embeddings, use_reentrant=False
                )
        else:
            for block in self.tf_row.blocks[:-1]:
                embeddings = block(embeddings, key_padding_mask=key_mask, rope=rope)

        # Last block: q = CLS tokens, k/v = full sequence
        last_block = self.tf_row.blocks[-1]
        if self.recompute:
            cls_outputs = checkpoint(
                lambda emb: last_block(
                    q=emb[..., : self.num_cls, :], k=emb, v=emb, key_padding_mask=key_mask, rope=rope
                ),
                embeddings,
                use_reentrant=False,
            )
        else:
            cls_outputs = last_block(
                q=embeddings[..., : self.num_cls, :], k=embeddings, v=embeddings, key_padding_mask=key_mask, rope=rope
            )
        del embeddings
        cls_outputs = self.out_ln(cls_outputs)

        return cls_outputs.flatten(-2)  # (B, T, C*E)

    def _train_forward(
        self,
        embeddings: Tensor,
        d: Optional[Tensor] = None,
        row_identity_permutation: Optional[Tensor] = None,
    ) -> Tensor:
        """Transform feature embeddings into row representations for training.

        Parameters
        ----------
        embeddings : Tensor
            Feature embeddings of shape (B, T, H+C, E) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features
             - C is the number of class tokens
             - E is the embedding dimension

        d : Optional[Tensor], default=None
            The number of features per dataset. Used only in training mode.

        Returns
        -------
        Tensor
            Row representations of shape (B, T, C*E) where C is the number of class tokens.
        """

        B, T, HC, E = embeddings.shape
        device = embeddings.device

        cls_tokens = self.cls_tokens.expand(B, T, self.num_cls, self.embed_dim)
        embeddings[:, :, : self.num_cls] = cls_tokens.to(embeddings.device)

        # Create mask to prevent from attending to empty features
        if d is None:
            key_mask = None
        else:
            d = d + self.num_cls
            indices = torch.arange(HC, device=device).view(1, 1, HC).expand(B, T, HC)
            key_mask = indices >= d.view(B, 1, 1)  # (B, T, HC)

        embeddings, key_mask = self._apply_temporary_feature_identity(
            embeddings, key_mask, row_identity_permutation
        )
        representations = self._aggregate_embeddings(embeddings, key_mask)  # (B, T, C*E)

        return representations  # (B, T, C*E)

    def _inference_forward(
        self,
        embeddings: Tensor,
        mgr_config: MgrConfig = None,
        row_identity_permutation: Optional[Tensor] = None,
    ) -> Tensor:
        """Transform feature embeddings into row representations for inference.

        Parameters
        ----------
        embeddings : Tensor
            Feature embeddings of shape (B, T, H+C, E) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features
             - C is the number of class tokens
             - E is the embedding dimension

        mgr_config : MgrConfig, default=None
            Configuration for InferenceManager.

        Returns
        -------
        Tensor
            Row representations of shape (B, T, C*E) where C is the number of class tokens.
        """
        # Configure inference parameters
        if mgr_config is None:
            mgr_config = InferenceConfig().ROW_CONFIG
        self.inference_mgr.configure(**mgr_config)

        B, T = embeddings.shape[:2]
        cls_tokens = self.cls_tokens.expand(B, T, self.num_cls, self.embed_dim)
        embeddings[:, :, : self.num_cls] = cls_tokens.to(embeddings.device)
        embeddings, _ = self._apply_temporary_feature_identity(
            embeddings, row_identity_permutation=row_identity_permutation
        )
        representations = self.inference_mgr(
            self._aggregate_embeddings, inputs=OrderedDict([("embeddings", embeddings)])
        )

        return representations  # (B, T, C*E)

    def forward(
        self,
        embeddings: Tensor,
        d: Optional[Tensor] = None,
        mgr_config: MgrConfig = None,
        row_identity_permutation: Optional[Tensor] = None,
    ) -> Tensor:
        """Transform feature embeddings into row representations.

        Parameters
        ----------
        embeddings : Tensor
            Feature embeddings of shape (B, T, H+C, E) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features
             - C is the number of class tokens
             - E is the embedding dimension

        d : Optional[Tensor], default=None
            The number of features per dataset. Used only in training mode.

        mgr_config : MgrConfig, default=None
            Configuration for InferenceManager. Used only in inference mode.

        Returns
        -------
        Tensor
            Row representations of shape (B, T, C*E) where C is the number of class tokens.
        """

        if self.training:
            representations = self._train_forward(embeddings, d, row_identity_permutation)
        else:
            representations = self._inference_forward(
                embeddings, mgr_config, row_identity_permutation
            )

        return representations  # (B, T, C*E)
