from __future__ import annotations

from typing import Optional, Literal
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
        fingerprint_dim: Optional[int] = None,
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

        if fingerprint_dim is not None and (
            isinstance(fingerprint_dim, bool)
            or not isinstance(fingerprint_dim, int)
            or fingerprint_dim < 1
        ):
            raise ValueError("fingerprint_dim must be a positive integer or None")
        if fingerprint_dim is not None and identity_mode != "none":
            raise ValueError("row fingerprint requires identity_mode='none'")
        self.fingerprint_dim = fingerprint_dim

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

        # The fingerprint arm has extra parameters, but constructing them must not
        # advance the global RNG and thereby change the shared ICL initialization.
        if fingerprint_dim is None:
            self.fingerprint_q_projections = None
            self.fingerprint_k_projections = None
            self.register_parameter("fingerprint_q_gates", None)
            self.register_parameter("fingerprint_k_gates", None)
        else:
            with torch.random.fork_rng(devices=[]):
                self.fingerprint_q_projections = nn.ModuleList(
                    [self._make_fingerprint_projection(fingerprint_dim) for _ in range(num_blocks)]
                )
                self.fingerprint_k_projections = nn.ModuleList(
                    [self._make_fingerprint_projection(fingerprint_dim) for _ in range(num_blocks)]
                )
            self.fingerprint_q_gates = nn.Parameter(torch.full((num_blocks,), 0.1))
            self.fingerprint_k_gates = nn.Parameter(torch.full((num_blocks,), 0.1))
        # This fallback exists for direct model use outside Trainer. Formal
        # training injects permutations from TrainerIdentityRNG explicitly.
        self._identity_generator = torch.Generator(device="cpu")
        self._identity_generator.manual_seed(torch.initial_seed())

    def _make_fingerprint_projection(self, fingerprint_dim: int) -> nn.Module:
        """Build a low-rank, bias-free identity projection."""
        return nn.Sequential(
            # No affine term: an all-zero CLS fingerprint must remain exactly
            # zero, even after the fingerprint projections have been trained.
            nn.LayerNorm(self.embed_dim, elementwise_affine=False),
            nn.Linear(self.embed_dim, fingerprint_dim, bias=False),
            nn.GELU(),
            nn.Linear(fingerprint_dim, self.embed_dim, bias=False),
        )

    def _prepare_row_fingerprint(
        self,
        row_fingerprint: Optional[Tensor],
        *,
        embeddings: Tensor,
        intervention: Literal["correct", "zero", "permuted", "collapsed"],
        permutation: Optional[Tensor],
    ) -> Optional[Tensor]:
        if self.fingerprint_dim is None:
            if row_fingerprint is not None:
                raise ValueError("row_fingerprint was supplied to a model with fingerprint disabled")
            return None
        if row_fingerprint is None:
            raise ValueError("fingerprint-enabled RowInteraction requires row_fingerprint")

        expected = (embeddings.shape[0], embeddings.shape[2], embeddings.shape[3])
        if tuple(row_fingerprint.shape) != expected:
            raise ValueError(
                f"row_fingerprint must have shape {expected}, got {tuple(row_fingerprint.shape)}"
            )
        if intervention not in {"correct", "zero", "permuted", "collapsed"}:
            raise ValueError(f"unknown fingerprint intervention: {intervention!r}")

        fingerprint = row_fingerprint.to(device=embeddings.device, dtype=embeddings.dtype)
        cls = fingerprint[:, : self.num_cls]
        if torch.count_nonzero(cls.detach()).item() != 0:
            raise ValueError("CLS fingerprint slots must be exactly zero")
        features = fingerprint[:, self.num_cls :]

        if intervention == "zero":
            features = torch.zeros_like(features)
        elif intervention == "collapsed":
            features = features.mean(dim=1, keepdim=True).expand_as(features)
        elif intervention == "permuted":
            batch_size, num_features = features.shape[:2]
            if permutation is None:
                raise ValueError("permuted fingerprint intervention requires a permutation")
            if tuple(permutation.shape) != (batch_size, num_features):
                raise ValueError(
                    "fingerprint_permutation must have shape "
                    f"{(batch_size, num_features)}, got {tuple(permutation.shape)}"
                )
            permutation = permutation.to(device=features.device, dtype=torch.long)
            expected_indices = torch.arange(num_features, device=features.device).expand(batch_size, -1)
            if not torch.equal(permutation.sort(dim=-1).values, expected_indices):
                raise ValueError("fingerprint_permutation rows must be feature permutations")
            gather_index = permutation[..., None].expand_as(features)
            features = features.gather(1, gather_index)

        cls_zeros = torch.zeros_like(cls)
        return torch.cat((cls_zeros, features), dim=1)

    def _fingerprint_identities(
        self,
        fingerprint: Optional[Tensor],
        layer_idx: int,
        layer_gates: Optional[Tensor],
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        if fingerprint is None:
            return None, None
        q_gate = self.fingerprint_q_gates[layer_idx]
        k_gate = self.fingerprint_k_gates[layer_idx]
        if layer_gates is not None:
            gates = torch.as_tensor(layer_gates, device=fingerprint.device, dtype=fingerprint.dtype)
            if tuple(gates.shape) == (self.num_blocks,):
                q_gate = q_gate * gates[layer_idx]
                k_gate = k_gate * gates[layer_idx]
            elif tuple(gates.shape) == (self.num_blocks, 2):
                q_gate = q_gate * gates[layer_idx, 0]
                k_gate = k_gate * gates[layer_idx, 1]
            else:
                raise ValueError(
                    "fingerprint_layer_gates must have shape "
                    f"{(self.num_blocks,)} or {(self.num_blocks, 2)}"
                )
        q_identity = self.fingerprint_q_projections[layer_idx](fingerprint) * q_gate
        k_identity = self.fingerprint_k_projections[layer_idx](fingerprint) * k_gate
        return q_identity, k_identity

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
            torch._assert_async(
                torch.all(permutations.sort(dim=-1).values == expected),
                "row_identity_permutation rows must each be a feature permutation",
            )
        feature_index = permutations[:, None, :, None].expand(batch_size, num_rows, num_features, embed_dim)
        permuted_features = embeddings[:, :, self.num_cls :].gather(2, feature_index)
        embeddings = torch.cat((embeddings[:, :, : self.num_cls], permuted_features), dim=2)

        if key_mask is not None:
            mask_index = permutations[:, None, :].expand(batch_size, num_rows, num_features)
            permuted_mask = key_mask[:, :, self.num_cls :].gather(2, mask_index)
            key_mask = torch.cat((key_mask[:, :, : self.num_cls], permuted_mask), dim=2)

        return embeddings, key_mask

    def _aggregate_embeddings(
        self,
        embeddings: Tensor,
        key_mask: Optional[Tensor] = None,
        row_fingerprint: Optional[Tensor] = None,
        fingerprint_layer_gates: Optional[Tensor] = None,
    ) -> Tensor:
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
            for layer_idx, block in enumerate(self.tf_row.blocks[:-1]):
                q_identity, k_identity = self._fingerprint_identities(
                    row_fingerprint, layer_idx, fingerprint_layer_gates
                )
                embeddings = checkpoint(
                    partial(
                        block,
                        key_padding_mask=key_mask,
                        rope=rope,
                        q_identity=q_identity,
                        k_identity=k_identity,
                    ),
                    embeddings,
                    use_reentrant=False,
                )
        else:
            for layer_idx, block in enumerate(self.tf_row.blocks[:-1]):
                q_identity, k_identity = self._fingerprint_identities(
                    row_fingerprint, layer_idx, fingerprint_layer_gates
                )
                embeddings = block(
                    embeddings,
                    key_padding_mask=key_mask,
                    rope=rope,
                    q_identity=q_identity,
                    k_identity=k_identity,
                )

        # Last block: q = CLS tokens, k/v = full sequence
        last_block = self.tf_row.blocks[-1]
        last_idx = self.num_blocks - 1
        q_identity, k_identity = self._fingerprint_identities(
            row_fingerprint, last_idx, fingerprint_layer_gates
        )
        cls_q_identity = None if q_identity is None else q_identity[..., : self.num_cls, :]
        if self.recompute:
            cls_outputs = checkpoint(
                lambda emb: last_block(
                    q=emb[..., : self.num_cls, :],
                    k=emb,
                    v=emb,
                    q_identity=cls_q_identity,
                    k_identity=k_identity,
                    key_padding_mask=key_mask,
                    rope=rope,
                ),
                embeddings,
                use_reentrant=False,
            )
        else:
            cls_outputs = last_block(
                q=embeddings[..., : self.num_cls, :],
                k=embeddings,
                v=embeddings,
                q_identity=cls_q_identity,
                k_identity=k_identity,
                key_padding_mask=key_mask,
                rope=rope,
            )
        del embeddings
        cls_outputs = self.out_ln(cls_outputs)

        return cls_outputs.flatten(-2)  # (B, T, C*E)

    def _train_forward(
        self,
        embeddings: Tensor,
        d: Optional[Tensor] = None,
        row_identity_permutation: Optional[Tensor] = None,
        row_fingerprint: Optional[Tensor] = None,
        fingerprint_intervention: Literal["correct", "zero", "permuted", "collapsed"] = "correct",
        fingerprint_permutation: Optional[Tensor] = None,
        fingerprint_layer_gates: Optional[Tensor] = None,
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
        row_fingerprint = self._prepare_row_fingerprint(
            row_fingerprint,
            embeddings=embeddings,
            intervention=fingerprint_intervention,
            permutation=fingerprint_permutation,
        )
        if row_fingerprint is not None:
            row_fingerprint = row_fingerprint[:, None].expand(B, T, HC, E)
        representations = self._aggregate_embeddings(
            embeddings, key_mask, row_fingerprint, fingerprint_layer_gates
        )  # (B, T, C*E)

        return representations  # (B, T, C*E)

    def _inference_forward(
        self,
        embeddings: Tensor,
        mgr_config: MgrConfig = None,
        row_identity_permutation: Optional[Tensor] = None,
        row_fingerprint: Optional[Tensor] = None,
        fingerprint_intervention: Literal["correct", "zero", "permuted", "collapsed"] = "correct",
        fingerprint_permutation: Optional[Tensor] = None,
        fingerprint_layer_gates: Optional[Tensor] = None,
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
        row_fingerprint = self._prepare_row_fingerprint(
            row_fingerprint,
            embeddings=embeddings,
            intervention=fingerprint_intervention,
            permutation=fingerprint_permutation,
        )
        if row_fingerprint is not None:
            row_fingerprint = row_fingerprint[:, None].expand(
                B, T, embeddings.shape[2], embeddings.shape[3]
            )
        aggregate = partial(
            self._aggregate_embeddings,
            fingerprint_layer_gates=fingerprint_layer_gates,
        )
        inputs = OrderedDict([("embeddings", embeddings)])
        if row_fingerprint is not None:
            inputs["row_fingerprint"] = row_fingerprint
        representations = self.inference_mgr(
            aggregate, inputs=inputs
        )

        return representations  # (B, T, C*E)

    def forward(
        self,
        embeddings: Tensor,
        d: Optional[Tensor] = None,
        mgr_config: MgrConfig = None,
        row_identity_permutation: Optional[Tensor] = None,
        row_fingerprint: Optional[Tensor] = None,
        fingerprint_intervention: Literal["correct", "zero", "permuted", "collapsed"] = "correct",
        fingerprint_permutation: Optional[Tensor] = None,
        fingerprint_layer_gates: Optional[Tensor] = None,
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
            if (
                self.fingerprint_dim is None
                and row_fingerprint is None
                and fingerprint_intervention == "correct"
                and fingerprint_permutation is None
                and fingerprint_layer_gates is None
            ):
                representations = self._train_forward(embeddings, d, row_identity_permutation)
            else:
                representations = self._train_forward(
                    embeddings,
                    d,
                    row_identity_permutation,
                    row_fingerprint,
                    fingerprint_intervention,
                    fingerprint_permutation,
                    fingerprint_layer_gates,
                )
        else:
            if (
                self.fingerprint_dim is None
                and row_fingerprint is None
                and fingerprint_intervention == "correct"
                and fingerprint_permutation is None
                and fingerprint_layer_gates is None
            ):
                representations = self._inference_forward(
                    embeddings, mgr_config, row_identity_permutation
                )
            else:
                representations = self._inference_forward(
                    embeddings,
                    mgr_config,
                    row_identity_permutation,
                    row_fingerprint,
                    fingerprint_intervention,
                    fingerprint_permutation,
                    fingerprint_layer_gates,
                )

        return representations  # (B, T, C*E)
