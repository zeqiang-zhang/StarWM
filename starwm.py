"""StarWM world model (Section 3 and Algorithm 1 of the paper).

Only the encoder and decoder of DreamerV3 are changed; the RSSM and the
actor-critic are inherited unchanged.

  1. Encoder interception: the first three conv stages give F_8 (128ch, 8x8),
     a 1x1 conv projects it to C=256 channels (L=64 spatial tokens). The fourth
     stage continues to F_4 (256ch, 4x4) for the residual stream.
  2. Cross-attention routing (Section 3.1): N learnable queries attend over
     K = F + E_pos (2D sinusoidal) with H heads; heads are averaged into
     A_t (N x L). Values are V = [F ; E_coord].
  3. Self-supervised attention training on e_attn = flatten(A_t F_t):
     inverse dynamics (CE for discrete, MSE for continuous actions), binary
     temporal InfoNCE and, for StarWM-R, reward prediction
     (Appendix E.1, ``attn_reward_scale`` > 0).
  4. Gradient isolation: the RSSM receives e_dyn = sg(A_t) V_t
     (``detach_attn_for_rssm``, "detach A") and the decoder uses sg(A_t)
     (``detach_attn_for_recon``, "detach B").
  5. Dual-stream decoder: dynamics stream A^T phi_fg([h;z]) (inverse spatial
     broadcast) blended with a residual Gaussian-bottleneck (VAE) stream from
     flatten(F_4) by the coverage mask M = clamp(sum_n A_n, 0, 1) at 8x8, then
     upsampled by transposed convolutions (Section 3.3). The residual latent is
     KL-regularised with beta_bg annealed 5.0 -> 0.5 over 15k steps and is not
     used during imagination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import networks
import tools
from tools import to_np


class CrossAttentionExtractor(nn.Module):
    """Cross-attention with N learnable queries on 8×8 feature maps.

    Q: (N, C) learnable queries
    K: F_8 + PE (positional encoding)
    V: F_8 + coord_grid (for position-aware extraction)
    Z = softmax(QK^T / sqrt(C)) × V → (N, C+2) entity tokens
    """
    def __init__(self, channels, spatial_size=8, num_queries=4, num_heads=4, pe_type='1d_sin'):
        super().__init__()
        self.channels = channels
        self.spatial_size = spatial_size
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.L = spatial_size * spatial_size  # 64
        self.pe_type = pe_type

        # Learnable queries
        self.queries = nn.Parameter(torch.randn(num_queries, channels) * 0.02)

        # Positional encoding for keys
        if pe_type == 'learnable':
            self.positional_encoding = nn.Parameter(
                torch.randn(spatial_size * spatial_size, channels) * 0.02
            )
        elif pe_type == 'learnable_2d_init':
            # Learnable PE initialized with 2d sinusoidal values (warm start)
            pe_init = self._make_true_2d_pe(spatial_size, channels)
            self.positional_encoding = nn.Parameter(pe_init)
        elif pe_type == '2d_sin':
            pe = self._make_true_2d_pe(spatial_size, channels)
            self.register_buffer("positional_encoding", pe)  # (L, C)
        else:  # '1d_sin' (original, backward compatible)
            pe = self._make_1d_pe(spatial_size, channels)
            self.register_buffer("positional_encoding", pe)  # (L, C)

        # Coordinate grid for values (normalized [0,1])
        grid = self._make_coord_grid(spatial_size)
        self.register_buffer("coord_grid", grid)  # (L, 2)

        # Multi-head attention projections
        assert channels % num_heads == 0
        self.head_dim = channels // num_heads
        self.Wq = nn.Linear(channels, channels, bias=False)
        self.Wk = nn.Linear(channels, channels, bias=False)
        self.Wv = nn.Linear(channels + 2, channels + 2, bias=False)

        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.xavier_uniform_(self.Wk.weight)
        nn.init.xavier_uniform_(self.Wv.weight)

    @staticmethod
    def _make_1d_pe(spatial_size, channels):
        """Create 1D sinusoidal positional encoding (original, treats grid as flat sequence)."""
        pe = torch.zeros(spatial_size * spatial_size, channels)
        position = torch.arange(0, spatial_size * spatial_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2, dtype=torch.float) * -(math.log(10000.0) / channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:channels // 2])
        return pe

    @staticmethod
    def _make_true_2d_pe(spatial_size, channels):
        """Create true 2D sinusoidal positional encoding.
        Splits channels in half: first half encodes y, second half encodes x.
        """
        pe = torch.zeros(spatial_size * spatial_size, channels)
        half_c = channels // 2
        div_term = torch.exp(torch.arange(0, half_c, 2, dtype=torch.float) * -(math.log(10000.0) / half_c))

        # y positions (row index)
        y_pos = torch.arange(0, spatial_size, dtype=torch.float).unsqueeze(1)  # (H, 1)
        y_pe = torch.zeros(spatial_size, half_c)
        y_pe[:, 0::2] = torch.sin(y_pos * div_term)
        y_pe[:, 1::2] = torch.cos(y_pos * div_term)

        # x positions (col index)
        x_pos = torch.arange(0, spatial_size, dtype=torch.float).unsqueeze(1)  # (W, 1)
        x_pe = torch.zeros(spatial_size, half_c)
        x_pe[:, 0::2] = torch.sin(x_pos * div_term)
        x_pe[:, 1::2] = torch.cos(x_pos * div_term)

        # Combine: each spatial position (y, x) gets [y_pe[y], x_pe[x]]
        for y in range(spatial_size):
            for x in range(spatial_size):
                idx = y * spatial_size + x
                pe[idx, :half_c] = y_pe[y]
                pe[idx, half_c:] = x_pe[x]
        return pe

    @staticmethod
    def _make_coord_grid(spatial_size):
        """Create normalized (x, y) coordinate grid."""
        coords = torch.linspace(0, 1, spatial_size)
        gy, gx = torch.meshgrid(coords, coords, indexing='ij')
        grid = torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # (L, 2)
        return grid

    def forward(self, fmap_flat):
        """
        Args:
            fmap_flat: (B, C, H, W) feature map at 8×8

        Returns:
            Z: (B, N, C+2) entity tokens (with coordinates)
            A: (B, N, L) attention weights
            pooled: (B, N, C) entity tokens without coordinates (for inverse dynamics)
        """
        B, C, H, W = fmap_flat.shape
        L = H * W  # 64

        # Flatten spatial → (B, L, C)
        V_raw = fmap_flat.reshape(B, C, L).permute(0, 2, 1)  # (B, L, C)

        # Keys: V + positional encoding
        K = V_raw + self.positional_encoding.unsqueeze(0)  # (B, L, C)

        # Values: V + coordinate grid → V_coord
        coord_expanded = self.coord_grid.unsqueeze(0).expand(B, -1, -1)  # (B, L, 2)
        V_coord = torch.cat([V_raw, coord_expanded], dim=-1)  # (B, L, C+2)

        # Queries
        Q = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, N, C)

        # Multi-head attention
        N = self.num_queries
        h = self.num_heads
        d = self.head_dim

        # Project Q and K (standard dim C), V has dim C+2
        Q_proj = self.Wq(Q).reshape(B, N, h, d).permute(0, 2, 1, 3)  # (B, h, N, d)
        K_proj = self.Wk(K).reshape(B, L, h, d).permute(0, 2, 1, 3)  # (B, h, L, d)

        # For V, we project (C+2) → (C+2) but split heads differently
        # Use a simpler approach: compute attention weights from Q,K then apply to V_coord directly
        attn_logits = torch.matmul(Q_proj, K_proj.transpose(-1, -2)) / math.sqrt(d)  # (B, h, N, L)
        attn_weights = torch.softmax(attn_logits, dim=-1)  # (B, h, N, L)

        # Average attention across heads for the final spatial attention map
        A = attn_weights.mean(dim=1)  # (B, N, L)

        # Extract entity tokens: Z = A × V_coord
        Z = torch.matmul(A, V_coord)  # (B, N, C+2)

        # Also extract without coordinates for inverse dynamics
        pooled = torch.matmul(A, V_raw)  # (B, N, C)

        return Z, A, pooled


class StarWMWorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super().__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}

        # ===== 1. Encoder (shared DreamerV3 encoder) =====
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.total_embed_size = self.encoder.outdim  # 4096

        # ===== 2. Encoder layer splitting for F_8 intercept =====
        cnn_depth = config.encoder.get('cnn_depth', 32)
        minres = config.encoder.get('minres', 4)
        image_size = config.size[0]
        stages = int(math.log2(image_size / minres))  # 4 for 64→4

        # F_8 is at layer 2 output: 128 channels, 8×8 spatial
        self.f8_channels = cnn_depth * (2 ** (stages - 2))  # 32 * 4 = 128
        self.f8_spatial = minres * 2  # 4 * 2 = 8
        # F_4 is at layer 3 output: 256 channels, 4×4 spatial
        self.f4_channels = cnn_depth * (2 ** (stages - 1))  # 32 * 8 = 256
        self.f4_spatial = minres  # 4

        # Split encoder CNN layers: layers[0..8] = first 3 stages (64→32→16→8), layers[9..11] = last stage (8→4)
        # Each stage has Conv + Norm + Act = 3 layers (or Conv + Act = 2 if no norm)
        encoder_cnn = self.encoder._cnn
        num_layers_per_stage = 3 if config.encoder.get('norm', True) else 2  # Conv + Norm + Act or Conv + Act
        split_idx = (stages - 1) * num_layers_per_stage  # After 3 stages (stages-1 = 3 for 4 total)
        self.encoder_to_8x8 = nn.Sequential(*list(encoder_cnn.layers)[:split_idx])
        self.encoder_8x8_to_4x4 = nn.Sequential(*list(encoder_cnn.layers)[split_idx:])

        # ===== 3. 1×1 Conv projection: 128 → proj_dim (256) =====
        self.proj_dim = getattr(config, 'crossattn_proj_dim', 256)
        self.conv_1x1 = nn.Conv2d(self.f8_channels, self.proj_dim, kernel_size=1)
        nn.init.xavier_uniform_(self.conv_1x1.weight)
        nn.init.zeros_(self.conv_1x1.bias)
        print(f"[XATTN] 1×1 Conv projection: {self.f8_channels} → {self.proj_dim}")

        # ===== 4. Attention mechanism (mode-dependent) =====
        num_queries = getattr(config, 'crossattn_num_queries', 4)
        self.num_queries = num_queries
        num_heads = getattr(config, 'crossattn_num_heads', 4)

        # Cross-attention with learnable queries at 8x8.
        pe_type = getattr(config, 'crossattn_pe_type', '2d_sin')
        self.cross_attn = CrossAttentionExtractor(
            channels=self.proj_dim,
            spatial_size=self.f8_spatial,
            num_queries=num_queries,
            num_heads=num_heads,
            pe_type=pe_type,
        )
        attn_pooled_dim = self.proj_dim * num_queries  # N x 256
        print(f"[XATTN] Cross-attention: N={num_queries}, heads={num_heads}, "
              f"pe={pe_type} at {self.f8_spatial}x{self.f8_spatial}")

        # ===== 5. Self-supervised training (inverse dynamics + contrastive + reward pred) =====
        self._infonce_scale = getattr(config, "infonce_scale", 1.0)
        self._infonce_temp = getattr(config, "infonce_temp", 0.1)
        self._attn_inv_scale = getattr(config, "attn_inv_scale", 1.0)
        self._attn_reward_scale = getattr(config, "attn_reward_scale", 0.0)
        # Inverse-dynamics window offset for the delayed-action experiment
        # (Appendix F.5): predict a_t from (e_{t+k}, e_{t+k+1}). 0 = standard.
        self._inv_delay = getattr(config, "inv_delay", 0)
        self._detach_attn_for_rssm = getattr(config, "detach_attn_for_rssm", False)
        if self._detach_attn_for_rssm:
            print(f"[XATTN] detach_attn_for_rssm=True: recon gradients will NOT flow to attention weights via RSSM")
        self._detach_attn_for_recon = getattr(config, "detach_attn_for_recon", True)
        if not self._detach_attn_for_recon:
            print(f"[XATTN] detach_attn_for_recon=False: decoder recon gradients WILL flow through attn_map")
        self._detach_encoder_for_attn = getattr(config, "detach_encoder_for_attn", False)
        if self._detach_encoder_for_attn:
            print(f"[XATTN] detach_encoder_for_attn=True: inverse/contrastive will NOT backprop into encoder")

        # Contrastive projection (from pooled entity features)
        infonce_proj_dim = getattr(config, "infonce_proj_dim", 128)
        inv_input_dim = self.proj_dim * num_queries
        self.infonce_proj = nn.Sequential(
            nn.Linear(inv_input_dim, inv_input_dim),
            nn.SiLU(),
            nn.Linear(inv_input_dim, infonce_proj_dim)
        ).to(config.device)

        # Inverse dynamics head
        self._num_actions = getattr(config, "num_actions", 4)
        task = getattr(config, 'task', '')
        self._continuous_actions = task.startswith('dmc_')
        self.attn_inv_head = nn.Sequential(
            nn.Linear(inv_input_dim * 2, inv_input_dim),
            nn.SiLU(),
            nn.Linear(inv_input_dim, self._num_actions)
        ).to(config.device)
        inv_type = 'MSE regression' if self._continuous_actions else f'CE classification ({self._num_actions} classes)'
        print(f"[XATTN] Inverse dynamics: {inv_type}, scale={self._attn_inv_scale}")

        # Attention reward prediction head (predicts reward from pooled attention features)
        if self._attn_reward_scale > 0:
            self.attn_reward_head = nn.Sequential(
                nn.Linear(inv_input_dim, inv_input_dim // 2),
                nn.SiLU(),
                nn.Linear(inv_input_dim // 2, 1)
            ).to(config.device)
            print(f"[XATTN] Attention reward prediction: scale={self._attn_reward_scale}")

        # ===== 6. proj_dyn -> RSSM =====
        rssm_embed_dim = self.proj_dim * num_queries + 2 * num_queries  # N*(256+2)

        self.dyn_dim = getattr(config, "dyn_dim", 0) or rssm_embed_dim
        self.proj_dyn = nn.Sequential(
            nn.Linear(rssm_embed_dim, config.units),
            nn.LayerNorm(config.units) if config.norm == 'layer' else nn.Identity(),
            nn.SiLU(),
            nn.Linear(config.units, self.dyn_dim)
        ).to(config.device)
        print(f"[XATTN] proj_dyn: {rssm_embed_dim} → MLP → {self.dyn_dim}")

        # ===== 7. proj_res: background encoding =====
        self.res_dim = getattr(config, "res_dim", 128)
        self._res_detach_encoder = getattr(config, "res_detach_encoder", False)

        # Standard residual: full flatten(F_4) = 4096
        res_input_dim = self.total_embed_size
        print(f"[XATTN] proj_res: {res_input_dim} -> VAE -> {self.res_dim}")

        self.proj_res = nn.Sequential(
            nn.Linear(res_input_dim, config.units),
            nn.LayerNorm(config.units) if config.norm == 'layer' else nn.Identity(),
            nn.SiLU(),
            nn.Linear(config.units, self.res_dim * 2)
        ).to(config.device)
        self.res_prior = torch.distributions.Normal(
            torch.zeros(self.res_dim, device=config.device),
            torch.ones(self.res_dim, device=config.device)
        )

        # KL annealing
        self._annealing = getattr(config, "Annealing", True)
        self._res_kl_scale = getattr(config, "res_kl_scale", 0.5)
        if self._annealing:
            self._res_kl_scale_init = getattr(config, "res_kl_scale_init", 5.0)
            self._res_kl_anneal_steps = getattr(config, "res_kl_anneal_steps", 15000)
            print(f"[XATTN] KL annealing: {self._res_kl_scale_init} → {self._res_kl_scale} over {self._res_kl_anneal_steps} steps")
        self.register_buffer("_wm_train_step", torch.tensor(0, dtype=torch.long))

        # ===== 8. RSSM =====
        self.dynamics = networks.RSSM(
            config.dyn_stoch, config.dyn_deter, config.dyn_hidden,
            config.dyn_rec_depth, config.dyn_discrete, config.act, config.norm,
            config.dyn_mean_act, config.dyn_std_act, config.dyn_min_std,
            config.unimix_ratio, config.initial, config.num_actions,
            self.dyn_dim, config.device,
        )

        # RSSM feature size
        if config.dyn_discrete:
            rssm_feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            rssm_feat_size = config.dyn_stoch + config.dyn_deter

        # ===== 9. Decoder: assembly at 8×8 or 4×4 =====
        some_shape = list({k: tuple(v.shape) for k, v in obs_space.spaces.items()
                         if k not in ('is_first', 'is_last', 'is_terminal') and len(tuple(v.shape)) == 3}.values())[0]
        img_shape = (some_shape[-1],) + some_shape[:-1]  # (3, 64, 64)

        # All modes assemble at 8×8, then ConvTranspose×3 to 64×64
        self.assembly_spatial = self.f8_spatial  # 8
        self.assembly_channels = self.proj_dim  # 256
        dec_start_res = self.f8_spatial
        dec_stages = int(math.log2(img_shape[1]) - math.log2(dec_start_res))  # 3

        # MLP_fg: rssm_feat -> N entity vectors of assembly_channels each
        # (used for inverse broadcasting onto the assembly grid).
        self.mlp_fg = nn.Sequential(
            nn.Linear(rssm_feat_size, config.units),
            nn.SiLU(),
            nn.Linear(config.units, num_queries * self.assembly_channels)  # N x 256
        ).to(config.device)
        print(f"[XATTN] MLP_fg: feat({rssm_feat_size}) -> "
              f"{num_queries} x {self.assembly_channels} "
              f"(inverse broadcast to {self.assembly_spatial}x{self.assembly_spatial})")

        # MLP_bg: z_res → assembly_channels × assembly_spatial^2
        bg_fmap_dim = self.assembly_channels * self.assembly_spatial * self.assembly_spatial
        self.mlp_bg = nn.Sequential(
            nn.Linear(self.res_dim, config.units),
            nn.SiLU(),
            nn.Linear(config.units, bg_fmap_dim)
        ).to(config.device)
        print(f"[XATTN] MLP_bg: z_res({self.res_dim}) → {bg_fmap_dim}")

        # ConvTranspose decoder: (assembly_channels, 8, 8) → (3, 64, 64)
        # Channel progression: 256 → 128 → 64 → 3  (3 stages of upsample ×2)
        kernel_size = config.decoder.get('kernel_size', 4)
        val = 1 * (kernel_size - 1) - 2 + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        decoder_layers = []
        in_dim = self.assembly_channels  # 256
        for i in range(dec_stages):
            if i == dec_stages - 1:
                out_dim = img_shape[0]  # 3 (final output channels)
                decoder_layers.append(
                    nn.ConvTranspose2d(in_dim, out_dim, kernel_size, 2,
                                       padding=(pad, pad), output_padding=(outpad, outpad), bias=True)
                )
            else:
                out_dim = in_dim // 2
                decoder_layers.append(
                    nn.ConvTranspose2d(in_dim, out_dim, kernel_size, 2,
                                       padding=(pad, pad), output_padding=(outpad, outpad), bias=False)
                )
                decoder_layers.append(networks.ImgChLayerNorm(out_dim))
                act_cls = getattr(torch.nn, config.decoder.get('act', 'SiLU'))
                decoder_layers.append(act_cls())
            in_dim = out_dim
        [m.apply(tools.weight_init) for m in decoder_layers[:-1]]
        decoder_layers[-1].apply(tools.uniform_weight_init(1.0))
        self.conv_decoder = nn.Sequential(*decoder_layers).to(config.device)
        self._img_shape = img_shape
        print(f"[XATTN] Shared ConvDecoder: ({self.assembly_channels}, {self.assembly_spatial}, {self.assembly_spatial}) → ConvTranspose×{dec_stages} → {img_shape}")

        # ===== 10. Reward + Cont heads =====
        self.heads = nn.ModuleDict()
        reward_shape = (255,) if config.reward_head["dist"] == "symlog_disc" else ()
        self.heads["reward"] = networks.MLP(
            rssm_feat_size, reward_shape, config.reward_head["layers"], config.units,
            config.act, config.norm, dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"], device=config.device, name="Reward",
        )
        self.heads["cont"] = networks.MLP(
            rssm_feat_size, (), config.cont_head["layers"], config.units,
            config.act, config.norm, dist="binary",
            outscale=config.cont_head["outscale"], device=config.device, name="Cont",
        )

        # ===== 11. Optimizer =====
        self._model_opt = tools.Optimizer(
            "model", self.parameters(), config.model_lr,
            config.opt_eps, config.grad_clip, config.weight_decay,
            opt=config.opt, use_amp=self._use_amp,
        )
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )
        print(f"[XATTN] StarWMWorldModel initialized; assembly at "
              f"{self.assembly_spatial}x{self.assembly_spatial}.")

    # ================================================================
    # Encoder with F_8 intercept
    # ================================================================
    def _encode_with_f8(self, obs):
        """Run encoder and return both F_8 (projected) and F_4 (standard).

        Returns:
            f8_proj: (B*T, proj_dim, 8, 8) — projected F_8
            raw_embed: (B*T, 4096) — standard encoder output (F_4 flattened)
            batch_shape: (B, T)
        """
        inputs = torch.cat([obs[k] for k in self.encoder.cnn_shapes], -1)
        x = inputs.reshape((-1,) + tuple(inputs.shape[-3:]))
        x = x.permute(0, 3, 1, 2)
        x -= 0.5

        # Run to 8×8
        f8 = self.encoder_to_8x8(x)  # (B*T, 128, 8, 8)
        # Project 128 → 256
        f8_proj = self.conv_1x1(f8)  # (B*T, 256, 8, 8)

        # Continue to 4×4 for standard embed
        f4 = self.encoder_8x8_to_4x4(f8)  # (B*T, 256, 4, 4)
        raw_embed = f4.reshape(f4.shape[0], -1)  # (B*T, 4096)

        batch_shape = inputs.shape[:-3]  # (B, T)
        raw_embed = raw_embed.reshape(list(batch_shape) + [raw_embed.shape[-1]])

        return f8_proj, raw_embed, batch_shape

    # ================================================================
    # Attention: mode-specific
    # ================================================================
    def _compute_attention(self, f8_proj):
        """Compute attention and extract entity features.

        Returns:
            pooled_for_rssm: entity features for RSSM input (with coords).
                Shape (B*T, N*(C+2)).
            pooled_for_attn: entity features for inverse-dynamics / InfoNCE
                training (without coords). Shape (B*T, N*C).
            attn_map: cross-attention weights, shape (B*T, N, L=64).
        """
        BT = f8_proj.shape[0]

        # Cross-attention extraction.
        Z, A, pooled = self.cross_attn(f8_proj)  # Z:(BT,N,C+2), A:(BT,N,L), pooled:(BT,N,C)

        if self._detach_attn_for_rssm:
            # Recompute Z with detached A: recon -> encoder OK, recon -> attention blocked.
            BT_loc, C_loc, H_loc, W_loc = f8_proj.shape
            L_loc = H_loc * W_loc
            V_raw = f8_proj.reshape(BT_loc, C_loc, L_loc).permute(0, 2, 1)  # (BT, L, C)
            coord_expanded = self.cross_attn.coord_grid.unsqueeze(0).expand(BT_loc, -1, -1)
            V_coord = torch.cat([V_raw, coord_expanded], dim=-1)  # (BT, L, C+2)
            A_sg = A.detach()
            Z_rssm = torch.matmul(A_sg, V_coord)  # (BT, N, C+2)
            pooled_for_rssm = Z_rssm.reshape(BT, -1)
        else:
            pooled_for_rssm = Z.reshape(BT, -1)

        if self._detach_encoder_for_attn:
            # Detach V_raw so inv / contrastive only train attention (Q/K/queries),
            # not the encoder.
            BT_loc2, C_loc2, H_loc2, W_loc2 = f8_proj.shape
            L_loc2 = H_loc2 * W_loc2
            V_raw_detached = f8_proj.detach().reshape(BT_loc2, C_loc2, L_loc2).permute(0, 2, 1)
            pooled_for_attn = torch.matmul(A, V_raw_detached).reshape(BT, -1)
        else:
            pooled_for_attn = pooled.reshape(BT, -1)

        return pooled_for_rssm, pooled_for_attn, A

    # ================================================================
    # Residual encoding
    # ================================================================
    def _compute_residual(self, f8_proj, raw_embed, attn_map, detach_encoder=False):
        """Compute z_res from the standard encoder output.

        Args:
            f8_proj: (B*T, C, 8, 8)   (kept for signature compatibility)
            raw_embed: (B, T, 4096)
            attn_map: attention weights (kept for signature compatibility)
            detach_encoder: whether to detach encoder gradients for res

        Returns:
            embed_res: (B, T, res_dim) sampled z_res
            res_kl_loss: KL divergence loss
        """
        res_input = raw_embed.detach() if detach_encoder else raw_embed
        res_stats = self.proj_res(res_input)
        res_mu, res_logvar = torch.split(res_stats, self.res_dim, dim=-1)
        res_std = torch.exp(0.5 * res_logvar)
        res_dist = torch.distributions.Normal(res_mu, res_std)
        embed_res = res_dist.rsample()

        res_kl_raw = torch.distributions.kl_divergence(res_dist, self.res_prior).sum(dim=-1)
        res_kl_free = getattr(self._config, 'res_kl_free', 1.0)
        res_kl_loss = torch.clamp(res_kl_raw - res_kl_free, min=0.0)

        return embed_res, res_kl_loss

    # ================================================================
    # Decoder assembly
    # ================================================================
    def _decode_assembled(self, rssm_feat, embed_res, attn_for_recon, batch_shape):
        """Compose fg+bg at 8×8 and decode with shared CNN."""
        BT = rssm_feat.shape[0] if rssm_feat.dim() == 2 else rssm_feat.shape[0] * rssm_feat.shape[1]
        feat_flat = rssm_feat.reshape(BT, -1)
        res_flat = embed_res.reshape(BT, -1)
        S = self.assembly_spatial  # 8

        # Inverse broadcasting: A^T x Z_hat at 8x8.
        N = self.num_queries
        L = S * S  # 64

        # MLP_fg: (BT, rssm_feat_dim) -> (BT, N, assembly_channels)
        fg_tokens = self.mlp_fg(feat_flat).reshape(BT, N, self.assembly_channels)

        # Inverse broadcast: A^T x fg_tokens -> (BT, L, 256)
        A_recon = attn_for_recon.reshape(BT, N, L)
        fg_fmap = torch.matmul(A_recon.transpose(-1, -2), fg_tokens)  # (BT, L, 256)
        fg_fmap = fg_fmap.permute(0, 2, 1).reshape(BT, self.assembly_channels, S, S)

        # bg_fmap: (BT, res_dim) -> (BT, 256, 8, 8)
        bg_fmap = self.mlp_bg(res_flat).reshape(BT, self.assembly_channels, S, S)

        # Attention-weighted spatial fusion.
        fg_coverage = torch.clamp(A_recon.sum(dim=1), 0.0, 1.0)  # (BT, L)
        fg_mask = fg_coverage.reshape(BT, 1, S, S)
        decoder_in = fg_fmap * fg_mask + bg_fmap * (1.0 - fg_mask)

        # Shared CNN decoder
        decoded = self.conv_decoder(decoder_in)  # (BT, 3, H, W)
        decoded = decoded.reshape(list(batch_shape) + list(self._img_shape))
        decoded = decoded.permute(0, 1, 3, 4, 2)  # (B, T, H, W, C)
        return decoded

    # ================================================================
    # Training
    # ================================================================
    def _train(self, data):
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                self._wm_train_step += 1
                step_val = self._wm_train_step.item()

                # KL annealing
                if self._annealing:
                    if step_val < self._res_kl_anneal_steps:
                        progress = step_val / self._res_kl_anneal_steps
                        res_kl_scale = self._res_kl_scale_init - progress * (self._res_kl_scale_init - self._res_kl_scale)
                    else:
                        res_kl_scale = self._res_kl_scale
                else:
                    res_kl_scale = self._res_kl_scale

                # 1. Encode with F_8 intercept
                f8_proj, raw_embed, _ = self._encode_with_f8(data)
                B, T, E = raw_embed.shape
                BT = B * T

                # 2. Compute attention and extract entity features
                pooled_for_rssm, pooled_for_attn, attn_map = self._compute_attention(f8_proj)

                # Attention gradient flow: optionally detach attention for the decoder.
                attn_alpha = 1.0
                if self._detach_attn_for_recon:
                    attn_for_recon = attn_map.detach()
                else:
                    attn_for_recon = attn_map
                pooled_for_rssm_final = pooled_for_rssm.reshape(B, T, -1)

                pooled_for_attn_bt = pooled_for_attn.reshape(B, T, -1)

                # 3. proj_dyn → RSSM
                embed_dyn = self.proj_dyn(pooled_for_rssm_final)
                post, prior = self.dynamics.observe(
                    embed_dyn, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )

                # 4. proj_res → z_res (VAE)
                embed_res, res_kl_loss = self._compute_residual(
                    f8_proj, raw_embed, attn_map,
                    detach_encoder=self._res_detach_encoder
                )

                # 5. Assembly + decode
                rssm_feat = self.dynamics.get_feat(post)
                L_dec = self.f8_spatial * self.f8_spatial
                attn_recon_bt = attn_for_recon.reshape(B, T, self.num_queries, L_dec)

                pred_img = self._decode_assembled(rssm_feat, embed_res, attn_recon_bt, (B, T))
                target_img = data["image"]

                # Image loss
                image_loss = (pred_img - target_img).pow(2).sum(dim=(-3, -2, -1))

                # 6. Other head losses
                losses = {"image": image_loss}
                for name in ["reward", "cont"]:
                    head = self.heads[name]
                    pred = head(rssm_feat)
                    if name in data:
                        losses[name] = -pred.log_prob(data[name])

                scaled = {key: value * self._scales.get(key, 1.0) for key, value in losses.items()}
                model_loss = sum(scaled.values()) + kl_loss + res_kl_scale * res_kl_loss

                # 7. Self-supervised attention training
                proj = self.infonce_proj(pooled_for_attn_bt)
                proj = F.normalize(proj, dim=-1)
                anchor = proj[:, :-1].reshape(-1, proj.shape[-1])
                positive = proj[:, 1:].reshape(-1, proj.shape[-1])
                N_samples = anchor.shape[0]
                neg_idx = torch.randint(0, N_samples, (N_samples,), device=anchor.device)
                negative = positive[neg_idx]
                pos_sim = (anchor * positive).sum(dim=-1) / self._infonce_temp
                neg_sim = (anchor * negative).sum(dim=-1) / self._infonce_temp
                attn_ctr_loss = -F.logsigmoid(pos_sim).mean() - F.logsigmoid(-neg_sim).mean()
                model_loss = model_loss + self._infonce_scale * attn_ctr_loss

                # Inverse dynamics. With an action delay of k (inv_delay=k), the issued
                # action a_j is applied at step j+k, so its visual effect is the
                # transition e_{j+k} -> e_{j+k+1}; we predict a_j from that pair.
                # k=0 is the standard adjacent-frame objective.
                k = self._inv_delay
                Ta = data['action'].shape[1]
                pooled_t = pooled_for_attn_bt[:, k:-1].reshape(-1, pooled_for_attn_bt.shape[-1])
                pooled_t1 = pooled_for_attn_bt[:, k + 1:].reshape(-1, pooled_for_attn_bt.shape[-1])
                inv_input = torch.cat([pooled_t, pooled_t1], dim=-1)
                inv_pred = self.attn_inv_head(inv_input)
                inv_target_raw = data['action'][:, 1:Ta - k].reshape(-1, data['action'].shape[-1])
                if self._continuous_actions:
                    # MSE regression for continuous actions (DMC)
                    attn_inv_loss = F.mse_loss(inv_pred, inv_target_raw)
                    inv_acc_val = 1.0 - attn_inv_loss.detach()
                else:
                    # CE classification for discrete actions (ShapeEnv)
                    inv_target_cls = inv_target_raw.argmax(dim=-1)
                    attn_inv_loss = F.cross_entropy(inv_pred, inv_target_cls)
                    inv_acc_val = (inv_pred.argmax(dim=-1) == inv_target_cls).float().mean()
                model_loss = model_loss + self._attn_inv_scale * attn_inv_loss

                # Attention reward prediction
                if self._attn_reward_scale > 0:
                    # Predict reward at each timestep from attention-pooled features
                    attn_reward_pred = self.attn_reward_head(pooled_for_attn_bt).squeeze(-1)  # (B, T)
                    reward_target = data['reward'].squeeze(-1) if data['reward'].dim() > 2 else data['reward']  # (B, T)
                    attn_reward_loss = F.mse_loss(attn_reward_pred, reward_target)
                    model_loss = model_loss + self._attn_reward_scale * attn_reward_loss
                else:
                    attn_reward_loss = torch.tensor(0.0)

            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        # Metrics
        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        metrics["res_kl_loss_raw"] = to_np(torch.mean(res_kl_loss))
        metrics["res_kl_loss"] = metrics["res_kl_loss_raw"] * res_kl_scale
        metrics["res_kl_scale_current"] = res_kl_scale
        metrics["attn_ctr_loss"] = to_np(attn_ctr_loss)
        metrics["attn_inv_loss"] = to_np(attn_inv_loss)
        metrics["attn_reward_loss"] = to_np(attn_reward_loss)
        metrics["attn_alpha"] = attn_alpha

        with torch.no_grad():
            metrics["attn_inv_acc"] = to_np(inv_acc_val) if torch.is_tensor(inv_acc_val) else inv_acc_val

            # Per-query cross-attention metrics.
            A = attn_map  # (BT, N, L)
            for qi in range(self.num_queries):
                aq = A[:, qi, :]
                entropy = -(aq * (aq + 1e-8).log()).sum(dim=-1).mean()
                metrics[f"xattn_entropy_q{qi}"] = to_np(entropy)
                metrics[f"xattn_max_q{qi}"] = to_np(aq.max(dim=-1).values.mean())
            # Query usage: fraction with max > 2/L (clearly non-uniform).
            max_per_query = A.max(dim=-1).values  # (BT, N)
            active = (max_per_query > 2.0 / (self.f8_spatial ** 2)).float().mean(dim=0)
            metrics["query_usage"] = to_np(active.mean())

        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(torch.mean(self.dynamics.get_dist(prior).entropy()))
            metrics["post_ent"] = to_np(torch.mean(self.dynamics.get_dist(post).entropy()))
            context = dict(
                embed=embed_dyn,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # ================================================================
    # Preprocessing
    # ================================================================
    def preprocess(self, obs):
        obs = {
            k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
            for k, v in obs.items()
        }
        obs["image"] = obs["image"] / 255.0
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            obs["discount"] = obs["discount"].unsqueeze(-1)
        assert "is_first" in obs
        assert "is_terminal" in obs
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        return obs

    # ================================================================
    # Video prediction (for logging)
    # ================================================================
    def video_pred(self, data):
        data = self.preprocess(data)
        f8_proj, raw_embed, _ = self._encode_with_f8(data)
        B_full, T_full, E = raw_embed.shape

        pooled_for_rssm, pooled_for_attn, attn_map = self._compute_attention(f8_proj)

        embed_dyn = self.proj_dyn(pooled_for_rssm.reshape(B_full, T_full, -1))
        res_stats = self.proj_res(raw_embed)
        embed_res, _ = torch.split(res_stats, self.res_dim, dim=-1)

        # Observe context
        states, _ = self.dynamics.observe(
            embed_dyn[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        feat = self.dynamics.get_feat(states)

        S = self.assembly_spatial
        L = S * S
        attn_ctx = attn_map.reshape(B_full, T_full, self.num_queries, L)[:6, :5]

        recon = self._decode_assembled(feat, embed_res[:6, :5], attn_ctx, (6, 5))

        # Imagine (open loop)
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        prior_feat = self.dynamics.get_feat(prior)
        T_open = prior_feat.shape[1]

        # Uniform attention during open-loop imagination: each query gets 1/L per position.
        uniform_attn = torch.ones(
            6, T_open, self.num_queries, L, device=prior_feat.device,
        ) / L

        dummy_res = torch.zeros(6, T_open, self.res_dim, device=prior_feat.device)
        openl = self._decode_assembled(prior_feat, dummy_res, uniform_attn, (6, T_open))

        model = torch.cat([recon, openl], 1)
        truth = data["image"][:6]
        error = (model - truth + 1.0) / 2.0
        return torch.cat([truth, model, error], 2)
