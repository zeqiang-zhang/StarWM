"""'+ attention only' ablation world model (Table 4).

Spatial attention routing without the dual-stream decoder: a single spatial
softmax map (1x1 conv over the 8x8 projected features, i.e. one "query") pools
the encoder features into the RSSM input, and the decoder reconstructs only the
attended content (attention-weighted broadcast of an MLP of [h; z]); there is
no residual/background stream. The attention is trained by the same inverse
dynamics + binary InfoNCE losses as StarWM and is stop-gradiented for both the
RSSM input and the decoder. This is the configuration run for the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import networks
import tools
from tools import to_np


class AttentionOnlyWorldModel(nn.Module):
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

        self.f8_channels = cnn_depth * (2 ** (stages - 2))  # 128
        self.f8_spatial = minres * 2  # 8
        self.f4_channels = cnn_depth * (2 ** (stages - 1))  # 256
        self.f4_spatial = minres  # 4

        encoder_cnn = self.encoder._cnn
        num_layers_per_stage = 3 if config.encoder.get('norm', True) else 2
        split_idx = (stages - 1) * num_layers_per_stage
        self.encoder_to_8x8 = nn.Sequential(*list(encoder_cnn.layers)[:split_idx])
        self.encoder_8x8_to_4x4 = nn.Sequential(*list(encoder_cnn.layers)[split_idx:])

        # ===== 3. 1×1 Conv projection: 128 → proj_dim (256) =====
        self.proj_dim = getattr(config, 'crossattn_proj_dim', 256)
        self.conv_1x1 = nn.Conv2d(self.f8_channels, self.proj_dim, kernel_size=1)
        nn.init.xavier_uniform_(self.conv_1x1.weight)
        nn.init.zeros_(self.conv_1x1.bias)
        print(f"[ATTN_ONLY] 1×1 Conv projection: {self.f8_channels} → {self.proj_dim}")

        # ===== 4. Spatial attention: 1x1 conv -> softmax over the 8x8 grid =====
        self.spatial_attn = nn.Conv2d(self.proj_dim, 1, kernel_size=1)
        nn.init.zeros_(self.spatial_attn.bias)
        print(f"[ATTN_ONLY] 1×1 conv spatial softmax at {self.f8_spatial}×{self.f8_spatial}")

        # ===== 5. Self-supervised training (inverse + contrastive) =====
        self._infonce_scale = getattr(config, "infonce_scale", 1.0)
        self._infonce_temp = getattr(config, "infonce_temp", 0.1)
        self._attn_inv_scale = getattr(config, "attn_inv_scale", 1.0)
        self._attn_anneal_steps = getattr(config, "attn_anneal_steps", 0)

        infonce_proj_dim = getattr(config, "infonce_proj_dim", 128)
        inv_input_dim = self.proj_dim
        self.infonce_proj = nn.Sequential(
            nn.Linear(inv_input_dim, inv_input_dim),
            nn.SiLU(),
            nn.Linear(inv_input_dim, infonce_proj_dim)
        ).to(config.device)

        self._num_actions = getattr(config, "num_actions", 4)
        task = getattr(config, 'task', '')
        self._continuous_actions = task.startswith('dmc_')
        self.attn_inv_head = nn.Sequential(
            nn.Linear(inv_input_dim * 2, inv_input_dim),
            nn.SiLU(),
            nn.Linear(inv_input_dim, self._num_actions)
        ).to(config.device)
        print(f"[ATTN_ONLY] Inverse dynamics: scale={self._attn_inv_scale}")

        # ===== 6. proj_dyn → RSSM =====
        rssm_embed_dim = self.proj_dim

        self.dyn_dim = getattr(config, "dyn_dim", 0) or rssm_embed_dim
        self.proj_dyn = nn.Sequential(
            nn.Linear(rssm_embed_dim, config.units),
            nn.LayerNorm(config.units) if config.norm == 'layer' else nn.Identity(),
            nn.SiLU(),
            nn.Linear(config.units, self.dyn_dim)
        ).to(config.device)
        print(f"[ATTN_ONLY] proj_dyn: {rssm_embed_dim} → {self.dyn_dim}")

        # ===== 7. RSSM =====
        self.dynamics = networks.RSSM(
            config.dyn_stoch, config.dyn_deter, config.dyn_hidden,
            config.dyn_rec_depth, config.dyn_discrete, config.act, config.norm,
            config.dyn_mean_act, config.dyn_std_act, config.dyn_min_std,
            config.unimix_ratio, config.initial, config.num_actions,
            self.dyn_dim, config.device,
        )

        if config.dyn_discrete:
            rssm_feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            rssm_feat_size = config.dyn_stoch + config.dyn_deter

        # ===== 8. Decoder: SINGLE STREAM (fg only) =====
        some_shape = list({k: tuple(v.shape) for k, v in obs_space.spaces.items()
                         if k not in ('is_first', 'is_last', 'is_terminal') and len(tuple(v.shape)) == 3}.values())[0]
        img_shape = (some_shape[-1],) + some_shape[:-1]  # (3, 64, 64)

        self.assembly_spatial = self.f8_spatial  # 8
        self.assembly_channels = self.proj_dim  # 256
        dec_start_res = self.f8_spatial
        dec_stages = int(math.log2(img_shape[1]) - math.log2(dec_start_res))  # 3

        # MLP_fg: rssm_feat → assembly_channels
        self.mlp_fg = nn.Sequential(
            nn.Linear(rssm_feat_size, config.units),
            nn.SiLU(),
            nn.Linear(config.units, self.assembly_channels)
        ).to(config.device)
        print(f"[ATTN_ONLY] MLP_fg: feat({rssm_feat_size}) → {self.assembly_channels} (broadcast to {self.assembly_spatial}×{self.assembly_spatial})")

        # ConvTranspose decoder: (assembly_channels, 8, 8) → (3, 64, 64)
        kernel_size = config.decoder.get('kernel_size', 4)
        val = 1 * (kernel_size - 1) - 2 + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        decoder_layers = []
        in_dim = self.assembly_channels
        for i in range(dec_stages):
            if i == dec_stages - 1:
                out_dim = img_shape[0]  # 3
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
        print(f"[ATTN_ONLY] Shared ConvDecoder: ({self.assembly_channels}, {self.assembly_spatial}, {self.assembly_spatial}) → {img_shape}")

        # ===== 9. Reward + Cont heads =====
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

        # ===== 10. Optimizer =====
        self._model_opt = tools.Optimizer(
            "model", self.parameters(), config.model_lr,
            config.opt_eps, config.grad_clip, config.weight_decay,
            opt=config.opt, use_amp=self._use_amp,
        )
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )
        print("[ATTN_ONLY] initialized (no residual stream, single-stream decoder).")
        self.register_buffer("_wm_train_step", torch.tensor(0, dtype=torch.long))

    # ================================================================
    # Encoder with F_8 intercept
    # ================================================================
    def _encode_with_f8(self, obs):
        inputs = torch.cat([obs[k] for k in self.encoder.cnn_shapes], -1)
        x = inputs.reshape((-1,) + tuple(inputs.shape[-3:]))
        x = x.permute(0, 3, 1, 2)
        x -= 0.5

        f8 = self.encoder_to_8x8(x)
        f8_proj = self.conv_1x1(f8)
        f4 = self.encoder_8x8_to_4x4(f8)
        raw_embed = f4.reshape(f4.shape[0], -1)

        batch_shape = inputs.shape[:-3]
        raw_embed = raw_embed.reshape(list(batch_shape) + [raw_embed.shape[-1]])

        return f8_proj, raw_embed, batch_shape

    # ================================================================
    # Attention: mode-specific
    # ================================================================
    def _compute_attention(self, f8_proj):
        BT = f8_proj.shape[0]

        attn_logits = self.spatial_attn(f8_proj)
        attn_weights = torch.softmax(attn_logits.reshape(BT, 1, -1), dim=-1)
        attn_weights = attn_weights.reshape(BT, 1, self.f8_spatial, self.f8_spatial)
        pooled = (attn_weights * f8_proj).sum(dim=(-2, -1))
        return pooled, pooled, attn_weights

    # ================================================================
    # Single-stream decoder
    # ================================================================
    def _decode_single(self, rssm_feat, attn_for_recon, batch_shape):
        """Decode using fg_fmap only (no background stream)."""
        BT = rssm_feat.shape[0] if rssm_feat.dim() == 2 else rssm_feat.shape[0] * rssm_feat.shape[1]
        feat_flat = rssm_feat.reshape(BT, -1)
        S = self.assembly_spatial

        # fg_vec: (BT, 256) -> broadcast to (BT, 256, 8, 8)
        fg_vec = self.mlp_fg(feat_flat)
        fg_fmap = fg_vec.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, S, S)
        attn = attn_for_recon.reshape(BT, 1, S, S)
        # Only the attended content is reconstructed
        decoder_in = fg_fmap * attn

        decoded = self.conv_decoder(decoder_in)
        decoded = decoded.reshape(list(batch_shape) + list(self._img_shape))
        decoded = decoded.permute(0, 1, 3, 4, 2)
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

                # 1. Encode with F_8 intercept
                f8_proj, raw_embed, _ = self._encode_with_f8(data)
                B, T, E = raw_embed.shape
                BT = B * T

                # 2. Compute attention
                pooled_for_rssm, pooled_for_attn, attn_map = self._compute_attention(f8_proj)

                # Stop-gradient attention for both the RSSM input and the decoder
                # (attn_anneal_steps=0 in the paper, i.e. attn_alpha = 1).
                H_sp, W_sp = self.f8_spatial, self.f8_spatial
                uniform_attn = torch.ones_like(attn_map) / (H_sp * W_sp)
                if self._attn_anneal_steps > 0:
                    attn_alpha = min(1.0, step_val / self._attn_anneal_steps)
                else:
                    attn_alpha = 1.0
                attn_for_recon = (1.0 - attn_alpha) * uniform_attn + attn_alpha * attn_map.detach()
                fmap = f8_proj
                pooled_sg = (attn_for_recon * fmap).sum(dim=(-2, -1))
                pooled_for_rssm_final = pooled_sg.reshape(B, T, -1)

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

                # 4. Single-stream decode (no proj_res)
                rssm_feat = self.dynamics.get_feat(post)
                attn_recon_bt = attn_for_recon.reshape(B, T, 1, self.f8_spatial, self.f8_spatial)

                pred_img = self._decode_single(rssm_feat, attn_recon_bt, (B, T))
                target_img = data["image"]

                # Image loss
                image_loss = (pred_img - target_img).pow(2).sum(dim=(-3, -2, -1))

                # 5. Other head losses
                losses = {"image": image_loss}
                for name in ["reward", "cont"]:
                    head = self.heads[name]
                    pred = head(rssm_feat)
                    if name in data:
                        losses[name] = -pred.log_prob(data[name])

                scaled = {key: value * self._scales.get(key, 1.0) for key, value in losses.items()}
                model_loss = sum(scaled.values()) + kl_loss

                # 6. Self-supervised attention training
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

                # Inverse dynamics
                pooled_t = pooled_for_attn_bt[:, :-1].reshape(-1, pooled_for_attn_bt.shape[-1])
                pooled_t1 = pooled_for_attn_bt[:, 1:].reshape(-1, pooled_for_attn_bt.shape[-1])
                inv_input = torch.cat([pooled_t, pooled_t1], dim=-1)
                inv_pred = self.attn_inv_head(inv_input)
                inv_target_raw = data['action'][:, 1:].reshape(-1, data['action'].shape[-1])
                if self._continuous_actions:
                    attn_inv_loss = F.mse_loss(inv_pred, inv_target_raw)
                    inv_acc_val = 1.0 - attn_inv_loss.detach()
                else:
                    inv_target_cls = inv_target_raw.argmax(dim=-1)
                    attn_inv_loss = F.cross_entropy(inv_pred, inv_target_cls)
                    inv_acc_val = (inv_pred.argmax(dim=-1) == inv_target_cls).float().mean()
                model_loss = model_loss + self._attn_inv_scale * attn_inv_loss

            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        # Metrics
        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        metrics["attn_ctr_loss"] = to_np(attn_ctr_loss)
        metrics["attn_inv_loss"] = to_np(attn_inv_loss)
        metrics["attn_alpha"] = attn_alpha

        with torch.no_grad():
            metrics["attn_inv_acc"] = to_np(inv_acc_val) if torch.is_tensor(inv_acc_val) else inv_acc_val

            aw = attn_map.reshape(BT, -1)
            attn_entropy = -(aw * (aw + 1e-8).log()).sum(dim=-1).mean()
            metrics["attn_entropy"] = to_np(attn_entropy)
            metrics["attn_max"] = to_np(aw.max(dim=-1).values.mean())

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

        pooled_for_rssm, _, attn_map = self._compute_attention(f8_proj)
        embed_dyn = self.proj_dyn(pooled_for_rssm.reshape(B_full, T_full, -1))

        states, _ = self.dynamics.observe(
            embed_dyn[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        feat = self.dynamics.get_feat(states)

        S = self.assembly_spatial
        attn_ctx = attn_map.reshape(B_full, T_full, 1, S, S)[:6, :5]

        recon = self._decode_single(feat, attn_ctx, (6, 5))

        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        prior_feat = self.dynamics.get_feat(prior)
        T_open = prior_feat.shape[1]

        uniform_attn = torch.ones(6, T_open, 1, S, S, device=prior_feat.device) / (S * S)

        openl = self._decode_single(prior_feat, uniform_attn, (6, T_open))

        model = torch.cat([recon, openl], 1)
        truth = data["image"][:6]
        error = (model - truth + 1.0) / 2.0
        return torch.cat([truth, model, error], 2)
