#!/usr/bin/env python3
"""CALVIN attention concentration with camera-projected masks (Appendix F.3 (iii);
helpers used by attn_table.py for Table 15).

Projects the end-effector position (robot_obs[:3]) and the three block centres
(scene_obs[6:9], [12:15], [18:21]) into the 64x64 image with the CALVIN static
camera (task_D_D calibration: look_from / look_at / up, fov 10, 200x200),
puts a disk around each (gripper r=5 px, blocks r=3 px) and measures how much
of StarWM's attention falls inside. Attention map per frame = max over the N
queries of the 8x8 maps, nearest-upsampled to 64x64, normalised to sum 1.
Concentration = attention share / area share; each region is compared to its
own shuffled null (same attention map scored against another frame's mask of
the same region, 5 permutations). The projection is sanity-checked by the
fraction of projected block pixels that are saturated (block colour).

Frames: 40 random held-out episodes of 500 frames (permutation seed 0),
t = 20, 45, ..., 495.

Example:
  python calvin/calvin_attn_metric_proj.py --ckpt runs/calvin/starwm/checkpoints/wm_step180000.pt \
      --data_dir <conv>/D_val --state_dir <conv>/D_val_state --device cuda:0
"""
import argparse
import glob
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from train_calvin_offline import build_config, build_wm  # noqa: E402

# CALVIN static camera (task_D_D .hydra/config.yaml)
LOOK_FROM = np.array([2.871459009488717, -2.166602199425597, 2.555159848480571])
LOOK_AT = np.array([-0.026242351159453392, -0.0302329882979393, 0.3920000493526459])
UP = np.array([0.4041403970338857, 0.22629790978217404, 0.8862616969685161])
FOV, ASPECT, NEAR, FAR, IMG = 10.0, 1.0, 0.01, 10.0, 200
BLK = [slice(6, 9), slice(12, 15), slice(18, 21)]   # red / blue / pink block xyz in scene_obs


def cam_matrix():
    f = LOOK_AT - LOOK_FROM; f /= np.linalg.norm(f)
    s = np.cross(f, UP); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    view = np.eye(4)
    view[0, :3] = s; view[1, :3] = u; view[2, :3] = -f
    view[0, 3] = -s @ LOOK_FROM; view[1, 3] = -u @ LOOK_FROM; view[2, 3] = f @ LOOK_FROM
    fy = 1.0 / np.tan(np.radians(FOV) / 2)
    proj = np.zeros((4, 4))
    proj[0, 0] = fy / ASPECT; proj[1, 1] = fy
    proj[2, 2] = (FAR + NEAR) / (NEAR - FAR); proj[2, 3] = 2 * FAR * NEAR / (NEAR - FAR)
    proj[3, 2] = -1
    return proj @ view


PV = cam_matrix()


def project(pts, out=64):
    """world (n,3) -> pixel (n,2) in an out x out image."""
    p = np.c_[pts, np.ones(len(pts))]
    clip = p @ PV.T
    ndc = clip[:, :3] / clip[:, 3:4]
    x = (ndc[:, 0] + 1) / 2 * IMG
    y = (1 - (ndc[:, 1] + 1) / 2) * IMG
    return np.c_[x, y] * (out / IMG)


def load_wm(ckpt, cfg):
    """StarWM from a train_calvin_offline.py checkpoint ({'wm': state_dict, 'step'})."""
    wm = build_wm(cfg)
    st = torch.load(ckpt, map_location=cfg.device, weights_only=False)
    wm.load_state_dict(st["wm"])
    print(f"[metric] loaded {os.path.basename(ckpt)} @step {st.get('step', '?')}", flush=True)
    return wm.eval()


@torch.no_grad()
def get_attn(wm, cfg, imgs):
    """imgs (T,64,64,3) uint8 -> per-query attention (T,N,64,64), nearest-upsampled, sum 1."""
    dev = cfg.device
    T = imgs.shape[0]
    data = {"image": torch.tensor(imgs[None], device=dev), "action": torch.zeros(1, T, 7, device=dev),
            "is_first": torch.zeros(1, T, device=dev), "is_terminal": torch.zeros(1, T, device=dev)}
    data = wm.preprocess(data)
    f8, _, _ = wm._encode_with_f8(data)
    _, _, attn = wm._compute_attention(f8)
    N, S = wm.num_queries, wm.f8_spatial
    a = attn.reshape(T, N, S, S).cpu().numpy()
    up = np.kron(a, np.ones((1, 1, 64 // S, 64 // S)))
    return up / (up.sum(axis=(-2, -1), keepdims=True) + 1e-8)


def disk(cx, cy, r, n=64):
    yy, xx = np.mgrid[0:n, 0:n]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def val_files(data_dir, n_eps, seed=0):
    """n_eps random full-length (500-frame) held-out episodes."""
    files = sorted(f for f in glob.glob(os.path.join(data_dir, "*.npz")) if f.endswith("-500.npz"))
    rng = np.random.RandomState(seed)
    return [files[i] for i in rng.permutation(len(files))[:n_eps]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True); ap.add_argument("--data_dir", required=True)
    ap.add_argument("--state_dir", required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_eps", type=int, default=40); ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--r_grip", type=int, default=5); ap.add_argument("--r_block", type=int, default=3)
    a = ap.parse_args()
    cfg = build_config("starwm", {"device": a.device, "seed": 0})
    wm = load_wm(a.ckpt, cfg)
    files = val_files(a.data_dir, a.n_eps)

    og, ob, oc, ag, ab_, red_ok, attn_all, mg_all, mc_all, mb_all = ([] for _ in range(10))
    for f in files:
        imgs = np.load(f)["image"]
        s = np.load(os.path.join(a.state_dir, os.path.basename(f)))
        robot, scene = s["robot_obs"], s["scene_obs"]
        idx = np.arange(20, 500, a.stride)
        attn = get_attn(wm, cfg, imgs[idx])
        comb = attn.max(axis=1); comb = comb / (comb.sum(axis=(-2, -1), keepdims=True) + 1e-8)
        for jj, t in enumerate(idx):
            ee = project(robot[t:t + 1, :3])[0]
            blks = project(np.stack([scene[t, b] for b in BLK]))
            H, W = 64, 64
            mg = disk(ee[0], ee[1], a.r_grip) if (0 <= ee[0] < W and 0 <= ee[1] < H) else np.zeros((64, 64), bool)
            mc = np.zeros((64, 64), bool)
            for bx, by in blks:
                if 0 <= bx < W and 0 <= by < H:
                    mc |= disk(bx, by, a.r_block)
            if not mg.any() and not mc.any():
                continue
            mall = mg | mc
            og.append(comb[jj][mg].sum() if mg.any() else 0.0)
            ob.append(comb[jj][mc].sum() if mc.any() else 0.0)
            oc.append(comb[jj][mall].sum())
            ag.append(mg.mean()); ab_.append(mall.mean())
            # projection check: is the projected first-block pixel saturated (a block colour)?
            bx, by = blks[0]
            if 0 <= bx < 64 and 0 <= by < 64:
                px = imgs[t][int(by), int(bx)]
                hsv = cv2.cvtColor(px.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0]
                red_ok.append(1.0 if hsv[1] > 100 else 0.0)
            attn_all.append(comb[jj]); mg_all.append(mg); mc_all.append(mall); mb_all.append(mc)
    og, ob, oc, ag, ab_ = map(np.array, (og, ob, oc, ag, ab_))

    def shuffled_null(masks, n_rep=5):
        """Score each attention map against ANOTHER frame's mask of the SAME region.
        Returns (mean area, mean attention share, mean concentration)."""
        ar, sh, cc = [], [], []
        for r in range(n_rep):
            pm = np.random.RandomState(100 + r).permutation(len(masks))
            for i in range(len(masks)):
                m = masks[pm[i]]
                if not m.any():
                    continue
                a_ = float(m.mean()); s_ = float(attn_all[i][m].sum())
                ar.append(a_); sh.append(s_); cc.append(s_ / (a_ + 1e-8))
        return np.mean(ar), np.mean(sh), np.mean(cc)

    null_g = shuffled_null(mg_all)      # gripper-tip null
    null_c = shuffled_null(mc_all)      # EE + blocks null
    null_b = shuffled_null(mb_all)      # blocks-only null
    print(f"\n=== CALVIN attention (projected masks, StarWM, {len(oc)} frames) ===")
    print(f"  projection check: {np.mean(red_ok) * 100:.0f}% of projected block pixels are saturated (block colour)")
    print(f"  gripper-end (r={a.r_grip}px): {np.mean(og) * 100:.1f}% attn / {np.mean(ag) * 100:.1f}% area = "
          f"{np.mean(og / (ag + 1e-8)):.1f}x (mean of per-frame ratios)")
    print(f"  blocks (r={a.r_block}px):     {np.mean(ob) * 100:.1f}% attn")
    print(f"  combined (gripper+blocks):  {np.mean(oc) * 100:.1f}% attn / {np.mean(ab_) * 100:.1f}% area = "
          f"{np.mean(oc / (ab_ + 1e-8)):.1f}x")
    print()
    print(f"  NULL gripper-tip : area {null_g[0] * 100:4.1f}%  attn {null_g[1] * 100:5.1f}%  = {null_g[2]:.1f}x")
    print(f"  NULL EE+blocks   : area {null_c[0] * 100:4.1f}%  attn {null_c[1] * 100:5.1f}%  = {null_c[2]:.1f}x")
    ab2 = np.mean([m.mean() for m in mb_all if m.any()])
    ob2 = np.mean([o for o, m in zip(ob, mb_all) if m.any()])
    print(f"  BLOCKS only      : area {ab2 * 100:4.1f}%  attn {ob2 * 100:5.1f}%  = {ob2 / ab2:.1f}x"
          f"   vs its own null {null_b[2]:.1f}x")


if __name__ == "__main__":
    main()
