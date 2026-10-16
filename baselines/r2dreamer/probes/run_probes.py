"""
NoisyShape probe analysis for R2-Dreamer checkpoints (StarWM paper, Fig. 3).

Loads a trained agent from <run>/latest.pt, collects trajectories on
NoisyShape (``shapeenv``), extracts RSSM latent features [stoch; deter], and
trains MLP probes for color/shape/size (classification) and pos_x/pos_y
(regression). Also runs imagined rollouts for H=1..15 and reports
corrected-label accuracy / MSE per horizon.

Output (<run>/probe_results/results.json) uses the same format as the StarWM
probe script, so the same plotting code reads it.

Copied into the R2-Dreamer checkout by ``baselines/setup_baselines.sh``.
Usage (from the R2-Dreamer checkout):
  python run_probes.py --logdir runs/baselines/r2dreamer/noisyshape \
      --run_prefix noisyshape --seeds 0 1 2 [--davis_path $DAVIS_DIR]
"""
import argparse
import json
import os
import pathlib
import pickle
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).parent))

import shapeenv
from dreamer import Dreamer
import envs.wrappers as wrappers


# DAVIS 2017 JPEGImages/480p directory (set from --davis_path / $DAVIS_DIR in main()).
DAVIS_PATH = None


# ----------------------------- data collection -----------------------------

def make_env(size=(64, 64), seed=42):
    raw = shapeenv.NoisyShapeEnv(size=size, seed=seed, davis_path=DAVIS_PATH)
    env = wrappers.OneHotAction(wrappers.FiveToFourTupleStep(raw))
    env = wrappers.TimeLimit(env, 200)
    return env, raw


def collect_trajectories(num_episodes, seed=42, ep_len=200):
    env, raw = make_env(seed=seed)
    all_images, all_actions, all_is_first = [], [], []
    labels = {k: [] for k in ["color", "shape", "size", "pos_x", "pos_y"]}
    for _ in tqdm(range(num_episodes), desc=f"Collect seed={seed}"):
        obs = env.reset()
        ep_imgs = [obs["image"]]
        ep_isf = [1.0]
        ep_acts = []
        ep_lab = {
            "color": [raw._color_id],
            "shape": [raw._shape_id],
            "size": [raw._size_id],
            "pos_x": [raw._pos[0]],
            "pos_y": [raw._pos[1]],
        }
        for _ in range(ep_len):
            ai = np.random.randint(0, 4)
            a = np.zeros(4, dtype=np.float32)
            a[ai] = 1.0
            obs, _r, done, info = env.step(a)
            ep_imgs.append(obs["image"])
            ep_isf.append(0.0)
            ep_acts.append(a)
            ep_lab["color"].append(info["color_id"])
            ep_lab["shape"].append(info["shape_id"])
            ep_lab["size"].append(info["size_id"])
            ep_lab["pos_x"].append(info["position_x"])
            ep_lab["pos_y"].append(info["position_y"])
            if done:
                break
        all_images.append(np.stack(ep_imgs))
        all_actions.append(np.stack(ep_acts))
        all_is_first.append(np.array(ep_isf, dtype=np.float32))
        for k in labels:
            labels[k].append(np.array(ep_lab[k]))
    env.close()
    # Ensure all episodes have the same length so we can stack.
    T = min(x.shape[0] for x in all_images)
    all_images = np.stack([x[:T] for x in all_images])
    all_actions = np.stack([x[:T - 1] for x in all_actions])
    all_is_first = np.stack([x[:T] for x in all_is_first])
    labels = {k: np.stack([v[:T] for v in labels[k]]) for k in labels}
    return all_images, all_actions, all_is_first, labels


def get_or_create_cache(cache_path, seed_curr=42, seed_imag=12345, num_episodes=500):
    if os.path.exists(cache_path):
        print(f"Loading trajectory cache: {cache_path}")
        with open(cache_path, "rb") as f:
            d = pickle.load(f)
        return d["curr"], d["imag"]
    print("Collecting trajectories (this takes a few minutes)...")
    curr = collect_trajectories(num_episodes, seed=seed_curr)
    imag = collect_trajectories(max(num_episodes // 2, 100), seed=seed_imag)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"curr": curr, "imag": imag}, f)
    print(f"Saved cache: {cache_path}")
    return curr, imag


# ----------------------------- model loading -----------------------------

def load_dreamer(logdir, device):
    """Reconstruct the Dreamer used during training and load the checkpoint."""
    cfg_path = pathlib.Path(logdir) / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing saved hydra config: {cfg_path}")
    config = OmegaConf.load(cfg_path)
    # OmegaConf interpolations like ${device} need a concrete value.
    OmegaConf.update(config, "device", device, force_add=True)
    OmegaConf.update(config, "model.device", device, force_add=True)

    # Build env to recover obs/action spaces the model was trained with.
    env, _ = make_env(size=tuple(config.env.size), seed=0)
    obs_space = env.observation_space
    act_space = env.action_space
    env.close()

    # Avoid torch.compile during eval (constructor triggers it otherwise).
    try:
        OmegaConf.update(config, "model.compile", False, force_add=True)
    except Exception:
        pass

    agent = Dreamer(config.model, obs_space, act_space).to(device)
    ckpt = torch.load(pathlib.Path(logdir) / "latest.pt", map_location=device, weights_only=False)
    missing, unexpected = agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
    if missing:
        print(f"  missing({len(missing)}): {missing[:3]}")
    if unexpected:
        print(f"  unexpected({len(unexpected)}): {unexpected[:3]}")
    agent.eval()
    return agent, config


# ----------------------------- feature extraction -----------------------------

@torch.no_grad()
def encode_current(agent, images, actions, is_first, device, batch_size=16):
    """Run RSSM posterior over trajectories and return flat (N_steps, F) features."""
    N_ep, Tp1 = images.shape[:2]
    T = Tp1 - 1
    feats_all = []
    for bs in tqdm(range(0, N_ep, batch_size), desc="Encode current"):
        be = min(bs + batch_size, N_ep)
        B = be - bs
        imgs = torch.tensor(images[bs:be], dtype=torch.float32, device=device) / 255.0  # (B,T+1,H,W,C)
        acts = torch.tensor(actions[bs:be], dtype=torch.float32, device=device)         # (B,T,A)
        isf = torch.tensor(is_first[bs:be], dtype=torch.bool, device=device)            # (B,T+1)

        stoch, deter = agent.rssm.initial(B)
        prev_action = torch.zeros(B, agent.act_dim, dtype=torch.float32, device=device)
        feats_ep = []
        for t in range(Tp1):
            op = {"image": imgs[:, t]}
            embed = agent.encoder(op)                  # (B, E)
            reset = isf[:, t]
            stoch, deter, _logit = agent.rssm.obs_step(stoch, deter, prev_action, embed, reset)
            feats_ep.append(agent.rssm.get_feat(stoch, deter).cpu().numpy())
            if t < T:
                prev_action = acts[:, t]
        feats_all.append(np.stack(feats_ep, axis=1).reshape(B * Tp1, -1))  # (B*T+1, F)
    return np.concatenate(feats_all, axis=0)


@torch.no_grad()
def encode_with_imagination(agent, images, actions, is_first, device, horizon=15, batch_size=16):
    """For every posterior state, run a prior rollout of length `horizon`.

    Returns:
        real_feats: (N_steps, F)
        imag_feats: (horizon, N_steps, F)
        imag_actions: (horizon, N_steps) int action indices
    """
    N_ep, Tp1 = images.shape[:2]
    T = Tp1 - 1
    real_all, imag_all, acts_all = [], [[] for _ in range(horizon)], [[] for _ in range(horizon)]
    for bs in tqdm(range(0, N_ep, batch_size), desc="Encode + imagine"):
        be = min(bs + batch_size, N_ep)
        B = be - bs
        imgs = torch.tensor(images[bs:be], dtype=torch.float32, device=device) / 255.0
        acts = torch.tensor(actions[bs:be], dtype=torch.float32, device=device)
        isf = torch.tensor(is_first[bs:be], dtype=torch.bool, device=device)

        stoch, deter = agent.rssm.initial(B)
        prev_action = torch.zeros(B, agent.act_dim, dtype=torch.float32, device=device)
        real_ep, imag_ep = [], [[] for _ in range(horizon)]
        act_ep = [[] for _ in range(horizon)]
        for t in range(Tp1):
            op = {"image": imgs[:, t]}
            embed = agent.encoder(op)
            reset = isf[:, t]
            stoch, deter, _ = agent.rssm.obs_step(stoch, deter, prev_action, embed, reset)
            real_ep.append(agent.rssm.get_feat(stoch, deter).cpu().numpy())
            # imagination from this posterior state
            istoch, ideter = stoch.clone(), deter.clone()
            for h in range(horizon):
                ri = np.random.randint(0, 4, size=B)
                ra = np.zeros((B, agent.act_dim), dtype=np.float32)
                ra[np.arange(B), ri] = 1.0
                ra_t = torch.tensor(ra, dtype=torch.float32, device=device)
                istoch, ideter = agent.rssm.img_step(istoch, ideter, ra_t)
                imag_ep[h].append(agent.rssm.get_feat(istoch, ideter).cpu().numpy())
                act_ep[h].append(ri)
            if t < T:
                prev_action = acts[:, t]
        real_all.append(np.stack(real_ep, axis=1).reshape(B * Tp1, -1))
        for h in range(horizon):
            imag_all[h].append(np.stack(imag_ep[h], axis=1).reshape(B * Tp1, -1))
            acts_all[h].append(np.stack(act_ep[h], axis=1).reshape(B * Tp1))
    real_feats = np.concatenate(real_all, axis=0)
    imag_feats = np.stack([np.concatenate(x, axis=0) for x in imag_all], axis=0)
    imag_actions = np.stack([np.concatenate(x, axis=0) for x in acts_all], axis=0)
    return real_feats, imag_feats, imag_actions


# ----------------------------- probes -----------------------------

class MLPClf(nn.Module):
    def __init__(self, d, c, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, c),
        )

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            dev = next(self.parameters()).device
            return self.forward(torch.tensor(x, dtype=torch.float32, device=dev)).argmax(-1).cpu().numpy()


class MLPReg(nn.Module):
    def __init__(self, d, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, 1),
        )

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            dev = next(self.parameters()).device
            return self.forward(torch.tensor(x, dtype=torch.float32, device=dev)).cpu().numpy().squeeze(-1)


def train_clf(feats, labels, n_cls, device, epochs=100, patience=10):
    Xtr, Xte, ytr, yte = train_test_split(feats, labels, test_size=0.3, random_state=42, stratify=labels)
    m = MLPClf(Xtr.shape[1], n_cls).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.long, device=device)
    Xv = torch.tensor(Xte, dtype=torch.float32, device=device)
    yv = torch.tensor(yte, dtype=torch.long, device=device)
    ld = DataLoader(TensorDataset(Xt, yt), batch_size=512, shuffle=True)
    bl, bs2, ni = float("inf"), None, 0
    for _ in range(epochs):
        m.train()
        for xb, yb in ld:
            opt.zero_grad()
            crit(m(xb), yb).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            tl = crit(m(Xv), yv).item()
        if tl < bl:
            bl, bs2, ni = tl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            ni += 1
        if ni >= patience:
            break
    if bs2:
        m.load_state_dict(bs2)
    m.eval()
    with torch.no_grad():
        tra = (m(Xt).argmax(-1) == yt).float().mean().item()
        tea = (m(Xv).argmax(-1) == yv).float().mean().item()
    return tra, tea, m


def train_reg(feats, labels, device, epochs=100, patience=10):
    Xtr, Xte, ytr, yte = train_test_split(feats, labels, test_size=0.3, random_state=42)
    m = MLPReg(Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.MSELoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.float32, device=device).unsqueeze(-1)
    Xv = torch.tensor(Xte, dtype=torch.float32, device=device)
    yv = torch.tensor(yte, dtype=torch.float32, device=device).unsqueeze(-1)
    ld = DataLoader(TensorDataset(Xt, yt), batch_size=512, shuffle=True)
    bl, bs2, ni = float("inf"), None, 0
    for _ in range(epochs):
        m.train()
        for xb, yb in ld:
            opt.zero_grad()
            crit(m(xb), yb).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            tl = crit(m(Xv), yv).item()
        if tl < bl:
            bl, bs2, ni = tl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            ni += 1
        if ni >= patience:
            break
    if bs2:
        m.load_state_dict(bs2)
    m.eval()
    with torch.no_grad():
        trm = crit(m(Xt), yt).item()
        tem = crit(m(Xv), yv).item()
    return trm, tem, m


def compute_imagined_labels(start_labels, imagined_actions, horizon):
    """V2 action logic: a0 color-1, a1 color+1, a2 shape_size-1, a3 shape_size+1."""
    cc = start_labels["color"].copy().astype(np.int64)
    sc = start_labels["shape"].copy().astype(np.int64) * 3 + start_labels["size"].copy().astype(np.int64)
    out = {}
    for h in range(horizon):
        a = imagined_actions[h]
        cc = cc.copy()
        sc = sc.copy()
        cc[a == 0] -= 1
        cc[a == 1] += 1
        sc[a == 2] -= 1
        sc[a == 3] += 1
        out[h + 1] = {
            "color": (cc % 4).astype(np.int64),
            "shape": ((sc % 12) // 3).astype(np.int64),
            "size": ((sc % 12) % 3).astype(np.int64),
        }
    return out


def evaluate_checkpoint(agent, curr_data, imag_data, device, horizon=15):
    ci, ca, cf, cl = curr_data
    ii, ia, isf, il = imag_data
    flat_cl = {k: v.reshape(-1) for k, v in cl.items()}
    flat_il = {k: v.reshape(-1) for k, v in il.items()}

    print("  Encoding current trajectories...")
    feats = encode_current(agent, ci, ca, cf, device)
    print(f"  feats: {feats.shape}")

    current_probes = {}
    for nm, nc in [("color", 4), ("shape", 4), ("size", 3)]:
        tra, tea, _ = train_clf(feats, flat_cl[nm], nc, device)
        current_probes[nm] = {"train_acc": tra, "test_acc": tea}
        print(f"    {nm}: test_acc={tea:.4f}")
    for nm in ["pos_x", "pos_y"]:
        trm, tem, _ = train_reg(feats, flat_cl[nm], device)
        current_probes[nm] = {"train_mse": trm, "test_mse": tem}
        print(f"    {nm}: test_mse={tem:.6f}")

    print("  Encoding + imagining...")
    rf, imf, imacts = encode_with_imagination(agent, ii, ia, isf, device, horizon=horizon)
    print(f"  real={rf.shape} imagined={imf.shape}")

    corrected = compute_imagined_labels(flat_il, imacts, horizon)

    imagined_probes = {"0": {}}
    clfs, regs = {}, {}
    for nm, nc in [("color", 4), ("shape", 4), ("size", 3)]:
        _, tea, clf = train_clf(rf, flat_il[nm], nc, device)
        clfs[nm] = clf
        imagined_probes["0"][nm] = float(tea)
    for nm in ["pos_x", "pos_y"]:
        _, tem, reg = train_reg(rf, flat_il[nm], device)
        regs[nm] = reg
        imagined_probes["0"][nm] = float(tem)

    for h in tqdm(range(horizon), desc="  Horizons"):
        hf = imf[h]
        imagined_probes[str(h + 1)] = {}
        for nm in ["color", "shape", "size"]:
            pred = clfs[nm].predict(hf)
            imagined_probes[str(h + 1)][nm] = float(accuracy_score(corrected[h + 1][nm], pred))
        for nm in ["pos_x", "pos_y"]:
            pred = regs[nm].predict(hf)
            # Position is re-randomized every step, so we expect the MSE to
            # stabilize at the variance of the marginal distribution.
            imagined_probes[str(h + 1)][nm] = float(mean_squared_error(flat_il[nm], pred))

    return {"current_probes": current_probes, "imagined_probes": imagined_probes}


# ----------------------------- main -----------------------------

def main():
    global DAVIS_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True,
                        help="directory containing the runs <run_prefix>_s<seed>")
    parser.add_argument("--run_prefix", default="noisyshape")
    parser.add_argument("--davis_path", default=os.environ.get("DAVIS_DIR"),
                        help="DAVIS 2017 JPEGImages/480p directory (default: $DAVIS_DIR)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--num_episodes", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.davis_path or not os.path.isdir(args.davis_path):
        raise SystemExit("Set --davis_path or $DAVIS_DIR to the DAVIS 2017 JPEGImages/480p directory.")
    DAVIS_PATH = args.davis_path

    device = args.device
    cache_path = os.path.join(args.logdir, f"traj_cache_noisyshape_{args.num_episodes}.pkl")
    curr_data, imag_data = get_or_create_cache(cache_path, num_episodes=args.num_episodes)
    print(f"curr: images={curr_data[0].shape}")
    print(f"imag: images={imag_data[0].shape}")

    for seed in args.seeds:
        exp_dir = os.path.join(args.logdir, f"{args.run_prefix}_s{seed}")
        ckpt = os.path.join(exp_dir, "latest.pt")
        out_dir = os.path.join(exp_dir, "probe_results")
        res_path = os.path.join(out_dir, "results.json")
        if not os.path.exists(ckpt):
            print(f"\n[seed {seed}] missing checkpoint, skip")
            continue
        if os.path.exists(res_path) and not args.force:
            print(f"\n[seed {seed}] already has results, skip (use --force to rerun)")
            continue
        print(f"\n==== seed {seed} ====")
        os.makedirs(out_dir, exist_ok=True)
        agent, _config = load_dreamer(exp_dir, device)
        results = evaluate_checkpoint(agent, curr_data, imag_data, device, horizon=args.horizon)
        with open(res_path, "w") as f:
            json.dump({"r2dreamer": results}, f, indent=2)
        print(f"  saved: {res_path}")
        del agent
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
