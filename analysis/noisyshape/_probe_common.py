"""Shared helpers for the NoisyShape-family probe scripts (Appendix C.2).

* Trajectory cache: random-policy episodes collected once per environment with
  :func:`shapeenv.collect.collect_trajectories` and stored as an ``.npz`` in
  ``--cache_dir`` so that every model is probed on identical data.
* Model loading through ``analysis/wm_loader.py`` (any ``--model`` type, release
  runs and the original paper runs alike).
* Seeded open-loop imagination actions.

Reproducibility note: the original probe scripts drew the random-policy and the
imagined actions from the unseeded global ``np.random`` (NoisyShape family), and
probe initialisation / minibatch order from the unseeded torch RNG. Here all of
them are seeded, so re-running gives identical numbers, but they can differ
slightly from the numbers in the paper.
"""
import argparse
import os
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # analysis/  (wm_loader)
sys.path.insert(0, str(HERE.parent.parent))    # StarWM/    (train, models, ...)

import wm_loader  # noqa: E402


def default_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def add_probe_args(parser, num_curr=500, num_imag=200):
    parser.add_argument("run_dirs", nargs="+", help="run directories (each with a checkpoint)")
    parser.add_argument("--env", default=None,
                        help="shapeenv name; default: inferred from the first run's config")
    parser.add_argument("--checkpoint", default="latest.pt",
                        help="checkpoint path relative to each run dir")
    parser.add_argument("--cache_dir", default="runs/probe_cache",
                        help="where the shared trajectory cache (.npz) is stored")
    parser.add_argument("--davis_path", default=os.environ.get("DAVIS_DIR"),
                        help="DAVIS JPEGImages/480p directory (default: $DAVIS_DIR)")
    parser.add_argument("--num_curr", type=int, default=num_curr,
                        help="episodes for the H0 (encoding) probes")
    parser.add_argument("--num_imag", type=int, default=num_imag,
                        help="episodes for the imagination probes")
    parser.add_argument("--episode_length", type=int, default=200)
    parser.add_argument("--curr_env_seed", type=int, default=42)
    parser.add_argument("--imag_env_seed", type=int, default=12345)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--imag_action_seed", type=int, default=0,
                        help="seed of the random actions used for open-loop imagination")
    parser.add_argument("--probe_seed", type=int, default=0, help="torch seed for probe training")
    parser.add_argument("--results_subdir", default="probe_results")
    parser.add_argument("--force", action="store_true", help="overwrite existing results.json")
    parser.add_argument("--device", default=default_device())


def infer_env(run_dir, device="cpu"):
    config = wm_loader.load_run_config(run_dir, device=device)
    return config.task.split("_", 1)[0]


def _cache_path(cache_dir, env_name, num_episodes, env_seed, action_seed, episode_length):
    return os.path.join(cache_dir, f"traj_{env_name}_n{num_episodes}_env{env_seed}"
                                   f"_act{action_seed}_T{episode_length}.npz")


def get_trajectories(env_name, num_episodes, env_seed, cache_dir, episode_length=200,
                     davis_path=None, action_seed=None):
    """Cached ``(images, actions, is_first, labels)`` of a random policy.

    The policy RNG is seeded with ``action_seed`` (default: ``env_seed``).
    Labels are keyed by the env's info names (``color_id``, ``position_x``, ...).
    """
    import shapeenv
    from shapeenv.collect import collect_trajectories

    action_seed = env_seed if action_seed is None else action_seed
    path = _cache_path(cache_dir, env_name, num_episodes, env_seed, action_seed, episode_length)
    if os.path.exists(path):
        print(f"Loading cached trajectories from {path}")
        d = np.load(path)
        labels = {k[len("label_"):]: d[k] for k in d.files if k.startswith("label_")}
        return d["images"], d["actions"], d["is_first"], labels
    kwargs = {}
    if env_name in shapeenv.USES_DAVIS:
        kwargs["davis_path"] = davis_path
    print(f"Collecting {num_episodes} episodes of {env_name} (env seed {env_seed})...")
    images, actions, is_first, labels = collect_trajectories(
        env_name, num_episodes, env_seed=env_seed, action_seed=action_seed,
        episode_length=episode_length, progress=True, **kwargs)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = path[:-len(".npz")] + f".tmp{os.getpid()}.npz"
    np.savez(tmp, images=images, actions=actions, is_first=is_first,
             **{f"label_{k}": v for k, v in labels.items()})
    os.replace(tmp, path)  # atomic: concurrent probe jobs never read a partial file
    print(f"Saved cache to {path}")
    return images, actions, is_first, labels


def get_probe_data(args, env_name):
    """(curr, imag) trajectory tuples with the protocol's two disjoint env seeds."""
    curr = get_trajectories(env_name, args.num_curr, args.curr_env_seed, args.cache_dir,
                            args.episode_length, args.davis_path)
    imag = get_trajectories(env_name, args.num_imag, args.imag_env_seed, args.cache_dir,
                            args.episode_length, args.davis_path)
    print(f"  curr: {curr[0].shape}, imag: {imag[0].shape}")
    return curr, imag


def rename_labels(labels, mapping):
    """Flatten (E, T+1) label arrays and rename info keys -> probe keys."""
    return {new: np.asarray(labels[old]).reshape(-1) for old, new in mapping.items()
            if old in labels}


def action_sampler(num_actions, seed):
    """Uniform one-hot actions for imagination: ``sampler(B) -> (onehot (B,A), int (B,))``."""
    rng = np.random.RandomState(seed)

    def sample(B):
        ri = rng.randint(0, num_actions, size=B)
        ra = np.zeros((B, num_actions), dtype=np.float32)
        ra[np.arange(B), ri] = 1.0
        return ra, ri
    return sample


def results_path(run_dir, results_subdir):
    return os.path.join(run_dir, results_subdir, "results.json")


def load_model(run_dir, checkpoint, device):
    ckpt = os.path.join(run_dir, checkpoint)
    if not os.path.exists(ckpt):
        print(f"WARNING: no checkpoint {ckpt}, skipping")
        return None, None
    wm, config = wm_loader.load_world_model(run_dir, checkpoint=checkpoint, device=device)
    print(f"Loaded {config.model} from {ckpt}")
    return wm, config


def free():
    """Release cached GPU memory (call after ``del wm``)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_parser(doc, **kw):
    p = argparse.ArgumentParser(description=doc.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_probe_args(p, **kw)
    return p
