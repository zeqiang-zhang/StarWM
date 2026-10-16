"""Smoke test for a patched baseline checkout (run after setup_baselines.sh).

Builds environments through the baseline's own env factory and steps them a few
times on CPU; no training. One baseline per process (each has its own `envs`).

    python baselines/common/smoke_test.py r2dreamer    [--src DIR] [--dmc] [--noisyshape]
    python baselines/common/smoke_test.py isodream     [--src DIR] [--dmc] [--noisyshape]
    python baselines/common/smoke_test.py denoised_mdp [--src DIR] [--dmc]

--dmc needs DMCGB_DATA_DIR (and DAVIS_DIR for the R2-Dreamer Reacher check) and a
working MuJoCo renderer (MUJOCO_GL, default egl); --noisyshape needs the shapeenv
package and DAVIS_DIR. Without --dmc/--noisyshape only the imports are checked.
"""
import argparse
import os
import pathlib
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
BASELINES_ROOT = pathlib.Path(__file__).resolve().parents[1]
N_STEPS = 5


def _enter(src):
    os.chdir(src)
    sys.path.insert(0, str(src))


def _onehot(n, i):
    a = np.zeros(n, dtype=np.float32)
    a[i % n] = 1.0
    return a


def _report(name, obs):
    img = obs["image"]
    print(f"  OK {name}: image {tuple(img.shape)} {img.dtype} mean={float(np.mean(img)):.1f}")


def check_r2dreamer(src, args):
    _enter(src)
    from omegaconf import OmegaConf

    import envs
    import envs.dmc  # noqa: F401  (patched module)

    def make(cfg_name, **overrides):
        cfg = OmegaConf.load(f"configs/env/{cfg_name}.yaml")
        cfg.seed, cfg.device = 0, "cpu"
        for k, v in overrides.items():
            cfg[k] = v
        return envs.make_env(cfg, 0)

    davis = os.environ.get("DAVIS_DIR", "")
    if args.noisyshape:
        env = make("noisyshape", distracting_background_path=davis)
        obs = env.reset()
        for t in range(N_STEPS):
            obs, rew, done, info = env.step(_onehot(4, t))
        assert {"color_id", "shape_id", "size_id"} <= set(info), info
        assert obs["is_last"] in (False, np.False_)
        _report("noisyshape", obs)
    if args.dmc:
        cases = [
            ("vh walker_walk", dict(task="dmc_walker_walk", video_hard=True)),
            ("vhr walker_walk low-contrast", dict(task="dmc_walker_walk", video_hard=True,
                                                  video_random_frames=True, recolor_agent=True)),
            ("vhr cheetah_run_small", dict(task="dmc_cheetah_run_small", video_hard=True,
                                           video_random_frames=True)),
            ("clean walker_walk_small", dict(task="dmc_walker_walk_small")),
            ("vh reacher_easy pixel_video_bg", dict(task="dmc_reacher_easy", pixel_video_bg=True,
                                                    distracting_background_path=davis)),
        ]
        for name, over in cases:
            env = make("dmc_vision", env_num=1, **over)
            obs = env.reset()
            for _ in range(N_STEPS):
                obs, *_ = env.step(env.action_space.sample())
            _report(name, obs)


def check_isodream(src, args):
    _enter(src / "iso_rl")
    import ruamel.yaml as yaml

    import dreamer
    import tools

    def make(name, *overrides):
        configs = yaml.YAML(typ="safe", pure=True).load(pathlib.Path("configs.yaml").read_text())
        defaults = {}
        for n in ["defaults", name]:
            defaults.update(configs[n])
        parser = argparse.ArgumentParser()
        for key, value in sorted(defaults.items(), key=lambda x: x[0]):
            parser.add_argument(f"--{key}", type=tools.args_type(value), default=tools.args_type(value)(value))
        config = parser.parse_args(list(overrides) + ["--seed", "0"])
        config.time_limit //= config.action_repeat
        return dreamer.make_env(config, None, "smoke", None, None)

    if args.noisyshape:
        env = make("noisyshape", "--davis_path", os.environ.get("DAVIS_DIR", ""))
        obs = env.reset()
        for t in range(N_STEPS):
            obs, rew, done, info = env.step({"action": _onehot(4, t)})
        _report("noisyshape", obs)
    if args.dmc:
        for name, flags in [("vh walker_walk", ["--use_video_hard", "True"]),
                            ("vhr cup_catch", ["--task", "dmc_cup_catch", "--use_video_hard", "True",
                                               "--video_random_frames", "True"])]:
            env = make("dmc", "--task", "dmc_walker_walk", *flags)
            obs = env.reset()
            for _ in range(N_STEPS):
                obs, *_ = env.step({"action": env.action_space.sample()})
            _report(name, obs)


def check_denoised_mdp(src, args):
    _enter(src)
    from denoised_mdp.envs.abc import EnvBase
    from denoised_mdp.envs.dmc import make_env

    if args.dmc:
        for spec in ["walker_walk_vh", "ball_in_cup_catch_vhr"]:
            env = make_env(spec, observation_output_kind=EnvBase.ObsOutputKind.image_uint8, seed=0,
                           max_episode_length=1000, action_repeat=2, batch_shape=())
            obs, _ = env.reset()
            for _ in range(N_STEPS):
                obs, rew, done, info = env.step(env.sample_random_action())
            obs = np.asarray(obs).transpose(1, 2, 0)  # CHW -> HWC for the report
            _report(spec, {"image": obs})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("baseline", choices=["r2dreamer", "isodream", "denoised_mdp"])
    p.add_argument("--src", type=pathlib.Path, default=None,
                   help="patched checkout (default: $BASELINES_SRC/<baseline> or baselines/upstream/<baseline>)")
    p.add_argument("--dmc", action="store_true")
    p.add_argument("--noisyshape", action="store_true")
    args = p.parse_args()
    src = args.src or pathlib.Path(os.environ.get("BASELINES_SRC", BASELINES_ROOT / "upstream")) / args.baseline
    src = src.resolve()
    print(f"{args.baseline}: {src}")
    globals()[f"check_{args.baseline}"](src, args)
    print("smoke test passed")


if __name__ == "__main__":
    main()
