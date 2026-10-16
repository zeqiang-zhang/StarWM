"""Registry of every training run behind a figure or table of the paper.

Each experiment is a ``train.py`` invocation (presets + flags) that is run for
seeds 0, 1, 2. Logs go to ``<root>/<name>_s<seed>`` (default root: ``runs``).

    python scripts/experiments.py list [PATTERN]          # names (glob pattern)
    python scripts/experiments.py show NAME [--seed 0]    # print the command
    python scripts/experiments.py run NAME [NAME ...] [--seeds 0 1 2]
                                       [--parallel] [--dry-run] [--device cuda:0]

``--parallel`` starts the seeds of one experiment as concurrent processes,
which is how the paper's runs were executed (3 seeds per GPU). NoisyShape,
Reacher and the NoisyShape variants need DAVIS 2017 (``$DAVIS_DIR``); the DMC
video distractors need the DMC-GB2 videos (``$DMCGB_DATA_DIR``).
"""
import argparse
import fnmatch
import os
import shlex
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2)

DMC_TASKS = ["walker_walk", "walker_run", "cheetah_run", "cartpole_swingup",
             "finger_spin", "cup_catch"]  # cup_catch = Ball-in-cup Catch
DMC_SETTINGS = OrderedDict([
    ("clean", {}),
    ("vh", {"use_video_hard": True}),
    ("vhr", {"use_video_hard": True, "video_random_frames": True}),
])
DMC_STEPS = {"clean": 1e6, "vh": 3e6, "vhr": 1e6}
# StarWM-R: weight of the reward-prediction loss on the attention features
# (Appendix E.1). Identical for VH and VHR.
STARWM_R_SCALE = {"walker_walk": 30, "walker_run": 30, "cheetah_run": 50,
                  "cartpole_swingup": 50, "finger_spin": 30, "cup_catch": 100,
                  "reacher_easy": 30}

EXPERIMENTS = OrderedDict()


def add(name, presets, flags=None, paper=""):
    assert name not in EXPERIMENTS, name
    EXPERIMENTS[name] = dict(presets=list(presets), flags=dict(flags or {}), paper=paper)


# ----------------------------------------------------------------------------- DMC
for setting, sflags in DMC_SETTINGS.items():
    for task in DMC_TASKS:
        base = {"task": f"dmc_{task}", "steps": DMC_STEPS[setting], "video_pred_log": False, **sflags}
        ref = "Fig. 5, Table 5, Figs. 1/10" if setting != "clean" else "Table 5 (Clean)"
        add(f"dmc/{setting}/{task}/starwm", ["dmc_vision"], base, ref)
        add(f"dmc/{setting}/{task}/dreamerv3", ["dmc_vision", "dreamerv3"], base, ref)
        if setting != "clean":
            add(f"dmc/{setting}/{task}/starwm_r", ["dmc_vision"],
                {**base, "attn_reward_scale": STARWM_R_SCALE[task]}, ref + ", App. E.1")

# Contrastive ablation (inverse dynamics only), Fig. 16
for setting, sflags in DMC_SETTINGS.items():
    for task in ["cartpole_swingup", "cheetah_run", "walker_run"]:
        add(f"dmc/{setting}/{task}/inv_only", ["dmc_vision", "inv_only"],
            {"task": f"dmc_{task}", "steps": DMC_STEPS[setting], "video_pred_log": False, **sflags},
            "Fig. 16")

# Gradient isolation on DMC (star1: no detach A/B with SS; star3: only detach B, no SS)
for setting in ["clean", "vhr"]:
    for task in ["walker_walk", "cheetah_run"]:
        for iso in ["iso_star1", "iso_star3"]:
            add(f"dmc/{setting}/{task}/{iso}", ["dmc_vision", iso],
                {"task": f"dmc_{task}", "steps": 1e6, "video_pred_log": False, **DMC_SETTINGS[setting]},
                "Figs. 7/13/14, Tables 7/8")

# Number of queries N (N=4 is dmc/vh/<task>/starwm), Fig. 17 / Table 10
for task in ["walker_walk", "cheetah_run", "cartpole_swingup"]:
    for n in [1, 2, 8]:
        add(f"dmc/vh/{task}/N{n}", ["dmc_vision"],
            {"task": f"dmc_{task}", "steps": 3e6, "video_pred_log": False,
             "crossattn_num_queries": n, **DMC_SETTINGS["vh"]}, "Fig. 17, Table 10")

# Small objects (agent rendered at half linear size), Fig. 20
for setting, sflags in DMC_SETTINGS.items():
    for task in ["cheetah_run", "walker_walk"]:
        base = {"task": f"dmc_{task}_small", "steps": DMC_STEPS[setting], "video_pred_log": False, **sflags}
        add(f"dmc_small/{setting}/{task}_small/starwm", ["dmc_vision"], base, "Fig. 20")
        add(f"dmc_small/{setting}/{task}_small/dreamerv3", ["dmc_vision", "dreamerv3"], base, "Fig. 20")

# Low-contrast (desaturated) Walker, Table 14 / Fig. 25
for setting in ["vh", "vhr"]:
    base = {"task": "dmc_walker_walk", "steps": DMC_STEPS[setting], "recolor_agent": True,
            "video_pred_log": False, **DMC_SETTINGS[setting]}
    add(f"dmc_lowcontrast/{setting}/walker_walk/starwm", ["dmc_vision"], base, "Table 14, Fig. 25")
    add(f"dmc_lowcontrast/{setting}/walker_walk/dreamerv3", ["dmc_vision", "dreamerv3"], base, "Table 14, Fig. 25")

# Reacher with DAVIS frames pasted behind the arm (top-down camera), Table 17 / Fig. 29
for setting in ["vh", "vhr"]:
    base = {"task": "dmc_reacher_easy", "steps": DMC_STEPS[setting], "pixel_video_bg": True,
            "video_pred_log": False}
    if setting == "vhr":
        base["video_random_frames"] = True
    add(f"dmc_reacher/{setting}/starwm", ["dmc_vision"], base, "Table 17, Fig. 29")
    add(f"dmc_reacher/{setting}/starwm_r", ["dmc_vision"],
        {**base, "attn_reward_scale": STARWM_R_SCALE["reacher_easy"]}, "Table 17, Fig. 29")
    add(f"dmc_reacher/{setting}/dreamerv3", ["dmc_vision", "dreamerv3"], base, "Table 17, Fig. 29")

# One-at-a-time hyperparameter perturbations, Walker Walk VHR 1M, Table 9
SENS = OrderedDict([("linv0.3", {"attn_inv_scale": 0.3}), ("linv3.0", {"attn_inv_scale": 3.0}),
                    ("bg_noanneal", {"Annealing": False}), ("bg_final2.0", {"res_kl_scale": 2.0})])
for key, f in SENS.items():
    add(f"sensitivity/{key}", ["dmc_vision"],
        {"task": "dmc_walker_walk", "steps": 1e6, "video_pred_log": False,
         **DMC_SETTINGS["vhr"], **f}, "Table 9")

# ----------------------------------------------------------------------- NoisyShape
NS_MODELS = OrderedDict([
    ("starwm", []), ("dreamerv3", ["dreamerv3"]), ("dualstream", ["dualstream"]),
    ("ss_only", ["ss_only"]), ("dualstream_ss", ["dualstream_ss"]),
    ("attention_only", ["attention_only"]),
])
for m, p in NS_MODELS.items():
    add(f"noisyshape/{m}", ["noisyshape", *p], {}, "Fig. 3, Table 4")
for iso in ["iso_star1", "iso_star2", "iso_star3", "iso_star4"]:
    add(f"noisyshape/{iso}", ["noisyshape", iso], {}, "Fig. 12")
for ss in ["inv_only", "ctr_only"]:
    add(f"noisyshape/{ss}", ["noisyshape", ss], {}, "Fig. 15")
for env in ["shape", "noisyshapeseq"]:
    for m in ["starwm", "dreamerv3"]:
        add(f"{env}/{m}", [env, *NS_MODELS[m]], {}, "Table 12, Fig. 19")
add("shapefixed/dreamerv3", ["shapefixed", "dreamerv3"], {}, "Table 12, Fig. 19")
for m in ["starwm", "dreamerv3"]:
    add(f"noisyshapemotion/{m}", ["noisyshapemotion", *NS_MODELS[m]], {}, "Table 11, Fig. 18")
    add(f"noisytwoshape/{m}", ["noisytwoshape", *NS_MODELS[m]], {}, "Figs. 21/22")
    for level in [1, 2, 3]:
        add(f"texnoisyshape/L{level}/{m}", ["texnoisyshape", *NS_MODELS[m]],
            {"tex_level": level}, "Table 13, Fig. 24")
add("passiveshape/starwm", ["passiveshape"], {}, "Table 16, Fig. 28")
add("passiveshapeseq/starwm", ["passiveshapeseq"], {}, "Table 16, Fig. 28")
# Delayed action effects (probed at checkpoints/agent_step180000.pt), Table 19
add("delay/k0_adjacent", ["noisyshape"], {"checkpoint_every": 20000}, "Table 19 (reference)")
for k in [1, 2, 4]:
    add(f"delay/k{k}_adjacent", ["noisyshape"],
        {"action_delay": k, "inv_delay": 0, "checkpoint_every": 20000}, "Table 19")
    add(f"delay/k{k}_shifted", ["noisyshape"],
        {"action_delay": k, "inv_delay": k, "checkpoint_every": 20000}, "Table 19")


# --------------------------------------------------------------------------- CLI
def _fmt(v):
    if isinstance(v, float) and v.is_integer() and abs(v) >= 1000:
        return f"{v:.0e}".replace("e+0", "e")   # 1000000.0 -> 1e6
    return str(v)


def command(name, seed, root="runs", device=None, python=sys.executable, display=False):
    """train.py command line; ``display=True`` gives a short, copy-pasteable
    form to be run from the repository root."""
    exp = EXPERIMENTS[name]
    entry = "train.py" if display else str(ROOT / "train.py")
    cmd = ["python" if display else python, entry, "--configs", *exp["presets"]]
    for k, v in exp["flags"].items():
        cmd += [f"--{k}", _fmt(v)]
    logroot = Path(root) if display else Path(root).resolve()
    cmd += ["--seed", str(seed), "--logdir", str(logroot / f"{name}_s{seed}")]
    if device:
        cmd += ["--device", device]
    return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list"); p.add_argument("pattern", nargs="?", default="*")
    p = sub.add_parser("show"); p.add_argument("name"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", default="runs")
    p = sub.add_parser("run"); p.add_argument("names", nargs="+")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--root", default="runs"); p.add_argument("--device", default=None)
    p.add_argument("--parallel", action="store_true"); p.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.cmd == "list":
        for n, e in EXPERIMENTS.items():
            if fnmatch.fnmatch(n, args.pattern):
                print(f"{n:45s} {e['paper']}")
        return
    if args.cmd == "show":
        print(shlex.join(command(args.name, args.seed, args.root, display=True)))
        return
    names = [n for pat in args.names for n in EXPERIMENTS if fnmatch.fnmatch(n, pat)]
    if not names:
        sys.exit(f"no experiment matches {args.names}")
    for n in names:
        cmds = [command(n, s, args.root, args.device) for s in args.seeds]
        for c in cmds:
            print(shlex.join(c))
        if args.dry_run:
            continue
        procs = []
        for c in cmds:
            log = Path(c[c.index("--logdir") + 1])
            log.mkdir(parents=True, exist_ok=True)
            out = open(log / "stdout.log", "a")
            proc = subprocess.Popen(c, stdout=out, stderr=subprocess.STDOUT, cwd=ROOT)
            if args.parallel:
                procs.append(proc)
            else:
                proc.wait()
        for proc in procs:
            proc.wait()


if __name__ == "__main__":
    main()
