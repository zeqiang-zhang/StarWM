"""Where the DMC runs live.

Two layouts are supported by every script in this directory:

``--layout release`` (default; ``--root runs``), the layout the README's training
commands produce::

    runs/dmc/{clean,vh,vhr}/<task>/<variant>_s<seed>
        variant: starwm, starwm_r, dreamerv3, inv_only, iso_star1, iso_star3, N1, N2, N8
    runs/dmc_small/{clean,vh,vhr}/<task>_small/{starwm,dreamerv3}_s<seed>
    runs/dmc_lowcontrast/{vh,vhr}/<task>/{starwm,dreamerv3}_s<seed>
    runs/dmc_reacher/{vh,vhr}/{starwm,starwm_r,dreamerv3}_s<seed>
    runs/sensitivity/{linv0.3,linv3.0,bg_noanneal,bg_final2.0}_s<seed>
    runs/baselines/r2dreamer/{vh,vhr}/<task>_s<seed>                 (main DMC)
    runs/baselines/r2dreamer/{clean,vh,vhr}/<task>_small_s<seed>     (small objects)
    runs/baselines/r2dreamer/lowcontrast_{vh,vhr}/<task>_s<seed>
    runs/baselines/r2dreamer/{vh,vhr}/reacher_easy_s<seed>
    runs/baselines/isodream/{vh,vhr}/<task>_s<seed>
    runs/baselines/denoised_mdp/{vh,vhr}/<task>_s<seed>   (TensorBoard events, possibly nested)

``<task>`` is the release task key (``cup_catch``, ``cartpole_swingup``,
``cheetah_run``, ``finger_spin``, ``walker_run``, ``walker_walk``).

``--layout paper --root <dir>`` reads the original research log directories,
``<dir>`` being the folder that contains the four code bases
(``icr_dreamerv3``, ``r2dreamer``, ``denoised_mdp``,
``MBRL_with_Isolated_Imaginations``); each can be overridden with
``--icr_root/--r2_root/--den_root/--iso_root``. The runs of the stress tests
(low-contrast agent, Reacher, sensitivity) live in a second log directory of the
StarWM and R2-Dreamer code bases, named with ``--stress_subdir``. This is what
the scripts were verified against.

``--seeds 0 1 2`` restricts every cell to the given seed ids.
"""
import glob
import os

from common import (STARWM_R_TASK_SCALE, find_tfevents_dir, paper_task, seed_runs)

DMC_VARIANTS = ["starwm", "starwm_r", "dreamerv3", "inv_only", "iso_star1", "iso_star3",
                "N1", "N2", "N8"]
BASELINES = ["r2dreamer", "isodream", "denoised_mdp"]
# metric holding the training-episode return of each source
TRAIN_METRIC = {
    "starwm": "jsonl:train_return", "dreamerv3": "jsonl:train_return",
    "r2dreamer": "jsonl:episode/score", "isodream": "jsonl:train_return",
    "denoised_mdp": "tb:train_reward",
}


def add_layout_args(parser, default_root="runs"):
    g = parser.add_argument_group("run locations")
    g.add_argument("--layout", choices=["release", "paper"], default="release",
                   help="directory convention of the runs (see layouts.py)")
    g.add_argument("--root", default=None,
                   help=f"release: runs directory (default: {default_root}); "
                        "paper: folder containing the original code bases")
    g.add_argument("--icr_root", default=None, help="paper layout: StarWM/DreamerV3 code base")
    g.add_argument("--r2_root", default=None, help="paper layout: R2-Dreamer code base")
    g.add_argument("--den_root", default=None, help="paper layout: DenoisedMDP code base")
    g.add_argument("--iso_root", default=None, help="paper layout: Iso-Dream++ code base")
    g.add_argument("--stress_subdir", default=None,
                   help="paper layout: log directory (inside the StarWM and R2-Dreamer code "
                        "bases) holding the low-contrast / Reacher / sensitivity runs")
    g.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="only use these seed ids")
    parser.set_defaults(_default_root=default_root)


def layout_from_args(args):
    return Layout(args.layout, args.root or (args._default_root if args.layout == "release" else "."),
                  icr=args.icr_root, r2=args.r2_root, den=args.den_root, iso=args.iso_root,
                  seeds=args.seeds, stress_subdir=args.stress_subdir)


class Layout:
    """Maps (experiment, setting, task, variant) to seed directories.

    Every method returns a list of ``(seed, run_dir)`` sorted by seed.
    """

    def __init__(self, kind="release", root="runs", icr=None, r2=None, den=None, iso=None,
                 seeds=None, stress_subdir=None):
        self.kind = kind
        self.root = root
        self.seeds = set(seeds) if seeds else None
        self.stress_subdir = stress_subdir
        if kind == "paper":
            self.icr = icr or os.path.join(root, "icr_dreamerv3")
            self.r2 = r2 or os.path.join(root, "r2dreamer")
            self.den = den or os.path.join(root, "denoised_mdp")
            self.iso = iso or os.path.join(root, "MBRL_with_Isolated_Imaginations")
        elif kind != "release":
            raise ValueError(kind)

    def _runs(self, prefix, style):
        return seed_runs(prefix, style, self.seeds)

    def _stress(self, root, *parts):
        if not self.stress_subdir:
            raise SystemExit("paper layout: pass --stress_subdir <log directory of the "
                             "low-contrast / Reacher / sensitivity runs>")
        return os.path.join(root, self.stress_subdir, *parts)

    # ── main DMC benchmark and its ablations ────────────────────────────────
    def dmc(self, setting, task, variant):
        """StarWM / DreamerV3 / ablation runs of the main benchmark (``setting`` in
        clean, vh, vhr)."""
        if variant not in DMC_VARIANTS:
            raise ValueError(variant)
        if self.kind == "release":
            return self._runs(os.path.join(self.root, "dmc", setting, task, variant), "s")
        lp = os.path.join(self.icr, "logdir_paper")
        pt = paper_task(task)
        if variant == "starwm":
            p = os.path.join(lp, "main_experiments", setting, pt)
        elif variant == "starwm_r":
            p = os.path.join(lp, "ablation_reward", setting,
                             f"{pt}_N4_R{STARWM_R_TASK_SCALE[task]}")
        elif variant == "dreamerv3":
            p = os.path.join(lp, "dreamer", setting, pt)
        elif variant == "inv_only":
            p = os.path.join(lp, "main_experiments", setting, f"{pt}_noNCE")
        elif variant == "iso_star1":
            p = os.path.join(lp, "ablation_noDetach_invCtr", setting, pt)
        elif variant == "iso_star3":
            p = os.path.join(lp, "ablation_reconRSSM", setting, pt)
        else:  # N1 / N2 / N8
            p = os.path.join(lp, "ablation_queries", setting, f"{pt}_{variant}")
        return self._runs(p, "d")

    def baseline(self, name, setting, task):
        """Baseline runs on the main benchmark. For ``denoised_mdp`` the returned
        directory is the one holding the TensorBoard event files."""
        if name not in BASELINES:
            raise ValueError(name)
        if self.kind == "release":
            runs = self._runs(os.path.join(self.root, "baselines", name, setting, task), "s")
            if name == "denoised_mdp":
                runs = [(s, find_tfevents_dir(d)) for s, d in runs]
                runs = [(s, d) for s, d in runs if d is not None]
            return runs
        pt = paper_task(task)
        if name == "r2dreamer":
            return self._runs(os.path.join(self.r2, "logdir_paper", "r2dreamer", setting, pt), "d")
        if name == "isodream":
            return self._runs(os.path.join(self.iso, "logdir_paper", "iso", setting, pt), "d")
        # DenoisedMDP: <task>_<seed>/dmc_<task>_<setting>/<hparams>_seed=<seed>/events...
        out = []
        for seed, d in self._runs(os.path.join(self.den, "logdir_paper", "denoised_mdp",
                                               setting, pt), "d"):
            parent = os.path.join(d, f"dmc_{pt}_{setting}")
            if not os.path.isdir(parent):
                continue
            for c in sorted(glob.glob(os.path.join(parent, f"*seed={seed}*"))):
                if os.path.isdir(c) and glob.glob(os.path.join(c, "events.out.tfevents.*")):
                    out.append((seed, c))
                    break
        return out

    # ── small-object variant (Appendix, Fig 20) ─────────────────────────────
    def small(self, setting, task, variant):
        """``variant`` in starwm, dreamerv3, r2dreamer; ``task`` without ``_small``."""
        if self.kind == "release":
            if variant == "r2dreamer":
                return self._runs(os.path.join(self.root, "baselines", "r2dreamer", setting,
                                               f"{task}_small"), "s")
            return self._runs(os.path.join(self.root, "dmc_small", setting, f"{task}_small",
                                           variant), "s")
        pt = f"{paper_task(task)}_small"
        if variant == "r2dreamer":
            base = os.path.join(self.r2, "logdir_paper", "r2dreamer")
        else:
            base = os.path.join(self.icr, "logdir_paper",
                                {"starwm": "starwm", "dreamerv3": "dreamer"}[variant])
        return self._runs(os.path.join(base, f"small_{setting}", pt), "d")

    # ── stress tests (EMA-0.3 convention) ───────────────────────────────────
    _PAPER_METHOD = {"starwm": "starwm", "starwm_r": "starwmR", "dreamerv3": "dreamer",
                     "r2dreamer": "r2"}

    def lowcontrast(self, setting, task, variant):
        """Low-contrast agent; ``variant`` in starwm, dreamerv3, r2dreamer."""
        if self.kind == "release":
            if variant == "r2dreamer":
                return self._runs(os.path.join(self.root, "baselines", "r2dreamer",
                                               f"lowcontrast_{setting}", task), "s")
            return self._runs(os.path.join(self.root, "dmc_lowcontrast", setting, task,
                                           variant), "s")
        dom = task.split("_")[0]
        m = self._PAPER_METHOD[variant]
        root = self.r2 if variant == "r2dreamer" else self.icr
        return self._runs(self._stress(root, "lowcontrast", f"{dom}_{setting}_{m}"), "s")

    def reacher(self, setting, variant):
        """Reacher (easy) with DAVIS backgrounds; ``variant`` in starwm, starwm_r,
        dreamerv3, r2dreamer."""
        if self.kind == "release":
            if variant == "r2dreamer":
                return self._runs(os.path.join(self.root, "baselines", "r2dreamer", setting,
                                               "reacher_easy"), "s")
            return self._runs(os.path.join(self.root, "dmc_reacher", setting, variant), "s")
        m = self._PAPER_METHOD[variant]
        root = self.r2 if variant == "r2dreamer" else self.icr
        return self._runs(self._stress(root, "reacher", f"{setting}_{m}"), "s")

    def sensitivity(self, key):
        """One-at-a-time hyper-parameter perturbations (Walker Walk VHR)."""
        if self.kind == "release":
            return self._runs(os.path.join(self.root, "sensitivity", key), "s")
        return self._runs(self._stress(self.icr, "sensitivity", key), "s")


def dirs(runs):
    """``[(seed, dir), ...]`` -> ``[dir, ...]``."""
    return [d for _, d in runs]
