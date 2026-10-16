"""Training entry point for StarWM, the DreamerV3 baseline and the ablations.

    python train.py --configs dmc_vision --task dmc_walker_walk --logdir runs/x
    python train.py --configs noisyshape --logdir runs/y --model dreamerv3

``--model`` selects the world model (see configs.yaml):
    starwm          StarWM (default)
    dreamerv3       DreamerV3 (PyTorch re-implementation, NM512/dreamerv3-torch)
    dualstream      ablation: residual VAE stream, no attention
    attention_only  ablation: spatial attention, no residual stream
All of them share the RSSM, the actor-critic and the training loop below,
which are those of dreamerv3-torch.
"""
import argparse
import functools
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from ruamel.yaml import YAML

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()


def build_world_model(obs_space, act_space, step, config):
    if config.model == "starwm":
        from starwm import StarWMWorldModel
        return StarWMWorldModel(obs_space, act_space, step, config)
    if config.model == "dreamerv3":
        return models.WorldModel(obs_space, act_space, step, config)
    if config.model == "dualstream":
        from ablations.dualstream import WorldModel as DualStreamWorldModel
        return DualStreamWorldModel(obs_space, act_space, step, config)
    if config.model == "attention_only":
        from ablations.attention_only import AttentionOnlyWorldModel
        return AttentionOnlyWorldModel(obs_space, act_space, step, config)
    raise ValueError(f"Unknown --model {config.model}")


def policy_embed(wm, obs, model):
    """Embedding of one observation that enters the RSSM (acting / probing)."""
    if model == "dreamerv3":
        return wm.encoder(obs)
    if model == "dualstream":
        return wm.proj_dyn(wm.encoder(obs))  # only the dynamics stream enters the RSSM
    # starwm / attention_only: encoder intercepted at 8x8 -> attention -> proj_dyn
    f8_proj, raw_embed, _ = wm._encode_with_f8(obs)
    pooled_for_rssm, _, _ = wm._compute_attention(f8_proj)
    pooled_for_rssm = pooled_for_rssm.reshape(raw_embed.shape[:-1] + (pooled_for_rssm.shape[-1],))
    return wm.proj_dyn(pooled_for_rssm)


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._wm = build_world_model(obs_space, act_space, self._step, config)
        print(f"World model: {config.model} ({type(self._wm).__name__})")
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = policy_embed(self._wm, obs, self._config.model)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        # Reward-free environments (the Shape family) train the world model only.
        if not self._config.skip_actor_critic:
            start = post
            reward = lambda f, s, a: self._wm.heads["reward"](
                self._wm.dynamics.get_feat(s)
            ).mode()
            metrics.update(self._task_behavior._train(start, reward)[-1])
            if self._config.expl_behavior != "greedy":
                mets = self._expl_behavior.train(start, context, data)[-1]
                metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id,
            pixel_video_bg=config.pixel_video_bg,
            video_random_frames=config.video_random_frames,
            davis_path=config.distracting_background_path,
        )
        if config.pixel_video_bg:
            pass  # DAVIS frames composited behind the agent inside DeepMindControl.render() (Reacher)
        elif config.use_video_hard:
            # DMC-GB2 video_hard background (VH) or random frames of it (VHR),
            # for both training and evaluation.
            from envs.video_hard_wrapper import VideoHardWrapper
            domain = task.split("_", 1)[0]
            if domain == "cup":
                domain = "ball_in_cup"
            recolor = config.recolor_rgba if config.recolor_agent else None
            env._env = VideoHardWrapper(env._env, domain, seed=config.seed + id + 2000,
                                        random_frames=config.video_random_frames,
                                        recolor_rgba=recolor)
            # VideoHardWrapper reloads physics from a new XML (adds the
            # video_screen site) -- re-apply the _small tweaks afterwards.
            if getattr(env, "_small", False):
                from envs.small_variant import (
                    apply_small_camera, apply_small_backcube,
                )
                env._env.reset()
                apply_small_camera(env._env.physics, domain)
                apply_small_backcube(env._env.physics, domain)
        env = wrappers.NormalizeActions(env)
    else:
        import shapeenv
        if suite not in shapeenv.ENVS:
            raise NotImplementedError(suite)
        kwargs = dict(size=config.size, seed=config.seed + id)
        if suite in shapeenv.USES_DAVIS:
            kwargs["davis_path"] = config.distracting_background_path
        if suite in ("passiveshape", "passiveshapeseq"):
            kwargs["reward_mode"] = config.passive_reward_mode
        if suite == "texnoisyshape":
            kwargs["level"] = config.tex_level
            kwargs["alpha"] = config.tex_alpha
        env = shapeenv.make(suite, **kwargs)
        env = wrappers.OneHotAction(env)
    # Delayed-action experiment (Appendix F.5): the env applies a_{t-k}.
    if config.action_delay > 0:
        env = wrappers.ActionDelay(env, config.action_delay)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    return env


def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    if not config.distracting_background_path:
        config.distracting_background_path = os.environ.get("DAVIS_DIR", "")
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    with open(logdir / "config.yaml", "w") as f:
        YAML(typ="safe", pure=True).dump(
            {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(config).items()}, f)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode, id: make_env(config, mode, id)
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]
    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space
    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(acts.low).repeat(config.envs, 1),
                    torch.tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt", map_location=config.device)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every:
        logger.write()
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
            )
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
        print("Start training.")
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
        # Optional archived snapshots checkpoints/agent_step{N}.pt every
        # `checkpoint_every` agent steps (used by the delayed-action probes).
        if config.checkpoint_every > 0:
            step_now = int(agent._step)
            ckpt_dir = logdir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            marker = step_now - (step_now % config.checkpoint_every)
            ckpt_path = ckpt_dir / f"agent_step{marker}.pt"
            if not ckpt_path.exists():
                torch.save(items_to_save, ckpt_path)
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


def load_config(config_names=(), argv=()):
    """Build the config Namespace exactly as the command line does."""
    yaml = YAML(typ='safe', pure=True)
    configs = yaml.load((pathlib.Path(__file__).parent / "configs.yaml").read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    defaults = {}
    for name in ["defaults", *config_names]:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    return parser.parse_args(list(argv))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    main(load_config(args.configs or (), remaining))
