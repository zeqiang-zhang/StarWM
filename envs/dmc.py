import glob
import os

import gymnasium as gym
import numpy as np
from PIL import Image


def _load_davis_videos(davis_path):
    """Return a list of videos, each a sorted list of frame paths."""
    videos = []
    if davis_path and os.path.isdir(davis_path):
        for vdir in sorted(glob.glob(os.path.join(davis_path, "*"))):
            if os.path.isdir(vdir):
                frames = sorted(glob.glob(os.path.join(vdir, "*.jpg")))
                if frames:
                    videos.append(frames)
    return videos


class DeepMindControl(gym.Env):
    metadata = {}

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, seed=0,
                 pixel_video_bg=False, video_random_frames=False, davis_path=None):
        # _small suffix: widen camera fov so the agent appears half-sized.
        # Physics are unchanged — reward/action/dynamics match the base task.
        self._small = name.endswith("_small")
        if self._small:
            name = name[: -len("_small")]
        domain, task = name.split("_", 1)
        if domain == "cup":  # Only domain with multiple words.
            domain = "ball_in_cup"
        self._domain = domain
        if isinstance(domain, str):
            from dm_control import suite

            self._env = suite.load(
                domain,
                task,
                task_kwargs={"random": seed},
            )
        else:
            assert task is None
            self._env = domain()
        if self._small:
            from envs.small_variant import apply_small_camera
            apply_small_camera(self._env.physics, domain)
        self._action_repeat = action_repeat
        self._size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera
        self.reward_range = [-np.inf, np.inf]

        # Pixel-level DAVIS video background (camera-agnostic; used for top-down
        # domains like reacher where the video_hard skybox is not in view).
        # Composites the DAVIS frame behind the agent using a segmentation mask.
        self._pixel_video_bg = pixel_video_bg
        if pixel_video_bg:
            self._pvb_rng = np.random.RandomState(seed + 3000)
            self._pvb_random = video_random_frames
            self._pvb_videos = _load_davis_videos(davis_path)
            self._pvb_bg_keywords = ("ground", "floor", "wall")
            self._pvb_bg_geoms = None  # cached geom ids treated as background
            if not self._pvb_videos:
                print(f"WARNING: pixel_video_bg on but no DAVIS frames at {davis_path}")
            print(f"[PixelVideoBg] {domain}_{task}: {len(self._pvb_videos)} videos, "
                  f"random_frames={video_random_frames}")
            self._pvb_reset_video()

    # ------- pixel video background helpers -------
    def _pvb_reset_video(self):
        if not self._pvb_videos:
            return
        self._pvb_vid = self._pvb_videos[self._pvb_rng.randint(len(self._pvb_videos))]
        self._pvb_idx = self._pvb_rng.randint(len(self._pvb_vid))

    def _pvb_frame(self):
        if not self._pvb_videos:
            return None
        if self._pvb_random:
            vid = self._pvb_videos[self._pvb_rng.randint(len(self._pvb_videos))]
            p = vid[self._pvb_rng.randint(len(vid))]
        else:
            p = self._pvb_vid[self._pvb_idx % len(self._pvb_vid)]
            self._pvb_idx += 1
        img = Image.open(p).convert("RGB").resize(self._size, Image.BILINEAR)
        return np.asarray(img, np.uint8)

    def _pvb_background_mask(self, physics):
        """Boolean (H,W): pixels that are empty space or a named background geom."""
        seg = physics.render(*self._size, camera_id=self._camera, segmentation=True)
        geom = seg[..., 0]                       # geom id per pixel, -1 = void
        mask = geom < 0
        if self._pvb_bg_geoms is None:           # resolve background geom ids once
            bg = set()
            for gid in np.unique(geom):
                if gid < 0:
                    continue
                nm = physics.model.id2name(int(gid), "geom") or ""
                if any(k in nm.lower() for k in self._pvb_bg_keywords):
                    bg.add(int(gid))
            self._pvb_bg_geoms = bg
        for gid in self._pvb_bg_geoms:
            mask |= (geom == gid)
        return mask

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            if len(value.shape) == 0:
                shape = (1,)
            else:
                shape = value.shape
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
        spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            if time_step.last():
                break
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["image"] = self.render()
        # There is no terminal state in DMC
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        done = time_step.last()
        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        time_step = self._env.reset()
        if self._pixel_video_bg:
            self._pvb_reset_video()
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["image"] = self.render()
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        rgb = self._env.physics.render(*self._size, camera_id=self._camera)
        if not self._pixel_video_bg:
            return rgb
        frame = self._pvb_frame()
        if frame is None:
            return rgb
        mask = self._pvb_background_mask(self._env.physics)
        out = rgb.copy()
        out[mask] = frame[mask]
        return out
