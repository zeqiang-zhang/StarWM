"""
VideoHardWrapper: Wraps a dm_control environment with video_hard background.

Uses the ColorVideoWrapper from DMC-GB2 (``video_hard`` level: the ground is
removed and one of 100 RealEstate10K-based videos is projected onto a backcube
behind the agent). Supports two modes:
  - VH  (sequential): frames advance one per physics step (standard video_hard).
  - VHR (random):     each step shows a random frame of the current episode's video.
The video of the first episode is drawn from RandomState(seed); later episodes
cycle through the pool in order.

This wrapper should be applied at the dm_control level (before pixel wrapper),
i.e., env._env = VideoHardWrapper(env._env, domain, seed, ...)
"""

import os
import sys
import numpy as np
import dm_env

from envs.dmcgb.dmcgb_photometric import ColorVideoWrapper


class VideoHardWrapper(dm_env.Environment):
    """
    Wraps a dm_control env with video_hard backgrounds.
    
    Args:
        env: The underlying dm_control environment
        domain: Domain name ('walker', 'cheetah', etc.)
        seed: Random seed
        random_frames: If True, each step picks a random frame instead of sequential
    """
    
    def __init__(self, env, domain, seed, random_frames=False, recolor_rgba=None):
        # Set domain name attribute required by ColorVideoWrapper
        env._domain_name = domain

        # Create the ColorVideoWrapper with video_hard mode
        self._wrapper = ColorVideoWrapper(env, mode='video_hard', seed=seed)
        self._random_frames = random_frames
        self._random_state = np.random.RandomState(seed + 42)
        # Low-contrast agent (Appendix F.3): desaturate the agent geoms so the
        # foreground shares the colour distribution of the video background.
        self._recolor_rgba = list(recolor_rgba) if recolor_rgba is not None else None
        self._bg_keywords = ('screen', 'video', 'sky', 'ground', 'floor', 'wall')

        print(f"[VideoHardWrapper] domain={domain}, random_frames={random_frames}, "
              f"recolor={self._recolor_rgba}")

    def _apply_recolor(self):
        """Set agent (non-background) geom colours to the desaturated target."""
        if self._recolor_rgba is None:
            return
        physics = self._wrapper._env.physics
        names = physics.named.model.geom_rgba.axes.row.names
        for nm in names:
            if not nm:  # skip unnamed structural geoms
                continue
            if any(k in nm.lower() for k in self._bg_keywords):
                continue
            physics.named.model.geom_rgba[nm] = self._recolor_rgba

    def reset(self):
        time_step = self._wrapper.reset()
        if self._random_frames:
            # Randomize first frame
            self._randomize_frame()
        self._apply_recolor()
        return time_step
    
    def step(self, action):
        if self._random_frames:
            # Before step, pick a random frame
            time_step = self._wrapper._env.step(action)
            # Apply random frame manually
            self._randomize_frame()
            self._wrapper._move_backcube()
            return time_step
        else:
            # Standard sequential behavior
            return self._wrapper.step(action)
    
    def _randomize_frame(self):
        """Pick a random frame from the current video and apply it."""
        if self._wrapper._background is not None and self._wrapper._current_video_len > 1:
            random_idx = self._random_state.randint(0, self._wrapper._current_video_len)
            self._wrapper._current_video_frame = random_idx
            self._wrapper._apply_video()
    
    def observation_spec(self):
        return self._wrapper.observation_spec()
    
    def action_spec(self):
        return self._wrapper.action_spec()
    
    def __getattr__(self, name):
        return getattr(self._wrapper, name)
