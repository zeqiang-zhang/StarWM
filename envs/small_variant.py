"""
Helper for the _small variant: widen the side camera's fov so the agent
appears ~50% smaller in the rendered frame, without changing any physics.

We use fov (field of view) rather than moving the camera:
  - The side camera uses `trackcom` mode, which keeps the agent framed
    regardless of camera pos, so translating pos has a muted effect.
  - Doubling the linear fov extent (tan(fov/2) *= 2) makes the agent
    occupy exactly half the picture width/height, independent of domain.

Usage (applied right after suite.load, and again after any reload_from_xml
that resets model params, e.g. VideoHardWrapper):
    physics = env.physics
    apply_small_camera(physics, domain)

For VHR, the backcube must be pushed back and enlarged proportionally to
keep filling the wider view. Call apply_small_backcube(physics, domain)
after VideoHardWrapper.reset().
"""

import math
import numpy as np

# Linear fov ratio: tan(new_half_fov) = FOV_RATIO * tan(orig_half_fov).
# 2.0 => agent appears half as large linearly.
FOV_RATIO = 2.0


def _widen_fov(orig_fov_deg: float, ratio: float) -> float:
    half = math.radians(orig_fov_deg) / 2.0
    new_half = math.atan(math.tan(half) * ratio)
    return math.degrees(new_half) * 2.0


def apply_small_camera(physics, domain):
    """Widen the side camera fov so the agent appears 1/FOV_RATIO as large."""
    if domain not in ("cheetah", "walker"):
        return
    orig_fov = float(physics.named.model.cam_fovy["side"])
    physics.named.model.cam_fovy["side"] = _widen_fov(orig_fov, FOV_RATIO)


def apply_small_backcube(physics, domain):
    """Enlarge the backcube so it still fills the wider view.

    The backcube size is linear in how much of the view it covers; since
    fov tan is scaled by FOV_RATIO, the backcube must grow by the same
    factor to remain edge-to-edge. Position doesn't need to change.
    """
    if domain not in ("cheetah", "walker"):
        return
    try:
        site_size = physics.named.model.site_size["video_screen"]
    except KeyError:
        return
    site_size[0] *= FOV_RATIO  # width
    site_size[2] *= FOV_RATIO  # height
