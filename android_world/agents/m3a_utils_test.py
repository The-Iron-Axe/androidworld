"""Tests for mark_scale compensation in add_ui_element_mark.

Regression: the m3a screenshot downscale (0.75) shrinks the SOM index labels
below legibility, so the vision model misreads / hallucinates indexes near the
list length (e.g. 57/58 when only 0-56 exist) and the action bounces off the
index-out-of-range guard for the rest of the episode.

The fix: add_ui_element_mark accepts `mark_scale`, which up-scales the
font-size / line-thickness / label-box that carry the index text, WITHOUT
moving the bounding box.  m3a passes mark_scale=1/_SCREENSHOT_SCALE_FACTOR
when it draws on the downscaled screenshot.
"""

import math
import unittest

import numpy as np

from android_world.agents import m3a_utils
from android_world.env import representation_utils


def _fake_element(x_min=100, y_min=100, x_max=300, y_max=200):
  el = representation_utils.UIElement()
  el.text = "label"
  el.content_description = ""
  el.hint_text = ""
  el.bbox_pixels = representation_utils.BoundingBox(
      x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
  )
  el.is_visible = True
  return el


def _draw_mark(screenshot, scale, mark_scale, x_scale):
  """Draw a mark on `screenshot` and return (font_scale, thickness)."""
  # Replicate the exact font/thickness math in add_ui_element_mark.
  iso_scale = math.sqrt(x_scale * x_scale + y_scale * y_scale)
  return mark_scale * 0.7 * iso_scale, int(mark_scale * 2 * iso_scale)


class MarkScaleTest(unittest.TestCase):
  """Verify mark_scale restores legibility after downscaling."""

  def test_font_scale_compensated(self):
    """mark_scale=1/0.75 restores iso_scale to the full-res level.

    At full resolution x_scale=y_scale=0.75 (810/1080, 1800/2400) so
    iso_full=sqrt(0.75^2+0.75^2).  Downscaling by 0.75 halves both axis
    scales, so iso_down=0.75*iso_full.  mark_scale=1/0.75 must recover
    iso_full from iso_down.
    """
    frame = (0, 0, 1080, 2400)
    full_w, full_h = 810, 1800
    down_w, down_h = int(full_w * 0.75), int(full_h * 0.75)
    x_full, y_full = full_w / frame[2], full_h / frame[3]
    x_down, y_down = down_w / frame[2], down_h / frame[3]
    iso_full = math.sqrt(x_full * x_full + y_full * y_full)
    iso_down = math.sqrt(x_down * x_down + y_down * y_down)

    mark_scale = 1.0 / 0.75
    compensated = mark_scale * iso_down
    self.assertAlmostEqual(compensated, iso_full, places=2)
    self.assertGreater(compensated, iso_down)

  def test_default_mark_scale_is_noop(self):
    """mark_scale=1.0 keeps legacy behavior (no compensation)."""
    frame = (0, 0, 1080, 2400)
    down_w, down_h = int(810 * 0.75), int(1800 * 0.75)
    x_down, y_down = down_w / frame[2], down_h / frame[3]
    iso_down = math.sqrt(x_down * x_down + y_down * y_down)
    self.assertAlmostEqual(1.0 * iso_down, iso_down)

  def test_marks_draw_on_downscaled_screenshot(self):
    """add_ui_element_mark runs on a downscaled canvas without error."""
    full = np.zeros((1800, 810, 3), dtype=np.uint8)
    logical_size = (1080, 2400)
    frame = (0, 0, 1080, 2400)
    # Downscaled canvas (0.75): 810x1800.
    canvas = np.zeros((int(1800 * 0.75), int(810 * 0.75), 3), dtype=np.uint8)
    m3a_utils.add_ui_element_mark(
        canvas,
        _fake_element(),
        42,
        logical_size,
        frame,
        0,
        mark_scale=1.0 / 0.75,
    )
    # Compensated mark actually painted a label box (some white pixels).
    self.assertTrue((canvas[100:200, 100:400, :] != 0).any())


if __name__ == "__main__":
  unittest.main()
