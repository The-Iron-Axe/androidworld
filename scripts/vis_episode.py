"""Render an AndroidWorld run directory into a per-step HTML report.

Reads every <task>_<n>.pkl.gz under a run dir and produces a single HTML
file with each step's summary, action, reason, and before/after screenshots
(SOM-marked, falling back to raw). Open the HTML in a browser to inspect a
trajectory step by step.

Usage:
    python scripts/vis_episode.py --run_dir=runs/run_20260806T212930284206
    python scripts/vis_episode.py --run_dir=... --out=my_report.html

The screenshot arrays are written as base64 data URIs directly into the HTML,
so the report is self-contained and can be opened / shared as one file.
"""

from __future__ import annotations

import base64
import gzip
import io
import os
import sys
import pickle
from pathlib import Path

from absl import app
from absl import flags
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FLAGS = flags.FLAGS
flags.DEFINE_string('run_dir', None, 'Path to a runs/<run_id> directory.')
flags.DEFINE_string('out', '', 'Output HTML path. Defaults to run_dir/report.html.')
flags.DEFINE_integer('max_steps', 200, 'Cap the number of steps rendered per episode.')
flags.DEFINE_integer('max_w', 420, 'Max width for displayed screenshots (keeps HTML small).')


def _load_episodes(run_dir: Path) -> list[tuple[str, dict]]:
  episodes = []
  for path in sorted(run_dir.glob('*.pkl.gz')):
    try:
      with gzip.open(path, 'rb') as f:
        data = pickle.load(f)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      print(f'  ! skip {path.name}: {exc}')
      continue
    if isinstance(data, list):
      for ep in data:
        if isinstance(ep, dict):
          episodes.append((path.name, ep))
    elif isinstance(data, dict):
      episodes.append((path.name, data))
  return episodes


def _img_to_data_uri(arr: np.ndarray | None, max_w: int) -> str:
  """Encode a uint8 HxWx3 image as a data URI, downscaled to max_w."""
  if arr is None or not isinstance(arr, np.ndarray) or arr.size == 0:
    return ''
  h, w = arr.shape[:2]
  scale = min(1.0, max_w / w)
  if scale < 1.0:
    arr = np.asarray(Image.fromarray(arr).resize(
        (int(w * scale), int(h * scale)), Image.LANCZOS))
  buf = io.BytesIO()
  Image.fromarray(arr).convert('RGB').save(buf, format='JPEG', quality=85)
  return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def _action_str(action) -> str:
  if action is None:
    return '(none)'
  s = str(action)
  return s if s else '(none)'


def _esc(text: str) -> str:
  return (str(text)
          .replace('&', '&amp;')
          .replace('<', '&lt;')
          .replace('>', '&gt;'))


def _render_episode(ep: dict, stem: str, max_steps: int, max_w: int) -> str:
  ed = ep.get('episode_data', {})
  n = len(ed.get('step_number') or [])
  n = min(n, max_steps)

  meta = [
      ('Task', ep.get('task_template', '')),
      ('Instance', stem),
      ('Goal', ep.get('goal', '')),
      ('Success', ep.get('is_successful')),
      ('Episode length', ep.get('episode_length')),
      ('Run time (s)', round(ep.get('run_time', 0), 1)),
      ('Agent', ep.get('agent_name', '')),
      ('Seed', ep.get('seed', '')),
  ]

  rows = []
  for i in range(n):
    action = _action_str(ed.get('action_output_json', [None] * n)[i]
                         if ed.get('action_output_json') else None)
    reason = _esc(ed.get('action_reason', [None] * n)[i]
                  if ed.get('action_reason') else '')
    summary = _esc(ed.get('summary', [None] * n)[i]
                   if ed.get('summary') else '')

    before_som = ed.get('before_screenshot_with_som', [None] * n)[i]
    after_som = ed.get('after_screenshot_with_som', [None] * n)[i]
    raw = ed.get('raw_screenshot', [None] * n)[i]
    if before_som is None:
      before_som = raw
    if after_som is None:
      after_som = ed.get('before_screenshot_with_som', [None] * n)[i + 1] \
          if i + 1 < n else None
      if after_som is None:
        after_som = raw

    before_uri = _img_to_data_uri(before_som, max_w)
    after_uri = _img_to_data_uri(after_som, max_w)

    rows.append(f'''
    <div class="step">
      <div class="step-head"><span class="step-n">Step {i}</span><code>{_esc(action)}</code></div>
      <div class="imgs">
        <figure><figcaption>before</figcaption>{f'<img src="{before_uri}">' if before_uri else '<span class="none">(no image)</span>'}</figure>
        <figure><figcaption>after</figcaption>{f'<img src="{after_uri}">' if after_uri else '<span class="none">(no image)</span>'}</figure>
      </div>
      <div class="text"><div class="lbl">Reason</div><p>{reason or '(none)'}</p></div>
      <div class="text"><div class="lbl">Summary</div><p>{summary or '(none)'}</p></div>
    </div>''')

  meta_html = ''.join(
      f'<div class="meta"><span class="lbl">{k}</span><span class="val">{_esc(v)}</span></div>'
      for k, v in meta)

  return f'''
  <div class="episode">
    <h2>{_esc(stem)}</h2>
    <div class="meta-grid">{meta_html}</div>
    {''.join(rows)}
  </div>'''


def main(argv):
  del argv
  if not FLAGS.run_dir:
    raise SystemExit('--run_dir is required')
  run_dir = Path(FLAGS.run_dir)
  if not run_dir.is_dir():
    raise SystemExit(f'not a directory: {run_dir}')

  episodes = _load_episodes(run_dir)
  if not episodes:
    raise SystemExit(f'no .pkl.gz episodes found under {run_dir}')

  print(f'Loaded {len(episodes)} episode(s) from {run_dir}')
  for stem, ep in episodes:
    print(f'  - {stem}: success={ep.get("is_successful")} '
          f'steps={len(ep.get("episode_data", {}).get("step_number") or [])}')

  body = ''.join(
      _render_episode(ep, stem, FLAGS.max_steps, FLAGS.max_w)
      for stem, ep in episodes)

  out_path = Path(FLAGS.out) if FLAGS.out else run_dir / 'report.html'
  html = f'''<!DOCTYPE html>
<html lang="zh"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Episode report — {run_dir.name}</title>
<style>
  body {{ font-family: "Segoe UI", "Microsoft YaHei", sans-serif; background: #1e1e2e; color: #cdd6f4; padding: 24px; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .sub {{ color: #7f849c; font-size: 13px; margin-bottom: 24px; }}
  .episode {{ background: #2a2a3c; border: 1px solid #3b3b52; border-radius: 12px; padding: 20px; margin-bottom: 24px; }}
  h2 {{ font-size: 17px; margin: 0 0 12px; color: #89b4fa; }}
  .meta-grid {{ display: grid; grid-template-columns: auto 1fr; gap: 4px 14px; font-size: 13px; margin-bottom: 16px; }}
  .meta .lbl {{ color: #7f849c; }}
  .step {{ border-top: 1px solid #3b3b52; padding: 14px 0; }}
  .step-head {{ display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }}
  .step-n {{ background: #89b4fa; color: #1e1e2e; border-radius: 6px; padding: 2px 8px; font-weight: 600; font-size: 13px; }}
  .step-head code {{ color: #a6e3a1; font-size: 13px; }}
  .imgs {{ display: flex; gap: 14px; margin-bottom: 10px; }}
  figure {{ margin: 0; }}
  figcaption {{ font-size: 11px; color: #7f849c; margin-bottom: 4px; }}
  img {{ border-radius: 8px; border: 1px solid #3b3b52; max-height: 420px; }}
  .none {{ color: #f38ba8; font-size: 12px; }}
  .text {{ margin: 6px 0; }}
  .lbl {{ color: #fab387; font-size: 11px; font-weight: 600; }}
  .text p {{ margin: 2px 0 0; font-size: 13px; white-space: pre-wrap; }}
</style></head>
<body>
  <h1>Episode report</h1>
  <div class="sub">Source: {run_dir} · {len(episodes)} episode(s)</div>
  {body}
</body></html>'''

  out_path.write_text(html, encoding='utf-8')
  print(f'Report written to {out_path.resolve()}')


if __name__ == '__main__':
  app.run(main)
