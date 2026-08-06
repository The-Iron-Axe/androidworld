"""Re-run NaN verify tasks and merge into the verify results JSON.

Usage:
  python scripts/rerun_verify_nans.py --run_id=20260804_135016 --seed=30
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)

_SPEC = importlib.util.spec_from_file_location(
    'test_u1_u2',
    os.path.join(_REPO_ROOT, 'scripts', 'test_u1_u2.py'),
)
t = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(t)

FLAGS = t.FLAGS

_NAN_TASKS = [
    'ExpenseAddMultipleFromMarkor',
    'MarkorTranscribeVideo',
]


def _is_nan(v) -> bool:
  return v is None or (isinstance(v, float) and math.isnan(v))


def _episode_row(e: dict) -> dict:
  return {
      'task_template': e.get('task_template', ''),
      'goal': e.get('goal', ''),
      'is_successful': e.get('is_successful'),
      'episode_length': e.get('episode_length'),
      'run_time': e.get('run_time'),
  }


def _merge_results(
    old_path: str,
    new_results: list[dict],
    run_id: str,
    phase: str,
) -> str:
  with open(old_path, encoding='utf-8') as f:
    data = json.load(f)

  replace = {e.get('task_template', '') for e in new_results}
  kept: list[dict] = []
  seen: set[str] = set()
  for e in data.get('episodes', []):
    name = e.get('task_template', '')
    if name in replace:
      continue
    if name in seen:
      continue
    if _is_nan(e.get('is_successful')):
      continue
    seen.add(name)
    kept.append(e)

  for e in new_results:
    kept.append(_episode_row(e))

  order = []
  for e in data.get('episodes', []):
    n = e.get('task_template', '')
    if n and n not in order:
      order.append(n)
  for n in replace:
    if n not in order:
      order.append(n)
  rank = {n: i for i, n in enumerate(order)}
  kept.sort(key=lambda e: rank.get(e.get('task_template', ''), 10**9))

  out = {
      'run_id': run_id,
      'phase': phase,
      'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
      'total': len(kept),
      'success': t._count_success(kept),
      'episodes': kept,
  }
  with open(old_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
  print(f'Merged into {old_path}: {out["success"]}/{out["total"]} ok')
  return old_path


def main():
  FLAGS(sys.argv)
  run_id = FLAGS.run_id.strip() or '20260804_135016'
  verify_seed = FLAGS.seed + 1
  phase = 'u1u2_verify'
  tasks = list(_NAN_TASKS)

  ckpt_dir = os.path.join(_REPO_ROOT, 'runs', f'run_{run_id}_{phase}')
  for name in tasks:
    pkl = os.path.join(ckpt_dir, f'{name}_0.pkl.gz')
    if os.path.isfile(pkl):
      os.remove(pkl)
      print(f'Deleted checkpoint {pkl}')
    else:
      print(f'No checkpoint to delete: {pkl}')

  FLAGS.tasks = tasks

  results_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      'results',
      f'{run_id}_{phase}.json',
  )
  if not os.path.isfile(results_path):
    raise FileNotFoundError(results_path)

  env, task_registry, suite_utils = t._make_env()
  try:
    print(f'Re-running {tasks} as {phase} with seed={verify_seed}')
    suite = t._make_suite(task_registry, suite_utils, verify_seed)
    new_results = t._run_phase(
        env,
        suite,
        suite_utils,
        phase,
        enable_u1=True,
        enable_u2=True,
        enable_u3=False,
        run_id=run_id,
    )
    print(
        f'Re-run done: {t._count_success(new_results)}/{len(new_results)} ok'
    )
    _merge_results(results_path, new_results, run_id, phase)
  finally:
    env.close()


if __name__ == '__main__':
  main()
