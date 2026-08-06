"""U2 episodic memory test — two-round protocol.

Running with no flags executes two rounds in sequence:

    Round 1: +U1+U2 record   (U2 bank empty → accumulates trajectories)
    Round 2: +U1+U2 verify   (reuses round-1 memory, NEW task params → tests
                              generalization, not memorization)

Each round uses a different task seed so parameter values differ between the
two rounds.  If U2 helps Round 2 despite the different params, it has learned
the underlying procedure rather than memorizing exact goals.

Each round's per-episode results are saved to scripts/results/<run_id>_<phase>.json.

Usage:
    python scripts/test_u1_u2.py [--tasks=T1,T2,...] [--n=1] [--seed=30]
                                 [--u2-store=./u2_store]
    python scripts/test_u1_u2.py --u2           # only one +U1+U2 round (no verify)
    python scripts/test_u1_u2.py --u3           # only one +U1+U2+U3 round (page-graph RAG)
    python scripts/test_u1_u2.py --run_id=20260804_135016   # resume same ckpt dir
    # Keep AutoDL tunnel up; RAG_URL defaults to http://127.0.0.1:18180

"""

from __future__ import annotations

import json
import os
import sys
import time

from absl import flags
from absl import logging

# Match run.py: hide adb_controller / android_env INFO spam; keep warnings+.
logging.set_verbosity(logging.WARNING)
os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')
os.environ.setdefault('GRPC_TRACE', 'none')

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)

FLAGS = flags.FLAGS

_PROGRESS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'progress.json'
)

flags.DEFINE_list(
    'tasks',
    None,
    'Task templates to run. If None, uses all "hard" tasks from task_metadata.json.',
)
flags.DEFINE_integer('n', 1, 'Number of task combinations per template.')
flags.DEFINE_integer('seed', 30, 'Task random seed.')
flags.DEFINE_integer('console_port', 5554, 'Android emulator console port.')
flags.DEFINE_string(
    'adb_path', '', 'Path to adb.exe (auto-discovered if empty).'
)
flags.DEFINE_string(
    'u2_store', os.path.join(_REPO_ROOT, 'u2_store'),
    'Directory where U2 episodic memory is persisted (shared across runs).',
)
flags.DEFINE_bool('u2', False, 'Only run a single +U1+U2 round (no verify round).')
flags.DEFINE_bool(
    'u3',
    False,
    'Enable U3 page-graph RAG with +U1+U2+U3. '
    'Alone: single +U1+U2+U3 round. Default (no --u2/--u3): U2 two-round protocol.',
)
flags.DEFINE_string(
    'rag_url',
    '',
    'U3 RAG URL. Empty = env RAG_URL or http://127.0.0.1:18180.',
)
flags.DEFINE_string(
    'run_id',
    '',
    'Reuse an existing run_id to resume from IncrementalCheckpointer '
    '(skips tasks that already have .pkl.gz under runs/run_<id>_<phase>/). '
    'Empty = new timestamp id.',
)


def _make_env():
  """Build the Android env and task registry.  Called once."""
  from android_world import registry
  from android_world import suite_utils
  from android_world.env import env_launcher

  # Reuse run.py's adb discovery logic.
  adb_path = FLAGS.adb_path
  if not adb_path:
    local_app_data = os.environ.get('LOCALAPPDATA', '')
    for path in (
        os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'),
        os.path.expanduser('~/Android/Sdk/platform-tools/adb'),
        os.path.join(local_app_data, 'Android', 'Sdk', 'platform-tools', 'adb.exe'),
        os.path.expanduser(r'~\AppData\Local\Android\Sdk\platform-tools\adb.exe'),
        r'D:\Data\Android\platform-tools\adb.exe',
    ):
      if path and os.path.isfile(path):
        adb_path = path
        break
  if not adb_path:
    raise RuntimeError('adb not found. Pass --adb_path or install Android SDK.')

  env = env_launcher.load_and_setup_env(
      console_port=FLAGS.console_port, emulator_setup=False, adb_path=adb_path
  )
  task_registry = registry.TaskRegistry()
  return env, task_registry, suite_utils


def _make_suite(task_registry, suite_utils, seed: int):
  """Build a task suite for one round with the given seed.

  Each round uses a different seed so task parameters differ across rounds.
  This forces U2 to generalize to new parameter values instead of memorizing
  the exact same goal (which would inflate reuse via identical embeddings).
  """
  # tasks=None → all hard tasks from task_metadata.json
  tasks = FLAGS.tasks or suite_utils.get_tasks_by_difficulty('hard')
  return suite_utils.create_suite(
      task_registry.get_registry(family='android_world'),
      n_task_combinations=FLAGS.n,
      seed=seed,
      tasks=tasks,
      use_identical_params=False,
  )


def _run_phase(
    env,
    suite,
    suite_utils,
    agent_name: str,
    enable_u1: bool,
    enable_u2: bool,
    enable_u3: bool = False,
    run_id: str = '',
) -> list[dict]:
  """Run one phase with a progress bar.  Returns episode metadata."""
  from android_world import checkpointer as checkpointer_lib
  from android_world.agents import infer
  from android_world.agents import memory_agent

  agent = memory_agent.MemoryAugmentedAgent(
      env,
      infer.Gpt4Wrapper('Qwen/Qwen3-VL-32B-Instruct'),
      enable_u1=enable_u1,
      enable_u2=enable_u2,
      enable_u3=enable_u3,
      u2_persistence_dir=FLAGS.u2_store,
      rag_url=FLAGS.rag_url or None,
  )
  agent.name = agent_name

  # Persist full episode trajectories as .pkl.gz under runs/
  ckpt_root = os.path.join(_REPO_ROOT, 'runs')
  ckpt_name = f'run_{run_id}_{agent_name}' if run_id else f'run_{agent_name}'
  checkpoint_dir = os.path.join(ckpt_root, ckpt_name)
  os.makedirs(checkpoint_dir, exist_ok=True)
  checkpointer = checkpointer_lib.IncrementalCheckpointer(checkpoint_dir)
  print(f'Episode checkpoints -> {checkpoint_dir}')

  total_episodes = sum(len(instances) for _, instances in suite.items())
  import tqdm
  pbar = tqdm.tqdm(total=total_episodes, desc=f'[{agent_name}]', ncols=90)

  # Current task name — the progress callback doesn't tell us which one, so
  # track it across the two loops in main() by pre-computing the flat order.
  task_order: list[str] = [
      name for name, instances in suite.items() for _ in instances
  ]

  _init_phase_progress(agent_name, task_order)

  # ── Step-level tracking ────────────────────────────────────────────
  # Wrap agent.step/reset so the dashboard can show the current step of the
  # current task.  Best-effort: a write failure never interrupts the run.
  _episodes_done = [0]
  _step_state = {'step': 0, 'max': 0}

  _orig_reset = agent.reset
  _orig_step = agent.step

  def _reset_wrap(*a, **kw):
    _step_state['step'] = 0
    _step_state['max'] = 0
    return _orig_reset(*a, **kw)

  def _step_wrap(goal):
    _step_state['step'] += 1
    _step_state['max'] = getattr(agent, '_max_steps', 0) or 0
    result = _orig_step(goal)
    _publish_step_progress(agent_name, task_order, _episodes_done[0], _step_state)
    return result

  agent.reset = _reset_wrap
  agent.step = _step_wrap

  def _progress(episodes_metadata, print_summary=False):
    del print_summary
    done = len(episodes_metadata)
    _episodes_done[0] = done
    pbar.n = done
    pbar.set_postfix_str(f'{_count_success(episodes_metadata)}/{done} ok')
    pbar.refresh()
    current = task_order[done] if done < len(task_order) else ''
    _write_progress(agent_name, done, total_episodes, episodes_metadata, current)

  results = suite_utils.run(
      suite,
      agent,
      checkpointer=checkpointer,
      process_episodes_fn=_progress,
  )
  pbar.close()
  print(f'Wrote episode pkl.gz files to {checkpoint_dir}')
  return results


# In-memory accumulator for the dashboard.  Updated after every episode.
_PROGRESS_DATA: dict = {
    'updated_at': '',
    'run_id': '',
    'active_phase': '',
    'tasks': [],
    'phases': {},
}


# Debounce for step-level writes to progress.json (seconds).  The dashboard
# polls every 15s, so writing more often than ~5s is wasted I/O.
_STEP_DEBOUNCE_S = 5.0
_last_step_write_ts = [0.0]


def _publish_step_progress(
    phase: str,
    task_order: list[str],
    episodes_done: int,
    step_state: dict,
) -> None:
  """Update the current-step info in progress.json (debounced).

  Reads the accumulator, patches only the 'current' section, and rewrites
  the file.  Called after every agent step but throttled to ~5s.
  """
  now = time.monotonic()
  if now - _last_step_write_ts[0] < _STEP_DEBOUNCE_S:
    return
  _last_step_write_ts[0] = now

  try:
    entry = _PROGRESS_DATA.get('phases', {}).get(phase)
    if entry is None:
      return
    # Which task is currently executing: episodes_done points at the next
    # un-completed episode in the flat task_order.
    task_idx = min(episodes_done, len(task_order) - 1) if task_order else -1
    task_name = task_order[task_idx] if task_idx >= 0 else ''
    _PROGRESS_DATA['current'] = {
        'phase': phase,
        'task_idx': task_idx + 1,  # 1-based: "task 5 of 19"
        'task_total': len(task_order),
        'task_name': task_name,
        'step': step_state['step'],
        'step_max': step_state['max'],
    }
    _PROGRESS_DATA['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(_PROGRESS_FILE, 'w', encoding='utf-8') as f:
      json.dump(_PROGRESS_DATA, f, ensure_ascii=False, indent=2)
  except Exception:
    pass  # Best-effort; never crash the run.


def _init_phase_progress(phase: str, task_order: list[str]) -> None:
  """Reset the per-phase progress slot before a phase starts."""
  # Per-task total = how many episodes belong to that task template
  from collections import Counter
  task_counts = Counter(task_order)
  _PROGRESS_DATA['active_phase'] = phase
  _PROGRESS_DATA['phases'][phase] = {
      'done': 0,
      'total': len(task_order),
      'ok': 0,
      'current_task': '',
      'per_task': {
          name: {'done': 0, 'ok': 0, 'total': task_counts[name]}
          for name in task_counts
      },
  }
  if not _PROGRESS_DATA['tasks']:
    # Preserve order, dedup
    seen = set()
    _PROGRESS_DATA['tasks'] = [
        n for n in task_order if not (n in seen or seen.add(n))
    ]


def _write_progress(
    phase: str,
    done: int,
    total: int,
    episodes: list[dict],
    current_task: str,
) -> None:
  """Update the in-memory accumulator and write progress.json.

  `episodes` is the cumulative list of all completed episodes this phase, so
  per-task counts are recomputed from scratch (idempotent), never incremented.
  """
  entry = _PROGRESS_DATA.get('phases', {}).get(phase)
  if entry is None:
    return
  entry['done'] = done
  entry['total'] = total
  entry['ok'] = _count_success(episodes)
  entry['current_task'] = current_task

  # Recompute per-task cells from the full cumulative list each time
  fresh: dict[str, dict] = {
      name: {'done': 0, 'ok': 0, 'total': cell['total']}
      for name, cell in entry['per_task'].items()
  }
  for e in episodes:
    name = e.get('task_template', '')
    cell = fresh.get(name)
    if cell is None:
      continue
    cell['done'] += 1
    if _is_ok(e.get('is_successful')):
      cell['ok'] += 1
  entry['per_task'] = fresh

  _PROGRESS_DATA['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
  try:
    with open(_PROGRESS_FILE, 'w', encoding='utf-8') as f:
      json.dump(_PROGRESS_DATA, f, ensure_ascii=False, indent=2)
  except Exception:
    pass  # Progress file is best-effort; never crash the run.


def _is_ok(v) -> bool:
  """True if is_successful counts as a success.

  AndroidWorld can report partial success (e.g. 0.5 for a multi-part goal that
  was half done).  Only strictly-greater-than-0.5 counts as success here so a
  half-completed task is not tallied as a full win.
  """
  try:
    return isinstance(v, (int, float)) and v == v and v > 0.5
  except Exception:
    return False


def _count_success(results: list[dict]) -> int:
  """Count successful episodes.  AndroidWorld stores is_successful as float."""
  return sum(1 for e in results if _is_ok(e.get('is_successful')))


def _save_results(results: list[dict], phase: str, run_id: str) -> str:
  """Persist phase results to scripts/results/ and return the file path.

  Saves per-episode metadata (task, success, episode_length, run_time) so the
  three runs can be compared later without re-running.
  """
  out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
  os.makedirs(out_dir, exist_ok=True)
  path = os.path.join(out_dir, f'{run_id}_{phase}.json')
  data = {
      'run_id': run_id,
      'phase': phase,
      'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
      'total': len(results),
      'success': _count_success(results),
      'episodes': [
          {
              'task_template': e.get('task_template', ''),
              'goal': e.get('goal', ''),
              'is_successful': e.get('is_successful'),
              'episode_length': e.get('episode_length'),
              'run_time': e.get('run_time'),
          }
          for e in results
      ],
  }
  with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
  print(f'  → saved {phase} results to {path}')
  return path


def _run_phase_and_save(
    env,
    suite,
    suite_utils,
    phase: str,
    enable_u1: bool,
    enable_u2: bool,
    run_id: str,
    enable_u3: bool = False,
) -> list[dict]:
  """Run one phase and persist its results."""
  results = _run_phase(
      env,
      suite,
      suite_utils,
      phase,
      enable_u1,
      enable_u2,
      enable_u3=enable_u3,
      run_id=run_id,
  )
  _save_results(results, phase, run_id)
  return results


def main():
  FLAGS(sys.argv)
  run_id = FLAGS.run_id.strip() or time.strftime('%Y%m%d_%H%M%S')
  _PROGRESS_DATA['run_id'] = run_id
  if FLAGS.run_id.strip():
    print(f'Resuming Run ID: {run_id} (skip completed episode pkls)')
  else:
    print(f'Run ID: {run_id}')
  if FLAGS.u3:
    rag = FLAGS.rag_url or os.environ.get('RAG_URL', 'http://127.0.0.1:18180')
    print(f'U3 enabled, RAG_URL={rag}')

  env, task_registry, suite_utils = _make_env()

  try:
    if FLAGS.u3:
      # Single round: +U1+U2+U3 (page-graph RAG on top of U1/U2)
      suite = _make_suite(task_registry, suite_utils, FLAGS.seed)
      r = _run_phase_and_save(
          env, suite, suite_utils, 'u1u2u3', True, True, run_id, enable_u3=True
      )
      print(f'+U1+U2+U3: {_count_success(r)}/{len(r)} ok')
    elif FLAGS.u2:
      # Single round: +U1+U2 only (no verify round)
      suite = _make_suite(task_registry, suite_utils, FLAGS.seed)
      r = _run_phase_and_save(
          env, suite, suite_utils, 'u1u2', True, True, run_id, enable_u3=False
      )
      print(f'+U1+U2: {_count_success(r)}/{len(r)} ok')
    else:
      # Two-round protocol for U2:
      #   Round 1: +U1+U2 record (empty bank, accumulates trajectories)
      #   Round 2: +U1+U2 verify (reuses round-1 memory, new params → tests
      #             generalization, not memorization of identical goals).
      # Each round uses a DIFFERENT seed so task parameters differ.
      seeds = [FLAGS.seed, FLAGS.seed + 1]
      print(f'Seeds per round: {seeds}')

      print('\n=== ROUND 1/2: +U1+U2 record (empty bank, collects trajectories) ===')
      suite1 = _make_suite(task_registry, suite_utils, seeds[0])
      r1 = _run_phase_and_save(
          env, suite1, suite_utils, 'u1u2_record', True, True, run_id
      )
      print(f'+U1+U2 record: {_count_success(r1)}/{len(r1)} ok')

      print('\n=== ROUND 2/2: +U1+U2 verify (reuses round-1 memory, new params) ===')
      suite2 = _make_suite(task_registry, suite_utils, seeds[1])
      r2 = _run_phase_and_save(
          env, suite2, suite_utils, 'u1u2_verify', True, True, run_id
      )
      print(f'+U1+U2 verify: {_count_success(r2)}/{len(r2)} ok')

      print('\n=== SUMMARY ===')
      print(f'  +U1+U2 record: {_count_success(r1)}/{len(r1)} ok')
      print(f'  +U1+U2 verify: {_count_success(r2)}/{len(r2)} ok')
      print(f'  Results saved to scripts/results/ with run_id {run_id}')
  finally:
    env.close()


if __name__ == '__main__':
  main()
