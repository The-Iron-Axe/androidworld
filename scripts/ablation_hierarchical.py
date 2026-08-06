"""Hierarchical ablation script for U1-U4 memory (record → verify per config).

Three stages in one run:

  Stage A  Record    — N rounds (default 3) with ALL memory enabled
                       (u1+u2+u3+u4).  Each round reuses the previous round's
                       stores, so U2 (DMS) matures (multi-round ecosystem,
                       paper §4.3), U3 grows its page graph, and U4 accumulates
                       successful trajectories into skills.  The stores are the
                       shared "trained memory" every config later reads.
  Stage C  Ablate    — hierarchical evaluation, one run per config, all reading
                       the SAME matured stores:
                         u12   → +U1+U2
                         u123  → +U1+U2+U3
                         u1234 → +U1+U2+U3+U4
                       Each config's gain over the previous is the marginal
                       contribution of the newly added layer.

(No baseline stage: the reference is u12, the first config, when every later
config is compared against it.)

The evaluation task set is a DIFFERENT task list from the record set by default
(record uses `--record-tasks`, ablate uses `--tasks`).  Splitting train/test
this way measures generalization, not memorization.  Every config sits on the
same matured memory, so differences are attributable purely to which layer is
enabled.

Usage:
    python scripts/ablation_hierarchical.py \
        --record-tasks=MarkorCreateNoteAndSms,MarkorMergeNotes,... \
        --tasks=MarkorCreateNoteAndSms,MarkorMergeNotes,... \
        --record-rounds=3 --seed=30

    # Single config (e.g. resume from a prior run, or only ablate u1234):
    python scripts/ablation_hierarchical.py --configs=u1234

Stores (shared): u2_store/u3_store/u4_store — see --*_store flags.
Requires: AutoDL RAG tunnel up (U3 reads RAG_URL, default http://127.0.0.1:18180).
"""

from __future__ import annotations

import json
import os
import sys
import time

from absl import app
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

flags.DEFINE_list(
    'record_tasks',
    None,
    'Task templates used to mature the memory stores (Stage A). '
    'If None, defaults to all "hard" tasks.',
)
flags.DEFINE_list(
    'tasks',
    None,
    'Task templates used to evaluate each config (Stage B/C). '
    'If None, defaults to all "hard" tasks. Should differ from --record-tasks '
    'to measure generalization.',
)
flags.DEFINE_integer(
    'record_rounds', 3, 'Number of all-memory record rounds in Stage A (DMS warm-up).'
)
flags.DEFINE_integer('seed', 30, 'Base task random seed. Ignored when --seeds is set.')
flags.DEFINE_list(
    'seeds',
    None,
    'Repeat the full Stage A + Stage C for each seed in this list, then print '
    'per-config success mean±std across seeds. Overrides --seed.',
)
flags.DEFINE_integer('n', 1, 'Task combinations per template per round.')
flags.DEFINE_integer('console_port', 5554, 'Android emulator console port.')
flags.DEFINE_string(
    'adb_path', '', 'Path to adb.exe (auto-discovered if empty).'
)
flags.DEFINE_string(
    'u2_store', os.path.join(_REPO_ROOT, 'u2_store'),
    'Shared U2 episodic memory store (accumulated across record rounds).',
)
flags.DEFINE_string(
    'u3_store', os.path.join(_REPO_ROOT, 'u3_store'),
    'Shared U3 page-graph store.',
)
flags.DEFINE_string(
    'u4_store', os.path.join(_REPO_ROOT, 'u4_store'),
    'Shared U4 skill-library store.',
)
flags.DEFINE_bool(
    'rag_on',
    False,
    'Enable U3 RAG. On by default for u123/u1234. Set --norag_on to force '
    'local page-graph only.',
)
flags.DEFINE_list(
    'configs',
    ['u12', 'u123', 'u1234'],
    'Which configs to evaluate in Stage C. Choices: u12, u123, u1234. '
    'Default runs all three.',
)
flags.DEFINE_string(
    'rag_url', '',
    'U3 RAG URL. Empty = env RAG_URL or http://127.0.0.1:18180.',
)
flags.DEFINE_string(
    'run_id', '',
    'Reuse an existing run_id to resume from IncrementalCheckpointer.',
)


# ── Memory construction flags ─────────────────────────────────────────

def _make_env():
  """Build the Android env and task registry.  Called once."""
  from android_world import registry
  from android_world import suite_utils
  from android_world.env import env_launcher

  adb_path = FLAGS.adb_path
  if not adb_path:
    local_app_data = os.environ.get('LOCALAPPDATA', '')
    for path in (
        os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'),
        os.path.expanduser('~/Android/Sdk/platform-tools/adb'),
        os.path.join(local_app_data, 'Android', 'Sdk', 'platform-tools', 'adb.exe'),
        r'D:\Data\Android\platform-tools\adb.exe',
    ):
      if os.path.exists(path):
        adb_path = path
        break
  if not adb_path:
    raise RuntimeError('adb not found. Pass --adb_path or install Android SDK.')

  env = env_launcher.load_and_setup_env(
      console_port=FLAGS.console_port, emulator_setup=False, adb_path=adb_path
  )
  task_registry = registry.TaskRegistry()
  return env, task_registry, suite_utils


def _make_suite(task_registry, suite_utils, seed: int, tasks):
  """Build a task suite for one round with the given seed."""
  t = tasks or suite_utils.get_tasks_by_difficulty('hard')
  return suite_utils.create_suite(
      task_registry.get_registry(family='android_world'),
      n_task_combinations=FLAGS.n,
      seed=seed,
      tasks=t,
      use_identical_params=False,
  )


def _run_phase(
    env,
    suite,
    suite_utils,
    agent_name: str,
    enable_u1: bool,
    enable_u2: bool,
    enable_u3: bool,
    enable_u4: bool,
    run_id: str,
    store_stage: bool,
) -> list[dict]:
  """Run one phase.  Returns episode metadata."""
  from android_world import checkpointer as checkpointer_lib
  from android_world.agents import infer
  from android_world.agents import memory_agent

  agent = memory_agent.MemoryAugmentedAgent(
      env,
      infer.Gpt4Wrapper('Qwen/Qwen3-VL-32B-Instruct'),
      enable_u1=enable_u1,
      enable_u2=enable_u2,
      enable_u3=enable_u3,
      enable_u4=enable_u4,
      u2_persistence_dir=FLAGS.u2_store,
      u3_persistence_dir=FLAGS.u3_store,
      u4_persistence_dir=FLAGS.u4_store,
      rag_url=FLAGS.rag_url or os.environ.get('RAG_URL', 'http://127.0.0.1:18180'),
  )
  agent.name = agent_name

  ckpt_root = os.path.join(_REPO_ROOT, 'runs')
  ckpt_name = f'run_{run_id}_{agent_name}' if run_id else f'run_{agent_name}'
  checkpoint_dir = os.path.join(ckpt_root, ckpt_name)
  os.makedirs(checkpoint_dir, exist_ok=True)
  checkpointer = checkpointer_lib.IncrementalCheckpointer(checkpoint_dir)
  print(f'Episode checkpoints -> {checkpoint_dir}')

  total_episodes = sum(len(instances) for _, instances in suite.items())
  import tqdm
  pbar = tqdm.tqdm(total=total_episodes, desc=f'[{agent_name}]', ncols=90)

  task_order = [name for name, instances in suite.items() for _ in instances]
  _init_phase_progress(agent_name, task_order)

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
  del store_stage  # (reserved: could record a per-phase "store snapshot" here)
  print(f'Wrote episode pkl.gz files to {checkpoint_dir}')
  return results


# ── Progress / results helpers (mirrors test_u1_u2.py) ─────────────────

_PROGRESS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'progress.json')
_PROGRESS_DATA: dict = {
    'updated_at': '',
    'run_id': '',
    'active_phase': '',
    'tasks': [],
    'phases': {},
}
_STEP_DEBOUNCE_S = 5.0
_last_step_write_ts = [0.0]


def _publish_step_progress(phase, task_order, episodes_done, step_state) -> None:
  import time as _t
  now = _t.monotonic()
  if now - _last_step_write_ts[0] < _STEP_DEBOUNCE_S:
    return
  _last_step_write_ts[0] = now
  try:
    entry = _PROGRESS_DATA.get('phases', {}).get(phase)
    if entry is None:
      return
    task_idx = min(episodes_done, len(task_order) - 1) if task_order else -1
    task_name = task_order[task_idx] if task_idx >= 0 else ''
    _PROGRESS_DATA['current'] = {
        'phase': phase, 'task_idx': task_idx + 1,
        'task_total': len(task_order), 'task_name': task_name,
        'step': step_state['step'], 'step_max': step_state['max'],
    }
    _PROGRESS_DATA['updated_at'] = _t.strftime('%Y-%m-%d %H:%M:%S')
    with open(_PROGRESS_FILE, 'w', encoding='utf-8') as f:
      json.dump(_PROGRESS_DATA, f, ensure_ascii=False, indent=2)
  except Exception:
    pass


def _init_phase_progress(phase, task_order) -> None:
  from collections import Counter
  task_counts = Counter(task_order)
  _PROGRESS_DATA['active_phase'] = phase
  _PROGRESS_DATA['phases'][phase] = {
      'done': 0, 'total': len(task_order), 'ok': 0, 'current_task': '',
      'per_task': {name: {'done': 0, 'ok': 0, 'total': task_counts[name]}
                   for name in task_counts},
  }
  if not _PROGRESS_DATA['tasks']:
    seen = set()
    _PROGRESS_DATA['tasks'] = [n for n in task_order if not (n in seen or seen.add(n))]


def _write_progress(phase, done, total, episodes, current_task) -> None:
  entry = _PROGRESS_DATA.get('phases', {}).get(phase)
  if entry is None:
    return
  entry['done'] = done
  entry['total'] = total
  entry['ok'] = _count_success(episodes)
  entry['current_task'] = current_task
  fresh = {name: {'done': 0, 'ok': 0, 'total': cell['total']}
           for name, cell in entry['per_task'].items()}
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
    pass


def _is_ok(v) -> bool:
  try:
    return isinstance(v, (int, float)) and v == v and v > 0.5
  except Exception:
    return False


def _count_success(results) -> int:
  return sum(1 for e in results if _is_ok(e.get('is_successful')))


def _save_results(results, phase, run_id) -> str:
  out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
  os.makedirs(out_dir, exist_ok=True)
  path = os.path.join(out_dir, f'{run_id}_{phase}.json')
  data = {
      'run_id': run_id, 'phase': phase,
      'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
      'total': len(results), 'success': _count_success(results),
      'episodes': [
          {'task_template': e.get('task_template', ''), 'goal': e.get('goal', ''),
           'is_successful': e.get('is_successful'),
           'episode_length': e.get('episode_length'),
           'run_time': e.get('run_time')}
          for e in results
      ],
  }
  with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
  print(f'  → saved {phase} results to {path}')
  return path


def _run_phase_and_save(env, suite, suite_utils, phase, u1, u2, u3, u4, run_id, store_stage):
  results = _run_phase(
      env, suite, suite_utils, phase, u1, u2, u3, u4, run_id, store_stage
  )
  _save_results(results, phase, run_id)
  print(f'{phase}: {_count_success(results)}/{len(results)} ok')
  return results


# ── Config table ─────────────────────────────────────────────────────

CONFIGS = {
    'u12':    dict(u1=True,  u2=True,  u3=False, u4=False),
    'u123':   dict(u1=True,  u2=True,  u3=True,  u4=False),
    'u1234':  dict(u1=True,  u2=True,  u3=True,  u4=True),
}


def main(argv):
  del argv
  FLAGS(sys.argv)
  run_id = FLAGS.run_id.strip() or time.strftime('%Y%m%d_%H%M%S')
  _PROGRESS_DATA['run_id'] = run_id
  print(f'Run ID: {run_id}')
  print(f'U3 RAG: {"on" if FLAGS.rag_on else "off (local graph only)"}')

  # Sanity: record tasks must be provided separately from eval tasks for a
  # clean generalization split.  Warn (not fail) if they're identical.
  record_tasks = FLAGS.record_tasks
  eval_tasks = FLAGS.tasks
  if record_tasks and eval_tasks and set(record_tasks) == set(eval_tasks):
    print('WARNING: --record_tasks == --tasks. This measures memorization, '
          'not generalization. Consider disjoint task lists.')

  # Multi-seed repetition: each seed runs a fresh Stage A + Stage C with its
  # OWN store directories, so every seed is an independent train/test
  # repetition (stores are not polluted across seeds).
  seeds = [int(x) for x in FLAGS.seeds] if FLAGS.seeds else [FLAGS.seed]
  print(f'Seeds: {seeds}  ({len(seeds)} independent repetitions)')

  env, task_registry, suite_utils = _make_env()

  def build_suite(seed, tasks):
    return _make_suite(task_registry, suite_utils, seed, tasks)

  # Per-config success rates across all seeds, for the final mean±std summary.
  # key: config name -> list of (success, total) per seed.
  across_seeds: dict[str, list[tuple[int, int]]] = {}

  for seed in seeds:
    seed_suffix = f'seed{seed}'
    sep = '=' * 70
    print(f'\n{sep}\nREPETITION: seed={seed}\n{sep}')

    # Per-seed stores (independent train/test repetition).
    stores = {
        'u2': os.path.join(FLAGS.u2_store, seed_suffix),
        'u3': os.path.join(FLAGS.u3_store, seed_suffix),
        'u4': os.path.join(FLAGS.u4_store, seed_suffix),
    }
    prev_store_u2 = FLAGS.u2_store
    prev_store_u3 = FLAGS.u3_store
    prev_store_u4 = FLAGS.u4_store
    FLAGS.u2_store = stores['u2']
    FLAGS.u3_store = stores['u3']
    FLAGS.u4_store = stores['u4']

    # ── Stage A: record (mature this seed's stores, all memory on) ────
    print(f'\n=== STAGE A: record, all memory on, {FLAGS.record_rounds} round(s) ===')
    for r in range(1, FLAGS.record_rounds + 1):
      s = seed + r - 1
      print(f'\n--- Record round {r}/{FLAGS.record_rounds} (seed={s}) ---')
      suite = build_suite(s, record_tasks)
      _run_phase_and_save(
          env, suite, suite_utils, f'stageA_r{r}_seed{seed}',
          True, True, FLAGS.rag_on, True, run_id, store_stage=True,
      )

    # ── Stage C: hierarchical ablation (this seed's matured stores) ───
    print(f'\n=== STAGE C: hierarchical ablation (seed={seed}) ===')
    for cfg in FLAGS.configs:
      spec = CONFIGS[cfg]
      print(f'\n--- Ablate {cfg}: +U1+U2' + ('+U3' if spec['u3'] else '')
            + ('+U4' if spec['u4'] else '') + ' ---')
      suite = build_suite(seed, eval_tasks)
      results = _run_phase_and_save(
          env, suite, suite_utils, f'stageC_{cfg}_seed{seed}',
          spec['u1'], spec['u2'], spec['u3'] and FLAGS.rag_on, spec['u4'],
          run_id, store_stage=False,
      )
      across_seeds.setdefault(cfg, []).append(
          (_count_success(results), len(results))
      )

    # Restore store flags for the next seed.
    FLAGS.u2_store = prev_store_u2
    FLAGS.u3_store = prev_store_u3
    FLAGS.u4_store = prev_store_u4

  # ── Summary: per-config mean ± std across seeds ────────────────────
  print('\n=== SUMMARY (mean±std across seeds) ===')
  if not across_seeds:
    print('(no Stage C configs were evaluated)')
    return
  for cfg in FLAGS.configs:
    if cfg not in across_seeds:
      continue
    rates = [ok / total for ok, total in across_seeds[cfg] if total]
    if not rates:
      continue
    mean = sum(rates) / len(rates)
    std = (sum((r - mean) ** 2 for r in rates) / len(rates)) ** 0.5
    total_ok = sum(ok for ok, _ in across_seeds[cfg])
    total_n = sum(t for _, t in across_seeds[cfg])
    print(f'  {cfg:8s}: {mean:.3f} ± {std:.3f}  ({total_ok}/{total_n} episodes)')
  print(f'\nRun ID {run_id}: per-phase results in scripts/results/')


if __name__ == '__main__':
  app.run(main)
