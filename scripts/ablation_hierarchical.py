"""Hierarchical ablation script for U1-U4 memory (accumulate → verify per config).

Two phases in one run:

  积累轮 (Accumulate)  — N rounds (default 3) with ALL memory enabled
                       (u1+u2+u3+u4).  Each round reuses the previous round's
                       stores, so U2 (DMS) matures (multi-round ecosystem,
                       paper §4.3), U3 grows its page graph, and U4 accumulates
                       successful trajectories into skills.  The stores are the
                       shared "trained memory" every config later reads.
  验证轮 (Verify)      — hierarchical evaluation, one run per config, all reading
                       the SAME matured stores:
                         u12   → +U1+U2
                         u123  → +U1+U2+U3
                         u1234 → +U1+U2+U3+U4
                       Each config's gain over the previous is the marginal
                       contribution of the newly added layer.

By default the SAME task set is used for accumulation and verification
(--record-tasks == --tasks).  This is intentional for a memory system: if the
verification tasks were never seen in the accumulation rounds, U2/U3/U4 would
have no entries to retrieve and the memory layers would be measured as useless
(systematic underestimation).  Running the same tasks first trains the store on
them, so every config is evaluated against memory that actually has relevant
prior experience.  The first accumulate round also doubles as the baseline of
"cold-start, no relevant memory".

Usage:
    python scripts/ablation_hierarchical.py \
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
    'Task templates used to mature the memory stores (积累轮). '
    'If None, defaults to all "hard" tasks.',
)
flags.DEFINE_list(
    'tasks',
    None,
    'Task templates used to evaluate each config (验证轮). '
    'If None, defaults to all "hard" tasks. Defaults to the same list as '
    '--record-tasks so evaluation tasks have prior experience in the store.',
)
flags.DEFINE_integer(
    'record_rounds', 3, 'Number of all-memory accumulation rounds (积累轮, DMS warm-up).'
)
flags.DEFINE_integer('seed', 30, 'Base task random seed. Ignored when --seeds is set.')
flags.DEFINE_list(
    'seeds',
    None,
    'Repeat the full 积累轮 + 验证轮 for each seed in this list, then print '
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
    'Ignored (deprecated). U3 is AutoDL-only; no local page-graph directory.',
)
flags.DEFINE_string(
    'u4_store', os.path.join(_REPO_ROOT, 'u4_store'),
    'Shared U4 skill-library store.',
)
flags.DEFINE_list(
    'configs',
    ['u12', 'u123', 'u1234'],
    'Which configs to evaluate in 验证轮. Choices: u1, u12, u123, u1234. '
    'U3 is on whenever any config contains it (u123 / u1234) — the '
    'accumulate phase enables U3 too so its page graph matures. '
    'No separate --rag_on switch.',
)
flags.DEFINE_string(
    'rag_url', '',
    'U3 RAG URL. Empty = env RAG_URL or http://127.0.0.1:18180.',
)
flags.DEFINE_string(
    'run_id', '',
    'Reuse an existing run_id to resume from IncrementalCheckpointer.',
)
flags.DEFINE_string(
    'results_dir', '',
    'Directory for per-phase result JSON + run.log. Empty = '
    '<repo_root>/scripts/results (back-compat default). Pass e.g. '
    'runs/u123 to store results under a config-named runs/ folder.',
)
flags.DEFINE_bool(
    'multiagent',
    False,
    'Enable the multi-agent orchestration layer (Planner/Executor/Reflector). '
    'Orthogonal to the U1-U4 flags; when off the agent behaves exactly like the '
    'existing memory-augmented agent.',
)
flags.DEFINE_bool(
    'ma_no_planner', False,
    'Disable the Planner module (decomposition, plan-block injection, replan).')
flags.DEFINE_bool(
    'ma_no_av', False,
    'Disable the Action Verifier module (per-step action check, U3 gating,'
    ' U4 step credit).')
flags.DEFINE_bool(
    'ma_no_pa', False,
    'Disable the Progress Auditor module (subgoal advancement, stall-replan,'
    ' progress lines in the plan block).')
flags.DEFINE_bool(
    'ma_no_ec', False,
    'Disable the Evidence Certifier module (completion veto, subgoal cert,'
    ' success fusion).')


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
    enable_multiagent: bool = False,
    results_dir: str = '',
) -> list[dict]:
  """Run one phase.  Returns episode metadata.

  Checkpoints (episode pkl.gz) go under `results_dir/<phase>/` when
  --results_dir is set, so a single run folder holds checkpoints + result
  JSON + run.log together.  Without it, checkpoints go to the legacy
  runs/run_<run_id>_<phase> location.
  """
  from android_world import checkpointer as checkpointer_lib
  from android_world.agents import infer
  from android_world.agents import memory_agent
  from android_world.agents import multi_agent

  agent_cls = (multi_agent.MultiAgentReflectorAgent if enable_multiagent
               else memory_agent.MemoryAugmentedAgent)
  rag_url = ''
  if enable_u3:
    rag_url = (
        FLAGS.rag_url or os.environ.get('RAG_URL', 'http://127.0.0.1:18180')
    ).strip()
    if not rag_url:
      raise ValueError(
          'U3 is enabled but rag_url is empty. Set --rag_url or RAG_URL '
          '(AutoDL tunnel). Local page-graph fallback is disabled.'
      )
  agent_kwargs = dict(
      env=env,
      llm=infer.Gpt4Wrapper('Qwen/Qwen3-VL-32B-Instruct'),
      enable_u1=enable_u1,
      enable_u2=enable_u2,
      enable_u3=enable_u3,
      enable_u4=enable_u4,
      u2_persistence_dir=FLAGS.u2_store,
      u3_persistence_dir=FLAGS.u3_store,
      u4_persistence_dir=FLAGS.u4_store,
      rag_url=rag_url,
  )
  if enable_multiagent:
    agent_kwargs['enable_multiagent'] = True
    agent_kwargs['enable_ma_planner'] = not FLAGS.ma_no_planner
    agent_kwargs['enable_ma_av'] = not FLAGS.ma_no_av
    agent_kwargs['enable_ma_pa'] = not FLAGS.ma_no_pa
    agent_kwargs['enable_ma_ec'] = not FLAGS.ma_no_ec
  agent = agent_cls(**agent_kwargs)
  agent.name = agent_name

  if results_dir:
    checkpoint_dir = os.path.join(results_dir, agent_name)
  else:
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


def _save_results(results, phase, run_id, results_dir='') -> str:
  out_dir = results_dir or os.path.join(
      os.path.dirname(os.path.abspath(__file__)), 'results')
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
           'run_time': e.get('run_time'),
           'token_usage': e.get('token_usage'),
           'memory_stats': e.get('memory_stats')}
          for e in results
      ],
  }
  with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
  print(f'  → saved {phase} results to {path}')
  return path


def _run_phase_and_save(env, suite, suite_utils, phase, u1, u2, u3, u4, run_id, store_stage, enable_multiagent=False, results_dir=''):
  results = _run_phase(
      env, suite, suite_utils, phase, u1, u2, u3, u4, run_id, store_stage,
      enable_multiagent=enable_multiagent,
      results_dir=results_dir,
  )
  _save_results(results, phase, run_id, results_dir)
  print(f'{phase}: {_count_success(results)}/{len(results)} ok')
  return results


# ── Config table ─────────────────────────────────────────────────────

CONFIGS = {
    'u12':    dict(u1=True,  u2=True,  u3=False, u4=False),
    'u123':   dict(u1=True,  u2=True,  u3=True,  u4=False),
    'u1234':  dict(u1=True,  u2=True,  u3=True,  u4=True),
}


def _install_run_logger(log_dir: str) -> None:
  """Tee stdout/stderr into <log_dir>/run.log (same helper as run.py).

  Wraps sys.stdout/stderr with a Tee that writes every line both to the
  terminal and to run.log, so full runtime output is persisted even if the
  terminal scrollback is lost.
  """
  os.makedirs(log_dir, exist_ok=True)
  log_path = os.path.join(log_dir, 'run.log')

  class _Tee:
    def __init__(self, *streams):
      self.streams = streams

    def write(self, data):
      for s in self.streams:
        try:
          s.write(data)
        except ValueError:
          pass
      return len(data)

    def flush(self):
      for s in self.streams:
        try:
          s.flush()
        except ValueError:
          pass

  try:
    log_file = open(log_path, 'a', encoding='utf-8')
  except OSError:
    return
  sys.stdout = _Tee(sys.__stdout__, log_file)
  sys.stderr = _Tee(sys.__stderr__, log_file)


def main(argv):
  del argv
  FLAGS(sys.argv)
  run_id = FLAGS.run_id.strip() or time.strftime('%Y%m%d_%H%M%S')
  _PROGRESS_DATA['run_id'] = run_id
  results_dir = FLAGS.results_dir.strip() or ''
  if results_dir:
    # A config-named results dir (e.g. runs/u123) gets a per-run timestamp
    # subfolder so multiple ablation runs of the same config never collide.
    # Format matches checkpointer.create_run_directory so the folder looks
    # like run_20260731T135847788342.
    import datetime
    ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')
    results_dir = os.path.join(_REPO_ROOT, results_dir, f'run_{ts}')
  _install_run_logger(results_dir or _REPO_ROOT)
  print(f'Run ID: {run_id}')
  if results_dir:
    print(f'Results + run.log -> {results_dir}')
  else:
    print('Results -> scripts/results/  (default; set --results_dir to override)')
  u3_any = any(CONFIGS[cfg]['u3'] for cfg in FLAGS.configs)
  print(f'U3 RAG: {"on (configs request u123/u1234)" if u3_any else "off (no config uses U3)"}')

  # By design the accumulation and verification task sets are the same by
  # default (eval tasks need prior experience in the store).  Warn only if a
  # user explicitly requests DIFFERENT task lists, since that measures
  # zero-shot generalization which memory layers usually can't satisfy.
  record_tasks = FLAGS.record_tasks
  eval_tasks = FLAGS.tasks
  if record_tasks and eval_tasks and set(record_tasks) == set(eval_tasks):
    print('Note: --record_tasks == --tasks (same tasks accumulate and evaluate). '
          'The first accumulate round provides the cold-start baseline; later '
          'configs are evaluated against memory with prior experience on these tasks.')
  elif record_tasks and eval_tasks:
    print('NOTE: --record_tasks differs from --tasks. Evaluation tasks were never '
          'seen in accumulation, so U2/U3/U4 will have no entries to retrieve — '
          'this measures zero-shot generalization, and memory layers may appear '
          'useless. Prefer the same task list unless generalization is the goal.')

  # Multi-seed repetition: each seed runs a fresh 积累轮 + 验证轮 with its
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

    # ── 积累轮: record (mature this seed's stores, all memory on) ────
    print(f'\n=== 积累轮: all memory on, {FLAGS.record_rounds} round(s) ===')
    for r in range(1, FLAGS.record_rounds + 1):
      s = seed + r - 1
      print(f'\n--- 积累 round {r}/{FLAGS.record_rounds} (seed={s}) ---')
      suite = build_suite(s, record_tasks)
      _run_phase_and_save(
          env, suite, suite_utils, f'acc_r{r}_seed{seed}',
          True, True, u3_any, True, run_id, store_stage=True,
          enable_multiagent=FLAGS.multiagent,
          results_dir=results_dir,
      )

    # ── 验证轮: hierarchical ablation (this seed's matured stores) ───
    # 验证轮使用一个与积累轮不重叠的 seed（seed + record_rounds），保证验证
    # 任务实例在积累轮中从未出现过 —— 这样测的是跨 seed 迁移（记忆把 "这类
    # 任务怎么做" 的经验迁移到新实例），而不是回放积累轮见过的答案。
    verify_seed = seed + FLAGS.record_rounds
    print(f'\n=== 验证轮: hierarchical ablation (acc seed={seed}, verify seed={verify_seed}) ===')
    for cfg in FLAGS.configs:
      spec = CONFIGS[cfg]
      print(f'\n--- Ablate {cfg}: +U1+U2' + ('+U3' if spec['u3'] else '')
            + ('+U4' if spec['u4'] else '') + ' ---')
      suite = build_suite(verify_seed, eval_tasks)
      results = _run_phase_and_save(
          env, suite, suite_utils, f'verify_{cfg}_seed{verify_seed}',
          spec['u1'], spec['u2'], spec['u3'], spec['u4'],
          run_id, store_stage=False,
          enable_multiagent=FLAGS.multiagent,
          results_dir=results_dir,
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
    print('(no verify configs were evaluated)')
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
