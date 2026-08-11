# Copyright 2026 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run eval suite.

The run.py module is used to run a suite of tasks, with configurable task
combinations, environment setups, and agent configurations. You can run specific
tasks or all tasks in the suite and customize various settings using the
command-line flags.
"""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
from absl import logging
from android_world import checkpointer as checkpointer_lib
from android_world import registry
from android_world import suite_utils
from android_world.agents import atmem_agent
from android_world.agents import base_agent
from android_world.agents import dms_agent
from android_world.agents import human_agent
from android_world.agents import infer
from android_world.agents import m3a
from android_world.agents import memory_agent
from android_world.agents import multi_agent
from android_world.agents import random_agent
from android_world.agents import seeact
from android_world.agents import t3a
from android_world.env import env_launcher
from android_world.env import interface

logging.set_verbosity(logging.WARNING)

os.environ['GRPC_VERBOSITY'] = 'ERROR'  # Only show errors
os.environ['GRPC_TRACE'] = 'none'  # Disable tracing


def _find_adb_directory() -> str:
  """Returns the path to the adb binary."""
  local_app_data = os.environ.get('LOCALAPPDATA', '')
  potential_paths = [
      os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'),
      os.path.expanduser('~/Android/Sdk/platform-tools/adb'),
      # Windows Android Studio default locations.
      os.path.join(local_app_data, 'Android', 'Sdk', 'platform-tools', 'adb.exe'),
      os.path.expanduser(r'~\AppData\Local\Android\Sdk\platform-tools\adb.exe'),
      r'D:\Data\Android\platform-tools\adb.exe',
  ]
  for path in potential_paths:
    if path and os.path.isfile(path):
      return path
  raise EnvironmentError(
      'adb not found in the common Android SDK paths. Please install Android'
      " SDK and ensure adb is in one of the expected directories. If it's"
      ' already installed, point to the installed location with --adb_path.'
  )


_ADB_PATH = flags.DEFINE_string(
    'adb_path',
    None,
    'Path to adb. Set if not installed through SDK.',
)
_EMULATOR_SETUP = flags.DEFINE_boolean(
    'perform_emulator_setup',
    False,
    'Whether to perform emulator setup. This must be done once and only once'
    ' before running Android World. After an emulator is setup, this flag'
    ' should always be False.',
)
_DEVICE_CONSOLE_PORT = flags.DEFINE_integer(
    'console_port',
    5554,
    'The console port of the running Android device. This can usually be'
    ' retrieved by looking at the output of `adb devices`. In general, the'
    ' first connected device is port 5554, the second is 5556, and'
    ' so on.',
)

_SUITE_FAMILY = flags.DEFINE_enum(
    'suite_family',
    registry.TaskRegistry.ANDROID_WORLD_FAMILY,
    [
        # Families from the paper.
        registry.TaskRegistry.ANDROID_WORLD_FAMILY,
        registry.TaskRegistry.MINIWOB_FAMILY_SUBSET,
        # Other families for more testing.
        registry.TaskRegistry.MINIWOB_FAMILY,
        registry.TaskRegistry.ANDROID_FAMILY,
        registry.TaskRegistry.INFORMATION_RETRIEVAL_FAMILY,
    ],
    'Suite family to run. See registry.py for more information.',
)
_TASK_RANDOM_SEED = flags.DEFINE_integer(
    'task_random_seed', 30, 'Random seed for task randomness.'
)

_TASKS = flags.DEFINE_list(
    'tasks',
    None,
    'List of specific tasks to run in the given suite family. If None, run all'
    ' tasks in the suite family.',
)
_DIFFICULTY = flags.DEFINE_enum(
    'difficulty',
    None,
    ['easy', 'medium', 'hard'],
    'If set, only run tasks with this difficulty from task_metadata.json.'
    ' Can be combined with --tasks (intersection).',
)
_N_TASK_COMBINATIONS = flags.DEFINE_integer(
    'n_task_combinations',
    1,
    'Number of task instances to run for each task template.',
)

_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir',
    '',
    'The directory to save checkpoints and resume evaluation from. If the'
    ' directory contains existing checkpoint files, evaluation will resume from'
    ' the latest checkpoint. If the directory is empty or does not exist, a new'
    ' directory will be created.',
)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs'),
    'The path to save results to if not resuming from a checkpoint is not'
    ' provided.',
)

# Agent specific.
_AGENT_NAME = flags.DEFINE_string('agent_name', 'm3a_gpt4v', help='Agent name.')
_U1 = flags.DEFINE_boolean(
    'u1',
    False,
    'Enable U1 task-state memory (episode-level structured state tracking).',
)
_U2 = flags.DEFINE_boolean(
    'u2',
    False,
    'Enable U2 episodic memory (cross-task DMS trajectory reuse).',
)
_U2_PERSISTENCE_DIR = flags.DEFINE_string(
    'u2_persistence_dir',
    '',
    'Directory for U2 episodic memory persistence. If empty, memory is not persisted.',
)
_U3_PERSISTENCE_DIR = flags.DEFINE_string(
    'u3_persistence_dir',
    '',
    'Ignored (deprecated). U3 is AutoDL-only; there is no local page-graph store.',
)
_U3 = flags.DEFINE_boolean(
    'u3',
    False,
    'Enable U3 environment knowledge (AutoDL page-graph RAG; requires '
    '--rag_url or RAG_URL, default http://127.0.0.1:18180).',
)
_U4 = flags.DEFINE_boolean(
    'u4',
    False,
    'Enable U4 procedural skill memory (mine reusable skills from successful trajectories).',
)
_U4_PERSISTENCE_DIR = flags.DEFINE_string(
    'u4_persistence_dir',
    '',
    'Directory for U4 skill-library persistence. If empty, skills are not persisted.',
)
_RAG_URL = flags.DEFINE_string(
    'rag_url',
    '',
    'U3 AutoDL RAG URL. Empty = env RAG_URL, else http://127.0.0.1:18180 when --u3.',
)

_MULTIAGENT = flags.DEFINE_boolean(
    'multiagent',
    False,
    'Enable the multi-agent orchestration layer (Planner/Executor/Reflector).'
    ' Orthogonal to the U1-U4 memory flags; when off the agent behaves exactly'
    ' like the existing memory-augmented agent.',
)

_SCREENSHOT_SCALE = flags.DEFINE_float(
    'screenshot_scale',
    1.0,
    'Downscale factor for screenshots fed to the LLM (e.g. 0.5 = 540x1200).',
)

_FIXED_TASK_SEED = flags.DEFINE_boolean(
    'fixed_task_seed',
    False,
    'Whether to use the same task seed when running multiple task combinations'
    ' (n_task_combinations > 1).',
)


# MiniWoB is very lightweight and new screens/View Hierarchy load quickly.
_MINIWOB_TRANSITION_PAUSE = 0.2

# Additional guidelines for the MiniWob tasks.
_MINIWOB_ADDITIONAL_GUIDELINES = [
    (
        'This task is running in a mock app, you must stay in this app and'
        ' DO NOT use the `navigate_home` action.'
    ),
]


def _install_run_logger(checkpoint_dir: str) -> None:
  """Tee stdout/stderr into a per-run log file inside checkpoint_dir.

  Wraps sys.stdout/stderr with a Tee that writes every line both to the
  terminal and to `run.log`, so full runtime output (LLM calls, memory
  traces, token lines) is persisted even if the terminal scrollback is lost.
  Replacing the streams is undone on process exit automatically.
  """
  os.makedirs(checkpoint_dir, exist_ok=True)
  log_path = os.path.join(checkpoint_dir, 'run.log')

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


def _get_agent(
    env: interface.AsyncEnv,
    family: str | None = None,
) -> base_agent.EnvironmentInteractingAgent:
  """Gets agent."""
  print('Initializing agent...')
  agent = None
  if _AGENT_NAME.value == 'human_agent':
    agent = human_agent.HumanAgent(env)
  elif _AGENT_NAME.value == 'random_agent':
    agent = random_agent.RandomAgent(env)
  # Gemini.
  elif _AGENT_NAME.value == 'm3a_gemini_gcp':
    agent = m3a.M3A(
        env, infer.GeminiGcpWrapper(model_name='gemini-1.5-pro-latest')
    )
  elif _AGENT_NAME.value == 't3a_gemini_gcp':
    agent = t3a.T3A(
        env, infer.GeminiGcpWrapper(model_name='gemini-1.5-pro-latest')
    )
  # GPT.
  elif _AGENT_NAME.value == 't3a_gpt4':
    agent = t3a.T3A(env, infer.Gpt4Wrapper('gpt-4-turbo-2024-04-09'))
  elif _AGENT_NAME.value == 'm3a_gpt4v':
    agent = m3a.M3A(env, infer.Gpt4Wrapper('gpt-4-turbo-2024-04-09'))
  elif _AGENT_NAME.value == 'm3a_qwen3_vl_32b':
    agent = m3a.M3A(env, infer.Gpt4Wrapper('Qwen/Qwen3-VL-32B-Instruct'))
  # ATMem agent.
  elif _AGENT_NAME.value == 'atmem_qwen3_vl_32b':
    agent = atmem_agent.ATMemAgent(
        env, infer.Gpt4Wrapper('Qwen/Qwen3-VL-32B-Instruct'),
    )
  # DMS v3 agents.
  elif _AGENT_NAME.value == 'dms_gpt4v':
    agent = dms_agent.DMSAgent(
        env, infer.Gpt4Wrapper('gpt-4-turbo-2024-04-09')
    )
  elif _AGENT_NAME.value == 'dms_gemini_gcp':
    agent = dms_agent.DMSAgent(
        env, infer.GeminiGcpWrapper(model_name='gemini-1.5-pro-latest')
    )
  elif _AGENT_NAME.value == 'dms_qwen3_vl_32b':
    agent = dms_agent.DMSAgent(
        env,
        infer.Gpt4Wrapper('Qwen/Qwen3-VL-32B-Instruct'),
        screenshot_scale=_SCREENSHOT_SCALE.value,
    )
  # Memory-augmented agent with orthogonal U1/U2/U3 flags.
  elif _AGENT_NAME.value == 'm3a_qwen3_vl_32b_mem':
    rag_url = ''
    if _U3.value:
      rag_url = (
          _RAG_URL.value
          or os.environ.get('RAG_URL', 'http://127.0.0.1:18180')
      ).strip()
      if not rag_url:
        raise ValueError(
            'U3 is enabled but rag_url is empty. Set --rag_url or RAG_URL '
            '(AutoDL tunnel). Local page-graph fallback is disabled.'
        )
    agent_kwargs = dict(
        env=env,
        llm=infer.Gpt4Wrapper('Qwen/Qwen3-VL-32B-Instruct'),
        enable_u1=_U1.value,
        enable_u2=_U2.value,
        enable_u3=_U3.value,
        enable_u4=_U4.value,
        u2_persistence_dir=_U2_PERSISTENCE_DIR.value,
        u3_persistence_dir=_U3_PERSISTENCE_DIR.value,  # ignored; AutoDL-only
        u4_persistence_dir=_U4_PERSISTENCE_DIR.value,
        rag_url=rag_url,
        screenshot_scale=_SCREENSHOT_SCALE.value,
    )
    if _MULTIAGENT.value:
      agent = multi_agent.MultiAgentReflectorAgent(
          enable_multiagent=True, **agent_kwargs
      )
    else:
      agent = memory_agent.MemoryAugmentedAgent(**agent_kwargs)
  # SeeAct.
  elif _AGENT_NAME.value == 'seeact':
    agent = seeact.SeeAct(env)

  if not agent:
    raise ValueError(f'Unknown agent: {_AGENT_NAME.value}')

  if (
      agent.name in ['M3A', 'T3A', 'SeeAct']
      and family
      and family.startswith('miniwob')
      and hasattr(agent, 'set_task_guidelines')
  ):
    agent.set_task_guidelines(_MINIWOB_ADDITIONAL_GUIDELINES)
  agent.name = _AGENT_NAME.value

  return agent


def _resolve_tasks() -> list[str] | None:
  """Resolves --tasks and --difficulty into a concrete task list."""
  tasks = list(_TASKS.value) if _TASKS.value else None
  if _DIFFICULTY.value:
    difficulty_tasks = suite_utils.get_tasks_by_difficulty(_DIFFICULTY.value)
    if tasks is None:
      tasks = difficulty_tasks
    else:
      tasks = [t for t in tasks if t in set(difficulty_tasks)]
    if not tasks:
      raise ValueError(
          f'No tasks left after applying --difficulty={_DIFFICULTY.value}'
          + (f' and --tasks={_TASKS.value}' if _TASKS.value else '')
      )
    print(
        f'Running {len(tasks)} task(s) with difficulty={_DIFFICULTY.value}:'
        f' {tasks}'
    )
  return tasks


def _main() -> None:
  """Runs eval suite and gets rewards back."""
  adb_path = _ADB_PATH.value or _find_adb_directory()
  env = env_launcher.load_and_setup_env(
      console_port=_DEVICE_CONSOLE_PORT.value,
      emulator_setup=_EMULATOR_SETUP.value,
      adb_path=adb_path,
  )

  n_task_combinations = _N_TASK_COMBINATIONS.value
  task_registry = registry.TaskRegistry()
  suite = suite_utils.create_suite(
      task_registry.get_registry(family=_SUITE_FAMILY.value),
      n_task_combinations=n_task_combinations,
      seed=_TASK_RANDOM_SEED.value,
      tasks=_resolve_tasks(),
      use_identical_params=_FIXED_TASK_SEED.value,
  )
  suite.suite_family = _SUITE_FAMILY.value

  agent = _get_agent(env, _SUITE_FAMILY.value)

  if _SUITE_FAMILY.value.startswith('miniwob'):
    # MiniWoB pages change quickly, don't need to wait for screen to stabilize.
    agent.transition_pause = _MINIWOB_TRANSITION_PAUSE
  else:
    agent.transition_pause = None

  if _CHECKPOINT_DIR.value:
    checkpoint_dir = _CHECKPOINT_DIR.value
  else:
    checkpoint_dir = checkpointer_lib.create_run_directory(_OUTPUT_PATH.value)

  # Tee all runtime output (prints, LLM token lines, memory traces) into a
  # per-run log file so nothing is lost to the terminal scrollback.
  _install_run_logger(checkpoint_dir)

  print(
      f'Starting eval with agent {_AGENT_NAME.value} and writing to'
      f' {checkpoint_dir}'
  )
  suite_utils.run(
      suite,
      agent,
      checkpointer=checkpointer_lib.IncrementalCheckpointer(checkpoint_dir),
      demo_mode=False,
  )
  print(
      f'Finished running agent {_AGENT_NAME.value} on {_SUITE_FAMILY.value}'
      f' family. Wrote to {checkpoint_dir}.'
  )
  env.close()


def main(argv: Sequence[str]) -> None:
  del argv
  _main()


if __name__ == '__main__':
  app.run(main)
