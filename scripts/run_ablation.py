"""One-click runner for the U1-U4 hierarchical ablation.

Wraps scripts/ablation_hierarchical.py (shared-store, multi-seed protocol)
with the RAG-tunnel health check and a fixed task split, so the whole
ablation is launched with one command.  No ps1 needed — pure Python.

Protocol (per seed, independent stores):
  Stage A  record — all memory on, `record_rounds` rounds over record_tasks,
                    maturing u2/u3/u4 stores (DMS multi-round warm-up).
  Stage C  ablate — u12 -> u123 -> u1234 over eval_tasks, all reading the
                    same matured stores; gain = marginal contribution of the
                    layer just added.
  Final             per-config success mean±std across seeds.

Requires: AutoDL RAG service up + local SSH tunnel (default 127.0.0.1:18180).
Run:
    python scripts/run_ablation.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)

_ABLATION = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'ablation_hierarchical.py')

# ── Default task split (override via CLI flags) ────────────────────────
DEFAULT_RECORD_TASKS = [
    'MarkorCreateNoteAndSms',
    'MarkorMergeNotes',
    'ExpenseAddMultipleFromMarkor',
]
DEFAULT_EVAL_TASKS = [
    'MarkorTranscribeVideo',
    'RecipeAddMultipleRecipesFromMarkor2',
    'ExpenseDeleteMultiple2',
]


def check_rag(url: str, timeout: float = 10.0) -> bool:
  """Best-effort RAG health check via the local tunnel.  Non-fatal if
  the requests library is unavailable — the ablation itself will surface it."""
  try:
    import requests
  except ImportError:
    print(f'[run_ablation] requests unavailable; skipping RAG check '
          f'({url}/health). The ablation will fail loudly if U3 needs it.')
    return True
  try:
    r = requests.get(f'{url}/health', timeout=timeout)
    ok = r.status_code == 200
    print(f'[run_ablation] RAG {url}/health -> {"OK" if ok else f"HTTP {r.status_code}"}')
    return ok
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f'[run_ablation] RAG {url}/health -> UNREACHABLE: {e}')
    return False


def main() -> int:
  p = argparse.ArgumentParser(description='One-click hierarchical ablation.')
  p.add_argument('--record_tasks', nargs='*', default=DEFAULT_RECORD_TASKS,
                 help='Tasks that mature the stores (Stage A).')
  p.add_argument('--tasks', nargs='*', default=DEFAULT_EVAL_TASKS,
                 help='Tasks that evaluate each config (Stage C).')
  p.add_argument('--seeds', default='30,31', help='Comma-separated seeds.')
  p.add_argument('--record_rounds', type=int, default=3,
                 help='All-memory record rounds (DMS warm-up). 1 = smoke test.')
  p.add_argument('--rag_url', default=os.environ.get('RAG_URL',
                 'http://127.0.0.1:18180'))
  p.add_argument('--skip_rag_check', action='store_true',
                 help='Skip the RAG health check and run anyway.')
  args = p.parse_args()

  print('=== One-click ablation: shared store / multi-seed / hierarchical ===')
  print(f'  record tasks : {", ".join(args.record_tasks)}')
  print(f'  eval tasks   : {", ".join(args.tasks)}')
  print(f'  seeds        : {args.seeds}   (per-seed independent stores)')
  print(f'  record rounds: {args.record_rounds}')

  # ── RAG tunnel health (U3 is an ablation axis; dead tunnel poisons its row)
  if not args.skip_rag_check and not check_rag(args.rag_url):
    print('\nRAG tunnel unreachable. Fix then rerun:')
    print('  1. AutoDL:  bash /root/pg_agent_rag/scripts/start_server.sh')
    print('  2. Local:   .\\pg_agent_rag\\tunnel\\start_tunnel.ps1')
    print('  3. Verify:  .\\pg_agent_rag\\tunnel\\verify_tunnel.ps1')
    return 1

  # ── Launch the ablation ─────────────────────────────────────────────
  cmd = [
      sys.executable, _ABLATION,
      '--record_tasks=' + ','.join(args.record_tasks),
      '--tasks=' + ','.join(args.tasks),
      '--seeds=' + args.seeds,
      '--record_rounds=%d' % args.record_rounds,
      '--rag_url=' + args.rag_url,
      '--rag_on',
  ]
  print('\n[run_ablation] launching: ' + ' '.join(cmd) + '\n')
  start = time.time()
  proc = subprocess.run(cmd, cwd=_REPO_ROOT)
  if proc.returncode != 0:
    print(f'\n[run_ablation] ablation failed (exit={proc.returncode}). See output above.')
    return proc.returncode

  print(f'\n[run_ablation] done in {time.time() - start:.0f}s.')
  print('Per-round results in scripts/results/; summary printed by the ablation.')
  return 0


if __name__ == '__main__':
  sys.exit(main())
