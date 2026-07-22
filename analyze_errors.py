#!/usr/bin/env python3
"""Analyze AndroidWorld checkpoint data for error patterns."""

import gzip
import io
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

# Fix Windows GBK encoding issue
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def load_episode(filepath):
    with open(filepath, 'rb') as f:
        compressed = f.read()
    with gzip.open(io.BytesIO(compressed), 'rb') as f_in:
        return pickle.load(f_in)


def load_all_episodes(directory):
    episodes = []
    for fname in sorted(os.listdir(directory)):
        if fname.endswith('.pkl.gz'):
            data = load_episode(os.path.join(directory, fname))
            if isinstance(data, list):
                for ep in data:
                    if isinstance(ep, dict):
                        ep['_file'] = fname
                        episodes.append(ep)
            elif isinstance(data, dict):
                data['_file'] = fname
                episodes.append(data)
    return episodes


def analyze_episode(ep):
    errors = []
    import numpy as np

    episode_data = ep.get('episode_data', {})
    if not isinstance(episode_data, dict):
        return [('UNRUNNABLE', 0, 'No step data (exception during run)', '')]
    try:
        if isinstance(episode_data, float) and np.isnan(episode_data):
            return [('UNRUNNABLE', 0,
                     f'Task exception: {ep.get("exception_info", "unknown")}', '')]
    except Exception:
        pass

    n_steps = len(episode_data.get('step_number', []))
    task_name = ep.get('task_template', 'Unknown')
    goal = ep.get('goal', '')
    is_successful = ep.get('is_successful', 0.0)

    def get_field(key, idx):
        val = episode_data.get(key, [])
        if isinstance(val, list) and idx < len(val):
            return val[idx]
        return None

    page_fingerprints = []
    clicked_indices = set()
    action_types_history = []

    for i in range(n_steps):
        summary = str(get_field('summary', i) or '')
        action_json = get_field('action_output_json', i)
        action_output = str(get_field('action_output', i) or '')
        reason = str(get_field('action_reason', i) or '')
        ui_elements = get_field('before_ui_elements', i) or []

        # 1. Format Error
        if 'not in the correct format' in summary:
            errors.append(('FORMAT_ERROR', i + 1,
                          'Output not in correct JSON format', summary[:200]))

        # 2. Hallucination (index out of range)
        if 'index is out of range' in summary:
            errors.append(('HALLUCINATION', i + 1,
                          'Model referenced nonexistent UI element index', summary[:200]))

        # 3. Model Refusal
        if 'Agent thinks' in summary or 'infeasible' in action_output.lower():
            errors.append(('MODEL_REFUSAL', i + 1,
                          'Agent thinks task is infeasible or gave up', summary[:200]))
        if 'safety' in summary.lower() or 'safety' in reason.lower():
            errors.append(('MODEL_REFUSAL', i + 1,
                          'Safety classifier triggered', summary[:200]))

        # 4. Suspected Grounding error
        if action_json is not None:
            try:
                at = getattr(action_json, 'action_type', '')
                ai = getattr(action_json, 'index', None)
                if (at in ('click', 'long_press', 'input_text')
                        and ai is not None
                        and isinstance(ui_elements, list)
                        and ai < len(ui_elements)):
                    target = ui_elements[ai]
                    ic = getattr(target, 'is_clickable', True)
                    txt = getattr(target, 'text', '') or ''
                    desc = getattr(target, 'content_description', '') or ''
                    if not ic:
                        errors.append(('GROUNDING_SUSPECT', i + 1,
                                      f'Clicked non-clickable element idx={ai} '
                                      f'text="{txt}" desc="{desc}"', ''))
            except Exception:
                pass

        # 5. Planning Loop
        fp = hash(str(ui_elements[:20]) if ui_elements else '')
        page_fingerprints.append(fp)
        atype = ''
        if action_json is not None:
            try:
                atype = getattr(action_json, 'action_type', '') or ''
            except Exception:
                pass
        action_types_history.append(atype)

        if i >= 2:
            if (len(set(page_fingerprints[-3:])) == 1
                    and len(set(action_types_history[-3:])) == 1
                    and action_types_history[-3:] != [''] * 3):
                errors.append(('PLANNING_LOOP', i + 1,
                              f'Repeated "{action_types_history[-1]}" 3x on same page', ''))

        # 6. History Amnesia
        if action_json is not None:
            try:
                ai = getattr(action_json, 'index', None)
                if ai is not None and ai in clicked_indices:
                    errors.append(('HISTORY_AMNESIA', i + 1,
                                  f'Repeated click on element idx={ai}', ''))
                if ai is not None:
                    clicked_indices.add(ai)
            except Exception:
                pass

        # 7. Knowledge Gap
        fail_kw = ['did not', 'not work', 'failed', 'unable', 'cannot',
                   "didn't", 'unsuccessful', 'wrong']
        if i >= 3:
            recent_sums = []
            for j in range(max(0, i - 3), i + 1):
                s = str(get_field('summary', j) or '')
                recent_sums.append(s)
            n_fail = sum(1 for s in recent_sums if any(kw in s.lower() for kw in fail_kw))
            recent_pages = page_fingerprints[max(0, i - 3):i + 1]
            same_page = sum(1 for p in recent_pages if p == fp)
            if n_fail >= 2 and same_page >= 3:
                errors.append(('KNOWLEDGE_GAP', i + 1,
                              f'Multiple failed attempts on same page '
                              f'({n_fail}/{len(recent_sums)} steps had failure keywords)', ''))

    # 8. Premature Termination
    agent_claimed_done = False
    for i in range(n_steps):
        aj = get_field('action_output_json', i)
        if aj is not None:
            try:
                if (getattr(aj, 'action_type', '') == 'status'
                        and getattr(aj, 'goal_status', '') == 'complete'):
                    agent_claimed_done = True
            except Exception:
                pass

    if agent_claimed_done and is_successful != 1.0:
        errors.insert(0, ('PREMATURE_TERMINATION', 0,
                         'Agent claimed complete but task.is_successful() returned False',
                         f'goal = {goal}'))

    return errors


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_errors.py <checkpoint_dir> [...]")
        sys.exit(1)

    all_errors_by_task = defaultdict(list)
    all_success = []
    total_tasks = 0

    for path in sys.argv[1:]:
        p = Path(path)
        if not p.is_dir():
            continue

        episodes = load_all_episodes(str(p))
        print(f"\n{'='*70}")
        print(f"Directory: {path}")
        print(f"Total: {len(episodes)} tasks")
        print(f"{'='*70}")

        for ep in episodes:
            task_name = ep.get('task_template', 'Unknown')
            is_successful = ep.get('is_successful', 0)
            total_tasks += 1
            all_success.append((task_name, is_successful))

            errs = analyze_episode(ep)
            if errs:
                all_errors_by_task[task_name].extend(errs)

            status = '[OK]' if is_successful == 1.0 else '[FAIL]'
            print(f"  {status} {task_name} - {len(errs)} error(s)")

    # ---- Summary ----
    error_counts = defaultdict(int)
    for task_errs in all_errors_by_task.values():
        for etype, step, desc, detail in task_errs:
            error_counts[etype] += 1

    n_ok = sum(1 for _, s in all_success if s == 1.0)
    print(f"\n{'='*70}")
    print(f"Summary: {total_tasks} tasks, {n_ok} successful "
          f"({100 * n_ok / max(total_tasks, 1):.1f}%)")
    print(f"{'='*70}")

    ERROR_LABELS = {
        'FORMAT_ERROR': ('Format Error (JSON parse failed)', 100),
        'HALLUCINATION': ('Hallucination (bad element index)', 100),
        'PREMATURE_TERMINATION': ('Premature Termination', 100),
        'MODEL_REFUSAL': ('Model Refusal / Giving Up', 100),
        'GROUNDING_SUSPECT': ('Suspected Grounding Error', 'medium'),
        'PLANNING_LOOP': ('Planning Loop (repeated action)', 'high'),
        'HISTORY_AMNESIA': ('History Amnesia (repeated click)', 'medium'),
        'KNOWLEDGE_GAP': ('Knowledge Gap (multiple failures)', 'medium'),
    }

    print(f"\n{'Error Type':<42} {'Count':<6} {'Certainty'}")
    print("-" * 65)
    for etype, (label, certainty) in ERROR_LABELS.items():
        cnt = error_counts.get(etype, 0)
        bar = '#' * cnt if cnt > 0 else ''
        print(f"  {label:<40} {cnt:<6} {str(certainty):<8} {bar}")

    # ---- Per-task breakdown ----
    print(f"\n{'='*70}")
    print("Per-task Breakdown")
    print(f"{'='*70}")
    for task_name, errs in sorted(all_errors_by_task.items()):
        etype_summary = defaultdict(int)
        for etype, step, desc, _ in errs:
            etype_summary[etype] += 1
        short_labels = {
            'FORMAT_ERROR': 'FMT', 'HALLUCINATION': 'HAL',
            'PREMATURE_TERMINATION': 'PRETERM', 'MODEL_REFUSAL': 'REFUSE',
            'GROUNDING_SUSPECT': 'GRD', 'PLANNING_LOOP': 'LOOP',
            'HISTORY_AMNESIA': 'AMNESIA', 'KNOWLEDGE_GAP': 'KGAP',
        }
        parts = [f"{short_labels.get(k, k)}:{v}" for k, v in etype_summary.items()]
        print(f"  [FAIL] {task_name}: {', '.join(parts)}")

    # ---- Detailed errors ----
    print(f"\n{'='*70}")
    print("Detailed Error Log (first 30)")
    print(f"{'='*70}")
    count = 0
    for task_name, errs in sorted(all_errors_by_task.items()):
        for etype, step, desc, detail in errs:
            if count >= 30:
                break
            print(f"\n  [{etype}] {task_name} step={step}")
            print(f"    {desc}")
            if detail:
                print(f"    detail: {detail[:150]}")
            count += 1


if __name__ == '__main__':
    main()
