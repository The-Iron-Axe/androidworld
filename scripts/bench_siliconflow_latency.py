"""Bench chat latency with a copied real action prompt (text only).

Prompt file: scripts/bench_siliconflow_prompt.txt

Usage:
  # SiliconFlow
  $env:OPENAI_API_KEY = "sk-..."
  $env:OPENAI_BASE_URL = "https://api.siliconflow.cn/v1/chat/completions"
  python scripts/bench_siliconflow_latency.py --model=Qwen/Qwen3-VL-32B-Instruct

  # OpenRouter + explicit prompt cache (Alibaba-style cache_control)
  $env:OPENAI_API_KEY = "sk-or-v1-..."
  $env:OPENAI_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
  python scripts/bench_siliconflow_latency.py `
    --model=qwen/qwen-plus --cache --n=5
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import requests

_PROMPT_PATH = Path(__file__).with_name('bench_siliconflow_prompt.txt')


def _usage_cache_info(data: dict) -> str:
  """Pull cached_tokens / cache_write from OpenRouter-style usage blobs."""
  usage = data.get('usage') or {}
  parts = []
  for k in (
      'prompt_tokens',
      'completion_tokens',
      'total_tokens',
      'cached_tokens',
      'cache_write_tokens',
  ):
    if k in usage and usage[k] is not None:
      parts.append(f'{k}={usage[k]}')
  details = (
      usage.get('prompt_tokens_details')
      or usage.get('input_tokens_details')
      or {}
  )
  if isinstance(details, dict):
    for k in ('cached_tokens', 'cache_write_tokens', 'cache_read_tokens'):
      if k in details and details[k] is not None:
        parts.append(f'details.{k}={details[k]}')
  return ' '.join(parts) if parts else 'usage=n/a'


def build_payload(
    *,
    model: str,
    text: str,
    max_tokens: int,
    cache: bool,
    session_id: str,
) -> dict:
  if cache:
    # OpenRouter / Alibaba explicit cache: content MUST be an array of
    # blocks; cache_control marks the stable prefix to cache.
    content: str | list = [
        {
            'type': 'text',
            'text': text,
            'cache_control': {'type': 'ephemeral'},
        }
    ]
  else:
    content = text

  payload: dict = {
      'model': model,
      'temperature': 0.0,
      'messages': [{'role': 'user', 'content': content}],
      'max_tokens': max_tokens,
  }
  if cache and session_id:
    # Sticky routing so follow-up requests hit the same provider cache.
    payload['session_id'] = session_id
  return payload


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument(
      '--url',
      default=os.environ.get(
          'OPENAI_BASE_URL', 'https://api.siliconflow.cn/v1/chat/completions'
      ),
  )
  p.add_argument('--api_key', default=os.environ.get('OPENAI_API_KEY', ''))
  p.add_argument('--model', default='Qwen/Qwen3-VL-32B-Instruct')
  p.add_argument('--n', type=int, default=3)
  p.add_argument('--timeout', type=float, default=180.0)
  p.add_argument('--max_tokens', type=int, default=4096)
  p.add_argument('--prompt', type=Path, default=_PROMPT_PATH)
  p.add_argument(
      '--cache',
      action='store_true',
      help=(
          'Enable OpenRouter/Alibaba-style explicit prompt caching '
          '(cache_control ephemeral + session_id).'
      ),
  )
  p.add_argument(
      '--session_id',
      default='androidworld-bench-cache-v1',
      help='Sticky session id used when --cache is set.',
  )
  args = p.parse_args()

  if not args.api_key:
    print('Missing OPENAI_API_KEY.', file=sys.stderr)
    return 1
  if not args.prompt.is_file():
    print(f'Prompt file not found: {args.prompt}', file=sys.stderr)
    return 1

  text = args.prompt.read_text(encoding='utf-8')
  print(f'URL    : {args.url}')
  print(f'Model  : {args.model}')
  print(f'Prompt : {args.prompt} ({len(text)} chars)')
  print(f'Cache  : {"ON (cache_control+session_id)" if args.cache else "off"}')
  if args.cache:
    print(f'Session: {args.session_id}')
  print(f'N      : {args.n}')
  print()

  payload = build_payload(
      model=args.model,
      text=text,
      max_tokens=args.max_tokens,
      cache=args.cache,
      session_id=args.session_id,
  )
  headers = {
      'Authorization': f'Bearer {args.api_key}',
      'Content-Type': 'application/json',
  }
  if args.cache:
    headers['HTTP-Referer'] = 'https://github.com/androidworld-bench'
    headers['X-Title'] = 'androidworld-bench'

  latencies: list[float] = []
  for i in range(1, args.n + 1):
    try:
      t0 = time.perf_counter()
      r = requests.post(
          args.url, headers=headers, json=payload, timeout=args.timeout
      )
      dt = time.perf_counter() - t0
    except requests.RequestException as e:
      print(f'[{i}/{args.n}] ERROR: {e}')
      continue
    latencies.append(dt)
    try:
      data = r.json()
    except json.JSONDecodeError:
      data = {}
    content = ''
    try:
      content = data['choices'][0]['message']['content'] or ''
    except (KeyError, IndexError, TypeError):
      content = r.text[:120]
    preview = content.replace('\n', ' ')[:120]
    cache_info = _usage_cache_info(data)
    provider = data.get('provider') or ''
    print(
        f'[{i}/{args.n}] status={r.status_code} latency_s={dt:.3f} '
        f'provider={provider} {cache_info}'
    )
    print(f'         content={preview}')

  if not latencies:
    print('No successful requests.')
    return 2

  print()
  print(f'count : {len(latencies)}')
  print(f'min_s : {min(latencies):.3f}')
  print(f'max_s : {max(latencies):.3f}')
  print(f'mean_s: {statistics.mean(latencies):.3f}')
  if len(latencies) >= 2:
    print(f'stdev : {statistics.stdev(latencies):.3f}')
  if args.cache:
    print(
        'Note: Alibaba explicit cache on OpenRouter only applies to listed '
        'models (e.g. qwen/qwen-plus, qwen/qwen3-max). VL-32B may ignore it; '
        'check cached_tokens above.'
    )
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
