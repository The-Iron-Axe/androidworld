"""Minimal prompt-caching probe for SiliconFlow / any OpenAI-compatible API.

Sends the same fixed prefix twice and reports cache hit fields.
Usage (must have OPENAI_API_KEY / OPENAI_BASE_URL in env):
  python scripts/probe_cache.py
"""
import os
import sys
import time

import requests

BASE = os.environ.get(
    'OPENAI_BASE_URL', 'https://api.siliconflow.cn/v1/chat/completions'
)
MODEL = os.environ.get('PROBE_MODEL', 'Qwen/Qwen3-VL-32B-Instruct')

# A long, identical fixed prefix to test caching on. Text-only probe (no images)
# to isolate the prefix-cache mechanism from vision prefill.
FIXED_PREFIX = ('Imagine an extremely long system instruction. ' * 40)
FIXED_PREFIX += '\nRepeat: keep every character identical between the two calls. '

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {os.environ["OPENAI_API_KEY"]}',
}


def call(suffix: str) -> dict:
    payload = {
        'model': MODEL,
        'messages': [
            {'role': 'user', 'content': FIXED_PREFIX + suffix},
        ],
        'max_tokens': 16,
    }
    r = requests.post(BASE, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def main() -> int:
    if 'OPENAI_API_KEY' not in os.environ:
        print('OPENAI_API_KEY not set', file=sys.stderr)
        return 1

    print(f'base={BASE}')
    print(f'model={MODEL}')
    print(f'prefix_chars={len(FIXED_PREFIX)}')

    # Call 1: cold cache.
    t0 = time.time()
    r1 = call('FIRST CALL, answer 1')
    dt1 = time.time() - t0
    # Call 2: same prefix, should hit if caching is supported.
    t0 = time.time()
    r2 = call('SECOND CALL, answer 2')
    dt2 = time.time() - t0

    for tag, r in (('call1(cold)', r1), ('call2(warm)', r2)):
        u = r.get('usage', {})
        print(f'--- {tag} raw usage JSON ---')
        print(u)
        td = u.get('prompt_tokens_details') or {}
        print(
            f"{tag}: prompt={u.get('prompt_tokens')} "
            f"hit={u.get('prompt_cache_hit_tokens')} "
            f"miss={u.get('prompt_cache_miss_tokens')} "
            f"cached_td={td.get('cached_tokens')} "
            f"latency={dt1 if tag.startswith('call1') else dt2:.1f}s"
        )

    u1 = r1.get('usage', {})
    u2 = r2.get('usage', {})
    h1 = u1.get('prompt_cache_hit_tokens') or 0
    h2 = u2.get('prompt_cache_hit_tokens') or 0
    print('RESULT:', 'CACHING ACTIVE' if h2 > h1 else 'NO CACHE HIT (unsupported or not auto)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
