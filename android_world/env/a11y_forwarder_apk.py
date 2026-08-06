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

"""Local cache + download helper for accessibility_forwarder.apk.

android_env downloads this APK via urllib on first env init. On some Windows
Python builds urllib SSL fails against Google Cloud Storage while curl/requests
work. We patch android_env to read a cached APK and download with requests.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import requests
from absl import logging
from android_env.wrappers import a11y_grpc_wrapper

_A11Y_FORWARDER_URL = (
    'https://storage.googleapis.com/android_env-tasks/'
    '2024.05.13-accessibility_forwarder.apk'
)
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / 'data'
_DEFAULT_CACHE_PATH = _DEFAULT_CACHE_DIR / '2024.05.13-accessibility_forwarder.apk'
_ENV_APK_PATH = 'ANDROID_WORLD_A11Y_FORWARDER_APK'

_patched = False


def _cache_path() -> Path:
  override = os.environ.get(_ENV_APK_PATH)
  if override:
    return Path(override).expanduser()
  return _DEFAULT_CACHE_PATH


def _download_with_curl(dest: Path, timeout_sec: float) -> None:
  """Downloads via curl.exe (Windows Schannel often works when Python SSL fails)."""
  curl = 'curl.exe' if sys.platform == 'win32' else 'curl'
  cmd = [
      curl,
      '-L',
      '--fail',
      '--silent',
      '--show-error',
      '--max-time',
      str(int(timeout_sec)),
      '-o',
      str(dest),
      _A11Y_FORWARDER_URL,
  ]
  logging.info('Downloading accessibility forwarder apk via curl to %s', dest)
  subprocess.run(cmd, check=True)


def download_a11y_forwarder_apk(
    dest: Path | None = None,
    timeout_sec: float = 120.0,
) -> Path:
  """Downloads accessibility_forwarder.apk to dest (or default cache path)."""
  dest = dest or _cache_path()
  dest.parent.mkdir(parents=True, exist_ok=True)
  tmp = dest.with_suffix(dest.suffix + '.tmp')
  try:
    logging.info('Downloading accessibility forwarder apk to %s', dest)
    try:
      response = requests.get(_A11Y_FORWARDER_URL, timeout=timeout_sec)
      response.raise_for_status()
      tmp.write_bytes(response.content)
    except requests.RequestException as error:
      logging.warning(
          'requests download failed (%s); trying curl fallback.', error
      )
      _download_with_curl(tmp, timeout_sec)
    tmp.replace(dest)
  finally:
    if tmp.exists() and not dest.exists():
      tmp.unlink(missing_ok=True)
  logging.info(
      'Saved accessibility forwarder apk (%d bytes).', dest.stat().st_size
  )
  return dest


def ensure_a11y_forwarder_apk() -> bytes:
  """Returns APK bytes, using local cache when available."""
  path = _cache_path()
  if path.is_file():
    logging.info('Using cached accessibility forwarder apk at %s', path)
    return path.read_bytes()
  download_a11y_forwarder_apk(path)
  return path.read_bytes()


def patch_a11y_forwarder_apk_download() -> None:
  """Monkey-patch android_env to use cached/requests-based APK fetch."""
  global _patched
  if _patched:
    return

  def _get_accessibility_forwarder_apk() -> bytes:
    return ensure_a11y_forwarder_apk()

  a11y_grpc_wrapper._get_accessibility_forwarder_apk = (  # pylint: disable=protected-access
      _get_accessibility_forwarder_apk
  )
  _patched = True
  logging.info('Patched android_env accessibility forwarder APK download.')
