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

"""Some LLM inference interface."""

import abc
import base64
import io
import os
import time
from typing import Any, Optional

import numpy as np
from PIL import Image
import requests

try:
  import google.generativeai as genai
  from google.generativeai import types
  from google.generativeai.types import answer_types
  from google.generativeai.types import content_types
  from google.generativeai.types import generation_types
  from google.generativeai.types import safety_types
except ImportError:  # Gemini optional; OpenAI-compatible wrappers still work.
  genai = None
  types = None
  answer_types = None
  content_types = None
  generation_types = None
  safety_types = None


ERROR_CALLING_LLM = 'Error calling LLM'


def array_to_jpeg_bytes(image: np.ndarray) -> bytes:
  """Converts a numpy array into a byte string for a JPEG image."""
  image = Image.fromarray(image)
  return image_to_jpeg_bytes(image)


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
  in_mem_file = io.BytesIO()
  image.save(in_mem_file, format='JPEG')
  # Reset file pointer to start
  in_mem_file.seek(0)
  img_bytes = in_mem_file.read()
  return img_bytes


class LlmWrapper(abc.ABC):
  """Abstract interface for (text only) LLM."""

  def __init__(self):
    self._token_usage = {'prompt': 0, 'completion': 0, 'cache_hit': 0}

  def reset_token_usage(self) -> None:
    self._token_usage = {'prompt': 0, 'completion': 0, 'cache_hit': 0}

  @property
  def token_usage(self) -> dict[str, int]:
    return dict(self._token_usage)

  def _record_usage(self, usage: dict | None) -> None:
    if not usage:
      return
    cached = usage.get('prompt_cache_hit_tokens') or 0
    cached_td = (usage.get('prompt_tokens_details') or {}).get('cached_tokens') or 0
    self._token_usage['prompt'] += usage.get('prompt_tokens') or 0
    self._token_usage['completion'] += usage.get('completion_tokens') or 0
    self._token_usage['cache_hit'] += cached or cached_td

  @abc.abstractmethod
  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    """Calling text-only LLM with a prompt.

    Args:
      text_prompt: Text prompt.

    Returns:
      Text output, is_safe, and raw output.
    """


class MultimodalLlmWrapper(abc.ABC):
  """Abstract interface for Multimodal LLM."""

  @abc.abstractmethod
  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    """Calling multimodal LLM with a prompt and a list of images.

    Args:
      text_prompt: Text prompt.
      images: List of images as numpy ndarray.

    Returns:
      Text output and raw output.
    """


SAFETY_SETTINGS_BLOCK_NONE = (
    {
        types.HarmCategory.HARM_CATEGORY_HARASSMENT: (
            types.HarmBlockThreshold.BLOCK_NONE
        ),
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: (
            types.HarmBlockThreshold.BLOCK_NONE
        ),
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: (
            types.HarmBlockThreshold.BLOCK_NONE
        ),
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: (
            types.HarmBlockThreshold.BLOCK_NONE
        ),
    }
    if types is not None
    else {}
)


class GeminiGcpWrapper(LlmWrapper, MultimodalLlmWrapper):
  """Gemini GCP interface."""

  def __init__(
      self,
      model_name: str | None = None,
      max_retry: int = 3,
      temperature: float = 0.0,
      top_p: float = 0.95,
      enable_safety_checks: bool = True,
  ):
    if genai is None:
      raise RuntimeError(
          'google-generativeai is not installed; required for GeminiGcpWrapper.'
      )
    if 'GCP_API_KEY' not in os.environ:
      raise RuntimeError('GCP API key not set.')
    genai.configure(api_key=os.environ['GCP_API_KEY'])
    self.llm = genai.GenerativeModel(
        model_name,  # pyrefly: ignore[bad-argument-type]
        safety_settings=None
        if enable_safety_checks
        else SAFETY_SETTINGS_BLOCK_NONE,
        generation_config=generation_types.GenerationConfig(
            temperature=temperature, top_p=top_p
        ),
    )
    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, 5)

  def predict(
      self,
      text_prompt: str,
      enable_safety_checks: bool = True,
      generation_config: generation_types.GenerationConfigType | None = None,
  ) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(
        text_prompt, [], enable_safety_checks, generation_config
    )

  def is_safe(self, raw_response):
    try:
      return (
          raw_response.candidates[0].finish_reason
          != answer_types.FinishReason.SAFETY
      )
    except Exception:  # pylint: disable=broad-exception-caught
      #  Assume safe if the response is None or doesn't have candidates.
      return True

  def predict_mm(
      self,
      text_prompt: str,
      images: list[np.ndarray],
      enable_safety_checks: bool = True,
      generation_config: generation_types.GenerationConfigType | None = None,
  ) -> tuple[str, Optional[bool], Any]:
    counter = self.max_retry
    retry_delay = 1.0
    output = None
    while counter > 0:
      try:
        output = self.llm.generate_content(
            [text_prompt] + [Image.fromarray(image) for image in images],
            safety_settings=None
            if enable_safety_checks
            else SAFETY_SETTINGS_BLOCK_NONE,
            generation_config=generation_config,
        )
        return output.text, True, output
      except Exception as e:  # pylint: disable=broad-exception-caught
        counter -= 1
        print('Error calling LLM, will retry in {retry_delay} seconds')
        print(e)
        if counter > 0:
          # Expo backoff
          time.sleep(retry_delay)
          retry_delay *= 2

    if (output is not None) and (not self.is_safe(output)):
      return ERROR_CALLING_LLM, False, output
    return ERROR_CALLING_LLM, None, None

  def generate(
      self,
      contents: (
          content_types.ContentsType | list[str | np.ndarray | Image.Image]
      ),
      safety_settings: safety_types.SafetySettingOptions | None = None,
      generation_config: generation_types.GenerationConfigType | None = None,
  ) -> tuple[str, Any]:
    """Exposes the generate_content API.

    Args:
      contents: The input to the LLM.
      safety_settings: Safety settings.
      generation_config: Generation config.

    Returns:
      The output text and the raw response.
    Raises:
      RuntimeError:
    """
    counter = self.max_retry
    retry_delay = 1.0
    response = None
    if isinstance(contents, list):
      contents = self.convert_content(contents)
    while counter > 0:
      try:
        response = self.llm.generate_content(
            contents=contents,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )
        return response.text, response
      except Exception as e:  # pylint: disable=broad-exception-caught
        counter -= 1
        print('Error calling LLM, will retry in {retry_delay} seconds')
        print(e)
        if counter > 0:
          # Expo backoff
          time.sleep(retry_delay)
          retry_delay *= 2
    raise RuntimeError(f'Error calling LLM. {response}.')

  def convert_content(
      self,
      contents: list[str | np.ndarray | Image.Image],
  ) -> content_types.ContentsType:
    """Converts a list of contents to a ContentsType."""
    converted = []
    for item in contents:
      if isinstance(item, str):
        converted.append(item)
      elif isinstance(item, np.ndarray):
        converted.append(Image.fromarray(item))
      elif isinstance(item, Image.Image):
        converted.append(item)
    return converted


class Gpt4Wrapper(LlmWrapper, MultimodalLlmWrapper):
  """OpenAI GPT4 wrapper.

  Attributes:
    openai_api_key: The class gets the OpenAI api key either explicitly, or
      through env variable in which case just leave this empty.
    max_retry: Max number of retries when some error happens.
    temperature: The temperature parameter in LLM to control result stability.
    model: GPT model to use based on if it is multimodal.
  """

  RETRY_WAITING_SECONDS = 20
  # Avoid hanging forever when the provider never responds (common with VL).
  REQUEST_TIMEOUT_SECONDS = 180

  def __init__(
      self,
      model_name: str,
      max_retry: int = 3,
      temperature: float = 0.0,
  ):
    if 'OPENAI_API_KEY' not in os.environ:
      raise RuntimeError('OpenAI API key not set.')
    self.openai_api_key = os.environ['OPENAI_API_KEY']
    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, 5)
    self.temperature = temperature
    self.model = model_name

  @classmethod
  def encode_image(cls, image: np.ndarray) -> str:
    return base64.b64encode(array_to_jpeg_bytes(image)).decode('utf-8')

  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(text_prompt, [])

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {self.openai_api_key}',
    }

    payload = {
        'model': self.model,
        'temperature': self.temperature,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': text_prompt},
            ],
        }],
        'max_tokens': 4096,
    }

    # Gpt-4v supports multiple images, just need to insert them in the content
    # list.
    for image in images:
      payload['messages'][0]['content'].append({
          'type': 'image_url',
          'image_url': {  # pyrefly: ignore[bad-assignment]
              'url': f'data:image/jpeg;base64,{self.encode_image(image)}'
          },
      })

    counter = self.max_retry
    wait_seconds = self.RETRY_WAITING_SECONDS
    while counter > 0:
      try:
        response = requests.post(
            os.environ.get(
                'OPENAI_BASE_URL',
                'https://api.openai.com/v1/chat/completions',
            ),
            headers=headers,
            json=payload,
            timeout=self.REQUEST_TIMEOUT_SECONDS,
        )
        if response.ok and 'choices' in response.json():
          # Log AND accumulate token usage (OpenAI-compatible APIs return it in
          # the body).  `_token_usage` is the episode-level counter read by
          # suite_utils at task end; the print keeps a live per-call trace.
          usage = response.json().get('usage')
          if usage:
            cached = usage.get('prompt_cache_hit_tokens')
            cached_td = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
            print(
                f"[tokens] model={self.model} prompt={usage.get('prompt_tokens')} "
                f"completion={usage.get('completion_tokens')} "
                f"total={usage.get('total_tokens')} "
                f"cache_hit={cached} cache_td={cached_td}"
            )
            self._record_usage(usage)
          return (
              response.json()['choices'][0]['message']['content'],
              None,
              response,
          )
        error_message = 'unknown error'
        try:
          error_message = response.json()['error']['message']
        except Exception:  # pylint: disable=broad-exception-caught
          error_message = response.text[:500]
        print(
            'Error calling OpenAI API with error message: ' + error_message
        )
        counter -= 1
        if counter <= 0:
          break
        time.sleep(wait_seconds)
        wait_seconds *= 2
      except requests.Timeout as e:
        counter -= 1
        print(
            f'LLM request timed out after {self.REQUEST_TIMEOUT_SECONDS}s,'
            f' retries left={counter}...'
        )
        print(e)
        if counter <= 0:
          break
        time.sleep(wait_seconds)
        wait_seconds *= 2
      except Exception as e:  # pylint: disable=broad-exception-caught
        # Want to catch all exceptions happened during LLM calls.
        counter -= 1
        print('Error calling LLM, will retry soon...')
        print(e)
        if counter <= 0:
          break
        time.sleep(wait_seconds)
        wait_seconds *= 2
    return ERROR_CALLING_LLM, None, None
