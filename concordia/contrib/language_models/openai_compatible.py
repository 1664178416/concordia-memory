# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Language models for OpenAI-compatible chat completion endpoints.

This wrapper intentionally uses only the Python standard library.  Different
providers expose slightly different subsets of the OpenAI request schema, so
it sends the portable ``max_tokens``/``top_p`` fields and omits provider-
specific reasoning parameters.
"""

from collections.abc import Collection, Mapping, Sequence
import json
import os
import time
from typing import Any, override
import urllib.error
import urllib.request

from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib

_MAX_MULTIPLE_CHOICE_ATTEMPTS = 5
_RETRYABLE_STATUS_CODES = frozenset((429, 500, 502, 503, 504))
_VECTOR_ENGINE_BASE_URL = 'https://api.vectorengine.ai/v1'


class OpenAICompatibleError(RuntimeError):
  """An error returned by an OpenAI-compatible endpoint."""

  def __init__(self, message: str, *, status_code: int | None = None):
    super().__init__(message)
    self.status_code = status_code


def _error_message(body: bytes) -> str:
  """Extracts a short provider error without exposing request headers."""
  try:
    data = json.loads(body.decode('utf-8'))
  except (UnicodeDecodeError, json.JSONDecodeError):
    return body.decode('utf-8', errors='replace')[:300]

  error = data.get('error') if isinstance(data, dict) else None
  if isinstance(error, dict) and error.get('message'):
    return str(error['message'])[:300]
  if isinstance(error, str):
    return error[:300]
  return 'The endpoint returned an error response.'


def _content_from_response(response: Mapping[str, Any]) -> str:
  """Reads text from the common chat-completions response variants."""
  choices = response.get('choices')
  if not isinstance(choices, list) or not choices:
    raise OpenAICompatibleError('Response did not contain any choices.')

  choice = choices[0]
  if not isinstance(choice, dict):
    raise OpenAICompatibleError('Response choice had an invalid shape.')

  message = choice.get('message')
  content: Any = message.get('content') if isinstance(message, dict) else None
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    text_parts = [
        part.get('text', '')
        for part in content
        if isinstance(part, dict) and isinstance(part.get('text'), str)
    ]
    if text_parts:
      return ''.join(text_parts)

  # A few older compatible servers return completion-style ``text``.
  text = choice.get('text')
  if isinstance(text, str):
    return text
  raise OpenAICompatibleError('Response choice did not contain text.')


def _normalise_choice(value: str) -> str:
  value = ' '.join(value.strip().split())
  value = value.strip('`"\'')
  if value.endswith('.'):
    value = value[:-1].rstrip()
  return value


class OpenAICompatibleLanguageModel(language_model.LanguageModel):
  """A small dependency-free wrapper for chat-completions compatible APIs."""

  def __init__(
      self,
      model_name: str,
      *,
      api_key: str | None = None,
      api_base: str | None = None,
      system_message: str | None = None,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
      max_retries: int = 2,
  ):
    self._model_name = model_name
    self._api_key = (
        api_key
        or os.getenv('OPENAI_API_KEY')
        or os.getenv('VECTORENGINE_API_KEY')
    )
    if not self._api_key:
      raise ValueError(
          'No API key found. Provide api_key or set OPENAI_API_KEY or '
          'VECTORENGINE_API_KEY.'
      )

    self._api_base = (
        api_base
        or os.getenv('OPENAI_BASE_URL')
        or os.getenv('VECTORENGINE_BASE_URL')
    )
    if not self._api_base:
      raise ValueError(
          'No API base URL found. Provide api_base or set OPENAI_BASE_URL '
          'or VECTORENGINE_BASE_URL.'
      )
    self._endpoint = self._api_base.rstrip('/') + '/chat/completions'
    self._system_message = system_message
    self._measurements = measurements
    self._channel = channel
    self._max_retries = max(0, max_retries)

  def _request_json(
      self, payload: Mapping[str, Any], *, timeout: float
  ) -> Mapping[str, Any]:
    request = urllib.request.Request(
        self._endpoint,
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Authorization': f'Bearer {self._api_key}',
            'Content-Type': 'application/json',
        },
        method='POST',
    )

    for attempt in range(self._max_retries + 1):
      try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
          body = response.read()
        data = json.loads(body.decode('utf-8'))
        if not isinstance(data, dict):
          raise OpenAICompatibleError('Endpoint returned a non-object JSON.')
        if 'error' in data:
          raise OpenAICompatibleError(_error_message(body))
        return data
      except urllib.error.HTTPError as error:
        body = error.read()
        if (
            error.code in _RETRYABLE_STATUS_CODES
            and attempt < self._max_retries
        ):
          time.sleep(min(2**attempt, 4))
          continue
        raise OpenAICompatibleError(
            f'Chat completion failed with HTTP {error.code}: '
            f'{_error_message(body)}',
            status_code=error.code,
        ) from error
      except (TimeoutError, urllib.error.URLError) as error:
        if attempt < self._max_retries:
          time.sleep(min(2**attempt, 4))
          continue
        raise TimeoutError(
            f'Chat completion request failed for {self._api_base}.'
        ) from error

    raise AssertionError('Retry loop did not return or raise.')

  def _chat(
      self,
      prompt: str,
      *,
      max_tokens: int,
      terminators: Collection[str],
      temperature: float,
      top_p: float,
      timeout: float,
      seed: int | None,
  ) -> str:
    messages: list[dict[str, str]] = []
    if self._system_message:
      messages.append({'role': 'system', 'content': self._system_message})
    messages.append({'role': 'user', 'content': prompt})

    payload: dict[str, Any] = {
        'model': self._model_name,
        'messages': messages,
        'temperature': temperature,
        'top_p': top_p,
        'max_tokens': max_tokens,
    }
    if terminators:
      payload['stop'] = list(terminators)
    if seed is not None:
      payload['seed'] = seed

    response = self._request_json(payload, timeout=timeout)
    result = _content_from_response(response)
    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel, {'raw_text_length': len(result)}
      )
    return result

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    del top_k  # OpenAI-compatible chat APIs do not share a top_k contract.
    return self._chat(
        prompt,
        max_tokens=max_tokens,
        terminators=terminators,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
        seed=seed,
    )

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    if not responses:
      raise ValueError('responses must contain at least one choice.')

    choice_prompt = (
        f'{prompt}\nRespond with exactly one option and no explanation.\n'
        + '\n'.join(f'- {response}' for response in responses)
    )
    answer = ''
    for attempt in range(_MAX_MULTIPLE_CHOICE_ATTEMPTS):
      answer = self.sample_text(
          choice_prompt,
          max_tokens=max(16, max(len(response) for response in responses) + 8),
          temperature=0.0 if attempt == 0 else 0.3,
          seed=seed,
      )
      normalised = _normalise_choice(answer)
      for index, response in enumerate(responses):
        if normalised == _normalise_choice(response):
          return index, response, {'attempts': attempt + 1}

    raise language_model.InvalidResponseError(
        'Too many multiple choice attempts. Last answer: ' + answer[:200]
    )


class VectorEngineLanguageModel(OpenAICompatibleLanguageModel):
  """OpenAI-compatible model with VectorEngine defaults."""

  def __init__(
      self,
      model_name: str,
      *,
      api_key: str | None = None,
      api_base: str | None = None,
      system_message: str | None = None,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
      max_retries: int = 2,
  ):
    super().__init__(
        model_name=model_name,
        api_key=api_key,
        api_base=(
            api_base
            or os.getenv('VECTORENGINE_BASE_URL')
            or _VECTOR_ENGINE_BASE_URL
        ),
        system_message=system_message,
        measurements=measurements,
        channel=channel,
        max_retries=max_retries,
    )
