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

"""Tests for the dependency-free OpenAI-compatible language model wrapper."""

from collections.abc import Mapping
import json
import os
from unittest import mock
import urllib.request

from absl.testing import absltest
from concordia.contrib import language_models
from concordia.contrib.language_models import openai_compatible


class _FakeResponse:

  def __init__(self, payload: Mapping[str, object]):
    self._body = json.dumps(payload).encode('utf-8')

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    del exc_type, exc_value, traceback

  def read(self):
    return self._body


class OpenAICompatibleLanguageModelTest(absltest.TestCase):

  def _model(self):
    return openai_compatible.OpenAICompatibleLanguageModel(
        model_name='test-model',
        api_key='test-key',
        api_base='https://example.test/v1',
        max_retries=0,
    )

  @mock.patch.object(urllib.request, 'urlopen')
  def test_sample_text_uses_portable_request_fields(self, urlopen):
    urlopen.return_value = _FakeResponse({
        'choices': [{'message': {'content': 'hello'}}],
    })

    result = self._model().sample_text(
        'say hello', max_tokens=12, temperature=0.2, seed=7
    )

    self.assertEqual(result, 'hello')
    request = urlopen.call_args.args[0]
    self.assertEqual(
        request.full_url, 'https://example.test/v1/chat/completions'
    )
    self.assertEqual(request.get_header('Authorization'), 'Bearer test-key')
    payload = json.loads(request.data)
    self.assertEqual(payload['model'], 'test-model')
    self.assertEqual(payload['max_tokens'], 12)
    self.assertEqual(payload['seed'], 7)
    self.assertNotIn('reasoning_effort', payload)
    self.assertNotIn('verbosity', payload)

  @mock.patch.object(urllib.request, 'urlopen')
  def test_sample_choice_matches_exact_option(self, urlopen):
    urlopen.return_value = _FakeResponse({
        'choices': [{'message': {'content': 'refuse'}}],
    })

    result = self._model().sample_choice(
        'Choose an action.', ('help', 'refuse')
    )

    self.assertEqual(result[0], 1)
    self.assertEqual(result[1], 'refuse')

  def test_vectorengine_defaults_use_environment_key(self):
    with mock.patch.dict(
        os.environ, {'VECTORENGINE_API_KEY': 'env-key'}, clear=True
    ):
      model = openai_compatible.VectorEngineLanguageModel('gpt-4o-mini')
    self.assertEqual(model._api_base, 'https://api.vectorengine.ai/v1')

  def test_registry_exposes_vectorengine(self):
    self.assertEqual(
        language_models._REGISTRY['vectorengine'],
        'openai_compatible.VectorEngineLanguageModel',
    )

  def test_setup_forwards_api_base(self):
    model = language_models.language_model_setup(
        api_type='openai_compatible',
        model_name='test-model',
        api_key='test-key',
        api_base='https://example.test/custom/v1',
    )
    self.assertEqual(model._api_base, 'https://example.test/custom/v1')


if __name__ == '__main__':
  absltest.main()
