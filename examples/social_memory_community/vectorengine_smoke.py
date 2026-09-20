"""Call a VectorEngine OpenAI-compatible model once for a connectivity check."""

from __future__ import annotations

import argparse
import json
import os

from concordia.contrib import language_models

_DEFAULT_BASE_URL = 'https://api.vectorengine.ai/v1'
_DEFAULT_MODEL = 'gpt-4o-mini'


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model', default=os.getenv('VECTORENGINE_MODEL', _DEFAULT_MODEL)
  )
  parser.add_argument(
      '--base_url',
      default=os.getenv('VECTORENGINE_BASE_URL', _DEFAULT_BASE_URL),
  )
  parser.add_argument(
      '--prompt',
      default='Reply with exactly: SOCIAL_MEMORY_API_OK',
  )
  parser.add_argument('--max_tokens', type=int, default=32)
  parser.add_argument('--timeout', type=float, default=60.0)
  args = parser.parse_args()

  api_key = os.getenv('VECTORENGINE_API_KEY')
  if not api_key:
    parser.error('Set VECTORENGINE_API_KEY in the current process first.')

  model = language_models.language_model_setup(
      api_type='vectorengine',
      model_name=args.model,
      api_key=api_key,
      api_base=args.base_url,
  )
  response = model.sample_text(
      args.prompt,
      max_tokens=args.max_tokens,
      temperature=0.0,
      timeout=args.timeout,
  )
  print(
      json.dumps(
          {
              'api_type': 'vectorengine',
              'base_url': args.base_url.rstrip('/'),
              'model': args.model,
              'response': response,
          },
          ensure_ascii=False,
          indent=2,
      )
  )


if __name__ == '__main__':
  main()
