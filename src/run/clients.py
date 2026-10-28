"""Provider clients for the four evaluated models, with the paper's configuration.

Each ``call_<provider>(request, setting)`` sends one output of
``src.prompts.<setting>.build_request`` (``{"system": str, "parts": [{"text": str} |
{"image": <local path>}]}``) as one independent conversation and returns a reply dict::

    {"text": str, "finish_reason": str, "complete": bool, "reasoning": bool,
     "returned_model": str, "response": <provider JSON>}

``complete`` is True only for a finished, non-truncated, non-refused answer. Decisions are
read only from replies that pass ``usable``: complete, without reasoning output, and produced
by the requested model.

Configuration (paper Appendix D.1):

| model    | identifier         | API                                                        |
|----------|--------------------|------------------------------------------------------------|
| Qwen     | `qwen3.5-flash`    | DashScope: OpenAI-compatible chat (text settings); native  |
|          |                    | multimodal generation with temporary uploads of the        |
|          |                    | original image files (retail marketplace)                  |
| DeepSeek | `deepseek-flash`   | DeepSeek chat completions                                  |
| Luna     | `gpt-5.6-luna`     | OpenAI chat completions (text); Responses API with         |
|          |                    | uploaded original images (retail marketplace)              |
| Haiku 5.5| `claude-haiku-5-5` | Anthropic Messages (images as uploaded files)              |

- Temperature 0 for Qwen, DeepSeek and Luna; thinking disabled for Qwen and DeepSeek; Luna
  ``reasoning_effort="none"``. Haiku 5.5 sends no temperature/top_p/top_k, disables thinking
  and uses medium effort.
- Maximum output tokens: 3,072 (Qwen, DeepSeek, Luna) and 4,096 (Haiku 5.5); housing 768 and
  1,024.
- Qwen, DeepSeek and Luna use the provider's JSON-object response format; Haiku 5.5 relies on
  the JSON instructions in the prompt (no provider JSON mode).
- Images are sent at original resolution (``detail: "original"`` where the API has the option).

API keys are read from DASHSCOPE_API_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY and
ANTHROPIC_API_KEY. Transport errors, HTTP 429 and 5xx responses are retried with exponential
backoff; other HTTP errors are raised.
"""
import base64
import mimetypes
import os
import time
import uuid
from pathlib import Path

import requests

MODELS = {'qwen': 'qwen3.5-flash', 'deepseek': 'deepseek-flash',
          'luna': 'gpt-5.6-luna', 'haiku': 'claude-haiku-5-5'}
MODEL_LABELS = {'qwen': 'Qwen', 'deepseek': 'DeepSeek', 'luna': 'Luna', 'haiku': 'Haiku 5.5'}
KEY_ENV = {'qwen': 'DASHSCOPE_API_KEY', 'deepseek': 'DEEPSEEK_API_KEY',
           'luna': 'OPENAI_API_KEY', 'haiku': 'ANTHROPIC_API_KEY'}
IMAGE_SETTINGS = ('marketplace',)          # settings whose requests contain images

QWEN_CHAT_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
QWEN_NATIVE_URL = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation'
QWEN_UPLOAD_POLICY_URL = 'https://dashscope.aliyuncs.com/api/v1/uploads'
DEEPSEEK_URL = 'https://api.deepseek.com/chat/completions'
DEEPSEEK_FILES_URL = 'https://api.deepseek.com/files'
OPENAI_CHAT_URL = 'https://api.openai.com/v1/chat/completions'
OPENAI_RESPONSES_URL = 'https://api.openai.com/v1/responses'
OPENAI_FILES_URL = 'https://api.openai.com/v1/files'
ANTHROPIC_MESSAGES_URL = 'https://api.anthropic.com/v1/messages'
ANTHROPIC_FILES_URL = 'https://api.anthropic.com/v1/files'
ANTHROPIC_VERSION = '2023-06-01'
ANTHROPIC_FILES_BETA = 'files-api-2025-04-14'
DEEPSEEK_INLINE_LIMIT = 40_000_000   # base64 bytes per request above which images go as files

TIMEOUT = 900
MAX_TRANSPORT_RETRIES = 5
JSON_OBJECT = {'type': 'json_object'}


def max_tokens(model, setting):
    """Maximum output tokens for a model in a setting."""
    if setting == 'housing':
        return 1024 if model == 'haiku' else 768
    return 4096 if model == 'haiku' else 3072


def api_key(model):
    key = os.environ.get(KEY_ENV[model], '').strip()
    if not key:
        raise RuntimeError(f'Set {KEY_ENV[model]} in the environment')
    return key


class APIError(RuntimeError):
    def __init__(self, status, body):
        super().__init__(f'HTTP {status}: {str(body)[:500]}')
        self.status, self.body = status, body


def _send(method, url, headers, **kwargs):
    """HTTP request with retries for transport errors, 429 and 5xx; returns parsed JSON."""
    for attempt in range(MAX_TRANSPORT_RETRIES + 1):
        try:
            response = requests.request(method, url, headers=headers, timeout=TIMEOUT, **kwargs)
        except requests.RequestException:
            if attempt == MAX_TRANSPORT_RETRIES:
                raise
        else:
            if response.status_code < 300:
                return response.json() if response.content else {}
            if response.status_code != 429 and response.status_code < 500 or attempt == MAX_TRANSPORT_RETRIES:
                try:
                    body = response.json()
                except ValueError:
                    body = response.text
                raise APIError(response.status_code, body)
        time.sleep(min(60, 2 ** (attempt + 1)))
    raise AssertionError('unreachable')


def _bearer(key):
    return {'Authorization': 'Bearer ' + key}


def _images(request):
    return [part['image'] for part in request['parts'] if 'image' in part]


def _single_text(request):
    """The user text if the request is exactly one text part, else None."""
    parts = request['parts']
    return parts[0]['text'] if len(parts) == 1 and 'text' in parts[0] else None


def _media_type(path):
    return mimetypes.guess_type(str(path))[0] or 'image/png'


def _data_url(path):
    return f'data:{_media_type(path)};base64,' + base64.b64encode(Path(path).read_bytes()).decode('ascii')


def _placeholder(model, path):
    """Image reference used only to display request bodies without uploading (dry run)."""
    return {'qwen': 'oss://<temporary upload>', 'deepseek': 'data:image/png;base64,<original bytes>',
            'luna': 'file-<uploaded original>', 'haiku': 'file_<uploaded original>'}[model]


# ----------------------------------------------------------------------------- request bodies
def qwen_chat_body(request, setting):
    """OpenAI-compatible DashScope chat body (text settings)."""
    if _images(request):
        raise ValueError('Image requests use the native multimodal body')
    user = _single_text(request)
    if user is None:
        user = [{'type': 'text', 'text': p['text']} for p in request['parts']]
    return {'model': MODELS['qwen'], 'temperature': 0, 'enable_thinking': False,
            'max_tokens': max_tokens('qwen', setting), 'response_format': JSON_OBJECT,
            'messages': [{'role': 'system', 'content': request['system']},
                         {'role': 'user', 'content': user}]}


def qwen_native_body(request, setting, image_ref):
    """Native DashScope multimodal-generation body; image_ref(path) -> oss:// URL."""
    content = [{'text': p['text']} if 'text' in p else {'image': image_ref(p['image'])}
               for p in request['parts']]
    return {'model': MODELS['qwen'],
            'input': {'messages': [{'role': 'system', 'content': [{'text': request['system']}]},
                                   {'role': 'user', 'content': content}]},
            'parameters': {'temperature': 0, 'enable_thinking': False,
                           'max_tokens': max_tokens('qwen', setting), 'response_format': JSON_OBJECT}}


def deepseek_body(request, setting, image_ref=None):
    """DeepSeek chat body; image_ref(path) -> {"url": data URL} or {"file_id": id}."""
    user = _single_text(request)
    if user is None:
        user = []
        for p in request['parts']:
            if 'text' in p:
                user.append({'type': 'text', 'text': p['text']})
                continue
            ref = image_ref(p['image'])
            user.append({'type': 'file', 'file_id': ref['file_id']} if 'file_id' in ref else
                        {'type': 'image_url', 'image_url': {'url': ref['url'], 'detail': 'original'}})
    return {'model': MODELS['deepseek'], 'temperature': 0,
            'messages': [{'role': 'system', 'content': request['system']},
                         {'role': 'user', 'content': user}],
            'response_format': JSON_OBJECT, 'max_tokens': max_tokens('deepseek', setting),
            'thinking': {'type': 'disabled'}}


def luna_chat_body(request, setting):
    """OpenAI chat-completions body (text settings)."""
    if _images(request):
        raise ValueError('Image requests use the Responses body')
    user = _single_text(request)
    if user is None:
        user = [{'type': 'text', 'text': p['text']} for p in request['parts']]
    return {'model': MODELS['luna'], 'temperature': 0, 'reasoning_effort': 'none',
            'max_completion_tokens': max_tokens('luna', setting), 'response_format': JSON_OBJECT,
            'store': False, 'messages': [{'role': 'system', 'content': request['system']},
                                         {'role': 'user', 'content': user}]}


def luna_responses_body(request, setting, image_ref):
    """OpenAI Responses body with uploaded original images; image_ref(path) -> file id."""
    content = [{'type': 'input_text', 'text': p['text']} if 'text' in p else
               {'type': 'input_image', 'file_id': image_ref(p['image']), 'detail': 'original'}
               for p in request['parts']]
    return {'model': MODELS['luna'],
            'input': [{'role': 'system', 'content': [{'type': 'input_text', 'text': request['system']}]},
                      {'role': 'user', 'content': content}],
            'temperature': 0, 'max_output_tokens': max_tokens('luna', setting),
            'reasoning': {'effort': 'none'}, 'text': {'format': JSON_OBJECT}, 'store': False}


def haiku_body(request, setting, image_ref=None):
    """Anthropic Messages body; image_ref(path) -> Files API file id. No sampling parameters."""
    user = _single_text(request)
    if user is None:
        user = [{'type': 'text', 'text': p['text']} if 'text' in p else
                {'type': 'image', 'source': {'type': 'file', 'file_id': image_ref(p['image'])}}
                for p in request['parts']]
    return {'model': MODELS['haiku'], 'system': request['system'],
            'messages': [{'role': 'user', 'content': user}],
            'max_tokens': max_tokens('haiku', setting), 'thinking': {'type': 'disabled'},
            'output_config': {'effort': 'medium'}}


def request_body(model, request, setting, image_ref=None):
    """The body sent for ``request``; images use ``image_ref`` or a display placeholder."""
    images = bool(_images(request))
    ref = image_ref or (lambda path: _placeholder(model, path))
    if model == 'qwen':
        return qwen_native_body(request, setting, ref) if images else qwen_chat_body(request, setting)
    if model == 'deepseek':
        return deepseek_body(request, setting, image_ref or (lambda path: {'url': _placeholder(model, path)}))
    if model == 'luna':
        return luna_responses_body(request, setting, ref) if images else luna_chat_body(request, setting)
    if model == 'haiku':
        return haiku_body(request, setting, ref)
    raise ValueError(f'Unknown model: {model}')


# ----------------------------------------------------------------------------- replies
def _chat_reply(body):
    choice = body['choices'][0]
    message = choice.get('message') or {}
    finish = choice.get('finish_reason')
    return {'text': message.get('content') or '', 'finish_reason': finish,
            'complete': finish == 'stop' and not message.get('refusal'),
            'reasoning': bool(message.get('reasoning_content') or message.get('reasoning')),
            'returned_model': body.get('model'), 'response': body}


def model_matches(model, returned):
    """True if the reply was produced by the requested model."""
    requested = MODELS[model]
    return isinstance(returned, str) and (returned == requested or returned.startswith(requested + '-'))


def usable(model, reply):
    """(True, '') if a decision may be read from ``reply``; otherwise (False, reason).

    A reply is usable only if it is final and complete (not truncated by the output limit,
    not refused), contains no reasoning output (thinking is disabled), and was produced by
    the requested model.
    """
    if not reply.get('complete'):
        return False, f"incomplete ({reply.get('finish_reason')})"
    if reply.get('reasoning'):
        return False, 'reasoning output present'
    if not model_matches(model, reply.get('returned_model')):
        return False, f"returned model {reply.get('returned_model')!r}"
    return True, ''


# ----------------------------------------------------------------------------- Qwen
def _qwen_upload(path, key, policy):
    """Temporary DashScope upload of one original image file; returns its oss:// URL."""
    path = Path(path)
    object_key = f"{policy['upload_dir']}/{uuid.uuid4().hex}{path.suffix or '.png'}"
    with path.open('rb') as stream:
        files = {'OSSAccessKeyId': (None, policy['oss_access_key_id']),
                 'Signature': (None, policy['signature']),
                 'policy': (None, policy['policy']),
                 'x-oss-object-acl': (None, policy['x_oss_object_acl']),
                 'x-oss-forbid-overwrite': (None, policy['x_oss_forbid_overwrite']),
                 'key': (None, object_key),
                 'success_action_status': (None, '200'),
                 'file': (path.name, stream, _media_type(path))}
        response = requests.post(policy['upload_host'], files=files, timeout=TIMEOUT)
    if response.status_code != 200:
        raise APIError(response.status_code, response.text)
    return 'oss://' + object_key


def call_qwen(request, setting):
    """Qwen: OpenAI-compatible chat for text; native multimodal API with temporary uploads
    of the original image files for requests that contain images."""
    key = api_key('qwen')
    if not _images(request):
        return _chat_reply(_send('POST', QWEN_CHAT_URL, _bearer(key), json=qwen_chat_body(request, setting)))
    urls, policy = {}, {'value': None, 'refresh_at': 0.}

    def reference(path):
        if path not in urls:
            if time.monotonic() >= policy['refresh_at']:   # upload policies expire after minutes
                value = _send('GET', QWEN_UPLOAD_POLICY_URL, _bearer(key),
                              params={'action': 'getPolicy', 'model': MODELS['qwen']})['data']
                lifetime = float(value.get('expire_in_seconds') or 300)
                policy.update(value=value, refresh_at=time.monotonic() + max(1., lifetime - 60))
            urls[path] = _qwen_upload(path, key, policy['value'])
        return urls[path]

    body = qwen_native_body(request, setting, reference)
    headers = {**_bearer(key), 'X-DashScope-OssResourceResolve': 'enable'}
    reply = _send('POST', QWEN_NATIVE_URL, headers, json=body)
    choice = reply['output']['choices'][0]
    content = choice['message'].get('content') or ''
    if isinstance(content, list):
        content = ''.join(item.get('text', '') for item in content)
    finish = choice.get('finish_reason')
    return {'text': content, 'finish_reason': finish, 'complete': finish == 'stop',
            'reasoning': bool(choice['message'].get('reasoning_content')),
            'returned_model': reply.get('model') or MODELS['qwen'], 'response': reply}


# ----------------------------------------------------------------------------- DeepSeek
def call_deepseek(request, setting):
    key = api_key('deepseek')
    images = _images(request)
    as_files = sum(Path(p).stat().st_size for p in images) * 4 / 3 > DEEPSEEK_INLINE_LIMIT
    uploaded = []

    def reference(path):
        if not as_files:
            return {'url': _data_url(path)}
        path = Path(path)
        with path.open('rb') as stream:
            result = _send('POST', DEEPSEEK_FILES_URL, _bearer(key), data={'purpose': 'user_data'},
                           files={'file': (path.name, stream, _media_type(path))})
        uploaded.append(result['id'])
        return {'file_id': result['id']}

    try:
        body = deepseek_body(request, setting, reference)
        return _chat_reply(_send('POST', DEEPSEEK_URL, _bearer(key), json=body))
    finally:
        _delete(DEEPSEEK_FILES_URL, _bearer(key), uploaded)


# ----------------------------------------------------------------------------- Luna
def call_luna(request, setting):
    key = api_key('luna')
    if not _images(request):
        return _chat_reply(_send('POST', OPENAI_CHAT_URL, _bearer(key), json=luna_chat_body(request, setting)))
    uploaded = []

    def reference(path):
        path = Path(path)
        with path.open('rb') as stream:
            result = _send('POST', OPENAI_FILES_URL, _bearer(key), data={'purpose': 'vision'},
                           files={'file': (path.name, stream, _media_type(path))})
        uploaded.append(result['id'])
        return result['id']

    try:
        reply = _send('POST', OPENAI_RESPONSES_URL, _bearer(key),
                      json=luna_responses_body(request, setting, reference))
    finally:
        _delete(OPENAI_FILES_URL, _bearer(key), uploaded)
    text = ''.join(c.get('text', '') for item in reply.get('output', []) if item.get('type') == 'message'
                   for c in item.get('content', []) if c.get('type') == 'output_text')
    refused = any(c.get('type') == 'refusal' for item in reply.get('output', []) if item.get('type') == 'message'
                  for c in item.get('content', []))
    reasoning = any(item.get('type') == 'reasoning' and (item.get('summary') or item.get('content'))
                    for item in reply.get('output', []))
    status = reply.get('status')
    finish = status if status != 'incomplete' else 'incomplete:' + str((reply.get('incomplete_details') or {}).get('reason'))
    return {'text': text, 'finish_reason': finish, 'complete': status == 'completed' and not refused,
            'reasoning': reasoning, 'returned_model': reply.get('model'), 'response': reply}


# ----------------------------------------------------------------------------- Haiku 5.5
def call_haiku(request, setting):
    key = api_key('haiku')
    headers = {'x-api-key': key, 'anthropic-version': ANTHROPIC_VERSION}
    uploaded = []
    if _images(request):
        headers['anthropic-beta'] = ANTHROPIC_FILES_BETA

    def reference(path):
        path = Path(path)
        with path.open('rb') as stream:
            result = _send('POST', ANTHROPIC_FILES_URL, headers,
                           files={'file': (path.name, stream, _media_type(path))})
        uploaded.append(result['id'])
        return result['id']

    try:
        reply = _send('POST', ANTHROPIC_MESSAGES_URL, headers, json=haiku_body(request, setting, reference))
    finally:
        _delete(ANTHROPIC_FILES_URL, headers, uploaded)
    text = ''.join(b.get('text', '') for b in reply.get('content', []) if b.get('type') == 'text')
    stop = reply.get('stop_reason')
    return {'text': text, 'finish_reason': stop, 'complete': stop == 'end_turn',
            'reasoning': any(b.get('type') in ('thinking', 'redacted_thinking') for b in reply.get('content', [])),
            'returned_model': reply.get('model'), 'response': reply}


def _delete(files_url, headers, file_ids):
    """Best-effort removal of images uploaded for one request."""
    for file_id in file_ids:
        try:
            requests.delete(f'{files_url}/{file_id}', headers=headers, timeout=60)
        except requests.RequestException:
            pass


CALLS = {'qwen': call_qwen, 'deepseek': call_deepseek, 'luna': call_luna, 'haiku': call_haiku}


def call(model, request, setting):
    """Send one request to ``model`` ('qwen', 'deepseek', 'luna' or 'haiku')."""
    return CALLS[model](request, setting)
