import io
import json
import pytest
from types import SimpleNamespace
from your_module import count_characters, handle_count_api, get_index_html, handle_static, app

def make_environ(path='/', method='GET', body=None, content_type='application/json'):
    environ = {
        'REQUEST_METHOD': method,
        'PATH_INFO': path,
        'CONTENT_TYPE': content_type,
        'CONTENT_LENGTH': str(len(body.encode('utf-8')) if body else 0),
        'wsgi.input': io.BytesIO(body.encode('utf-8') if body else b''),
        'SERVER_NAME': 'localhost',
        'SERVER_PORT': '8080',
        'wsgi.url_scheme': 'http',
    }
    return environ

# ---- count_characters ----
def test_count_characters_basic():
    assert count_characters('abc') == 3
    assert count_characters('') == 0
    assert count_characters('hello world!') == 12

# ---- handle_count_api ----
def test_handle_count_api_ok():
    data = json.dumps({'text': 'openai'})
    environ = make_environ(path='/api/count', method='POST', body=data)
    status, headers, response = handle_count_api(environ)
    assert status == '200 OK'
    assert ('Content-Type', 'application/json') in headers
    body = json.loads(response)
    assert body['count'] == 6

def test_handle_count_api_missing_text():
    data = '{}'  # Missing 'text', should default to ''
    environ = make_environ(path='/api/count', method='POST', body=data)
    status, headers, response = handle_count_api(environ)
    assert status == '200 OK'
    body = json.loads(response)
    assert body['count'] == 0

def test_handle_count_api_text_not_str():
    data = json.dumps({'text': 123})
    environ = make_environ(path='/api/count', method='POST', body=data)
    status, headers, response = handle_count_api(environ)
    assert status == '400 Bad Request'
    body = json.loads(response)
    assert 'error' in body
    assert 'must be a string' in body['error']

def test_handle_count_api_bad_json():
    data = 'not a json'
    environ = make_environ(path='/api/count', method='POST', body=data)
    status, headers, response = handle_count_api(environ)
    assert status == '400 Bad Request'
    body = json.loads(response)
    assert 'error' in body

def test_handle_count_api_no_body():
    environ = make_environ(path='/api/count', method='POST', body=None)
    status, headers, response = handle_count_api(environ)
    assert status == '200 OK'
    body = json.loads(response)
    assert body['count'] == 0

def test_handle_count_api_invalid_content_length():
    environ = make_environ(path='/api/count', method='POST', body=json.dumps({'text': 'abc'}))
    environ['CONTENT_LENGTH'] = 'not_a_number'
    status, headers, response = handle_count_api(environ)
    assert status == '200 OK'
    body = json.loads(response)
    assert body['count'] == 0

# ---- get_index_html ----
def test_get_index_html_basic():
    html = get_index_html()
    assert '<!DOCTYPE html>' in html
    assert 'Character Counter' in html
    assert 'ReactDOM.render' in html

def test_handle_static_returns_html():
    environ = make_environ()
    status, headers, body = handle_static(environ)
    assert status == '200 OK'
    assert ('Content-Type', 'text/html; charset=utf-8') in headers
    assert body.startswith(b'<!DOCTYPE html>')

# ---- app (WSGI) ----
def start_response_collector():
    info = {}
    def start_response(status, headers):
        info['status'] = status
        info['headers'] = headers
    return info, start_response

def test_app_get_index_html():
    environ = make_environ(path='/', method='GET')
    info, start_response = start_response_collector()
    result = app(environ, start_response)
    body = b''.join(result)
    assert info['status'] == '200 OK'
    assert b'<html' in body


def test_app_api_ok():
    body_json = json.dumps({'text': 'abcde'})
    environ = make_environ(path='/api/count', method='POST', body=body_json)
    info, start_response = start_response_collector()
    result = app(environ, start_response)
    response = b''.join(result)
    parsed = json.loads(response)
    assert info['status'] == '200 OK'
    assert parsed['count'] == 5


def test_app_api_invalid_method():
    environ = make_environ(path='/api/count', method='GET')
    info, start_response = start_response_collector()
    result = app(environ, start_response)
    response = b''.join(result)
    parsed = json.loads(response)
    assert info['status'] == '404 Not Found'
    assert 'error' in parsed


def test_app_unknown_path():
    environ = make_environ(path='/unknown', method='GET')
    info, start_response = start_response_collector()
    result = app(environ, start_response)
    response = b''.join(result)
    parsed = json.loads(response)
    assert info['status'] == '404 Not Found'
    assert 'error' in parsed

# Optional: test catastrophic error route

def test_app_internal_error(monkeypatch):
    def broken_count_api(environ):
        raise RuntimeError('fail')
    monkeypatch.setattr('your_module.handle_count_api', broken_count_api)
    environ = make_environ(path='/api/count', method='POST', body='{"text":"x"}')
    info, start_response = start_response_collector()
    result = app(environ, start_response)
    response = b''.join(result)
    parsed = json.loads(response)
    assert info['status'] == '500 Internal Server Error'
    assert 'error' in parsed
