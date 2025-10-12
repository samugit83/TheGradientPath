from typing import Any, Dict, Optional
import logging
from wsgiref.simple_server import make_server, WSGIServer
from wsgiref.util import setup_testing_defaults
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def count_characters(text: str) -> int:
    """
    Counts the number of characters in the given text.

    Args:
        text (str): The input text to count characters from.

    Returns:
        int: The number of characters in the input text.

    Example:
        >>> count_characters('hello')
        5
    """
    return len(text)

def handle_count_api(environ: Dict[str, Any]) -> Tuple[str, list, bytes]:
    """
    Handles the /api/count POST endpoint to count characters in a text string.

    Args:
        environ (Dict[str, Any]): WSGI environment dictionary.

    Returns:
        Tuple[str, list, bytes]: (status, headers, response body)
    """
    try:
        try:
            content_length = int(environ.get('CONTENT_LENGTH', '0'))
        except (ValueError, TypeError):
            content_length = 0

        request_body = environ['wsgi.input'].read(content_length).decode('utf-8')
        logging.debug(f"Received API request body: {request_body}")
        data = json.loads(request_body) if request_body else {}
        text = data.get('text', '')
        if not isinstance(text, str):
            raise ValueError("Input 'text' must be a string.")
        char_count = count_characters(text)
        response = json.dumps({'count': char_count}).encode('utf-8')
        status = '200 OK'
        headers = [
            ('Content-Type', 'application/json'),
            ('Access-Control-Allow-Origin', '*')
        ]
        logging.info(f"Counted {char_count} chars for text length {len(text)}.")
        return status, headers, response
    except Exception as e:
        logging.error(f"Error in handle_count_api: {str(e)}")
        response = json.dumps({'error': str(e)}).encode('utf-8')
        status = '400 Bad Request'
        headers = [
            ('Content-Type', 'application/json'),
            ('Access-Control-Allow-Origin', '*')
        ]
        return status, headers, response

def get_index_html() -> str:
    """
    Returns the HTML content for the character counting app frontend.

    Returns:
        str: HTML source as string.
    """
    # Embedding a minimal React app via CDN
    return """
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Character Counter App</title>
  <style>
    body { font-family: Arial, sans-serif; background: #fafafa; margin: 2em; }
    #root { max-width: 500px; margin: auto; padding: 1.5em; background: #fff; border-radius: 8px; box-shadow: 0 2px 8px #ddd; }
    textarea { width: 100%; height: 120px; margin-bottom: 1em; font-size: 1em; padding: 8px; border-radius: 4px; border: 1px solid #bbb; }
    .count { font-size: 1.2em; margin-bottom: 1em; }
    .error { color: red; margin-top: 0.8em; }
    @media (max-width: 600px) { #root { width: 95%; } }
  </style>
</head>
<body>
  <div id=\"root\"></div>
  <script crossorigin src=\"https://unpkg.com/react@17/umd/react.development.js\"></script>
  <script crossorigin src=\"https://unpkg.com/react-dom@17/umd/react-dom.development.js\"></script>
  <script>
    const e = React.createElement;
    function CharCounterApp() {
      const [text, setText] = React.useState('');
      const [count, setCount] = React.useState(0);
      const [error, setError] = React.useState(null);
      React.useEffect(() => {
        const controller = new AbortController();
        async function fetchCount() {
          try {
            setError(null);
            const resp = await fetch('/api/count', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ text }),
              signal: controller.signal
            });
            const data = await resp.json();
            if (resp.ok) {
              setCount(data.count);
            } else {
              setError(data.error || 'Failed to compute character count.');
              setCount(0);
            }
          } catch (e) {
            if (e.name !== 'AbortError') {
              setError('Request failed: ' + e.message);
              setCount(0);
            }
          }
        }
        fetchCount();
        return () => controller.abort();
      }, [text]);
      return e('div', null,
        e('h2', null, 'Character Counter'),
        e('textarea', {
          value: text,
          onChange: e => setText(e.target.value),
          placeholder: 'Type your text here...'
        }),
        e('div', { className: 'count' },
          'Characters: ', e('b', null, count)
        ),
        error && e('div', { className: 'error' }, error)
      );
    }
    ReactDOM.render(React.createElement(CharCounterApp), document.getElementById('root'));
  </script>
</body>
</html>
"""

def handle_static(environ: Dict[str, Any]) -> Tuple[str, list, bytes]:
    """
    Handles serving the static index HTML for the app.

    Args:
        environ (Dict[str, Any]): WSGI environment.

    Returns:
        Tuple[str, list, bytes]: (status, headers, response body)
    """
    html = get_index_html()
    return (
        '200 OK',
        [('Content-Type', 'text/html; charset=utf-8')],
        html.encode('utf-8')
    )

def app(environ: Dict[str, Any], start_response: Any) -> Any:
    """
    The main WSGI application callable.

    Args:
        environ (Dict[str, Any]): WSGI environment.
        start_response (callable): WSGI start_response callback.

    Returns:
        Iterable of bytes (the response body).
    """
    setup_testing_defaults(environ)
    path = environ.get('PATH_INFO', '/')
    method = environ.get('REQUEST_METHOD', 'GET')
    try:
        if path == '/' and method == 'GET':
            status, headers, body = handle_static(environ)
        elif path == '/api/count' and method == 'POST':
            status, headers, body = handle_count_api(environ)
        else:
            status = '404 Not Found'
            headers = [('Content-Type', 'application/json')]
            body = json.dumps({'error': 'Endpoint not found.'}).encode('utf-8')
        start_response(status, headers)
        return [body]
    except Exception as e:
        logging.error(f"Unhandled error: {str(e)}")
        status = '500 Internal Server Error'
        headers = [('Content-Type', 'application/json')]
        body = json.dumps({'error': f'Internal server error: {str(e)}'}).encode('utf-8')
        start_response(status, headers)
        return [body]

def main(host: str = '127.0.0.1', port: int = 8080) -> int:
    """
    Starts the WSGI web server serving the React character counter app and its API.

    Args:
        host (str, optional): The host/interface to bind. Defaults to '127.0.0.1'.
        port (int, optional): The port to bind. Defaults to 8080.

    Returns:
        int: Status code (0 for success).

    Example:
        main('0.0.0.0', 8000)
    """
    try:
        httpd: WSGIServer = make_server(host, port, app)
        logging.info(f"Serving Character Counter app at http://{host}:{port}/ ...")
        httpd.serve_forever()
        return 0
    except KeyboardInterrupt:
        logging.info('Server shutdown by user.')
        return 0
    except Exception as exc:
        logging.error(f"Failed to start server: {repr(exc)}")
        return 1

if __name__ == "__main__":
    main()
