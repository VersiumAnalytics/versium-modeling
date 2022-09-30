"""response_handler_map.py: Maps response handler functions to url patterns."""
from .response_handlers import *
import urllib
import re

# Wildcard '*' can be used within uri to match any characters
RESPONSE_HANDLER_MAP = {
    'api*.versium.com/*': api_versium_com
}


def handler_from_url(url, default=default_response_handler):
    # If we don't have response handler, check the RESPONSE_HANDLER_MAP for matching url, fallback to default handler
    parsed_url = urllib.parse.urlparse(url)
    netloc_path = parsed_url.netloc + parsed_url.path

    for pattern, handler in RESPONSE_HANDLER_MAP.items():
        pattern = pattern.split('*')
        pattern = '.*'.join(map(re.escape, pattern))
        if re.match(pattern, netloc_path, re.IGNORECASE):
            return handler

    return default

