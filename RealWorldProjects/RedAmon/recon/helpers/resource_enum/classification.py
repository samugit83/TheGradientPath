"""
RedAmon - Classification Helpers
================================
Parameter and endpoint classification utilities for resource enumeration.
"""

import re
from typing import Dict, List


# =============================================================================
# Parameter Name Patterns for Classification
# =============================================================================

PARAM_PATTERNS = {
    'id_params': [
        r'^id$', r'_id$', r'Id$', r'^uid$', r'^pid$', r'^aid$', r'^cid$',
        r'^user_?id$', r'^product_?id$', r'^item_?id$', r'^post_?id$',
        r'^article_?id$', r'^page_?id$', r'^cat_?id$', r'^category_?id$',
        r'^artist$', r'^cat$', r'^pic$', r'^num$', r'^no$', r'^index$'
    ],
    'file_params': [
        r'^file$', r'^filename$', r'^path$', r'^filepath$', r'^download$',
        r'^include$', r'^require$', r'^read$', r'^load$', r'^src$',
        r'^template$', r'^page$', r'^doc$', r'^document$', r'^img$',
        r'^image$', r'^attachment$'
    ],
    'search_params': [
        r'^q$', r'^query$', r'^search$', r'^s$', r'^keyword$', r'^term$',
        r'^find$', r'^filter$', r'^text$', r'^input$'
    ],
    'auth_params': [
        r'^user$', r'^username$', r'^login$', r'^email$', r'^mail$',
        r'^password$', r'^passwd$', r'^pass$', r'^pwd$', r'^token$',
        r'^auth$', r'^key$', r'^apikey$', r'^api_key$', r'^secret$',
        r'^session$', r'^cookie$'
    ],
    'redirect_params': [
        r'^url$', r'^redirect$', r'^return$', r'^next$', r'^goto$',
        r'^target$', r'^dest$', r'^destination$', r'^continue$', r'^ref$',
        r'^callback$', r'^returnurl$', r'^return_url$'
    ],
    'command_params': [
        r'^cmd$', r'^command$', r'^exec$', r'^execute$', r'^run$',
        r'^shell$', r'^system$', r'^ping$', r'^host$', r'^ip$'
    ]
}


# =============================================================================
# Parameter Classification
# =============================================================================

def classify_parameter(param_name: str) -> str:
    """
    Classify a parameter name into a category.
    
    Categories: id_params, file_params, search_params, auth_params, 
                redirect_params, command_params, other
    """
    param_lower = param_name.lower()

    for category, patterns in PARAM_PATTERNS.items():
        for pattern in patterns:
            if re.match(pattern, param_lower, re.IGNORECASE):
                return category

    return 'other'


def infer_parameter_type(param_name: str, sample_values: List[str]) -> str:
    """
    Infer the data type of a parameter from its name and sample values.
    
    Returns: integer, path, email, url, datetime, boolean, or string
    """
    param_lower = param_name.lower()

    # Check sample values first
    if sample_values:
        # Check if all values are numeric
        all_numeric = all(
            v.isdigit() or (v.startswith('-') and v[1:].isdigit())
            for v in sample_values if v
        )
        if all_numeric:
            return 'integer'

        # Check if values look like file paths
        if any('/' in v or '\\' in v or '.' in v for v in sample_values if v):
            if any(v.endswith(('.jpg', '.png', '.gif', '.pdf', '.txt', '.html', '.php', '.js'))
                   for v in sample_values if v):
                return 'path'

        # Check if values look like emails
        if any('@' in v and '.' in v for v in sample_values if v):
            return 'email'

        # Check if values look like URLs
        if any(v.startswith(('http://', 'https://')) for v in sample_values if v):
            return 'url'

    # Infer from parameter name
    if any(p in param_lower for p in ['id', 'num', 'count', 'page', 'limit', 'offset', 'size']):
        return 'integer'
    if any(p in param_lower for p in ['file', 'path', 'dir', 'template', 'include']):
        return 'path'
    if any(p in param_lower for p in ['email', 'mail']):
        return 'email'
    if any(p in param_lower for p in ['url', 'link', 'redirect', 'callback']):
        return 'url'
    if any(p in param_lower for p in ['date', 'time', 'timestamp']):
        return 'datetime'
    if any(p in param_lower for p in ['bool', 'flag', 'enabled', 'active', 'is_']):
        return 'boolean'

    return 'string'


# =============================================================================
# Endpoint Classification
# =============================================================================

def classify_endpoint(path: str, methods: List[str], params: Dict) -> str:
    """
    Classify an endpoint into a category based on path, methods, and parameters.

    Categories:
    - authentication: login, signup, logout, auth-related
    - file_access: file download, image serving, document access
    - api: REST API endpoints
    - admin: admin panels, dashboards
    - dynamic: PHP/ASP/JSP pages with parameters
    - static: HTML, CSS, JS, images
    - upload: file upload endpoints
    - search: search functionality
    - other: default category
    """
    path_lower = path.lower()

    # Check for authentication endpoints
    auth_patterns = ['/login', '/signin', '/signup', '/register', '/logout', '/signout',
                     '/auth', '/oauth', '/password', '/reset', '/forgot', '/session',
                     '/token', '/jwt', '/sso']
    if any(p in path_lower for p in auth_patterns):
        return 'authentication'

    # Check for admin endpoints
    admin_patterns = ['/admin', '/dashboard', '/panel', '/manage', '/control',
                      '/backend', '/cms', '/wp-admin', '/administrator']
    if any(p in path_lower for p in admin_patterns):
        return 'admin'

    # Check for API endpoints
    api_patterns = ['/api/', '/v1/', '/v2/', '/v3/', '/rest/', '/graphql',
                    '/json', '/xml', '/rpc']
    if any(p in path_lower for p in api_patterns):
        return 'api'

    # Check for file access endpoints
    file_patterns = ['/download', '/file', '/image', '/img', '/media', '/upload',
                     '/attachment', '/document', '/doc', '/pdf', '/export']
    if any(p in path_lower for p in file_patterns):
        # Check if it's upload vs download
        if any(p in path_lower for p in ['/upload', '/import']):
            return 'upload'
        return 'file_access'

    # Check for search endpoints
    search_patterns = ['/search', '/find', '/query', '/filter', '/browse']
    if any(p in path_lower for p in search_patterns):
        return 'search'

    # Check body params for auth indicators
    body_params = params.get('body', [])
    body_param_names = [p.get('name', '').lower() for p in body_params]
    if any(p in body_param_names for p in ['username', 'password', 'email', 'login']):
        return 'authentication'

    # Check for static files
    static_extensions = ['.html', '.htm', '.css', '.js', '.txt', '.xml', '.json',
                        '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico', '.webp',
                        '.woff', '.woff2', '.ttf', '.eot', '.pdf', '.zip']
    if any(path_lower.endswith(ext) for ext in static_extensions):
        return 'static'

    # Check for dynamic pages (with query params)
    dynamic_extensions = ['.php', '.asp', '.aspx', '.jsp', '.cgi', '.pl']
    if any(path_lower.endswith(ext) for ext in dynamic_extensions):
        return 'dynamic'

    # If has query params, likely dynamic
    if params.get('query'):
        return 'dynamic'

    # Default
    return 'other'

