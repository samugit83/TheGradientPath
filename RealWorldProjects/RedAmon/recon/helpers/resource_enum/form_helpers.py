"""
RedAmon - Form Parsing Helpers
==============================
HTML form parsing and extraction utilities for resource enumeration.
"""

from html.parser import HTMLParser
from typing import Dict, List
from urllib.parse import urljoin


# =============================================================================
# HTML Form Parser Class
# =============================================================================

class FormParser(HTMLParser):
    """Parse HTML to extract form elements and their inputs."""

    def __init__(self):
        super().__init__()
        self.forms = []
        self.current_form = None
        self.in_form = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)

        if tag == 'form':
            self.in_form = True
            self.current_form = {
                'action': attrs_dict.get('action', ''),
                'method': attrs_dict.get('method', 'GET').upper(),
                'enctype': attrs_dict.get('enctype', 'application/x-www-form-urlencoded'),
                'inputs': []
            }

        elif self.in_form and tag == 'input':
            input_info = {
                'name': attrs_dict.get('name', ''),
                'type': attrs_dict.get('type', 'text'),
                'value': attrs_dict.get('value', ''),
                'required': 'required' in attrs_dict,
                'placeholder': attrs_dict.get('placeholder', '')
            }
            if input_info['name']:  # Only add inputs with names
                self.current_form['inputs'].append(input_info)

        elif self.in_form and tag == 'textarea':
            input_info = {
                'name': attrs_dict.get('name', ''),
                'type': 'textarea',
                'value': '',
                'required': 'required' in attrs_dict
            }
            if input_info['name']:
                self.current_form['inputs'].append(input_info)

        elif self.in_form and tag == 'select':
            input_info = {
                'name': attrs_dict.get('name', ''),
                'type': 'select',
                'value': '',
                'required': 'required' in attrs_dict
            }
            if input_info['name']:
                self.current_form['inputs'].append(input_info)

        elif self.in_form and tag == 'button':
            btn_type = attrs_dict.get('type', 'submit')
            if btn_type == 'submit' and attrs_dict.get('name'):
                input_info = {
                    'name': attrs_dict.get('name', ''),
                    'type': 'submit',
                    'value': attrs_dict.get('value', '')
                }
                self.current_form['inputs'].append(input_info)

    def handle_endtag(self, tag):
        if tag == 'form' and self.in_form:
            self.in_form = False
            if self.current_form:
                self.forms.append(self.current_form)
            self.current_form = None


# =============================================================================
# Form Parsing Functions
# =============================================================================

def parse_forms_from_html(html_content: str, base_url: str) -> List[Dict]:
    """
    Parse HTML content to extract form information.

    Args:
        html_content: Raw HTML string
        base_url: Base URL for resolving relative form actions

    Returns:
        List of form dictionaries with action, method, and inputs
    """
    if not html_content:
        return []

    try:
        parser = FormParser()
        parser.feed(html_content)

        forms = []
        for form in parser.forms:
            # Resolve relative action URLs
            action = form['action']
            if action:
                if not action.startswith(('http://', 'https://')):
                    action = urljoin(base_url, action)
            else:
                action = base_url  # Form submits to current URL

            form['action'] = action
            form['found_at'] = base_url
            forms.append(form)

        return forms
    except Exception:
        return []
