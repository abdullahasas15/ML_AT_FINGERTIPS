from django import template
import json
import re
from django.utils.safestring import mark_safe

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Get item from dictionary by key"""
    if dictionary is None:
        return 0
    return dictionary.get(key, 0)

@register.filter
def percentage(value, decimals=0):
    """Convert a fractional value (0-1) to a percentage string"""
    try:
        decimals = int(decimals)
    except (TypeError, ValueError):
        decimals = 0
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return '0%'
    format_string = "{:." + str(decimals) + "f}%"
    return format_string.format(numeric_value * 100)

@register.filter
def replace(value, args):
    """Replace occurrences of a string with another string
    Usage: {{ value|replace:"old,new" }}
    """
    if not args or ',' not in args:
        return value
    old, new = args.split(',', 1)
    return str(value).replace(old, new)

@register.filter
def parse_json(value):
    """Parse JSON string to Python object"""
    if not value:
        return {}
    try:
        if isinstance(value, str):
            return json.loads(value)
        return value
    except (json.JSONDecodeError, TypeError):
        return {}

@register.filter
def markdown_to_html(text):
    """Convert markdown-style text to HTML with proper formatting"""
    if not text:
        return ''
    
    html = str(text)
    
    # Convert headers
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
    
    # Convert bold text
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    
    # Convert italic text
    html = re.sub(r'_(.+?)_', r'<em>\1</em>', html)
    
    # Convert inline code
    html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)
    
    # Convert code blocks
    html = re.sub(
        r'```(\w+)?\n(.*?)```',
        lambda m: f'<pre><code class="language-{m.group(1) or "python"}">{m.group(2)}</code></pre>',
        html,
        flags=re.DOTALL
    )
    
    # Convert unordered lists
    lines = html.split('\n')
    in_list = False
    result = []
    
    for line in lines:
        # Check for list items
        if re.match(r'^[-*]\s+', line):
            if not in_list:
                result.append('<ul>')
                in_list = True
            item_text = re.sub(r'^[-*]\s+', '', line)
            result.append(f'<li>{item_text}</li>')
        else:
            if in_list:
                result.append('</ul>')
                in_list = False
            result.append(line)
    
    if in_list:
        result.append('</ul>')
    
    html = '\n'.join(result)
    
    # Convert paragraphs
    html = re.sub(r'\n\n+', '</p><p>', html)
    if html and not html.startswith('<h') and not html.startswith('<ul') and not html.startswith('<pre'):
        html = '<p>' + html + '</p>'
    
    # Clean up extra paragraph tags around headers
    html = re.sub(r'<p>(<h[1-6]>)', r'\1', html)
    html = re.sub(r'(</h[1-6]>)</p>', r'\1', html)
    html = re.sub(r'<p>(<ul>)', r'\1', html)
    html = re.sub(r'(</ul>)</p>', r'\1', html)
    html = re.sub(r'<p>(<pre>)', r'\1', html)
    html = re.sub(r'(</pre>)</p>', r'\1', html)
    
    return mark_safe(html)
