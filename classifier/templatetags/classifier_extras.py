from django import template
import json

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
