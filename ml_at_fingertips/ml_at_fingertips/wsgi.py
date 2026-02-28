"""
WSGI config for ml_at_fingertips project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/wsgi/
"""

import os
import sys
from pathlib import Path

# Add the repository root to sys.path so sibling apps (classifier/, accounts/)
# are importable when gunicorn loads this file directly (manage.py does this
# automatically but wsgi.py is loaded by gunicorn without that setup).
repo_root = str(Path(__file__).resolve().parent.parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ml_at_fingertips.settings')

application = get_wsgi_application()
