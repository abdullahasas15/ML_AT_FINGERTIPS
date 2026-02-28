#!/usr/bin/env bash
# build.sh - Render runs this script during every deployment build
# It installs dependencies, collects static files, and applies DB migrations.

set -o errexit  # Exit immediately if any command fails

# ── 1. Install Python dependencies ──────────────────────────────────────────
pip install -r requirements.txt

# ── 2. Collect static files into STATIC_ROOT (served by WhiteNoise) ─────────
cd ml_at_fingertips
python manage.py collectstatic --no-input

# ── 3. Apply database migrations to the Neon PostgreSQL database ─────────────
python manage.py migrate
