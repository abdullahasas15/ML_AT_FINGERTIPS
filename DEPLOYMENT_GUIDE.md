# Deployment Guide — ML at Fingertips on Render

This file explains **every single change made** and **why it was necessary** for production deployment.
Read this once and you'll understand the full deployment process.

---

## What Was Changed & Why

---

### 1. `requirements.txt` — Added Production Server & Static File Packages

**Added two packages:**

```
gunicorn>=21.2.0
whitenoise>=6.6.0
```

**Why gunicorn?**
Django's built-in dev server (`manage.py runserver`) is NOT safe for production. It handles only one request at a time and has no process management. **Gunicorn** is a production-grade Python WSGI HTTP server — it's what Render (and most platforms) expect you to use.

**Why whitenoise?**
In development, Django serves your CSS/JS/images directly. In production with `DEBUG=False`, Django refuses to serve static files — it expects a CDN or a dedicated web server (like Nginx). **WhiteNoise** is a Python package that lets Django serve static files safely in production, with gzip compression and long-lived caching headers — perfect for platforms like Render where you don't configure Nginx yourself.

---

### 2. `settings.py` — Four Production-Ready Changes

#### a) WhiteNoise Middleware (line ~71)

```python
'django.middleware.security.SecurityMiddleware',
'whitenoise.middleware.WhiteNoiseMiddleware',  # ← ADDED
'django.contrib.sessions.middleware.SessionMiddleware',
```

**Why?** WhiteNoise must be placed directly after `SecurityMiddleware` and before all others. It intercepts requests for static files (CSS, JS, images) so they never reach Django's view routing — this is fast and efficient.

#### b) STATIC_ROOT and STORAGES (line ~163)

```python
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Django 5.1+ uses STORAGES dict instead of old STATICFILES_STORAGE
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage"
    },
}
```

**Why STATIC_ROOT?** Django's `collectstatic` command gathers all static files from all apps and copies them into a single folder (`staticfiles/`). WhiteNoise then serves everything from that folder. Without `STATIC_ROOT`, `collectstatic` has nowhere to write.

**Why STORAGES with WhiteNoise backend?** Django 5.1+ replaced the old `STATICFILES_STORAGE` setting with the new `STORAGES` dict. The WhiteNoise backend:
  - Compresses static files (gzip + brotli)
  - Adds content-hash fingerprints to filenames (`style.abc123.css`) so browsers cache aggressively

#### c) PERPLEXITY_API_KEY from environment (line ~177)

```python
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY', '')
```

**Why?** Your Perplexity API key was hardcoded in 3 places inside `views.py`. Hardcoding secrets in code is a **critical security risk** — anyone who sees your GitHub repo can steal your API key. Now the key is read from an environment variable, keeping it out of your code entirely.

---

### 3. `views.py` — Removed Hardcoded API Key (3 places)

**Before (dangerous ❌):**
```python
api_key = "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"
```

**After (safe ✓):**
```python
api_key = settings.PERPLEXITY_API_KEY
```

This was changed in all 3 functions:
- `get_prediction_explanation()`
- `get_feature_impact_explanation()`
- `get_perplexity_recommendations()`

---

### 4. `.env` — Added All Required Environment Variables

```env
DATABASE_URL=postgresql://...neon.tech/neondb?sslmode=require
DJANGO_SECRET_KEY=hafs)1c$w#!ry#(3fm$l@uk_9r-ko%r6-...
DJANGO_DEBUG=False
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1
PERPLEXITY_API_KEY=pplx-dXYm8tUaRloHLhwzlfjs3R1xUX...
```

**Why DJANGO_SECRET_KEY?** A new, strong secret key was generated. The old key in `settings.py` was the insecure default (`django-insecure-...`) — never use that in production.

**Why DJANGO_DEBUG=False?** With `DEBUG=True`, Django shows full error tracebacks to users (including your code, file paths, settings). In production this is a massive security leak. With `False`, users see a generic error page instead.

**Why DJANGO_ALLOWED_HOSTS?** Django refuses to serve requests to unrecognized domains when `DEBUG=False`. This prevents HTTP Host header attacks.

> **Note:** `.env` is listed in `.gitignore` and will NOT be committed to GitHub. You will set these values manually in the Render dashboard (see Step 6 below).

---

### 5. `build.sh` — Created at Project Root

```bash
#!/usr/bin/env bash
set -o errexit
pip install -r requirements.txt
cd ml_at_fingertips
python manage.py collectstatic --no-input
python manage.py migrate
```

**Why?** Render runs this script every time you push new code and trigger a deployment. It:
1. Installs all Python packages
2. Collects static files into `staticfiles/`
3. Applies any new database migrations to your Neon PostgreSQL database automatically

`set -o errexit` means if ANY command fails, the build stops immediately — you won't deploy broken code.

---

### 6. `render.yaml` — Created at Project Root

```yaml
services:
  - type: web
    name: ml-at-fingertips
    runtime: python
    buildCommand: ./build.sh
    startCommand: cd ml_at_fingertips && gunicorn ml_at_fingertips.wsgi:application --bind 0.0.0.0:$PORT --workers 1 --timeout 120
    envVars:
      - key: DJANGO_SECRET_KEY
        generateValue: true   # Render auto-generates a strong key
      - key: DJANGO_DEBUG
        value: "False"
      - key: DJANGO_ALLOWED_HOSTS
        value: ".onrender.com"
      - key: DATABASE_URL
        sync: false            # You set this manually - it's your Neon DB URL
      - key: PERPLEXITY_API_KEY
        sync: false            # You set this manually - it's your API key
```

**Why --workers 1?** TensorFlow (used by the DeepETA model) consumes a lot of memory. Render's free plan gives 512MB RAM. Using 1 worker keeps memory usage safe. You can increase to 2 workers if you upgrade your Render plan.

**Why --timeout 120?** TensorFlow model loading can take 10-30 seconds on first request. Default gunicorn timeout is 30 seconds — it would kill the worker before the model loads. Setting 120 seconds gives it plenty of time.

**Why generateValue: true for DJANGO_SECRET_KEY?** Render can auto-generate a cryptographically secure random key so you don't have to.

---

### 7. `.gitignore` — Created at Project Root

Prevents sensitive and unnecessary files from being committed to GitHub:
- `.env` — contains your database password, API keys, and secret key
- `staticfiles/` — generated by collectstatic, recreated on every deploy
- `db.sqlite3` — local SQLite database (you use Neon in production)
- `__pycache__/`, `.venv/`, `.DS_Store` etc.

---

## Step-by-Step: How to Deploy to Render

### Step 1: Make sure your code is on GitHub
```bash
git add .
git commit -m "Add Render deployment configuration"
git push origin main
```

> **Important:** Your `.env` file is in `.gitignore` so it will NOT be pushed to GitHub. That's correct — secrets should never go to GitHub.

---

### Step 2: Create a Render Account
Go to [https://render.com](https://render.com) and sign up (free).

---

### Step 3: Create a New Web Service
1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub account
3. Select your `ML_AT_FINGERTIPS` repository
4. Render will auto-detect `render.yaml` — click **"Use render.yaml"** if prompted

Or set manually:
- **Runtime:** Python 3
- **Build Command:** `./build.sh`
- **Start Command:** `cd ml_at_fingertips && gunicorn ml_at_fingertips.wsgi:application --bind 0.0.0.0:$PORT --workers 1 --timeout 120`

---

### Step 4: Set Environment Variables in Render Dashboard

Go to your service → **Environment** tab and add:

| Key | Value |
|-----|-------|
| `DATABASE_URL` | Your full Neon connection string: `postgresql://neondb_owner:...@...neon.tech/neondb?sslmode=require&channel_binding=require` |
| `PERPLEXITY_API_KEY` | `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` |
| `DJANGO_DEBUG` | `False` |
| `DJANGO_SECRET_KEY` | Click "Generate" — or paste: `hafs)1c$w#!ry#(3fm$l@uk_9r-ko%r6-q&g8pv0be24l5czv-` |
| `DJANGO_ALLOWED_HOSTS` | `your-app-name.onrender.com` (your actual render URL) |

> **Tip:** After your first deploy, Render gives you a URL like `ml-at-fingertips.onrender.com`. Go back to the env vars and update `DJANGO_ALLOWED_HOSTS` to that exact domain.

---

### Step 5: Deploy
Click **"Deploy"**. Render will:
1. Clone your GitHub repo
2. Run `build.sh` (installs packages, collectstatic, migrate)
3. Start gunicorn server

Watch the **Logs** tab — if it says `Listening at: http://0.0.0.0:PORT` you're live!

---

### Step 6: Add admin user (optional but useful)
After deployment, open the Render Shell (or use the terminal with your database):
```bash
cd ml_at_fingertips
python manage.py createsuperuser
```

---

## Architecture After Deployment

```
========================================
User Browser
    |
    ↓ HTTPS
Render Web Service (gunicorn, 1 worker)
    |
    ├── Static files (CSS/JS/Images)
    │       served by WhiteNoise from /staticfiles/
    │
    ├── Django App (views, templates, ML predictions)
    │       ↓
    ├── Neon PostgreSQL (cloud database)
    │       stores: users, ML model configs, contributions
    │
    └── Perplexity AI API (external)
            provides: prediction explanations & recommendations
========================================
```

---

## Quick Checklist Before Every Deployment

- [ ] `git add .` and `git commit -m "your message"`
- [ ] `git push origin main`
- [ ] Watch Render logs for errors
- [ ] If you add new models/migrations locally — they auto-apply via `build.sh`
- [ ] Never commit `.env` to GitHub

---

## Local Development (unchanged)

For local development, set `DJANGO_DEBUG=True` in your `.env`. The app will:
- Use Neon PostgreSQL (from `DATABASE_URL` in `.env`)
- Be accessible at `http://localhost:8000`

```bash
cd ml_at_fingertips
python manage.py runserver
```
