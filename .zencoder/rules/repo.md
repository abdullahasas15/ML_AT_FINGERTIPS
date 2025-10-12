# ML_AT_FINGERTIPS Repository Guide

## Overview
- **Project Type**: Django web application offering machine learning models (classification/regression) via a web interface.
- **Primary Apps**:
  - **ml_at_fingertips**: Django project configuration, settings, and URLs.
  - **classifier**: Handles classification problems (e.g., heart disease), templates, static assets, and saved models.
  - **regressor**: Manages regression-focused functionality (currently minimal scaffolding).
  - **accounts**: Authentication-related views and routes.

## Key Paths
1. **Project Root**: `/Users/mohammadabdullah/Documents/GitHub/ML_AT_FINGERTIPS`
2. **Main Django Project**: `/Users/mohammadabdullah/Documents/GitHub/ML_AT_FINGERTIPS/ml_at_fingertips`
3. **Classifier App**:
   - Templates: `/Users/mohammadabdullah/Documents/GitHub/ML_AT_FINGERTIPS/classifier/templates`
   - Static: `/Users/mohammadabdullah/Documents/GitHub/ML_AT_FINGERTIPS/classifier/static`
   - Saved models: `/Users/mohammadabdullah/Documents/GitHub/ML_AT_FINGERTIPS/classifier/models1`
4. **Shared Templates**: `/Users/mohammadabdullah/Documents/GitHub/ML_AT_FINGERTIPS/templates`
5. **Global Static Files**: `/Users/mohammadabdullah/Documents/GitHub/ML_AT_FINGERTIPS/static`

## Setup & Environment
1. **Python Version**: 3.13.x (update virtualenv if needed).
2. **Dependencies**: Install from `requirements.txt`.
   ```bash
   pip install -r /Users/mohammadabdullah/Documents/GitHub/ML_AT_FINGERTIPS/requirements.txt
   ```
3. **Environment Variables**: Configure `.env` file in project root (contains secrets and Django settings overrides).
4. **Database**: Default SQLite at `/Users/mohammadabdullah/Documents/GitHub/ML_AT_FINGERTIPS/ml_at_fingertips/db.sqlite3`.

## Common Commands
1. **Run Development Server**:
   ```bash
   python /Users/mohammadabdullah/Documents/GitHub/ML_AT_FINGERTIPS/ml_at_fingertips/manage.py runserver
   ```
2. **Apply Migrations**:
   ```bash
   python /Users/mohammadabdullah/Documents/GitHub/ML_AT_FINGERTIPS/ml_at_fingertips/manage.py migrate
   ```
3. **Create Superuser**:
   ```bash
   python /Users/mohammadabdullah/Documents/GitHub/ML_AT_FINGERTIPS/ml_at_fingertips/manage.py createsuperuser
   ```

## Notable Files
- `ml_at_fingertips/ml_at_fingertips/settings.py`: Core Django settings.
- `classifier/views.py`: Main views for classification pages.
- `classifier/templates/models1.html`: Template listing available classification models.
- `classifier/templatetags/classifier_extras.py`: Custom template filters (e.g., for accessing dict values).

## Tips
- Keep saved model artifacts in `classifier/models1` and reference them via appropriate views.
- Use template inheritance (e.g., `base.html`) for consistent layout.
- Custom template tags/filters must be loaded in templates via `{% load classifier_extras %}`.
- When adding new apps, register them in `INSTALLED_APPS` inside `settings.py` and include URL confs in the project `urls.py`.

This guide should help orient quick navigation and setup. Update it as the project evolves.