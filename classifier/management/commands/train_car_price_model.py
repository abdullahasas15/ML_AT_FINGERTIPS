from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from classifier.models import ProblemStatement
from django.db.models import Q
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump

# Import shared feature function so the pickled pipeline loads at runtime
from classifier.models1.transformers import add_squared_features

class Command(BaseCommand):
    help = "Train and (re)save the car price prediction pipeline. Usage: python manage.py train_car_price_model --csv path/to/data.csv"

    def add_arguments(self, parser):
        parser.add_argument('--csv', type=str, required=True, help='Path to car price dataset CSV')
        parser.add_argument('--target', type=str, default='selling_price', help='Target column name (default: selling_price)')
        parser.add_argument('--output', type=str, default='classifier/models1/secondhandcarprice.pkl', help='Where to save the trained pipeline')

    def handle(self, *args, **options):
        csv_path = options['csv']
        target_col = options['target']
        output_rel = options['output']
        output_path = os.path.join(settings.BASE_DIR, output_rel)

        if not os.path.exists(csv_path):
            raise CommandError(f"CSV not found: {csv_path}")

        self.stdout.write(self.style.NOTICE(f"Loading dataset: {csv_path}"))
        df = pd.read_csv(csv_path)

        # Normalize column names
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

        # Derive 'age' if the dataset has 'year'
        from datetime import datetime
        if 'age' not in df.columns and 'year' in df.columns:
            current_year = datetime.now().year
            df['age'] = (current_year - pd.to_numeric(df['year'], errors='coerce')).clip(lower=0)

        # Normalize km column variants
        if 'km_driven' not in df.columns:
            for alt in ['kms_driven', 'km', 'kms']:
                if alt in df.columns:
                    df['km_driven'] = pd.to_numeric(df[alt], errors='coerce')
                    break

        required_cols = ['age', 'km_driven', 'name', 'fuel', 'seller_type', 'transmission', 'owner']
        missing = [c for c in required_cols + [target_col] if c not in df.columns]
        if missing:
            raise CommandError(f"Dataset missing columns: {missing}")

        # Basic cleaning
        df = df.dropna(subset=required_cols + [target_col])
        df = df.reset_index(drop=True)

        # Train/Val split
        X = df[required_cols].copy()
        y = df[target_col].astype(float)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocessing: engineered features + encoding + scaling
        add_feats = FunctionTransformer(add_squared_features, validate=False)

        numeric_features = ['age', 'km_driven', 'age_sq', 'km_driven_sq']
        categorical_features = ['name', 'fuel', 'seller_type', 'transmission', 'owner']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[('scaler', StandardScaler())]), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ],
            remainder='drop'
        )

        ridge = Pipeline(steps=[
            ('add', add_feats),
            ('prep', preprocessor),
            ('model', Ridge(alpha=5.0))
        ])

        rf = Pipeline(steps=[
            ('add', add_feats),
            ('prep', preprocessor),
            ('model', RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ))
        ])

        # Train and evaluate both, pick the better R2 on validation
        models = {
            'Ridge': ridge,
            'RandomForest': rf
        }
        scores = {}

        for name, pipe in models.items():
            self.stdout.write(self.style.NOTICE(f"Training {name}..."))
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            r2 = r2_score(y_test, pred)
            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            scores[name] = {'r2': float(r2), 'rmse': rmse}
            self.stdout.write(self.style.SUCCESS(f"{name} -> R2={r2:.4f}, RMSE={rmse:,.0f}"))

        # Choose best by R2
        best_model_name = max(scores.keys(), key=lambda k: scores[k]['r2'])
        best_pipe = models[best_model_name]

        # Save pipeline
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dump(best_pipe, output_path)
        self.stdout.write(self.style.SUCCESS(f"Saved pipeline: {output_path}"))

        # Update ProblemStatement if present
        try:
            ps = ProblemStatement.objects.filter(Q(title__icontains='car') & Q(title__icontains='price')).first()
            if ps is None:
                raise ProblemStatement.DoesNotExist()
            ps.selected_model = best_model_name
            ps.model_file = output_rel
            # Build accuracy_scores
            acc = {k: v['r2'] for k, v in scores.items()}
            ps.accuracy_scores = acc
            # Update model_info with car names
            car_names = sorted(list(pd.Series(df['name']).dropna().astype(str).unique()))[:1000]
            info = ps.model_info
            if isinstance(info, str):
                try:
                    info = json.loads(info)
                except Exception:
                    info = {}
            if not isinstance(info, dict):
                info = {}
            info['features'] = required_cols
            info['target'] = target_col
            info['r2_score'] = scores[best_model_name]['r2']
            info['rmse'] = scores[best_model_name]['rmse']
            info['car_names'] = car_names
            ps.model_info = json.dumps(info)
            ps.save()
            self.stdout.write(self.style.SUCCESS("ProblemStatement updated with new metrics and model file."))
        except ProblemStatement.DoesNotExist:
            self.stdout.write(self.style.WARNING("ProblemStatement entry not found; skipping DB update."))

        self.stdout.write(self.style.SUCCESS(f"Training complete. Best: {best_model_name} R2={scores[best_model_name]['r2']:.4f}"))
