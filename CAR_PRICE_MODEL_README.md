# Car Price Prediction – Current Setup and Usage

This app is wired to use a trained scikit-learn Pipeline for second-hand car price prediction. The web UI is ready: choose the car price model, fill inputs, and get a ₹ prediction with explanations.

## Current model

- File: `classifier/models1/secondhandcarprice.pkl`
- Type: sklearn Pipeline = FunctionTransformer(add_squared_features) → ColumnTransformer (scale + one-hot) → Ridge
- Validation: R² ≈ 0.555, RMSE ≈ ₹368k (on provided dataset split)
- Database: `ProblemStatement` updated with selected_model, accuracy_scores, and `model_info` (includes `car_names` for autocomplete)

## How to use in the website

1. Start the server:
   ```bash
   cd ml_at_fingertips
   python3 manage.py runserver
   ```
2. Log in and open the Models Dashboard.
3. Click the “Car Price” problem → Try.
4. Fill the fields:
   - name (autocomplete list)
   - age (years)
   - km_driven
   - fuel, seller_type, transmission, owner
5. Click “Get Prediction” → Price is shown in ₹ with top factors and tips.

## Retraining the model (optional)

Use the management command to retrain with a CSV (supports `year` → derives `age`):

```bash
cd ml_at_fingertips
python3 manage.py train_car_price_model --csv \
"/absolute/path/to/CAR DETAILS FROM CAR DEKHO.csv"
```

What it does:
- Normalizes columns, derives `age` (from `year`) and squared features.
- Trains Ridge and RandomForest Pipelines; picks best by R².
- Saves to `classifier/models1/secondhandcarprice.pkl` and updates the `ProblemStatement` (metrics + car_names).

## Inputs expected

- name, age, km_driven, fuel, seller_type, transmission, owner

## Notes

- The UI’s car name field uses `model_info.car_names` for autocomplete.
- Predictions are made end-to-end inside the Pipeline; no manual encoders needed at inference.
- If prices seem high/low for a segment, consider retraining with:
  - log-transform target, brand extracted from name, tuned Ridge/RandomForest.
