from django.contrib.auth import get_user
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import pickle
import joblib
import os
from django.conf import settings
import requests
from .models import ProblemStatement
import numpy as np
from sklearn.inspection import permutation_importance
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

def landing_page(request):
    user = get_user(request)
    if user.is_authenticated:
        return redirect('home')
    return render(request, 'base.html')

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def manhattan_km(lon1, lat1, lon2, lat2):
    return haversine_km(lon1, lat1, lon2, lat1) + haversine_km(lon1, lat1, lon1, lat2)

def calculate_feature_importance(model, scaler, input_data, feature_names, prediction, model_type):
    try:
        base_data = np.array(input_data).reshape(1, -1)
        n_samples = 50
        synthetic_data = np.tile(base_data, (n_samples, 1))
        noise = np.random.normal(0, 0.1, synthetic_data.shape)
        synthetic_data += noise
        synthetic_data_scaled = scaler.transform(synthetic_data)
        perm_importance = permutation_importance(
            model, synthetic_data_scaled, 
            model.predict(synthetic_data_scaled), 
            n_repeats=10, random_state=42
        )
        importance_scores = perm_importance.importances_mean
        feature_importance = {}
        for i, feature_name in enumerate(feature_names):
            feature_importance[feature_name] = {
                'importance': float(importance_scores[i]),
                'value': float(input_data[i]),
                'contribution': 'positive' if importance_scores[i] > 0 else 'negative'
            }
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1]['importance'], reverse=True)
        return sorted_features
    except Exception as e:
        print(f"Error calculating feature importance: {e}")
        return create_simple_feature_importance(input_data, feature_names, prediction, model_type)

def create_simple_feature_importance(input_data, feature_names, prediction, model_type):
    if model_type.lower() == 'regression':
        eta_weights = {
            'distance': 0.25,
            'pickup_hour': 0.20,
            'pickup_latitude': 0.15,
            'pickup_longitude': 0.15,
            'dropoff_latitude': 0.10,
            'dropoff_longitude': 0.10,
            'passenger_count': 0.03,
            'pickup_day': 0.02,
            'pickup_month': 0.00
        }
        feature_importance = {}
        for i, feature_name in enumerate(feature_names):
            value = input_data[i]
            weight = eta_weights.get(feature_name, 0.05)
            if prediction > 30:
                if feature_name in ['distance', 'pickup_hour']:
                    adjusted_importance = weight * (1 + value / 10)
                else:
                    adjusted_importance = weight * (1 + value / 100)
            else:
                adjusted_importance = weight * (1 - value / 100)
            feature_importance[feature_name] = {
                'importance': max(0.01, adjusted_importance),
                'value': float(value),
                'contribution': 'positive' if (prediction > 30 and value > 5) or (prediction <= 30 and value < 5) else 'negative'
            }
    else:
        medical_weights = {
            'age': 0.15,
            'sex': 0.08,
            'cp': 0.12,
            'trestbps': 0.10,
            'chol': 0.10,
            'fbs': 0.06,
            'restecg': 0.05,
            'thalach': 0.08,
            'exang': 0.08,
            'oldpeak': 0.10,
            'slope': 0.06,
            'ca': 0.10,
            'thal': 0.12
        }
        feature_importance = {}
        for i, feature_name in enumerate(feature_names):
            value = input_data[i]
            weight = medical_weights.get(feature_name, 0.05)
            if prediction == 1:
                if feature_name in ['age', 'trestbps', 'chol', 'oldpeak', 'ca']:
                    adjusted_importance = weight * (1 + value / 100)
                else:
                    adjusted_importance = weight * (1 + value / 10)
            else:
                adjusted_importance = weight * (1 - value / 100)
            feature_importance[feature_name] = {
                'importance': max(0.01, adjusted_importance),
                'value': float(value),
                'contribution': 'positive' if (prediction == 1 and value > 50) or (prediction == 0 and value < 50) else 'negative'
            }
    return sorted(feature_importance.items(), key=lambda x: x[1]['importance'], reverse=True)

@login_required
def models1_view(request):
    problems = ProblemStatement.objects.all()
    return render(request, 'models1.html', {'problems': problems})

@login_required
def model_detail_view(request, problem_id):
    problem = get_object_or_404(ProblemStatement, id=problem_id)
    return render(request, 'unified_model_detail.html', {'problem': problem})

@csrf_exempt
@login_required
def predict_view(request, problem_id):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)
    try:
        try:
            problem = ProblemStatement.objects.get(id=problem_id)
        except ProblemStatement.DoesNotExist:
            available = list(ProblemStatement.objects.values_list('id', flat=True))
            print(f"Predict called for problem_id={problem_id} but not found. Available ids: {available}")
            return JsonResponse({'error': f'ProblemStatement with id {problem_id} not found'}, status=404)
        data = json.loads(request.body)
        features = data.get('features', {})

        # Ensure NYC features to calculate distances
        req_keys = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
        for k in req_keys:
            if k not in features:
                return JsonResponse({'error': f'Missing required input: {k}'}, status=400)
        # Calculate haversine/manhattan if missing
        features['haversine_km'] = haversine_km(
            float(features['pickup_longitude']), float(features['pickup_latitude']),
            float(features['dropoff_longitude']), float(features['dropoff_latitude'])
        )
        features['manhattan_km'] = manhattan_km(
            float(features['pickup_longitude']), float(features['pickup_latitude']),
            float(features['dropoff_longitude']), float(features['dropoff_latitude'])
        )

        model_path = os.path.join(settings.BASE_DIR, problem.model_file)
        model = None
        scaler = None
        if model_path.endswith('.h5'):
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(
                    model_path,
                    custom_objects={
                        'mse': tf.keras.metrics.MeanSquaredError(),
                        'mean_squared_error': tf.keras.metrics.MeanSquaredError(),
                        'MeanSquaredError': tf.keras.metrics.MeanSquaredError()
                    }
                )
            except Exception as e:
                print(f"Error loading Keras model: {e}")
        if model is None and (model_path.endswith('.joblib') or model_path.endswith('.pkl')):
            try:
                model = joblib.load(model_path)
            except:
                pass
        if model is None:
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            except:
                pass
        if model is None:
            raise Exception("Could not load model from any supported format")
        if problem.scaler_file:
            scaler_path = os.path.join(settings.BASE_DIR, problem.scaler_file)
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            except:
                try:
                    scaler = joblib.load(scaler_path)
                except:
                    pass
        feature_order = [
            'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
            'passenger_count', 'hour', 'dow', 'month', 'haversine_km', 'manhattan_km'
        ]
        # Always calculate haversine_km and manhattan_km
        try:
            features['haversine_km'] = haversine_km(
                float(features['pickup_longitude']), float(features['pickup_latitude']),
                float(features['dropoff_longitude']), float(features['dropoff_latitude'])
            )
            features['manhattan_km'] = manhattan_km(
                float(features['pickup_longitude']), float(features['pickup_latitude']),
                float(features['dropoff_longitude']), float(features['dropoff_latitude'])
            )
        except Exception as e:
            print(f"[ERROR] Could not calculate distance features: {e}")
        input_data = [features.get(feature, 0) for feature in feature_order]
        print(f"[DEBUG] Received features: {features}")
        print(f"[DEBUG] Input data for model: {input_data}")
        if problem.model_type.lower() == 'regression' and problem.selected_model.lower().startswith('deepeta'):
            if len(input_data) != 10:
                print(f"[ERROR] DeepETA_DNN expects 10 features, got {len(input_data)}.")
                print(f"Received features: {features}")
                print(f"Expected features: {feature_order}")
                return JsonResponse({
                    'error': f'DeepETA_DNN expects 10 features, but received {len(input_data)}. Please check your input.'
                }, status=400)
        if problem.model_type.lower() == 'regression' and problem.selected_model.lower().startswith('deepeta'):
            if scaler and hasattr(scaler, 'transform'):
                input_scaled = scaler.transform([input_data])
            else:
                input_scaled = np.array([input_data])
            prediction = model.predict(input_scaled)
            if isinstance(prediction, np.ndarray):
                eta_minutes = float(prediction[0][0]) if prediction.ndim == 2 else float(prediction[0])
            else:
                eta_minutes = float(prediction)
            prediction_value = eta_minutes
            prediction_proba = None
        else:
            if scaler:
                input_scaled = scaler.transform([input_data])
            else:
                input_scaled = np.array([input_data])
            if hasattr(model, 'predict'):
                prediction = model.predict(input_scaled)[0]
            else:
                raise Exception("Model does not have predict method")
            if isinstance(prediction, (list, np.ndarray)):
                prediction = prediction[0] if len(prediction) > 0 else prediction
            prediction_value = int(prediction)
            prediction_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    prediction_proba = model.predict_proba(input_scaled)[0]
                except:
                    pass
            try:
                prediction_proba = model.predict_proba(input_scaled)[0]
            except:
                pass
        feature_names = list(problem.features_description.keys())
        feature_importance = calculate_feature_importance(model, scaler, input_data, feature_names, prediction, problem.model_type)
        recommendations = get_perplexity_recommendations(prediction, features, problem.title, problem.model_type)
        # Return prediction and advice
        return JsonResponse({
            'prediction': prediction_value,
            'feature_importance': feature_importance,
            'recommendations': recommendations
        }, status=200)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def get_perplexity_recommendations(prediction, features, problem_title, model_type):
    def get_fallback_recommendations(prediction, features, model_type):
        if model_type.lower() == 'regression':
            eta_minutes = int(prediction)
            if eta_minutes <= 15:
                recommendations = []
                recommendations.append("🚗 **Great News! Fast ETA Predicted**")
                recommendations.append("\n**Current Conditions:**")
                recommendations.append("• Traffic is light and flowing well")
                recommendations.append("• Optimal route conditions")
                recommendations.append("• Good time to travel")
                recommendations.append("\n**Tips to Maintain Speed:**")
                recommendations.append("• Avoid unnecessary stops")
                recommendations.append("• Use main roads and highways")
                recommendations.append("• Check real-time traffic updates")
                recommendations.append("• Consider alternative routes if delays occur")
                return "\n".join(recommendations)
            elif eta_minutes <= 30:
                recommendations = []
                recommendations.append("⏰ **Moderate ETA: Normal Traffic Conditions**")
                recommendations.append("\n**Current Situation:**")
                recommendations.append("• Standard traffic patterns")
                recommendations.append("• Some congestion expected")
                recommendations.append("• Reasonable travel time")
                recommendations.append("\n**Optimization Tips:**")
                recommendations.append("• Plan for minor delays")
                recommendations.append("• Consider timing adjustments")
                recommendations.append("• Monitor traffic updates")
                recommendations.append("• Have backup routes ready")
                return "\n".join(recommendations)
            else:
                recommendations = []
                recommendations.append("🚦 **Long ETA: Heavy Traffic Expected**")
                recommendations.append("\n**Current Challenges:**")
                recommendations.append("• Significant traffic congestion")
                recommendations.append("• Peak hour conditions")
                recommendations.append("• Possible route delays")
                recommendations.append("\n**Alternative Strategies:**")
                recommendations.append("• Consider different departure time")
                recommendations.append("• Explore alternative routes")
                recommendations.append("• Check for road closures or incidents")
                recommendations.append("• Plan for extra travel time")
                recommendations.append("• Consider public transportation if available")
                return "\n".join(recommendations)
        else:
            if prediction == 1:
                age = features.get('age', 0)
                chol = features.get('chol', 0)
                trestbps = features.get('trestbps', 0)
                recommendations = []
                recommendations.append("⚠️ **Immediate Actions Required:**")
                recommendations.append("• Schedule an appointment with a cardiologist as soon as possible")
                recommendations.append("• Monitor your blood pressure daily")
                if chol > 240:
                    recommendations.append("• Reduce cholesterol through diet: limit saturated fats, increase fiber intake")
                if trestbps > 140:
                    recommendations.append("• Manage blood pressure: reduce sodium intake, maintain healthy weight")
                if age > 55:
                    recommendations.append("• Regular cardiac check-ups every 3-6 months recommended")
                recommendations.append("\n**Lifestyle Changes:**")
                recommendations.append("• Exercise: 30 minutes of moderate activity, 5 days/week")
                recommendations.append("• Diet: Mediterranean diet with fruits, vegetables, whole grains, lean proteins")
                recommendations.append("• Stress management: Practice meditation, yoga, or deep breathing")
                recommendations.append("• Quit smoking and limit alcohol consumption")
                recommendations.append("• Maintain healthy weight (BMI 18.5-24.9)")
                return "\n".join(recommendations)
            else:
                recommendations = []
                recommendations.append("✅ **Good News! Low Risk Detected**")
                recommendations.append("\n**Maintain Your Heart Health:**")
                recommendations.append("• Continue regular physical activity (150 min/week)")
                recommendations.append("• Maintain a balanced, heart-healthy diet")
                recommendations.append("• Annual health check-ups recommended")
                recommendations.append("• Monitor blood pressure and cholesterol levels")
                recommendations.append("• Stay hydrated and manage stress effectively")
                recommendations.append("• Avoid smoking and excessive alcohol")
                recommendations.append("\n**Prevention is Key:** Keep up your healthy lifestyle to maintain low risk!")
                return "\n".join(recommendations)
    try:
        api_key = "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"
        if model_type.lower() == 'regression':
            eta_minutes = int(prediction)
            if eta_minutes <= 15:
                traffic_level = "light traffic with fast ETA"
                advice_type = "maintaining speed and optimizing route"
            elif eta_minutes <= 30:
                traffic_level = "moderate traffic with normal ETA"
                advice_type = "route optimization and timing strategies"
            else:
                traffic_level = "heavy traffic with longer ETA"
                advice_type = "alternative strategies and route planning"
            distance = features.get('distance', 0)
            pickup_hour = features.get('pickup_hour', 0)
            prompt = f"""Based on an Uber ETA prediction showing {traffic_level} ({eta_minutes} minutes), provide specific, actionable travel recommendations for a {distance} km trip starting at {pickup_hour}:00 with these route parameters:
- Distance: {distance} km
- Pickup Time: {pickup_hour}:00
- Route Features: {json.dumps(features, indent=2)}
Focus on {advice_type}. Provide practical recommendations for optimizing travel time and avoiding delays. Keep response concise but comprehensive (max 200 words)."""
        else:
            if prediction == 1:
                risk_level = "high risk"
                advice_type = "preventive measures and lifestyle changes"
            else:
                risk_level = "low risk"
                advice_type = "maintenance and prevention strategies"
            age = features.get('age', 0)
            chol = features.get('chol', 0)
            trestbps = features.get('trestbps', 0)
            prompt = f"""Based on a {problem_title} prediction showing {risk_level}, provide specific, actionable health recommendations for a {age}-year-old patient with:
- Cholesterol: {chol} mg/dl
- Blood Pressure: {trestbps} mm Hg
- Other risk factors: {json.dumps(features, indent=2)}
Focus on {advice_type}. Provide evidence-based, practical recommendations that are easy to follow. Keep response concise but comprehensive (max 200 words)."""
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': 'sonar-pro',
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 300
        }
        response = requests.post(
            'https://api.perplexity.ai/chat/completions',
            headers=headers,
            json=data,
            timeout=15
        )
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
    except:
        pass
    return get_fallback_recommendations(prediction, features, model_type)
