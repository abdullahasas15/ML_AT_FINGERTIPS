from django.shortcuts import render, get_object_or_404
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

# Create your views here.

def calculate_feature_importance(model, scaler, input_data, feature_names, prediction):
    """Calculate feature importance using permutation importance"""
    try:
        # For permutation importance, we need some reference data
        # We'll create synthetic data based on the input for demonstration
        # In production, you'd want to use actual training data
        
        # Create synthetic reference data around the input
        base_data = np.array(input_data).reshape(1, -1)
        
        # Generate variations of the input data for permutation importance
        n_samples = 50
        synthetic_data = np.tile(base_data, (n_samples, 1))
        
        # Add some noise to create variations
        noise = np.random.normal(0, 0.1, synthetic_data.shape)
        synthetic_data += noise
        
        # Scale the synthetic data if scaler exists
        if scaler:
            synthetic_data_scaled = scaler.transform(synthetic_data)
        else:
            synthetic_data_scaled = synthetic_data
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            model, synthetic_data_scaled, 
            model.predict(synthetic_data_scaled), 
            n_repeats=10, random_state=42
        )
        
        # Get importance scores
        importance_scores = perm_importance.importances_mean
        
        # Create feature importance dictionary
        feature_importance = {}
        for i, feature_name in enumerate(feature_names):
            feature_importance[feature_name] = {
                'importance': float(importance_scores[i]),
                'value': float(input_data[i]),
                'contribution': 'positive' if importance_scores[i] > 0 else 'negative'
            }
        
        # Sort features by importance (descending)
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: x[1]['importance'], 
            reverse=True
        )
        
        return sorted_features
        
    except Exception as e:
        # Fallback: create simple importance based on feature values
        print(f"Error calculating feature importance: {e}")
        return create_simple_feature_importance(input_data, feature_names, prediction)

def create_simple_feature_importance(input_data, feature_names, prediction):
    """Create simple feature importance based on feature values and domain knowledge"""
    # ETA feature importance weights (based on domain knowledge)
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
        
        # Adjust importance based on value and prediction
        if prediction > 30:  # Long ETA
            # Higher values for time-extending factors increase importance
            if feature_name in ['distance', 'pickup_hour']:
                adjusted_importance = weight * (1 + value / 10)
            else:
                adjusted_importance = weight * (1 + value / 100)
        else:  # Short ETA
            # Lower values for time-extending factors increase importance
            adjusted_importance = weight * (1 - value / 100)
        
        feature_importance[feature_name] = {
            'importance': max(0.01, adjusted_importance),  # Ensure positive importance
            'value': float(value),
            'contribution': 'positive' if (prediction > 30 and value > 5) or (prediction <= 30 and value < 5) else 'negative'
        }
    
    # Sort by importance
    return sorted(feature_importance.items(), key=lambda x: x[1]['importance'], reverse=True)

@login_required
def models2_view(request):
    """View for the regression models page - shows all problem statements"""
    problems = ProblemStatement.objects.all()
    return render(request, 'models2.html', {'problems': problems})

@login_required
def model_detail_view(request, problem_id):
    """View for specific model details and prediction"""
    problem = get_object_or_404(ProblemStatement, id=problem_id)
    return render(request, 'model_detail.html', {'problem': problem})

@csrf_exempt
@login_required
def predict_view(request, problem_id):
    """Handle model predictions"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)
    
    try:
        problem = get_object_or_404(ProblemStatement, id=problem_id)
        
        # Get input data from request
        data = json.loads(request.body)
        features = data.get('features', {})
        
        # Load model and scaler
        model_path = os.path.join(settings.BASE_DIR, problem.model_file)
        
        # Try to load as different model types
        model = None
        scaler = None
        
        # Try loading as H5 (Keras/TensorFlow model)
        if model_path.endswith('.h5'):
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(model_path)
            except:
                pass
        
        # Try loading as joblib
        if model is None and (model_path.endswith('.joblib') or model_path.endswith('.pkl')):
            try:
                model = joblib.load(model_path)
            except:
                pass
        
        # Try loading as pickle
        if model is None:
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            except:
                pass
        
        if model is None:
            raise Exception("Could not load model from any supported format")
        
        # Load scaler if it exists
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
        
        # Prepare features in correct order
        feature_order = list(problem.features_description.keys())
        input_data = [features.get(feature, 0) for feature in feature_order]
        
        # Scale the input if scaler exists
        if scaler:
            input_scaled = scaler.transform([input_data])
        else:
            input_scaled = np.array([input_data])
        
        # Make prediction
        if hasattr(model, 'predict'):
            prediction = model.predict(input_scaled)[0]
        else:
            raise Exception("Model does not have predict method")
        
        # Handle different prediction outputs
        if isinstance(prediction, (list, np.ndarray)):
            prediction = prediction[0] if len(prediction) > 0 else prediction
        
        prediction_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                prediction_proba = model.predict_proba(input_scaled)[0]
            except:
                pass
        
        # Calculate feature importance
        feature_names = list(problem.features_description.keys())
        feature_importance = calculate_feature_importance(model, scaler, input_data, feature_names, prediction)
        
        # Get recommendations from Perplexity API
        recommendations = get_perplexity_recommendations(prediction, features, problem.title)
        
        return JsonResponse({
            'prediction': float(prediction),
            'prediction_proba': prediction_proba.tolist() if prediction_proba is not None else None,
            'recommendations': recommendations,
            'feature_importance': feature_importance
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def get_perplexity_recommendations(prediction, features, problem_title):
    """Get recommendations from Perplexity API with fallback"""
    
    # Fallback recommendations based on prediction
    def get_fallback_recommendations(prediction, features):
        eta_minutes = int(prediction)
        
        if eta_minutes <= 15:  # Fast ETA
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
        elif eta_minutes <= 30:  # Moderate ETA
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
        else:  # Long ETA
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
    
    # Try Perplexity API first
    try:
        api_key = "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"
        
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
        
        # Create more detailed prompt with route data
        distance = features.get('distance', 0)
        pickup_hour = features.get('pickup_hour', 0)
        
        prompt = f"""Based on an Uber ETA prediction showing {traffic_level} ({eta_minutes} minutes), provide specific, actionable travel recommendations for a {distance} km trip starting at {pickup_hour}:00 with these route parameters:
- Distance: {distance} km
- Pickup Time: {pickup_hour}:00
- Route Features: {json.dumps(features, indent=2)}

Focus on {advice_type}. Provide practical recommendations for optimizing travel time and avoiding delays. Keep response concise but comprehensive (max 200 words)."""
        
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
    
    # Return fallback recommendations if API fails
    return get_fallback_recommendations(prediction, features)
