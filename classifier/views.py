from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import pickle
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
        
        # Scale the synthetic data
        synthetic_data_scaled = scaler.transform(synthetic_data)
        
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
    """Create simple feature importance based on feature values and medical knowledge"""
    # Medical feature importance weights (based on medical research)
    medical_weights = {
        'age': 0.15,
        'sex': 0.08,
        'cp': 0.12,  # chest pain
        'trestbps': 0.10,  # blood pressure
        'chol': 0.10,  # cholesterol
        'fbs': 0.06,  # fasting blood sugar
        'restecg': 0.05,  # resting ECG
        'thalach': 0.08,  # max heart rate
        'exang': 0.08,  # exercise induced angina
        'oldpeak': 0.10,  # ST depression
        'slope': 0.06,  # slope of peak exercise ST segment
        'ca': 0.10,  # number of major vessels
        'thal': 0.12   # thalassemia
    }
    
    feature_importance = {}
    for i, feature_name in enumerate(feature_names):
        value = input_data[i]
        weight = medical_weights.get(feature_name, 0.05)
        
        # Adjust importance based on value and prediction
        if prediction == 1:  # High risk
            # Higher values for risk factors increase importance
            if feature_name in ['age', 'trestbps', 'chol', 'oldpeak', 'ca']:
                adjusted_importance = weight * (1 + value / 100)
            else:
                adjusted_importance = weight * (1 + value / 10)
        else:  # Low risk
            # Lower values for risk factors increase importance
            adjusted_importance = weight * (1 - value / 100)
        
        feature_importance[feature_name] = {
            'importance': max(0.01, adjusted_importance),  # Ensure positive importance
            'value': float(value),
            'contribution': 'positive' if (prediction == 1 and value > 50) or (prediction == 0 and value < 50) else 'negative'
        }
    
    # Sort by importance
    return sorted(feature_importance.items(), key=lambda x: x[1]['importance'], reverse=True)

@login_required
def models1_view(request):
    """View for the first model page - shows all problem statements"""
    problems = ProblemStatement.objects.all()
    return render(request, 'models1.html', {'problems': problems})

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
        scaler_path = os.path.join(settings.BASE_DIR, problem.scaler_file)
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Prepare features in correct order
        feature_order = list(problem.features_description.keys())
        input_data = [features.get(feature, 0) for feature in feature_order]
        
        # Scale the input
        input_scaled = scaler.transform([input_data])
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0] if hasattr(model, 'predict_proba') else None
        
        # Calculate feature importance
        feature_names = list(problem.features_description.keys())
        feature_importance = calculate_feature_importance(model, scaler, input_data, feature_names, prediction)
        
        # Get recommendations from Perplexity API
        recommendations = get_perplexity_recommendations(prediction, features, problem.title)
        
        return JsonResponse({
            'prediction': int(prediction),
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
        if prediction == 1:  # High risk
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
        else:  # Low risk
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
    
    # Try Perplexity API first
    try:
        api_key = "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"
        
        if prediction == 1:
            risk_level = "high risk"
            advice_type = "preventive measures and lifestyle changes"
        else:
            risk_level = "low risk"
            advice_type = "maintenance and prevention strategies"
        
        # Create more detailed prompt with patient data
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
    
    # Return fallback recommendations if API fails
    return get_fallback_recommendations(prediction, features)
