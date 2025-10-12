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

# Create your views here.

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
        
        # Get recommendations from Perplexity API
        recommendations = get_perplexity_recommendations(prediction, features, problem.title)
        
        return JsonResponse({
            'prediction': int(prediction),
            'prediction_proba': prediction_proba.tolist() if prediction_proba is not None else None,
            'recommendations': recommendations
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
