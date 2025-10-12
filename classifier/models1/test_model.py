#!/usr/bin/env python3
"""
Test script for heart disease prediction models
Run this to verify the models are working correctly
"""

import pickle
import numpy as np
import os

def test_models():
    """Test the heart disease prediction models"""
    
    print("🧪 Testing Heart Disease Prediction Models")
    print("=" * 50)
    
    # Check if files exist
    model_path = "heart_disease_svm_model.pkl"
    scaler_path = "scalerheart.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    if not os.path.exists(scaler_path):
        print(f"❌ Scaler file not found: {scaler_path}")
        return False
    
    print(f"✅ Found model file: {model_path}")
    print(f"✅ Found scaler file: {scaler_path}")
    
    # Load models
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded successfully: {type(model)}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✅ Scaler loaded successfully: {type(scaler)}")
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")
        return False
    
    # Test predictions with sample data
    test_cases = [
        {
            "name": "High Risk Case",
            "data": [65, 1, 3, 145, 280, 1, 0, 150, 1, 3.5, 0, 2, 2],  # High risk
            "expected": "High risk"
        },
        {
            "name": "Low Risk Case", 
            "data": [35, 0, 0, 120, 200, 0, 1, 180, 0, 0.5, 1, 0, 1],  # Low risk
            "expected": "Low risk"
        }
    ]
    
    print("\n🔬 Testing Predictions:")
    print("-" * 30)
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            # Prepare data
            sample_data = np.array([test_case["data"]])
            scaled_data = scaler.transform(sample_data)
            
            # Make prediction
            prediction = model.predict(scaled_data)[0]
            probability = model.predict_proba(scaled_data)[0]
            
            # Interpret results
            risk_level = "High risk" if prediction == 1 else "Low risk"
            confidence = max(probability) * 100
            
            print(f"Test {i}: {test_case['name']}")
            print(f"  Input: {test_case['data'][:5]}... (first 5 features)")
            print(f"  Prediction: {risk_level}")
            print(f"  Confidence: {confidence:.1f}%")
            print(f"  Probabilities: [Low: {probability[0]:.3f}, High: {probability[1]:.3f}]")
            print()
            
        except Exception as e:
            print(f"❌ Error in test {i}: {e}")
            return False
    
    print("✅ All tests passed! Models are working correctly.")
    print("=" * 50)
    return True

if __name__ == "__main__":
    # Change to the directory containing this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_models()
