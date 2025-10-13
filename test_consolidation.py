#!/usr/bin/env python3
"""
Test script to verify the consolidated ML application works correctly
"""

import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_file_structure():
    """Test if all required files are present"""
    print("🔍 Testing File Structure")
    print("=" * 50)
    
    required_files = [
        'classifier/models1/heart_disease_svm_model.pkl',
        'classifier/models1/scalerheart.pkl',
        'classifier/models1/deepeta_nyc_taxi.h5',
        'classifier/models1/deepeta_assets.joblib',
        'classifier/models1/train.csv',
        'classifier/templates/models1.html',
        'classifier/templates/unified_model_detail.html',
        'classifier/management/commands/populate_models.py',
        'ml_at_fingertips/ml_at_fingertips/settings.py',
        'ml_at_fingertips/ml_at_fingertips/urls.py',
        'classifier/urls.py',
        'classifier/views.py',
        'classifier/models.py'
    ]
    
    all_present = True
    for file_path in required_files:
        full_path = os.path.join(project_root, file_path)
        if os.path.exists(full_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING!")
            all_present = False
    
    return all_present

def test_models():
    """Test if both models can be loaded"""
    print("\n🧪 Testing Model Loading")
    print("=" * 50)
    
    try:
        # Test Heart Disease model
        print("Testing Heart Disease model...")
        import pickle
        import numpy as np
        
        model_path = os.path.join(project_root, 'classifier', 'models1', 'heart_disease_svm_model.pkl')
        scaler_path = os.path.join(project_root, 'classifier', 'models1', 'scalerheart.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            with open(model_path, 'rb') as f:
                heart_model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                heart_scaler = pickle.load(f)
            
            # Test prediction
            test_data = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])
            test_scaled = heart_scaler.transform(test_data)
            prediction = heart_model.predict(test_scaled)
            print(f"✅ Heart Disease model works - Prediction: {prediction[0]}")
        else:
            print("❌ Heart Disease model files missing")
            return False
        
        # Test Uber model
        print("Testing Uber ETA model...")
        import tensorflow as tf
        import joblib
        
        uber_model_path = os.path.join(project_root, 'classifier', 'models1', 'deepeta_nyc_taxi.h5')
        uber_assets_path = os.path.join(project_root, 'classifier', 'models1', 'deepeta_assets.joblib')
        
        if os.path.exists(uber_model_path) and os.path.exists(uber_assets_path):
            uber_model = tf.keras.models.load_model(uber_model_path)
            uber_assets = joblib.load(uber_assets_path)
            
            # Test prediction
            test_data = np.array([[5.2, 40.7589, -73.9851, 40.7614, -73.9776, 2, 14, 3, 6]])
            prediction = uber_model.predict(test_data)
            eta = float(prediction[0][0])
            print(f"✅ Uber ETA model works - Prediction: {eta:.1f} minutes")
        else:
            print("❌ Uber ETA model files missing")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure required packages are installed:")
        print("   pip install tensorflow scikit-learn joblib")
        return False
    except Exception as e:
        print(f"❌ Error testing models: {e}")
        return False

def test_templates():
    """Test if templates are properly configured"""
    print("\n📄 Testing Templates")
    print("=" * 50)
    
    try:
        # Check if templates exist and have correct content
        models1_path = os.path.join(project_root, 'classifier', 'templates', 'models1.html')
        detail_path = os.path.join(project_root, 'classifier', 'templates', 'unified_model_detail.html')
        
        if os.path.exists(models1_path):
            with open(models1_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'ML Models Dashboard' in content and 'problem-card' in content:
                    print("✅ models1.html template looks good")
                else:
                    print("❌ models1.html template missing key content")
                    return False
        else:
            print("❌ models1.html template missing")
            return False
        
        if os.path.exists(detail_path):
            with open(detail_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'unified_model_detail' in content and 'prediction' in content:
                    print("✅ unified_model_detail.html template looks good")
                else:
                    print("❌ unified_model_detail.html template missing key content")
                    return False
        else:
            print("❌ unified_model_detail.html template missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing templates: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 ML at Fingertips - Consolidation Test Suite")
    print("=" * 60)
    
    # Test 1: File structure
    files_ok = test_file_structure()
    
    # Test 2: Model loading
    models_ok = test_models()
    
    # Test 3: Templates
    templates_ok = test_templates()
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 50)
    print(f"File Structure: {'✅ PASS' if files_ok else '❌ FAIL'}")
    print(f"Model Loading:  {'✅ PASS' if models_ok else '❌ FAIL'}")
    print(f"Templates:      {'✅ PASS' if templates_ok else '❌ FAIL'}")
    
    if files_ok and models_ok and templates_ok:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ The consolidated ML application is ready to use!")
        print(f"\n🚀 Next steps:")
        print(f"   1. Run: python manage.py makemigrations")
        print(f"   2. Run: python manage.py migrate")
        print(f"   3. Run: python manage.py populate_models")
        print(f"   4. Run: python manage.py runserver")
        print(f"   5. Open: http://127.0.0.1:8000/")
    else:
        print(f"\n❌ SOME TESTS FAILED!")
        print(f"⚠️  Please fix the issues before proceeding")

if __name__ == "__main__":
    main()
