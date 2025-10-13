import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_uber_model():
    """Test the Uber ETA prediction model"""
    print("🚗 Testing Uber ETA Prediction Model")
    print("=" * 50)
    
    # Model paths
    model_path = os.path.join(project_root, 'deepeta_nyc_taxi.h5')
    assets_path = os.path.join(project_root, 'deepeta_assets.joblib')
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    if not os.path.exists(assets_path):
        print(f"❌ Assets file not found: {assets_path}")
        return False
    
    print(f"✅ Model file found: {model_path}")
    print(f"✅ Assets file found: {assets_path}")
    
    try:
        # Load the model
        print("\n📦 Loading model...")
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully")
        
        # Load assets (scaler, etc.)
        print("\n📦 Loading assets...")
        import joblib
        assets = joblib.load(assets_path)
        print("✅ Assets loaded successfully")
        
        # Test data (sample Uber trip features)
        test_features = {
            'distance': 5.2,           # km
            'pickup_latitude': 40.7589,
            'pickup_longitude': -73.9851,
            'dropoff_latitude': 40.7614,
            'dropoff_longitude': -73.9776,
            'passenger_count': 2,
            'pickup_hour': 14,         # 2 PM
            'pickup_day': 3,           # Wednesday
            'pickup_month': 6          # June
        }
        
        print(f"\n🧪 Test Features:")
        for key, value in test_features.items():
            print(f"   {key}: {value}")
        
        # Prepare input data
        import numpy as np
        feature_order = list(test_features.keys())
        input_data = [test_features[feature] for feature in feature_order]
        input_array = np.array([input_data])
        
        print(f"\n🔢 Input array shape: {input_array.shape}")
        
        # Make prediction
        print("\n🔮 Making prediction...")
        prediction = model.predict(input_array)
        
        if isinstance(prediction, np.ndarray):
            eta_minutes = float(prediction[0][0])
        else:
            eta_minutes = float(prediction[0])
        
        print(f"✅ Prediction successful!")
        print(f"🚗 Predicted ETA: {eta_minutes:.1f} minutes")
        
        # Interpret result
        if eta_minutes <= 15:
            status = "Fast ETA - Light traffic"
            emoji = "🚀"
        elif eta_minutes <= 30:
            status = "Moderate ETA - Normal traffic"
            emoji = "⏰"
        else:
            status = "Long ETA - Heavy traffic"
            emoji = "🚦"
        
        print(f"{emoji} Status: {status}")
        
        print(f"\n✅ Uber model test passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure TensorFlow is installed: pip install tensorflow")
        return False
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Uber Model Test")
    print("=" * 50)
    
    success = test_uber_model()
    
    if success:
        print(f"\n🎉 SUCCESS: Uber model is working perfectly!")
        print(f"✅ Ready to delete regressor app")
    else:
        print(f"\n❌ FAILURE: Uber model has issues")
        print(f"⚠️  Do not delete regressor app yet")