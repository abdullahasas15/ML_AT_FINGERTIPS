import os
import shutil
import sys

def copy_models_and_test():
    """Copy models from regressor to classifier and test"""
    print("🔄 Copying models from regressor to classifier...")
    
    # Source and destination paths
    source_dir = "regressor/models2"
    dest_dir = "classifier/models1"
    
    # Files to copy
    files_to_copy = ["deepeta_assets.joblib", "deepeta_nyc_taxi.h5", "train.csv"]
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"❌ Source directory not found: {source_dir}")
        return False
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Copy files
    for file in files_to_copy:
        source_path = os.path.join(source_dir, file)
        dest_path = os.path.join(dest_dir, file)
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, dest_path)
            print(f"✅ Copied {file}")
        else:
            print(f"⚠️  File not found: {file}")
    
    # Test if files are now present
    print("\n🧪 Testing if models are present...")
    all_present = True
    for file in files_to_copy:
        dest_path = os.path.join(dest_dir, file)
        if os.path.exists(dest_path):
            print(f"✅ {file} is present")
        else:
            print(f"❌ {file} is missing")
            all_present = False
    
    return all_present

def test_uber_model():
    """Simple test of the Uber model"""
    print("\n🚗 Testing Uber model...")
    
    try:
        import tensorflow as tf
        import joblib
        import numpy as np
        
        model_path = "classifier/models1/deepeta_nyc_taxi.h5"
        assets_path = "classifier/models1/deepeta_assets.joblib"
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
        
        if not os.path.exists(assets_path):
            print(f"❌ Assets file not found: {assets_path}")
            return False
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully")
        
        # Load assets
        assets = joblib.load(assets_path)
        print("✅ Assets loaded successfully")
        
        # Test prediction
        test_data = np.array([[5.2, 40.7589, -73.9851, 40.7614, -73.9776, 2, 14, 3, 6]])
        prediction = model.predict(test_data)
        eta = float(prediction[0][0])
        
        print(f"✅ Prediction successful: {eta:.1f} minutes")
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

def delete_regressor_app():
    """Delete the regressor app directory"""
    print("\n🗑️  Deleting regressor app...")
    
    if os.path.exists("regressor"):
        shutil.rmtree("regressor")
        print("✅ Regressor app deleted successfully")
        return True
    else:
        print("⚠️  Regressor app directory not found")
        return False

if __name__ == "__main__":
    print("🧪 Model Migration and Test Suite")
    print("=" * 50)
    
    # Step 1: Copy models
    models_copied = copy_models_and_test()
    
    if models_copied:
        # Step 2: Test the model
        model_works = test_uber_model()
        
        if model_works:
            # Step 3: Delete regressor app
            deleted = delete_regressor_app()
            
            if deleted:
                print("\n🎉 SUCCESS: Migration completed successfully!")
                print("✅ Models copied, tested, and regressor app deleted")
            else:
                print("\n⚠️  Models work but regressor app deletion failed")
        else:
            print("\n❌ Model test failed - keeping regressor app")
    else:
        print("\n❌ Model copy failed - keeping regressor app")
