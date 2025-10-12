# Heart Disease Prediction Models

This folder contains the working machine learning models for heart disease risk prediction.

## Files:

### 1. `heart_disease_svm_model.pkl`
- **Type**: Support Vector Machine (SVM) classifier
- **Purpose**: Main prediction model for heart disease risk
- **Accuracy**: 79%
- **Features**: 13 health parameters (age, sex, chest pain, blood pressure, etc.)
- **Output**: Binary prediction (0 = low risk, 1 = high risk) with probability scores

### 2. `scalerheart.pkl`
- **Type**: StandardScaler from scikit-learn
- **Purpose**: Normalizes input features to ensure consistent model performance
- **Usage**: Applied to all input data before making predictions

## Model Details:

- **Training Data**: 1000 synthetic samples based on real heart disease patterns
- **Algorithm**: SVM with RBF kernel
- **Features**: 13 medical parameters
- **Validation**: 80/20 train-test split
- **Performance**: 79% accuracy with probability predictions

## Usage:

These models are automatically loaded by the Django application when users make predictions through the web interface. The models are used in the `classifier/views.py` file in the `predict_view` function.

## Created:
- **Date**: October 12, 2025
- **Status**: ✅ Working and tested
- **Integration**: Fully integrated with Django web application

## Notes:
- These models replaced the original corrupted PKL files
- The models include probability predictions for better user experience
- All features are properly scaled for consistent predictions
