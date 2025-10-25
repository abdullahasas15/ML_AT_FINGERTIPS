python3 ml_at_fingertips/manage.py runserver

# 🔄 Complete ML Prediction Flow Documentation

## 📌 Overview
This document explains the complete flow of how Machine Learning models work in your Django application - from training in Google Colab to making predictions and getting recommendations.

---

## 🎯 PART 1: MODEL TRAINING & UPLOAD (Google Colab)

### Step 1: Train Your Model in Colab
```python
# In Google Colab Notebook
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# 1. Load your dataset
data = pd.read_csv('heart_disease.csv')

# 2. Split features and target
X = data.drop('target', axis=1)
y = data['target']

# 3. Create and train scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train your model
model = RandomForestClassifier()
model.fit(X_scaled, y)

# 5. Save model and scaler as PKL files
with open('heart_disease_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

### Step 2: Download PKL Files
- Download `heart_disease_model.pkl` (your trained model)
- Download `scaler.pkl` (your feature scaler)

### Step 3: Upload to Django Project
Upload both files to:
```
/classifier/models1/
├── heart_disease_model.pkl  ← Your model
├── scaler.pkl                ← Your scaler
```

---

## 🗄️ PART 2: DATABASE SETUP (ProblemStatement Model)

### File Location: `/classifier/models.py`

```python
class ProblemStatement(models.Model):
    id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=200)  # e.g., "Heart Disease Prediction"
    description = models.TextField()
    
    # 🔑 KEY FIELDS - These tell Django where your model files are
    model_file = models.CharField(max_length=200)    # Path: "classifier/models1/heart_disease_model.pkl"
    scaler_file = models.CharField(max_length=200)   # Path: "classifier/models1/scaler.pkl"
    
    model_type = models.CharField(max_length=100)    # "Classification" or "Regression"
    features_description = models.JSONField()        # JSON with feature names and descriptions
    accuracy_scores = models.JSONField()             # Model accuracy metrics
    selected_model = models.CharField(max_length=100) # "SVM", "Random Forest", etc.
    code_snippet = models.TextField()                # Python code shown to users
```

### How to Add Your Model to Database

#### Option 1: Using Django Management Command
File: `/classifier/management/commands/populate_models.py`

```python
python3 ml_at_fingertips/manage.py populate_models
```

This command creates a database entry like:
```python
ProblemStatement.objects.create(
    title="Heart Disease Prediction",
    model_file="classifier/models1/heart_disease_svm_model.pkl",  # ← Your PKL file path
    scaler_file="classifier/models1/scalerheart.pkl",              # ← Your scaler path
    model_type="Classification",
    features_description={
        "age": "Age of the patient",
        "sex": "Gender (1=male, 0=female)",
        "cp": "Chest pain type (0-3)",
        # ... all your features
    },
    accuracy_scores={
        "SVM": 0.85,
        "Random Forest": 0.88
    },
    selected_model="SVM"
)
```

#### Option 2: Django Admin Panel
1. Go to `http://localhost:8000/admin`
2. Add a new ProblemStatement
3. Fill in model_file path and scaler_file path

---

## 🌐 PART 3: FRONTEND TO BACKEND FLOW

### Step 1: User Opens Model Page
**URL**: `http://localhost:8000/models/`

**View Function**: `/classifier/views.py` → `models1_view()`
```python
@login_required
def models1_view(request):
    # Fetches ALL problem statements from database
    problems = ProblemStatement.objects.all()
    # Renders page showing all available models
    return render(request, 'models1.html', {'problems': problems})
```

**Template**: `/classifier/templates/models1.html`
- Shows cards for each model (Heart Disease, Uber ETA, Car Price)
- User clicks "Try" button → Goes to model detail page

---

### Step 2: User Opens Model Detail Page
**URL**: `http://localhost:8000/model/1/` (where 1 is the problem_id)

**View Function**: `/classifier/views.py` → `model_detail_view()`
```python
@login_required
def model_detail_view(request, problem_id):
    # Get specific problem from database using ID
    problem = get_object_or_404(ProblemStatement, id=problem_id)
    # Render detail page with model info
    return render(request, 'unified_model_detail.html', {'problem': problem})
```

**Template**: `/classifier/templates/unified_model_detail.html`
- Shows input form based on `features_description` from database
- Shows model accuracy, code snippet, dataset sample
- User fills form and clicks "Predict"

---

### Step 3: User Submits Prediction Form
**Frontend JavaScript** (in `unified_model_detail.html`):
```javascript
function submitPrediction() {
    // Collect all input values from form
    const features = {
        age: document.getElementById('age').value,
        sex: document.getElementById('sex').value,
        cp: document.getElementById('cp').value,
        // ... all features
    };
    
    // Send AJAX POST request to backend
    fetch(`/predict/${problemId}/`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ features: features })
    })
    .then(response => response.json())
    .then(data => {
        // Display prediction results
        displayPrediction(data.prediction);
        displayRecommendations(data.recommendations);
        displayFeatureImportance(data.feature_importance);
    });
}
```

---

## 🔧 PART 4: BACKEND PREDICTION FLOW

### Main Prediction Function
**File**: `/classifier/views.py` → `predict_view()`

**URL**: `POST /predict/<problem_id>/`

### 🔄 Complete Flow Breakdown:

#### Step 1: Receive Input Data
```python
@csrf_exempt
@login_required
def predict_view(request, problem_id):
    # Parse JSON data from frontend
    data = json.loads(request.body)
    features = data.get('features', {})
    
    # Example features received:
    # {
    #     "age": 63,
    #     "sex": 1,
    #     "cp": 3,
    #     "trestbps": 145,
    #     ...
    # }
```

#### Step 2: Load Model Info from Database
```python
    # Get problem statement from database
    problem = ProblemStatement.objects.get(id=problem_id)
    
    # Now we know:
    # - problem.model_file = "classifier/models1/heart_disease_svm_model.pkl"
    # - problem.scaler_file = "classifier/models1/scalerheart.pkl"
    # - problem.model_type = "Classification"
    # - problem.features_description = {"age": "...", "sex": "...", ...}
```

#### Step 3: Load Model & Scaler from Disk
```python
    import pickle
    import joblib
    from django.conf import settings
    
    # Build full path to model file
    model_path = os.path.join(settings.BASE_DIR, problem.model_file)
    # Full path: /Users/mohammadabdullah/.../classifier/models1/heart_disease_svm_model.pkl
    
    # Load model (try multiple formats)
    if model_path.endswith('.pkl'):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)  # ✅ Your trained model is now loaded!
    
    # Load scaler
    scaler_path = os.path.join(settings.BASE_DIR, problem.scaler_file)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)  # ✅ Your scaler is now loaded!
```

#### Step 4: Prepare Input Data
```python
    # Get feature order from database
    feature_order = list(problem.features_description.keys())
    # ['age', 'sex', 'cp', 'trestbps', 'chol', ...]
    
    # Create input array in correct order
    input_data = [float(features.get(feature, 0)) for feature in feature_order]
    # [63.0, 1.0, 3.0, 145.0, 233.0, ...]
```

#### Step 5: Scale Input Data
```python
    # Scale the input using loaded scaler
    input_scaled = scaler.transform([input_data])
    # Input is now normalized, just like during training!
```

#### Step 6: Make Prediction
```python
    # Use loaded model to make prediction
    prediction = model.predict(input_scaled)[0]
    # Classification: Returns 0 or 1
    # Regression: Returns a number (e.g., 23.5 minutes)
    
    # For classification, also get probability
    if hasattr(model, 'predict_proba'):
        prediction_proba = model.predict_proba(input_scaled)[0]
        # [0.15, 0.85] = 15% no disease, 85% disease
```

#### Step 7: Calculate Feature Importance
```python
    # Shows which features influenced the prediction most
    feature_importance = calculate_feature_importance(
        model, scaler, input_data, feature_names, prediction, problem.model_type
    )
    
    # Returns:
    # [
    #     ('cp', {'importance': 0.25, 'value': 3, 'contribution': 'positive'}),
    #     ('age', {'importance': 0.20, 'value': 63, 'contribution': 'positive'}),
    #     ...
    # ]
```

#### Step 8: Generate Recommendations (AI-Powered)
```python
    # Get personalized recommendations from Perplexity AI
    recommendations = get_perplexity_recommendations(
        prediction, features, problem.title, problem.model_type
    )
```

---

## 🤖 PART 5: PERPLEXITY AI INTEGRATION

### Where API is Stored & Used
**File**: `/classifier/views.py` → `get_perplexity_recommendations()`

```python
def get_perplexity_recommendations(prediction, features, problem_title, model_type):
    # 🔑 API KEY - Stored directly in code
    api_key = "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"
    
    # Build prompt based on prediction
    if model_type.lower() == 'regression':
        # For Uber ETA prediction
        prompt = f"""Based on an Uber ETA prediction showing {eta_minutes} minutes,
        provide specific travel recommendations for a {distance} km trip..."""
    else:
        # For Heart Disease prediction
        prompt = f"""Based on a Heart Disease prediction showing {risk_level},
        provide health recommendations for a {age}-year-old patient..."""
    
    # Call Perplexity AI API
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': 'sonar-pro',  # Perplexity AI model
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 300
    }
    
    # Make HTTP POST request to Perplexity API
    response = requests.post(
        'https://api.perplexity.ai/chat/completions',
        headers=headers,
        json=data,
        timeout=15
    )
    
    if response.status_code == 200:
        result = response.json()
        # Extract AI-generated recommendations
        return result['choices'][0]['message']['content']
    else:
        # Fallback to hardcoded recommendations if API fails
        return get_fallback_recommendations(prediction, features, model_type)
```

### Fallback Recommendations
If Perplexity API fails, system uses pre-written recommendations:

```python
def get_fallback_recommendations(prediction, features, model_type):
    if prediction == 1:  # High risk
        recommendations = [
            "⚠️ Immediate Actions Required:",
            "• Schedule appointment with cardiologist",
            "• Monitor blood pressure daily",
            "• Reduce cholesterol through diet",
            ...
        ]
    else:  # Low risk
        recommendations = [
            "✅ Good News! Low Risk Detected",
            "• Continue regular physical activity",
            "• Maintain balanced diet",
            ...
        ]
    return "\n".join(recommendations)
```

---

## 📤 PART 6: RETURNING RESULTS TO FRONTEND

### Backend Response
```python
    # Return JSON response to frontend
    return JsonResponse({
        'prediction': prediction_value,        # 0 or 1 (or regression value)
        'feature_importance': feature_importance,  # Which features mattered most
        'recommendations': recommendations     # AI-generated or fallback advice
    }, status=200)
```

### Frontend Receives & Displays
**JavaScript** (in `unified_model_detail.html`):

```javascript
.then(data => {
    // 1. Display Prediction
    displayPrediction(data.prediction);
    // Shows: "⚠️ High Risk" or "✅ Low Risk"
    
    // 2. Display Feature Importance
    displayFeatureImportance(data.feature_importance);
    // Shows bar chart of which features influenced prediction
    
    // 3. Display Recommendations
    displayRecommendations(data.recommendations);
    // Shows AI-generated or fallback health/travel advice
})
```

---

## 📊 COMPLETE FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────┐
│                    1. MODEL TRAINING (Google Colab)                 │
├─────────────────────────────────────────────────────────────────────┤
│  • Train model with your dataset                                    │
│  • Save as PKL file: model.pkl, scaler.pkl                         │
│  • Download to your computer                                        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    2. UPLOAD TO DJANGO PROJECT                      │
├─────────────────────────────────────────────────────────────────────┤
│  Upload PKL files to: /classifier/models1/                          │
│    ├── heart_disease_model.pkl                                      │
│    └── scaler.pkl                                                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              3. REGISTER IN DATABASE (ProblemStatement)             │
├─────────────────────────────────────────────────────────────────────┤
│  File: /classifier/models.py                                        │
│  Fields:                                                            │
│    • title = "Heart Disease Prediction"                            │
│    • model_file = "classifier/models1/heart_disease_model.pkl"     │
│    • scaler_file = "classifier/models1/scaler.pkl"                 │
│    • features_description = {"age": "...", "sex": "...", ...}      │
│    • model_type = "Classification"                                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   4. USER VISITS WEBSITE                            │
├─────────────────────────────────────────────────────────────────────┤
│  URL: /models/                                                      │
│  View: models1_view() → templates/models1.html                      │
│  Shows: All available models in cards                               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│               5. USER CLICKS MODEL (e.g., Heart Disease)            │
├─────────────────────────────────────────────────────────────────────┤
│  URL: /model/1/                                                     │
│  View: model_detail_view() → unified_model_detail.html             │
│  Shows:                                                             │
│    • Input form (age, sex, cp, trestbps, ...)                      │
│    • Model info (accuracy, code snippet)                           │
│    • Dataset samples                                                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                6. USER FILLS FORM & CLICKS "PREDICT"                │
├─────────────────────────────────────────────────────────────────────┤
│  Frontend JavaScript collects input:                                │
│    features = {age: 63, sex: 1, cp: 3, trestbps: 145, ...}         │
│                                                                     │
│  AJAX POST Request:                                                 │
│    URL: /predict/1/                                                 │
│    Body: {"features": {...}}                                        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    7. BACKEND RECEIVES REQUEST                      │
├─────────────────────────────────────────────────────────────────────┤
│  File: /classifier/views.py                                         │
│  Function: predict_view(request, problem_id)                        │
│                                                                     │
│  Steps:                                                             │
│    A. Get ProblemStatement from database (id=1)                     │
│    B. Load model: pickle.load('classifier/models1/model.pkl')       │
│    C. Load scaler: pickle.load('classifier/models1/scaler.pkl')     │
│    D. Prepare input: [63, 1, 3, 145, ...]                          │
│    E. Scale input: scaler.transform(input_data)                     │
│    F. Predict: prediction = model.predict(input_scaled)             │
│    G. Calculate feature importance                                  │
│    H. Get AI recommendations from Perplexity API                    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              8. CALL PERPLEXITY AI FOR RECOMMENDATIONS              │
├─────────────────────────────────────────────────────────────────────┤
│  Function: get_perplexity_recommendations()                         │
│  API Key: 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    │
│  API URL: https://api.perplexity.ai/chat/completions                │
│                                                                     │
│  Request:                                                           │
│    {                                                                │
│      "model": "sonar-pro",                                          │
│      "messages": [{                                                 │
│        "role": "user",                                              │
│        "content": "Based on heart disease prediction..."           │
│      }]                                                             │
│    }                                                                │
│                                                                     │
│  Response:                                                          │
│    AI-generated health recommendations                              │
│    OR                                                               │
│    Fallback hardcoded recommendations (if API fails)                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   9. BACKEND RETURNS JSON RESPONSE                  │
├─────────────────────────────────────────────────────────────────────┤
│  {                                                                  │
│    "prediction": 1,                                                 │
│    "feature_importance": [                                          │
│      {"cp": {"importance": 0.25, "value": 3}},                     │
│      {"age": {"importance": 0.20, "value": 63}},                   │
│      ...                                                            │
│    ],                                                               │
│    "recommendations": "⚠️ Immediate Actions Required:..."          │
│  }                                                                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                10. FRONTEND DISPLAYS RESULTS                        │
├─────────────────────────────────────────────────────────────────────┤
│  JavaScript receives response and displays:                         │
│                                                                     │
│  ┌─────────────────────────────────────────┐                       │
│  │  🎯 Prediction Result                   │                       │
│  │  ⚠️ High Risk Detected (85% confidence)│                       │
│  └─────────────────────────────────────────┘                       │
│                                                                     │
│  ┌─────────────────────────────────────────┐                       │
│  │  📊 Feature Importance                  │                       │
│  │  • Chest Pain (cp): 25% ████████        │                       │
│  │  • Age: 20% ██████                      │                       │
│  │  • Blood Pressure: 15% █████            │                       │
│  └─────────────────────────────────────────┘                       │
│                                                                     │
│  ┌─────────────────────────────────────────┐                       │
│  │  💡 AI Recommendations                  │                       │
│  │  ⚠️ Immediate Actions Required:         │                       │
│  │  • Schedule cardiologist appointment    │                       │
│  │  • Monitor blood pressure daily         │                       │
│  │  • Reduce cholesterol intake...         │                       │
│  └─────────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ FILE STRUCTURE & RESPONSIBILITIES

```
ML_AT_FINGERTIPS/
│
├── classifier/
│   ├── models.py                          ← DATABASE MODEL (stores model metadata)
│   │   └── ProblemStatement               ← Links to your PKL files
│   │       ├── model_file                 ← Path to PKL file
│   │       ├── scaler_file                ← Path to scaler PKL
│   │       └── features_description       ← Feature names & descriptions
│   │
│   ├── views.py                           ← BACKEND LOGIC (prediction happens here)
│   │   ├── models1_view()                 ← Shows all models
│   │   ├── model_detail_view()            ← Shows model detail page
│   │   ├── predict_view()                 ← 🔥 MAIN PREDICTION FUNCTION
│   │   ├── calculate_feature_importance() ← Calculates which features matter
│   │   └── get_perplexity_recommendations() ← 🤖 Calls Perplexity AI API
│   │
│   ├── urls.py                            ← URL ROUTING
│   │   ├── /models/                       ← List all models
│   │   ├── /model/<id>/                   ← Model detail page
│   │   └── /predict/<id>/                 ← Prediction endpoint (POST)
│   │
│   ├── templates/
│   │   ├── models1.html                   ← Shows all models (cards)
│   │   └── unified_model_detail.html      ← Model detail & prediction form
│   │       ├── Input form (dynamic)       ← Based on features_description
│   │       ├── submitPrediction()         ← JavaScript: sends AJAX request
│   │       ├── displayPrediction()        ← Shows prediction result
│   │       └── displayRecommendations()   ← Shows AI recommendations
│   │
│   └── models1/                           ← 📦 YOUR PKL FILES GO HERE
│       ├── heart_disease_svm_model.pkl    ← Your trained model
│       ├── scalerheart.pkl                ← Your scaler
│       ├── deepeta_nyc_taxi.h5            ← Uber ETA model (TensorFlow)
│       ├── deepeta_assets.joblib          ← Uber scaler
│       └── secondhandcarprice.pkl         ← Car price model
│
└── ml_at_fingertips/
    └── settings.py                        ← BASE_DIR defined here (for file paths)
```

---

## 🔍 KEY FILES EXPLAINED

### 1. `/classifier/models.py` - Database Model
**Purpose**: Stores metadata about your ML models
- Links to PKL file locations
- Stores feature names
- Stores accuracy scores
- Stores model type (Classification/Regression)

### 2. `/classifier/views.py` - Backend Logic
**Purpose**: Handles all prediction logic
- Loads PKL files from disk
- Processes input data
- Makes predictions
- Calls Perplexity AI
- Returns JSON results

**Key Function**: `predict_view()`
- Line 138-310: Main prediction logic
- Loads model and scaler
- Scales input
- Makes prediction
- Returns JSON response

### 3. `/classifier/templates/unified_model_detail.html` - Frontend
**Purpose**: Shows model interface to users
- Dynamic form generation (based on features_description)
- JavaScript for AJAX prediction requests
- Display functions for results

### 4. `/classifier/models1/` - Model Storage
**Purpose**: Stores all your PKL files
- Upload your models here
- Django loads them from here during prediction

---

## 🚀 ADDING A NEW MODEL - COMPLETE CHECKLIST

### ✅ Step 1: Train Model in Colab
```python
# Train and save
model = RandomForestClassifier()
model.fit(X_train, y_train)

with open('my_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('my_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

### ✅ Step 2: Upload PKL Files
```
Upload to: /classifier/models1/
  ├── my_model.pkl
  └── my_scaler.pkl
```

### ✅ Step 3: Add to Database
Edit `/classifier/management/commands/populate_models.py`:

```python
ProblemStatement.objects.create(
    title="My New Prediction Model",
    model_file="classifier/models1/my_model.pkl",       # ← Your model path
    scaler_file="classifier/models1/my_scaler.pkl",     # ← Your scaler path
    model_type="Classification",                         # or "Regression"
    features_description={
        "feature1": "Description 1",
        "feature2": "Description 2",
        # ... all your features
    },
    accuracy_scores={"Random Forest": 0.92},
    selected_model="Random Forest",
    code_snippet="# Your training code here"
)
```

Run: `python3 ml_at_fingertips/manage.py populate_models`

### ✅ Step 4: Test Prediction
1. Visit `http://localhost:8000/models/`
2. Find your new model card
3. Click "Try"
4. Fill form and test prediction

---

## 🎓 SUMMARY

### Where Things Are Stored:
1. **PKL Files**: `/classifier/models1/` (uploaded by you)
2. **Model Metadata**: Database `ProblemStatement` table
3. **Perplexity API Key**: Hardcoded in `views.py` line 636
4. **Prediction Logic**: `/classifier/views.py` → `predict_view()`
5. **Frontend Forms**: `/classifier/templates/unified_model_detail.html`

### How Prediction Works:
1. User fills form → JavaScript collects data
2. AJAX POST to `/predict/<id>/`
3. Backend loads PKL model from disk
4. Backend scales input data
5. Backend calls `model.predict()`
6. Backend calls Perplexity AI for recommendations
7. Backend returns JSON response
8. Frontend displays results

### Data Flow:
```
User Input → JavaScript → AJAX POST → Django View → Load PKL → 
Scale Data → Model.predict() → Perplexity AI → JSON Response → 
Display Results
```

### Key Technologies:
- **Django**: Web framework
- **Pickle/Joblib**: Model serialization
- **Scikit-learn**: ML models & preprocessing
- **TensorFlow**: Deep learning models (Uber ETA)
- **Perplexity AI**: Recommendation generation
- **JavaScript/AJAX**: Frontend-backend communication

---

## 📞 TROUBLESHOOTING

### Model Not Loading?
- Check PKL file path in database matches actual file location
- Verify file permissions (should be readable)

### Wrong Predictions?
- Ensure features are in correct order (match training order)
- Check scaler is applied before prediction
- Verify input data types (all numeric)

### API Recommendations Failing?
- Check Perplexity API key is valid
- System will automatically use fallback recommendations if API fails

---

**Created**: October 24, 2025  
**Author**: ML_AT_FINGERTIPS Documentation  
**Version**: 1.0
