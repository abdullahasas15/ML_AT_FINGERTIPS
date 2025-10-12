from django.core.management.base import BaseCommand
from classifier.models import ProblemStatement

class Command(BaseCommand):
    help = 'Populate the database with problem statements'

    def handle(self, *args, **options):
        # Heart Disease Risk Management Problem Statement
        heart_disease_data = {
            'title': 'Heart Disease Risk Management',
            'description': 'Predict the risk of heart disease based on various health parameters including age, blood pressure, cholesterol levels, and other medical indicators. This model helps in early detection and prevention of cardiovascular diseases.',
            'dataset_sample': [
                {
                    'age': 63,
                    'sex': 1,
                    'cp': 3,
                    'trestbps': 145,
                    'chol': 233,
                    'fbs': 1,
                    'restecg': 0,
                    'thalach': 150,
                    'exang': 0,
                    'oldpeak': 2.3,
                    'slope': 0,
                    'ca': 0,
                    'thal': 1,
                    'target': 0
                },
                {
                    'age': 37,
                    'sex': 1,
                    'cp': 2,
                    'trestbps': 130,
                    'chol': 250,
                    'fbs': 0,
                    'restecg': 1,
                    'thalach': 187,
                    'exang': 0,
                    'oldpeak': 3.5,
                    'slope': 0,
                    'ca': 0,
                    'thal': 2,
                    'target': 1
                }
            ],
            'model_type': 'Classification',
            'model_file': 'classifier/models1/heart_disease_svm_model.pkl',
            'scaler_file': 'classifier/models1/scalerheart.pkl',
            'features_description': {
                'age': 'Age in years',
                'sex': 'Sex (1 = male; 0 = female)',
                'cp': 'Chest pain type (0-3)',
                'trestbps': 'Resting blood pressure (mm Hg)',
                'chol': 'Serum cholesterol (mg/dl)',
                'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
                'restecg': 'Resting electrocardiographic results (0-2)',
                'thalach': 'Maximum heart rate achieved',
                'exang': 'Exercise induced angina (1 = yes; 0 = no)',
                'oldpeak': 'ST depression induced by exercise relative to rest',
                'slope': 'Slope of the peak exercise ST segment (0-2)',
                'ca': 'Number of major vessels colored by flourosopy (0-3)',
                'thal': 'Thalassemia (1-3)'
            },
            'accuracy_scores': {
                'SVM': 0.79,
                'Random Forest': 0.76,
                'Logistic Regression': 0.73
            },
            'selected_model': 'SVM',
            'code_snippet': '''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Create synthetic heart disease dataset
def create_heart_dataset():
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.normal(55, 15, n_samples).astype(int),
        'sex': np.random.choice([0, 1], n_samples),
        'cp': np.random.choice([0, 1, 2, 3], n_samples),
        'trestbps': np.random.normal(130, 20, n_samples).astype(int),
        'chol': np.random.normal(250, 50, n_samples).astype(int),
        'fbs': np.random.choice([0, 1], n_samples),
        'restecg': np.random.choice([0, 1, 2], n_samples),
        'thalach': np.random.normal(150, 25, n_samples).astype(int),
        'exang': np.random.choice([0, 1], n_samples),
        'oldpeak': np.random.exponential(1.5, n_samples),
        'slope': np.random.choice([0, 1, 2], n_samples),
        'ca': np.random.choice([0, 1, 2, 3], n_samples),
        'thal': np.random.choice([1, 2, 3], n_samples),
    }
    
    # Create target based on risk factors
    target = []
    for i in range(n_samples):
        risk_score = 0
        if data['age'][i] > 60: risk_score += 1
        if data['trestbps'][i] > 140: risk_score += 1
        if data['chol'][i] > 300: risk_score += 1
        if data['exang'][i] == 1: risk_score += 1
        if data['oldpeak'][i] > 2: risk_score += 1
        risk_score += np.random.choice([0, 1], p=[0.7, 0.3])
        target.append(1 if risk_score >= 3 else 0)
    
    data['target'] = target
    return pd.DataFrame(data)

# Create dataset and train model
df = create_heart_dataset()
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_model = SVC(kernel='rbf', random_state=42, probability=True)
svm_model.fit(X_train_scaled, y_train)

y_pred = svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Save model and scaler
pickle.dump(svm_model, open('heart_disease_svm_model.pkl', 'wb'))
pickle.dump(scaler, open('scalerheart.pkl', 'wb'))'''
        }

        # Create or update the problem statement
        problem, created = ProblemStatement.objects.get_or_create(
            title=heart_disease_data['title'],
            defaults=heart_disease_data
        )

        if created:
            self.stdout.write(
                self.style.SUCCESS(f'Successfully created problem statement: {problem.title}')
            )
        else:
            self.stdout.write(
                self.style.WARNING(f'Problem statement already exists: {problem.title}')
            )
