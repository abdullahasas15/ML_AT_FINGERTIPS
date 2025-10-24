from django.core.management.base import BaseCommand
from classifier.models import ProblemStatement
import json

class Command(BaseCommand):
    help = 'Add Car Price Prediction problem statement'

    def handle(self, *args, **kwargs):
        # Read car names from file
        try:
            with open('carnames', 'r') as f:
                lines = f.readlines()
                car_names = [line.strip() for line in lines if line.strip() and not line.startswith('Unique')]
        except FileNotFoundError:
            car_names = []
            self.stdout.write(self.style.WARNING('carnames file not found, using empty list'))

        problem, created = ProblemStatement.objects.update_or_create(
            title="Can we predict second-hand car prices using ML?",
            defaults={
                'description': 'Discover how machine learning models analyze car features like age, kilometers driven, fuel type, and transmission to accurately predict second-hand car market prices.',
                'model_type': 'Regression',
                'selected_model': 'LinearRegression',
                'model_file': 'classifier/models1/secondhandcarprice.pkl',
                'scaler_file': '',  # No scaler file for this model
                'features_description': {
                    'age': 'Age of the car in years',
                    'km_driven': 'Total kilometers driven',
                    'name': 'Car model name',
                    'fuel': 'Fuel type (Petrol/Diesel/CNG/LPG)',
                    'seller_type': 'Seller type (Individual/Dealer/Trustmark Dealer)',
                    'transmission': 'Transmission type (Manual/Automatic)',
                    'owner': 'Number of previous owners (First Owner/Second Owner/etc.)'
                },
                'dataset_sample': [
                    {
                        'age': 18,
                        'km_driven': 70000,
                        'name': 'Maruti 800 AC',
                        'fuel': 'Petrol',
                        'seller_type': 'Individual',
                        'transmission': 'Manual',
                        'owner': 'First Owner'
                    },
                    {
                        'age': 18,
                        'km_driven': 50000,
                        'name': 'Maruti Wagon R LXI Minor',
                        'fuel': 'Petrol',
                        'seller_type': 'Individual',
                        'transmission': 'Manual',
                        'owner': 'First Owner'
                    }
                ],
                'accuracy_scores': {
                    'LinearRegression': 0.5635,  # R2 score
                    'DecisionTree': 0.4821,
                    'RandomForest': 0.5206,
                    'XGBoost': 0.5128
                },
                'model_info': json.dumps({
                    'algorithm': 'Linear Regression',
                    'features': ['age', 'km_driven', 'name', 'fuel', 'seller_type', 'transmission', 'owner'],
                    'target': 'selling_price',
                    'r2_score': 0.5635,
                    'rmse': 439242.57,
                    'car_names': car_names
                }),
                'code_snippet': '''# Car Price Prediction Model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load dataset
df = pd.read_csv('car_data.csv')

# Encode categorical variables
le_name = LabelEncoder()
le_fuel = LabelEncoder()
le_seller = LabelEncoder()
le_transmission = LabelEncoder()
le_owner = LabelEncoder()

df['name_encoded'] = le_name.fit_transform(df['name'])
df['fuel_encoded'] = le_fuel.fit_transform(df['fuel'])
df['seller_type_encoded'] = le_seller.fit_transform(df['seller_type'])
df['transmission_encoded'] = le_transmission.fit_transform(df['transmission'])
df['owner_encoded'] = le_owner.fit_transform(df['owner'])

# Features and target
X = df[['age', 'km_driven', 'name_encoded', 'fuel_encoded', 
        'seller_type_encoded', 'transmission_encoded', 'owner_encoded']]
y = df['selling_price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'R2 Score: {r2:.4f}')
print(f'RMSE: {rmse:.2f}')
'''
            }
        )

        if created:
            self.stdout.write(self.style.SUCCESS(f'Successfully created problem: {problem.title}'))
        else:
            self.stdout.write(self.style.SUCCESS(f'Successfully updated problem: {problem.title}'))
