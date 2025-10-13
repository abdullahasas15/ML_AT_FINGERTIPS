from django.core.management.base import BaseCommand
from regressor.models import ProblemStatement


class Command(BaseCommand):
    help = 'Populate the database with Uber time prediction problem statements'

    def handle(self, *args, **options):
        # Uber Time Prediction Problem Statement
        uber_eta_data = {
            'title': 'How Uber Calculates Time to Reach Destination - Are You Curious?',
            'description': 'Ever wondered how Uber predicts your arrival time so accurately? Our Deep ETA model uses advanced machine learning to analyze traffic patterns, distance, time of day, and route conditions to predict travel time. Discover how artificial intelligence powers modern transportation and logistics!',
            'dataset_sample': [
                {
                    'distance': 2.5,
                    'pickup_latitude': 40.7589,
                    'pickup_longitude': -73.9851,
                    'dropoff_latitude': 40.7505,
                    'dropoff_longitude': -73.9934,
                    'passenger_count': 2,
                    'pickup_hour': 14,
                    'pickup_day': 3,
                    'pickup_month': 6,
                    'trip_duration': 15.5
                },
                {
                    'distance': 8.2,
                    'pickup_latitude': 40.7128,
                    'pickup_longitude': -74.0060,
                    'dropoff_latitude': 40.6892,
                    'dropoff_longitude': -74.0445,
                    'passenger_count': 1,
                    'pickup_hour': 18,
                    'pickup_day': 5,
                    'pickup_month': 9,
                    'trip_duration': 28.3
                }
            ],
            'model_type': 'Regression',
            'model_file': 'regressor/models2/deepeta_nyc_taxi.h5',
            'scaler_file': 'regressor/models2/deepeta_assets.joblib',
            'features_description': {
                'distance': 'Distance between pickup and dropoff locations in kilometers',
                'pickup_latitude': 'Latitude coordinate of pickup location',
                'pickup_longitude': 'Longitude coordinate of pickup location',
                'dropoff_latitude': 'Latitude coordinate of dropoff location',
                'dropoff_longitude': 'Longitude coordinate of dropoff location',
                'passenger_count': 'Number of passengers in the trip',
                'pickup_hour': 'Hour of day when trip started (0-23)',
                'pickup_day': 'Day of week when trip started (1-7)',
                'pickup_month': 'Month when trip started (1-12)'
            },
            'accuracy_scores': {
                'Deep ETA Model': 0.89,
                'Random Forest': 0.82,
                'Linear Regression': 0.74,
                'XGBoost': 0.85
            },
            'selected_model': 'Deep ETA Model',
            'model_info': '''Deep ETA (Estimated Time of Arrival) Model - How It Works:

🚗 What is Deep ETA?
A Deep ETA model predicts "How long will it take for something (like delivery/ride) to reach the destination?" Traditional ETA models used simple formulas like distance ÷ average speed, but they failed in real-world scenarios because of traffic variations, signal delays, road closures, weather, driver's behavior patterns, pickup/delivery preparation time, vehicle type, road structure, and peak hours.

🧠 How Deep ETA Model Works:

✅ Step 1: Input Features (Data Collection)
The model collects multiple real-time and historical signals:
• Geo Features: Latitude, longitude of source & destination
• Route Features: Distance, road type, number of turns, past delays on that road
• Time Features: Time of day, weekday/weekend, festival, peak traffic hour
• Traffic Signals: Live congestion score, historical average speed on that route
• Driver/Vehicle Features: Driver speed profile, vehicle type (bike/car/truck)
• Context Features: Weather, rain, roadblocks, event nearby

✅ Step 2: Sequence Modeling (Path as a Sequence)
A route is broken into multiple segments (like map tiles/grids). Each small road segment has features — speed pattern, stop-light probability, congestion chance. These segments are fed into models like LSTM / GRU / Transformer, because ETA prediction is a time-sequence prediction problem.

✅ Step 3: Deep Neural Network Architecture
Most Deep ETA models use:
• Embedding Layer: Converts categorical data (like area codes, driver IDs) into dense vectors
• LSTM/Transformer Layer: Understands movement sequence through route segments
• Attention Mechanism: Identifies critical segments that contribute most to delay
• Fully Connected (Dense) Layers: Combines all insights to give final ETA prediction

✅ Step 4: Continuous Learning & Feedback Loop
After each delivery/trip, the system compares: Predicted ETA vs Actual Arrival Time. The error (loss) is backpropagated. Model automatically self-improves based on real-world data.

⚡ Why Deep ETA is Powerful:
Traditional ETA: Only distance & speed, One formula fits all, Errors during traffic/peak time, Static
Deep ETA: Considers 50+ real-world signals, Personalized ETA per driver/road/time, Learns traffic patterns using history, Self-learning and updating daily

💡 Real-World Applications:
Companies like Uber, Google Maps, Amazon Logistics, Swiggy, Zomato, Blinkit use Deep ETA models for accurate time predictions that save millions of hours and improve user experience.''',
            'code_snippet': '''import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Deep ETA Model Implementation
def create_deep_eta_model(input_shape):
    """Create a Deep ETA model using TensorFlow/Keras"""
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # Hidden layers
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        
        # Output layer (regression)
        tf.keras.layers.Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

# Feature Engineering for ETA Prediction
def engineer_features(df):
    """Engineer features for ETA prediction"""
    # Calculate distance using Haversine formula
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in kilometers
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    # Create distance feature
    df['distance'] = haversine_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    
    # Time-based features
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_day'] = df['pickup_datetime'].dt.dayofweek
    df['pickup_month'] = df['pickup_datetime'].dt.month
    
    # Traffic patterns
    df['is_peak_hour'] = df['pickup_hour'].apply(lambda x: 1 if x in [7,8,9,17,18,19] else 0)
    df['is_weekend'] = df['pickup_day'].apply(lambda x: 1 if x >= 5 else 0)
    
    return df

# Training the Deep ETA Model
def train_deep_eta_model():
    # Load and preprocess data
    df = pd.read_csv('nyc_taxi_data.csv')
    df = engineer_features(df)
    
    # Select features
    feature_cols = [
        'distance', 'pickup_latitude', 'pickup_longitude',
        'dropoff_latitude', 'dropoff_longitude', 'passenger_count',
        'pickup_hour', 'pickup_day', 'pickup_month'
    ]
    
    X = df[feature_cols]
    y = df['trip_duration']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    model = create_deep_eta_model(X_train_scaled.shape[1])
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=100,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate model
    predictions = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    
    print(f"Mean Absolute Error: {mae:.2f} minutes")
    print(f"Mean Squared Error: {mse:.2f}")
    
    # Save model and scaler
    model.save('deepeta_nyc_taxi.h5')
    joblib.dump(scaler, 'deepeta_assets.joblib')
    
    return model, scaler

# Usage
if __name__ == "__main__":
    model, scaler = train_deep_eta_model()
    print("Deep ETA model trained and saved successfully!")'''
        }

        # Create or update the problem statement
        problem, created = ProblemStatement.objects.get_or_create(
            title=uber_eta_data['title'],
            defaults=uber_eta_data
        )

        if created:
            self.stdout.write(
                self.style.SUCCESS(f'Successfully created problem statement: {problem.title}')
            )
        else:
            self.stdout.write(
                self.style.WARNING(f'Problem statement already exists: {problem.title}')
            )
