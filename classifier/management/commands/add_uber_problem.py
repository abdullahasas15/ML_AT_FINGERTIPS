from django.core.management.base import BaseCommand
from classifier.models import ProblemStatement
import json

class Command(BaseCommand):
    help = 'Add Uber ETA prediction problem to classifier_problemstatement table.'

    def handle(self, *args, **options):
        title = 'How Uber Calculates Time to Destination'
        description = (
            'Are you curious how Uber predicts the time to reach your destination? '
            'This model uses real NYC taxi trip data and advanced deep learning to estimate ETA based on pickup/dropoff, distance, time, and more.'
        )
        dataset_sample = [
            {
                'distance': 5.2,
                'pickup_latitude': 40.7589,
                'pickup_longitude': -73.9851,
                'dropoff_latitude': 40.7614,
                'dropoff_longitude': -73.9776,
                'passenger_count': 2,
                'pickup_hour': 14,
                'pickup_day': 3,
                'pickup_month': 6
            },
            {
                'distance': 2.8,
                'pickup_latitude': 40.7306,
                'pickup_longitude': -73.9352,
                'dropoff_latitude': 40.7411,
                'dropoff_longitude': -73.9897,
                'passenger_count': 1,
                'pickup_hour': 9,
                'pickup_day': 5,
                'pickup_month': 7
            }
        ]
        model_type = 'Regression'
        model_file = 'classifier/models1/deepeta_nyc_taxi.h5'
        scaler_file = 'classifier/models1/deepeta_assets.joblib'
        features_description = {
            'distance': 'Trip distance in kilometers',
            'pickup_latitude': 'Latitude of pickup location',
            'pickup_longitude': 'Longitude of pickup location',
            'dropoff_latitude': 'Latitude of dropoff location',
            'dropoff_longitude': 'Longitude of dropoff location',
            'passenger_count': 'Number of passengers',
            'pickup_hour': 'Hour of pickup (0-23)',
            'pickup_day': 'Day of week (1=Mon, 7=Sun)',
            'pickup_month': 'Month of year (1-12)'
        }
        accuracy_scores = {
            'DeepETA': 0.87,
            'RandomForest': 0.81
        }
        selected_model = 'DeepETA'
        code_snippet = (
            'import tensorflow as tf\nmodel = tf.keras.models.load_model("deepeta_nyc_taxi.h5")\n'\
            'import joblib\nscaler = joblib.load("deepeta_assets.joblib")\n'\
            'input_data = scaler.transform([[...features...]])\nprediction = model.predict(input_data)'
        )
        model_info = (
            'DeepETA is a deep learning model trained on thousands of NYC taxi trips. It uses features like distance, pickup/dropoff coordinates, time, and passenger count to predict ETA. '
            'The model leverages Keras and TensorFlow for high accuracy and real-time predictions.'
        )
        problem = ProblemStatement.objects.create(
            title=title,
            description=description,
            dataset_sample=dataset_sample,
            model_type=model_type,
            model_file=model_file,
            scaler_file=scaler_file,
            features_description=features_description,
            accuracy_scores=accuracy_scores,
            selected_model=selected_model,
            code_snippet=code_snippet,
            model_info=model_info
        )
        self.stdout.write(self.style.SUCCESS(f'Uber ETA problem added: {problem.title}'))
