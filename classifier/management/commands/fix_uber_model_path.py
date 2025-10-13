from django.core.management.base import BaseCommand
from classifier.models import ProblemStatement

class Command(BaseCommand):
    help = 'Fix Uber ETA model and scaler file paths in classifier_problemstatement table.'

    def handle(self, *args, **options):
        problems = ProblemStatement.objects.filter(title__icontains='Uber')
        count = 0
        for problem in problems:
            problem.model_file = 'classifier/models1/deepeta_nyc_taxi.h5'
            problem.scaler_file = 'classifier/models1/deepeta_assets.joblib'
            problem.save()
            count += 1
        self.stdout.write(self.style.SUCCESS(f'Updated {count} Uber ETA problem(s) with correct model/scaler path.'))