from django.core.management.base import BaseCommand
from django.db import transaction
import os
import shutil

class Command(BaseCommand):
    help = 'Migrate data from regressor app to classifier app and clean up'

    def handle(self, *args, **options):
        self.stdout.write('Starting migration from regressor to classifier...')
        
        try:
            with transaction.atomic():
                # Copy model files from regressor/models2 to classifier/models1
                self.migrate_model_files()
                
                # Update existing ProblemStatement records to use classifier paths
                self.update_problem_statements()
                
                self.stdout.write(
                    self.style.SUCCESS('Successfully migrated from regressor to classifier!')
                )
                
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Migration failed: {str(e)}')
            )
            raise

    def migrate_model_files(self):
        """Copy model files from regressor/models2 to classifier/models1"""
        self.stdout.write('Migrating model files...')
        
        source_dir = 'regressor/models2'
        dest_dir = 'classifier/models1'
        
        files_to_copy = ['deepeta_assets.joblib', 'deepeta_nyc_taxi.h5', 'train.csv']
        
        for file in files_to_copy:
            source_path = os.path.join(source_dir, file)
            dest_path = os.path.join(dest_dir, file)
            
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
                self.stdout.write(f'  Copied {file}')
            else:
                self.stdout.write(f'  File {file} not found, skipping...')

    def update_problem_statements(self):
        """Update ProblemStatement records to use classifier paths"""
        self.stdout.write('Updating ProblemStatement records...')
        
        from classifier.models import ProblemStatement
        
        # Update regression models to use classifier paths
        regression_problems = ProblemStatement.objects.filter(model_type='Regression')
        
        for problem in regression_problems:
            # Update model file path
            if 'regressor/models2/' in problem.model_file:
                problem.model_file = problem.model_file.replace('regressor/models2/', 'classifier/models1/')
            
            # Update scaler file path if it exists
            if problem.scaler_file and 'regressor/models2/' in problem.scaler_file:
                problem.scaler_file = problem.scaler_file.replace('regressor/models2/', 'classifier/models1/')
            
            problem.save()
            self.stdout.write(f'  Updated problem: {problem.title}')
        
        self.stdout.write(f'Updated {regression_problems.count()} regression problems')
