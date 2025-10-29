from django.db import models
from django.conf import settings
import os

# Create your models here.

class ProblemStatement(models.Model):
    id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=200)
    description = models.TextField()
    dataset_sample = models.JSONField(help_text="Sample data rows for display")
    model_type = models.CharField(max_length=100, default="Classification")
    model_file = models.CharField(max_length=200, help_text="Path to the main model file")
    scaler_file = models.CharField(max_length=200, help_text="Path to the scaler file", blank=True, null=True)
    features_description = models.JSONField(help_text="Description of input features")
    accuracy_scores = models.JSONField(help_text="Accuracy scores for different models")
    selected_model = models.CharField(max_length=100, help_text="The model currently being used")
    code_snippet = models.TextField(help_text="Python code for the model")
    model_info = models.TextField(help_text="Detailed information about the model", blank=True)
    
    # New fields for learning experience
    problem_statement_detail = models.TextField(help_text="Detailed problem explanation for beginners", blank=True, null=True)
    approach_explanation = models.TextField(help_text="Our approach and model selection reasoning", blank=True, null=True)
    preprocessing_steps = models.TextField(help_text="Data preprocessing and feature engineering techniques", blank=True, null=True)
    model_architecture = models.TextField(help_text="Model architecture, parameters, and hyperparameters", blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.title
    
    class Meta:
        verbose_name = "Problem Statement"
        verbose_name_plural = "Problem Statements"


class ModelContribution(models.Model):
    """User-submitted ML model contributions pending admin approval"""
    
    STATUS_CHOICES = [
        ('pending', 'Pending Review'),
        ('approved', 'Approved'),
        ('rejected', 'Rejected'),
    ]
    
    MODEL_TYPE_CHOICES = [
        ('Classification', 'Classification'),
        ('Regression', 'Regression'),
        ('Clustering', 'Clustering'),
        ('Deep Learning', 'Deep Learning'),
        ('NLP', 'Natural Language Processing'),
        ('Computer Vision', 'Computer Vision'),
        ('Other', 'Other'),
    ]
    
    # User and status information
    contributor = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='contributions')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    admin_notes = models.TextField(blank=True, help_text="Admin feedback or notes")
    
    # Basic model information
    title = models.CharField(max_length=200, help_text="Title of your ML project")
    description = models.TextField(help_text="Brief description of the problem and solution")
    model_type = models.CharField(max_length=100, choices=MODEL_TYPE_CHOICES, default="Classification")
    
    # File uploads
    model_file = models.FileField(
        upload_to='contributions/models/%Y/%m/',
        help_text="Upload your trained model file (.pkl, .joblib, .h5, etc.)"
    )
    scaler_file = models.FileField(
        upload_to='contributions/scalers/%Y/%m/',
        blank=True,
        null=True,
        help_text="Upload scaler/preprocessor file if applicable"
    )
    dataset_file = models.FileField(
        upload_to='contributions/datasets/%Y/%m/',
        blank=True,
        null=True,
        help_text="Upload sample dataset (CSV, Excel, etc.)"
    )
    code_file = models.FileField(
        upload_to='contributions/code/%Y/%m/',
        blank=True,
        null=True,
        help_text="Upload your Python code file (.py or .ipynb)"
    )
    
    # Data and features
    dataset_sample = models.JSONField(
        help_text="Sample data rows (will be auto-extracted from dataset_file or paste JSON)",
        blank=True,
        null=True
    )
    features_description = models.JSONField(
        help_text="Description of input features as JSON object",
        blank=True,
        null=True
    )
    
    # Model details
    code_snippet = models.TextField(
        help_text="Python code for model training/prediction",
        blank=True
    )
    selected_model = models.CharField(
        max_length=100,
        help_text="Model algorithm used (e.g., Random Forest, XGBoost, Neural Network)",
        blank=True
    )
    accuracy_scores = models.JSONField(
        help_text="Model performance metrics as JSON",
        blank=True,
        null=True
    )
    
    # Learning content
    problem_statement_detail = models.TextField(
        help_text="Detailed problem explanation for beginners",
        blank=True
    )
    approach_explanation = models.TextField(
        help_text="Your approach and model selection reasoning",
        blank=True
    )
    preprocessing_steps = models.TextField(
        help_text="Data preprocessing and feature engineering techniques",
        blank=True
    )
    model_architecture = models.TextField(
        help_text="Model architecture, parameters, and hyperparameters",
        blank=True
    )
    model_info = models.TextField(
        help_text="Additional model information",
        blank=True
    )
    
    # Dependencies
    requirements = models.TextField(
        help_text="Python packages required (one per line)",
        blank=True
    )
    
    # Timestamps
    submitted_at = models.DateTimeField(auto_now_add=True)
    reviewed_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.title} - {self.contributor.username} ({self.status})"
    
    def get_file_extension(self, file_field):
        """Get file extension from uploaded file"""
        if file_field:
            return os.path.splitext(file_field.name)[1].lower()
        return None
    
    class Meta:
        verbose_name = "Model Contribution"
        verbose_name_plural = "Model Contributions"
        ordering = ['-submitted_at']
