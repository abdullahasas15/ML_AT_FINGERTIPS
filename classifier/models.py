from django.db import models

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
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.title
    
    class Meta:
        verbose_name = "Problem Statement"
        verbose_name_plural = "Problem Statements"
