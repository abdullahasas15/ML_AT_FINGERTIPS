from django import forms
from .models import ModelContribution
import json


class ModelContributionForm(forms.ModelForm):
    """Form for users to submit their ML model contributions"""
    
    # Additional text fields for JSON inputs
    dataset_sample_text = forms.CharField(
        widget=forms.Textarea(attrs={
            'rows': 4,
            'placeholder': 'Paste sample data as JSON array, e.g., [{"feature1": value1, "feature2": value2}, ...]'
        }),
        required=False,
        help_text="Paste sample data rows as JSON (optional if uploading dataset file)"
    )
    
    features_description_text = forms.CharField(
        widget=forms.Textarea(attrs={
            'rows': 4,
            'placeholder': '{"feature1": "Description of feature 1", "feature2": "Description of feature 2", ...}'
        }),
        required=False,
        help_text="Describe your input features as JSON object"
    )
    
    accuracy_scores_text = forms.CharField(
        widget=forms.Textarea(attrs={
            'rows': 3,
            'placeholder': '{"accuracy": 0.95, "precision": 0.93, "recall": 0.94, "f1_score": 0.935}'
        }),
        required=False,
        help_text="Model performance metrics as JSON"
    )
    
    class Meta:
        model = ModelContribution
        fields = [
            'title',
            'description',
            'model_type',
            'model_file',
            'scaler_file',
            'dataset_file',
            'code_file',
            'dataset_sample_text',
            'features_description_text',
            'code_snippet',
            'selected_model',
            'accuracy_scores_text',
            'problem_statement_detail',
            'approach_explanation',
            'preprocessing_steps',
            'model_architecture',
            'model_info',
            'requirements',
        ]
        
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., House Price Prediction Model'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Brief overview of your ML project...'
            }),
            'model_type': forms.Select(attrs={'class': 'form-control'}),
            'model_file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.pkl,.joblib,.h5,.pt,.pth,.sav'
            }),
            'scaler_file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.pkl,.joblib,.sav'
            }),
            'dataset_file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.csv,.xlsx,.xls,.json'
            }),
            'code_file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.py,.ipynb'
            }),
            'code_snippet': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 10,
                'placeholder': 'Paste your model training/prediction code here...'
            }),
            'selected_model': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., Random Forest, XGBoost, LSTM'
            }),
            'problem_statement_detail': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 5,
                'placeholder': 'Explain the problem in detail for beginners...'
            }),
            'approach_explanation': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 5,
                'placeholder': 'Explain your approach and why you chose this model...'
            }),
            'preprocessing_steps': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 5,
                'placeholder': 'Describe data preprocessing and feature engineering steps...'
            }),
            'model_architecture': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 5,
                'placeholder': 'Describe model architecture, parameters, hyperparameters...'
            }),
            'model_info': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Any additional information about the model...'
            }),
            'requirements': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'scikit-learn==1.3.0\npandas==2.0.0\nnumpy==1.24.0'
            }),
        }
    
    def clean_model_file(self):
        """Validate model file extension"""
        model_file = self.cleaned_data.get('model_file')
        if model_file:
            valid_extensions = ['.pkl', '.joblib', '.h5', '.pt', '.pth', '.sav']
            ext = model_file.name.split('.')[-1].lower()
            if f'.{ext}' not in valid_extensions:
                raise forms.ValidationError(
                    f'Invalid file type. Allowed: {", ".join(valid_extensions)}'
                )
        return model_file
    
    def clean_dataset_sample_text(self):
        """Validate and convert dataset sample text to JSON"""
        text = self.cleaned_data.get('dataset_sample_text', '').strip()
        if text:
            try:
                data = json.loads(text)
                if not isinstance(data, list):
                    raise forms.ValidationError('Dataset sample must be a JSON array')
                return data
            except json.JSONDecodeError:
                raise forms.ValidationError('Invalid JSON format')
        return None
    
    def clean_features_description_text(self):
        """Validate and convert features description to JSON"""
        text = self.cleaned_data.get('features_description_text', '').strip()
        if text:
            try:
                data = json.loads(text)
                if not isinstance(data, dict):
                    raise forms.ValidationError('Features description must be a JSON object')
                return data
            except json.JSONDecodeError:
                raise forms.ValidationError('Invalid JSON format')
        return None
    
    def clean_accuracy_scores_text(self):
        """Validate and convert accuracy scores to JSON"""
        text = self.cleaned_data.get('accuracy_scores_text', '').strip()
        if text:
            try:
                data = json.loads(text)
                if not isinstance(data, dict):
                    raise forms.ValidationError('Accuracy scores must be a JSON object')
                return data
            except json.JSONDecodeError:
                raise forms.ValidationError('Invalid JSON format')
        return None
    
    def save(self, commit=True):
        """Save the form and populate JSON fields from text inputs"""
        instance = super().save(commit=False)
        
        # Convert text fields to JSON fields
        if self.cleaned_data.get('dataset_sample_text'):
            instance.dataset_sample = self.cleaned_data['dataset_sample_text']
        
        if self.cleaned_data.get('features_description_text'):
            instance.features_description = self.cleaned_data['features_description_text']
        
        if self.cleaned_data.get('accuracy_scores_text'):
            instance.accuracy_scores = self.cleaned_data['accuracy_scores_text']
        
        if commit:
            instance.save()
        
        return instance


class ContributionReviewForm(forms.ModelForm):
    """Form for admin to review and approve/reject contributions"""
    
    class Meta:
        model = ModelContribution
        fields = ['status', 'admin_notes']
        widgets = {
            'status': forms.Select(attrs={'class': 'form-control'}),
            'admin_notes': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Provide feedback to the contributor...'
            }),
        }
