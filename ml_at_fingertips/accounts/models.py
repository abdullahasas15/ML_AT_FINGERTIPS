from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    FAVORITE_CHOICES = [
        ('classification', 'Classification'),
        ('regression', 'Regression'),
        ('random_trees', 'Random Trees'),
    ]
    favorite_ml_type = models.CharField(max_length=32, choices=FAVORITE_CHOICES, blank=True)
    # email and password are already included in AbstractUser
