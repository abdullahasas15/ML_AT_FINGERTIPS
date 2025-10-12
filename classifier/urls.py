from django.urls import path
from . import views

urlpatterns = [
    path('models1/', views.models1_view, name='models1_view'),
    path('model/<int:problem_id>/', views.model_detail_view, name='model_detail_view'),
    path('predict/<int:problem_id>/', views.predict_view, name='predict_view'),
]
