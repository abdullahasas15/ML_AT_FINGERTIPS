from django.urls import path
from . import views

urlpatterns = [
    path('models2/', views.models2_view, name='models2_view'),
    path('model/<int:problem_id>/', views.model_detail_view, name='model_detail_view_regressor'),
    path('predict/<int:problem_id>/', views.predict_view, name='predict_view_regressor'),
]
