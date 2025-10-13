from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing_page, name='landing_page'),  # Public landing page
    path('dashboard/', views.models1_view, name='home'),  # Authenticated dashboard
    path('models1/', views.models1_view, name='models1_view'),
    path('models2/', views.models1_view, name='models2_view'),  # Use same view for both
    path('model/<int:problem_id>/', views.model_detail_view, name='model_detail_view'),
    path('predict/<int:problem_id>/', views.predict_view, name='predict_view'),
]
