from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing_page, name='landing_page'),  # Public landing page
    path('dashboard/', views.models1_view, name='home'),  # Authenticated dashboard
    path('models1/', views.models1_view, name='models1_view'),
    path('models2/', views.models1_view, name='models2_view'),  # Use same view for both
    path('model/<int:problem_id>/', views.model_detail_view, name='model_detail_view'),
    path('predict/<int:problem_id>/', views.predict_view, name='predict_view'),
    
    # Contribution URLs
    path('contribute/', views.submit_contribution, name='submit_contribution'),
    path('my-contributions/', views.my_contributions, name='my_contributions'),
    
    # Admin review URLs (use 'review' instead of 'admin/review' to avoid conflict)
    path('review/', views.review_contributions, name='review_contributions'),
    path('review/<int:contribution_id>/', views.review_contribution_detail, name='review_contribution_detail'),
    path('review/approve/<int:contribution_id>/', views.approve_contribution, name='approve_contribution'),
    path('review/reject/<int:contribution_id>/', views.reject_contribution, name='reject_contribution'),
]
