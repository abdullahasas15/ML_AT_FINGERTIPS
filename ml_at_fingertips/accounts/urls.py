from django.urls import path
from . import views

urlpatterns = [
    path('signup/', views.signup_page, name='signup_page'),
    path('login/', views.login_page, name='login_page'),
    path('logout/', views.logout_view, name='logout_view'),
]