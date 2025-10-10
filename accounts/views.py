from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.contrib.auth import authenticate, login
from django.shortcuts import redirect
from .models import User
from django.contrib.auth.hashers import make_password
from django.views.decorators.csrf import csrf_exempt

@api_view(['POST'])
@csrf_exempt
def signup(request):
    email = request.data.get('email')
    password = request.data.get('password')
    favorite_ml_type = request.data.get('favorite_ml_type')
    if not email or not password or not favorite_ml_type:
        return Response({'error': 'All fields required.'}, status=status.HTTP_400_BAD_REQUEST)
    if User.objects.filter(email=email).exists():
        return Response({'error': 'Email already exists.'}, status=status.HTTP_400_BAD_REQUEST)
    user = User.objects.create(
        username=email,
        email=email,
        password=make_password(password),
        favorite_ml_type=favorite_ml_type
    )
    login(request, user)
    return redirect('/base2/')

@api_view(['POST'])
@csrf_exempt
def user_login(request):
    email = request.data.get('email')
    password = request.data.get('password')
    user = authenticate(request, username=email, password=password)
    if user is not None:
        login(request, user)
        return redirect('/base2/')
    else:
        return Response({'error': 'Invalid credentials.'}, status=status.HTTP_401_UNAUTHORIZED)
