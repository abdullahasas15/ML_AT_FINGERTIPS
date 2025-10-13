from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from .models import User
from django.contrib.auth.hashers import make_password

def signup_page(request):
    if request.method == 'POST':
        full_name = request.POST.get('full_name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        favorite_ml_type = request.POST.get('favorite_ml_type')
        if not full_name or not email or not password or not favorite_ml_type:
            return render(request, 'signup.html', {'error': 'All fields required.'})
        if User.objects.filter(email=email).exists():
            return render(request, 'signup.html', {'error': 'Email already exists.'})
        # Split full name into first and last name
        name_parts = full_name.strip().split(' ', 1)
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else ''
        
        user = User.objects.create(
            username=email,
            email=email,
            password=make_password(password),
            favorite_ml_type=favorite_ml_type,
            first_name=first_name,
            last_name=last_name
        )
    login(request, user)
    return redirect('/')
    return render(request, 'signup.html')

def login_page(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        user = authenticate(request, username=email, password=password)
        if user is not None:
            login(request, user)
            return redirect('/')
        else:
            return render(request, 'login.html', {'error': 'Invalid credentials.'})
    return render(request, 'login.html')

def logout_view(request):
    logout(request)
    return redirect('/')
