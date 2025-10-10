// JavaScript for ML Everywhere Website

document.addEventListener('DOMContentLoaded', function() {
    const navLinks = document.querySelectorAll('a[href^="#"]');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    const navbar = document.querySelector('.navbar');
    window.addEventListener('scroll', function() {
        if (window.scrollY > 100) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });

    initDemos();
    initAuthForms();
});

function initDemos() {
    window.predictWeather = function() {
        const temp = parseFloat(document.getElementById('temperature').value);
        const humidity = parseFloat(document.getElementById('humidity').value);
        const resultDiv = document.getElementById('weather-result');
        
        if (isNaN(temp) || isNaN(humidity)) {
            resultDiv.innerHTML = '<div class="alert alert-warning">Please enter valid temperature and humidity values.</div>';
            return;
        }

        let prediction = '';
        const confidence = Math.random() * 20 + 80;
        
        if (temp > 30 && humidity > 70) {
            prediction = 'High probability of thunderstorms 🌩️';
        } else if (temp > 25 && humidity > 60) {
            prediction = 'Possible rain showers 🌧️';
        } else if (temp > 20) {
            prediction = 'Partly cloudy with sunny intervals ⛅';
        } else if (temp > 10) {
            prediction = 'Clear skies with mild temperatures ☀️';
        } else {
            prediction = 'Cold with possible frost ❄️';
        }

        resultDiv.innerHTML = `
            <div class="alert alert-success">
                <h5>Weather Prediction Result</h5>
                <p><strong>Prediction:</strong> ${prediction}</p>
                <p><strong>Confidence:</strong> ${confidence.toFixed(1)}%</p>
                <p><strong>Temperature:</strong> ${temp}°C</p>
                <p><strong>Humidity:</strong> ${humidity}%</p>
            </div>
        `;
    };

    window.assessWine = function() {
        const alcohol = parseFloat(document.getElementById('alcohol').value);
        const acidity = parseFloat(document.getElementById('acidity').value);
        const resultDiv = document.getElementById('wine-result');
        
        if (isNaN(alcohol) || isNaN(acidity)) {
            resultDiv.innerHTML = '<div class="alert alert-warning">Please enter valid alcohol and acidity values.</div>';
            return;
        }

        let score = 0;
        
        if (alcohol >= 12 && alcohol <= 14) score += 3;
        else if (alcohol > 14) score += 2;
        else score += 1;

        if (acidity >= 0.4 && acidity <= 0.6) score += 3;
        else if (acidity > 0.6) score += 2;
        else score += 1;

        let qualityLevel = '';
        if (score >= 5) qualityLevel = 'Excellent Quality 🏆';
        else if (score >= 3) qualityLevel = 'Good Quality 👍';
        else qualityLevel = 'Average Quality ⚖️';

        const recommendations = [
            'Ideal for aging 2-3 years',
            'Best consumed within 1 year',
            'Pair with red meat dishes',
            'Excellent with cheese platters',
            'Serve at room temperature'
        ];

        const randomRecommendations = recommendations
            .sort(() => Math.random() - 0.5)
            .slice(0, 2);

        resultDiv.innerHTML = `
            <div class="alert alert-info">
                <h5>Wine Quality Assessment</h5>
                <p><strong>Quality:</strong> ${qualityLevel}</p>
                <p><strong>Score:</strong> ${score}/6</p>
                <p><strong>Alcohol:</strong> ${alcohol}%</p>
                <p><strong>Acidity:</strong> ${acidity}g/L</p>
                <p><strong>Recommendations:</strong></p>
                <ul>
                    ${randomRecommendations.map(rec => `<li>${rec}</li>`).join('')}
                </ul>
            </div>
        `;
    };

    const weatherInputs = document.querySelectorAll('#temperature, #humidity');
    weatherInputs.forEach(input => {
        input.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                predictWeather();
            }
        });
    });

    const wineInputs = document.querySelectorAll('#alcohol, #acidity');
    wineInputs.forEach(input => {
        input.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                assessWine();
            }
        });
    });
}

function initAuthForms() {
    // CSRF token helper
    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
    const csrftoken = getCookie('csrftoken');
    // No custom JS for login/signup forms; Django handles POST natively
}

function initAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, observerOptions);

    const sections = document.querySelectorAll('section');
    sections.forEach(section => {
        observer.observe(section);
    });
}

document.addEventListener('DOMContentLoaded', initAnimations);

function showLoading(element) {
    element.innerHTML = '<div class="text-center"><div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div><p class="mt-2">Analyzing data...</p></div>';
}

function simulateProcessing(callback, minDelay = 1000, maxDelay = 2000) {
    const delay = Math.random() * (maxDelay - minDelay) + minDelay;
    setTimeout(callback, delay);
}
