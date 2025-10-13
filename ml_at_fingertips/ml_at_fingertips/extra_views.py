from django.shortcuts import render
from django.contrib.auth.decorators import login_required

# Import ProblemStatement models from both apps
from classifier.models import ProblemStatement as ClassProblem
from regressor.models import ProblemStatement as RegrProblem


@login_required
def problems_list(request):
    """Unified problems list showing both classification and regression problem statements."""
    classifier_problems = ClassProblem.objects.all()
    regressor_problems = RegrProblem.objects.all()

    combined = []
    for p in classifier_problems:
        combined.append({
            'id': p.id,
            'title': p.title,
            'description': p.description,
            'model_type': getattr(p, 'model_type', 'Classification'),
            'app': 'classifier'
        })
    for p in regressor_problems:
        combined.append({
            'id': p.id,
            'title': p.title,
            'description': p.description,
            'model_type': getattr(p, 'model_type', 'Regression'),
            'app': 'regressor'
        })

    # Optional: stable sort by model_type then title
    combined = sorted(combined, key=lambda x: (x['model_type'], x['title']))

    return render(request, 'problems_list.html', {'problems': combined})
