from django.urls import path
from .views import NumeralRecognitionForm, canvas


urlpatterns = [
    path('', NumeralRecognitionForm.as_view()),
    path('a/', canvas),
]