from django.urls import path
from .views import NumeralRecognitionForm


urlpatterns = [
    path('', NumeralRecognitionForm.as_view()),
]