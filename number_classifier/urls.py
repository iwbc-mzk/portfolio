from django.urls import path
from .views import NumberClassifierForm, canvas


urlpatterns = [
    path('', NumberClassifierForm.as_view()),
    path('canvas/', canvas),
]