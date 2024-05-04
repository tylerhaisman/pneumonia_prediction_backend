from django.urls import path, include
from .views import (
    PredictionApiView,
)

urlpatterns = [
    path("predict", PredictionApiView.as_view()),
]
