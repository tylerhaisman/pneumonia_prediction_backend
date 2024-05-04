from django.db import models
from django.contrib.auth.models import User
from django.core.validators import FileExtensionValidator


class Prediction(models.Model):
    image = models.FileField(
        upload_to="images/",
        validators=[
            FileExtensionValidator(["jpg", "jpeg", "png"]),
        ],
    )
