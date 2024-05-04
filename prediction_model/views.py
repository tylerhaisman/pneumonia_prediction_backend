from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import permissions
from .serializers import PredictionSerializer
from .predict import predict_pneumonia


class PredictionApiView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = PredictionSerializer(data=request.data)
        if serializer.is_valid():
            image = request.FILES["image"].read()
            pneumonia_likely, overall_confidence, pneumonia_type, type_confidence = (
                predict_pneumonia(image)
            )
            data = {
                "pneumonia_likely": pneumonia_likely,
                "overall_confidence": overall_confidence,
                "pneumonia_type": pneumonia_type,
                "type_confidence": type_confidence,
            }
            return Response(data, status=status.HTTP_202_ACCEPTED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
