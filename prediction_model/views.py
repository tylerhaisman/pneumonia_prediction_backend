from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import permissions
from .serializers import PredictionSerializer
from .predict import predict_pneumonia, is_chest_xray


class PredictionApiView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = PredictionSerializer(data=request.data)
        if serializer.is_valid():
            image = request.FILES["image"].read()
            chest_xray = is_chest_xray(image)
            if chest_xray == False:
                data = {
                    "chest_xray": chest_xray,
                    "pneumonia_likely": None,
                    "overall_confidence": None,
                    "pneumonia_type": None,
                    "type_confidence": None,
                }
                return Response(data, status=status.HTTP_202_ACCEPTED)
            else:
                (
                    pneumonia_likely,
                    overall_confidence,
                    pneumonia_type,
                    type_confidence,
                ) = predict_pneumonia(image)
                data = {
                    "chest_xray": chest_xray,
                    "pneumonia_likely": pneumonia_likely,
                    "overall_confidence": overall_confidence,
                    "pneumonia_type": pneumonia_type,
                    "type_confidence": type_confidence,
                }
                return Response(data, status=status.HTTP_202_ACCEPTED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
