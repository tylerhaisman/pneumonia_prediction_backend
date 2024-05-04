from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import colored
from colored import stylize
from PIL import Image
import io
import os


def predict_pneumonia(file):
    img = Image.open(io.BytesIO(file))
    img = img.convert("RGB")
    img = img.resize((150, 150))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array.astype("float32") / 255.0

    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "pneumonia_prediction.keras"
    )
    model = load_model(model_path)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence_percentages = [round(prob * 100, 2) for prob in prediction[0]]
    if predicted_class == 0 or predicted_class == 2:
        print("Pneumonia likely")
        print("Confidence:", (confidence_percentages[0] + confidence_percentages[2]))
        if predicted_class == 0:
            print("Bacterial")
            print("Confidence:", confidence_percentages[0])
            return (
                True,
                (confidence_percentages[0] + confidence_percentages[2]),
                "Bacterial",
                confidence_percentages[0],
            )
        else:
            print("Viral")
            print("Confidence:", confidence_percentages[2])
            return (
                True,
                (confidence_percentages[0] + confidence_percentages[2]),
                "Viral",
                confidence_percentages[2],
            )
    else:
        print("Pneumonia unlikely")
        print("Confidence:", confidence_percentages[1])
        return False, confidence_percentages[1], None, None


# images = [
#     "BACTERIAL/person80_bacteria_389",
#     "BACTERIAL/person82_bacteria_404",
#     "BACTERIAL/person155_bacteria_731",
#     "BACTERIAL/person171_bacteria_826",
#     "NORMAL/IM-0015-0001",
#     "NORMAL/NORMAL2-IM-0372-0001",
#     "NORMAL/NORMAL2-IM-0351-0001",
#     "NORMAL/IM-0099-0001",
#     "VIRAL/person1_virus_13",
#     "VIRAL/person1671_virus_2887",
#     "VIRAL/person1613_virus_2799",
#     "VIRAL/person42_virus_89",
# ]

# for file_path in images:
#     print(stylize(file_path, colored.fg("blue")))

#     img_path = "./x-ray_data/test/" + file_path + ".jpeg"
#     img = image.load_img(img_path, target_size=(150, 150))

#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)

#     img_array /= 255.0

#     model = load_model("pneumonia_prediction.keras")

#     prediction = model.predict(img_array)
#     predicted_class = np.argmax(prediction)
#     confidence_percentages = [round(prob * 100, 2) for prob in prediction[0]]
#     if predicted_class == 0 or predicted_class == 2:
#         print("Pneumonia likely")
#         print("Confidence:", (confidence_percentages[0] + confidence_percentages[2]))
#         if predicted_class == 0:
#             print("Bacterial")
#             print("Confidence:", confidence_percentages[0])
#             if "BACTERIAL" in file_path:
#                 print(stylize("Correct", colored.fg("green")))
#             else:
#                 print(stylize("Incorrect", colored.fg("red")))
#         else:
#             print("Viral")
#             print("Confidence:", confidence_percentages[2])
#             if "VIRAL" in file_path:
#                 print(stylize("Correct", colored.fg("green")))
#             else:
#                 print(stylize("Incorrect", colored.fg("red")))
#     else:
#         print("Pneumonia unlikely")
#         print("Confidence:", confidence_percentages[1])
#         if "NORMAL" in file_path:
#             print(stylize("Correct", colored.fg("green")))
#         else:
#             print(stylize("Incorrect", colored.fg("red")))
#     print("\n|||||||||||||||||||\n")
