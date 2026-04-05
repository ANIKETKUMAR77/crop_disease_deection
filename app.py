from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ---------------- MODEL ----------------

class CheckpointedResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

CLASS_LABELS = [
    "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy",
    "Blueberry___healthy","Cherry_(including_sour)___Powdery_mildew","Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight","Corn_(maize)___healthy","Grape___Black_rot",
    "Grape___Esca_(Black_Measles)","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)","Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot","Peach___healthy",
    "Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy","Potato___Early_blight",
    "Potato___Late_blight","Potato___healthy","Raspberry___healthy","Soybean___healthy",
    "Squash___Powdery_mildew","Strawberry___Leaf_scorch","Strawberry___healthy",
    "Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight","Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus","Tomato___healthy"
]

model = CheckpointedResNet(len(CLASS_LABELS))
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- SMART DISEASE INFO ----------------

DISEASE_TYPE = {
    "blight": "fungal",
    "rot": "fungal",
    "rust": "fungal",
    "mildew": "fungal",
    "virus": "viral",
    "bacterial": "bacterial"
}

def get_disease_info(disease, lang="en"):
    name = disease.replace("___", " ").replace("_", " ")

    if "healthy" in disease.lower():
        if lang == "hi":
            return {
                "desc": "यह पौधा स्वस्थ है और इसमें कोई बीमारी नहीं है।",
                "solution": "सही पानी, धूप और पोषण बनाए रखें।"
            }
        else:
            return {
                "desc": "The plant is healthy and shows no disease symptoms.",
                "solution": "Maintain proper watering, sunlight, and nutrition."
            }

    dtype = "plant"
    for key in DISEASE_TYPE:
        if key in disease.lower():
            dtype = DISEASE_TYPE[key]

    if lang == "hi":
        return {
            "desc": f"{name} एक {dtype} रोग है जो पौधों की पत्तियों को नुकसान पहुंचाता है।",
            "solution": "संक्रमित पत्तियां हटाएं, उचित दवा (फंगीसाइड/कीटनाशक) का उपयोग करें और देखभाल बनाए रखें।"
        }
    else:
        return {
            "desc": f"{name} is a {dtype} disease that damages plant leaves.",
            "solution": "Remove infected leaves, use proper fungicides/pesticides, and maintain plant care."
        }

# ---------------- ROUTES ----------------

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    lang = request.form.get("lang", "en")

    image = Image.open(file).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted_class = torch.max(outputs, 1)
        disease = CLASS_LABELS[predicted_class.item()]

    info = get_disease_info(disease, lang)

    return jsonify({
        "disease": disease.replace("___", " ").replace("_", " "),
        "description": info["desc"],
        "solution": info["solution"]
    })

# ---------------- RUN ----------------

if __name__ == "__main__":
    app.run(debug=True)