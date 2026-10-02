import os
import json
import torch
import torch.nn as nn
from PIL import Image
import open_clip
import cv2
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/seed_classifier.pth"
CONFIG_PATH = "config/species_config_5.json"

# Variables globales para reutilizar en memoria
bioclip_model = None
preprocess_val = None
tokenizer = None
classifier = None
classes = []
common_names_map = {}


def smart_crop_pil(pil_img, padding_ratio=0.2):
    """Aplica Smart Crop con OpenCV para enfocar la semilla."""
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    width, height = pil_img.size
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w // 2, y + h // 2
        max_side = max(w, h)
        crop_size = int(max_side * (1 + padding_ratio))
        
        left = max(0, cx - crop_size // 2)
        top = max(0, cy - crop_size // 2)
        right = min(width, cx + crop_size // 2)
        bottom = min(height, cy + crop_size // 2)
        return pil_img.crop((left, top, right, bottom))
    else:
        return pil_img


def load_bioclip_model(device=None):
    """Carga BioCLIP y la capa lineal entrenada, devolviendo los objetos que app.py espera."""
    global bioclip_model, preprocess_val, tokenizer, classifier, classes, common_names_map, DEVICE

    if device is not None:
        DEVICE = device

    print(f"Cargando BioCLIP en {DEVICE}...")
    bioclip_model, _, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
    tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
    
    bioclip_model = bioclip_model.to(DEVICE)
    bioclip_model.eval()

    if os.path.exists(MODEL_PATH):
        print(f"Cargando clasificador lineal desde {MODEL_PATH}...")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        classes = checkpoint['classes']
        
        classifier = nn.Linear(512, len(classes))
        classifier.load_state_dict(checkpoint['state_dict'])
        classifier = classifier.to(DEVICE)
        classifier.eval()
    else:
        print(f"⚠ ADVERTENCIA: No se encontró el archivo de pesos en '{MODEL_PATH}'.")

    # Cargar mapa de nombres comunes desde species_config.json
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config_data = json.load(f)
            for sp in config_data.get("species", []):
                common_names_map[sp["folder_name"]] = sp.get("common_name", sp["scientific_name"])

    print("¡Modelos e infraestructura cargados con éxito!")
    return bioclip_model, preprocess_val, tokenizer


def predict_seed(image_input, top_k=3):
    """Realiza la predicción sobre una imagen usando la capa lineal entrenada."""
    global bioclip_model, preprocess_val, classifier, classes, common_names_map, DEVICE

    if bioclip_model is None or classifier is None:
        load_bioclip_model(DEVICE)

    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")

    cropped = smart_crop_pil(image)
    tensor_img = preprocess_val(cropped).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = bioclip_model.encode_image(tensor_img)
        features /= features.norm(dim=-1, keepdim=True)
        
        outputs = classifier(features)
        probabilities = torch.softmax(outputs, dim=-1)[0]

    top_probs, top_indices = torch.topk(probabilities, top_k)
    
    results = []
    for prob, idx in zip(top_probs, top_indices):
        folder_class = classes[idx.item()]
        scientific_name = folder_class.replace("_", " ")
        common_name = common_names_map.get(folder_class, scientific_name)
        
        results.append({
            "scientific_name": scientific_name,
            "common_name": common_name,
            "confidence": round(prob.item() * 100, 2)
        })

    return results, cropped