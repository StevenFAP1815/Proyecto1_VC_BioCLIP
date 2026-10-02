import os
import json
import torch
import torch.nn as nn
from PIL import Image
import open_clip
import cv2
import numpy as np

# Configuración de rutas y dispositivo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/seed_classifier.pth"
CONFIG_PATH = "config/species_config_5.json"

# 1. Cargar el modelo base BioCLIP
print("Cargando BioCLIP...")
bioclip_model, _, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
bioclip_model = bioclip_model.to(DEVICE)
bioclip_model.eval()

# 2. Cargar el clasificador entrenado (.pth)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
classes = checkpoint['classes']
num_classes = len(classes)

classifier = nn.Linear(512, num_classes)  # BioCLIP genera embeddings de dimensión 512
classifier.load_state_dict(checkpoint['state_dict'])
classifier = classifier.to(DEVICE)
classifier.eval()

# 3. Cargar diccionario de nombres comunes desde species_config.json
common_names_map = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config_data = json.load(f)
        for sp in config_data.get("species", []):
            # Mapear folder_name o scientific_name al nombre común
            common_names_map[sp["folder_name"]] = sp.get("common_name", sp["scientific_name"])

def smart_crop_pil(pil_img, padding_ratio=0.2):
    """Aplica Smart Crop con OpenCV para eliminar fondos oscuros."""
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

def predict_seed(image_input, top_k=3):
    """
    Recibe una imagen (PIL or filepath), aplica Smart Crop, extrae el embedding
    con BioCLIP y predice las probabilidades con el clasificador entrenado.
    """
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")

    # Crop inteligente y preprocesamiento
    cropped = smart_crop_pil(image)
    tensor_img = preprocess_val(cropped).unsqueeze(0).to(DEVICE)

    # Inferencia
    with torch.no_grad():
        # Extracción de embedding
        features = bioclip_model.encode_image(tensor_img)
        features /= features.norm(dim=-1, keepdim=True)
        
        # Inferencia de la capa lineal
        outputs = classifier(features)
        probabilities = torch.softmax(outputs, dim=-1)[0]

    # Obtener Top K resultados
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