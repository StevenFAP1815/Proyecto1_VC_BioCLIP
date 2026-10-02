import os
from PIL import Image
import cv2
import numpy as np

def process_image(input_path: str, output_path: str, target_size=(224, 224), padding_ratio=0.2) -> None:
    """
    Carga una imagen, la recorta al centro haciendo un cuadrado
    y la redimensiona al tamaño objetivo requerido por BioCLIP (224x224),
    detecta la semilla en fondos oscuros/claros
    """

    # 1. Cargar imagen con OpenCV
    cv_img = cv2.imread(input_path)
    if cv_img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en: {input_path}")
    
    # Convertir a escala de grises y aplicar desenfoque suave para eliminar ruido
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Umbralización para segmentar la semilla del fondo (Otsu's Thresholding)
    # Funciona detectando automáticamente el contraste entre la semilla brillante y el fondo oscuro
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Cargar versión PIL para el recorte final
    pil_img = Image.open(input_path).convert("RGB")
    width, height = pil_img.size

    if contours:
        # Tomar el contorno más grande (que corresponde a la semilla)
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # Calcular el centro de la semilla y su dimensión máxima
        cx, cy = x + w // 2, y + h // 2
        max_side = max(w, h)
        
        # Añadir un margen (padding) proporcional alrededor de la semilla
        crop_size = int(max_side * (1 + padding_ratio))
        
        # Definir bordes garantizando que sea un cuadrado centrado en la semilla
        left = max(0, cx - crop_size // 2)
        top = max(0, cy - crop_size // 2)
        right = min(width, cx + crop_size // 2)
        bottom = min(height, cy + crop_size // 2)
        
        # Recortar la zona detectada
        img_cropped = pil_img.crop((left, top, right, bottom))
    else:
        # Si no detecta ningún contorno, vuelve al Center Crop de respaldo
        print("Aviso: No se detectó objeto claro. Aplicando Center Crop estándar.")
        min_dim = min(width, height)
        left = (width - min_dim) / 2
        top = (height - min_dim) / 2
        right = (width + min_dim) / 2
        bottom = (height + min_dim) / 2
        img_cropped = pil_img.crop((left, top, right, bottom))

    # 4. Redimensionar a 224x224 usando LANCZOS (alta calidad)
    img_resized = img_cropped.resize(target_size, Image.Resampling.LANCZOS)
    
    # Guardar resultado
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
    img_resized.save(output_path, "JPEG", quality=95)
    print(f"Procesada con éxito (Smart Crop): {output_path}")

if __name__ == "__main__":
    raw_dir = "data/raw_examples"
    processed_dir = "data/processed_examples"
    
    input_file = os.path.join(raw_dir, "tangerine_02.jpg")
    output_file = os.path.join(processed_dir, "tangerine_02.jpg")
    
    if os.path.exists(input_file):
        process_image(input_file, output_file)
    else:
        print(f"No se encontró el archivo de origen: {input_file}")