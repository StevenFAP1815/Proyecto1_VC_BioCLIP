import os
from pathlib import Path
from PIL import Image

# 1. Rutas de origen y destino especificadas
SOURCE_DIR = Path(r"C:\Users\cjant\Downloads\Seed dataset\Seed dataset")
TARGET_DIR = Path(r"C:\Users\cjant\Downloads\Seed dataset") / "data_seed_ajustado"

# Tamaño objetivo
TARGET_SIZE = (224, 224)
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

def resize_and_center_crop(img, target_size=(224, 224)):
    """
    Escala la imagen manteniendo la relación de aspecto según su lado menor
    y realiza un recorte central exacto al tamaño deseado.
    """
    orig_w, orig_h = img.size
    target_w, target_h = target_size
    
    # Calcular la escala basada en la dimensión menor
    scale = max(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    # Redimensionar con filtro LANCZOS para mantener la máxima calidad
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Calcular coordenadas para el recorte central
    left = (new_w - target_w) / 2
    top = (new_h - target_h) / 2
    right = (new_w + target_w) / 2
    bottom = (new_h + target_h) / 2
    
    # Aplicar el recorte
    img_cropped = img_resized.crop((left, top, right, bottom))
    return img_cropped

# 2. Procesamiento de la estructura de carpetas
processed_count = 0
error_count = 0

print(f"Iniciando procesamiento...")
print(f"Origen:  {SOURCE_DIR}")
print(f"Destino: {TARGET_DIR}\n")

for category_dir in SOURCE_DIR.iterdir():
    # Evitar procesar la carpeta destino si ya fue creada previamente
    if category_dir.is_dir() and category_dir.name != "data_seed_ajustado":
        # Crear la subcarpeta correspondiente en la ruta destino
        target_category_dir = TARGET_DIR / category_dir.name
        target_category_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in category_dir.iterdir():
            if img_path.suffix.lower() in VALID_EXTENSIONS:
                try:
                    with Image.open(img_path) as img:
                        # Convertir imágenes a RGB si no lo están
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Aplicar ajuste de dimensión menor y recorte central
                        processed_img = resize_and_center_crop(img, TARGET_SIZE)
                        
                        # Guardar la imagen procesada
                        save_path = target_category_dir / img_path.name
                        processed_img.save(save_path, quality=95)
                        processed_count += 1
                        
                except Exception as e:
                    print(f"Error procesando {img_path.name}: {e}")
                    error_count += 1

print(f"\n¡Proceso completado exitosamente!")
print(f"Imágenes guardadas en: {TARGET_DIR}")
print(f"Total procesadas: {processed_count}")
if error_count > 0:
    print(f"Imágenes con error: {error_count}")