import os
from PIL import Image

def process_image(input_path: str, output_path: str, target_size=(224, 224)) -> None:
    """
    Carga una imagen, la recorta al centro haciendo un cuadrado
    y la redimensiona al tamaño objetivo requerido por BioCLIP (224x224).
    """
    with Image.open(input_path) as img:
        # Convertir a RGB por si la foto viene en RGBA o escala de grises
        img = img.convert("RGB")
        
        # Recorte cuadrado central (Center Crop)
        width, height = img.size
        min_dim = min(width, height)
        
        left = (width - min_dim) / 2
        top = (height - min_dim) / 2
        right = (width + min_dim) / 2
        bottom = (height + min_dim) / 2
        
        img_cropped = img.crop((left, top, right, bottom))
        
        # Redimensionar al tamaño final requerido por el modelo
        img_resized = img_cropped.resize(target_size, Image.Resampling.LANCZOS)
        
        # Crear la carpeta de salida si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Guardar imagen procesada
        img_resized.save(output_path, "JPEG", quality=95)
        print(f"Procesada con éxito: {output_path}")

if __name__ == "__main__":
    # Prueba rápida procesando la primera imagen de mandarina
    raw_dir = "data/raw_examples"
    processed_dir = "data/processed_examples"
    
    input_file = os.path.join(raw_dir, "mandarina_04.jpg")
    output_file = os.path.join(processed_dir, "mandarina_04.jpg")
    
    if os.path.exists(input_file):
        process_image(input_file, output_file)
    else:
        print(f"No se encontró el archivo de origen: {input_file}")