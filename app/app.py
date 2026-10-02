import os
import torch
import gradio as gr
from src.preprocess import process_image
from src.predict import load_bioclip_model, predict_species

# 1. Configurar dispositivo y cargar modelo al iniciar la aplicación
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Cargando BioCLIP en dispositivo: {device}...")

model, preprocess, tokenizer = load_bioclip_model(device=device)
config_path = "config/species_config.json"

# Ruta temporal para guardar la imagen preprocesada antes de inferir
TEMP_PROCESSED_PATH = "data/processed_examples/temp_input.jpg"

def classify_seed_image(input_image):
    """
    Función de callback para Gradio:
    Recibe la imagen cargada por el usuario, la preprocesa y ejecuta la inferencia.
    """
    if input_image is None:
        return "Por favor, carga una imagen para clasificar."

    # Si la imagen viene como array de NumPy (por defecto en Gradio), la guardamos temporalmente
    os.makedirs(os.path.dirname(TEMP_PROCESSED_PATH), exist_ok=True)
    input_image.save("data/raw_examples/temp_raw.jpg")

    # Step 1: Preprocesar la imagen (Center crop 224x224)
    process_image("data/raw_examples/temp_raw.jpg", TEMP_PROCESSED_PATH)

    # Step 2: Inferencia Zero-Shot con BioCLIP
    predictions = predict_species(
        image_path=TEMP_PROCESSED_PATH,
        config_path=config_path,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device
    )

    # Gradio Label acepta un diccionario {clase: probabilidad}
    return predictions

# 3. Diseñar la interfaz con Gradio
with gr.Blocks(title="BioCLIP - Clasificador Zero-Shot de Semillas") as demo:
    gr.Markdown(
        """
        # 🌱 Clasificador Zero-Shot de Semillas con BioCLIP
        Sube una imagen de una semilla para identificar su especie botánica utilizando el modelo **BioCLIP**.
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Cargar o Tomar Foto de la Semilla")
            submit_btn = gr.Button("Clasificar Semilla", variant="primary")

        with gr.Column():
            output_labels = gr.Label(num_top_classes=5, label="Predicciones de BioCLIP")

    # Ejemplos de prueba rápida si existen en la carpeta
    if os.path.exists("data/raw_examples/mandarina_01.jpg"):
        gr.Examples(
            examples=["data/raw_examples/mandarina_01.jpg"],
            inputs=image_input,
            label="Ejemplos rápidos"
        )

    submit_btn.click(
        fn=classify_seed_image,
        inputs=image_input,
        outputs=output_labels
    )

if __name__ == "__main__":
    # Lanzar la aplicación
    demo.launch(share=True)