import json
import torch
import open_clip
from PIL import Image

def load_bioclip_model(device="cpu"):
    """Carga el modelo y preprocesador de BioCLIP."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        'hf-hub:imageomics/bioclip',
        device=device
    )
    tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
    model.eval()
    return model, preprocess, tokenizer

def predict_species(image_path: str, config_path: str, model, preprocess, tokenizer, device="cpu"):
    """Realiza la clasificación Zero-Shot sobre una imagen."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    labels = [sp['common_name'] for sp in config['species']]
    prompts = [config['templates'][0].format(scientific_name=sp['scientific_name']) for sp in config['species']]

    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = tokenizer(prompts).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = similarity[0].cpu().numpy()

    # Retorna un diccionario con las probabilidades ordenadas
    results = dict(zip(labels, [float(p) for p in probs]))
    return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))