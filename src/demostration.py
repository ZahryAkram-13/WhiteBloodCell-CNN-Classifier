import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger

from .models import CellClassifier
from data.utils import get_random_image, CLASSES_NAMES, DATA_TESTB_PATH, DATA_TESTA_PATH
from paths import MODELS_DIR


from torchvision import transforms

def get_single_image_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def predict_image(image_path, model_path, model_name="mobilenet_v3_small"):
    
    logger.info(f"Loading model: {model_name}")
    
    model = CellClassifier(
        model_name=model_name,
        num_classes=len(CLASSES_NAMES),
        native_model=False
    )
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=model.device,
            weights_only=True
        )
    )
    model.to(model.device)
    model.eval()
    
    logger.info(f"Loading image: {image_path}")

    from src.data_augmentation_transforms import ImageData
    
    image = Image.open(image_path).convert('RGB')
    # transform = get_single_image_transform()
    transform = ImageData.get_eval_transform()

    image_tensor = transform(image).unsqueeze(0).to(model.device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()
    
    predicted_class = CLASSES_NAMES[predicted_idx]
    
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence*100:.1f}%", 
             fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Predicted Class: {predicted_class}")
    logger.info(f"Confidence: {confidence*100:.2f}%")
    logger.info(f"{'='*50}\n")
    
    return predicted_class, confidence


if __name__ == "__main__":
    
    IMAGE_PATH = get_random_image(DATA_TESTA_PATH / "Lymphocyte")
    # IMAGE_PATH = get_random_image(DATA_TESTA_PATH / "Neutrophil")

    MODELS = [
        "best_efficientnet_b0.pth",
        "best_mobilenet_v2.pth", 
        "best_mobilenet_v3_small.pth",
        "best_shufflenet_v2.pth",
        "best_squeezenet.pth",
        "best_resnet18.pth",
    ]

    MODELS_NAME = [
        "efficientnet_b0",
        "mobilenet_v2",
        "mobilenet_v3_small", 
        "shufflenet_v2",
        "squeezenet"
    ]
    
    index = 4
    MODEL_PATH = MODELS_DIR / MODELS[index]
    MODEL_NAME = MODELS_NAME[index]
    
    predict_image(IMAGE_PATH, MODEL_PATH, MODEL_NAME)