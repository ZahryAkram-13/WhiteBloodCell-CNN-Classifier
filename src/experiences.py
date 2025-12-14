import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


from src.data_augmentation_transforms import ImageData
from data.utils import (
    DATA_TRAIN_PATH,
    DATA_TESTA_PATH,
    CLASSES_INDEX,
    CLASSES_NAMES
)
from paths import BEST_MODEL_DIR, MODELS_DIR, PLOTS_DIR


import torch
import numpy as np
from loguru import logger

from models import CellClassifier
from plots import plot_confusion_matrix, plot_per_class_metrics
from src.data_augmentation_transforms import ImageData
from data.utils import DATA_TRAIN_PATH, DATA_TESTA_PATH

import matplotlib.pyplot as plt
import seaborn as sns




if __name__ == "__main__":

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
    MODEL_NAME = MODELS_NAME[index]
    MODEL_PATH = MODELS_DIR / MODELS[index]
    NUM_CLASSES = 5
    BATCH_SIZE = 64
    IMAGE_SIZE = 224
    
    
    logger.info("Loading model...")
    model = CellClassifier(
        model_name = MODEL_NAME,
        num_classes = NUM_CLASSES,
        native_model = False
    )
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=model.device))
    model.to(model.device)
    model.eval()
    
    logger.info(f"Model loaded from {MODEL_PATH}")
    logger.info(f"Device: {model.device}")
    
    logger.info("Loading dataset...")
    data = ImageData(
        train_dir=DATA_TRAIN_PATH,
        test_dir=DATA_TESTA_PATH,
        batch_size=BATCH_SIZE,
        num_workers=4,
        balance_classes=False
    )
    
    test_loader = data.test_loader()
    val_loader = data.val_loader()
    
    logger.info(f"Test batches: {len(test_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATING ON TEST SET")
    logger.info("="*80)
    
    test_results = model.evaluate_detailed(test_loader, class_names=CLASSES_NAMES)
        
    plot_confusion_matrix(
        test_results['confusion_matrix'],
        CLASSES_NAMES,
        title=f'Test Set Confusion Matrix - {MODEL_NAME}',
        save_path= PLOTS_DIR / f'confusion_matrix_test_{MODEL_NAME.upper()}.png'
    )
    
    plot_per_class_metrics(
        test_results['classification_report'],
        CLASSES_NAMES,
        save_path= PLOTS_DIR / f'per_class_metrics_test_{MODEL_NAME.upper()}.png'
    )
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATING ON VALIDATION SET")
    logger.info("="*80)
    
    val_results = model.evaluate_detailed(val_loader, class_names=CLASSES_NAMES)
    
    
    plot_confusion_matrix(
        val_results['confusion_matrix'],
        CLASSES_NAMES,
        title = f'Validation Set Confusion Matrix - {MODEL_NAME}',
        save_path = PLOTS_DIR / f'confusion_matrix_val_{MODEL_NAME.upper()}.png'
    )
    
    print("\n" + "="*80)
    print("TEST vs VALIDATION COMPARISON")
    print("="*80)
    print(f"{'Metric':<30} {'Test':<15} {'Validation':<15} {'Difference':<15}")
    print("-"*80)
    
    test_acc = test_results['classification_report']['accuracy']
    val_acc = val_results['classification_report']['accuracy']
    print(f"{'Accuracy':<30} {test_acc:<15.4f} {val_acc:<15.4f} {abs(test_acc - val_acc):<15.4f}")
    
    test_f1 = test_results['classification_report']['macro avg']['f1-score']
    val_f1 = val_results['classification_report']['macro avg']['f1-score']
    print(f"{'Macro F1-Score':<30} {test_f1:<15.4f} {val_f1:<15.4f} {abs(test_f1 - val_f1):<15.4f}")
    
    print(f"{'Cohen Kappa':<30} {test_results['cohen_kappa']:<15.4f} {val_results['cohen_kappa']:<15.4f} {abs(test_results['cohen_kappa'] - val_results['cohen_kappa']):<15.4f}")
    
    print("="*80)
    
    if abs(test_acc - val_acc) > 0.05:
        print("\n WARNING: Large difference between test and validation accuracy!")
        print("   This suggests potential overfitting or data distribution issues.")
    else:
        print("\n Test and validation performance are similar - good generalization!")
    
    logger.info("\n Evaluation complete!")















