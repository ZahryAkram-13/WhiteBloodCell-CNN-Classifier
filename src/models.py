import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from loguru import logger

from data.data_augmentation_transforms import ImageData
from data.utils import DATA_TRAIN_PATH, DATA_TESTA_PATH
from paths import BEST_MODEL_DIR


class EarlyStopping:
    def __init__(self, patience = 1):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True





class CellClassifier(nn.Module):

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            return torch.device("cpu")

        try:
            x = torch.randn(2000, 2000, device=device)
            y = torch.randn(2000, 2000, device=device)
            _ = torch.matmul(x, y)

        except Exception as e:
            logger.debug(f"GPU detected but failed compute test: {e}")
            logger.debug("Falling back to CPU")
            return torch.device("cpu")

        logger.debug(f"GPU is working: {torch.cuda.get_device_name(device)}")
        return device

    BASE_MODELS = {
        "resnet18": lambda pretrained=True: models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None),
        "resnet34": lambda pretrained=True: models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None),
        "mobilenet_v2": lambda pretrained=True: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None),
        "efficientnet_b0": lambda pretrained=True: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None),
        "googlenet": lambda pretrained=True: models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT if pretrained else None),
    }

    def __init__(self, model_name="resnet18", num_classes=5, pretrained=True, native_model = False):
        super().__init__()

        self.device = CellClassifier.get_device()

        assert model_name in CellClassifier.BASE_MODELS, f"Available: {list(CellClassifier.BASE_MODELS.keys())}"
        self.model_name = model_name

        self.base_model = CellClassifier.BASE_MODELS[model_name](pretrained=pretrained)

        if model_name == "mobilenet_v2":
            in_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()
        else:
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()

        self.classifier_head = self.native_model(in_features, num_classes) if native_model else self.build_classifier(in_features, num_classes)

    def native_model(self, in_features, num_classes):
        return nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.4),
                nn.Linear(in_features, num_classes)
            )

    def build_classifier(self, in_features, num_classes):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def build_model(self):
        return nn.Sequential(
            self.base_model,
            self.classifier_head
        )

    def validate(self, loader, criterion):
        self.eval()
        total_loss = 0
        correct = 0
        samples = 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self(images)

                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                samples += labels.size(0)

        accuracy = correct / samples
        return total_loss / len(loader), accuracy

    def fit(self, train_loader, val_loader, optimizer, criterion, scheduler,
        epochs=10, early_stopping=None):

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_loss = float("inf")

        self.to(self.device)

        
        logger.info("\n===================== TRAINING START =====================")

        for epoch in range(epochs):
            logger.debug(f"\nEpoch {epoch+1}/{epochs}")
            torch.cuda.empty_cache()

            train_loss, train_acc = self.train_one_epoch(
                train_loader, optimizer, criterion
            )

            val_loss, val_acc = self.validate(
                val_loader, criterion
            )

            if scheduler is not None:
                scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            
            logger.debug(f"Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%")
            logger.debug(f"Val Loss:   {val_loss:.4f} | Acc: {val_acc*100:.2f}%")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.state_dict(), f"best_{self.model_name}.pth")
                
                logger.debug("New best model saved")

            if early_stopping is not None:
                early_stopping.step(val_loss)
                if early_stopping.should_stop:
                    logger.debug("Early stopping triggered")
                    break

            torch.cuda.empty_cache()

        return history


    def train_one_epoch(self, loader, optimizer, criterion):
        self.train()
        total_loss = 0
        correct = 0
        samples = 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad(set_to_none=True)
            outputs = self(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            samples += labels.size(0)

        accuracy = correct / samples
        return total_loss / len(loader), accuracy

    def forward(self, x):
        x = self.base_model(x)          
        x = self.classifier_head(x)
        return x



if __name__ == "__main__":

    logger.info("\nLoading dataset...")

    data = ImageData(
        train_dir=DATA_TRAIN_PATH,
        test_dir=DATA_TESTA_PATH,
        batch_size=32,
        num_workers=2,
        balance_classes=True
    )

    train_loader = data.train_loader()
    val_loader = data.val_loader()
    test_loader = data.test_loader()

    logger.info(f"Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")


    model = CellClassifier(model_name="resnet18", num_classes=5, native_model = True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4 
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    early = EarlyStopping(patience = 5)

    history = model.fit(
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler,
        epochs = 20,
        early_stopping=early,
    )
