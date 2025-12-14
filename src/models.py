import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


import matplotlib.pyplot as plt
import numpy as np

from loguru import logger

from data.data_augmentation_transforms import ImageData
from data.utils import DATA_TRAIN_PATH, DATA_TESTA_PATH
from paths import BEST_MODEL_DIR, MODELS_DIR, PLOTS_DIR


import os
#os.environ['MIOPEN_DEBUG_DISABLE_FIND_DB'] = '1'
#os.environ['MIOPEN_FIND_ENFORCE'] = '3'
#os.environ['MIOPEN_FIND_MODE'] = '0' 


#os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:512'
#os.environ['HSA_ENABLE_SDMA'] = '0'  # Disable DMA for stability


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
        "mobilenet_v2": lambda pretrained=True: models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        ),
        "mobilenet_v3_small": lambda pretrained=True: models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        ),
        "efficientnet_b0": lambda pretrained=True: models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        ),
        "shufflenet_v2": lambda pretrained=True: models.shufflenet_v2_x1_0(
            weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        ),
        "squeezenet": lambda pretrained=True: models.squeezenet1_1(
            weights=models.SqueezeNet1_1_Weights.DEFAULT if pretrained else None
        ),
    }


    def __init__(self, model_name="resnet18", num_classes=5, pretrained=True, native_model = False):
        super().__init__()

        self.device = CellClassifier.get_device()

        assert model_name in CellClassifier.BASE_MODELS, f"Available: {list(CellClassifier.BASE_MODELS.keys())}"
        self.model_name = model_name

        self.base_model = CellClassifier.BASE_MODELS[model_name](pretrained=pretrained)

        if "mobilenet_v3" in model_name:
            in_features = self.base_model.classifier[0].in_features
            self.base_model.classifier = nn.Identity()

        elif "mobilenet_v2" in model_name:
            in_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()
            
        elif "efficientnet_b0" in model_name:
            in_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()
            
        elif "shufflenet" in model_name:
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
            
        elif "squeezenet" in model_name:
            in_features = 512
            self.base_model.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unknown model architecture: {model_name}")

        self.classifier_head = self.native_model(in_features, num_classes) if native_model else self.build_classifier(in_features, num_classes)

    def native_model(self, in_features, num_classes):
        return nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(in_features, num_classes)
            )
            
    def build_classifier(self, in_features, num_classes):
        return nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, num_classes)
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
    
    def evaluate_detailed(self, loader, class_names=None):
       
        self.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                outputs = self(images)
                
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, matthews_corrcoef
        
        cm = confusion_matrix(all_labels, all_preds)
        
        report = classification_report(
            all_labels, 
            all_preds, 
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        kappa = cohen_kappa_score(all_labels, all_preds)
        mcc = matthews_corrcoef(all_labels, all_preds)
        
        per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
        
        return {
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'confusion_matrix': cm,
            'classification_report': report,
            'cohen_kappa': kappa,
            'matthews_corrcoef': mcc,
            'per_class_accuracy': per_class_acc
        }

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
                torch.save(self.state_dict(), MODELS_DIR / f"best_{self.model_name}.pth")
                
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
        if "squeezenet" in self.model_name:
            x = self.base_model.features(x)
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        else:
            x = self.base_model(x)
        
        if x.dim() == 4:
            x = x.flatten(1)
        
        x = self.classifier_head(x)
        return x
        
    def freeze_base_layers(self, freeze_ratio=0.7):

        total_layers = len(list(self.base_model.parameters()))
        layers_to_freeze = int(total_layers * freeze_ratio)
        
        for i, param in enumerate(self.base_model.parameters()):
            if i < layers_to_freeze:
                param.requires_grad = False
        
        logger.info(f"Froze {layers_to_freeze}/{total_layers} base model layers")



if __name__ == "__main__":

    logger.info("\nLoading dataset...")

    data = ImageData(
        train_dir=DATA_TRAIN_PATH,
        test_dir=DATA_TESTA_PATH,
        batch_size=32,
        num_workers=5,
        balance_classes=True
    )

    train_loader = data.train_loader()
    val_loader = data.val_loader()
    test_loader = data.test_loader()

    logger.info(f"Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

    NATIVE = False
    model = CellClassifier(model_name="mobilenet_v3_small", num_classes=5, native_model = NATIVE)
    model.freeze_base_layers(freeze_ratio=0.7)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-5,
        weight_decay=1e-3
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-7,       
    )
    early = EarlyStopping(patience = 5)

    history = model.fit(
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler,
        epochs = 50,
        early_stopping=early,
    )
    
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    history_filename = MODELS_DIR / f"history_{'NATIVE' if NATIVE else ''}_{model.model_name}_64BATCH_{timestamp}.npz"
    
    np.savez(
        history_filename,
        train_loss=np.array(history['train_loss']),
        val_loss=np.array(history['val_loss']),
        train_acc=np.array(history['train_acc']),
        val_acc=np.array(history['val_acc'])
    )
    
    logger.info(f"Training history saved to {history_filename}")
    
