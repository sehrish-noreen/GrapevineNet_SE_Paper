from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return acc, prec, rec, f1


output_dir = Path("/content/drive/MyDrive/TermPaper/GrapeVineDataSplits/output")
output_dir.mkdir(parents=True, exist_ok=True)

learning_rate = 1e-3

backbone_params, classifier_params = [], []
for name, param in model.named_parameters():
    if "backbone" in name:
        backbone_params.append(param)
    else:
        classifier_params.append(param)

optimizer = optim.AdamW(
    [
        {"params": backbone_params, "lr": learning_rate * 0.1},
        {"params": classifier_params, "lr": learning_rate},
    ],
    weight_decay=1e-4,
)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

num_epochs = 40
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=learning_rate,
    epochs=num_epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.4,
    anneal_strategy="cos",
)

scaler = torch.amp.GradScaler()

best_val_acc = 0.0

train_losses, val_losses = [], []
train_accs, val_accs = [], []
train_precs, val_precs = [], []
train_recs, val_recs = [], []
train_f1s, val_f1s = [], []

final_train_true, final_train_pred = None, None
best_val_true, best_val_pred = None, None

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print("-" * 60)

    model.train()
    train_true, train_pred = [], []
    train_loss = 0.0

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = (
            images.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss += loss.item() * images.size(0)
        train_true.extend(labels.cpu().numpy())
        train_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    train_loss /= len(train_loader.dataset)
    train_acc, train_prec, train_rec, train_f1 = compute_metrics(train_true, train_pred)

    final_train_true = train_true
    final_train_pred = train_pred

    model.eval()
    val_true, val_pred = [], []
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = (
                images.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )

            with torch.amp.autocast(device_type=device):
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            val_true.extend(labels.cpu().numpy())
            val_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    val_loss /= len(val_loader.dataset)
    val_acc, val_prec, val_rec, val_f1 = compute_metrics(val_true, val_pred)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    train_precs.append(train_prec)
    val_precs.append(val_prec)
    train_recs.append(train_rec)
    val_recs.append(val_rec)
    train_f1s.append(train_f1)
    val_f1s.append(val_f1)

    print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_true = val_true.copy()
        best_val_pred = val_pred.copy()
        torch.save(
            model.state_dict(), output_dir / "best_model_se_no_pretrained_weights.pth"
        )
        print(f"Best model updated with accuracy: {best_val_acc:.4f}")

history_df = pd.DataFrame(
    {
        "epoch": range(1, len(train_losses) + 1),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_accuracy": train_accs,
        "val_accuracy": val_accs,
        "train_precision": train_precs,
        "val_precision": val_precs,
        "train_recall": train_recs,
        "val_recall": val_recs,
        "train_f1": train_f1s,
        "val_f1": val_f1s,
    }
)

history_path = output_dir / "training_history_se0.csv"
history_df.to_csv(history_path, index=False)
print(f"\nTraining history saved to: {history_path}")

model.load_state_dict(
    torch.load(output_dir / "best_model_se_no_pretrained_weights.pth")
)
model.eval()

test_true, test_pred = [], []
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = (
            images.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )

        with torch.amp.autocast(device_type=device):
            outputs = model(images)

        test_true.extend(labels.cpu().numpy())
        test_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())

test_acc, test_prec, test_rec, test_f1 = compute_metrics(test_true, test_pred)

test_df = pd.DataFrame({"true_label": test_true, "predicted_label": test_pred})

test_path = output_dir / "test_predictions_se_no_pretrained_weights.csv"
test_df.to_csv(test_path, index=False)
print(f"Test predictions saved to: {test_path}")

class_names = ["Fresh", "Stage1", "Stage-2", "Stage-3"]

print("\nTest Results:")
print(f"Accuracy: {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall: {test_rec:.4f}")
print(f"F1-Score: {test_f1:.4f}")
print("\nClassification Report:")
print(classification_report(test_true, test_pred, target_names=class_names, digits=4))
