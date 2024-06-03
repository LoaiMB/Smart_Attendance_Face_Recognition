import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import seaborn as sns
import random
import numpy as np
import torch

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def visualize_predictions(model, dataset, device, num_samples=5):
    model.eval()
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        img1, img2, label = dataset[idx]  # Get a pair and the label

        img1 = img1.to(device).unsqueeze(0)
        img2 = img2.to(device).unsqueeze(0)

        with torch.no_grad():
            output = model(img1, img2).squeeze()
            pred = (output > 0.4).float()

        if pred.item() == 1:
          res = "Matched"
        else:
          res = "Not Matched"

        img1 = img1.cpu().squeeze(0)
        img2 = img2.cpu().squeeze(0)

        img1 = denormalize(img1, mean, std).permute(1, 2, 0).numpy()
        img2 = denormalize(img2, mean, std).permute(1, 2, 0).numpy()

        img1 = (img1 * 255).astype(np.uint8)
        img2 = (img2 * 255).astype(np.uint8)

        axes[i, 0].imshow(img1)
        axes[i, 0].set_title("Anchor Image")
        axes[i, 0].axis('off')

        pair_type = "Positive" if label == 1 else "Negative"
        axes[i, 1].imshow(img2)
        axes[i, 1].set_title(f"{pair_type} Pair:\n" + res)
        axes[i, 1].axis('off')

    plt.show()



# Utils for evaluation
def evaluate_model(model, dataloader, device):
    model.eval()
    labels = []
    predictions = []

    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device).float()

            output = model(img1, img2).squeeze()
            pred = (output > 0.4).float()  # Binary classification threshold at 0.4

            labels.extend(label.cpu().numpy())
            predictions.extend(pred.cpu().numpy())

    labels = np.array(labels)
    predictions = np.array(predictions)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {roc_auc:.4f}")

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Plot confusion matrix
    conf_matrix = confusion_matrix(labels, predictions)
    plt.figure()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
