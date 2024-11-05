import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import label_binarize
import numpy as np


def evaluate(
    encoder: nn.Module,
    finetuned_head: nn.Module,
    test_dataset: Dataset,
    dataset_name: str,
    audio_length: int,
    device,
) -> dict:
    est_array = []
    gt_array = []

    encoder = encoder.to(device)
    encoder.eval()

    if finetuned_head is not None:
        finetuned_head = finetuned_head.to(device)
        finetuned_head.eval()

    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset))):
            waveform, label = test_dataset[idx]
            if waveform.shape[-1] < audio_length:
                padding = audio_length - waveform.shape[-1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            elif waveform.shape[-1] > audio_length:
                waveform = waveform[:, :audio_length]
                
            waveform = waveform.unsqueeze(0).to(device)

            output = encoder(waveform)
            if finetuned_head:
                output = finetuned_head(output)

            # Apply softmax to get probabilities
            if dataset_name in ["magnatagatune", "msd"]:
                output = torch.sigmoid(output)
            else:
                output = F.softmax(output, dim=1)

            # Store predictions and ground truth
            est_array.append(output.cpu().numpy())
            gt_array.append(label)

    est_array = np.concatenate(est_array, axis=0)
    gt_array = np.array(gt_array)

    if dataset_name in ["magnatagatune", "msd"]:
        # Multi-label classification metrics
        roc_aucs = roc_auc_score(gt_array, est_array, average="macro")
        pr_aucs = average_precision_score(gt_array, est_array, average="macro")
        return {
            "PR-AUC": pr_aucs,
            "ROC-AUC": roc_aucs,
        }
    else:
        # Multi-class classification metrics
        preds = np.argmax(est_array, axis=1)
        accuracy = accuracy_score(gt_array, preds)

        # Binarize labels for AUC metrics
        n_classes = test_dataset.n_classes if hasattr(test_dataset, 'n_classes') else len(np.unique(gt_array))
        gt_binarized = label_binarize(gt_array, classes=range(n_classes))

        #show ROC-AUC and PR-AUC
        roc_auc = roc_auc_score(gt_binarized, est_array, average='macro', multi_class='ovr')
        pr_auc = average_precision_score(gt_binarized, est_array, average='macro')

        # get confusion matrix
        conf_matrix = confusion_matrix(gt_array, preds)

        #Compute classification report
        if hasattr(test_dataset, 'labels'):
            target_names = test_dataset.labels
        else:
            target_names = [str(i) for i in range(n_classes)]
        class_report = classification_report(gt_array, preds, target_names=target_names)

        return {
            "Accuracy": accuracy,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
            "Confusion Matrix": conf_matrix,
            "Classification Report": class_report
        }
