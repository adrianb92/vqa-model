import torch
from sklearn.metrics import precision_score, recall_score
import numpy as np

def calculate_metrics(preds, labels):
    tp = (preds * labels).sum()
    fp = (preds * (1 - labels)).sum()
    fn = ((1 - preds) * labels).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return precision.mean(), recall.mean()

def train_epoch(epoch, model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    num_batches = 0
    accumulation_steps = 4
    optimizer.zero_grad()

    for i, (text_inputs, attention_mask, image_inputs, labels) in enumerate(dataloader):
        text_inputs = text_inputs.to(device)
        attention_mask = attention_mask.to(device)
        image_inputs = image_inputs.to(device)
        labels = labels.to(device)

        outputs = model(text_inputs, attention_mask, image_inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item()
        predictions = (torch.sigmoid(outputs) > 0.5).float()

        precision, recall = calculate_metrics(predictions, labels)
        cumulative_precision += precision
        cumulative_recall += recall
        num_batches += 1

    train_loss = running_loss / len(dataloader)
    precision = cumulative_precision / num_batches
    recall = cumulative_recall / num_batches

    print(f"Epoch {epoch}, Training Loss: {train_loss}, Precision: {precision}, Recall: {recall}")

    return train_loss, precision, recall

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    num_batches = 0
    with torch.no_grad():
        for text_inputs, attention_mask, image_inputs, labels in dataloader:
            text_inputs = text_inputs.to(device)
            attention_mask = attention_mask.to(device)
            image_inputs = image_inputs.to(device)
            labels = labels.to(device)

            outputs = model(text_inputs, attention_mask, image_inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            predictions = (torch.sigmoid(outputs) > 0.5).float()

            precision, recall = calculate_metrics(predictions, labels)
            cumulative_precision += precision
            cumulative_recall += recall
            num_batches += 1

    val_loss = running_loss / len(dataloader)
    precision = cumulative_precision / num_batches
    recall = cumulative_recall / num_batches

    print(f"Validation Loss: {val_loss}, Precision: {precision}, Recall: {recall}")

    return val_loss, precision, recall