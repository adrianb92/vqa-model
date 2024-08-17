import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from transformers import DistilBertTokenizer
import wandb
from torchvision import transforms
from data.data_loading import load_datasets
from datasets.dataset import VQADataset, collate_fn
from models.model import VQAModel
from training.train import train_epoch, validate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset_raw, val_dataset_raw = load_datasets()

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = VQADataset(train_dataset_raw, tokenizer, transform)
val_dataset = VQADataset(val_dataset_raw, tokenizer, transform)

all_possible_answers = list(set(train_dataset.possible_answers) | set(val_dataset.possible_answers))
train_dataset.possible_answers = all_possible_answers
val_dataset.possible_answers = all_possible_answers

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, collate_fn=collate_fn, num_workers=4)

num_classes = len(all_possible_answers)
vqa_model = VQAModel(num_classes).to(device)

weights = torch.ones(num_classes).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
optimizer = optim.Adam(vqa_model.parameters(), lr=1e-4)

wandb.init(project="vqa-project-v2")
wandb.watch(vqa_model, log="all")

num_epochs = 50
for epoch in range(num_epochs):
    train_loss, train_precision, train_recall = train_epoch(epoch, vqa_model, train_loader, optimizer, criterion, device)
    val_loss, val_precision, val_recall = validate(vqa_model, val_loader, criterion, device)

    wandb.log({
        "Epoch": epoch,
        "Training Loss": train_loss,
        "Training Precision": train_precision,
        "Training Recall": train_recall,
        "Validation Loss": val_loss,
        "Validation Precision": val_precision,
        "Validation Recall": val_recall
    })

torch.save(vqa_model.state_dict(), 'model_v2.pth')
artifact = wandb.Artifact('model', type='model')
artifact.add_file('model_v2.pth')
wandb.log_artifact(artifact)
wandb.finish()