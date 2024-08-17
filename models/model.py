import torch
from torch import nn
from transformers import DistilBertModel
import timm

class TextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.embedding_dim = self.backbone.config.dim
        self.CLS_token_idx = 0

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = x.last_hidden_state
        x = last_hidden_state[:, self.CLS_token_idx, :]
        return x

class ImageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = timm.create_model('resnet50', pretrained=True, num_classes=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return x

class VQAModel(nn.Module):
    def __init__(self, num_classes):
        super(VQAModel, self).__init__()
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()
        self.classifier = nn.Linear(self.text_encoder.embedding_dim + self.image_encoder.backbone.num_features, num_classes)

    def forward(self, input_ids, attention_mask, image_inputs):
        text_outputs = self.text_encoder(input_ids, attention_mask)
        image_outputs = self.image_encoder(image_inputs)
        combined = torch.cat((text_outputs, image_outputs), dim=1)
        return self.classifier(combined)