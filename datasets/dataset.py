import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence

class VQADataset(Dataset):
    def __init__(self, data, tokenizer, transform=None):
        self.data = data
        self.tokenizer = tokenizer
        self.transform = transform
        self.possible_answers = self.extract_possible_answers(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        image_path = item['image_id']
        labels = item['label']['ids']
        weights = item['label']['weights']

        text_inputs = self.tokenizer(question, return_tensors='pt', padding=True, truncation=True)

        if text_inputs['input_ids'].nelement() == 0:
            return None

        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image '{image_path}': {e}")
            return None

        label_tensor = torch.zeros(len(self.possible_answers))
        for label, weight in zip(labels, weights):
            try:
                label_index = self.possible_answers.index(label)
                label_tensor[label_index] = weight
            except ValueError:
                print(f"Label '{label}' not in possible_answers, skipping.")
                return None

        return (text_inputs['input_ids'].squeeze(), text_inputs['attention_mask'].squeeze(), image, label_tensor)

    @staticmethod
    def extract_possible_answers(data):
        possible_answers = set()
        for item in data:
            labels = item['label']['ids']
            possible_answers.update(labels)
        return list(possible_answers)

def collate_fn(batch):
    batch = [item for item in batch if item is not None]

    input_ids = [item[0] for item in batch]
    attention_mask = [item[1] for item in batch]
    images = [item[2] for item in batch]
    labels = [item[3] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    images = torch.stack(images)
    labels = torch.stack(labels)

    return input_ids, attention_mask, images, labels
