import torch
from PIL import Image
from transformers import DistilBertTokenizer
from torchvision import transforms
import argparse
from models.model import VQAModel
from datasets.dataset import VQADataset
from data.data_loading import load_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def load_model(model_path, num_classes):
    model = VQAModel(num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def infer_on_custom_image(model, image_path, question, possible_answers):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    text_inputs = tokenizer(question, return_tensors='pt', padding=True, truncation=True)
    input_ids = text_inputs['input_ids'].to(device)
    attention_mask = text_inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, image)
        preds = torch.sigmoid(outputs).squeeze().cpu().numpy()
    
    predicted_index = preds.argmax()
    
    return possible_answers[predicted_index]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test VQA Model with custom image and question.')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--question', type=str, required=True, help='The question to ask about the image.')
    args = parser.parse_args()

    model_path = 'model_v2.pth'
    
    train_dataset_raw, val_dataset_raw = load_datasets()
    
    train_dataset = VQADataset(train_dataset_raw, tokenizer, transform)
    possible_answers = train_dataset.possible_answers
    
    num_classes = len(possible_answers)
    
    model = load_model(model_path, num_classes)
    
    predicted_answer = infer_on_custom_image(model, args.image, args.question, possible_answers)
    
    print(f"Predicted Answer: {predicted_answer}")