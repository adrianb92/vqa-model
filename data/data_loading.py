from datasets import load_dataset

def load_datasets():
    train_dataset = load_dataset("Graphcore/vqa", split="train[:90000]")
    val_dataset = load_dataset("Graphcore/vqa", split="validation[:10000]")

    train_dataset = train_dataset.remove_columns(['question_type', 'question_id', 'answer_type'])
    val_dataset = val_dataset.remove_columns(['question_type', 'question_id', 'answer_type'])

    return train_dataset, val_dataset