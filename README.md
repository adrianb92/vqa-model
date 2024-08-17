# Visual Question Answering (VQA) Project

## Project Overview

This project implements a Visual Question Answering (VQA) system using PyTorch. The model combines text and image inputs to answer questions about images. It leverages pre-trained models such as DistilBERT for text encoding and ResNet50 for image encoding. The project also uses the Hugging Face `transformers` library for natural language processing and `timm` for the image model.

The project is modular and organized into various components for data loading, model definition, training, validation and testing.

## Project Structure

The project is organized as follows:

```
vqa-model/
├── data/
│   └── data_loading.py           # Data loading scripts
├── datasets/
│   └── dataset.py                # Dataset class and related code
├── models/
│   └── model.py                  # Model architecture files
├── training/
│   └── train.py                  # Training and validation logic
├── testing/
│   └── test.py                   # Script for testing the model with custom image and question
├── main.py                       # Main entry point of the project
└── requirements.txt              # Dependencies
```

### Key Components:

- **`data/`**
  - `data_loading.py`: Contains code to load and preprocess the datasets. This includes loading the dataset from Hugging Face's `datasets` library and preparing it for training and validation.
  
- **`datasets/`**
  - `dataset.py`: Contains the custom `VQADataset` class, which handles tokenization, image preprocessing, and label preparation. Also includes a custom `collate_fn()` function for batch preparation.

- **`models/`**
  - `model.py`: Contains the neural network model definitions, including the `TextEncoder` (DistilBERT), `ImageEncoder` (ResNet50), and the combined `VQAModel` for the VQA task.

- **`training/`**
  - `train.py`: Contains the training and validation loops, as well as functions for calculating precision and recall metrics.

- **`testing/`**
   - `test.py`: Script for testing the model with a custom image and question provided via command-line arguments. This script loads the pre-trained model and dynamically retrieves the possible answers from the dataset, allowing the model to select the best answer based on its training.

- **`main.py`**
  - The main script that ties everything together, handling data loading, model initialization, training, validation, and logging with `WandB`.

- **`requirements.txt`**
  - Lists the necessary Python packages to run the project.

## Getting Started

### Prerequisites

Ensure you have Python 3.7 or above installed. Install the required packages by running the following command:

```bash
pip install -r requirements.txt
```

### Required Libraries

- `transformers`: For using pre-trained language models.
- `datasets`: For loading and handling datasets.
- `torch` and `torchvision`: For building and training the PyTorch models.
- `timm`: Pretrained models for image tasks.
- `wandb`: For experiment tracking and logging.
- `scikit-learn`: For precision and recall calculations.
- `numpy`: For numerical operations.
- `Pillow`: For image processing.

### Usage

1. **Clone the repository**:

   ```bash
   git clone https://github.com/adrianb92/vqa-model.git
   cd vqa-model
   ```

2. **Install dependencies**:

   Install the required packages listed in the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model**:

   You can train the model using the `main.py` script:

   ```bash
   python main.py
   ```
   
  This will handle everything from data loading to model training and logging the results.

4. **Test the Model with a Custom Image and Question**:

   Once the model is trained, you can test it on a custom image and question using the `test.py` script located in the `testing/` folder. This script will load the trained model and dynamically retrieve the possible answers from the dataset.

   To run the script:

   ```bash
   python testing/test.py --image path/to/your/image.jpg --question "What is in the picture?"
   ```

   This will print the most likely answer predicted by the model based on the learned possible answers from the training data.

5. **Tracking with WandB**:

   The project uses `WandB` for experiment tracking. You need to set up a `WandB` account and authenticate using your API key. You can initialize a new project on `WandB` by simply running the `main.py` script, and the logging will be automatically handled.

6. **Model Checkpoints**:

   After training, the model will be saved as `model_v2.pth`. This file can be logged and saved as an artifact using `WandB`.

## Configuration

The model hyperparameters and other configurations, such as the number of epochs and learning rate, are set in the `main.py` file and can be adjusted as needed. Key configurations:

- **Epochs**: 50
- **Batch size**: 100
- **Learning rate**: 1e-4

To modify these, simply update the configuration section in the `main.py` file.

## Project Workflow

1. **Data Loading**: 
   - The `data_loading.py` script loads the VQA dataset and removes unnecessary columns.

2. **Dataset Preparation**:
   - The `VQADataset` class prepares the data for the model, including tokenizing questions and preprocessing images.

3. **Model Definition**:
   - The `VQAModel` class combines the text encoder (DistilBERT) and image encoder (ResNet50) into a unified model that performs classification on the combined features.

4. **Training and Validation**:
   - The `train.py` script handles training and validation, including calculating metrics like precision and recall.

5. **Testing**:
   - The `test.py` script allows you to test the trained model with custom images and questions, printing the most likely answer.

6. **Experiment Logging**:
   - The project integrates with `WandB` for logging and tracking metrics during training and validation.

7. **Model Saving**:
   - After training, the model is saved and logged as an artifact.

## Future Work

- **Hyperparameter Tuning**: Experiment with different learning rates, optimizers, and batch sizes to improve performance.
- **Model Architecture Enhancements**: Consider testing other model architectures or improving the current one with additional features like attention mechanisms.
- **Data Augmentation**: Implement data augmentation techniques to improve generalization.
- **Deployment**: Package the model for deployment in a production environment, such as serving via a REST API.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Hugging Face Transformers** for providing pre-trained models and easy-to-use APIs.
- **Torchvision** and **timm** for pretrained image models.
- **Weights and Biases (WandB)** for experiment tracking and logging.
