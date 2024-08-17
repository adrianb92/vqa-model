# Visual Question Answering (VQA) Project

## Project Overview

This project implements a Visual Question Answering (VQA) system using PyTorch. The model combines text and image inputs to answer questions about images. It leverages pre-trained models such as DistilBERT for text encoding and ResNet50 for image encoding. The project also uses the Hugging Face `transformers` library for natural language processing and `timm` for the image model.

The project is modular and organized into various components for data loading, model definition, training, and validation.

## Project Structure

The project is organized as follows:

```
my_project/
├── data/
│   └── data_loading.py           # Data loading scripts
├── datasets/
│   └── dataset.py                # Dataset class and related code
├── models/
│   └── model.py                  # Model architecture files
├── training/
│   └── train.py                  # Training and validation logic
├── main.py                       # Main entry point of the project
├── requirements.txt              # Dependencies
└── utils.py                      # Utility/helper functions (optional)
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

- **`main.py`**
  - The main script that ties everything together, handling data loading, model initialization, training, validation, and logging with `WandB`.

- **`requirements.txt`**
  - Lists the necessary Python packages to run the project.

- **`utils.py`**
  - Placeholder for any utility/helper functions that may be needed across multiple modules.

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
   git clone https://github.com/your-repository/vqa-project.git
   cd vqa-project
   ```

2. **Install dependencies**:

   Install the required packages listed in the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the project**:

   The main script `main.py` orchestrates the entire pipeline from data loading to training and validation. Run the script as follows:

   ```bash
   python main.py
   ```

4. **Tracking with WandB**:

   The project uses `WandB` for experiment tracking. You need to set up a `WandB` account and authenticate using your API key. You can initialize a new project on `WandB` by simply running the `main.py` script, and the logging will be automatically handled.

5. **Model Checkpoints**:

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

5. **Experiment Logging**:
   - The project integrates with `WandB` for logging and tracking metrics during training and validation.

6. **Model Saving**:
   - After training, the model is saved and logged as an artifact.

## Future Work

- **Hyperparameter Tuning**: Experiment with different learning rates, optimizers, and batch sizes to improve performance.
- **Model Architecture Enhancements**: Consider testing other model architectures or improving the current one with additional features like attention mechanisms.
- **Data Augmentation**: Implement data augmentation techniques to improve generalization.
- **Deployment**: Package the model for deployment in a production environment, such as serving via a REST API.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request with any improvements, bug fixes, or new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Hugging Face Transformers** for providing pre-trained models and easy-to-use APIs.
- **Torchvision** and **timm** for pretrained image models.
- **Weights and Biases (WandB)** for experiment tracking and logging.
