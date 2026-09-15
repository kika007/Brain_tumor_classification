# Brain Tumor Classification with CNN

This project uses a TensorFlow/Keras notebook to classify brain MRI images into two classes: `Brain Tumor` and `Healthy`. It prepares an image dataset, trains a convolutional neural network, compares it with a simple sigmoid baseline, and evaluates both models with accuracy, classification reports, and confusion matrices.

## Main Features
* Loads and inspects a two-class brain MRI image dataset.
* Splits the source images into training, validation, and test sets using a 70/20/10 split.
* Preprocesses images to `128 x 128` RGB tensors and scales pixel values to `[0, 1]`.
* Trains a CNN with convolution, max-pooling, dense, batch-normalization, dropout, and sigmoid output layers.
* Compares the CNN with a flattened-input sigmoid baseline and reports accuracy, precision, recall, F1-score, and confusion matrices.

## Used Technologies
**Machine Learning:** TensorFlow, Keras, `tensorflow.keras`, CNN layers, binary cross-entropy, Adam optimizer, early stopping.

**Data Processing:** NumPy, pandas, Pillow, `ImageDataGenerator`, random sampling, filesystem-based dataset splitting.

**Evaluation and Visualization:** scikit-learn metrics, Matplotlib, seaborn.

**Environment:** Python 3, Jupyter Notebook.

## Installation and Setup

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data / Dataset Preparation
Download the dataset from [Kaggle: Brain Tumor Dataset](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset). The dataset is not included in the Git repository because it is excluded by `.gitignore`. After downloading and extracting it, provide it manually with this structure before running the notebook:

```text
Brain Tumor Data Set/
└── Brain Tumor Data Set/
    ├── Brain Tumor/
    │   └── <brain tumor image files>
    └── Healthy/
        └── <healthy image files>
```

The notebook creates the following directories and copies images into them using a 70% training, 20% validation, and 10% test split:

```text
Brain Tumor Data Set/
├── train/
│   ├── Brain Tumor/
│   └── Healthy/
├── val/
│   ├── Brain Tumor/
│   └── Healthy/
└── test/
    ├── Brain Tumor/
    └── Healthy/
```

Supported image extensions are `.jpg`, `.jpeg`, and `.png`. The four standalone sample images used for manual prediction are stored in `sample_test_images/`.

## Usage
Start Jupyter from the project root:

```bash
jupyter notebook brain.ipynb
```

Alternatively, open the notebook in VS Code with the Python environment containing the installed requirements.

Run the notebook cells in order. The workflow is:

1. Inspect the source dataset and create the train/validation/test directories.
2. Build the TensorFlow data generators.
3. Train the CNN and save it as `model.keras`.
4. Evaluate the test set and generate plots and confusion matrices.
5. Train and evaluate the simple sigmoid baseline.
6. Run manual prediction using:

```text
sample_test_images/Test_h.jpg
```

The notebook is the primary executable for this project; there are no standalone Python scripts or deployment services.
