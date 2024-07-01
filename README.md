# Multilabel Image Classification of Bird Species

## Getting Started

### Prerequisites

Before you begin:
  * if you are just browsing, please visit the jupyter notebook: `DenseNet bird classification.ipynb`
  * if you want to play with the project ensure you have met the following `environment.yaml`, (you can still access notebook in preview mode without this step)

### Installing Dependencies

### Using conda

To use same conda enviroment, run following line:

```
conda env create -f environment.yaml
```

### Data Source

The dataset used in this project was sourced from [525 bird spiecies dataset](https://www.kaggle.com/datasets/gpiosenka/100-bird-species). Download the original dataset into 
`dataset` directory, so that structure looks like the one provided in `config.py`. 
By default, it is: `dataset/train`,`dataset/test`,`dataset/valid`, but you can customize it from within config.

Please refer to the original source for licensing and usage restrictions.

## Running the Project

### In jupyter notebook

To run the project in jupyter notebook, execute the following commands in your terminal or command prompt:

```
jupyter lab
```

Navigate to the notebook named `DenseNet bird classification.ipynb` for an easy preview of the project.

### Localy

To run the project locally, follow these steps:

1. Download the original dataset and place it into the matching directory specified in the settings as Step One.
2. (Optional) Use `dataset_merger.py` and `dataset_redistribution.py` functions to tailor the dataset according to your needs.
3. Access saved weights from my training session using the `evaluate.py`, `predict.py`, and `single_img_predict.py` functions.
4. Try training your own model using the `train.py` function.

By following these steps, you'll be able to replicate the project's setup and experiment with the dataset and model configurations as needed.

