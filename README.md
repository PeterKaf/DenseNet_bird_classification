# Multilabel Image Classification of Bird Species

## Getting Started

### Prerequisites

Before you begin:
  * If you are just browsing, please visit the jupyter notebook: `DenseNet bird classification.ipynb`
  * If you wish to interact with the project, ensure you have met the requirements outlined in `environment.yaml`. 
Note that this environment was originally built on Linux, and some packages may face difficulties when attempting to 
install on Windows due to platform-specific compatibility issues. You can still access the notebook in preview mode
without completing this step, but full functionality may be limited.


### Installing Dependencies

### Using conda

To use same conda enviroment, run following line:

```
conda env create -f environment.yaml -p ../DenseNet_bird_classification/env
```
and after that:

```
conda activate ../DenseNet_bird_classification/env
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

1. Download the original dataset and place it into the matching directory specified in the `config.py` (dataset/train, dataset/test, dataset/valid by default).
2. (Optional) Use `dataset_merger.py` and `dataset_redistribution.py` functions to tailor the dataset according to your needs.
3. Access saved weights from my training session using the `evaluate.py`, `predict.py`, and `single_img_predict.py` functions.
4. Try training your own model using the `train.py` function.

By following these steps, you'll be able to replicate the project's setup and experiment with the dataset and model configurations as needed.


## License

This project has been released under the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) open source license.

