# CS-433: Class Project 1

Code for the first project of the Machine Learning Course (CS-433) at EPFL.

In this document, we provide instructions on how to run this code, and information about how the project is organized. Please refer to the report in `report.pdf` for a detailed description of the project, motivation of the choices we made, and the results we obtained.

## Summary
- [CS-433: Class Project 1](#cs-433-class-project-1)
  - [Summary](#summary)
  - [Getting Started](#getting-started)
  - [Implementation of ML methods](#implementation-of-ml-methods)
  - [Medical dataset](#medical-dataset)
    - [How to run the pipeline](#how-to-run-the-pipeline)
    - [Configuration](#configuration)
    - [Generating final AICrowd predictions](#generating-final-aicrowd-predictions)
  - [Authors](#authors)

## Getting Started
1. Make sure `python` is installed on your system. We used `v3.11.8` for development.
2. Install all the packages listed in `requirements.txt` (`numpy` and `matplotlib`). For example, if you use pip for package management, you can do so with:
```bash
pip install -r requirements.txt
```
3. Place the raw data (`x_train.csv`, `y_train.csv` and `x_test.csv` files) in the `data_raw` folder. You can download the data [here](https://www.cdc.gov/brfss/annual_data/2015/files/LLCP2015XPT.zip).

## Implementation of ML methods
The implementation of the ML methods is in the `implementations.py` file. The methods are implemented as functions with the interface specified by the project description. The methods implemented are:
- `mean_squared_error_gd(y, tx, initial_w, max_iters, gamma)`: Linear regression using gradient descent
- `mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma)`: Linear regression using stochastic gradient descent
- `least_squares(y, tx)`: Least squares regression using normal equations
- `ridge_regression(y, tx, lambda_)`: Ridge regression using normal equations
- `logistic_regression(y, tx, initial_w, max_iters, gamma)`: Logistic regression using gradient descent
- `reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)`: Regularized logistic regression using gradient descent

## Medical dataset

The original data is expected to be stored in the `data_raw` folder. This includes the `x_train.csv`, `y_train.csv` and `x_test.csv` files. The pipeline we use for the medical dataset preprocesses this raw data, and stores the preprocessed data in the `data_clean` folder, along with checkpoints and results. If you want to change these paths (`data_raw` and `data_clean`), you can do so in the configuration as explained below.

All data preprocessing is done in the `data_preprocessing.py` file, where the main entry point is the `get_all_data` function. This function reads the raw data, preprocesses it according to the configuration, and returns the preprocessed data. Please refer to its doc string and comments in the code for more information.

### How to run the pipeline

The main pipeline is implemented in the `run.py` file. To run the pipeline, you can simply run the following command:
```bash
python run.py
```

The current settings evaluates all the models on all the data preprocessing configurations trying all possible combinations of hyperparameters. **This takes several hours on a laptop!** If you want to run only the best model which produced the best submission predictions on AICrowd, the sections [Configuration](#configuration) and [Generating final AICrowd predictions](#generating-final-aicrowd-predictions) explain how to do that. Furthermore, they explain the configuration options that allows to narrow down the search space of hyperparameters and data pipelines.

### Configuration
The configuration of the pipeline is done at the top of the `run.py` file, in a global dictionary called `cfg`: 
```python
### global config
cfg = {
    "raw_data_path": "data_raw",
    "clean_data_path": "data_clean",
    "allow_load_clean_data": False,
    "remap_labels_to_01": True,
    "seed": 0,
    "scoring_fn": f1,
    "eval_frac": 0.1,
    "retrain_on_all_data_after_eval": True,
    "train": {
        "retrain_selected_on_all_data": True,
        "cv": {
            "k_folds": 5,
            "shuffle": True,
        },
        # "holdout": {
        #     "split_frac": 0.2,
        #     "seed": 0,
        # },
    },
}
```
Explanation of the configuration:
- `raw_data_path`: path to the folder containing the raw data (default: `data_raw`)
- `clean_data_path`: path to the folder where the preprocessed data and results will be stored (default: `data_clean`)
- `allow_load_clean_data`: if `True`, the pipeline will try to load the preprocessed data from the `clean_data_path` folder, falling back to preprocessing the raw data if it is not found. If `False`, the pipeline will always preprocess the raw data and store it in the `clean_data_path` folder (default: `False`)
- `remap_labels_to_01`: if `True`, the pipeline will remap the labels to 0 and 1. For the methods to work correctly, the labels should be 0 and 1 (default: `True`).
- `seed`: seed to assure reproducibility of the results (default: `0`)
- `scoring_fn`: scoring function to use for the cross-validation (default: `f1`, options: `f1`, `accuracy`)
- `eval_frac`: fraction of the data to use for the final evaluation of the models (default: `0.1`)
- `retrain_on_all_data_after_eval`: if `True`, the pipeline will retrain the best selected model on all the data after the final evaluation (default: `True`)
- `train`: configuration for the training part of the pipeline
  - `retrain_selected_on_all_data`: if `True`, the pipeline will retrain the selected model on all the data after the cross-validation (default: `True`)
  - `cv`: configuration for the cross-validation
    - `k_folds`: number of folds for the cross-validation (default: `5`)
    - `shuffle`: if `True`, the data will be shuffled before the cross-validation, otherwise it will assign every `k`-th sample to the same fold (default: `True`)
  - `holdout`: configuration for the holdout validation (computationally cheaper than cross-validation)
    - `split_frac`: fraction of the data to use for the holdout validation (default: `0.2`)
    - `seed`: seed to assure reproducibility of the results (default: `0`)

As you can see, `holdout` is commented out by default. This is the way of specifying that we want to use cross-validation instead of holdout validation. If you want to use holdout validation, you can simply uncomment the `holdout` configuration, specify the desired parameters, and comment out the `cv` configuration.

The configuration of the hyperparameter search is done separately from `cfg`, in the `runs` dictionary that has the following structure:
```python
### data-model combinations to run
runs = {
    "data": {
        "<data preprocessing name>": <preprocessing config or None>,
    },
    "models": {
        "<model name>": {
            "model_cls": <model class>,
            "hyperparam_search": <hyperparameter search space config>,
        },
    },
}
```
Explanation of the `runs` configuration:
- `data`: dictionary containing the data preprocessing configurations. The key is the name of the data preprocessing, and the value is the configuration for the data preprocessing (`dict` or `None` if no preprocessing is needed). Currently supported are:
  - `process_cols`: columns to clean and use (`all`, `selected`, or integer representing the percentage of columns to use)
  - `pca_kwargs`: configuration for the PCA preprocessing: `None` if this PCA step should be omitted, and a dictionary with the following keys if it should be included:
    - `max_frac_of_nan`: maximum fraction of NaN values in a column to keep it in the PCA preprocessing (between 0 and 1)
    - `n_components`: number of components to keep in the PCA preprocessing (you need to specify either this or `min_explained_variance`, not both)
    - `min_explained_variance`: minimum explained variance to keep in the PCA preprocessing (between 0 and 1; you need to specify either this or `n_components`, not both)
  - `standardize_num`: if `True`, the numerical columns will be standardized (default: `True`)
  - `onehot_cat`: if `True`, the categorical columns will be one-hot encoded (default: `True`)
  - `skip_rule_transformations`: if `True`, the rule-based transformations will be skipped (default: `False`)
- `models`: dictionary containing the models to run and their hyperparameter search spaces. The key is the name of the model, and the value is a dictionary with the following keys:
  - `model_cls`: the class of the model to run
  - `hyperparam_search`: the hyperparameter search space configuration. This is a dictionary with the following keys:
    - `param1`: list of values to search for the first hyperparameter
    - `param2`: list of values to search for the second hyperparameter (optional)
    - ...
    - `paramN`: list of values to search for the N-th hyperparameter (optional)
    - Please refer to the `models.py` file to see the available models and their hyperparameters.

### Generating final AICrowd predictions
To generate the submission we used for AICrowd, comment out or remove all the data pipelines other than the `"All columns": {"process_cols": "all", "pca_kwargs": None}`. Furthermore, use only the following model dictionary in the `models` configuration (currently commented out in the `run.py` file):
```python
"Logistic Regression": { ### AICrowd submission
    "model_cls": LogisticRegression,
    "hyperparam_search": {
        "gamma": [None],
        "use_line_search": [True],
        "optim_algo": ["lbfgs"],
        "optim_kwargs": [{"epochs": 1}],
        "class_weights": [{0: 1, 1: 4}],
        "reg_mul": [0],
        "verbose": [False],
    },
},
```
The `cfg` configuration should be kept as is (seed=0).

After running `python run.py`, the pipeline will train the model on all the data and store the predictions in the `data_clean/runs/<current-timestep>` folder (default path; `<current-timestep>` will be a timestamp of the run). The predictions will be stored in a file ending with `submission.csv` file in the same folder (for logistic regression this would be `Logistic_Regression_submission.csv`). This file can be submitted to AICrowd.

## Authors
  - **Andrej Kotevski** [[link]](https://people.epfl.ch/andrej.kotevski/?lang=en)
  - **Mikuláš Vanoušek** [[link]](https://people.epfl.ch/mikulas.vanousek/?lang=en)
  - **Jan Sobotka** [[link]](https://people.epfl.ch/jan.sobotka/?lang=en)