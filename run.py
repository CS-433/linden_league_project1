import os
import numpy as np
import pickle
from copy import deepcopy

from models import LogisticRegression, DecisionTreeBinaryClassifier, SVM
from data_preprocessing import get_all_data, resave_csv_as_npy
from helpers import create_csv_submission
from utils import split_data, get_cross_val_scores, prep_hyperparam_search, now_str, accuracy, f1, seed_all


### global config
cfg = {
    "raw_data_path": "data_raw",
    "clean_data_path": "data_clean",
    "allow_load_clean_data": True,
    "remap_labels_to_01": True,
    "seed": 0,
    "scoring_fn": f1,
    "train": {
        "retrain_selected_on_all_data": False,
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
if cfg["train"].get("cv", None) is not None:
    cfg["train"]["cv"]["scoring_fn"] = cfg["scoring_fn"]

### data-model combinations to run
runs = {
    "data": {
        "No PCA": None,
        # "PCA var > 0.4, nan < 0.4": {"max_frac_of_nan": 0.4, "min_explained_variance": 0.4},
        # "PCA var > 0.7, nan < 0.4": {"max_frac_of_nan": 0.4, "min_explained_variance": 0.7},
        # "PCA var > 0.95, nan < 0.4": {"max_frac_of_nan": 0.4, "min_explained_variance": 0.95},
        # "PCA var > 0.4, nan < 0.8": {"max_frac_of_nan": 0.8, "min_explained_variance": 0.4},
        # "PCA var > 0.7, nan < 0.8": {"max_frac_of_nan": 0.8, "min_explained_variance": 0.7},
        # "PCA var > 0.95, nan < 0.8": {"max_frac_of_nan": 0.8, "min_explained_variance": 0.95},
    },
    "models": {
        # "Logistic Regression": {
        #     "model_cls": LogisticRegression,
        #     "hyperparam_search": {
        #         "gamma": [None],
        #         "use_line_search": [True],
        #         "optim_algo": ["lbfgs"],
        #         "optim_kwargs": [{"epochs": 1}, {"epochs": 10}],
        #         "class_weights": [{0: 1, 1: 1}, {0: 1, 1: 2}, {0: 1, 1: 4}],
        #         "reg_mul": [0, 1e-2, 1e-4],
        #         "verbose": [False],
        #     },
        # },
        # "Decision Tree": {
        #     "model_cls": DecisionTreeBinaryClassifier,
        #     "hyperparam_search": {
        #         "max_depth": [3, 5],
        #         "min_samples_split": [5],
        #         "criterion": ["gini"],
        #         "class_weights": [{0: 1, 1: 1}, {0: 1, 1: 4}],
        #         "eval_max_n_thresholds_per_split": [4],
        #     },
        # }
        "SVM": {
            "model_cls": SVM,
            "hyperparam_search": {
                "_lambda": [.001, .005, .01,],
                "class_weights": [{0: 1, 1: i} for i in [2, 4, 6, 10]],
            }
        },
    }
}


def get_best_model(model_runs, x, y, verbose=1):
    """ Find the best model-hyperparameters combination using cross-validation or holdout validation.

    Args:
        model_runs : dict : model runs to try
        x : np.ndarray(N, D) : data for training/validation
        y : np.ndarray(N) : labels for training/validation
        verbose : int : verbosity level

    Returns:
        best : dict : best model-hyperparameters combination
        model_runs : dict : updated runs with validation scores
    """
    if verbose > 1: print("  Searching for the best model-hyperparameters combination...")

    ### keep track of best model
    best = {"model_name": None, "model": None, "val_score": 0, "hyperparams": None}
    for name, run_dict in model_runs.items():
        if verbose: print("  " + "-" * 3 + f" {name} " + "-" * 3)

        ### preaper hyperparam search space
        hyperparam_search = prep_hyperparam_search(run_dict["hyperparam_search"])
        if verbose > 1: print(f"  Searching hyperparameters among {len(hyperparam_search)} options...")
        seed_all(cfg["seed"])
        ### choose best hyperparameters using validation (CV, holdout, or none)
        if cfg["train"].get("cv", None) is not None:
            ### cross-validation
            if verbose > 1: print(f"  Cross-validating with {cfg['train']['cv']['k_folds']}-fold cross-validation...")
            models, cv_scores = [], []
            for hp_comb in hyperparam_search:
                ### train with these hyperparameters
                seed_all(cfg["seed"])
                model = run_dict["model_cls"](**hp_comb)
                hp_scores = get_cross_val_scores(model, x, y, **cfg["train"]["cv"])
                cv_scores.append(np.mean(hp_scores))
                models.append(model)
            run_dict["all_val_scores"] = cv_scores
            run_dict["hyperparam_search_list"] = hyperparam_search

            ### choose best hyperparameters
            best_hp_comb_idx = np.argmax(cv_scores)
            run_dict["hyperparams"] = hyperparam_search[best_hp_comb_idx]
            run_dict["val_score"] = cv_scores[best_hp_comb_idx]
            run_dict["model"] = models[best_hp_comb_idx]
        elif cfg["train"].get("holdout", None) is not None:
            ### validation holdout
            if verbose > 1: print(f"  Holdout validation with {cfg['train']['holdout']['split_frac']} split...")
            x_train, x_val, y_train, y_val = split_data(x, y, **cfg["train"]["holdout"])
            models, val_scores = [], []
            for hp_comb in hyperparam_search:
                ### train with these hyperparameters
                model = run_dict["model_cls"](**hp_comb)
                model.fit(x_train, y_train)
                y_val_pred = model.predict(x_val)
                val_score = cfg["scoring_fn"](y_val, y_val_pred)
                val_scores.append(val_score)
                models.append(model)
            run_dict["all_val_scores"] = val_scores
            run_dict["hyperparam_search_list"] = hyperparam_search

            ### choose best hyperparameters
            best_hp_comb_idx = np.argmax(val_scores)
            run_dict["hyperparams"] = hyperparam_search[best_hp_comb_idx]
            run_dict["val_score"] = val_scores[best_hp_comb_idx]
            run_dict["model"] = models[best_hp_comb_idx]
        else:
            ### just train on all data and hope for the best
            if verbose > 1: print("  No validation method specified. Training on all data with the first hyperparameter combination.")
            run_dict["all_val_scores"] = []
            run_dict["hyperparam_search_list"] = hyperparam_search
            run_dict["hyperparams"] = hyperparam_search[0]
            run_dict["val_score"] = -1
            run_dict["model"] = run_dict["model_cls"](**run_dict["hyperparams"]).fit(x, y)

        ### re-train on all data
        if cfg["train"]["retrain_selected_on_all_data"]:
            if verbose > 1: print(f"  Training on all data with best hyperparameters...")
            run_dict["model"] = run_dict["model_cls"](**run_dict["hyperparams"]).fit(x, y)

        ### print results of the best model
        if verbose:
            print(
                f"  Validation {cfg['scoring_fn'].__name__}: {run_dict['val_score']:.4f}"
                f"\n  Hyperparameters: {' '.join([f'{k}={v}' for k, v in run_dict['hyperparams'].items()])}"
            )

        ### remember best
        if run_dict["val_score"] > best["val_score"]:
            best["model_name"] = name
            best["model"] = run_dict["model"]
            best["val_score"] = run_dict["val_score"]
            best["hyperparams"] = run_dict["hyperparams"]

    ### logging
    if verbose > 1:
        print("  " + "\n" + "-" * 3 + f" Best: {best['model_name']} " + "-" * 3)
        print(f"  Hyperparameters: {' '.join([f'{k}={v}' for k, v in best['hyperparams'].items()])}")
        print(f"  Validation {cfg['scoring_fn'].__name__}: {best['val_score']:.4f}")
        print("  " + "-" * (14 + len(best['model_name'])))

    return best, model_runs


def main():
    """ Run the main pipeline """
    ### prepare saving
    cfg["run_name"] = now_str()
    cfg["dir_name"] = os.path.join(cfg["clean_data_path"], "runs", cfg["run_name"])
    os.makedirs(cfg["dir_name"], exist_ok=True)

    ### resave raw data as npy for faster loading
    print("Saving the raw data into .npy files for faster loading...")
    resave_csv_as_npy(cfg["raw_data_path"])

    ### find best [data processing]-[model] combination
    best_data_model_comb_dict = {"best_model": None, "best_data": None, "val_score": 0}
    for data_preproc_name, data_preproc_kwargs in runs["data"].items():
        print("\n" + "-" * 5 + f" {data_preproc_name} " + "-" * 5)

        ### get data
        x, x_final, y, ids, ids_final, col_idx_map, cleaned_col_idx_map = get_all_data(cfg=cfg, pca_kwargs=data_preproc_kwargs, verbose=False)

        ### get best model for this data
        best_model_dict, results = get_best_model(model_runs=deepcopy(runs["models"]), x=x, y=y, verbose=1)
        if best_model_dict["val_score"] > best_data_model_comb_dict["val_score"]:
            best_data_model_comb_dict = best_model_dict
            best_data_model_comb_dict["data_name"] = data_preproc_name
            best_data_model_comb_dict["data_kwargs"] = data_preproc_kwargs

        ### save results
        if cfg["dir_name"] is not None:
            with open(os.path.join(cfg["dir_name"], f"{data_preproc_name.replace(' ', '_')}_results.pkl"), "wb") as f:
                pickle.dump(results, f)
            with open(os.path.join(cfg["dir_name"], f"{data_preproc_name.replace(' ', '_')}_best.pkl"), "wb") as f:
                pickle.dump(best_model_dict, f)

    ### print best data-model combination
    print("\n" + "=" * 20)
    print("Best data-model combination found:")
    print(f"  Data: {best_data_model_comb_dict['data_name']}")
    print(f"  Model: {best_data_model_comb_dict['model_name']}")
    print(f"  Validation {cfg['scoring_fn'].__name__}: {best_data_model_comb_dict['val_score']:.4f}")
    print(f"  Hyperparameters: {' '.join([f'{k}={v}' for k, v in best_data_model_comb_dict['hyperparams'].items()])}")
    print("=" * 20 + "\n")

    ### save best data-model combination
    if cfg["dir_name"] is not None:
        with open(os.path.join(cfg["dir_name"], "best_data_model_comb.pkl"), "wb") as f:
            pickle.dump(best_data_model_comb_dict, f)

    ### predict on final data
    print("Predicting on final data...")
    _, x_final, _, _, ids_final, _, cleaned_col_idx_map = get_all_data(cfg=cfg, pca_kwargs=best_data_model_comb_dict["data_kwargs"], verbose=False)
    y_final_pred = best_data_model_comb_dict["model"].predict(x_final)
    if cfg["remap_labels_to_01"]:
        ### remap labels back to -1, 1
        y_final_pred = (y_final_pred * 2 - 1).astype(int)

    ### save submission
    save_path = os.path.join(cfg["dir_name"], f"{best_data_model_comb_dict['model_name']}_submission.csv")
    create_csv_submission(
        ids=ids_final,
        y_pred=y_final_pred,
        name=save_path,
    )
    print(f"Submission saved to: {save_path}")


if __name__ == "__main__":
    main()
