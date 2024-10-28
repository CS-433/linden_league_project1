import os
import numpy as np
import pickle
from copy import deepcopy

from models import LogisticRegression, DecisionTreeBinaryClassifier, SVM, Ensemble
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
if cfg["train"].get("cv", None) is not None:
    cfg["train"]["cv"]["scoring_fn"] = cfg["scoring_fn"]

### data-model combinations to run
runs = {
    "data": {
        "All columns": {"process_cols": "all", "pca_kwargs": None},
        "Selected columns": {"process_cols": "selected", "pca_kwargs": None},
        "Selected columns + Remaining columns PCA var > 0.99": {"process_cols": "selected", "pca_kwargs": {"min_explained_variance": 0.99, "max_frac_of_nan": 0.8}},
        "Selected columns + All columns PCA var > 0.99": {"process_cols": "selected", "pca_kwargs": {"min_explained_variance": 0.99, "max_frac_of_nan": 0.8, "all_cols": True}},

        "Raw": {"process_cols": "all", "pca_kwargs": None, "only_impute": True, "skip_rule_transformations": True},

        # "70% of columns": {"process_cols": 70, "pca_kwargs": None},
        # "35% of columns": {"process_cols": 35, "pca_kwargs": None},
        # "Selected columns + Remaining columns PCA var > 0.8": {"process_cols": "selected", "pca_kwargs": {"min_explained_variance": 0.8, "max_frac_of_nan": 0.8}},
        # "Selected columns + All columns PCA var > 0.8": {"process_cols": "selected", "pca_kwargs": {"min_explained_variance": 0.8, "max_frac_of_nan": 0.8, "all_cols": True}},
        
        # "70% of columns + PCA var > 0.6": {"process_cols": 70, "pca_kwargs": {"min_explained_variance": 0.6, "max_frac_of_nan": 0.8}},
        # "35% of columns + PCA var > 0.6": {"process_cols": 35, "pca_kwargs": {"min_explained_variance": 0.6, "max_frac_of_nan": 0.8}},

        # "70% of columns + PCA var > 0.85": {"process_cols": 70, "pca_kwargs": {"min_explained_variance": 0.85, "max_frac_of_nan": 0.8}},
        # "35% of columns + PCA var > 0.85": {"process_cols": 35, "pca_kwargs": {"min_explained_variance": 0.85, "max_frac_of_nan": 0.8}},

        # "70% of columns + PCA var > 0.99": {"process_cols": 70, "pca_kwargs": {"min_explained_variance": 0.99, "max_frac_of_nan": 0.8}},
        # "35% of columns + PCA var > 0.99": {"process_cols": 35, "pca_kwargs": {"min_explained_variance": 0.99, "max_frac_of_nan": 0.8}},
    },
    "models": {
        ### Table I
        # "Logistic Regression": {
        #     "model_cls": LogisticRegression,
        #     "hyperparam_search": {
        #         "gamma": [None],
        #         "use_line_search": [True],
        #         "optim_algo": ["lbfgs"],
        #         "optim_kwargs": [{"epochs": 1}],
        #         "class_weights": [{0: 1, 1: i} for i in [1, 2, 4, 6, 8]],
        #         "reg_mul": [0, 1e-4, 1e-2, 1],
        #         "verbose": [False],
        #     },
        # },
        # "SVM": {
        #     "model_cls": SVM,
        #     "hyperparam_search": {
        #         "_lambda": [1e-4, 1e-3, 1e-2, 1],
        #         "class_weights": [{0: 1, 1: i} for i in [1, 2, 4, 6, 8]],
        #     }
        # },
        
        # "Decision Tree": {
        #     "model_cls": DecisionTreeBinaryClassifier,
        #     "hyperparam_search": {
        #         "max_depth": [3, 5, 7],
        #         "min_samples_split": [5],
        #         "criterion": ["gini"],
        #         "class_weights": [{0: 1, 1: i} for i in [1, 2, 3, 4, 5, 6]],
        #         "eval_frac_of_features": [0.3],
        #         "eval_max_n_thresholds_per_split": [4],
        #     },
        # },

        # "Logistic Regression (baseline)": {
        #     "model_cls": LogisticRegression,
        #     "hyperparam_search": {
        #         "gamma": [None],
        #         "use_line_search": [True],
        #         "optim_algo": ["gd"],
        #         "optim_kwargs": [{"epochs": 200}],
        #         "class_weights": [{0: 1, 1: 1}],
        #         "reg_mul": [0],
        #         "verbose": [False],
        #     },
        # },
        "Logistic Regression (baseline)": {
            "model_cls": LogisticRegression,
            "hyperparam_search": {
                "gamma": [1, 5e-1, 1e-1, 5e-2, 1e-2],
                "use_line_search": [False],
                "optim_algo": ["gd"],
                "optim_kwargs": [{"epochs": 150}, {"epochs": 250}],
                "class_weights": [{0: 1, 1: 1}],
                "reg_mul": [0],
                "verbose": [False],
            },
        },
        # "SVM (baseline)": {
        #     "model_cls": SVM,
        #     "hyperparam_search": {
        #         "_lambda": [1e-4, 1e-3, 1e-2, 1],
        #         "class_weights": [{0: 1, 1: 1}],
        #     }
        # },

        # "Ensemble": {
        #     "model_cls": Ensemble,
        #     "hyperparam_search": [
        #         # {
        #         #     "model_cls_list": [LogisticRegression, LogisticRegression, LogisticRegression],
        #         #     "model_kwargs_list": [
        #         #         {"gamma": None, "use_line_search": True, "optim_algo": "lbfgs", "optim_kwargs": {"epochs": 1}, "class_weights": {0: 1, 1: 3.5}, "reg_mul": 1e-4, "verbose": False},
        #         #         {"gamma": None, "use_line_search": True, "optim_algo": "lbfgs", "optim_kwargs": {"epochs": 1}, "class_weights": {0: 1, 1: 4}, "reg_mul": 1e-4, "verbose": False},
        #         #         {"gamma": None, "use_line_search": True, "optim_algo": "lbfgs", "optim_kwargs": {"epochs": 1}, "class_weights": {0: 1, 1: 4.5}, "reg_mul": 1e-4, "verbose": False},
        #         #     ],
        #         #     "fit_frac_per_model_majority": 1.,
        #         # },
        #         {
        #             "model_cls_list": [LogisticRegression, LogisticRegression, LogisticRegression, LogisticRegression, LogisticRegression],
        #             "model_kwargs_list": [
        #                 {"gamma": None, "use_line_search": True, "optim_algo": "lbfgs", "optim_kwargs": {"epochs": 1}, "class_weights": {0: 1, 1: 4}, "reg_mul": 1e-4, "verbose": False, "init_w": "normal"},
        #                 {"gamma": None, "use_line_search": True, "optim_algo": "lbfgs", "optim_kwargs": {"epochs": 1}, "class_weights": {0: 1, 1: 4}, "reg_mul": 1e-4, "verbose": False, "init_w": "normal"},
        #                 {"gamma": None, "use_line_search": True, "optim_algo": "lbfgs", "optim_kwargs": {"epochs": 1}, "class_weights": {0: 1, 1: 4}, "reg_mul": 1e-4, "verbose": False, "init_w": "normal"},
        #                 {"gamma": None, "use_line_search": True, "optim_algo": "lbfgs", "optim_kwargs": {"epochs": 1}, "class_weights": {0: 1, 1: 4}, "reg_mul": 1e-4, "verbose": False, "init_w": "normal"},
        #                 {"gamma": None, "use_line_search": True, "optim_algo": "lbfgs", "optim_kwargs": {"epochs": 1}, "class_weights": {0: 1, 1: 4}, "reg_mul": 1e-4, "verbose": False, "init_w": "normal"},
        #             ],
        #             "fit_frac_per_model_majority": 1.,
        #         },
        #         {
        #             "model_cls_list": [LogisticRegression, LogisticRegression, LogisticRegression, LogisticRegression, LogisticRegression],
        #             "model_kwargs_list": [
        #                 {"gamma": None, "use_line_search": True, "optim_algo": "lbfgs", "optim_kwargs": {"epochs": 1}, "class_weights": {0: 1, 1: 4}, "reg_mul": 5e-4, "verbose": False},
        #                 {"gamma": None, "use_line_search": True, "optim_algo": "lbfgs", "optim_kwargs": {"epochs": 1}, "class_weights": {0: 1, 1: 4}, "reg_mul": 3e-4, "verbose": False},
        #                 {"gamma": None, "use_line_search": True, "optim_algo": "lbfgs", "optim_kwargs": {"epochs": 1}, "class_weights": {0: 1, 1: 4}, "reg_mul": 1e-4, "verbose": False},
        #                 {"gamma": None, "use_line_search": True, "optim_algo": "lbfgs", "optim_kwargs": {"epochs": 1}, "class_weights": {0: 1, 1: 4}, "reg_mul": 8e-5, "verbose": False},
        #                 {"gamma": None, "use_line_search": True, "optim_algo": "lbfgs", "optim_kwargs": {"epochs": 1}, "class_weights": {0: 1, 1: 4}, "reg_mul": 5e-5, "verbose": False},
        #             ],
        #             "fit_frac_per_model_majority": 1.,
        #         },
        #     ]
        # },
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
    best = {"model_name": None, "model": None, "val_score": -1, "hyperparams": None}
    for name, run_dict in model_runs.items():
        if verbose: print("  " + "-" * 3 + f" {name} " + "-" * 3)

        ### prepare hyperparam search space
        if type(run_dict["hyperparam_search"]) == dict:
            hyperparam_search = prep_hyperparam_search(run_dict["hyperparam_search"])
        elif type(run_dict["hyperparam_search"]) == list:
            hyperparam_search = run_dict["hyperparam_search"]
        else:
            raise ValueError("Invalid hyperparam search type")
        if verbose > 1: print(f"  Searching hyperparameters among {len(hyperparam_search)} options...")

        ### choose best hyperparameters using validation (CV, holdout, or none)
        seed_all(cfg["seed"])
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

    ### save config
    with open(os.path.join(cfg["dir_name"], "config.pkl"), "wb") as f:
        pickle.dump(cfg, f)

    ### resave raw data as npy for faster loading
    print("Saving the raw data into .npy files for faster loading...")
    resave_csv_as_npy(data_path=cfg["raw_data_path"], transform_values=True)

    ### find best [data processing]-[model] combination
    best_data_model_comb_dict = {"best_model": None, "best_data": None, "val_score": -1}
    for data_preproc_name, data_preproc_kwargs in runs["data"].items():
        print("\n" + "-" * 5 + f" {data_preproc_name} " + "-" * 5)

        ### get data
        seed_all(cfg["seed"])
        x, x_final, y, ids, ids_final, col_idx_map, cleaned_col_idx_map = get_all_data(
            cfg=cfg,
            process_cols=data_preproc_kwargs.get("process_cols", None),
            pca_kwargs=data_preproc_kwargs.get("pca_kwargs", None),
            only_impute=data_preproc_kwargs.get("only_impute", False),
            skip_rule_transformations=data_preproc_kwargs.get("skip_rule_transformations", False),
            verbose=False,
        )

        ### get best model for this data
        best_model_dict, results = get_best_model(model_runs=deepcopy(runs["models"]), x=x, y=y, verbose=1)
        if best_model_dict["val_score"] > best_data_model_comb_dict["val_score"]:
            best_data_model_comb_dict = deepcopy(best_model_dict)
            best_data_model_comb_dict["data_name"] = data_preproc_name
            best_data_model_comb_dict["data_kwargs"] = data_preproc_kwargs
            best_data_model_comb_dict["final_data"] = (x_final, ids_final, cleaned_col_idx_map)

        ### save results
        if cfg["dir_name"] is not None:
            with open(os.path.join(cfg["dir_name"], f"{data_preproc_name.replace(' ', '_')}__{best_model_dict['model_name'].replace(' ', '_')}__results.pkl"), "wb") as f:
                pickle.dump(results, f)
            with open(os.path.join(cfg["dir_name"], f"{data_preproc_name.replace(' ', '_')}__{best_model_dict['model_name'].replace(' ', '_')}__best.pkl"), "wb") as f:
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
    seed_all(cfg["seed"])
    x_final, ids_final, _ = best_data_model_comb_dict["final_data"]
    y_final_pred = best_data_model_comb_dict["model"].predict(x_final)
    if cfg["remap_labels_to_01"]:
        ### remap labels back to -1, 1
        y_final_pred = (y_final_pred * 2 - 1).astype(int)

    ### save submission
    save_path = os.path.join(cfg["dir_name"], f"{best_data_model_comb_dict['model_name'].replace(' ', '_')}_submission.csv")
    create_csv_submission(
        ids=ids_final,
        y_pred=y_final_pred,
        name=save_path,
    )
    print(f"Submission saved to: {save_path}")


if __name__ == "__main__":
    main()
