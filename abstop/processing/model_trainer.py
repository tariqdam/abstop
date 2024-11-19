import logging
import os
from logging import Logger
import copy

import optuna
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, \
    recall_score, f1_score, confusion_matrix, balanced_accuracy_score, \
    average_precision_score, make_scorer
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
import shap as shap

import cloudpickle as cloudpickle
from pypdf import PdfWriter

from typing import Any, Union
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin


import tadam as td
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import lightgbm as lgb
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt

from abstop.config import Config

logger: Logger = logging.getLogger(__name__)


CV_SCORERS = {
    "accuracy": make_scorer(accuracy_score),
    "balanced_accuracy_score": make_scorer(balanced_accuracy_score),
    "rocauc": make_scorer(roc_auc_score, response_method="predict_proba"),
    "average_precision": make_scorer(
        average_precision_score, response_method="predict_proba"
    ),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "specificity": make_scorer(recall_score, pos_label=0),
    "f1": make_scorer(f1_score),
}

def save_to_file(path: Union[str, Path], model: Any) -> Any:
    with open(path, "wb") as f:
        return cloudpickle.dump(model, f)


def load_from_file(path: Union[str, Path]) -> Any:
    with open(path, "rb") as f:
        return cloudpickle.load(f)


def rm(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min == 0 and c_max == 0:
                df[col] = df[col].astype(np.int8)
            elif c_min == 1 and c_max == 1:
                df[col] = df[col].astype(np.int8)
            elif c_min == 0 and c_max == 1:
                df[col] = df[col].astype(np.int8)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

class ModelTrainer:
    """
    Constructs aggregated features based on events and measurements.

    The feature selector class is initialized with a Config object. This object
    contains the settings for the experiment.

    The run() method transforms incoming hospital data into a usable format for the
    experiment. The resulting table is saved to the processed data directory.
    """

    def __init__(self, config: Config):
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.logger.debug(f"from {__name__} instantiate {self.__class__.__name__}")
        self.config = config
        self._cfg = self.config.settings.model_trainer

        self.X_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.X_val = pd.DataFrame()
        self.y_val = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_test = pd.DataFrame()

        self.evaluator: KFoldEvaluator | None = None
        self.kfe_results = {}



    def run(self) -> None:
        """
        Run the Feature Selector with the settings defined in the config file.
        :return: pd.DataFrame
        """

        self.logger.info("Running model trainer")

        # get data
        df = self._load_data(
            "model_data"
        )

        df = df[self.config.settings.feature_selector.c.model_columns].copy()

        df = df.astype(float).copy()

        # split data
        random_state = 168
        ## Commented code was used to find a stratified split which kept the outcome labels as balanced as possible
        ### because of the grouping structure, stratified splits are approximated, but still resulted in splits where
        ### the outcome label was disbalanced (20% vs. 16%). By finding the split seed with the smallest standard
        ### deviation, we can find a split which has an as equal as possible split.
        # results = {}
        # best_trial = 0
        # lowest_std = 1
        # for i in range(1001):
        #     print(f"Running {i}/1000")
        #     self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = (
        #         StratifiedGroupShuffleSplit(  # StratifiedGroupShuffleSplit( # grouped_train_val_test_split(
        #             data=df,
        #             group="pid",
        #             label=self.config.settings.feature_selector.c.outcomes[0],
        #             random_state=i,
        #             val_size=0.2,
        #             test_size=0.2,
        #         )
        #     )
        #     results[i] = {
        #         "train": self.y_train.mean(),
        #         "test": self.y_test.mean(),
        #         "val": self.y_val.mean(),
        #     }
        #     std = np.std(list(results[i].values()))
        #     if std < lowest_std:
        #         print(f"{i} ({std}): {results[i]}")
        #         lowest_std = std
        #         best_trial = i
        # print(results)
        # print(best_trial, lowest_std, results[best_trial])
        # random_state = best_trial

        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = (
            StratifiedGroupShuffleSplit( # StratifiedGroupShuffleSplit( # grouped_train_val_test_split(
                data=df,
                group="pid",
                label=self.config.settings.feature_selector.c.outcomes[0],
                random_state=random_state,
                val_size=0.2,
                test_size=0.2,
            )
        )

        print("="*30)
        print(f"""
        {random_state = }
        {self.y_train.mean() = }
        {self.y_val.mean() = }
        {self.y_test.mean() = }
        {np.std([self.y_train.mean(), self.y_val.mean(), self.y_test.mean()]) = }
        """)
        print("=" * 30)

        self.evaluator = KFoldEvaluator(
            X=self.X_test,
            y=self.y_test,
            threshold=self.y_train.mean(),
            k_folds=10,
            random_state=self.config.settings.model_trainer.seed,
        )

        td.dump((self.X_train, self.y_train), os.path.join(self.config.directory("models"), "train_data.pkl"))
        td.dump((self.X_val, self.y_val), os.path.join(self.config.directory("models"), "val_data.pkl"))
        td.dump((self.X_test, self.y_test), os.path.join(self.config.directory("models"), "test_data.pkl"))

        for model in self._cfg.models:
            self.logger.info(f"Running model {model['model']}")
            self._run_step(model)

        kfe_results = self.evaluator.transform_results(self.kfe_results)
        kfe_results.to_csv(
            os.path.join(self.config.directory("models"), "kfe_results.csv"),
            float_format="%.3f"
        )
    # build model
    # evaluate model
    # save model
    # save scaler
    # save imputer

    # next model type

    def _load_data(self, name: str) -> pd.DataFrame:
        """
        Load data from file.
        :param name: string name of dictionary key in config file
        :return: pd.DataFrame
        """
        _path, _file = self._cfg.files.get(name, ("processed", None))
        if _file:
            file_path = os.path.join(self.config.directory(_path), _file)
            if file_path.endswith(".csv"):
                return pd.read_csv(file_path)
            else:
                return td.load(file_path)
        else:
            raise ValueError(f"no file found for {name}")

    def _build_model(self, model: str):
        if model == "LogisticRegression":
            return LogisticRegression()
        elif model == "LGBMClassifier":
            return LGBMClassifier(importance_type='gain')
        elif model == "SVM":
            return SVC(probability=True)
        else:
            raise ValueError(f"Model {model} not supported")

    def _run_step(self, step: dict) -> None:
        """
        Run the Feature Selector with the settings defined in the config file.
        :return: pd.DataFrame
        """

        pipeline = self._get_pipeline(step)

        param_grid = self._build_param_grid(step)

        self._run_grid_search(
            step,
            pipeline,
            param_grid
        )

    def grabdict(self, tisdict, tree_index, split_index, depth, splits, leaves):
        # recursive function to unravel nested dictionaries
        depth += 1
        if 'split_index' in tisdict.keys():
            tis = tisdict.copy()
            del tis['left_child']
            del tis['right_child']
            tis['tree_index'] = tree_index
            split_index = tis['split_index']
            splits = pd.concat([splits, pd.DataFrame(tis, index=[len(splits)])])
            splits, leaves = self.grabdict(tisdict['left_child'], tree_index, split_index, depth, splits, leaves)
            splits, leaves = self.grabdict(tisdict['right_child'], tree_index, split_index, depth, splits, leaves)
        else:
            tis = tisdict.copy()
            tis['tree_index'] = tree_index
            tis['split_index'] = split_index
            tis['depth'] = depth
            leaves = pd.concat([leaves, pd.DataFrame(tis, index=[len(leaves)])])
        return splits, leaves

    def grabtrees(self, model):
        # wrapper function to call grabdict
        splits, leaves = pd.DataFrame(), pd.DataFrame()
        tree_info = model.dump_model()['tree_info']
        for tisdict in tree_info:
            splits, leaves = self.grabdict(tisdict['tree_structure'], tisdict['tree_index'], 0, 0, splits, leaves)
        leaves = leaves.merge(splits, left_on=['tree_index', 'split_index'], right_on=['tree_index', 'split_index'],
                              how='left')
        return tree_info, leaves

    def _run_grid_search(self, step, pipeline: Pipeline, param_grid: dict) -> None:
        """
        Run the Feature Selector with the settings defined in the config file.
        :return: pd.DataFrame
        """

        name = step["model"]

        self.logger.info(f"Running grid search for {name}")

        use_optuna = True
        # 2. Fit model
        if use_optuna:

            parameters_base = {}
            if name in ['LGBMClassifier']:
                parameters_base.update({
                    'classifier__scale_pos_weight': [param_grid['classifier__scale_pos_weight'][0]],
                    'classifier__learning_rate': [0.01],
                    'classifier__objective': ['binary'],
                    'classifier__metric': ['binary__logloss'],
                    'classifier__random_state': [42],
                    'classifier__n_jobs': [-1],
                    'classifier__device': ['gpu'],
                    'classifier__importance_type': ['gain'],
                })
            elif name in ['LogisticRegression']:
                parameters_base.update({
                    'classifier__max_iter': [1000],
                    'classifier__n_jobs': [-1],
                    'classifier__random_state': [42],
                    'classifier__class_weight': ['balanced'],
                })
            elif name in ['SVM']:
                parameters_base.update({
                    'classifier__class_weight': ['balanced'],
                    'classifier__random_state': [42],
                    'classifier__max_iter': [-1],
                    'classifier__probability': [True],
                })

            def objective(trial):

                if name in ['LGBMClassifier']:
                    param = {
                        'classifier': LGBMClassifier(),
                        'classifier__lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
                        'classifier__lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
                        'classifier__num_leaves': trial.suggest_int('num_leaves', 2, 256),
                        'classifier__feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
                        'classifier__bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
                        'classifier__bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                        'classifier__min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
                    }
                elif name in ['LogisticRegression']:
                    solver = trial.suggest_categorical('solver', ['liblinear', 'saga', 'newton-cg', 'lbfgs'])
                    # Ensure compatibility between solver and penalty
                    if solver in ['saga']:
                        penalty = trial.suggest_categorical('penalty__llsaga', ['elasticnet', 'l1', 'l2', None])
                    elif solver in ['liblinear']:
                        penalty = trial.suggest_categorical('penalty__liblinear', ['l1', 'l2'])
                    else:
                        penalty = trial.suggest_categorical('penalty__other', ['l2', None])

                    if penalty == 'elasticnet':
                        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
                    else:
                        l1_ratio = None
                    param = {
                        'classifier': LogisticRegression(),
                        'classifier__C': trial.suggest_float('C', 1e-10, 1e3, log=True),
                        'classifier__solver': solver,
                        'classifier__penalty': penalty,
                        'classifier__l1_ratio': l1_ratio,
                    }
                elif name in ['SVM']:
                    param = {
                        'classifier': SVC(),
                        'classifier__C': trial.suggest_float('C', 1e-10, 1e3, log=True),
                        'classifier__gamma': trial.suggest_float('gamma', 1e-10, 1e3, log=True),
                        'classifier__kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
                        'classifier__degree': trial.suggest_int('degree', 2, 4),
                    }
                else:
                    raise ValueError(f"No eligible model selected. {name} is not supported in optuna.")
                base_parameters = {k: v[0] for k,v in parameters_base.items()}
                param.update(base_parameters)
                pl = copy.deepcopy(pipeline)
                pl.set_params(**param)

                if name in ['LGBMClassifier']:
                    pl.fit(
                        X=self.X_train.copy(),
                        y=self.y_train.copy(),
                        classifier__eval_set=[(self.X_train.copy(), self.y_train.copy()),
                                              (self.X_val.copy(), self.y_val.copy())],
                        classifier__eval_names=["train", "val"],
                        classifier__eval_metric="auc",
                        classifier__callbacks=[
                            lgb.early_stopping(
                                stopping_rounds=20,
                                first_metric_only=True,
                                verbose=True,
                                min_delta=0.0,
                            ),
                            lgb.log_evaluation(period=0)
                        ],
                       )
                else:
                    pl.fit(X=self.X_train.copy(), y=self.y_train.copy())
                y_proba = pl.predict_proba(X=self.X_val)[:, 1]
                return roc_auc_score(y_true=self.y_val, y_score=y_proba)

            study_db = os.path.join(self.config.directory('models'), f"{self.config.experiment_name}.db")
            study = optuna.create_study(
                storage=self.config.optuna_db_uri,
                study_name=f"{self.config.experiment_name}__{name}",
                direction="maximize",
                load_if_exists=True,
            )

            n_total_trials = 300 # 300
            trials_completed = len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))
            remaining_trials = max(n_total_trials - trials_completed, 0)
            study.optimize(
                objective,
                n_trials=remaining_trials,
                n_jobs=-1)
            print('Number of finished trials:', len(study.trials))
            print('Best trial:', study.best_trial.params)
            self.logger.critical(f'Number of finished trials: {len(study.trials)}')
            self.logger.critical(f'Best trial: {study.best_trial.params}')
            param_grid = {f"classifier__{k.split('__')[0]}": [v] for k, v in study.best_trial.params.items()}
            param_grid.update(parameters_base)

        clf = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=CV_SCORERS,
            refit="rocauc",
            cv=10,
            verbose=1,
            n_jobs=-1,
            return_train_score=True,
            error_score=0.0,
        )

        if name in ["LGBMClassifier"]:
            # old code; keep 20240730
            best_clf = clf.fit(
                X=self.X_train.copy(),
                y=self.y_train.copy(),
                classifier__eval_set=[(self.X_train.copy(), self.y_train.copy()), (self.X_val.copy(), self.y_val.copy())],
                classifier__eval_names=["train", "val"],
                classifier__eval_metric="auc",
                classifier__callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=20,
                        first_metric_only=True,
                        verbose=True,
                        min_delta=0.0,
                    ),
                    lgb.log_evaluation(period=0)
                ],
            )
        else:
            best_clf = clf.fit(self.X_train.copy(), self.y_train.copy())

        self.logger.info(f"Grid search completed for {name}")
        self.logger.info(f"Best params: {best_clf.best_params_}")

        scores = pd.DataFrame(best_clf.cv_results_).sort_values(by=["rank_test_rocauc"], key=abs, ascending=True)
        print(type(scores))
        print(scores.shape)
        # self.logger.debug(f"CV results: {scores}")
        self.logger.debug("Saving results to file")
        scores.to_csv(os.path.join(self.config.directory("models"), f"{name}__cv_results.csv"))

        self.logger.debug("Saving model to file")
        save_to_file(path=os.path.join(self.config.directory("models"), f"{name}__model.pkl"), model=best_clf)
        # td.dump(best_clf, os.path.join(self.config.directory("models"), f"{name}__model.pkl"))

        if name in ['LGBMClassifier']:
            self.logger.debug("New method for lgbm")
            tree_info, leaves = self.grabtrees(best_clf.best_estimator_.named_steps["classifier"].booster_)
            leaves.to_csv(os.path.join(self.config.directory("models"), f"{name}__tree_info.csv"))

            # Use only for debugging with low n_estimators or select only the last tree
            # for tree_index in range(len(tree_info)):
            #
            #     fig, ax = plt.subplots()
            #     ax = lgb.plot_tree(
            #         booster=best_clf.best_estimator_.named_steps["classifier"].booster_, ax=ax, tree_index=tree_index, figsize=(30, 15)
            #     )
            #     fig.savefig(os.path.join(self.config.directory("models"), f"{name}__tree__{tree_index}.pdf"), dpi=1200)
            #
            # tree_plots = glob.glob(os.path.join(self.config.directory("models"), f"{name}__tree__*.pdf"))
            # self.merge_pdfs(pdfs=tree_plots, filename=os.path.join(self.config.directory("models"), f"{name}__tree.pdf"))


        self.logger.debug("Evaluating model")
        self.evaluate_model(step, best_clf)

    def merge_pdfs(self, pdfs, filename, suffix="", keep_original=False):
        """Search, combine and delete matching pdf files."""
        merger = PdfWriter()
        saved_plots = sorted(pdfs)
        for pdf in saved_plots:
            merger.append(pdf)
        merger.write(filename)
        merger.close()
        if not keep_original:
            for pdf in saved_plots:
                os.remove(pdf)

    def evaluate_model(self, step, model: GridSearchCV):
        """
        Evaluate the model.
        :param df: pd.DataFrame
        :return: pd.DataFrame
        """
        self.logger.info("Evaluating model")
        y_test_pred = model.predict(self.X_test)
        y_test_pred_proba = model.predict_proba(self.X_test)[:, 1]

        y_train_pred = model.predict(self.X_train)
        y_train_pred_proba = model.predict_proba(self.X_train)[:, 1]

        y_val_pred = model.predict(self.X_val)
        y_val_pred_proba = model.predict_proba(self.X_val)[:, 1]

        scores = {
            "train": {
                "accuracy": accuracy_score(self.y_train, y_train_pred),
                "balanced_accuracy_score": balanced_accuracy_score(self.y_train, y_train_pred),
                "roc_auc": roc_auc_score(self.y_train, y_train_pred_proba),
                "average_precision": average_precision_score(self.y_train, y_train_pred_proba),
                "precision": precision_score(self.y_train, y_train_pred),
                "recall": recall_score(self.y_train, y_train_pred),
                "specificity": recall_score(self.y_train, y_train_pred, pos_label=0),
                "f1": f1_score(self.y_train, y_train_pred),
                "positive": self.y_train.sum(),
                "negative": len(self.y_train) - self.y_train.sum(),
                "all": len(self.y_train),
            },
            "test": {
                "accuracy": accuracy_score(self.y_test, y_test_pred),
                "balanced_accuracy_score": balanced_accuracy_score(self.y_test,
                                                                   y_test_pred),
                "roc_auc": roc_auc_score(self.y_test, y_test_pred_proba),
                "average_precision": average_precision_score(self.y_test,
                                                             y_test_pred_proba),
                "precision": precision_score(self.y_test, y_test_pred),
                "recall": recall_score(self.y_test, y_test_pred),
                "specificity": recall_score(self.y_test, y_test_pred, pos_label=0),
                "f1": f1_score(self.y_test, y_test_pred),
                "positive": self.y_test.sum(),
                "negative": len(self.y_test) - self.y_test.sum(),
                "all": len(self.y_test),
            },
            "val": {
                "accuracy": accuracy_score(self.y_val, y_val_pred),
                "balanced_accuracy_score": balanced_accuracy_score(self.y_val,
                                                                   y_val_pred),
                "roc_auc": roc_auc_score(self.y_val, y_val_pred_proba),
                "average_precision": average_precision_score(self.y_val,
                                                             y_val_pred_proba),
                "precision": precision_score(self.y_val, y_val_pred),
                "recall": recall_score(self.y_val, y_val_pred),
                "specificity": recall_score(self.y_val, y_val_pred, pos_label=0),
                "f1": f1_score(self.y_val, y_val_pred),
                "positive": self.y_val.sum(),
                "negative": len(self.y_val) - self.y_val.sum(),
                "all": len(self.y_val),
            },
        }

        for t, kv in scores.items():
            for k, v in kv.items():
                self.logger.info(f"{t} - {k}: {v}")

        pd.DataFrame(scores).to_csv(os.path.join(self.config.directory("models"), f"{step['model']}__scores.csv"))

        self.logger.info(f"Test - Confusion matrix: {confusion_matrix(self.y_test, y_test_pred)}")
        self.logger.info(f"Train - Confusion matrix: {confusion_matrix(self.y_train, y_train_pred)}")

        rn = self.config.settings.feature_selector.c.plot_names

        if step["model"] == "LogisticRegression":
            # get coefficients
            coefs = pd.DataFrame(
                model.best_estimator_.named_steps['classifier'].coef_,
                columns=[rn.get(x, x) for x in list(model.feature_names_in_)],
            ).T.sort_values(0, ascending=False, key=abs)
            coefs.to_csv(os.path.join(self.config.directory("models"), f"{step['model']}__coefs.csv"))
            self.logger.debug(f"Coefficients: {coefs}")
        elif step["model"] == "SVM":
            kernel = model.best_estimator_.named_steps['classifier'].kernel

            if kernel == "rbf":
                from sklearn.inspection import permutation_importance
                r = permutation_importance(model,
                                           self.X_val,
                                           self.y_val,
                                           n_repeats=30,
                                           random_state=0)

                df_perm = pd.DataFrame(
                    [r.importances_mean, r.importances_std],
                    index=['mean', 'std'],
                    columns=[rn.get(k, k) for k in model.feature_names_in_],
                ).T.sort_values(['mean'], key=abs, ascending=[False])

                df_perm.to_csv(
                    os.path.join(
                        self.config.directory('models'),
                        f"{step['model']}__permutation_importances.csv"
                    )
                )
            elif kernel == "linear":
                # coefs
                m: SVC = model.best_estimator_.named_steps['classifier']
                coefs = pd.DataFrame(
                    m.coef_,
                    columns=[rn.get(x, x) for x in list(model.feature_names_in_)],
                ).T.sort_values(0, ascending=False, key=abs)
                coefs.to_csv(
                    os.path.join(
                        self.config.directory("models"),
                        f"{step['model']}__coefs.csv"
                    )
                )

        elif step["model"] == "LGBMClassifier":

            metric = "auc"

            # plot eval progress
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            ax = lgb.plot_metric(
                booster=model.best_estimator_.named_steps["classifier"],
                metric=metric,
                ax=ax,
                figsize=(10, 10),
                title=f"Evaluation results - {metric}",
            )
            fig.savefig(os.path.join(self.config.directory("models"), f"{step['model']}__eval_results.png"))
            plt.close(fig)


            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            ax = lgb.plot_importance(
                booster=model.best_estimator_.named_steps["classifier"],
                ax=ax,
                importance_type="gain",
                figsize=(10, 10),
                title="Feature importance (gain)",
            )

            __labels = [item.get_text() for item in ax.get_yticklabels()]
            __labels = [self.config.settings.feature_selector.c.plot_names.get(l, l) for l in __labels]
            ax.set_yticklabels(__labels)

            plt.tight_layout()

            fig.savefig(os.path.join(self.config.directory("models"), f"{step['model']}__feature_importance.png"))

            print(dir(model))
            print(model.best_estimator_.named_steps['classifier'].feature_importances_)

            # get feature importance
            feature_importance = pd.DataFrame(
                model.best_estimator_.named_steps['classifier'].feature_importances_,
                index=model.feature_names_in_,
                columns=["gain"],
            ).sort_values(by=["gain"], ascending=False, key=abs)
            plt.close(fig)

            ax2 = feature_importance["gain"].sort_values(ascending=True).plot(
                kind='barh',
                figsize = (10, 10),
                title="Feature importance (gain)",
            )

            __labels = [item.get_text() for item in ax2.get_yticklabels()]
            __labels = [self.config.settings.feature_selector.c.plot_names.get(l, l) for
                        l in __labels]
            ax2.set_yticklabels(__labels)
            f_2 = ax2.get_figure()

            plt.tight_layout()
            f_2.savefig(os.path.join(self.config.directory("models"), f"{step['model']}__feature_importance_bar.png"))
            plt.close(f_2)

            feature_importance.to_csv(
                os.path.join(
                    self.config.directory("models"),
                    f"{step['model']}__feature_importance.csv"
                )
            )
            self.logger.info(f"Feature importance: {feature_importance}")

            # self.plot_shap_values(model, step)

        else:
            raise ValueError(f"Model {step['model']} not supported")

        self.plot_shap_values(model, step)

        self.plot_calibration_curve(model, step)

        self.kfe_results[step['model']] = self.evaluator.evaluate(
            model=model.best_estimator_,
            calibrated=None,
        )

        calibrated_model = self.calibrate(model, step)

        self.logger.critical("CALIBRATED MODEL")

        self.kfe_results[step['model'] + "_calibrated"] = self.evaluator.evaluate(
            model=model.best_estimator_,
            calibrated=calibrated_model,
        )

        self.plot_calibration_curve(
            calibrated_model,
            step={"model": f"{step['model']}__calibrated"},
            uncalibrated_model=model,
        )

        threshold = self.y_train.mean()

        calibrated_y_train_pred = model.predict_proba(self.X_train)[:, 1]
        y_pred_proba_train_calibrated = calibrated_model.predict_proba(calibrated_y_train_pred.reshape(-1,1))[:, 1]
        y_pred_train_calibrated = y_pred_proba_train_calibrated > threshold

        calibrated_y_test_pred = model.predict_proba(self.X_test)[:, 1]
        y_pred_proba_test_calibrated = calibrated_model.predict_proba(
            calibrated_y_test_pred.reshape(-1, 1))[:, 1]
        y_pred_test_calibrated = y_pred_proba_test_calibrated > threshold

        calibrated_y_val_pred = model.predict_proba(self.X_val)[:, 1]
        y_pred_proba_val_calibrated = calibrated_model.predict_proba(
            calibrated_y_val_pred.reshape(-1, 1))[:, 1]
        y_pred_val_calibrated = y_pred_proba_val_calibrated > threshold

        scores = {
            "train": {
                "accuracy": accuracy_score(self.y_train, y_pred_train_calibrated),
                "balanced_accuracy_score": balanced_accuracy_score(self.y_train,
                                                                   y_pred_train_calibrated),
                "roc_auc": roc_auc_score(self.y_train, y_pred_proba_train_calibrated),
                "average_precision": average_precision_score(self.y_train,
                                                             y_pred_proba_train_calibrated),
                "precision": precision_score(self.y_train, y_pred_train_calibrated),
                "recall": recall_score(self.y_train, y_pred_train_calibrated),
                "specificity": recall_score(self.y_train, y_pred_train_calibrated, pos_label=0),
                "f1": f1_score(self.y_train, y_pred_train_calibrated),
                "positive": self.y_train.sum(),
                "negative": len(self.y_train) - self.y_train.sum(),
                "all": len(self.y_train),
            },
            "test": {
                "accuracy": accuracy_score(self.y_test, y_pred_test_calibrated),
                "balanced_accuracy_score": balanced_accuracy_score(self.y_test,
                                                                   y_pred_test_calibrated),
                "roc_auc": roc_auc_score(self.y_test, y_pred_proba_test_calibrated),
                "average_precision": average_precision_score(self.y_test,
                                                             y_pred_proba_test_calibrated),
                "precision": precision_score(self.y_test, y_pred_test_calibrated),
                "recall": recall_score(self.y_test, y_pred_test_calibrated),
                "specificity": recall_score(self.y_test, y_pred_test_calibrated,
                                            pos_label=0),
                "f1": f1_score(self.y_test, y_pred_test_calibrated),
                "positive": self.y_test.sum(),
                "negative": len(self.y_test) - self.y_test.sum(),
                "all": len(self.y_test),
            },
            "val": {
                "accuracy": accuracy_score(self.y_val, y_pred_val_calibrated),
                "balanced_accuracy_score": balanced_accuracy_score(self.y_val,
                                                                   y_pred_val_calibrated),
                "roc_auc": roc_auc_score(self.y_val, y_pred_proba_val_calibrated),
                "average_precision": average_precision_score(self.y_val,
                                                             y_pred_proba_val_calibrated),
                "precision": precision_score(self.y_val, y_pred_val_calibrated),
                "recall": recall_score(self.y_val, y_pred_val_calibrated),
                "specificity": recall_score(self.y_val, y_pred_val_calibrated,
                                            pos_label=0),
                "f1": f1_score(self.y_val, y_pred_val_calibrated),
                "positive": self.y_val.sum(),
                "negative": len(self.y_val) - self.y_val.sum(),
                "all": len(self.y_val),
            },
        }

        for t, kv in scores.items():
            for k, v in kv.items():
                self.logger.info(f"{t} - {k}: {v}")

        pd.DataFrame(scores).to_csv(os.path.join(self.config.directory("models"), f"{step['model']}__calibrated_scores.csv"))

    def calibrate(self, model, step):
        calibrated_model = CalibratedClassifierCV(
            estimator=LogisticRegression(),
            method="sigmoid",
            cv=10,
            n_jobs=-1,
        )
        y_pred_proba = model.predict_proba(self.X_train)[:, 1]
        calibrated_model.fit(y_pred_proba.reshape(-1, 1), self.y_train.values.reshape(-1, 1))
        save_to_file(path=os.path.join(self.config.directory("models"), f"{step['model']}__calibrated_model.pkl"), model=calibrated_model)
        # td.dump(calibrated_model, os.path.join(self.config.directory("models"), f"{step['model']}__calibrated_model.pkl"))
        return calibrated_model

    def plot_calibration_curve(self, model, step, uncalibrated_model = None):

        if uncalibrated_model is None:
            y_prob_test = model.predict_proba(self.X_test)[:, 1]
            y_prob_train = model.predict_proba(self.X_train)[:, 1]
        else:
            y_prob_test_uncalibrated = uncalibrated_model.predict_proba(self.X_test)[:, 1]
            y_prob_test = model.predict_proba(y_prob_test_uncalibrated.reshape(-1,1))[:, 1]

            y_prob_train_uncalibrated = uncalibrated_model.predict_proba(self.X_train)[:, 1]
            y_prob_train = model.predict_proba(y_prob_train_uncalibrated.reshape(-1,1))[:, 1]

        plt.close()

        fig = plot_calibration_curve(
            y_true=self.y_test,
            y_prob=y_prob_test,
            x_min=0.0,
            x_max=1.0,
        )
        if uncalibrated_model is None:
            filename = os.path.join(
                self.config.directory("models"),
                f"{step['model']}__calibration_curve__test"
            )
        else:
            filename = os.path.join(
                self.config.directory("models"),
                f"{step['model']}__calibration_curve__test__calibrated"
            )
        plt.tight_layout()
        # fig.savefig(filename + ".png", dpi=1200)
        fig.savefig(filename + ".pdf", dpi=1200)
        plt.close()

        fig = plot_calibration_curve(
            y_true=self.y_train,
            y_prob=y_prob_train,
            x_min=0.0,
            x_max=1.0,
        )
        if uncalibrated_model is None:
            filename = os.path.join(
                self.config.directory("models"),
                f"{step['model']}__calibration_curve__train"
            )
        else:
            filename = os.path.join(
                self.config.directory("models"),
                f"{step['model']}__calibration_curve__train__calibrated"
            )
        plt.tight_layout()
        # fig.savefig(filename + ".png", dpi=1200)
        fig.savefig(filename + ".pdf", dpi=1200)
        plt.close()

    def plot_shap_values(self, model, step):
        _best_model = model.best_estimator_.named_steps["classifier"]
        # _X_transformed = model.best_estimator_.named_steps["scaler"].transform(self.X_train)
        # print(dir(model.estimator))
        if step["model"] in ["LogisticRegression", "SVM"]:
            return # LogReg calculation not useful as we have coefficients
        else:
            X_train = model.best_estimator_.named_steps["name_tracker"].transform(self.X_train)
            X_test = model.best_estimator_.named_steps["name_tracker"].transform(self.X_test)

            # Use TreeExplainer as this uses tree structure by ignoring decision paths relying on missing features
            explainer = shap.TreeExplainer(
                model=_best_model,
                # data=pd.DataFrame(X_train, columns=model.feature_names_in_),
            )

            # explainer = shap.KernelExplainer(
            #     model=_best_model.predict_proba, data=pd.DataFrame(X_train, columns=model.feature_names_in_), link="logit",
            # )
            shap_values_train = explainer.shap_values(
                pd.DataFrame(X_train, columns=model.feature_names_in_),
            )
            shap_values_test = explainer.shap_values(
                pd.DataFrame(X_test, columns=model.feature_names_in_),
            )

        td.dump(shap_values_train, os.path.join(self.config.directory("models"), f"{step['model']}__shap_values_train.pkl"))
        td.dump(shap_values_test, os.path.join(self.config.directory("models"),
                                                f"{step['model']}__shap_values_test.pkl"))

        rn = self.config.settings.feature_selector.c.plot_names
        feature_names = [rn.get(x, x) for x in X_train.columns]
        rf_resultX_test = pd.DataFrame(shap_values_test[1], columns=feature_names)
        rf_resultX_train = pd.DataFrame(shap_values_train[1], columns=feature_names)

        vals_test = np.abs(rf_resultX_test.values).mean(0)
        vals_train = np.abs(rf_resultX_train.values).mean(0)

        shap_importance_test = pd.DataFrame(list(zip(feature_names, vals_test)),
                                       columns=['col_name', 'feature_importance_vals'])
        shap_importance_test.sort_values(by=['feature_importance_vals'],
                                    ascending=False, inplace=True)

        shap_importance_train = pd.DataFrame(list(zip(feature_names, vals_train)),
                                            columns=['col_name', 'feature_importance_vals'])
        shap_importance_train.sort_values(by=['feature_importance_vals'],
                                         ascending=False, inplace=True)

        shap_importance_test.to_csv(
            os.path.join(self.config.directory('models'), f"{step['model']}__shap_importance_test.csv"))
        shap_importance_train.to_csv(os.path.join(self.config.directory('models'), f"{step['model']}__shap_importance_train.csv"))

        td.dump(self.X_train, os.path.join(self.config.directory("models"), f"{step['model']}__X_train.pkl"))
        td.dump(self.X_test, os.path.join(self.config.directory("models"), f"{step['model']}__X_test.pkl"))
        td.dump(model.feature_names_in_, os.path.join(self.config.directory("models"), f"{step['model']}__feature_names_in.pkl"))

        feature_rename = self.config.settings.feature_selector.c.plot_names
        feature_names = [feature_rename.get(f, f) for f in model.feature_names_in_]
        self.shap_plot(
            shap_values=shap_values_train,
            features=X_train,
            feature_names=feature_names,
            filename=f"{step['model']}__shap_values__train"
        )
        self.shap_plot(
            shap_values=shap_values_test,
            features=X_test,
            feature_names=feature_names,
            filename=f"{step['model']}__shap_values__test"
        )

    def shap_plot(self, shap_values, features, feature_names, filename):

        shap.summary_plot(
            shap_values=shap_values[0],
            features=features,
            show=False,
            feature_names=feature_names,
            plot_type="dot",
        )
        plt.tight_layout()
        # plt.savefig(os.path.join(self.config.directory("models"), filename + "__dot_0.png"), dpi=1200)
        plt.savefig(
            os.path.join(self.config.directory("models"), filename + "__dot_0.pdf"),
            dpi=1200)
        plt.close()

        shap.summary_plot(
            shap_values=shap_values[1],
            features=features,
            show=False,
            feature_names=feature_names,
            plot_type="dot",
        )
        plt.tight_layout()
        # plt.savefig(
        #     os.path.join(self.config.directory("models"), filename + "__dot_1.png"), dpi=1200)
        plt.savefig(
            os.path.join(self.config.directory("models"), filename + "__dot_1.pdf"),
            dpi=1200)
        plt.close()

        shap.summary_plot(
            shap_values=shap_values,
            features=features,
            show=False,
            feature_names=feature_names,
            plot_type="bar",
        )
        plt.tight_layout()
        # plt.savefig(os.path.join(self.config.directory("models"), filename + "__bar.png"), dpi=1200)
        plt.savefig(
            os.path.join(self.config.directory("models"), filename + "__bar.pdf"),
            dpi=1200)
        plt.close()


    def _get_pipeline(self, step: dict) -> Pipeline:
        _scaler = ('scaler', StandardScaler(with_mean=True, with_std=True))
        # _imputer = ('imputer', SimpleImputer(strategy="median"))
        # _custom_imputer = ('custom_imputer', CustomImputer())
        _imputer = ('imputer', IterativeImputer(initial_strategy="median", random_state=42,))
        _name_tracker = ("name_tracker", FeatureNameTracker())
        _classifier = ("classifier", self._build_model(step["model"]))
        # _polynomial = ("polynomial", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))

        _steps = []
        if step["model"] == "LogisticRegression":
            _steps.append(_imputer)
            _steps.append(_scaler)
            # _steps.append(_polynomial)
        elif step['model'] == 'SVM':
            _steps.append(_imputer)
            _steps.append(_scaler)
        elif step["model"] == "LGBMClassifier":
            _steps.append(_name_tracker)
            pass
        _steps.append(_classifier)
        return Pipeline(steps=_steps)

    def _build_param_grid(self, step: dict) -> dict | list:
        model = self._build_model(step["model"])

        def __process(model, grid):
            param_grid = {"classifier": [model]}
            for name, values in grid.items():
                if name == "scale_pos_weight":
                    values = [(self.y_train.shape[
                                   0] - self.y_train.sum()) / self.y_train.sum()]
                param_grid[f"classifier__{name}"] = values
            return param_grid

        if isinstance(step["grid"], list):
            param_grid = []
            for grid in step["grid"]:
                param_grid.append(__process(model, grid))
        elif isinstance(step["grid"], dict):
            param_grid = __process(model, step["grid"])
        else:
            raise ValueError(f"Grid for model {step['model']} not supported")
        return param_grid

def StratifiedGroupShuffleSplit(
        data: pd.DataFrame,
        group: str,
        label: str,
        val_size: float = None,
        test_size: float = 0.2,
        random_state: int = 42,
        ) -> tuple:
    """
    Split data into chunks keeping the same group within the same set and stratifying for the outcome

    :param data:
    :param group:
    :param label:
    :param val_size:
    :param test_size:
    :param random_state:
    :return:
    """

    splitter_test = StratifiedGroupKFold(
        n_splits=int(1/test_size), # note: to accept fractions like 0.3 or 0.25 we may need to up the 1 to 100 increasing the number of splits
        random_state=random_state,
        shuffle=True
    )
    split_test = splitter_test.split(data, y=data[label], groups=data[group])
    train_ids, test_ids = next(split_test)

    train = data.iloc[train_ids]
    test = data.iloc[test_ids]

    X_test = test.drop(columns=[group, label])
    y_test = test[label]

    if val_size is not None:
        splitter_val = StratifiedGroupKFold(
            n_splits=int(1/test_size), # note: to accept fractions like 0.3 or 0.25 we may need to up the 1 to 100 increasing the number of splits
            random_state=random_state,
            shuffle=True
        )
        split_val = splitter_val.split(train, y=train[label], groups=train[group])
        train_train_ids, train_val_ids = next(split_val)

        train_train = train.iloc[train_train_ids]
        train_val = train.iloc[train_val_ids]

        X_train = train_train.drop(columns=[group, label])
        y_train = train_train[label]
        X_val = train_val.drop(columns=[group, label])
        y_val = train_val[label]
        return X_train, y_train, X_val, y_val, X_test, y_test
    else:
        X_train = train.drop(columns=[group, label])
        y_train = train[label]
        return X_train, y_train, X_test, y_test

def grouped_train_val_test_split(data: pd.DataFrame, group: str, label: str,
                                 val_size: float = None, test_size: float = 0.2,
                                 random_state: int = 42) -> tuple:
    """
    Splits data into a train, validation and test set.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """

    splitter_test = GroupShuffleSplit(
        test_size=test_size,
        n_splits=2,
        random_state=random_state
    )

    split_test = splitter_test.split(data, groups=data[group])
    train_ids, test_ids = next(split_test)

    train = data.iloc[train_ids]
    test = data.iloc[test_ids]

    X_test = test.drop(columns=[group, label])
    y_test = test[label]

    if val_size is not None:
        splitter_val = GroupShuffleSplit(
            test_size=val_size,
            n_splits=2,
            random_state=random_state
        )
        split_val = splitter_val.split(train, groups=train[group])
        train_train_ids, train_val_ids = next(split_val)

        train_train = train.iloc[train_train_ids]
        train_val = train.iloc[train_val_ids]

        X_train = train_train.drop(columns=[group, label])
        y_train = train_train[label]
        X_val = train_val.drop(columns=[group, label])
        y_val = train_val[label]
        return X_train, y_train, X_val, y_val, X_test, y_test
    else:
        X_train = train.drop(columns=[group, label])
        y_train = train[label]
        return X_train, y_train, X_test, y_test



class CustomImputer(TransformerMixin, BaseEstimator):
    def __init__(self, strategy: str, random_state: int = 42):
        self.feature_names_in_ = None

        self.constant_imputers = {}

        self.columns_to_impute_as_false = [
            "mic__is_positive__tip__event_m3d__any",
            "mic__is_positive__sputum__event_m3d__any",
            "mic__is_positive__blood__event_m3d__any",
            "mic__is_positive__urine__event_m3d__any",
            "mic__group_cns__event_m3d__ohe",
            "mic__group_hospital_pathogens__event_m3d__ohe",
            "mic__group_negative__event_m3d__ohe",
        ]

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = X.columns

        columns_to_fit = [x for x in self.columns_to_impute_as_false if x in X.columns]
        if columns_to_fit:
            self.constant_imputers[False] = SimpleImputer(
                strategy="constant", fill_value=False)
            self.constant_imputers[False].fit(X[columns_to_fit])
        return self

    def transform(self, X):
        X = X.copy()
        columns_to_transform = [x for x in self.columns_to_impute_as_false if x in X.columns]
        if columns_to_transform:
            X[columns_to_transform] = self.constant_imputers[
                False].transform(X[columns_to_transform])
        return X




class FeatureNameTracker(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Store the feature names during fit
        self.feature_names_ = list(X.columns)
        self.feature_names_in_ = self.feature_names_
        return self

    def transform(self, X):
        # Return the original data without any modification
        return X

    def get_feature_names_out(self):
        return self.feature_names_

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.figure import Figure
from sklearn.calibration import calibration_curve, CalibratedClassifierCV


def plot_calibration_curve(
    y_true,
    y_prob,
    x_min: float | None = None,
    x_max: float | None = None,
    n_bins: int = 10,
    strategy: str | None = "quantile",
    fig_size: tuple = (15, 10),
    show: bool = False,
):
    """Plot a calibration curve.

    This curve shows if predicted probabilities and observed frequencies are inline.
    For example, if well calibrated 100 observations with y_pred = 0.1 should contain 10
    observation with y_true = 1.

    "Specifically, the predicted probabilities are divided up into a fixed number of buckets along
    the x-axis. The number of events (class=1) are then counted for each bin
    (e.g. the relative observed frequency). Finally, the counts are normalized.
    The results are then plotted as a line plot."
    https://machinelearningmastery.com/calibrated-classification-model-in-scikit-learn/

    Parameters
    ----------
    y_true
        Target value of y.
    y_prob
        Predicted probability values of y.
    x_min
        Minimum value of the x axis to plot.
    x_max
        Maximum value of the x axis to plot.
    n_bins
        Number of bins of the histograms.
    show
        Whether to show the plot.
    fig_size
        Size of the figure to plot learning curve on.
    strategy
        Method to bin predicted probabilities for calibration calculation.
        Options are "quantile" (each group has an equal number of patients)
        or "uniform" (the bins are equally spaced)

    Returns
    -------
    A plot
    """
    if strategy not in ["quantile", "uniform"]:
        raise ValueError(
            f"`strategy` should be either `quantile` or `uniform`, not {strategy}"
        )

    fig = plt.figure(figsize=fig_size)
    ax = plt.gca()

    y_pred_to_plot = [x for x, y in zip(y_prob, y_true) if pd.notnull(x)]
    y_true_to_plot = [y for x, y in zip(y_prob, y_true) if pd.notnull(x)]

    # getting calibration points usings sklearns calibration curve function
    prob_true, prob_pred = calibration_curve(
        y_true_to_plot, y_pred_to_plot, strategy=strategy, n_bins=n_bins
    )

    # filtering on x-range to plot
    if x_max is None:
        x_max = np.max(y_prob)
    if x_min is None:
        x_min = np.min(y_prob)

    prob_pred_to_plot = [
        x for x, y in zip(prob_pred, prob_true) if (x_min < x) and (x < x_max)
    ]
    prob_true_to_plot = [
        y for x, y in zip(prob_pred, prob_true) if (x_min < x) and (x < x_max)
    ]

    # plotting the calibration line
    ax.plot(
        prob_pred_to_plot,
        prob_true_to_plot,
        marker="*",
        color="darkblue",
        label="calibration of predictions",
    )

    # plotting line of perfect calibration
    ax.plot(
        [x_min, x_max],
        [x_min, x_max],
        marker=None,
        color="grey",
        label="perfect calibration",
    )

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("% restarted <72h on ICU")
    ax.legend(loc="center right")

    # plotting histogram of predictions distribution
    ax1 = ax.twinx()
    ax1.grid(None)
    ax1.hist(
        y_prob, color="lightgrey", alpha=0.5, label="prediction distribution", bins=20
    )
    ax1.legend(loc="upper right")
    ax1.set_ylabel("number of predictions")
    ax1.set_xlim(x_min, x_max)

    if show:
        plt.show()

    plt.close(fig)
    return fig


CV_SCORERS_CALIBRATED = {'accuracy': accuracy_score,
 'balanced_accuracy_score': balanced_accuracy_score,
 'rocauc': roc_auc_score,
 'average_precision': average_precision_score,
 'precision': precision_score,
 'recall': recall_score,
 'specificity': recall_score,
 'f1': f1_score}



class KFoldEvaluator:

    def __init__(self, X, y, threshold: float,  k_folds: int = 10, random_state: int = 42):
        self.k_folds = k_folds
        self.random_state = random_state
        self.threshold = threshold
        self.skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

        self.X = X
        self.y = y


    def evaluate(self, model, calibrated = None) -> dict:
        result = {}

        # the regular evaluation takes the model and X
        # but the calibrated model needs to make a prediction based on
        # the predictions by the original model
        # therefore, to evaluate the calibrated model, the original model needs
        # to be passed as well

        for i, (idx_train, idx_test) in enumerate(self.skf.split(self.X, self.y)):
            if calibrated:
                X_proba = model.predict_proba(self.X.iloc[idx_train])[:,1].reshape(-1,1)
                result[i] = self.score(calibrated, X_proba, self.y.iloc[idx_train], calibrated=True)
            else:
                result[i] = self.score(model, self.X.iloc[idx_train], self.y.iloc[idx_train], calibrated=False)
            result[i]['size'] = idx_train.shape[0]
            result[i]['pos'] = self.y.iloc[idx_train].sum()
        return result

    @staticmethod
    def transform_results(result: dict):
        dfs = list()
        for name, _result in result.items():
            _df = pd.DataFrame(_result).T
            _df.columns = pd.MultiIndex.from_tuples([(name, k) for k in _df.columns])
            dfs.append(_df)

        df_result = pd.concat(dfs, axis=1).agg(["mean", "std"]).round(3).T
        return df_result

    def evaluate_models(self, models: dict[str, any], calibrated: dict[str, any] | None = None):
        result = {}
        for model_name, model in models.items():
            if calibrated is not None:
                calibrated_model = calibrated[model_name]
            else:
                calibrated_model = None
            result[model_name] = self.evaluate(model=model, calibrated=calibrated_model)

        return self.transform_results(result)

    def score(self, model, X, y, calibrated: bool = False):
        _probas = ["rocauc", "average_precision"]
        _zero_division = ["precision", "recall", "specificity"]
        _pos_label_0 = ["specificity"]

        _results = dict()
        if not calibrated:
            for metric, fun in CV_SCORERS.items():
                _results[metric] = fun(model, X, y)
        else:
            y_pred_proba = model.predict_proba(X)[:, 1]
            y_pred = y_pred_proba > self.threshold

            for metric, fun in CV_SCORERS_CALIBRATED.items():
                __kwargs = {}
                if metric in _zero_division:
                    __kwargs["zero_division"] = 0
                if metric in _pos_label_0:
                    __kwargs["pos_label"] = 0
                if metric in _probas:
                    y_passed = y_pred_proba
                else:
                    y_passed = y_pred

                _results[metric] = fun(y, y_passed, **__kwargs)

        return _results


if __name__ == "__main__":

    _config = Config(root="C:\\TADAM\\projects\\abstop")

    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(
        os.path.join(_config.directory("logs"), "experiment_runner.log")
    )
    fh.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)

    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - [%(filename)s:%(lineno)s - %(funcName)20s() ] "
        "[%(levelname)s] - %(message)s"
    )

    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    mt = ModelTrainer(config=_config)
    mt.run()
    print("Done")

