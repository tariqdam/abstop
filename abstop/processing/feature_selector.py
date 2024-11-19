import logging
import os
import warnings
from logging import Logger
from typing import Any

import duckdb as duckdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

import tadam as td
import tadam.dataprocessing as dp
import seaborn as sns
import textwrap

from scipy import stats
import math


from abstop.config import Config

logger: Logger = logging.getLogger(__name__)


class FeatureSelector:
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
        self._cfg = self.config.settings.feature_selector

        self._boolean_columns_for_imputation = []

    def run(self) -> None:
        """
        Run the Feature Selector with the settings defined in the config file.
        :return: pd.DataFrame
        """

        self.logger.info("Running feature selector")
        # Load the data
        df = self._load_data("features")[["pid"] + self._cfg.c.full].copy()
        self.logger.critical(f"Loaded data with shape {df.shape}")
        self.logger.critical(["pid"] + self._cfg.c.full)

        # 0. Process data
        df = self._process(df)

        # 1. Describe the data
        self._describe(df, suffix="before_boolean_imputation")

        # 2. Reprocess data: impute missing values for booleans
        __mic_group = [x for x in df if "mic__group_" in x]
        __mic_pos = [x for x in df if "mic__is_positive" in x]
        __impute_columns = __mic_group + __mic_pos
        df = self._impute_booleans_as_false(df, __impute_columns)
        df = self._impute_set_features(df)

        # 3. Describe the data again: check if imputation worked
        # and check if any patients have too many missing features
        # followed by check if any features are too high in missing features
        self._describe(df, suffix="after_boolean_imputation")

        # Save dataframe for training pipeline
        df.to_csv(
            os.path.join(self.config.directory("processed"), "model_data.csv"),
            index=False,
        )

        # 2. Select features
        # export feature x feature correlation
        self._feature_x_feature_correlation(df)

        # export feature x outcome correlation
        self._feature_x_outcome_correlation(df)

        # self._plot_feature_distribution(df)

    def _feature_x_feature_correlation(self, df: pd.DataFrame) -> None:
        """
        Export feature x feature correlation.
        :param df: pd.DataFrame
        :return: None
        """
        _features = df[self._cfg.c.features].copy()

        # Calculate correlation matrix
        _corr = _features.corr()
        _cols = list(sorted(list(_corr.index)))
        _corr = _corr[_cols]
        _corr = _corr.reindex(_cols, axis="index")
        _path, _name = self._cfg.files.get("corr_features", ("results", None))
        _corr.to_csv(os.path.join(self.config.directory(_path), _name))

    def _plot_feature_distribution(self, df: pd.DataFrame) -> None:
        """
        Plot feature distribution.
        :param df: pd.DataFrame
        :return: None
        """

        _features = self._cfg.c.features
        _outcome = self._cfg.c.outcomes[0]

        self.logger.debug(f"Plotting feature distribution for {_features} and {_outcome}")

        self.logger.debug("Setting seaborn context to paper")
        sns.set_context("paper", rc={"axes.labelsize": 14})

        self.logger.debug("Renaming columns")
        _feature_rename = {x: " ".join(" ".join(x.split("__")).split("_")) for x in
                           _features}
        _outcome_rename = {_outcome: " ".join(" ".join(_outcome.split("__")).split("_"))}

        _data = df[_features + [_outcome]].rename(columns={**_feature_rename, **_outcome_rename})

        self.logger.debug("Converting objects to float")
        _objects_bool_to_float = _data.select_dtypes(include=["object"]).columns
        _data[_objects_bool_to_float] = _data[_objects_bool_to_float].astype(float)

        self.logger.debug("Retrieving path and name from config")
        _path, _name = self._cfg.files.get("pairplot", ("figures", None))

        if _name is None:
            self.logger.critical("No name for pairplot found, using default name")
            _name = "pairplot.png"

        self.logger.debug("Plotting pairplot")
        _plot = sns.pairplot(
                    _data,
                    kind="reg",
                    hue=_outcome_rename.get(_outcome),
                    height=3,
                    corner=True,
                    plot_kws={
                        "scatter_kws": {
                            "alpha": 0.05
                        }
                    },
                )
        self.logger.debug("Plotted pairplot")

        self.logger.debug("Wrapping labels")
        _plot = wrap_labels(pairgrid=_plot, width=20)
        self.logger.debug("Wrapped labels")

        save_path = os.path.join(
            self.config.directory(_path),
            _name
        )

        self.logger.debug(f"Saving pairplot to {save_path}")
        _plot.savefig(save_path)
        self.logger.debug(f"Saved pairplot to {save_path}")

    def _feature_x_outcome_correlation(self, df: pd.DataFrame) -> None:
        """
        Export feature x outcome correlation.
        :param df: pd.DataFrame
        :return: None
        """

        _columns = self._cfg.c.features + self._cfg.c.outcomes

        # Calculate correlation matrix
        _corr = df[_columns].corr().filter(self._cfg.c.outcomes).drop(self._cfg.c.outcomes)

        _cols = list(sorted(list(_corr.index)))
        _corr = _corr.sort_values(by=self._cfg.c.outcomes[0], key=abs, ascending=False)
        _path, _name = self._cfg.files.get("corr_outcomes", ("results", None))
        _corr.to_csv(os.path.join(self.config.directory(_path), _name))

    def _load_data(self, name: str) -> pd.DataFrame:
        """
        Load data from file.
        :param name: string name of dictionary key in config file
        :return: pd.DataFrame
        """
        _path, _file = self._cfg.files.get(name, ("processed", None))
        if _file:
            file_path = os.path.join(self.config.directory(_path), _file)
            return td.load(file_path)
        else:
            raise ValueError(f"no file found for {name}")

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the data.
        :param df:
        :return:
        """

        # Transform all timedelta into days
        df = self._timedelta_to_days(df)
        df = self._object_to_bool_without_imputation(df)
        df = self._calculate_missing_from_absent_booleans(df)

        df = self._process_limits(df)
        return df

    def _calculate_missing_from_absent_booleans(self, df: pd.DataFrame) -> pd.DataFrame:
        __mic_m3d = [x for x in df.columns if "mic__group" in x if "event_m3d__ohe" in x]
        df["mic__group__missing__event_m3d__ohe"] = (df[__mic_m3d].astype(float).sum(axis=1) ==
                                       0).astype(int)

        __mic_m1d_p1d = [x for x in df.columns if "mic__group" in x if "start_m1d_p1d__ohe" in x]
        df["mic__group__missing__start_m1d_p1d__ohe"] = (
                    df[__mic_m1d_p1d].astype(float).sum(axis=1) ==
                    0).astype(int)

        __adm_hosp = [x for x in df.columns if "adm_hosp_specialism" in x]
        df["adm_hosp_specialism__missing"] = (df[__adm_hosp].astype(float).sum(axis=1) == 0).astype(int)
        return df

    def _process_limits(self, df: pd.DataFrame) -> pd.DataFrame:
        limit_instructions = self._cfg.limits
        for method, limits in limit_instructions.items():
            for column, limit in limits.items():
                if column in df.columns:
                    lower_limit = limit.get("min", None)
                    upper_limit = limit.get("max", None)
                    if method == "clip":
                        df[column] = df[column].clip(lower=lower_limit, upper=upper_limit)
                    elif method == "drop":
                        df[column] = df[column].where((df[column] >= lower_limit) & (df[column] <= upper_limit))
        return df

    def _timedelta_to_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform all timedelta columns into days.
        :param df: pd.DataFrame
        :return: pd.DataFrame
        """

        timedelta_columns = df[self._cfg.c.full].select_dtypes(
            include=["timedelta64[ns]"]).columns
        df[timedelta_columns] = df[timedelta_columns] / pd.Timedelta(1, "d")
        return df

    def _object_to_bool_without_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform all object columns into boolean without imputation as these may still
        need to be described with their missing propotions.
        :param df: pd.DataFrame
        :return: pd.DataFrame
        """
        object_columns = df[self._cfg.c.full].select_dtypes(
            include=["object"]).columns
        # set to float, as bool will inpute missing as true
        df[object_columns] = df[object_columns].astype(float)
        return df

    def _impute_booleans_as_false(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Impute all boolean columns with missing values as False.
        :param df: pd.DataFrame
        :return: pd.DataFrame
        """
        df[columns] = df[columns].fillna(0).astype(int)
        return df

    def _describe(self, df: pd.DataFrame, suffix=None) -> None:
        """
        Describe the data.
        :param df: pd.DataFrame
        :return: pd.DataFrame
        """

        desc_all = list(sorted(self._cfg.c.full))
        features = list(sorted(self._cfg.c.features))

        # TableOne overview with outcomes etc.

        # Missing per feature
        df_missing_per_feature = df[desc_all].isna().mean()
        _path, _name = self._cfg.files.get("missing_per_feature", ("results", "missing_per_feature.csv"))
        _name = add_suffix_to_filename(_name, suffix=suffix)
        df_missing_per_feature.to_csv(os.path.join(self.config.directory(_path), _name))
        self.logger.info(f"Missing per feature: {df_missing_per_feature.describe()}")
        self.logger.info(f"Missing >0.4 per feature: {df_missing_per_feature[df_missing_per_feature > 0.4]}")

        # Missing per patient
        # features only!
        df_missing_per_patient = df[features].isna().mean(axis=1)
        _path, _name = self._cfg.files.get("missing_per_patient", ("results", "missing_per_patient.csv"))
        _name = add_suffix_to_filename(_name, suffix=suffix)
        df_missing_per_patient.to_csv(os.path.join(self.config.directory(_path), _name))
        self.logger.info(f"Missing per patient: {df_missing_per_patient.describe()}")
        self.logger.info(f"Missing >0.4 per patient: {df_missing_per_patient[df_missing_per_patient > 0.4]}")

        self.logger.debug("Describing data per feature")
        # Mean, std, median, iqr, min, max, missing etc. per feature
        df_describe_per_feature = df[desc_all].describe(include="all").T
        df_describe_per_feature["missing"] = df_missing_per_feature
        _path, _name = self._cfg.files.get("describe_per_feature", ("results", "describe_per_feature.csv"))
        _name = add_suffix_to_filename(_name, suffix=suffix)
        df_describe_per_feature.to_csv(os.path.join(self.config.directory(_path), _name))

        self.logger.debug("Describing data per feature per outcome")
        # Mean, std, median, iqr, min, max, missing etc. per feature, per outcome
        df_per_feature_per_outcome = describe_feature_per_group(
            df=df,
            features=desc_all,
            groups=[self._cfg.c.outcomes[0]]
        )
        _path, _name = self._cfg.files.get("describe_per_feature_per_outcome", ("results", "describe_per_feature_per_outcome.csv"))
        _name = add_suffix_to_filename(_name, suffix=suffix)
        df_per_feature_per_outcome.to_csv(os.path.join(self.config.directory(_path), _name))

        self.logger.debug("Describing data per feature per hospital")
        df_per_feature_per_hospital = describe_feature_per_group(
            df=df,
            features=desc_all,
            groups=["adm_hosp_loc__vumc__ohe"]
        )
        _path, _name = self._cfg.files.get("describe_per_feature_per_hospital", (
        "results", "describe_per_feature_per_hospital.csv"))
        _name = add_suffix_to_filename(_name, suffix=suffix)
        df_per_feature_per_hospital.to_csv(
            os.path.join(self.config.directory(_path), _name))

        self.logger.debug("Describing data per feature per outcome per hospital")
        df_per_feature_per_outcome_per_hospital = describe_feature_per_group(
            df=df,
            features=desc_all,
            groups=["adm_hosp_loc__vumc__ohe", self._cfg.c.outcomes[0]]
        )
        _path, _name = self._cfg.files.get("describe_per_feature_per_outcome_per_hospital", (
        "results", "describe_per_feature_per_outcome_per_hospital.csv"))
        _name = add_suffix_to_filename(_name, suffix=suffix)
        df_per_feature_per_outcome_per_hospital.to_csv(
            os.path.join(self.config.directory(_path), _name))

        self.__describe_new_method(df, suffix)

    def __describe_new_method(self, df:pd.DataFrame, suffix:str) -> None:
        """
        Describe the data.
        :param df: pd.DataFrame
        :return: pd.DataFrame
        """

        __num = list(sorted(set(self._cfg.c.features_numeric) | set(self._cfg.c.descriptives_numeric)))
        __bool = list(sorted(set(self._cfg.c.features_bool) | set(self._cfg.c.descriptives_bool)))

        _df = df.copy()
        _df[__bool] = _df[__bool].astype(float).copy()

        __bool += ["mic__group__missing__event_m3d__ohe", "mic__group__missing__start_m1d_p1d__ohe", "adm_hosp_specialism__missing", "records"]
        __num += ["events_per_patient"]

        # set groups to false if outcome is false
        _outcome = self._cfg.c.outcomes[0]
        _mask = _df[_outcome] == 0
        __next_atc_columns = [x for x in _df.columns if "outcome_next_atc__groups__j" in x]
        _df.loc[_mask, __next_atc_columns] = 0
        _df.loc[_mask, "outcome_next_atc__groups__none__ohe"] = 1

        # TableOne overview with outcomes etc.
        _agg = get_aggregate(data=_df.copy(), groupby=None)
        _agg_proc = get_table(data=_agg, normal=[], nonnormal=__num, booleans=__bool).reset_index()
        _total = get_total_patients(data=_agg_proc, search_keyword="adm_hosp_loc")
        __agg_proc = process_df(data=_agg_proc, rename=self._cfg.c.table1, total=_total)
        path = self.config.directory("results")
        file = f"table1.csv"
        if suffix:
            file = f"{file[:-4]}_{suffix}.csv"
        __agg_proc.to_csv(os.path.join(path, file), index=False)

        # TableOne per outcome:
        df_grouped = get_aggregate(data=_df.copy(), groupby=[_outcome])
        df_grouped.index.name = "index"
        _dfs = []
        for col in df_grouped.columns._get_level_values(0).unique():
            __TOTAL = df_grouped[col].loc["records", "sum"]

            _dfg = get_table(df_grouped[col].copy(), normal=[], nonnormal=__num, booleans=__bool).reset_index()
            __df = process_df(_dfg, rename=self._cfg.c.table1,
                              total=__TOTAL).set_index(["Feature", "Aggregate"])
            __df.columns = pd.MultiIndex.from_tuples([(col, i) for i in __df.columns])
            _dfs.append(__df)
        dfs = pd.concat(_dfs, axis=1)

        file = f"table1_per_outcome.csv"
        if suffix:
            file = f"{file[:-4]}_{suffix}.csv"
        dfs.to_csv(os.path.join(path, file))


        # TableOne per hospital location
        print("TableOne!")
        df_grouped = get_aggregate(data=_df.copy(), groupby=["adm_hosp_loc__vumc__ohe"])
        df_grouped.index.name = "index"
        _dfs = []
        for col in df_grouped.columns._get_level_values(0).unique():
            __TOTAL = df_grouped[col].loc["records", "sum"]

            _dfg = get_table(df_grouped[col].copy(), normal=[], nonnormal=__num, booleans=__bool).reset_index()
            __df = process_df(_dfg, rename=self._cfg.c.table1,
                              total=__TOTAL).set_index(["Feature", "Aggregate"])
            __df.columns = pd.MultiIndex.from_tuples([(col, i) for i in __df.columns])
            _dfs.append(__df)
        dfs = pd.concat(_dfs, axis=1)

        file = f"table1_per_hospital__is_vumc.csv"
        if suffix:
            file = f"{file[:-4]}_{suffix}.csv"
        dfs.to_csv(os.path.join(path, file))


    def _impute_set_features(self, df):
        __impute_columns = [x for x in df if "gcs_total" in x]
        for col in __impute_columns:
            df[col] = df[col].fillna(15)
        return df


def add_suffix_to_filename(filename: str, suffix: str) -> str:
    """
    Add suffix to filename.
    :param filename: str
    :param suffix: str
    :return: str
    """
    _name, _ext = filename.split(".")
    return f"{_name}__{suffix}.{_ext}"

def describe_feature_per_group(df: pd.DataFrame, features: list, groups: list[str]) -> pd.DataFrame:
    """
    Describe features per group.
    :param df: pd.DataFrame
    :param features: list of features to describe
    :param groups: list of groups to describe
    :return: pd.DataFrame
    """

    group_by = [df[group] for group in groups]
    grouped_describe = df[features].groupby(by=group_by).describe(include="all").T.reset_index().rename(
        columns={
            "level_1": "agg",
            "level_0": "feature",
        },
    )

    grouped_missing = df[features].isna().groupby(by=group_by).mean().T
    grouped_missing = pd.concat({"missing": grouped_missing}).reset_index().rename(
        columns={"level_0": "agg", "level_1": "feature"})

    grouped_combined = pd.concat(objs=[grouped_describe, grouped_missing], axis=0,
                                 ignore_index=False)
    return grouped_combined.pivot(columns=["agg"], index=["feature"])

def wrap_labels(pairgrid, width, break_long_words=False):
    figure = pairgrid._figure
    for ax in figure._localaxes:
        ylabel = ax.get_ylabel()
        new_ylabel = "\n".join(textwrap.wrap(ylabel, width))
        ax.set_ylabel(new_ylabel)

        xlabel = ax.get_xlabel()
        new_xlabel = "\n".join(textwrap.wrap(xlabel, width))
        ax.set_xlabel(new_xlabel)
    return pairgrid

# Table One functions
def get_p_from_p(p1, n1, p2, n2):
    N = n1 + n2
    if N == 0:
        return np.nan

    p_pooled = (p1 + p2) / N
    upper = p1 - p2

    if n1 == 0:
        lower_1 = 0
    else:
        lower_1 = (p1*(1-p1))/n1
    if n2 == 0:
        lower_2 = 0
    else:
        lower_2 = (p2*(1-p2))/n2

    lower = math.sqrt(
        lower_1 + lower_2
    )

    if (lower == 0) and (upper == 0):
        p_value = np.nan
    elif (lower == 0) or (upper == 0):
        p_value = 0
    else:
        p_value = stats.norm.sf(abs(upper/lower))*2
    return p_value

def freq(x):
    _unique_counts = np.unique(x, return_counts=True)
    item = {k: v for k,v in list(zip(*_unique_counts))}
    n_pos = item.get(1, None)
    n_neg = item.get(0, None)

    if n_pos is not None:
        if n_neg is not None:
            freq = n_pos / (n_pos + n_neg)
        else:
            freq = 1
    else:
        if n_neg is not None:
            freq = 0
        else:
            freq = np.nan
    return freq

def missing(x):
    n_missing = x.isna().sum()
    n_total = x.shape[0]
    return n_missing / n_total

def missing_n(x):
    return x.isna().sum()

def missing_p(x):
    return x.isna().sum() / x.shape[0]

def p(n):
    def percentile_(x):
        return x.quantile(n)
    percentile_.__name__ = 'p{:02.0f}'.format(n*100)
    return percentile_


def derive_extra_features(data):
    if "records" not in data.columns:
        data["records"] = 1
    return data

def get_aggregate(data, groupby = None):
    agg = ["mean", "std", "count", "sum", "min", "max", missing_n, missing_p, p(0.5), p(0.25), p(0.75)]
    has_numeric = False
    has_categorical = False

    full = list()
    grouped = list()
    data = derive_extra_features(data)

    numeric_data = data.select_dtypes(include='number')
    categorical_data = data.select_dtypes(exclude="number")

    if numeric_data.shape[1] > 0:
        has_numeric = True
    if categorical_data.shape[1] > 0:
        has_categorical = True

    if groupby is None:
        if has_numeric:
            numerics = numeric_data.agg(agg).T
            full.append(numerics)
        if has_categorical:
            ohe = OneHotEncoder()
            categorical_data = pd.DataFrame(
                ohe.fit_transform(
                    X=data.select_dtypes(exclude="number")).toarray(),
                    columns=ohe.get_feature_names_out(),
            )
            categoricals = categorical_data.agg(agg).T
            full.append(categoricals)

        ___df = data[["pid"]].value_counts().reset_index()
        ___df = ___df["count"].agg(
            agg).to_frame("events_per_patient").T
        full.append(___df)

        if len(full) > 0:
            return_df = pd.concat(full, axis=0)
    elif isinstance(groupby, (str, list, tuple)):
        group_by = None
        if isinstance(groupby, str):
            group_by = data[group_by]
        else:
            group_by = [data[col] for col in groupby]
        __dfs = list()
        if has_numeric:
            __dfs.append(
                numeric_data.groupby(group_by).agg(agg).T.reset_index().rename(columns={"level_1": "agg", "level_0": "feature"})
            )

        if has_categorical:
            ohe = OneHotEncoder()
            categorical_data = pd.DataFrame(
                ohe.fit_transform(
                    X=data.select_dtypes(exclude="number")).toarray(),
                    columns=ohe.get_feature_names_out(),
            )
            __dfs.append(
                categorical_data.groupby(group_by).agg(agg).T.reset_index().rename(columns={"level_1": "agg", "level_0": "feature"})
            )

        # ___df = data.groupby(group_by)[["pid"]].value_counts().reset_index().groupby(group_by)["count"].agg(agg).T.reset_index().rename(columns={"index": "agg"})
        ___df = data.groupby(group_by)[["pid"]].value_counts().reset_index()
        ___df = ___df.groupby(list(___df.columns)[:len(group_by)])["count"].agg(agg).T.reset_index().rename(columns={"index": "agg"})
        ___df["feature"] = "events_per_patient"
        __dfs.append(___df)

        return pd.concat(__dfs, axis=0).pivot(columns=["agg"], index=["feature"])

    else:
        if has_numeric:
            grouped_numeric = numeric_data.groupby(groupby).agg(agg).T
            grouped_numeric.columns.name = "ood"
            grouped_numeric = grouped_numeric.rename(columns={True: "out-domain", False: "in-domain"})
            g_num_piv = grouped_numeric.reset_index().rename(columns={"level_0": "item", "level_1": "agg"}).pivot_table(index=["item"], columns=["agg"])
            g_num_piv[("", "p-value")] = g_num_piv.apply(lambda x: stats.ttest_ind_from_stats(
                    mean1 = x[("in-domain", "mean")],
                    std1 = x[("in-domain", "std")],
                    nobs1 = x[("in-domain", "count")],
                    mean2 = x[("out-domain", "mean")],
                    std2 = x[("out-domain", "std")],
                    nobs2 = x[("out-domain", "count")],
                    equal_var=False,
                    alternative="two-sided",
                )[-1], axis=1)
            g_num_piv[("", "p-value-missing")] = g_num_piv.apply(lambda x: get_p_from_p(
                p1=x[("in-domain", "missing")],
                n1=x[("in-domain", "count")],
                p2=x[("out-domain", "missing")],
                n2=x[("out-domain", "count")],
            ), axis=1)
            grouped.append(g_num_piv)
            full_numeric = numeric_data.agg(agg).T
            full.append(full_numeric)

        if has_categorical:
            ohe = OneHotEncoder()
            categorical_data = pd.DataFrame(
                ohe.fit_transform(
                    X=data.select_dtypes(exclude="number")).toarray(),
                    columns=ohe.get_feature_names_out(),
            )
            grouped_categorical = categorical_data.groupby(groupby).agg(agg).T
            grouped_categorical.columns.name = "ood"
            grouped_categorical = grouped_categorical.rename(columns={True: "out-domain", False: "in-domain"})
            g_cat_piv = grouped_categorical.reset_index().rename(columns={"level_0": "item", "level_1": "agg"}).pivot_table(index=["item"], columns=["agg"])
            g_cat_piv[("", "p-value")] = g_cat_piv.apply(lambda x: get_p_from_p(
                p1=x[("in-domain", "mean")],
                n1=x[("in-domain", "count")],
                p2=x[("out-domain", "mean")],
                n2=x[("out-domain", "count")],
            ), axis=1)
            g_cat_piv[("", "p-value-missing")] = g_cat_piv.apply(lambda x: get_p_from_p(
                p1=x[("in-domain", "missing")],
                n1=x[("in-domain", "count")],
                p2=x[("out-domain", "missing")],
                n2=x[("out-domain", "count")],
            ), axis=1)
            grouped.append(g_cat_piv)

            full_categorical = categorical_data.agg(agg).T
            full.append(full_categorical)

        final = list()
        if len(full) > 0:
            full_df = pd.concat(full, axis=0)
            full_df.columns.name = "agg"
            full_df.columns = pd.MultiIndex.from_product([pd.Index(["full"], name="ood"), full_df.columns])
            final.append(full_df)
        if len(grouped) > 0:
            grouped_df_pivoted = pd.concat(grouped, axis=0)
            final.append(grouped_df_pivoted)

        return_df = pd.concat(final, axis=1)
    return return_df


def format_t1(df: pd.DataFrame) -> pd.DataFrame:
    # fix mean + std
    df["mean_std"] = df["mean"].round(2).astype(str) + " (" + df["std"].round(2).astype(
        str) + ")"

    # set median [IQR]
    df["median"] = df["p50"].round(2).astype(str) + " [" + df["p25"].round(2).astype(
        str) + ", " + df["p75"].round(2).astype(str) + "]"

    # set booleans to N + percentage
    df["N (%)"] = df["sum"].round(0).astype(str).apply(lambda x: x[:-2]) + " (" + (
                100 * df["mean"]).round(2).astype(str) + "%)"
    _booleans = (
            (df["max"] == 1) & (df["min"] == 0) |
            (df["max"] == 0) & (df["min"] == 0) |
            (df["max"] == 1) & (df["min"] == 1)
    )
    df.loc[~_booleans, "N (%)"] = np.nan

    # convert missing fracctions to percentages
    df["Missing, N (%)"] = df["missing_n"].round(0).astype(str).apply(
        lambda x: x) + " (" + (100 * df["missing_p"]).round(2).astype(str) + "%)"

    return df[["mean_std", "median", "N (%)", "Missing, N (%)"]]


def get_table(data, normal: list[str], nonnormal: list[str], booleans: list[str], order: list[str] = None) -> pd.DataFrame:
    d = format_t1(data)
    _d = d[["Missing, N (%)"]].copy()
    _aggs = list()
    if normal:
        _normal = d.loc[normal, "mean_std"].to_frame("value")
        _normal["description"] = "Mean (std)"
        _aggs.append(_normal)
    if nonnormal:
        _nonnormal = d.loc[nonnormal, "median"].to_frame("value")
        _nonnormal["description"] = "Median [IQR]"
        _aggs.append(_nonnormal)
    if booleans:
        _bool = d.loc[booleans, "N (%)"].to_frame("value")
        _bool["description"] = "N (%)"
        _aggs.append(_bool)
    _df = pd.concat(_aggs, axis=0, ignore_index=False)

    df = pd.concat([_d, _df], axis=1).reset_index()

    if order:
        # set order according to order, filtering only on mentioned values
        pass

    return df.set_index(["index", "description"])


def combine_boolean_categories(data: pd.DataFrame, rename: dict, total: int,
                               search_keyword="ADD_TO_OTHER", target_keyword="other"):
    _others = {k: i for k, i in rename.items() if search_keyword in i}

    if len(_others) == 0:
        return data, rename

    _split = data.loc[data["index"].isin(_others.keys())]["value"].str.replace("(",
                                                                               "").str.replace(
        ")", "").str.replace("%", "").str.split(" ", expand=True).astype(float)
    _sum = _split.sum().values
    _str = f"{int(_sum[0])} ({round(_sum[0] / total * 100, 2)}%)"
    _others_to_remove = {
        k: i.replace(search_keyword, "REMOVE") if target_keyword not in k else "- Other"
        for k, i in _others.items()}

    data.loc[data["index"] == target_keyword, "value"] = _str

    rename.update(_others_to_remove)
    return data, rename


def __to_string(primary, secondary, method="bool"):
    if method == "bool":
        _str = f"{primary} ({round(primary / secondary * 100, 2)}%)"
    elif method == "median":
        _str = f"{primary} [{secondary}]"
    elif method == "mean":
        _str = f"{primary} ({secondary})"
    return _str


def adjust_missing_counts(data: pd.DataFrame, records_to_group: list, total: int):
    _known_count = data.loc[data["index"].isin(records_to_group)]["value"].str.split(" ", expand=True)[
        0].astype(int).sum()
    _missing_count = total - _known_count
    _str = __to_string(_missing_count, total)

    data.loc[data["index"].isin(records_to_group), "Missing, N (%)"] = _str
    return data


def adjust_missing_counts_all_zeroes(data: pd.DataFrame, records_to_group: list, total: int):
    _sums = data[records_to_group].astype(float).sum(axis=1)
    _missing = _sums == 0
    _missing_count = _missing.sum()
    print("MIC MISSING = {}".format(_missing_count))
    _str = __to_string(_missing_count, total)
    data.loc[data["index"].isin(records_to_group), "Missing, N (%)"] = _str
    return data


def add_headers(data: pd.DataFrame, rename: dict):
    names = list(rename.keys())
    known_names = list(data.loc[data["index"].isin(names)]["index"].unique())
    unknown_names = [x for x in names if x not in known_names]
    return pd.concat(
        [
            data,
            pd.DataFrame(unknown_names, columns=["index"]),
        ],
        axis=0).fillna("")


def sort_by_rename(data: pd.DataFrame, rename: dict):
    order = {k: i for i, k in enumerate(rename.keys())}
    return data.sort_values("index", key=lambda x: x.map(order))


def rename_features(data: pd.DataFrame, rename: dict):
    data["index"] = data["index"].replace(rename)
    return data


def remove_unused_records(data: pd.DataFrame, rename: dict):
    return data.loc[~data["index"].str.contains("__REMOVE")].copy()


def fix_headers(data: pd.DataFrame):
    __rename = {"index": "Feature", "description": "Aggregate", "value": "Value",
                "Missing, N (%)": "Missing, N (%)"}
    return data.rename(columns=__rename)[list(__rename.values())]


def adjust_missing_counts_for_positive_culture_sites(data: pd.DataFrame, total: int,
                                                     culture_site_positives: str,
                                                     culture_site_counts: str) -> pd.DataFrame:

    count_positives = data.loc[data["index"].isin([culture_site_positives])]["value"].str.split(" ", expand=True)[
        0].astype(int).sum()
    count_culture_site = data.loc[data["index"].isin([culture_site_counts])]["value"].str.split(" ", expand=True)[
        0].astype(int).sum()

    count_missing = total - count_culture_site

    mean_positive = count_positives / count_culture_site * 100

    _value = f"{count_positives} ({mean_positive:.2f}%)"
    _missing = f"{count_missing} ({count_missing / total * 100:.2f}%)"

    data.loc[data["index"].isin([culture_site_positives]), "value"] = _value
    data.loc[data["index"].isin([culture_site_positives]), "Missing, N (%)"] = _missing

    return data


def process_df(data: pd.DataFrame, rename: dict, total: int):
    _data = data.copy()
    _rename = {k: i for k, i in rename.items()}

    _data, _rename = combine_boolean_categories(
        data=_data,
        rename=_rename,
        total=total,
        search_keyword="ADD_SPECIALISM_TO_OTHER",
        target_keyword="mic__group_other__event_m3d__ohe"
    )

    _data, _rename = combine_boolean_categories(
        data=_data,
        rename=_rename,
        total=total,
        search_keyword="ADD_TO_OTHER_M3D",
        target_keyword="mic__group_other__event_m3d__ohe"
    )

    _data, _rename = combine_boolean_categories(
        data=_data,
        rename=_rename,
        total=total,
        search_keyword="ADD_TO_OTHER_M1D_P1D",
        target_keyword="mic__group_other__start_m1d_p1d__ohe"
    )

    _data, _rename = combine_boolean_categories(
        data=_data,
        rename=_rename,
        total=total,
        search_keyword="ADD_TO_OTHER_M1D_P3D",
        target_keyword="mic__group_other__start_m1d_p3d__ohe"
    )


    _data = adjust_missing_counts(
        data=_data,
        records_to_group=["is_sex__male", "is_sex__female"],
        total=total,
    )

    _data = adjust_missing_counts(
        data=_data,
        records_to_group=["adm_hosp_route__elective__ohe", "adm_hosp_route__unplanned__ohe"],
        total=total,
    )

    adjustable_missing_counts = [
        ("mic__is_positive__blood__start_m1d_p1d__any", "mic__is_blood__start_m1d_p1d__any"),
        ("mic__is_positive__blood__start_m1d_p3d__any", "mic__is_blood__start_m1d_p3d__any"),
        ("mic__is_positive__blood__event_m3d__any", "mic__is_blood__event_m3d__any"),

        ("mic__is_positive__sputum__start_m1d_p1d__any", "mic__is_sputum__start_m1d_p1d__any"),
        ("mic__is_positive__sputum__start_m1d_p3d__any", "mic__is_sputum__start_m1d_p3d__any"),
        ("mic__is_positive__sputum__event_m3d__any", "mic__is_sputum__event_m3d__any"),

        ("mic__is_positive__urine__start_m1d_p1d__any", "mic__is_urine__start_m1d_p1d__any"),
        ("mic__is_positive__urine__start_m1d_p3d__any", "mic__is_urine__start_m1d_p3d__any"),
        ("mic__is_positive__urine__event_m3d__any", "mic__is_urine__event_m3d__any"),

        ("mic__is_positive__tip__start_m1d_p1d__any", "mic__is_tip__start_m1d_p1d__any"),
        ("mic__is_positive__tip__start_m1d_p3d__any", "mic__is_tip__start_m1d_p3d__any"),
        ("mic__is_positive__tip__event_m3d__any", "mic__is_tip__event_m3d__any"),
    ]

    for pos, count in adjustable_missing_counts:
        _data = adjust_missing_counts_for_positive_culture_sites(
            data=_data,
            total=total,
            culture_site_positives=pos,
            culture_site_counts=count,
        )

    _data = add_headers(data=_data, rename=_rename)
    _data = sort_by_rename(data=_data, rename=_rename)
    _data = rename_features(data=_data, rename=_rename)
    _data = remove_unused_records(data=_data, rename=_rename)
    _data = fix_headers(data=_data)

    return _data


def get_total_patients(data, search_keyword: str = "hospital") -> int:
    return data.loc[data["index"].str.contains(search_keyword)]["value"].str.split(" ",
                                                                                   expand=True)[
        0].astype(int).sum()


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

    fs = FeatureSelector(config=_config)
    fs.run()
    print("Done")
