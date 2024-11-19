import logging
import os
from logging import Logger
from typing import Any, Iterable

from matplotlib import pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import tadam as td
import seaborn as sns

from abstop.config import Config

logger: Logger = logging.getLogger(__name__)


class Featurizer:
    """
    Constructs aggregated features based on events and measurements.

    The Aggregator class is initialized with a Config object. This object
    contains the settings for the experiment.

    The run() method transforms incoming hospital data into a usable format for the
    experiment. The resulting table is saved to the processed data directory.
    """

    def __init__(self, config: Config):
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.logger.debug(f"from {__name__} instantiate {self.__class__.__name__}")
        self.config = config
        self._cfg = self.config.settings.featurizer

        self.data = self._load_data("aggregated")

    def run(self) -> pd.DataFrame:
        """
        Run the aggregator with the settings defined in the config file.
        :return: pd.DataFrame
        """
        self.logger.info("Running featurizer")

        # the next atc should always only be considered if the outcome label is present
        _outcome = self.data["outcome_restart_in_72h_on_icu"] == 1
        self.data.loc[~_outcome, "outcome_next_atc"] = np.nan

        self.process_transformations()

        self.process_ohe_features()

        self.process_time_differences()

        self.exclude_patients()

        # Calculate restart of same antibiotic
        self.logger.debug("Calculating restart of same antibiotic")
        self.data["outcome_restart_same_antibiotic"] = self.data.apply(
            lambda x: _contains(
                x=x,
                col_a="atc_last_24h",
                col_b="outcome_next_atc",
            ),
            axis=1,
        )
        # Calculate restart of same antibiotic group
        self.logger.debug("Calculating restart of same antibiotic group")
        self.data["outcome_restart_same_antibiotic_group"] = self.data.apply(
            lambda x: _contains(
                x=x,
                col_a="atc_last_24h__groups",
                col_b="outcome_next_atc__groups",
            ),
            axis=1,
        )

        self.data["outcome_restart_same_antibiotic_g2"] = self.data.apply(
            lambda x: _contains(
                x=x,
                col_a="atc_last_24h__g2",
                col_b="outcome_next_atc__g2",
            ),
            axis=1,
        )


        # Calculate overlap in antibiotics for restarts
        self.log_overlap_in_restarts(
            outcome="outcome_restart_in_72h_on_icu",
        )

        self.plot_restart_events()

        # cast feature names to lowercase
        self.data.columns = [c.lower() for c in self.data.columns]

        # impute missing values in boolean columns
        # MIC__GROUP_negative: True | False | NaN --> False
        # is_male: True | False | NaN --> True

        # cast boolean to int
        bool_columns = self.data.select_dtypes(include="bool").columns
        columns_bool_to_int = {c: int for c in bool_columns}
        self.data = self.data.astype(columns_bool_to_int)

        # save data
        self.logger.info("Saving data")
        self.logger.debug(f"{self.data.shape = }")
        output: tuple[str, str] = self._cfg.files.get(
            "output",
            ("processed", "featurized.pkl.gz"),
        )
        output_path = os.path.join(
            self.config.directory(output[0]),
            output[1]
        )
        self.logger.debug(f"Saving data to {output_path}")
        td.dump(self.data, output_path)

        return self.data

    def log_overlap_in_restarts(self, outcome: str) -> None:
        matching = self.data["atc_last_24h"].astype("object").apply(lambda x: sorted(x)).apply(
            lambda x: "|".join(x)) == self.data["outcome_next_atc"].astype("object").apply(
            lambda x: sorted(x) if isinstance(x, tuple) else []).apply(
            lambda x: "|".join(x))
        self.logger.info("OVERLAP - EXACT MATCH - RESTARTS")
        self.logger.info(matching.loc[self.data[outcome].astype(bool)].agg(
            ["mean", "sum"]))

        matching = self.data["atc_last_24h__groups"].astype("object").apply(
            lambda x: sorted(x)).apply(lambda x: "|".join(x)) == self.data[
                       "outcome_next_atc__groups"].astype("object").apply(
            lambda x: sorted(x) if isinstance(x, tuple) else []).apply(
            lambda x: "|".join(x))
        self.logger.info("OVERLAP - EXACT MATCH - RESTARTS - GROUPS")
        self.logger.info(matching.loc[self.data[outcome].astype(bool)].agg(
            ["mean", "sum"]))

        self.logger.info("OVERLAP - ANY MATCH - RESTARTS")
        self.logger.info(self.data.loc[
            self.data[outcome].astype(bool)
        ]["outcome_restart_same_antibiotic"].agg(["mean", "sum"]))

        self.logger.info("OVERLAP - ANY MATCH - RESTARTS - GROUPS")
        self.logger.info(
            self.data.loc[self.data[outcome].astype(bool)
            ]["outcome_restart_same_antibiotic_group"].agg(["mean", "sum"])
        )

    def plot_restart_events(self) -> None:

        # Define the groups from the dataset
        microbiology_culture_result_groups = [x for x in self.data if 'MIC' in x if 'GROUP' in x if 'START_M1D_P1D' in x]
        microbiology_culture_gram_groups = [x for x in self.data if 'MIC__GRAM' in x]
        microbiology_culture_type_groups = [x for x in self.data if 'IS_POSITIVE' in x if 'M1D_P1D' in x]

        # sanity check for is_positive and culture type groups
        assert ((~self.data[microbiology_culture_type_groups].any(axis=1)) & (~self.data[
            "MIC__GROUP_negative__START_M1D_P1D__OHE"].fillna(True))).mean() == 0.0, "There are records with a positive culture without any culture site"

        # {str: str}
        mic_cul_res = {f"{x.split('__')[1][6:].replace('_', ' ')}": x for x in microbiology_culture_result_groups}
        mic_cul_gram = {f"{x.split('__')[1]}": x for x in microbiology_culture_gram_groups}
        mic_cul_type = {f"{x.split('__')[2]}": x for x in microbiology_culture_type_groups}

        # {str: list[str, str]}
        # NOTE: we do not do this part because:
        #   1. Combining the groups results in an explosion of plots (4*13=52)
        #   2. The grouping results will contain artifacts: e.g. a positive blood culture with a negative urine culture etc.

        self.plot_restart_heatmap_annotated(
            df=self.data,
            name="restart_events__all",
            description="all"
        )
        self.plot_restart_heatmap_annotated(
            df=self.data,
            name="restart_events__all__g2",
            description="all",
            col_last="atc_last_24h__g2",
            col_next="outcome_next_atc__g2",
        )

        # Filter on positive matches, and plot the heatmap
        # Note: this includes all positive matches, and some antibiotics may be given
        #  for a positive culture for another infectious agent or site.
        # groups_to_plot = {}
        # groups_to_plot.update(mic_cul_res)
        # groups_to_plot.update(mic_cul_gram)
        # groups_to_plot.update(mic_cul_type)

        # for description, column in groups_to_plot.items():
        #     self.plot_restart_heatmap_annotated(
        #         df=self.data.loc[self.data[column].fillna(False)],
        #         name=f"restart_events__{description}",
        #         description=description)

        # repeat the plots, but only for records with a single site or single culture type
        # For site data: sum of the columns max 1, then filter on the same columns
        # For culture type data: sum of the columns w/o negative, or only negative
        # single_site = self.data[microbiology_culture_type_groups].sum(axis=1) == 1
        # single_group = (
        #         (self.data[microbiology_culture_result_groups].sum(axis=1) == 1) |
        #         (
        #                 (self.data[microbiology_culture_result_groups].sum(axis=1) == 2)
        #                 & self.data["MIC__GROUP_negative__START_M1D_P1D__OHE"]
        #         )
        # )
        #
        # self.plot_restart_heatmap_annotated(
        #     df=self.data.loc[single_site],
        #     name="restart_events__single_site__all",
        #     description="single_site__all"
        # )

        # for description, column in groups_to_plot.items():
        #     self.plot_restart_heatmap_annotated(
        #         df=self.data.loc[single_site & self.data[column].fillna(False)],
        #         name=f"restart_events__single_site__{description}",
        #         description=description)
        #
        # self.plot_restart_heatmap_annotated(
        #     df=self.data.loc[single_group],
        #     name="restart_events__single_group__all",
        #     description="single_group__all"
        # )
        #
        # for description, column in groups_to_plot.items():
        #     self.plot_restart_heatmap_annotated(
        #         df=self.data.loc[single_group & self.data[column].fillna(False)],
        #         name=f"restart_events__single_group__{description}",
        #         description=description)


    def process_time_differences(self) -> None:
        self.logger.info("Processing time differences")
        self.logger.debug(f"{self.data.shape = }")
        time_differences: list[dict] = self._cfg.time_differences
        self.logger.debug("Time differences: %s", time_differences)
        if time_differences:
            self._calc_time_differences(time_differences)
        else:
            self.logger.debug("No time differences specified")
        self.logger.debug(f"{self.data.shape = }")

    def process_ohe_features(self) -> None:
        self.logger.info("Processing one-hot encoded features")
        self.logger.debug(f"{self.data.shape = }")
        ohe_features: list[str] = self._cfg.ohe_features
        self.logger.debug("One-hot encoded features: %s", ohe_features)
        if ohe_features:
            self._process_ohe_features(ohe_features)
        else:
            self.logger.debug("No one-hot encoded features specified")
        self.logger.debug(f"{self.data.shape = }")

    def _process_ohe_features(self, features: list[str]) -> None:
        new_data: list[pd.DataFrame] = list()
        for feature in features:
            self.logger.debug(f"Processing one-hot encoded feature {feature}")
            self.data = pd.concat([
                self.data,
                self.to_ohe(feature),
                ], axis=1)

    def process_transformations(self) -> None:
        self.logger.info("Processing transformations")
        transformations: list[dict[str, str]] = self._cfg.transformations
        self.logger.debug("Transformations: %s", transformations)
        self._process_transformations(ctx=transformations)
        self.logger.debug(f"{self.data.shape = }")

    @staticmethod
    def get_atc_groups(x: Any) -> tuple | float:
        if isinstance(x, Iterable):
            return tuple(set(sorted([i[:4] for i in x if i])))
        return np.nan

    def transform_atc_to_groups(self, column: str) -> pd.Series:
        """Transform a column of atc codes to a column of atc groups.

        :param column: column name
        :return: pd.Series
        """

        col = self.data[column]
        if isinstance(col.dtype, CategoricalDtype):
            ser = col.astype("object").apply(self.get_atc_groups).astype("category")
        else:
            ser = col.apply(self.get_atc_groups)
        return ser

    def transform_atc_to_g2(self, column: str) -> pd.Series:
        """
        Transform a column of atc codes to a column of atc groups using the new grouping structure
        :param column: colum name
        :return: pd.Series
        """

        __nan_value__ = "____"

        def map_within_tuple(x: tuple, d: dict):
            return tuple([d.get(i, i) for i in x])

        col = self.data[column]
        if isinstance(col.dtype, CategoricalDtype):
            ser = col.astype("object").fillna(__nan_value__).apply(lambda x: map_within_tuple(x, self._cfg.atc_to_group)).astype("category")
        else:
            ser = col.fillna(__nan_value__).apply(lambda x: map_within_tuple(x, self._cfg.atc_to_group))

        ser.loc[ser == __nan_value__] = np.nan
        ser.loc[ser == (__nan_value__, )] = np.nan

        return ser

    def _process_transformations(self, ctx: list[dict[str, str]]) -> None:
        """
        """
        for c in ctx:
            method = c.get("method", None)
            if method is None:
                raise ValueError("method must be defined")
            target_name = c.get("target_column", None)
            if target_name is None:
                raise ValueError("target_name must be defined")
            source_name = c.get("source_column", None)
            if source_name is None:
                raise ValueError("source_name must be defined")

            match method:
                case "get_atc_groups":
                    self.data[target_name] = self.transform_atc_to_groups(source_name)
                case "get_atc_g2":
                    self.data[target_name] = self.transform_atc_to_g2(source_name)

    def to_ohe(self, column: Any) -> pd.DataFrame:
        """Transform a categorical column into a dataframe of one-hot encoded columns.

        :param column: column name
        :return: pd.DataFrame
        """
        self.logger.critical(f"to_ohe: {column}")
        mlb = MultiLabelBinarizer()
        self.logger.debug(self.data[column])

        data = self.data[column]
        if isinstance(data.dtype, CategoricalDtype):
            data = data.cat.add_categories([("NONE", )])

        idx_arrays = np.where(data.isnull())
        # idx_tups = zip(idx_arrays[0], idx_arrays[1])
        for tup in idx_arrays:
            if len(tup) > 0:
                data.iloc[tup] = [("NONE", )] * len(tup)

        if isinstance(data.iloc[0], str):
            # works for now as the string columns do not have missing values right now
            self.logger.critical(data.iloc[0])
            data = data.apply(lambda x: (x,) if is_iterable(x) else (np.nan, ))
        _dum = mlb.fit_transform(data)
        dum = pd.DataFrame(
            _dum,
            columns=mlb.classes_,
            index=self.data.index)
        dum_columns = [
            column + "__" + str(col) + "__ohe" for col in dum.columns
        ]
        dum.columns = dum_columns
        return dum

    def _calc_time_differences(self, ctx: list[dict[str, Any]]) -> None:
        """Calculate time differences between events and measurements.

        :param ctx: list of dicts with context information
        :return: None, modifies the data object
        """

        for c in ctx:
            fun = c.get("fun", None)
            if fun is None:
                raise ValueError("function must be defined")
            target_name = c.get("name", None)
            if target_name is None:
                raise ValueError("target_name must be defined")

            kwargs = c.get("kwargs", {})
            for name, value in kwargs.items():
                if name.startswith('df_'):
                    kwargs[name] = self.data[value]
            if fun:
                self.data[target_name] = fun(**kwargs)

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

    def plot_restart_heatmap_annotated(self,
                                       df,
                                       name,
                                       description,
                                       col_last='atc_last_24h',
                                       col_next='outcome_next_atc'):

        self.logger.info(f"Plotting restart events heatmap for {description}")

        if df.shape[0] == 0:
            self.logger.info(f"No records for {description}")
            return

        # Note: outcome_next_atc is already masked for outcome within time window at the parent function

        # For each group, perform the function which plots the heatmap
        # reduce makes the ATC level be reduced to the first 4 characters of the ATC system
        # when using new groups, as long as the lenght of the name is below 4 characters, the entire
        # name will be used.
        a = _reduce_atc_cat(df[col_last])
        b = _reduce_atc_cat(df[col_next])
        a, b = _combine_cats(a, b)
        self.logger.info(f"{(a == b).mean() = }")

        self.logger.info(
            "Records where at least one antibiotic overlaps at ATC level: "
            f""
            f""
            f"{self.data.apply(lambda x: _contains(x=x, col_a=col_last, col_b=col_next), axis=1).mean()}"
        )

        list_of_lists = list(
            pd.DataFrame([a, b]).T.apply(lambda x: _get_product_elements(x,
                                                                         col_a=col_last,
                                                                         col_b=col_next),
                                         axis=1).values)
        q = [item for sublist in list_of_lists for item in sublist]

        name_last_ab = "Last Antibiotic"
        name_next_ab = "Restarted Antibiotic"

        r = pd.DataFrame(q).fillna("Stopped").groupby(0, dropna=False)[1].value_counts(
            dropna=False, normalize=True).reset_index().rename(
            columns={0: name_last_ab, 1: name_next_ab})
        r.to_csv(os.path.join(self.config.directory("figures"),
                              f"heatmap__r__{description}.csv"))
        s = r.pivot(columns=name_last_ab, index=name_next_ab,
                    values="proportion")

        s.to_csv(os.path.join(self.config.directory("figures"),
                              f"heatmap__s__{description}.csv"))

        labels = np.asarray([round(value, 3) for value in s.values.flatten()]).reshape(
            s.shape)

        atc_rename = {
            "Stopped": "Stopped",
            "J01A": "Tetracyclines",
            "J01C": "β-lactam\n(Penicillins)",
            "J01D": "Other\nβ-lactam",
            "J01E": "Sulfonamides\n& trimethoprim",
            "J01F": "Macrolides,\nLincosamides\n& Streptogramins",
            "J01G": "Aminoglycosides",
            "J01M": "Quinolones",
            "J01X": "Other\nAntibacterials",
            "J01Z": "Vancomycin",
        }
        atc_rename.update(self._cfg.g2_rename)

        t = [i[0] for i in q]
        x_name = "Last Antibiotic"
        u = pd.Series(t).to_frame(x_name)

        ## COUNTPLOT INCLUDED WITH HUMAN LABELS AND PERCENTAGES
        sns.set(style="white", font_scale=0.8)
        fig, ax = plt.subplots(
            nrows=2,
            ncols=2,
            sharex="col",
            figsize=(8, 8),
            gridspec_kw={
                "height_ratios": [5, 1],
                "width_ratios": [100, 5]
            }
        )
        ax[1, 1].remove()
        ax2 = sns.countplot(
            data=u.sort_values(x_name),
            x=x_name,
            color="grey",
            order=sorted(u[x_name].unique()),
            ax=ax[1, 0],
        )

        # set x-axis labels to rotated by 90 degrees
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, fontsize=10)
        ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=10)
        ax2.set_ylabel("Count")

        yscale = "linear"
        if yscale == "log":
            ax2.set_yscale("log")
            ax2.set_ylim((0, int(round(u.value_counts().max() * 2.5, 0))))
        else:
            ax2.set_ylim((0, int(round(u.value_counts().max() * 1.20, 0))))

        # Shift the countplot half a step to the right
        new_value = 1.0
        recenter = True
        for patch in ax2.patches:
            current_width = patch.get_width()
            patch.set_width(new_value)
            if recenter == True:
                patch.set_x(patch.get_x() + current_width * .5)  # To recenter the bars

            x = patch.get_bbox().get_points()[:, 0]
            y = patch.get_bbox().get_points()[1, 1]
            ax2.annotate("{:.0f}".format(patch.get_height()), (x.mean(), y),
                         ha="center", va="bottom")
            # patch.annotate("{:.0f}".format(patch.get_height()), (patch.get_x() + patch.get_width() / 2., patch.get_height()+1))

        z = sns.heatmap(
            s.rename(columns=atc_rename, index=atc_rename) * 100,
            annot=labels,
            annot_kws={"size": 10},
            fmt=".1%",
            vmin=0,
            vmax=100,
            cmap="viridis",
            square=False,
            ax=ax[0, 0],
            cbar_ax=ax[0, 1],
        )

        z.axes.set_yticklabels(z.axes.get_ymajorticklabels(), fontsize=10)

        # ax[0, 1].set_xticklabels([round(x*100) for x in ax[0, 1].get_xticklabels()], fontsize=12)
        z.set_facecolor("xkcd:white")
        z.set(xlabel="")
        plt.tight_layout()
        # filepath = os.path.join(self.config.directory("figures"),
        #                         f"{name}.png")
        # fig.savefig(filepath, dpi=1200)
        filepath = os.path.join(self.config.directory("figures"),
                                f"{name}.pdf")
        fig.savefig(filepath, dpi=1200)
        plt.close()
        print("debug")

    def exclude_patients(self):

        # Exclude patients dying within 72 hours after stopping antibiotics
        self.logger.critical(f"Excluding mortality <72: {self.data.shape[0]} --> ")
        # self.data = self.data.loc[
        #     ~(self.data['outcome_timedelta_mortality'] <= pd.Timedelta(79.2, 'h'))
        # ].copy()
        self.logger.critical(f"{self.data.shape[0]}")

        # Excluding patients dying before the outcome could be determined
        self.logger.critical(f"Excluding undetermined outcomes: {self.data.shape[0]} --> ")
        self.data = self.data.loc[
            ( # Keep patients where timedelta mortality is greater than the limit
                ~(self.data['outcome_timedelta_mortality'] < pd.Timedelta(79.2, 'h'))
            ) |
            ( # And keep patients dying within 72h, but where the outcome is already determined
                    (self.data['outcome_timedelta_mortality'] < pd.Timedelta(79.2, 'h')) &
                    (self.data['outcome_restart_in_72h_on_icu'])
            )
            ].copy()
        self.logger.critical(f"{self.data.shape[0]}")

        # Exclude patients being discharged within 72 hours after stopping antibiotics
        self.logger.critical(f"Excluding discharge <24: {self.data.shape[0]} --> ")
        self.data = self.data.loc[
            ~(self.data['outcome_timedelta_icu_discharge'] <= pd.Timedelta(26.4, 'h'))
        ].copy()
        self.logger.critical(f"{self.data.shape[0]}")


def is_iterable(x):
    try:
        iter(x)
        return True
    except TypeError:
        return False


def __reduce(x):
    if isinstance(x, float):
        return np.nan
    if isinstance(x, tuple):
        return tuple(set([i[:4] for i in x if i]))

def _reduce_atc_cat(col):
    return col.astype("object").apply(lambda x: __reduce(x)).astype("category")

def _combine_cats(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    old_a = set(a.cat.categories)
    old_b = set(b.cat.categories)
    new_a = list(old_b - old_a)
    new_b = list(old_a - old_b)
    return a.cat.add_categories(new_a), b.cat.add_categories(new_b)


def _contains(x: pd.Series, col_a: str, col_b: str) -> bool:
    a = x[col_a]
    b = x[col_b]

    if isinstance(a, tuple):
        if isinstance(b, tuple):
            for i in b:
                if i in a:
                    return True
    return False


def _get_product_elements(x: pd.Series, col_a: str, col_b: str) -> list[tuple] | float:
    _a = x[col_a]
    _b = x[col_b]
    _new = list()
    if isinstance(_a, tuple):
        for i in _a:
            if isinstance(_b, tuple):
                for j in _b:
                    _new.append((i, j))
            else:
                _new.append((i, np.nan))
    else:
        return np.nan
    return _new


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

    ft = Featurizer(config=_config)
    ft.run()
