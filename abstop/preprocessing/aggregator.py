import logging
import os
import warnings
from logging import Logger
from typing import Any

import duckdb as duckdb
import pandas as pd
import tadam as td
import tadam.dataprocessing as dp

from abstop.config import Config

logger: Logger = logging.getLogger(__name__)


class Aggregator:
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
        self._cfg = self.config.settings.aggregator

        self.data = self._load_data("measurements")
        self.events = self._load_data("events")
        self.windows, self.window_id_columns = self._load_windows()

    def run(self) -> pd.DataFrame:
        """
        Run the aggregator with the settings defined in the config file.
        :return: pd.DataFrame
        """

        self.logger.info("Running aggregator")
        features = self._get_features()
        self.logger.info(f"{features.shape = }")
        self.logger.info(f"{features.columns = }")

        features = self._post_process(features)

        self.logger.info("Combining features with events data")
        data = pd.concat(
            [self.events, features], axis=1, ignore_index=False, copy=False
        )
        self.logger.info(f"{data.shape = }")
        self.logger.info(f"{data.columns = }")
        self.logger.info(f"{data.dtypes = }")
        self.logger.info(f"{round(data.memory_usage(deep=True) / 1024 ** 2, 2) = }")

        _path, _file = self._cfg.files.get(
            "output", ("processed", "__aggregated__.pkl.gz")
        )
        file_path = os.path.join(self.config.directory(_path), _file)
        td.dump(obj=data, path=file_path)
        self.logger.debug(f"saved aggregated data to {file_path}")
        return data

    def _post_process(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Post process features.

        This method is called after the features are aggregated. This method
        is used to add additional columns to the features table.

        :param features: pd.DataFrame
        :return: pd.DataFrame
        """

        self.logger.info("Post processing features")
        _post_features: list[dict] = self._cfg.post_features # type: ignore[assignment]
        for _feature in _post_features:
            _columns = _feature.get("features", [])
            _method = _feature.get("method")
            _target = _feature.get("target_feature")

            if _method == "bool_or_above_threshold":
                _threshold = _feature.get("threshold")

                if _threshold is not None:
                    _above_threshold = features[_columns] > _threshold
                    if isinstance(_above_threshold, pd.DataFrame):
                        _above_threshold = _above_threshold.any(axis=1)
                    elif isinstance(_above_threshold, pd.Series):
                        pass
                else:
                    raise ValueError(f"no threshold defined for {_feature}")
                features[_target] = _above_threshold
            elif _method == "proportional_difference":
                if len(_columns) != 2:
                    raise ValueError(f"expected 2 columns for {_feature}")
                features[_target] = (
                    features[_columns[1]] - features[_columns[0]]
                ) / features[_columns[0]]
        return features

    def _get_features(self) -> pd.DataFrame:
        """
        Get features from config file.

        Based on the settings defined in the config file, aggregate the measurements
        table into features. The features are defined by a list of dictionaries.
        features: list[dict] = [
            {
                "variables": list[str], # list of names in variable column
                "window": str,  # name of window used to specifying feature names
                "agg_func": str, # name of agg func for specifying feature names
                "merge_kwargs": dict[str, Any] = { # passed to dp.merge_windows
                    "on": list[str],  # [patient id]
                    "windows_start": str,  # column name windows start time
                    "windows_end": str,  # column name windows end time
                    "measurements_start": str,  # column name measurements start time
                    "measurements_end": str,  # column name measurements end time
                    "group_by": list[str],  # [patient id, window id!!, {variable_id}]
                    "variable_id": str,  # column name holding the variables
                    "agg_func": str,
                    "map_columns": bool,
                },
            },
        ]
        :return: pd.DataFrame
        """
        ctx = self._cfg.features

        _dfs = list()

        for feature in ctx:
            _variables = feature.get("variables")
            _window_name = feature.get("window")
            _agg_func_name = feature.get("agg_func")
            _suffix = f"{_window_name}__{_agg_func_name}"

            _merge_kwargs: dict = feature.get(
                "merge_kwargs", {}
            )  # type: ignore[assignment]

            window_ids = set(self.window_id_columns)

            __on = _merge_kwargs.get("on", [])

            _window_columns = {
                *__on,
                _merge_kwargs["windows_start"],
                _merge_kwargs["windows_end"],
            }
            _window_columns.update(window_ids)
            _window = self.windows[list(_window_columns)].copy()

            _data = self.data.loc[self.data["variable"].isin(_variables)]

            _feature_df = dp.merge_windows(
                windows=_window,
                measurements=_data,
                **_merge_kwargs,
            )

            # _target_vars = set(_variable_rename_dict.values())
            # _keep_columns = _target_vars  # | window_ids

            _drop_columns = [x for x in _window.columns if x in _feature_df.columns]
            _feature_df = _feature_df.drop(columns=_drop_columns)

            _variable_rename_dict = {v: f"{v}__{_suffix}" for v in _feature_df.columns}
            _feature_df = _feature_df.rename(columns=_variable_rename_dict)

            _dfs.append(_feature_df)

        return pd.concat(_dfs, axis=1, ignore_index=False, copy=False)

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

    def _load_windows(self) -> pd.DataFrame:
        """
        Load windows from events table.
        :return:
        """
        self.logger.debug("load windows")
        ctx = self._cfg.windows

        event_columns = ctx.get("event_columns", ["pid", "stop"])
        self.logger.debug(f"{event_columns = }")
        windows = self.events[event_columns].copy()
        self.logger.debug(f"{self.events.shape = }")
        self.logger.debug(f"{windows.shape = }")

        rename_columns = ctx.get("rename_columns")
        if rename_columns:
            self.logger.debug(f"{rename_columns = }")
            windows = windows.rename(columns=rename_columns)
        else:
            self.logger.debug("no rename columns")

        if "window_id" not in windows.columns:
            if len(set(windows.index)) != windows.shape[0]:
                _msg = "Index is not unique"
                warnings.warn(_msg)
                self.logger.warning(_msg)
                windows["window_id"] = pd.RangeIndex(
                    start=0, stop=windows.shape[0], step=1
                )
            else:
                windows["window_id"] = windows.index

        window_id_columns = list(windows.columns)
        self.logger.debug(f"{window_id_columns = }")

        create_columns: dict = ctx.get("create_columns")  # type: ignore[assignment]
        if create_columns:
            self.logger.debug(f"{create_columns = }")
            self.logger.debug(f"{windows.shape = }")
            windows = self._create_columns(windows, create_columns)
            self.logger.debug(f"{windows.shape = }")
            self.logger.debug(f"{windows.columns = }")
        else:
            self.logger.debug("no create columns")

        return windows, window_id_columns

    def _create_columns(
        self, windows: pd.DataFrame, create_columns: dict[Any, Any]
    ) -> pd.DataFrame:
        for target_column, settings in create_columns.items():
            windows[target_column] = windows[settings["source"]] + settings["offset"]
        return windows


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

    ag = Aggregator(config=_config)
    df = ag.run()
    print("Done")
