import logging
import os
import warnings
from logging import Logger
from typing import Any

import duckdb as duckdb
import pandas as pd
import tadam as td
import tadam.dataprocessing as dp
from pandas import Interval
from pandas.core.dtypes.common import is_categorical_dtype, is_numeric_dtype

from abstop.config import Config
from abstop.utils.converter import Converter

logger: Logger = logging.getLogger(__name__)


class MeasurementsProcessor:
    """
    Constructs events based on antibiotics data.

    The MeasurementsProcessor class is initialized with a Config object. This object
    contains the settings for the experiment. The AntibioticsPreprocessor class loads
    the tables from the raw data directory.

    The run() method transforms incoming hospital data into a usable format for the
    experiment. The resulting table is saved to the processed data directory.
    """

    def __init__(self, config: Config):
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.logger.debug(f"from {__name__} instantiate {self.__class__.__name__}")
        self.config = config
        self._cfg = self.config.settings.measurements

        self.patients = self._load_patients()
        self.patient_ids = set(self.patients["pid"].unique())

    def run(self) -> pd.DataFrame:
        medication = self._load_medication()
        measurements = self._load_measurements()

        ctx = self._cfg.data_loader.singles
        retype_columns: dict = ctx.get("retype")  # type: ignore[assignment]
        if retype_columns:
            medication = medication.astype(retype_columns)[list(retype_columns.keys())]
            measurements = measurements.astype(retype_columns)[
                list(retype_columns.keys())
            ]
            medication, measurements = self._combine_categories(
                dfs=[medication, measurements],
                column="variable",
                upper=True,
            )
        _concat = pd.concat(
            [medication, measurements],
            axis=0,
            ignore_index=True,
            copy=False,
        )
        filename = self._cfg.files.get("output")
        if filename is None:
            filename = "measurements.pkl"
            error_msg = (
                "No filename specified for measurements, defaulting to" f"{filename}"
            )
            self.logger.error(error_msg)
        dump_path = os.path.join(self.config.directory("processed"), filename)
        td.dump(obj=_concat, path=dump_path)
        return _concat

    def _load_patients(self) -> pd.DataFrame:
        filename = self._cfg.data_loader.patients.get("filename")
        if filename:
            path = os.path.join(
                self.config.directory("processed"),
                filename,
            )
            return td.load(path)
        else:
            error_msg = "No filename specified for patients"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

    def _combine_categories(
        self,
        dfs: list[pd.DataFrame],
        column: str,
        upper: bool = True,
    ) -> list[pd.DataFrame]:
        categories = set()
        for df in dfs:
            df[column] = df[column].cat.remove_unused_categories()
            categories.update(df[column].cat.categories)

        if upper:
            categories = set([x.upper() for x in categories])

        for df in dfs:
            existing_categories = set(df[column].cat.categories)
            if upper:
                rename_categories = {x: x.upper() for x in existing_categories}
                df[column] = df[column].cat.rename_categories(
                    new_categories=rename_categories
                )
                existing_categories = set(df[column].cat.categories)
            new_categories = categories - existing_categories
            df[column] = df[column].cat.add_categories(new_categories)

        return dfs

    def _load_measurements(self) -> pd.DataFrame:
        _dfs = list()
        _dfs.append(self._load_single_timestamps())
        _dfs.append(self._load_range_timestamps())
        _dfs.append(self._load_microbiology())

        ctx = self._cfg.data_loader.singles
        retype_columns: dict = ctx.get("retype")  # type: ignore[assignment]
        if retype_columns:
            for i, df in enumerate(_dfs):
                _dfs[i] = df.astype(retype_columns)[list(retype_columns.keys())]

            _dfs = self._combine_categories(
                dfs=_dfs,
                column="variable",
                upper=True,
            )

        merged = pd.concat(
            objs=_dfs,
            axis=0,
            ignore_index=True,
            copy=False,
        )

        return merged

    def _load_microbiology(self) -> pd.DataFrame:
        ctx = self._cfg.data_loader.microbiology
        mic = self.get_data("microbiology")

        # 2024-06-25
        # Drop records which are not in the sites of interest
        sites_of_interest = ['is_blood', 'is_urine', 'is_sputum', 'is_tip']
        print(mic.shape)
        print(mic.columns)
        # self.logger.critical(f"{mic.shape = }")
        # mic = mic[mic[sites_of_interest].sum(axis=1) > 0].copy()
        # self.logger.critical("Dropped records not in sites of interest")
        # self.logger.critical(f"{mic.shape = }")

        _post_processing: dict = ctx.get("post_processing")  # type: ignore[assignment]
        if _post_processing:
            mic = self._post_processing(data=mic, ctx=_post_processing)
            _retype = _post_processing.get("retype")
            if _retype:
                mic = mic.astype(_retype)
        _melt: dict = ctx.get("melt")  # type: ignore[assignment]
        if _melt:
            _drop_nan_after_melt = _melt.get("drop_nan_after_melt")
            if _drop_nan_after_melt:
                mic = mic.dropna()
        return mic

    def _load_medication(self) -> pd.DataFrame:
        medication = self.get_data("medication")
        medication = self._process_medication(data=medication)
        medication = medication.rename(
            columns={
                "target_name": "variable",
                "dose": "value",
            }
        )
        return medication

    def _process_medication(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Processed medication from hetergenous names and dose units
        to standardized names and units.

        :param data: pd.DataFrame containing medication

        :return:
        """

        ctx = self._cfg.data_loader.medication

        # merge with patients to get current weights
        data = self._merge_medication_with_patients(
            data=data,
            patients=self.patients,
        )

        # convert units
        data = self._convert_medication_units(data=data)

        # Discrete, adjoining,
        _overlap_to_discrete = ctx.get("overlap_to_discrete")
        if _overlap_to_discrete:
            self.logger.debug("Discretizing medication")
            self.logger.debug(f"{_overlap_to_discrete = }")
            self.logger.debug(f"{data.shape = }")
            data = dp.overlap_to_discrete(
                data=data,
                timestamp_start="timestamp_start",
                timestamp_end="timestamp_end",
                value="dose",
                group_by=["pid", "target_name"],
                keep="sum",
            )
            self.logger.debug(f"{data.shape = }")

        _join_adjoining = ctx.get("join_adjoining_records")
        if _join_adjoining:
            self.logger.debug("Joining adjoining medication records")
            self.logger.debug(f"{_join_adjoining = }")
            self.logger.debug(f"{data.shape = }")
            data = dp.join_adjoining(
                data=data,
                **_join_adjoining,
            )
            self.logger.debug(f"{data.shape = }")

        _impute_missing = ctx.get("impute_missing_as_zero")
        if _impute_missing:
            self.logger.debug("Imputing missing medication records with 0")
            self.logger.debug(f"{_impute_missing = }")
            self.logger.debug(f"{data.shape = }")
            data = dp.impute_missing(
                data=data,
                windows=self.patients,
                **_impute_missing,
            )
            self.logger.debug(f"{data.shape = }")
        return data

    def _merge_medication_with_patients(
        self,
        data: pd.DataFrame,
        patients: pd.DataFrame,
    ) -> pd.DataFrame:
        """

        :param data:
        :param patients:
        :return:
        """

        merge: dict = self._cfg.data_loader.medication.get(
            "merge_patients"
        )  # type: ignore[assignment]

        if merge is None:
            _msg = "No merge settings found for medication"
            self.logger.error(_msg)
            raise ValueError(_msg)

        on = merge.get("on", "pid")
        data_timestamp = merge.get("data_timestamp", "timestamp")
        patients_lower = merge.get("patients_lower", "adm_icu_adm")
        patients_upper = merge.get("patients_upper", "adm_icu_dis")

        q = f"""
        SELECT d.*, p.weight, p.adjusted_weight, p.predicted_weight
        FROM data d 
        LEFT JOIN patients p 
        ON d.{on} = p.{on}
        AND d.{data_timestamp} >= p.{patients_lower}
        AND d.{data_timestamp} <= p.{patients_upper}
        """

        merged = duckdb.query(q).to_df()

        return merged

    def _convert_medication_units(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """

        :param data:
        :return:
        """
        ctx = self._cfg.data_loader.medication
        # convert non-continuous dose units
        # impute using standard dose if not convertible
        convert_weight_based: dict = ctx.get(
            "convert_weight_based"
        )  # type: ignore[assignment]

        if convert_weight_based:
            self.logger.debug("Converting weight based doses")
            self.logger.debug(f"{data.shape = }")
            data = self._convert_weight_based_dose(
                data=data, convert_weight_based=convert_weight_based
            )
            self.logger.debug(f"{data.shape = }")

        if is_categorical_dtype(data["dose_unit"]):
            required_categories = {"mg/hr", "mg", "microgr/hr"}
            used_categories = set(data["dose_unit"].cat.categories)
            new_categories = required_categories - used_categories
            if new_categories:
                data["dose_unit"] = data["dose_unit"].cat.add_categories(
                    list(new_categories)
                )

        converter = Converter()
        self.logger.debug("Converting dose and dose units")
        self.logger.debug(f"{data.shape = }")
        factors = converter.factors.factors
        source_factor = data["dose_unit"].map(factors)
        target_factor = pd.Series("mg/hr", index=data.index).map(factors)
        new_dose = data["dose"] * (source_factor / target_factor)
        self.logger.debug(f"{new_dose.shape = }")

        has_new_dose = new_dose.notna()
        self.logger.debug(f"Calculated {has_new_dose.sum()} doses")
        data.loc[has_new_dose, "dose"] = new_dose

        self.logger.debug("Correcting dose units")
        is_continuous = data["dose_unit"].str.contains(
            "|".join(["uur", "hr", "hour", "min"]), na=False
        )  # TODO: change to SPUITENPOMP | PROPOFOL WWSP?
        self.logger.debug(
            f"Correcting {(has_new_dose & is_continuous).sum()} continuous"
        )
        data.loc[has_new_dose & is_continuous, "dose_unit"] = "mg/hr"
        self.logger.debug(
            f"Correcting {(has_new_dose & ~is_continuous).sum()} non-continuous"
        )
        data.loc[has_new_dose & ~is_continuous, "dose_unit"] = "mg"

        # convert continuous dose units
        # calculate concentrations
        self.logger.debug("Calculating concentrations based on rate/dose")
        data["conc"] = data["dose"] / data["rate"]
        self.logger.debug(f"Calculated {data['conc'].notna().sum()} concentrations")

        # for noradrenaline adjust concentrations to nearest most likely value
        concentration_correction: dict = ctx.get(
            "concentration_correction_upper_limits"
        )  # type: ignore[assignment]
        if concentration_correction:
            self.logger.debug("Correcting concentrations")
            for medication_name, settings in concentration_correction.items():
                self.logger.debug(f"for {medication_name} with {settings}")
                medication_filter = data["target_name"].str.match(
                    medication_name, na=False
                )
                _min = settings.get("min")
                if _min:
                    conc_filter = data["conc"] < _min
                    data.loc[medication_filter & conc_filter, "conc"] = pd.NA
                _max = settings.get("max")
                if _max:
                    conc_filter = data["conc"] > _max
                    data.loc[medication_filter & conc_filter, "conc"] = pd.NA

                _bins = settings.get("bins")
                if _bins:
                    _keys = list(_bins.keys())
                    _values = list(_bins.values())[1:]
                    __keys = [
                        Interval(x, y, closed="right")
                        for x, y in zip(_keys[:-1], _keys[1:])
                    ]
                    __dict = {k: v for k, v in zip(__keys, _values)}

                    values = data.loc[medication_filter, "conc"]
                    rounding_vals = list(_bins.keys())
                    rounded = custom_round(col=values, rounding_vals=rounding_vals)
                    rounded = (
                        pd.Series(rounded, index=values.index).map(__dict).astype(float)
                    )

                    data.loc[medication_filter, "conc"] = rounded

        self.logger.debug(
            "Filling missing concentration with previous record and next record"
        )
        data_grouped = data.groupby(["pid", "prescription_id", "original_name"])["conc"]
        conc_prev = data_grouped.shift(1)
        conc_next = data_grouped.shift(-1)
        self.logger.debug(f"Missing concentrations: {data['conc'].isna().sum()}")
        self.logger.debug("Filling with previous concentrations")
        data["conc"] = data["conc"].fillna(conc_prev)
        self.logger.debug(f"Missing concentrations: {data['conc'].isna().sum()}")
        self.logger.debug("Filling with next concentrations")
        data["conc"] = data["conc"].fillna(conc_next)
        self.logger.debug(f"Missing concentrations: {data['conc'].isna().sum()}")

        self.logger.debug("Filling with the default concentration")
        data.loc[data["conc"].isna(), "conc"] = data["concentration"]
        self.logger.debug(f"Missing concentrations: {data['conc'].isna().sum()}")

        # calculate dose rate based on infusion rate and concentration
        is_continuous = data["rate"].notna()
        data.loc[is_continuous, "dose"] = data["conc"] * data["rate"]

        # TODO: calculate dose based on the number of tablets/units etc.
        _unit_doses = ctx.get("unit_doses")
        if _unit_doses:
            self.logger.debug("Converting unit doses")
            _new = data["dose"].astype(float) * data["dose_per_unit"].astype(float)
            _filter = data["dose_unit"].isin(_unit_doses) & _new.notna()
            self.logger.debug(f"{_filter.sum()} unit doses found")
            self.logger.debug(f"{data.loc[_filter]['dose'].describe() = }")
            data.loc[_filter, "dose"] = _new
            self.logger.debug(f"{data.loc[_filter]['dose'].describe() = }")
            data.loc[_filter, "dose_unit"] = "mg"
        # TODO: calculate dose based on the number of ml * concentration
        _ml_doses = ctx.get("ml_doses")
        if _ml_doses:
            self.logger.debug("Converting ml doses")
            _new = data["dose"].astype(float) * data["conc"].astype(float)
            _filter = data["dose_unit"].isin(_ml_doses) & _new.notna()
            self.logger.debug(f"{_filter.sum()} ml doses found")
            self.logger.debug(f"{data.loc[_filter]['dose'].describe() = }")
            data.loc[_filter, "dose"] = _new
            self.logger.debug(f"{data.loc[_filter]['dose'].describe() = }")
            data.loc[_filter, "dose_unit"] = "mg"

        _equivalents: dict = ctx.get("benzo_equivalents")  # type: ignore[assignment]
        if _equivalents:
            data = self._convert_benzo_equivalents(data=data, equivalents=_equivalents)
        else:
            self.logger.debug("No benzodiazepine equivalents defined")

        # adjust specific medication based on predicted_weight | adjusted_weight
        if convert_weight_based:
            if is_categorical_dtype(data["dose_unit"]):
                old_categories = set(data["dose_unit"].cat.categories)
                new_categories = set([f"{x}/kg" for x in old_categories])
                new_categories = new_categories - old_categories
                data["dose_unit"] = data["dose_unit"].cat.add_categories(
                    list(new_categories)
                )

            _predicted = convert_weight_based.get("predicted_weight")
            if _predicted:
                _records = data["target_name"].isin(_predicted)
                data.loc[_records, "dose"] /= (
                    data["predicted_weight"].fillna(data["weight"]).fillna(80)
                )

                if is_categorical_dtype(data["dose_unit"]):
                    data.loc[_records & data["dose_unit"].notna(), "dose_unit"] = (
                        data["dose_unit"].astype(str) + "/kg"
                    )
                else:
                    data.loc[_records & data["dose_unit"].notna(), "dose_unit"] += "/kg"
            _adjusted = convert_weight_based.get("adjusted_weight")
            if _adjusted:
                _records = data["target_name"].isin(_adjusted)
                data.loc[_records, "dose"] /= (
                    data["adjusted_weight"].fillna(data["weight"]).fillna(80)
                )
                if is_categorical_dtype(data["dose_unit"]):
                    data.loc[_records & data["dose_unit"].notna(), "dose_unit"] = (
                        data["dose_unit"].astype(str) + "/kg"
                    )
                else:
                    data.loc[_records & data["dose_unit"].notna(), "dose_unit"] += "/kg"

        # split records into continuous and single dose
        split_continuous: dict = ctx.get(
            "continuous_records"
        )  # type: ignore[assignment]
        if split_continuous:
            _names: list[str] = split_continuous.get(
                "names"
            )  # type: ignore[assignment]
            if _names:
                if is_categorical_dtype(data["dose_unit"]):
                    old_categories = set(data["dose_unit"].cat.categories)
                    new_categories = set([f"{x}/hr" for x in old_categories])
                    new_categories = new_categories - old_categories
                    data["dose_unit"] = data["dose_unit"].cat.add_categories(
                        list(new_categories)
                    )

                is_continuous = data["original_name"].str.contains(
                    "|".join(_names), na=False
                )

                common_cols = [
                    "pid",
                    "timestamp_start",
                    "timestamp_end",
                    "original_name",
                    "prescription_id",
                    "rate",
                    "rate_unit",
                    "dose",
                    "dose_unit",
                    "conc",
                    "target_name",
                    "target_atc",
                    "target_group",
                    "concentration",
                    "dose_per_unit",
                ]

                cont = data.loc[is_continuous]

                # transform to range
                cont = dp.single_to_range(
                    data=cont,
                    timestamp="timestamp",
                    group_by=["pid", "prescription_id", "original_name"],
                    direction="forward",
                    max_duration=pd.Timedelta(2, "h"),
                    fill_duration=pd.Timedelta(1, "h"),
                )

                cont = cont[common_cols]

                # set duration for non-continuous
                bolus = data.loc[~is_continuous]
                bolus = bolus.rename(columns={"timestamp": "timestamp_start"})
                bolus["timestamp_end"] = bolus["timestamp_start"] + pd.Timedelta(1, "m")
                bolus["dose"] *= 60  # 1 minute duration, and dose/hr

                if is_categorical_dtype(data["dose_unit"]):
                    new_values = bolus["dose_unit"].astype(str) + "/hr"
                    set(new_values.unique())
                    bolus.loc[bolus["dose_unit"].notna(), "dose_unit"] = new_values
                else:
                    bolus.loc[bolus["dose_unit"].notna(), "dose_unit"] = (
                        bolus["dose_unit"] + "/hr"
                    )

                bolus = bolus[common_cols]
                data = pd.concat([cont, bolus], axis=0, ignore_index=True, copy=False)

        # overlapping to discrete; based on pid and target name
        do_overlap_to_discrete = ctx.get("overlap_to_discrete")
        if do_overlap_to_discrete:
            pass

        # Remove records if infusion rate is 0.1 ml/hr
        _remove_rate: dict = ctx.get("remove_rate_based")  # type: ignore[assignment]
        if _remove_rate:
            for med, settings in _remove_rate.items():
                _med_filter = data["target_group"].str.match(med)

                _min = settings.get("min")
                if _min:
                    _rate_filter = data["rate"] < _min
                    _keep = ~(_med_filter & _rate_filter)
                    data = data.loc[_keep]

                _max = settings.get("max")
                if _max:
                    _rate_filter = data["rate"] > _max
                    _keep = ~(_med_filter & _rate_filter)
                    data = data.loc[_keep]

        return data

    def _convert_benzo_equivalents(
        self,
        data: pd.DataFrame,
        equivalents: dict,
    ) -> pd.DataFrame:
        self.logger.debug("Converting benzodiazepine doses")
        _mapper = {k: v["factor"] for k, v in equivalents.items()}
        self.logger.debug(f"{_mapper = }")
        _factors = data["target_name"].map(_mapper)
        self.logger.debug(f"{_factors.notna().sum() = }")
        _filter = data["target_name"].isin(_mapper.keys())
        self.logger.debug(f"{data.loc[_filter]['dose'].describe() = }")
        data.loc[_filter, "dose"] /= _factors
        self.logger.debug(f"{data.loc[_filter]['dose'].describe() = }")
        self.logger.debug(f"{_filter.sum()} benzodiazepine doses converted")
        self.logger.debug("Renaming benzodiazepines")
        data.loc[_filter, "target_name"] = "BENZO"
        return data

    def _convert_weight_based_dose(
        self, data: pd.DataFrame, convert_weight_based: dict
    ) -> pd.DataFrame:
        self.logger.debug("Converting weight based dose!")
        records_to_modify = data["dose_unit"].str.contains("/kg")
        if is_categorical_dtype(data["dose_unit"]):
            self.logger.debug("Dose unit is categorical")
            existing_categories = set(list(data["dose_unit"].cat.categories))
            self.logger.debug(f"{existing_categories = }")
            _renamed = set([x.replace("/kg", "") for x in existing_categories])
            self.logger.debug(f"{_renamed = }")
            new_categories = _renamed - existing_categories
            self.logger.debug(f"{new_categories = }")
            if new_categories:
                self.logger.debug("New categories found!")
                data["dose_unit"] = data["dose_unit"].cat.add_categories(new_categories)
                self.logger.debug(f"{data['dose_unit'].cat.categories = }")

        predicted_weight = convert_weight_based.get("predicted_weight")
        if predicted_weight:
            name_filter = data["target_name"].isin(predicted_weight)
            record_filter = records_to_modify & name_filter
            data.loc[record_filter, "dose"] *= (
                data["predicted_weight"].fillna(data["weight"]).fillna(80)
            )

            data.loc[record_filter, "dose_unit"] = data["dose_unit"].str.replace(
                "/kg", ""
            )
        adjusted_weight = convert_weight_based.get("adjusted_weight")
        if adjusted_weight:
            name_filter = data["target_name"].isin(adjusted_weight)
            record_filter = records_to_modify & name_filter
            data.loc[record_filter, "dose"] *= (
                data["adjusted_weight"].fillna(data["weight"]).fillna(80)
            )
            data.loc[record_filter, "dose_unit"] = data["dose_unit"].str.replace(
                "/kg", ""
            )
        records_to_modify = data["dose_unit"].str.contains("/kg", na=False)
        data.loc[records_to_modify, "dose"] *= data["weight"].fillna(80)
        data.loc[records_to_modify, "dose_unit"] = data["dose_unit"].str.replace(
            "/kg", ""
        )

        return data

    def _get_lab(self) -> pd.DataFrame:
        lab = self.get_data("lab")
        lab = lab.dropna().copy()
        ctx = self._cfg.data_loader.singles
        return self._post_processing(data=lab, ctx=ctx)

    def _load_range_timestamps(self) -> pd.DataFrame:
        dfs = dict()

        # medication: pid, start, stop, medication, rate
        dfs["urine_production"] = self._get_urine_production()

        range_timestamps = pd.concat(
            objs=dfs.values(),
            axis=0,
            ignore_index=True,
        )
        return range_timestamps

    def _load_single_timestamps(self) -> pd.DataFrame:
        ctx = self._cfg.data_loader.singles

        dfs = dict()

        dfs["hemodynamics"] = self._get_hemodynamics()
        dfs["respiratory"] = self._get_respiratory()
        dfs["neurology"] = self._get_neurology()
        dfs["temperature"] = self._get_temperature()
        dfs["lab"] = self._get_lab()

        self.logger.debug(f"Concatenating {len(dfs)} single timestamps")
        self.logger.debug(f"{dfs.keys() = }")
        for name, df in dfs.items():
            self.logger.debug(f"{name}: {df.shape} - {df.columns}")
        self.logger.debug(f"{sum([df.shape[0] for df in dfs.values()]) = }")

        single_timestamps = pd.concat(
            objs=dfs.values(),
            axis=0,
            ignore_index=True,
            copy=False,
        )

        self.logger.debug(f"{single_timestamps.shape = }")
        self.logger.debug(f"{single_timestamps.columns = }")
        self.logger.debug(f"{single_timestamps.dtypes = }")

        del dfs

        # retype columns
        retype = ctx.get("retype")
        if retype:
            self.logger.debug("Retyping columns")
            self.logger.debug(f"{retype = }")
            single_timestamps = single_timestamps.astype(retype)
            self.logger.debug(f"{single_timestamps.dtypes = }")

        single_timestamps = self._calculate_pao2_fio2_ratio(single_timestamps)

        return single_timestamps

    def _calculate_pao2_fio2_ratio(self, data: pd.DataFrame) -> pd.DataFrame:
        self.logger.debug("Calculating PaO2/FiO2 ratio")
        self.logger.debug(f"{data.shape = }")
        self.logger.debug(f"{data.columns = }")
        self.logger.debug(f"{data.dtypes = }")
        f_po2 = data.loc[
            (data["variable"] == "PO2__arterial") &
            (data["value"] > 10)
            ].copy()
        f_po2 = f_po2.rename(columns={
            "value": "pao2",
            "timestamp_start": "window_start",
            "timestamp_end": "window_end",
        })

        f_po2["window_end"] = f_po2["window_start"]
        f_po2["window_start"] = f_po2["window_end"] - pd.Timedelta(8, "h")
        f_po2['window_id'] = f_po2.reset_index().index

        f_fio2 = data.loc[
            (data["variable"] == "fio2") &
            (data["value"] > 0)].copy()
        f_fio2["timestamp_end"] = f_fio2["timestamp_start"]

        f_pf = td.dataprocessing.merge_windows(
            windows=f_po2,
            measurements=f_fio2,
            on=['pid'],
            windows_start='window_start',
            windows_end='window_end',
            measurements_start='timestamp_start',
            measurements_end='timestamp_end',
            group_by=['pid', 'variable', 'window_id'],
            variable_id='variable',
            value='value',
            # value_unit=(1, "h"),  # ml/h
            agg_func='last',
            map_columns=False,
        )

        f_pf["PO2__arterial"] = f_pf["PO2__arterial"].fillna(21)
        f_pf["value"] = (f_pf["pao2"] / f_pf["PO2__arterial"]) * 100
        f_pf["variable"] = "pao2_fio2_ratio"
        f_pf = f_pf.rename(columns={
            "window_start": "timestamp_start",
            "window_end": "timestamp_end",
        })
        f_pf = f_pf[data.columns].dropna().copy()
        self.logger.debug(f"{f_pf.shape = }")
        self.logger.debug(f"{f_pf.columns = }")
        self.logger.debug(f"{f_pf.dtypes = }")
        self.logger.debug(f"{f_pf['value'].describe() = }")
        return pd.concat([data, f_pf], axis=0, ignore_index=True, copy=False)

    def _post_processing(self, data: pd.DataFrame, ctx: dict) -> pd.DataFrame:
        self.logger.debug("Post processing data")
        self.logger.debug(f"{data.shape = }")
        self.logger.debug(f"{data.columns = }")
        self.logger.debug(f"{data.dtypes = }")

        self.logger.debug(f"{ctx = }")

        _rename = ctx.get("rename")
        if _rename:
            for column, rename_values in _rename.items():
                self.logger.debug(f"Renaming values in {column}")
                data[column] = data[column].map(rename_values).fillna(data[column])

        sort_by = ctx.get("sort_post_processing")
        if sort_by:
            self.logger.debug("Sorting data")
            self.logger.debug(f"{sort_by = }")
            data = data.sort_values(**sort_by)

        _single_to_range = ctx.get("transform_single_to_range")
        if _single_to_range:
            self.logger.debug("Transforming single to range")
            self.logger.debug(f"{_single_to_range = }")
            self.logger.debug(f"{data.shape = }")
            data = dp.single_to_range(
                data=data,
                **_single_to_range,
            )
            self.logger.debug(f"{data.shape = }")

        _add_timestamps = ctx.get("add_timestamps")
        if _add_timestamps:
            self.logger.debug("Adding timestamps")
            self.logger.debug(f"{_add_timestamps = }")
            self.logger.debug(f"{data.shape = }")
            data = self.add_timestamps(
                data=data,
                **_add_timestamps,
            )
            self.logger.debug(f"{data.shape = }")

        _overlap_to_discrete = ctx.get("transform_overlapping_to_discrete_series")
        if _overlap_to_discrete:
            self.logger.debug("Transforming overlapping to discrete series")
            self.logger.debug(f"{_overlap_to_discrete = }")
            self.logger.debug(f"{data.shape = }")
            data = dp.overlap_to_discrete(
                data=data,
                **_overlap_to_discrete,
            )
            self.logger.debug(f"{data.shape = }")

        _join_adjoining = ctx.get("join_adjoining_records")
        if _join_adjoining:
            self.logger.debug("Joining adjoining records")
            self.logger.debug(f"{_join_adjoining = }")
            self.logger.debug(f"{data.shape = }")
            data = dp.join_adjoining(
                data=data,
                **_join_adjoining,
            )
            self.logger.debug(f"{data.shape = }")

        return data

    def add_timestamps(
        self, data: pd.DataFrame, **kwargs: dict[Any, Any]
    ) -> pd.DataFrame:
        _rename = kwargs.get("rename")
        if _rename:
            data = data.rename(columns=_rename)

        _adjustments = kwargs.get("adjustments")
        if _adjustments:
            for target_column, settings in _adjustments.items():
                if target_column in data.columns:
                    self.logger.warning(
                        f"Column {target_column} already exists and "
                        f"will be overwritten"
                    )
                if settings["method"] == "add":
                    data[target_column] = data[settings["column"]] + settings["value"]
                elif settings["method"] == "subtract":
                    data[target_column] = data[settings["column"]] - settings["value"]
                else:
                    raise ValueError(f"Unknown method {settings['method']}")
        else:
            self.logger.debug("No adjustments found")
        return data

    def _get_urine_production(self) -> pd.DataFrame:
        urine = self.get_data("urine_production")

        if urine.shape[0] == 0:
            raise ValueError("No urine data loaded")

        # calculate urine output
        context = self._cfg.data_loader.__getattribute__("urine_production")
        self.logger.debug(f"{context = }")

        self.logger.debug("Transforming single to range")
        self.logger.debug(f"{urine.shape = }")
        self.logger.debug(f"{urine.columns = }")
        self.logger.debug(f"{urine.dtypes = }")
        urine_ranged = dp.single_to_range(
            data=urine, **context.get("transform_single_to_range")
        )
        self.logger.debug(f"{urine_ranged.shape = }")
        self.logger.debug(f"{urine_ranged.columns = }")
        self.logger.debug(f"{urine_ranged.dtypes = }")

        self.logger.debug("Calculating rate")
        urine_ranged["rate"] = -1 * self._calculate_rate(
            data=urine_ranged, **context.get("calculate_rate")
        )
        self.logger.debug(f"{urine_ranged.shape = }")
        self.logger.debug(f"{urine_ranged.dtypes = }")

        urine_imputed = dp.impute_missing(
            data=urine_ranged,
            windows=self.patients,
            **context.get("impute_missing_as_zero"),
        )

        return urine_imputed.rename(columns={"rate": "value"})

    def _calculate_rate(
        self,
        data: pd.DataFrame,
        value: str,
        start: str,
        stop: str,
        duration_unit: pd.Timedelta | int,
    ) -> pd.Series:
        """
        Calculate the rate of a value over a duration.
        :param data: pd.DataFrame containing value start and stop as columns
        :param value: string to specify a numerical column
        :param start: string to specify the start time
        :param stop: string to specify the end time
        :param duration_unit: number or timedelta to divide by (/hr or /min)
        :return:
        """
        self.logger.debug(f"Calculating rate of {value} over {duration_unit}")
        self.logger.debug(f"data: {data.shape}, {data.columns}")
        self.logger.debug(f"data dtypes: {data.dtypes}")
        self.logger.debug(f"start: {start}")
        self.logger.debug(f"stop: {stop}")
        self.logger.debug(f"value: {value}")
        self.logger.debug(f"duration_unit: {duration_unit}")
        return data.apply(
            lambda row: (row[value] / ((row[stop] - row[start]) / duration_unit)),
            axis=1,
        )

    def _get_neurology(self) -> pd.DataFrame:
        gcs = self.get_data("neurology")

        ctx = self._cfg.data_loader.singles
        gcs = self._post_processing(data=gcs, ctx=ctx)

        return gcs

    def _get_temperature(self) -> pd.DataFrame:
        temperature = self.get_data("temperature")

        ctx = self._cfg.data_loader.singles
        temperature = self._post_processing(data=temperature, ctx=ctx)

        return temperature

    def _get_respiratory(self) -> pd.DataFrame:
        ventilator = self.get_data("respiratory_ventilator")
        default = self.get_data("respiratory_default")
        respiratory = pd.DataFrame()

        dl = self._cfg.data_loader.__getattribute__("respiratory_merge")
        if dl:
            merge_on = dl.get("on")
            if merge_on:
                respiratory = pd.merge(
                    left=ventilator,
                    right=default,
                    on=merge_on,
                )

                combine_columns = dl.get("combine_columns")
                if combine_columns:
                    respiratory = self._combine_columns_iterative(
                        data=respiratory, combine_columns=combine_columns
                    )
                else:
                    self.logger.debug("No columns to combine")

                melt = dl.get("melt")
                if melt:
                    respiratory = self._melt(data=respiratory, melt=melt)

                    drop_nan_after_melt = dl.get("drop_nan_after_melt")
                    if drop_nan_after_melt:
                        respiratory = respiratory.dropna(how="any", subset="value")
                else:
                    self.logger.debug("No melt specified")
            else:
                self.logger.critical("No columns to merge on")
        else:
            self.logger.critical("No merge settings specified")

        ctx = self._cfg.data_loader.singles
        respiratory = self._post_processing(data=respiratory, ctx=ctx)

        return respiratory

    def _get_hemodynamics(self) -> pd.DataFrame:
        hemodynamics = self.get_data("hemodynamics")

        ctx = self._cfg.data_loader.singles
        hemodynamics = self._post_processing(data=hemodynamics, ctx=ctx)

        return hemodynamics

    def _combine_columns(
        self, data: pd.DataFrame, final_column: str | float | int, settings: dict
    ) -> pd.DataFrame:
        self.logger.debug(f"Combining columns to {final_column}")
        method: str | None = settings.get("method")
        columns: list | None = settings.get("columns")
        self.logger.debug(f"Method: {method}")
        self.logger.debug(f"Columns: {columns}")
        if method:
            if columns:
                if method == "fillna":
                    data[final_column] = pd.NA
                    for column in columns:
                        data[final_column] = data[final_column].fillna(data[column])
                elif method == "bool_and":
                    data[final_column] = (
                        data[columns]
                        .fillna(False)
                        .all(
                            axis=1,
                            bool_only=True,
                            skipna=True,
                        )
                    )
                elif method == "bool_or":
                    data[final_column] = (
                        data[columns]
                        .fillna(False)
                        .any(
                            axis=1,
                            bool_only=True,
                            skipna=True,
                        )
                    )
                elif method == "bool_ohe":
                    data[final_column] = pd.NA
                    data.loc[data[columns[0]].fillna(False), final_column] = data[columns[1]]
                else:
                    data[final_column] = data[columns].agg(method, axis=1)
                self.logger.debug(f"Columns to drop: {columns}")
                self.logger.debug(f"Starting shape: {data.shape}")
                _drop = settings.get("drop", True)
                if _drop:
                    data = data.drop(columns=columns)
                self.logger.debug(f"Remaining shape: {data.shape}")
            else:
                self.logger.critical("No columns specified")
        else:
            self.logger.critical("No method specified")
        return data

    def _to_long_format(self, data: pd.DataFrame, id_vars: list[str]) -> pd.DataFrame:
        """
        Transform the data to long format.

        The data is transformed to long format. The columns are split into a
        measurement column and a value column. The resulting table is returned.

        Args:
            data (pd.DataFrame): The data to transform.

        Returns:
            pd.DataFrame: The transformed table.
        """

        self.logger.debug("Transforming data to long format")

        _columns = set(data.columns)
        _id_vars = set(id_vars)

        if _id_vars.issubset(_columns):
            _variables = _columns - _id_vars
            self.logger.debug(f"Melting variables: {_variables}")

            data = data.melt(
                id_vars=id_vars,
            )
        else:
            self.logger.error(f"{_id_vars} not in {_columns}")

        self.logger.debug(f"Data shape after: {data.shape}")

        return data

    def get_data(self, name: str) -> pd.DataFrame:
        """Load data from raw data directory and apply transformations.

        The get_data() method loads data from the raw data directory. It applies
        transformations as specified in the config file. The resulting table is
        returned.

        Data columns are renamed and retyped, NaN values are dropped for specified
        columns and rows are filtered based on specified values. Finally, the data is
        merged with a table containing the columns to keep.

        filename: The filename of the data to load.
        rename: A dictionary with the columns to rename as keys and the new names as
                values.
        drop_nan_any: A list of columns to drop rows with any NaN values.
        drop_nan_all: A list of columns to drop rows with all NaN values.
        retype: A dictionary with the columns to retype as keys and the new datatypes
                as values.
        keep: A dictionary with the columns to filter as keys and the values to keep as
                values.
        merge_keep: A dictionary specifying a table to merge with the data.
            filename: The filename of the table to merge.
            rename: A dictionary with the columns to rename as keys and the new names
                    as values.
            keep: The column to filter the table on. Either boolean or int.
            on: The column(s) to merge on.
            name: The column to keep from the table to merge.

        Args:
            name (str): Name of the data to load. This name should be specified in the
            config file.

        Returns:
            pd.DataFrame: The resulting table.
        """

        settings = self.config.settings.measurements.data_loader
        context = settings.__getattribute__(name)

        data = self._load_data(context, name)

        self.logger.debug(data.notna().mean())

        rename_columns = context.get("rename")
        if rename_columns:
            data = self._rename_columns(data, rename_columns)
        else:
            self.logger.debug("No columns to rename")
        self.logger.debug(data.notna().mean())

        drop_nan_any = context.get("drop_nan_any")
        if drop_nan_any:
            data = self._drop_nan_any(data, drop_nan_any)
        else:
            self.logger.debug("No columns defined to filter NaN values on (any)")
        self.logger.debug(data.notna().mean())

        drop_nan_all = context.get("drop_nan_all")
        if drop_nan_all:
            data = self._drop_nan_all(data, drop_nan_all)
        else:
            self.logger.debug("No columns defined to filter NaN values on (all)")
        self.logger.debug(data.notna().mean())

        drop_pids = True
        if drop_pids:
            data = data.loc[data["pid"].isin(self.patient_ids)]
        self.logger.debug(data.notna().mean())

        remap = context.get("remap")
        if remap:
            data = self._remap(data, remap)
        self.logger.debug(data.notna().mean())

        encode_columns = context.get("encode_columns")
        if encode_columns:
            data = self._encode_columns(data, encode_columns)
        self.logger.debug(data.notna().mean())

        retype_columns = context.get("retype")
        if retype_columns:
            data = self._retype_columns(data=data, retype_columns=retype_columns)
        self.logger.debug(data.notna().mean())

        sorting = context.get("sort")
        if sorting:
            data = data.sort_values(**sorting)
        self.logger.debug(data.notna().mean())

        duplicates = context.get("duplicates")
        if duplicates:
            data = data.drop_duplicates(**duplicates)
        self.logger.debug(data.notna().mean())

        set_timestamp_column = context.get("set_timestamp_column")
        if set_timestamp_column:
            data = self._set_timestamp_column(data, set_timestamp_column)
        else:
            self.logger.debug("No timestamp column specified")
        self.logger.debug(data.notna().mean())

        keep_columns = context.get("keep")
        if keep_columns:
            data = self._keep_columns(data=data, keep_columns=keep_columns)
        else:
            self.logger.debug("No columns to filter data on")
        self.logger.debug(data.notna().mean())

        merge_keep = context.get("merge_keep")
        if merge_keep:
            data = self._merge_keep(data=data, merge_keep=merge_keep)
        else:
            self.logger.debug("No columns to merge")
        self.logger.debug(data.notna().mean())

        adjust_variables = context.get("adjust_variables")
        if adjust_variables:
            data = self._adjust_variables(data, adjust_variables)
        else:
            self.logger.debug("No variables to adjust")
        self.logger.debug(data.notna().mean())

        split_columns = context.get("split_columns")
        if split_columns:
            data = self._split_columns(data, split_columns)
        else:
            self.logger.debug("No columns to split")
        self.logger.debug(data.notna().mean())

        filter_columns = context.get("filter_columns")
        if filter_columns:
            # filter on values within a single column
            data = self._filter_columns(data, filter_columns)
        else:
            self.logger.debug("No columns to filter")
        self.logger.debug(data.notna().mean())

        filter_values = context.get("filter_values")
        if filter_values:
            # filter on values within one column, based on values in other columns
            data = self._filter_values(data, filter_values)
        else:
            self.logger.debug("No values to filter")
        self.logger.debug(data.notna().mean())

        unit_conversion = context.get("unit_conversion")
        if unit_conversion:
            data = self._unit_conversion(data, unit_conversion)
        else:
            self.logger.debug("No unit conversion to perform")
        self.logger.debug(data.notna().mean())

        combine_columns = context.get("combine_columns")
        if combine_columns:
            data = self._combine_columns_iterative(data, combine_columns)
        else:
            self.logger.debug("No columns to combine")
        self.logger.debug(data.notna().mean())
        
        filter_on_combined_columns = context.get("filter_on_combined_columns")
        if filter_on_combined_columns:
            data = self._filter_on_combined_columns(data, filter_on_combined_columns)
        else:
            self.logger.debug("No combined columns to filter on")
        self.logger.debug(data.notna().mean())

        melt = context.get("melt")
        if melt:
            data = self._melt(data=data, melt=melt)

            drop_nan_after_melt = context.get("drop_nan_after_melt")
            if drop_nan_after_melt:
                data = data.dropna(how="any", subset=["value"])
        self.logger.debug(data.notna().mean())

        self.logger.debug(f"Finished loading data with shape {data.shape}")
        self.logger.debug(data.notna().mean())

        _return = context.get("return")
        if _return:
            data = self._return(data, _return)

        return data

    def _filter_values(self, data: pd.DataFrame, filter_values: dict) -> pd.DataFrame:
        mappers: dict = filter_values.get("mappers")  # type: ignore[assignment]
        value_column = filter_values.get("value_column")

        _values_to_drop = pd.Series(False, index=data.index)
        for column, values in mappers.items():
            _min = {k: v.get("min") for k, v in values.items() if v.get("min")}
            _max = {k: v.get("max") for k, v in values.items() if v.get("max")}

            _below_min = data[value_column] < data[column].map(_min)
            _above_max = data[value_column] > data[column].map(_max)

            _values_to_drop |= _below_min | _above_max

        return data[~_values_to_drop].copy()

    def _unit_conversion(
        self, data: pd.DataFrame, unit_conversion: dict
    ) -> pd.DataFrame:
        column_unit = unit_conversion.get("column_unit")
        column_value = unit_conversion.get("column_value")
        factors = unit_conversion.get("factors")
        rename_units = unit_conversion.get("rename_units")

        if column_unit and column_value and factors:
            data[column_value] = (
                data[column_value] * data[column_unit].map(factors)
            ).fillna(data[column_value])
            if rename_units:
                data[column_unit] = (
                    data[column_unit].map(rename_units).fillna(data[column_unit])
                )
        return data

    def _return(self, data: pd.DataFrame, settings: dict) -> pd.DataFrame:
        self.logger.debug("Modifying returning data")
        self.logger.debug(f"{settings = }")
        self.logger.debug(f"{data.shape = }")
        _rename = settings.get("rename")
        if _rename:
            self.logger.debug("Renaming data")
            data = data.rename(columns=_rename)
            self.logger.debug("Data renamed")
        _columns = settings.get("columns")
        if _columns:
            self.logger.debug(f"Filtering on columns = {_columns}")
            data = data[_columns]
        self.logger.debug(f"{data.shape = }")
        return data

    def _adjust_variables(
        self, data: pd.DataFrame, adjust_variables: dict
    ) -> pd.DataFrame:
        for method, settings in adjust_variables.items():
            if method == "concat":
                data = self._concat_variables(data, settings)
            else:
                _msg = f"Unknown method {method} to adjust variables"
                self.logger.critical(_msg)
                raise ValueError(_msg)
        return data

    def _concat_variables(self, data: pd.DataFrame, settings: dict) -> pd.DataFrame:
        self.logger.debug("Concatenating variables")
        self.logger.debug(f"{settings = }")
        self.logger.debug(f"{data.shape = }")
        self.logger.debug(f"{data.columns = }")
        self.logger.debug(f"{data.dtypes = }")

        _columns = settings.get("columns")
        if not _columns:
            _msg = "No columns specified to concatenate"
            self.logger.critical(_msg)
            raise ValueError(_msg)

        _target_column = settings.get("target_column")
        if not _target_column:
            _msg = "No target_column specified"
            self.logger.critical(_msg)
            raise ValueError(_msg)

        _sep = settings.get("sep")
        if not _sep:
            _msg = "No separator specified"
            self.logger.critical(_msg)
            raise ValueError(_msg)

        _filters = settings.get("filters")
        if _filters:
            _data_filter = pd.Series(False, index=data.index)
            for _filter in _filters:
                _inner_filter = pd.Series(True, index=data.index)
                for column, values in _filter.items():
                    _inner_filter &= data[column].isin(values)
                _data_filter |= _inner_filter
        else:
            _data_filter = pd.Series(True, index=data.index)

        _values = data[_columns].astype(str).agg(_sep.join, axis=1)
        if _target_column in data.columns:
            if is_categorical_dtype(data[_target_column]):
                _old_cats = set(data[_target_column].cat.categories)
                _new_cats = set(_values.loc[_data_filter].unique())
                _new_cats -= _old_cats
                if _new_cats:
                    self.logger.debug(f"Adding new categories {_new_cats}")
                    data[_target_column] = data[_target_column].cat.add_categories(
                        _new_cats
                    )
        else:
            _msg = f"Creating new column {_target_column}"
            self.logger.warning(_msg)
            warnings.warn(_msg)
            data[_target_column] = pd.NA

        data.loc[_data_filter, _target_column] = _values
        self.logger.debug(f"{data.shape = }")
        self.logger.debug(f"{data.columns = }")
        self.logger.debug(f"{data.dtypes = }")
        return data

    def _set_timestamp_column(
        self, data: pd.DataFrame, set_timestamp_column: dict
    ) -> pd.DataFrame:
        self.logger.debug(f"{set_timestamp_column = }")
        method = set_timestamp_column.get("method")
        if not method:
            error_msg = "No method specified to set timestamp column"
            self.logger.critical(error_msg)
            raise ValueError(error_msg)

        columns = set_timestamp_column.get("columns")
        if not columns:
            error_msg = "No columns specified to set timestamp column"
            self.logger.critical(error_msg)
            raise ValueError(error_msg)

        return_column_name = set_timestamp_column.get("return_column_name")
        if not return_column_name:
            _msg = "No return column name specified, using 'timestamp'"
            self.logger.warning(_msg)
            warnings.warn(_msg)
            return_column_name = "timestamp"

        _data = data[columns].copy()

        adjustments = set_timestamp_column.get("adjustments")
        if adjustments:
            _msgs: list[str] = list()
            for col, adjustment in adjustments.items():
                if col not in columns:
                    _msgs.append(
                        f"Adjustment requested for column {col} "
                        "but is not present in the evaluated columns"
                        f" {columns}"
                    )
                    self.logger.critical(_msgs[-1])
                    continue
                _data[col] += adjustment
            if _msgs:
                raise ValueError("\n+++\n".join(_msgs))

        if method == "fillna":
            _timestamps = pd.Series(pd.NA, index=data.index)
            for column in columns:
                _timestamps = _timestamps.fillna(_data[column])
        elif method == "max":
            _timestamps = _data[columns].max(axis=1)
        elif method == "min":
            _timestamps = _data[columns].min(axis=1)
        else:
            error_msg = f"Unknown method {method} to set timestamp column"
            self.logger.critical(error_msg)
            raise ValueError(error_msg)

        self.logger.debug(f"Setting timestamp column to {set_timestamp_column}")
        data[return_column_name] = _timestamps
        return data

    def _encode_columns(self, data: pd.DataFrame, encode_columns: dict) -> pd.DataFrame:
        self.logger.debug("Encoding columns")
        self.logger.debug(f"Data shape before: {data.shape}")
        self.logger.debug(f"{encode_columns = }")

        for column, encoding in encode_columns.items():
            self.logger.debug(f"Encoding column {column}")
            self.logger.debug(f"Encoding values {encoding}")
            method = encoding.get("method")
            if method == "map":
                self.logger.debug(f"Mapping values in column {column}")
                _mapper = encoding.get("values")
                if _mapper:
                    self.logger.debug(f"Mapping values {_mapper}")
                    data[column] = data[column].map(_mapper).astype(float)
                else:
                    self.logger.critical(f"No values specified for {column}")
        self.logger.debug(f"Data shape after: {data.shape}")
        return data

    def _combine_columns_iterative(
        self, data: pd.DataFrame, combine_columns: dict
    ) -> pd.DataFrame:
        self.logger.debug("Combining columns")
        for final_column, settings in combine_columns.items():
            data = self._combine_columns(
                data=data, final_column=final_column, settings=settings
            )
        return data

    def _filter_columns(self, data: pd.DataFrame, filter_columns: dict) -> pd.DataFrame:
        for column, settings in filter_columns.items():
            self.logger.debug(f"Filtering column {column}")
            self.logger.debug(f"Filtering values {settings}")
            self.logger.debug(f"Data shape before: {data.shape}")
            _min = settings.get("min")
            _max = settings.get("max")

            self.logger.debug(f"Min: {_min}")
            if _min:
                self.logger.debug(f"Shape: {data.shape}")
                data.loc[data[column] < _min, column] = pd.NA
                self.logger.debug(f"Shape: {data.shape}")

            self.logger.debug(f"Max: {_max}")
            if _max:
                self.logger.debug(f"Shape: {data.shape}")
                data.loc[data[column] > _max, column] = pd.NA
                self.logger.debug(f"Shape: {data.shape}")
        return data

    def _split_columns(self, data: pd.DataFrame, split_columns: dict) -> pd.DataFrame:
        self.logger.debug("Splitting columns")
        for column, new_columns in split_columns.items():
            self.logger.debug(f"Splitting column {column} into {new_columns}")
            new_columns = list(new_columns)
            data[new_columns] = data[column].str.split("/", expand=True)
            self.logger.debug(f"Converting columns to float: {new_columns}")
            data[new_columns] = data[new_columns].astype(float)
            self.logger.debug(f"Dropping column {column}")
            data = data.drop(columns=column)
        return data

    def _load_data(self, context: dict, name: str) -> pd.DataFrame:
        self.logger.debug(f"Loading {name} data")
        filename: str = context.get("filename")  # type: ignore[assignment]
        if filename:
            path = context.get("path", "raw")
            filepath = os.path.join(self.config.directory(path), filename)
            self.logger.debug(f"Loading {name} data from {filepath}")
            data: pd.DataFrame = td.load(filepath)
            self.logger.debug(f"Loaded {name} data with shape {data.shape}")
            self.logger.debug(
                "Memory usage: " f"{data.memory_usage(deep=True).sum() / 1024 ** 2} MB"
            )
            return data
        else:
            self.logger.critical(f"No filename specified for {name} data")
            raise ValueError(f"No filename specified for {name} data")

    def _rename_columns(self, data: pd.DataFrame, rename_columns: dict) -> pd.DataFrame:
        self.logger.debug(f"Renaming columns to {rename_columns}")
        data = data.rename(columns=rename_columns)
        self.logger.debug(f"Keeping columns {rename_columns.values()}")
        data = data[rename_columns.values()].copy()
        return data

    def _drop_nan_any(self, data: pd.DataFrame, drop_nan_any: list) -> pd.DataFrame:
        self.logger.debug("Dropping rows with any NaN values")
        self.logger.debug(f"Data shape before: {data.shape}")
        data = data.dropna(axis=0, how="any", subset=drop_nan_any)
        self.logger.debug(f"Data shape after: {data.shape}")
        return data

    def _drop_nan_all(self, data: pd.DataFrame, drop_nan_all: list) -> pd.DataFrame:
        self.logger.debug(f"Dropping rows with all NaN values over {drop_nan_all}")
        self.logger.debug(f"Data shape before: {data.shape}")
        data = data.dropna(axis=0, how="all", subset=drop_nan_all)
        self.logger.debug(f"Data shape after: {data.shape}")
        return data

    def _remap(self, data: pd.DataFrame, remap: dict) -> pd.DataFrame:
        for column in remap:
            self.logger.debug(f"Remapping values in column {column}")
            data[column] = data[column].map(remap[column]).fillna(data[column])
            self.logger.debug(f"Dropping rows with 'drop' value in {column}")
            self.logger.debug(f"Data shape before: {data.shape}")
            data = data.loc[data[column] != "drop"].copy()
            self.logger.debug(f"Data shape after: {data.shape}")
        return data

    def _retype_columns(self, data: pd.DataFrame, retype_columns: dict) -> pd.DataFrame:
        self.logger.debug("Converting columns to dataypes")
        self.logger.debug(
            "Memory usage before: "
            f"{data.memory_usage(deep=True).sum() / 1024 ** 2} MB"
        )
        data = data.astype(retype_columns)
        self.logger.debug(
            "Memory usage after: "
            f"{data.memory_usage(deep=True).sum() / 1024 ** 2} MB"
        )
        return data

    def _keep_columns(self, data: pd.DataFrame, keep_columns: dict) -> pd.DataFrame:
        self.logger.debug(f"Columns to filter data: {keep_columns.keys()}")
        for column in keep_columns:
            if column in data.columns:
                self.logger.debug(f"Keeping column {column}")
                data = data.loc[data[column].isin(keep_columns[column])]
            else:
                self.logger.warning(f"Column {column} not in data")
        return data

    def _merge_keep(self, data: pd.DataFrame, merge_keep: dict) -> pd.DataFrame:
        filename = merge_keep.get("filename")
        if filename:
            filepath = os.path.join(self.config.directory("definitions"), filename)
            self.logger.debug(f"Loading {filename} data from {filepath}")
            merge_data = pd.read_csv(filepath)

            self.logger.debug(
                f"Loaded {filename} data with " f"shape {merge_data.shape}"
            )

            merge_rename = merge_keep.get("rename")
            if merge_rename:
                merge_data = merge_data.rename(columns=merge_rename)

            keep_column = merge_keep.get("keep")
            if merge_data[keep_column].dtype == "object":
                merge_data[keep_column] = merge_data[keep_column].astype(int)

            if is_numeric_dtype(merge_data[keep_column]):
                merge_data = merge_data.loc[merge_data[keep_column] == 1].copy()
            elif merge_data[keep_column].dtype == "bool":
                merge_data = merge_data.loc[merge_data[keep_column]].copy()
            self.logger.debug(f"Keeping rows with {keep_column} == 1 or True")
            self.logger.debug(f"merge_data shape after: {merge_data.shape}")

            merge_on = merge_keep.get("on")
            if merge_on:
                self.logger.debug(f"Merging on {merge_on}")
                if not isinstance(merge_on, set):
                    if isinstance(merge_on, str):
                        merge_on = {merge_on}
                    else:
                        merge_on = set(merge_on)

                __add_columns: list = merge_keep.get(
                    "add_columns"
                )  # type: ignore[assignment]
                if __add_columns:
                    columns_to_add = set(__add_columns)
                    merge_columns = list(merge_on | columns_to_add)
                else:
                    merge_columns = list(merge_on)

                merge_data = merge_data[merge_columns]

                data = pd.merge(
                    left=data,
                    right=merge_data,
                    on=list(merge_on),
                    how="inner",
                )
            else:
                self.logger.critical("No columns to merge on")

            self.logger.debug(f"Data shape after: {data.shape}")
        else:
            self.logger.warning(f"No filename specified for {merge_keep}")
        return data

    def _melt(self, data: pd.DataFrame, melt: dict) -> pd.DataFrame:
        self.logger.debug("Transforming data to long format")
        id_vars = melt.get("id_vars")
        self.logger.debug(f"Data shape before: {data.shape}")
        if id_vars:
            subset = melt.get("subset")
            if subset:
                self.logger.debug(f"Subsetting data to {id_vars + subset}")
                data = data[id_vars + subset]
            data = self._to_long_format(data=data, id_vars=id_vars)
        else:
            self.logger.critical("No id_vars specified while melt is specified")
        self.logger.debug(f"Data shape after: {data.shape}")
        return data

    def _filter_on_combined_columns(self, data, filter_on_combined_columns):
        self.logger.debug("Filtering on combined columns")
        print("Filtering on combined columns")
        self.logger.debug(data.shape)
        print(data.shape)
        for method, columns in filter_on_combined_columns.items():
            if method == "any":
                data = data.loc[data[columns].any(axis=1)].copy()
            elif method == "all":
                data = data.loc[data[columns].all(axis=1)].copy()
            elif method == "none":
                data = data.loc[~data[columns].any(axis=1)].copy()
        print(data.shape)
        return data


def custom_round(col: pd.Series, rounding_vals: list[float]) -> pd.Series:
    rounding_logic = pd.Series(rounding_vals)
    return pd.cut(x=col, bins=rounding_logic)


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

    mp = MeasurementsProcessor(config=_config)
    dfs = mp.run()
    print("Done")
