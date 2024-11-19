import logging
import os

import duckdb as duckdb
import numpy as np
import pandas as pd
import tadam as td  # Personal repo

from abstop.config import Config

logger = logging.getLogger(__name__)


class AntibioticsPreprocessor:
    """
    Constructs antibiotics data for the experiment.

    The AntibioticsPreprocessor class is initialized with a Config object. This object
    contains the settings for the experiment. The AntibioticsPreprocessor class loads
    the tables from the raw data directory.

    Using settings from the config class, a distinction is made between most-likely
    therapeutic and most-likely prophylactic antibiotics.
    """

    def __init__(self, config: Config):
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.logger.debug(f"from {__name__} instantiate {self.__class__.__name__}")
        self.config = config
        self._cfg = self.config.settings.antibiotic_selection

    def run(self) -> None:
        self.logger.debug(f"Running {self.__class__.__name__}")

        # Load patients
        patients = self.load_patients()
        patient_ids = patients["pid"].unique()

        # Load medication table
        medication_filepath = os.path.join(
            self.config.directory("raw"),
            self._cfg.ab_raw_filename,
        )
        medication = self._get_medication_table(
            path=medication_filepath,
            ids=patient_ids,
        )

        self.logger.debug(medication.shape)

        # Construct atc series
        medication = self._create_atc_series(medication=medication)

        medication = self._handle_tdm_records(medication=medication, ids=patient_ids)
        self.logger.info(medication.shape)
        self.logger.info(medication.columns)
        self.logger.info(f"\n{medication.head()}")

        # Remove single dose antibiotics
        medication = self.remove_single_records_within_groups(
            medication=medication,
            group_label="tdm_series_id",
        )

        # Label prophylactics and SDD series
        ## ATC, duration, timing relative to SDD cultures/medication
        medication = self._label_prophylaxis(medication=medication)

        # Construct Super Series of overlapping series
        medication = self._construct_super_series(medication=medication)

        # Save to processed data directory
        self.save(medication=medication)

    def save(self, medication: pd.DataFrame) -> None:
        medication = self.retype(
            medication=medication, types=self._cfg.ab_column_final_type_dict
        )

        save_path = os.path.join(
            self.config.directory("processed"),
            self._cfg.ab_processed_filename,
        )

        td.dump(obj=medication, path=save_path, hash_type=["sha512"])

    def retype(self, medication: pd.DataFrame, types: dict) -> pd.DataFrame:
        if self.logger.level == logging.DEBUG:
            memory_usage = medication.memory_usage(deep=True).sum() / (1024**2)
            self.logger.debug(f"Memory usage before reducing: {memory_usage:.2f} MB")

        retype_dict = {k: v for k, v in types.items() if k in medication.columns}

        medication = medication.astype(retype_dict)

        if self.logger.level == logging.DEBUG:
            memory_usage = medication.memory_usage(deep=True).sum() / (1024**2)
            self.logger.debug(f"Memory usage after reducing: {memory_usage:.2f} MB")

        return medication

    def _construct_super_series(self, medication: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Split medication into prophylaxis and therapeutic")
        med_prophylaxis = medication.loc[medication.is_prophylaxis]
        med_therapeutic = medication.loc[~medication.is_prophylaxis]
        self.logger.debug(f"{med_prophylaxis.shape[0] = }")
        self.logger.debug(f"{med_therapeutic.shape[0] = }")

        med_prophylaxis.groupby(["name"], dropna=False)["original_name"].value_counts()
        med_therapeutic["name"].value_counts()

        """
        Creating super series ids for therapeutic records.

        When handling TDM records, these series are joined if they are within 52.8h 
        (48+10%). However, when creating super series, we also allow for a period of 
        26.4h difference, therefore, we only adjust the TDM timestamp for 26.4 hours as 
        this will result in a maximal look-ahead of 52.8h.

        As the super series ids are created based on the min and max of the tdm_series 
        (atc grouped), adjusting the timestamps will not result in accidentally breaking
         up these series when creating super series ids.
        """

        # TDM is joined if next record occurs within 52.8. Therefore, we adjust the
        # timestamps for therapeutic records to allow for a 26.4h difference
        medication.loc[
            (medication["tdm_flag"] == 1) & medication["is_last_in_atc_group"],
            "timestamp_tdm_adjusted",
        ] = medication["timestamp"] + pd.Timedelta(26.4, "h")
        medication["timestamp_tdm_adjusted"] = medication[
            "timestamp_tdm_adjusted"
        ].fillna(medication["timestamp"])

        # For therapeutic records, construct super series based on overlapping ids
        groups = (
            medication.loc[~medication.is_prophylaxis]
            .groupby(["pid", "tdm_series_id"])
            .agg({"timestamp_tdm_adjusted": ["min", "max"]})
            .reset_index()
        )
        groups.columns = ["pid", "tdm_series_id", "start", "end"]
        groups["end"] = groups["end"] + pd.Timedelta(26.4, "h")

        # label series within pids with overlapping start and end timestamps
        a = groups.melt(
            id_vars=["pid", "tdm_series_id"],
            value_vars=["start", "end"],
            var_name="type",
            value_name="timestamp",
        )
        a["value"] = a["type"].map({"start": 1, "end": -1})
        a.sort_values(["pid", "timestamp"], inplace=True)
        a["running"] = a.groupby(["pid"])["value"].cumsum()
        a["newwin"] = a["running"].eq(1) & a["value"].eq(1)
        a["group"] = a["newwin"].cumsum()

        # map series to group numbers
        medication["ab_series_id"] = medication["tdm_series_id"].map(
            a[["tdm_series_id", "group"]]
            .drop_duplicates()
            .set_index("tdm_series_id")["group"]
            .to_dict()
        )
        # make sure only one pid per series
        self.assert_max_pids_per_groups(medication)

        # For prophylaxis records, construct super series based on overlapping ids
        groups = (
            medication.loc[medication.is_prophylaxis]
            .groupby(["pid", "tdm_series_id"])
            .agg({"timestamp_tdm_adjusted": ["min", "max"]})
            .reset_index()
        )
        groups.columns = ["pid", "tdm_series_id", "start", "end"]
        groups["end"] = groups["end"] + pd.Timedelta(26.4, "h")

        # label series within pids with overlapping start and end timestamps
        a = groups.melt(
            id_vars=["pid", "tdm_series_id"],
            value_vars=["start", "end"],
            var_name="type",
            value_name="timestamp",
        )
        a["value"] = a["type"].map({"start": 1, "end": -1})
        a.sort_values(["pid", "timestamp"], inplace=True)
        a["running"] = a.groupby(["pid"])["value"].cumsum()
        a["newwin"] = a["running"].eq(1) & a["value"].eq(1)
        a["group"] = a["newwin"].cumsum()
        a["group"] = a["group"] + medication["ab_series_id"].max()

        # map series to group numbers
        medication["ab_series_id"] = medication["ab_series_id"].fillna(
            medication["tdm_series_id"].map(
                a[["tdm_series_id", "group"]]
                .drop_duplicates()
                .set_index("tdm_series_id")["group"]
                .to_dict()
            )
        )

        self.assert_max_pids_per_groups(medication)

        list_of_ids = set(medication.ab_series_id.unique())
        list_of_ints = set(range(int(medication.ab_series_id.max()) + 1))
        output = [i for i in list_of_ints if i not in list_of_ids]
        self.logger.debug(f"ints not in ids: {output}")
        assert len(output) == 1, f"ints not in ids: {output}"
        assert output[0] == 0, f"ints not in ids: {output}"

        return medication

    def assert_max_pids_per_groups(self, medication: pd.DataFrame) -> None:
        # make sure only one pid per series
        max_n_pids = medication.groupby(["ab_series_id"])["pid"].nunique().max()
        self.logger.debug(f"{max_n_pids = }")
        assert max_n_pids == 1, f"max_n_pids = {max_n_pids}"

        msg = (
            "Number of old groups:"
            f"{medication.tdm_series_id.nunique()}"
            " Number of new groups:"
            f"{medication.ab_series_id.nunique()}"
        )
        self.logger.info(msg)

    def _label_prophylaxis(self, medication: pd.DataFrame) -> pd.DataFrame:
        """
        Label prophylactic antibiotics based on ATC, duration and timing relative to
        SDD cultures/medication. The prophylactic antibiotics are labeled as

        :param medication:
        :return:
        """
        # Prophylaxis based on ATC, duration and timing relative to SDD cultures
        medication = self._label_prophylaxis_based_on_atc_duration_timing(
            medication=medication,
        )

        # Prophylaxis based on ATC, duration without timing relative to SDD
        medication = self._label_prophylaxis_based_on_atc_without_timing(
            medication=medication,
        )

        return medication

    def _label_prophylaxis_based_on_atc_without_timing(
        self,
        medication: pd.DataFrame,
    ) -> pd.DataFrame:
        # kefzol
        medication = self._label_prophylaxis_cefazolin(medication=medication)
        # cotrim
        medication = self._label_prophylaxis_cotrimoxazole(medication=medication)
        # ciprofloxacin
        medication = self._label_prophylaxis_ciprofloxacin(medication=medication)

        return medication

    def _label_prophylaxis_ciprofloxacin(
        self, medication: pd.DataFrame
    ) -> pd.DataFrame:
        # Prophylaxis time dependent:
        # - Ciprofloxacin (J01MA02) in Hematologie, if given after any PDD culture
        # - Retrieve all PDD cultures
        # - Retrieve all Ciprofloxacin medication records

        # And we also select the timestamps for the collection of PDD cultures from the
        # microbiology lab.
        mic = td.load(
            os.path.join(
                self.config.directory("processed"),
                self._cfg.mic_processed_filename,
            ),
            hash_type=["sha512"],
        )

        mic_pdd = (
            mic.loc[mic.is_pdd][["pid", "sample_date"]]
            .drop_duplicates()
            .set_index("pid")
        )
        mic_pdd.columns = ["timestamp"]
        mic_pdd["timestamp"] = pd.to_datetime(mic_pdd["timestamp"]) + pd.Timedelta(
            12, "h"
        )

        mic_pdd["new_sdd"] = (
            (mic_pdd.groupby(["pid"]).timestamp.diff() > pd.Timedelta(26.4, "h"))
            | mic_pdd.groupby(["pid"]).timestamp.diff().isna()
        ).astype(int)
        mic_pdd["pdd_series_id"] = mic_pdd["new_sdd"].cumsum()
        mic_pdd_series = mic_pdd.groupby(["pid", "pdd_series_id"])["timestamp"].agg(
            ["min", "max", "count"]
        )

        # Next, we create PDD series based on the medication and culture records, so we
        # can identify the first occurrence of PDD
        med_pdd_series = (
            medication.loc[(medication["atc"] == "J01MA02")]  # Ciprofloxacin
            .groupby(["pid", "tdm_series_id"])["timestamp"]
            .agg(["min", "max", "count"])
        )
        med_pdd_series["duration"] = med_pdd_series["max"] - med_pdd_series["min"]
        print(
            "Medication potentially PDD series duration:\n",
            med_pdd_series.duration.describe(),
        )

        # Merge the PDD medication/cultures with the medication records which may fall
        # surrounding/within the PDD periods
        med_rec_pdd = pd.merge(
            med_pdd_series.reset_index(),
            mic_pdd_series.reset_index(),
            on=["pid"],
            how="left",
            suffixes=("_med", "_rec"),
        )

        # Filter out SDD series which are not within 26.4h of a SDD culture/medication
        pdd_series_ids = med_rec_pdd.loc[
            (med_rec_pdd["min_med"] >= med_rec_pdd["min_rec"] - pd.Timedelta(60, "d"))
            & (
                med_rec_pdd["min_med"]
                <= med_rec_pdd["min_rec"] + pd.Timedelta(180, "d")
            )
        ]["tdm_series_id"].unique()

        msg = (
            f"{len(pdd_series_ids)}/{med_rec_pdd.tdm_series_id.nunique()} "
            f"({len(pdd_series_ids) / med_rec_pdd.tdm_series_id.nunique():.2%})"
        )
        self.logger.info("Number of Ciprofloxacin series labeled as PDD:")
        self.logger.info(f"\n{msg}")

        # PDD records:
        self.logger.info(
            f"Number of old PDD series: "
            f"{medication.loc[medication.is_pdd]['tdm_series_id'].nunique()} "
            f"({(medication.is_pdd).sum()} records)"
        )

        medication.loc[
            medication["tdm_series_id"].isin(pdd_series_ids), "is_pdd"
        ] = True
        medication.loc[
            medication["tdm_series_id"].isin(pdd_series_ids), "is_prophylaxis"
        ] = True

        self.logger.info(
            f"Number of new PDD series: "
            f"{medication.loc[medication.is_pdd]['tdm_series_id'].nunique()} "
            f"({(medication.is_pdd).sum()} records)"
        )

        return medication

    def _cotrim_fix_dose(self, medication: pd.DataFrame) -> pd.DataFrame:
        # First, we fix dosage calculation based on parameter names
        # As we can see, most doses are 480, 960 or 1920mg, but we have some alternative
        # doses as well.
        self.logger.info("Cotrimoxazole dose:")
        msg = (
            medication.loc[medication["atc"] == "J01EE01"]
            .groupby(["pid", "tdm_series_id"])["dose"]
            .agg(["min", "max", "count"])
            .describe()
        )
        self.logger.info(f"\n{msg}")

        # Grouping by original names and looking at rates, we see that some records are
        # describing continuous infusion
        msg = (
            medication.loc[medication["atc"] == "J01EE01"]
            .groupby(["original_name"])[["rate", "dose"]]
            .agg(["min", "max", "count"])
            .describe()
        )
        self.logger.info("Cotrim dose by original name, showing continuous infusion:")
        self.logger.info(f"\n{msg}")

        # Frequency counts to show whether we can expect this problem to have a major
        # impact
        msg = medication.loc[medication["atc"] == "J01EE01"][
            "original_name"
        ].value_counts()
        self.logger.info("Cotrim frequency counts")
        self.logger.info(f"\n{msg}")

        cotrim_dose_dict = self._cfg.cotrim_dose_dict

        self.logger.info("Filling cotrim records without a dose:")
        n_records_pre = (
            medication.loc[medication["atc"] == "J01EE01"]
            .loc[medication["dose"].isna()]
            .shape[0]
        )
        self.logger.info(f"pre-fill: {n_records_pre} records without a dose")

        # impute based on medication name
        medication["dose"] = medication["dose"].fillna(
            medication["original_name"].map(cotrim_dose_dict)
        )
        n_records_post = (
            medication.loc[medication["atc"] == "J01EE01"]
            .loc[medication["dose"].isna()]
            .shape[0]
        )
        self.logger.info(f"post-fill: {n_records_post} records without a dose")

        return medication

    def _cotrim_assign_labels(self, medication: pd.DataFrame) -> pd.DataFrame:
        """
        if dose >480/day or continuous, label as therapeutic
        therefore, rates >20mg/hr (margin of 5: 25mg/hr) or continous infusions are
        labeled as therapeutic records without an end time will be evaluated based on
        their dose total: if <=500 then profylactic

        Prophylactic:
        - hourly dose <25mg/hr
        OR
        - dose <= 480 (Note: will be rectified if multiple doses per day)

        Therapeutic (will overwrite previous prophylactic):
        - continouous infusion (handles doses in mg/kg if continuously infused)
        OR
        - time difference with previous and/or next < 16 hours (handle multiple daily
           doses, or doses in mg/kg/day)
        """

        self.logger.info("J01EE01 & is_prophylaxis (dose hourly):")
        self.logger.info(
            medication.loc[
                (medication["atc"] == "J01EE01") & (medication["is_prophylaxis"])
            ].shape[0]
        )
        medication["dose_hourly"] = medication["dose"] / (
            (medication["timestamp_diff_next"]) / pd.Timedelta(1, "h")
        )
        medication.loc[
            (medication["atc"] == "J01EE01")
            & (medication["dose_hourly"] <= 25)
            & (medication["dose_hourly"] >= 1),
            "is_prophylaxis",
        ] = True
        self.logger.info(
            medication.loc[
                (medication["atc"] == "J01EE01") & (medication["is_prophylaxis"])
            ].shape[0]
        )

        self.logger.info("J01EE01 & is_prophylaxis (240<=dose=>480):")
        self.logger.info(
            medication.loc[
                (medication["atc"] == "J01EE01") & (medication["is_prophylaxis"])
            ].shape[0]
        )
        medication.loc[
            (medication["atc"] == "J01EE01")
            & (medication["dose"] <= 480)
            & (medication["dose"] >= 240),
            "is_prophylaxis",
        ] = True
        self.logger.info(
            medication.loc[
                (medication["atc"] == "J01EE01") & (medication["is_prophylaxis"])
            ].shape[0]
        )

        self.logger.info("J01EE01 & is_prophylaxis (rate > 0):")
        self.logger.info(
            medication.loc[
                (medication["atc"] == "J01EE01") & (medication["is_prophylaxis"])
            ].shape[0]
        )
        medication.loc[
            (medication["atc"] == "J01EE01") & (medication["rate"] > 0),
            "is_prophylaxis",
        ] = False
        self.logger.info(
            medication.loc[
                (medication["atc"] == "J01EE01") & (medication["is_prophylaxis"])
            ].shape[0]
        )

        self.logger.info("J01EE01 & is_prophylaxis (next < 16h):")
        self.logger.info(
            medication.loc[
                (medication["atc"] == "J01EE01") & (medication["is_prophylaxis"])
            ].shape[0]
        )
        medication.loc[
            (medication["atc"] == "J01EE01")
            & (medication["timestamp_diff_next"] < pd.Timedelta(16, "h")),
            "is_prophylaxis",
        ] = False
        self.logger.info(
            medication.loc[
                (medication["atc"] == "J01EE01") & (medication["is_prophylaxis"])
            ].shape[0]
        )

        self.logger.info("J01EE01 & is_prophylaxis (prev < 16h):")
        self.logger.info(
            medication.loc[
                (medication["atc"] == "J01EE01") & (medication["is_prophylaxis"])
            ].shape[0]
        )
        medication.loc[
            (medication["atc"] == "J01EE01")
            & (medication["timestamp_diff_prev"] < pd.Timedelta(16, "h")),
            "is_prophylaxis",
        ] = False
        self.logger.info(
            medication.loc[
                (medication["atc"] == "J01EE01") & (medication["is_prophylaxis"])
            ].shape[0]
        )

        return medication

    def _cotrim_fix_labels_within_series(
        self, medication: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Fix Cotrim discrepant prophylaxis records within atc_series/tdm_series_ids
        At the start of an atc_series, the first record may be labeled as therapeutic
        due to starting medication in the evening and continuing the next morning. For
        series where the first record is therapeutic, and followed by a prophylactic
        record, set the first record to prophylactic as well.

        Where a single tdm_series_id contains both prophylactic and therapeutic records,
         split the tdm_series into multiple prophylactic/therapeutic series, where new
         tdm_series_ids will be assigned.
        """

        medication["tdm_series_id_record_number"] = medication.groupby(
            ["tdm_series_id"]
        ).cumcount()

        # select cotrim records
        df_cotrim = pd.DataFrame(
            medication.loc[medication["atc"] == "J01EE01"]
            .groupby(["tdm_series_id"])["is_prophylaxis"]
            .value_counts(dropna=False)
        )
        df_cotrim.columns = ["count"]
        df_cotrim.reset_index(inplace=True)
        cotrim_series_ids = df_cotrim["tdm_series_id"][
            df_cotrim.tdm_series_id.duplicated()
        ].unique()
        self.logger.debug(f"{len(cotrim_series_ids) = }")
        df_cotrim = medication.loc[
            medication["tdm_series_id"].isin(cotrim_series_ids)
        ].copy()

        # Sometimes records are falsely labeled as therapeutic because they are a result
        # of ordering when admitting patients:
        # set these records to prophylaxis
        df_cotrim["is_prophylaxis_next"] = df_cotrim.groupby(["tdm_series_id"])[
            "is_prophylaxis"
        ].shift(-1)
        first_record_in_series_is_therapeutic_and_next_record_is_prophylaxis = (
            (df_cotrim["tdm_series_id_record_number"] == 0)
            & (~df_cotrim.is_prophylaxis)
            & (df_cotrim["is_prophylaxis_next"])
        )
        false_therapeutics = df_cotrim.loc[
            first_record_in_series_is_therapeutic_and_next_record_is_prophylaxis
        ].index
        medication.loc[false_therapeutics, "is_prophylaxis"] = True

        # re-select cotrim records
        df_cotrim = pd.DataFrame(
            medication.loc[medication["atc"] == "J01EE01"]
            .groupby(["tdm_series_id"])["is_prophylaxis"]
            .value_counts(dropna=False)
        )
        df_cotrim.columns = ["count"]
        df_cotrim.reset_index(inplace=True)
        cotrim_series_ids = df_cotrim["tdm_series_id"][
            df_cotrim.tdm_series_id.duplicated()
        ].unique()
        self.logger.debug(f"{len(cotrim_series_ids) = }")
        df_cotrim = medication.loc[
            medication["tdm_series_id"].isin(cotrim_series_ids)
        ].copy()

        # Construct new series where is_prophylaxis in either True or False
        # Look at the original logic for constructing groups to create sub_series_ids
        # which are groups within groups
        # Let all other tdm_series (non-cotrim) have a sub_series_id of 0
        # Let within cotrim series, the sub_series increment
        self.logger.debug("Counting series ids during editing of series:")
        series_ids = set(medication["tdm_series_id"].unique())
        self.logger.debug(f"{len(series_ids) = }")
        ranges = set(range(int(medication.tdm_series_id.max())))
        self.logger.debug(f"{len(ranges) = }")
        self.logger.debug(f"{len([x for x in ranges if x not in series_ids]) = }")
        START_SERIES_ID = medication.tdm_series_id.max() + 1

        df_cotrim["is_prophylaxis_prev"] = df_cotrim["is_prophylaxis"].shift(1)
        df_cotrim["is_prophylaxis_change"] = (
            df_cotrim.is_prophylaxis != df_cotrim.is_prophylaxis_prev
        ).fillna(False)
        df_cotrim["cotrim_sub_series_id"] = df_cotrim["is_prophylaxis_change"].cumsum()
        df_cotrim["cotrim_series_id"] = (
            df_cotrim.groupby(["tdm_series_id", "cotrim_sub_series_id"]).ngroup()
            + START_SERIES_ID
        )
        cotrim_new_series_id_dict = df_cotrim.set_index("record_id")[
            "cotrim_series_id"
        ].to_dict()
        medication["tdm_series_id"] = (
            medication["record_id"]
            .map(cotrim_new_series_id_dict)
            .fillna(medication["tdm_series_id"])
        )

        # Recount
        self.logger.debug("Counting series ids during editing of series:")
        self.logger.debug(
            "Note: missing series ids may originate from dropping "
            "individual antibiotics doses earlier"
        )
        series_ids = set(medication["tdm_series_id"].unique())
        self.logger.debug(f"{len(series_ids) = }")
        ranges = set(range(int(medication.tdm_series_id.max())))
        self.logger.debug(f"{len(ranges) = }")
        self.logger.debug(f"{len([x for x in ranges if x not in series_ids]) = }")

        return medication

    def _label_prophylaxis_cotrimoxazole(
        self, medication: pd.DataFrame
    ) -> pd.DataFrame:
        medication = self._cotrim_fix_dose(medication=medication)

        medication = self._cotrim_assign_labels(medication=medication)

        medication = self._cotrim_fix_labels_within_series(medication=medication)

        return medication

    def _label_prophylaxis_cefazolin(self, medication: pd.DataFrame) -> pd.DataFrame:
        # Remove all Kefzol with a duration <48h
        # Kefzol | Cefazolin | J01DB04
        kefzol = (
            medication.loc[medication["atc"] == "J01DB04"]
            .groupby(["pid", "tdm_series_id"])["timestamp"]
            .agg(["min", "max", "count"])
        )
        kefzol["duration"] = kefzol["max"] - kefzol["min"]

        self.logger.info("Kefzol series duration:\n", kefzol.duration.describe())
        self.logger.info("Kefzol series duration < 48h (52.8):")
        msg_value = kefzol.loc[
            kefzol["duration"] < pd.Timedelta(52.8, "h")
        ].duration.describe()
        self.logger.info(f"\n{msg_value}")

        self.logger.info("Kefzol series duration >= 48h (52.8):")
        msg_value = kefzol.loc[
            kefzol["duration"] >= pd.Timedelta(52.8, "h")
        ].duration.describe()
        self.logger.info(f"\n{msg_value}")
        self.logger.debug(f"{kefzol.columns = }")

        kefzol_series_ids_to_drop = (
            kefzol.loc[kefzol["duration"] < pd.Timedelta(52.8, "h")]
            .reset_index()["tdm_series_id"]
            .unique()
        )
        medication.loc[
            medication["tdm_series_id"].isin(kefzol_series_ids_to_drop),
            "is_prophylaxis",
        ] = True
        return medication

    def _label_prophylaxis_based_on_atc_duration_timing(
        self,
        medication: pd.DataFrame,
    ) -> pd.DataFrame:
        mic_filepath = os.path.join(
            self.config.directory("processed"),
            self._cfg.mic_processed_filename,
        )
        mic = td.load(mic_filepath, hash_type=["sha512"])

        # SDD records:
        # All SDD medication labeled with is_sdd is definitely sdd, so we can select
        # those records for the start of an SDD period
        # is_sdd is defined in /data/definitions/antibiotics_selection.csv
        med_sdd = medication.loc[medication.is_sdd][["pid", "timestamp"]].set_index(
            "pid"
        )

        # And we also select the timestamps for the collection of SDD cultures from the
        # microbiology lab.
        mic_sdd = (
            mic.loc[mic.is_sdd][["pid", "sample_date"]]
            .drop_duplicates()
            .set_index("pid")
        )
        mic_sdd.columns = ["timestamp"]
        mic_sdd["timestamp"] = pd.to_datetime(mic_sdd["timestamp"]) + pd.Timedelta(
            12, "h"
        )

        # We create a common dataframe with all timestamps for any SDD medication or
        # culture
        rec_sdd = pd.concat([med_sdd, mic_sdd]).sort_values(["pid", "timestamp"])
        rec_sdd_diff = rec_sdd.groupby(["pid"]).timestamp.diff()
        rec_sdd_diff_more_than = rec_sdd_diff > pd.Timedelta(26.4, "h")
        rec_sdd_diff_isna = rec_sdd_diff.isna()
        rec_sdd["new_sdd"] = (rec_sdd_diff_more_than | rec_sdd_diff_isna).astype(int)
        rec_sdd["sdd_series_id"] = rec_sdd["new_sdd"].cumsum()
        rec_sdd_series = rec_sdd.groupby(["pid", "sdd_series_id"])["timestamp"].agg(
            ["min", "max", "count"]
        )

        # Next, we create SDD series based on the medication and culture records, so we
        # can identify the first occurrence of SDD
        med_sdd_series = (
            medication.loc[(medication["atc"] == "J01DD01")]  # Cefotaxime
            .groupby(["pid", "tdm_series_id"])["timestamp"]
            .agg(["min", "max", "count"])
        )
        med_sdd_series["duration"] = med_sdd_series["max"] - med_sdd_series["min"]
        print(
            "Medication potentially SDD series duration:\n",
            med_sdd_series.duration.describe(),
        )

        # Merge the SDD medication/cultures with the medication records which may fall
        # surrounding/within the SDD periods
        med_rec = pd.merge(
            med_sdd_series.reset_index(),
            rec_sdd_series.reset_index(),
            on=["pid"],
            how="left",
            suffixes=("_med", "_rec"),
        )

        # Select record series starting within 26.4h of SDD culture/medication
        # And having a duration of less than 132h (5 days + 10%)
        rec_late = med_rec["min_med"] >= med_rec["min_rec"] - pd.Timedelta(26.4, "h")
        rec_early = med_rec["min_med"] <= med_rec["min_rec"] + pd.Timedelta(26.4, "h")
        rec_duration = med_rec["duration"] <= pd.Timedelta(132, "h")
        sdd_series_ids = med_rec.loc[rec_late & rec_early & rec_duration][
            "tdm_series_id"
        ].unique()
        msg = (
            "Number of series labeled as SDD: "
            f"{len(sdd_series_ids)}/{med_rec.tdm_series_id.nunique()} "
            f"({len(sdd_series_ids) / med_rec.tdm_series_id.nunique():.2%})"
        )
        self.logger.info(msg)

        msg = (
            "Number of old SDD series: "
            f"{medication.loc[medication.is_sdd]['tdm_series_id'].nunique()} "
            f"({(medication.is_sdd).sum()} records)"
        )
        self.logger.info(msg)

        medication.loc[
            medication["tdm_series_id"].isin(sdd_series_ids), "is_sdd"
        ] = True
        medication.loc[
            medication["tdm_series_id"].isin(sdd_series_ids), "is_prophylaxis"
        ] = True
        msg = (
            "Number of new SDD series: "
            f"{medication.loc[medication.is_sdd]['tdm_series_id'].nunique()} "
            f"({(medication.is_sdd).sum()} records)"
        )
        self.logger.info(msg)

        # describe SDD series which are now labeled as therapeutic:
        mask = (
            (med_rec["min_med"] >= med_rec["min_rec"] - pd.Timedelta(26.4, "h"))
            & (med_rec["min_med"] <= med_rec["min_rec"] + pd.Timedelta(26.4, "h"))
            & (med_rec["duration"] >= pd.Timedelta(132, "h"))
            & (med_rec["duration"] < pd.Timedelta(168, "h"))
        )  # 7 days

        self.logger.info("SDD series which are now labeled as therapeutic:")
        self.logger.info(med_rec.loc[mask][["duration", "count_med"]].describe())

        return medication

    def remove_single_records_within_groups(
        self,
        medication: pd.DataFrame,
        group_label: str,
    ) -> pd.DataFrame:
        group_size_label = f"{group_label}_size"
        group_size_dict = medication.groupby(group_label).size().to_dict()
        self.logger.debug(f"{len(group_size_dict) = }")

        medication[group_size_label] = medication[group_label].map(group_size_dict)
        self.logger.info("Dropping single records within groups:")
        self.logger.debug(f"{medication.shape = }")
        medication.drop(
            medication.loc[medication[group_size_label] == 1].index, inplace=True
        )
        self.logger.debug(f"{medication.shape = }")
        return medication

    def _handle_tdm_records(
        self, medication: pd.DataFrame, ids: list | np.ndarray
    ) -> pd.DataFrame:
        """

        :param medication:
        :param ids:
        :return:
        """

        tdm_records = self._get_tdm_records(ids=ids)

        medication = self._flag_series_for_tdm(
            medication=medication,
            tdm_records=tdm_records,
        )

        medication = self._create_tdm_series_ids(
            medication=medication,
        )

        self.logger.info("VANCOMYCIN")
        self.logger.info(
            medication.loc[medication["name"] == "VANCOMYCIN"][
                ["atc_series_id", "tdm_series_id"]
            ].nunique()
        )

        self.logger.info("GENTAMICIN")
        self.logger.info(
            medication.loc[medication["name"] == "GENTAMICIN"][
                ["atc_series_id", "tdm_series_id"]
            ].nunique()
        )

        self.logger.info("Non-TDM records changes: (should be no difference)")
        self.logger.info(
            medication.loc[
                ~medication["name"].str.match("|".join(["VANCOMYCIN", "GENTAMICIN"]))
            ][["atc_series_id", "tdm_series_id"]].nunique()
        )

        return medication

    def _create_tdm_series_ids(
        self,
        medication: pd.DataFrame,
    ) -> pd.DataFrame:
        """

        :param medication:
        :return:
        """

        medication["is_last_in_atc_group"] = medication["record_id"].isin(
            medication.groupby("atc_series_id").tail(1)["record_id"].to_list()
        )

        medication[
            "timestamp_diff_next_more_than_26h"
        ] = medication.timestamp_diff_next > pd.Timedelta(value=26.4, unit="h")

        # TDM flag change = all records where a tdm flag is raised,
        # the record is the last record in the atc_series and the next record is within
        # 52.8 hours
        medication["tdm_flag_change"] = (
            (medication["tdm_flag"] == 1)
            & (medication["is_last_in_atc_group"])
            & (medication["timestamp_diff_next"] < pd.Timedelta(value=52.8, unit="h"))
        )

        # ts_diff_prev_tdm is all non-TDM records which are more than 26h apart, thus
        # setting a positive label on all records which should be labeled as a new end
        # therefore, positive TDM flags, will not be regarded for the end of a series
        # and thus include the next record as well.
        medication["timestamp_diff_prev_tdm"] = (
            medication["timestamp_diff_prev_more_than_26h"]
            & ~medication["tdm_flag_change"]
        )
        medication["tdm_sub_series_id"] = medication.groupby(
            by=["pid", "atc"],
        )["timestamp_diff_prev_tdm"].cumsum()
        medication["tdm_series_id"] = medication.groupby(
            by=["pid", "atc", "tdm_sub_series_id"],
        ).ngroup()

        self.logger.info(f"{medication['atc_series_id'].nunique() = }")
        self.logger.info(f"{medication['atc_series_id'].max() = }")
        self.logger.info(f"{medication['tdm_series_id'].nunique() = }")
        self.logger.info(f"{medication['tdm_series_id'].max() = }")

        return medication

    def _flag_series_for_tdm(
        self,
        medication: pd.DataFrame,
        tdm_records: pd.DataFrame,
    ) -> pd.DataFrame:
        """

        :param medication:
        :param tdm_records:
        :return:
        """

        med = (
            medication.loc[
                medication["name"].str.match("|".join(self._cfg.tdm_positive_values))
            ]
            .sort_values(["atc_series_id", "timestamp"])
            .drop_duplicates(
                subset=["atc_series_id"],
                keep="last",
            )
        )
        self.logger.debug(f"{med['atc_series_id'].nunique() = }")

        join_query = """
        SELECT DISTINCT m.atc_series_id
        FROM med m
        LEFT JOIN tdm_records t
        ON m.pid = t.pid
        AND m.name = t.test_name
        AND m.timestamp >= t.result_timestamp - INTERVAL '24 HOURS'
        AND m.timestamp <= t.result_timestamp + INTERVAL '24 HOURS'
        """

        tdm_positive = duckdb.query(join_query).to_df()["atc_series_id"].values
        medication["tdm_flag"] = (
            medication["atc_series_id"].isin(tdm_positive).astype(int)
        )
        return medication

    def _get_tdm_records(self, ids: list | np.ndarray) -> pd.DataFrame:
        """

        :param ids:
        :return:
        """

        lab_filepath = os.path.join(
            self.config.directory("raw"),
            self._cfg.lab_raw_filename,
        )
        lab = td.load(lab_filepath)

        lab = lab.rename(columns=self._cfg.lab_column_rename_dict)[
            self._cfg.lab_column_rename_dict.values()
        ].copy()

        m_pids = lab["pid"].isin(ids)
        m_test = (
            lab["test_name"]
            .str.lower()
            .str.contains("|".join(self._cfg.tdm_search_items), na=False)
        )
        m_numeric = lab["result_numeric"].notna()

        tdm_lab = lab.loc[m_pids & m_test & m_numeric].copy()

        m_peak_level = (
            tdm_lab["result_comment"].str.lower().str.contains("topspiegel", na=False)
        )

        tdm_lab = tdm_lab.loc[~m_peak_level].copy()

        self.logger.info(
            "Unique patients with TDM laboratory measurements: "
            f"{tdm_lab.pid.nunique()}"
        )

        tdm_lab["test_name"] = tdm_lab["test_name"].map(self._cfg.tdm_test_name_dict)
        tdm_lab = tdm_lab.loc[tdm_lab["test_name"] != "drop"].copy()

        m_tdm = pd.Series(False, index=tdm_lab.index)
        for k, v in self._cfg.tdm_positive_values.items():
            m_tdm |= (tdm_lab["test_name"] == k) & (tdm_lab["result_numeric"] >= v)

        tdm_lab = tdm_lab.loc[m_tdm].copy()

        m_result_timestamp_isna = tdm_lab["result_timestamp"].isna()
        tdm_lab.loc[m_result_timestamp_isna, "result_timestamp"] = tdm_lab.loc[
            m_result_timestamp_isna, "sample_timestamp"
        ]

        tdm_lab["result_timestamp"] = pd.to_datetime(tdm_lab["result_timestamp"])
        tdm_lab = tdm_lab.sort_values(by=["pid", "result_timestamp"]).copy()

        return tdm_lab

    def _create_atc_series(
        self,
        medication: pd.DataFrame,
        sort_by: list | np.ndarray | None = None,
        group_by: list | np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Create series of antibiotics based on their ATC code.

        If the time difference between two medication records is less than 24 hours,
        then the medication is considered to be in series. If the time difference is
        more than 24 hours, then the medication is considered to be a new series.

        The cumsum ensures that a new series number will be generated if the time
        difference is more than 24 hours and the same series number will be generated
        if the time difference is less than 24 hours subsequently. This behavior is
        different from a direct groupby which will group all records with a time
        difference less than 24 hours in the same series.

        :param medication: pd.DataFrame with medication records and ATC codes
        :return: pd.DataFrame with medication records and ATC series ids
        """

        if sort_by is None:
            sort_by = ["pid", "atc", "timestamp"]
        if group_by is None:
            group_by = ["pid", "atc"]

        medication.sort_values(by=sort_by, inplace=True)
        medication["timestamp_diff_prev"] = medication.groupby(by=group_by)[
            "timestamp"
        ].diff(
            periods=1
        )  # default, previous row
        medication["timestamp_diff_next"] = (
            medication.groupby(by=group_by)["timestamp"].diff(periods=-1) * -1
        )  # get positive timedeltas

        medication["timestamp_diff_prev_more_than_26h"] = medication[
            "timestamp_diff_prev"
        ] > pd.Timedelta(value=self._cfg.series_time_limit, unit="h")

        medication["atc_sub_series_id"] = medication.groupby(by=group_by)[
            "timestamp_diff_prev_more_than_26h"
        ].cumsum()
        medication["atc_series_id"] = medication.groupby(
            by=group_by + ["atc_sub_series_id"]
        ).ngroup()
        medication["record_id"] = medication.reset_index().index
        return medication

    def load_patients(self) -> pd.DataFrame:
        """
        Load patients from the processed data directory.
        """
        return td.load(
            os.path.join(
                self.config.directory("processed"),
                self.config.settings.patient_selection.processed_filename,
            ),
            hash_type=["sha512"],
        )

    def _get_medication_table(
        self, path: str | os.PathLike, ids: list | np.ndarray
    ) -> pd.DataFrame:
        """
        Load the medication table and reduce it to the included patients and
        antibiotics.
        :param path: string or PathLike for the raw data path
        :param ids: list or np.ndarray of patient ids to select
        :return:
        """
        medication = td.load(path)

        self.logger.info("Reducing selection to included patients:")
        self.logger.debug(f"{medication.shape[0] = }")
        medication = medication.loc[medication["Pseudo_id"].isin(ids)].copy()
        self.logger.debug(f"{medication.shape[0] = }")

        memory_usage = medication.memory_usage(index=False, deep=True).sum()
        memory_usage = round(memory_usage / (1024 * 1024), 2)
        self.logger.debug(f"{memory_usage = } MB")

        self.logger.debug("Renaming and reducing medication columns:")
        self.logger.debug(f"{medication.columns = }")
        col_rename = self.config.settings.antibiotic_selection.ab_column_rename_dict
        medication = medication.rename(columns=col_rename)[col_rename.values()].copy()

        memory_usage = medication.memory_usage(index=False, deep=True).sum()
        memory_usage = round(memory_usage / (1024 * 1024), 2)
        self.logger.debug(f"{memory_usage = } MB")
        self.logger.debug(f"{medication.columns = }")
        self.logger.debug(f"{medication.shape = }")

        medication = self._fix_medication_table_dtypes(medication)

        self._describe_medication_value_counts(medication)

        medication = self._clean_table_values(
            medication,
            self.config.settings.antibiotic_selection.columns_to_clean,
        )

        self._export_medication_frequencies(medication)

        medication = self._append_medication_selection(medication)

        medication = self._remove_not_administered_records(medication)

        return medication

    def _remove_not_administered_records(
        self, medication: pd.DataFrame
    ) -> pd.DataFrame:
        medication_status_to_drop = {
            k for k, v in self._cfg.ab_status_dict.items() if v == "drop"
        }
        self.logger.debug(f"{medication_status_to_drop = }")
        n_records_pre_drop = medication.shape[0]
        self.logger.debug(f"{medication.shape = }")
        medication = medication.loc[
            ~medication["status"].isin(medication_status_to_drop)
        ].copy()
        self.logger.debug(f"{medication.shape = }")
        n_records_post_drop = medication.shape[0]
        self.logger.debug(f"{n_records_pre_drop - n_records_post_drop = }")

        self.logger.info("Remaining status value counts")
        remaining_status_value_counts = medication.loc[
            (medication["rate"].isna() & medication["dose"].isna())
        ]["status"].value_counts()
        self.logger.info(f"\n{remaining_status_value_counts}")

        return medication

    def _append_medication_selection(self, medication: pd.DataFrame) -> pd.DataFrame:
        definitions = self._load_medication_definitions()

        for col in self._cfg.medication_definitions_merge_cols:
            definitions[col] = definitions[col].astype(str)
            medication[col] = medication[col].astype(str)

        definitions = definitions.set_index(self._cfg.medication_definitions_merge_cols)
        self.logger.debug(f"{definitions.shape = }")
        self.logger.debug(f"{definitions.columns = }")

        medication = medication.set_index(self._cfg.medication_definitions_merge_cols)
        self.logger.debug(f"{medication.shape = }")
        self.logger.debug(f"{medication.columns = }")

        self.logger.info("Merging medication with medication definitions:")
        self.logger.debug(f"{medication.shape = }")
        self.logger.debug(f"{medication.columns = }")
        medication = medication.merge(
            definitions,
            how="left",
            left_index=True,
            right_index=True,
            validate="m:1",
        )
        self.logger.debug(f"{medication.shape = }")
        self.logger.debug(f"{medication.columns = }")

        med_include_isna = medication["include"].isna()
        if med_include_isna.sum() > 0:
            missing_frequencies = medication.loc[med_include_isna]

            self.logger.debug(f"{missing_frequencies.shape = }")

            self._export_medication_frequencies(missing_frequencies)

            raise ValueError(
                "Some records were not merged with the medication definitions."
                "Please check the logs directory for the medication frequency file "
                "and add any missing records to the medication definitions file."
            )
        else:
            self.logger.info(
                "Removing all medication records which are not selected antibiotics: "
                f"{medication.shape} --> "
            )
            medication = medication.loc[medication.include == 1].copy()
            self.logger.info(medication.shape)

            self.logger.debug("Setting is_prophylaxis and is_pdd columns")
            medication["is_prophylaxis"] = medication["is_sdd"].copy()
            medication["is_pdd"] = False

            medication = medication.reset_index()

            medication = medication.rename(columns=self._cfg.definition_rename_dict)
            medication = medication.drop(columns=self._cfg.definition_drop_columns)

            if len(set(medication.columns)) != len(medication.columns):
                raise AssertionError("Duplicate columns in medication dataframe.")

            return medication

    def _load_medication_definitions(self) -> pd.DataFrame:
        definitions_path = os.path.join(
            self.config.directory("definitions"),
            self.config.settings.antibiotic_selection.ab_med_definitions_filename,
        )
        if os.path.isfile(definitions_path):
            definitions = pd.read_csv(definitions_path)
            definitions["include"] = definitions["is_ab_med"].fillna(0).astype(int)
            definitions["is_sdd"] = definitions["is_sdd"].fillna(0).astype(bool)

            definitions = self._clean_table_values(
                definitions,
                self.config.settings.antibiotic_selection.columns_to_clean,
            )
        else:
            raise FileNotFoundError(
                "Medication definitions file not found. Please run check the logs "
                "directory for the medication frequency file and create the "
                "definitions file through selecting which records to include."
                "Column include: 0 or 1, is_sdd: 0 or 1, atc: J01DB04, target_name: "
                "CEFAZOLIN, target_group: unused, check: label for manual check, "
            )
        return definitions

    def _fix_medication_table_dtypes(self, medication: pd.DataFrame) -> pd.DataFrame:
        self.logger.debug("Ensuring medication columns are of the correct type:")
        self.logger.debug(f"{medication.dtypes = }")
        memory_usage = medication.memory_usage(index=False, deep=True).sum()
        memory_usage = round(memory_usage / (1024 * 1024), 2)
        self.logger.debug(f"{memory_usage = } MB")
        medication = medication.astype(
            self.config.settings.antibiotic_selection.ab_column_type_dict
        )
        self.logger.debug(f"{medication.dtypes = }")
        memory_usage = medication.memory_usage(index=False, deep=True).sum()
        memory_usage = round(memory_usage / (1024 * 1024), 2)
        self.logger.debug(f"{memory_usage = } MB")
        return medication

    def _describe_medication_value_counts(self, medication: pd.DataFrame) -> None:
        self.logger.debug("class_therapeutic.value_counts:")
        self.logger.debug(
            f"\n"
            f"{medication['class_therapeutic'].value_counts(dropna=False).head(20)}"
        )

        self.logger.debug("class_pharmaceutical.value_counts:")
        self.logger.debug(
            f"\n"
            f"{medication['class_pharmaceutical'].value_counts(dropna=False).head(30)}"
        )

        self.logger.debug("atc.value_counts:")
        self.logger.debug(
            f"\n" f"{medication['atc'].value_counts(dropna=False).head(30)}"
        )

        self.logger.debug("status.value_counts:")
        self.logger.debug(
            f"\n" f"{medication['status'].value_counts(dropna=False).head(30)}"
        )

    def _clean_table_values(
        self, df: pd.DataFrame, columns: list | np.ndarray
    ) -> pd.DataFrame:
        for col in columns:
            self.logger.debug(f"{col}: ")
            self.logger.debug(f"{df[col].nunique()} --> ")
            unique_values = df[col].unique()

            rename_dict = {}
            for k in unique_values:
                rename_dict[k] = " ".join(str(k).split()).replace("'", "")

            df[col] = df[col].map(rename_dict)
            self.logger.debug(f"{df[col].nunique() = }")
        return df

    def _export_medication_frequencies(self, medication: pd.DataFrame) -> None:
        medication_frequencies = (
            medication.groupby(
                self.config.settings.antibiotic_selection.columns_for_medication_frequency,
                dropna=False,
                observed=True,
            )["pid"]
            .count()
            .sort_values(ascending=False)
        )

        columns_to_add = self._cfg.freq_columns_to_add

        for col in columns_to_add:
            medication_frequencies[col] = None

        self.logger.debug(f"{medication_frequencies.shape = }")
        self.logger.debug(f"{medication_frequencies.head(30)}")
        medication_frequencies.to_csv(
            os.path.join(
                self.config.directory("logs"),
                self.config.settings.antibiotic_selection.ab_med_freq_filename,
            )
        )


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

    ap = AntibioticsPreprocessor(config=_config)
    ap.run()
