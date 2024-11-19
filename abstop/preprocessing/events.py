import logging
import os

import duckdb as duckdb
import pandas as pd
import tadam as td  # Personal repo

from abstop.config import Config

logger = logging.getLogger(__name__)


class EventsCreator:
    """
    Constructs events based on antibiotics data.

    The EventsCreator class is initialized with a Config object. This object
    contains the settings for the experiment. The AntibioticsPreprocessor class loads
    the tables from the raw data directory.

    Using settings from the config class, a distinction is made between most-likely
    therapeutic and most-likely prophylactic antibiotics.
    """

    def __init__(self, config: Config):
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.logger.debug(f"from {__name__} instantiate {self.__class__.__name__}")
        self.config = config
        self._cfg = self.config.settings.events_creator

    def run(self) -> None:
        self.logger.info("Creating events")

        abpat = self.load_patients()

        # load antibiotics
        ab = self.load_antibiotics()

        # create events from antibiotics
        events = self.create_events(ab=ab)

        # combine events with demographic data
        events = self.merge_events_with_demographics(events=events, abpat=abpat)

        # combine events with antibiotic features
        events = self.get_features_antibiotics(
            events=events, medication=ab, patients=abpat
        )

        # create events from microbiology
        # TODO: determine which features to create from microbiology

        # save events
        self.save(events=events)

    def get_features_antibiotics(
        self, events: pd.DataFrame, medication: pd.DataFrame, patients: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get features from antibiotics.

        :param events: Events table.
        :param ab: Antibiotics table.
        :return: Events table with antibiotic features.
        """

        self.logger.debug("Getting features from antibiotics")

        events = self.get_atc_in_first_last_overall_periods(
            events=events, medication=medication
        )

        events = self.get_ab_durations(events=events, medication=medication)

        events = self.get_sdd_iv_in_last_14d(events=events, medication=medication)

        events = self.get_outcomes(
            events=events,
            patients=patients,
        )

        return events

    def save(self, events: pd.DataFrame) -> None:
        filepath = os.path.join(
            self.config.directory("processed"),
            self._cfg.events_filename,
        )
        td.dump(obj=events, path=filepath, hash_type=["sha512"])

    def get_outcomes(
        self,
        events: pd.DataFrame,
        patients: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Get outcomes from antibiotics.

        :param events: Events table.
        :param patients: Patients table.
        :param medication: Antibiotics table.
        :return: Events table with antibiotic outcomes.
        """

        events = self._get_outcome_restart(events=events)

        events = self._get_outcome_readmission(events=events, patients=patients)

        events = self._get_outcome_mortality(events=events, patients=patients)

        events = self._get_outcome_composite(events=events)

        events = self._get_outcome_next_atc(events=events)

        events = self._get_outcome_is_primary_series(events=events)

        events = self._handle_events_surrounding_icu_admission(events=events)

        return events

    def _get_outcome_is_primary_series(self, events: pd.DataFrame) -> pd.DataFrame:
        self.logger.debug("Getting outcome is primary series")
        # determine if an antibiotic series is the primary series if no therapeutic
        # series were given within the restart window before the start of the series

        # time diff with previous series
        events["outcome_timestamp_previous_series_stop"] = events.groupby(["pid"])[
            "stop"
        ].shift(1)
        events["outcome_timedelta_previous_series_stop"] = (
            events["start"] - events["outcome_timestamp_previous_series_stop"]
        )

        _is_primary = events["outcome_timedelta_previous_series_stop"] >= pd.Timedelta(
            79.2, "h"
        )

        events["outcome_is_primary_series"] = _is_primary.fillna(True)

        n_events = len(events)
        n_primary = events["outcome_is_primary_series"].sum()

        self.logger.info(
            f"Overall, {n_primary}/{n_events} "
            f"({round(n_primary / n_events * 100, 2)}%) primary series"
        )

        return events

    def _get_outcome_next_atc(self, events: pd.DataFrame) -> pd.DataFrame:
        self.logger.debug("Getting outcome next ATC")

        events["outcome_next_atc"] = events.groupby(["pid"])["atc_first_24h"].shift(-1)
        return events

    def _handle_events_surrounding_icu_admission(
        self, events: pd.DataFrame
    ) -> pd.DataFrame:
        event_is_stopped_after_icu_admission = events["stop"] > events["adm_icu_adm"]
        event_is_stopped_before_icu_discharge = events["stop"] <= (
            events["adm_icu_dis"] - pd.Timedelta(26.4, "h")
        )
        events = events.loc[
            event_is_stopped_after_icu_admission & event_is_stopped_before_icu_discharge
        ].copy()

        self._report_events(events=events)

        return events

    def _report_events(self, events: pd.DataFrame) -> None:
        # report on: number of events, number of restarts
        n_events = len(events)
        n_restarts = events["outcome_restart_in_72h"].sum()
        n_restarts_pct = round(n_restarts / n_events * 100, 2)
        self.logger.info(
            f"Overall, {n_restarts}/{n_events} ({n_restarts_pct}%) restart events"
        )

        n_primary = events["outcome_is_primary_series"].sum()
        self.logger.info(
            f"Overall, {n_primary}/{n_events} "
            f"({round(n_primary / n_events * 100, 2)}%) primary series"
        )

        restart_on_icu = events.loc[
            events["outcome_on_icu_restart"] & events["outcome_restart_in_72h"]
        ]
        n_restarts_on_icu = restart_on_icu.shape[0]
        n_restarts_on_icu_pct = round(n_restarts_on_icu / n_events * 100, 2)
        self.logger.info(
            f"On ICU, {n_restarts_on_icu}/{n_events} ({n_restarts_on_icu_pct}%)"
            " restart events"
        )

        n_restarts_after_icu = (
            ~events["outcome_on_icu_restart"] & events["outcome_restart_in_72h"]
        ).sum()
        n_restarts_after_icu_pct = round(n_restarts_after_icu / n_events * 100, 2)
        self.logger.info(
            "After ICU,"
            f" {n_restarts_after_icu}/{n_events} ({n_restarts_after_icu_pct}%)"
            " restart events"
        )

        # Show the proportion of antibiotic series where the series is restarted
        # within 72h and on the ICU, while this is the first time the patient received
        # antibiotics in 72h.
        n_primary_on_icu = restart_on_icu["outcome_is_primary_series"].sum()
        self.logger.info(
            f"On ICU, {n_primary_on_icu}/{n_events} "
            f"({round(n_primary_on_icu / n_events * 100, 2)}%) "
            "primary series with restart"
        )

    def _get_outcome_composite(self, events: pd.DataFrame) -> pd.DataFrame:
        # Composite outcome
        self.logger.debug("Getting composite outcome")
        events["outcome_safely_stopped"] = ~(
            events["outcome_restart_in_72h_on_icu"]
            | events["outcome_readmission_in_72h"]
            | events["outcome_mortality_in_72h"]
        )
        events["outcome_safely_stopped"].fillna(False)

        events["outcome_restart_in_72h_on_icu_or_mortality_in_72h"] = (
                events["outcome_restart_in_72h_on_icu"]
                | events["outcome_mortality_in_72h"]
        )
        return events

    def _get_outcome_mortality(
        self, events: pd.DataFrame, patients: pd.DataFrame
    ) -> pd.DataFrame:
        # Outcome mortality
        events["outcome_timestamp_mortality"] = events["pid"].map(
            patients.set_index(["pid"])["dod"].to_dict()
        )
        events["outcome_timedelta_mortality"] = (
            events["outcome_timestamp_mortality"] - events["stop"]
        )
        events["outcome_mortality_in_72h"] = events[
            "outcome_timedelta_mortality"
        ] < pd.Timedelta(79.2, "h")
        events["outcome_mortality_adm_30d"] = (
            events["outcome_timestamp_mortality"] - events["adm_icu_adm"]
        ) < pd.Timedelta(30, "d")
        events["outcome_mortality_adm_90d"] = (
            events["outcome_timestamp_mortality"] - events["adm_icu_adm"]
        ) < pd.Timedelta(90, "d")
        return events

    def _get_outcome_readmission(
        self, events: pd.DataFrame, patients: pd.DataFrame
    ) -> pd.DataFrame:
        # Outcome readmission
        next_admission_dict = (
            patients.set_index(patients["adm_icu_adm_id"])
            .groupby(["pid"])["adm_icu_adm"]
            .shift(-1)
            .to_dict()
        )
        events["outcome_timestamp_readmission"] = events["adm_icu_adm_id"].map(
            next_admission_dict
        )
        events["outcome_timedelta_readmission"] = (
            events["outcome_timestamp_readmission"] - events["stop"]
        )
        events['outcome_timedelta_icu_discharge'] = (
            events['adm_icu_dis'] - events['stop']
        )
        events["outcome_readmission_in_72h"] = events[
            "outcome_timedelta_readmission"
        ] < pd.Timedelta(79.2, "h")
        return events

    def _get_outcome_restart(self, events: pd.DataFrame) -> pd.DataFrame:
        # Determine if the restarting series is started on the ICU
        events["start_on_icu"] = (events["start"] >= events["adm_icu_adm"]) & (
            events["start"] <= events["adm_icu_dis"]
        )

        events = events.sort_values(["pid", "start"], ascending=[True, True]).copy()
        events["outcome_timestamp_restart"] = events.groupby(["pid"])["start"].shift(-1)

        events["outcome_on_icu_restart"] = (
            events.groupby(["pid"])["start_on_icu"].shift(-1).fillna(False)
        )

        events["outcome_timedelta_restart"] = (
            events["outcome_timestamp_restart"] - events["stop"]
        )
        # sanity check
        try:
            assert events["outcome_timedelta_restart"].min() >= pd.Timedelta(0, "s")
        except AssertionError as e:
            logger.critical("Negative timedelta_restart!")
            raise e

        events["outcome_restart_in_72h"] = events[
            "outcome_timedelta_restart"
        ] < pd.Timedelta(79.2, "h")

        n_restarts_overall = events["outcome_restart_in_72h"].sum()
        n_restarts_overall_pct = round(n_restarts_overall / len(events) * 100, 2)
        self.logger.info(
            f"Overall, {n_restarts_overall} ({n_restarts_overall_pct}%) restart events"
        )

        events["outcome_restart_in_72h_on_icu"] = (
            events["outcome_restart_in_72h"] & events["outcome_on_icu_restart"]
        )
        restart_on_icu = events.loc[events["outcome_on_icu_restart"]]
        n_restarts_on_icu = restart_on_icu["outcome_restart_in_72h"].sum()
        n_restarts_on_icu_pct = round(
            n_restarts_on_icu / events["start_on_icu"].shape[0] * 100, 2
        )
        self.logger.info(
            f"On ICU, {n_restarts_on_icu} ({n_restarts_on_icu_pct}%) restart events"
        )

        return events

    def get_sdd_iv_in_last_14d(
        self, events: pd.DataFrame, medication: pd.DataFrame
    ) -> pd.DataFrame:
        self.logger.debug("Getting SDD IV in last 14d")

        sdd_iv_meds = self._cfg.sdd_iv_meds

        sdd_iv_meds = medication.loc[
            medication.atc.str.contains("|".join(sdd_iv_meds))
        ][["pid", "tdm_series_id", "timestamp_tdm_adjusted"]]

        sdd_ev = pd.merge(
            events[["pid", "ab_series_id", "start", "stop"]], sdd_iv_meds, on=["pid"]
        )

        sdd_before_end_of_ab_series = sdd_ev["timestamp_tdm_adjusted"] < sdd_ev["stop"]
        sdd_not_before_14d_before_end_of_ab_series = sdd_ev[
            "timestamp_tdm_adjusted"
        ] > (sdd_ev["stop"] - pd.Timedelta(14, "d"))

        tdm_sdd_within_target_period = sdd_ev.loc[
            sdd_before_end_of_ab_series & sdd_not_before_14d_before_end_of_ab_series
        ]["tdm_series_id"].unique()
        ab_series_containing_tdm_within_target = (
            medication[["ab_series_id", "tdm_series_id"]]
            .loc[medication["tdm_series_id"].isin(tdm_sdd_within_target_period)]
            .drop_duplicates()
        )
        ab_series_with_sdd_iv_within_target = ab_series_containing_tdm_within_target[
            "ab_series_id"
        ].unique()

        events["is_sdd_within_14d_before_stop"] = False
        events.loc[
            events["ab_series_id"].isin(ab_series_with_sdd_iv_within_target),
            "is_sdd_within_14d_before_stop",
        ] = True

        sdd_after_start = sdd_ev["timestamp_tdm_adjusted"] > sdd_ev["start"]
        sdd_before_end_limit = sdd_ev["timestamp_tdm_adjusted"] < (
            sdd_ev["start"] + pd.Timedelta(14, "d")
        )

        tdm_sdd_within_target_period = sdd_ev.loc[
            sdd_after_start & sdd_before_end_limit
        ]["tdm_series_id"].unique()
        ab_series_containing_tdm_within_target = (
            medication[["ab_series_id", "tdm_series_id"]]
            .loc[medication["tdm_series_id"].isin(tdm_sdd_within_target_period)]
            .drop_duplicates()
        )
        ab_series_with_sdd_iv_within_target = ab_series_containing_tdm_within_target[
            "ab_series_id"
        ].unique()

        events["is_sdd_within_14d_after_start"] = False
        events.loc[
            events["ab_series_id"].isin(ab_series_with_sdd_iv_within_target),
            "is_sdd_within_14d_after_start",
        ] = True

        return events

    def get_ab_durations(
        self, events: pd.DataFrame, medication: pd.DataFrame
    ) -> pd.DataFrame:
        self.logger.debug("Getting antibiotic durations")
        events = self._duration_full_course(events=events)
        events = self._duration_shortest_in_last_24h(
            events=events, medication=medication
        )
        return events

    def _duration_shortest_in_last_24h(
        self, events: pd.DataFrame, medication: pd.DataFrame
    ) -> pd.DataFrame:
        # Shortest duration of any antibiotic which has been given therapeutically in
        # the last 24h

        self.logger.debug("Getting duration shortest in last 24h")

        events_atc = self._get_events_atc(events=events, medication=medication)

        events_last_24h = events_atc.loc[
            events_atc["timestamp_tdm_adjusted"]
            >= (events_atc["stop"] - pd.Timedelta(26.4, "h"))
        ]

        tdm_in_last_24h = events_last_24h["tdm_series_id"].unique()
        tdm = (
            medication.loc[medication["tdm_series_id"].isin(tdm_in_last_24h)]
            .groupby(["ab_series_id", "tdm_series_id"])["timestamp"]
            .agg(["min", "max"])
        )
        tdm_duration = tdm["max"] - tdm["min"]
        tdm_duration_min = tdm_duration.reset_index().groupby(["ab_series_id"])[0].min()
        tdm_duration_min_dict = tdm_duration_min.to_dict()
        events["ab_duration_shortest_in_last_24h"] = events["ab_series_id"].map(
            tdm_duration_min_dict
        )
        return events

    def _get_events_atc(
        self, events: pd.DataFrame, medication: pd.DataFrame
    ) -> pd.DataFrame:
        self.logger.debug("Getting events ATC")

        df_events = events[["ab_series_id", "start", "stop"]]
        df_medication = medication[
            ["ab_series_id", "atc", "timestamp_tdm_adjusted", "tdm_series_id"]
        ]
        events_atc = pd.merge(
            df_events, df_medication, on=["ab_series_id"]
        ).sort_values(["ab_series_id", "timestamp_tdm_adjusted"])

        return events_atc

    def _duration_full_course(self, events: pd.DataFrame) -> pd.DataFrame:
        events["ab_duration_full"] = events["stop"] - events["start"]
        return events

    def get_atc_in_first_last_overall_periods(
        self, events: pd.DataFrame, medication: pd.DataFrame
    ) -> pd.DataFrame:
        self.logger.debug("Getting ATC in first, last and overall periods")

        events_atc = self._get_events_atc(events=events, medication=medication)

        atc_features: dict[str, dict] = dict()
        atc_features["first"] = (
            events_atc.groupby(["ab_series_id"])
            .head(1)
            .set_index("ab_series_id")["atc"]
            .to_dict()
        )
        events_first_24h = events_atc.loc[
            events_atc["timestamp_tdm_adjusted"]
            <= (events_atc["start"] + pd.Timedelta(26.4, "h"))
        ]
        atc_features["first_24h"] = (
            events_first_24h.groupby(["ab_series_id"])["atc"]
            .apply(lambda x: tuple(sorted(set(x))))
            .to_dict()
        )
        atc_features["last"] = (
            events_atc.groupby(["ab_series_id"])
            .tail(1)
            .set_index("ab_series_id")["atc"]
            .to_dict()
        )
        events_last_24h = events_atc.loc[
            events_atc["timestamp_tdm_adjusted"]
            >= (events_atc["stop"] - pd.Timedelta(26.4, "h"))
        ]
        atc_features["last_24h"] = (
            events_last_24h.groupby(["ab_series_id"])["atc"]
            .apply(lambda x: tuple(sorted(set(x))))
            .to_dict()
        )
        atc_features["overall"] = (
            events_atc.groupby(["ab_series_id"])["atc"]
            .apply(lambda x: tuple(sorted(set(x))))
            .to_dict()
        )

        for key, feature in atc_features.items():
            events[f"atc_{key}"] = (
                events["ab_series_id"].map(feature).astype("category")
            )
        return events

    def merge_events_with_demographics(
        self, events: pd.DataFrame, abpat: pd.DataFrame
    ) -> pd.DataFrame:
        self.logger.debug("Merging events with demographics")

        self.logger.debug(f"{events.shape = }")
        self.logger.debug(f"{abpat.shape = }")

        query = """
        SELECT *
        FROM events e
        LEFT JOIN abpat a 
        ON e.pid = a.pid
        AND e.stop >= a.adm_icu_adm
        AND e.stop <= a.adm_icu_dis
        """

        new_events = duckdb.query(query).to_df()

        self.logger.debug(f"{new_events.shape = }")
        self.logger.debug(f"{new_events.pid_2.notna().mean() = }")

        return new_events

    def create_events(self, ab: pd.DataFrame) -> pd.DataFrame:
        """
        Create events from antibiotics and patients.

        :param abpat: Antibiotics and patients table.
        :param ab: Antibiotics table.
        :return: Events table.
        """

        self.logger.debug("Creating events")

        # Create events from antibiotics
        events = (
            ab.loc[~ab.is_prophylaxis]
            .groupby(["pid", "ab_series_id"])["timestamp"]
            .agg(["min", "max"])
            .reset_index()
            .rename(columns={"min": "start", "max": "stop"})
            .sort_values(["pid", "start"])
        )
        self.logger.info(f"Created {len(events)} events from antibiotics")
        return events

    def load_antibiotics(self) -> pd.DataFrame:
        self.logger.debug("Loading antibiotics")
        filepath = os.path.join(
            self.config.directory("processed"),
            self._cfg.antibiotics_filename,
        )
        return td.load(filepath, hash_type=["sha512"])

    def load_patients(self) -> pd.DataFrame:
        self.logger.debug("Loading patients")
        filepath = os.path.join(
            self.config.directory("processed"),
            self._cfg.patients_filename,
        )
        return td.load(filepath, hash_type=["sha512"])


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

    ap = EventsCreator(config=_config)
    ap.run()
