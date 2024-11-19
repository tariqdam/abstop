import logging
import os

import duckdb as duckdb
import numpy as np
import pandas as pd
import tadam as td  # Personal repo

from abstop.config import Config
from abstop.utils.processing import create_unique_identifier

logger = logging.getLogger(__name__)


class MicrobiologyPreprocessor:
    """
    Selects patients from the various tables in the RDP dataset of Amsterdam UMC.

    The PatientSelector class is initialized with a Config object. This object
    contains the settings for the experiment. The PatientSelector class loads
    the tables from the raw data directory and stores them in a dictionary.

    """

    def __init__(self, config: Config):
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.logger.debug(f"from {__name__} instantiate {self.__class__.__name__}")
        self.config = config

    def run(self) -> None:
        self.logger.debug(f"Running {self.__class__.__name__}")

        patients = self.load_patients()
        patient_ids = patients["pid"].unique()
        self.logger.debug(td.utils.summary(patients))

        df_side = self.load_data_sideloaded(patient_ids=patient_ids)
        df_labtrain = self.load_data_labtrain(patient_ids=patient_ids)

        # Remove patients from merge, as we will verify all available microbe records
        # are included in the merge.
        df_merged = self.merge_data(df_side=df_side, df_labtrain=df_labtrain)

        # Verify grouping structures
        df_merged = self.add_grouping_structures(df_merged)

        # Describe grouping structures
        self.describe_grouping_structures(df_merged, patients)

        # Left join patients to keep only records from patients in the patient selection
        df_merged = df_merged.loc[df_merged["pid"].isin(patient_ids)].copy()

        # set boolean values where applicable
        df_merged = self.set_boolean_values(df_merged, patients)
        self.describe_boolean_values(df_merged)

        self.save_data(df_merged)

    def set_boolean_values(
        self,
        df_merged: pd.DataFrame,
        patients: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Set boolean values for the following columns:
        - is_hematology: department is hematology
        - is_sdd: result_name contains SDD
        - is_pdd: is_pdd | (is_hematology & is_sdd)
        - is_icu:
        - is_positive: groupings != negative
        - is_blood: sample_description_groups == blood
        - is_sputum: sample_description_groups == sputum
        - is_urine: sample_description_groups == urine
        - is_tip: sample_description_groups == tip

        :param df_merged: pandas DataFrame with merged microbiology data
        :param patients: pandas DataFrame with patient selection
        :return:
        """
        df_merged["is_hematology"] = (
            df_merged["department"].map(
                self.config.settings.microbiology.hematology_department_dict
            )
            == "hematology"
        )

        m_contains_sdd = df_merged["result_name"].str.contains("SDD", na=False)
        m_contains_pdd = df_merged["result_name"].str.contains("PDD", na=False)

        df_merged["is_sdd"] = m_contains_sdd
        df_merged["is_pdd"] = m_contains_pdd & ~m_contains_sdd
        df_merged["is_pdd"] = df_merged.is_pdd | (
            df_merged.is_hematology & df_merged.is_sdd
        )


        # determine which records were collected within the ICU
        mic_merge_icu_patients_query = """
        SELECT d.id
        FROM df_merged d
        JOIN patients p
        ON d.pid = p.pid
        WHERE p.adm_icu_adm <= d.sample_date
        AND p.adm_icu_dis >= d.sample_date
        """

        mic_records_on_icu = duckdb.query(mic_merge_icu_patients_query).to_df()

        df_merged["is_icu"] = df_merged["id"].isin(mic_records_on_icu["id"])
        df_merged["is_positive"] = df_merged["groupings"] != "negative"
        df_merged["is_blood"] = df_merged["sample_description_groups"] == "blood"
        df_merged["is_sputum"] = df_merged["sample_description_groups"] == "sputum"
        df_merged["is_urine"] = df_merged["sample_description_groups"] == "urine"
        df_merged["is_tip"] = df_merged["sample_description_groups"] == "tip"

        df_merged['sdd_positive'] = (
                                        df_merged['is_sdd'] &
                                        (df_merged['gram'] == "negative") &
                                        df_merged['shape'].str.match("|".join(["rods", "coccobacillus"]), na=False)
        ).fillna(False)

        return df_merged

    def describe_boolean_values(self, df_merged: pd.DataFrame) -> None:
        self.logger.info("PDD/SDD value counts and ranges: ")
        spdd_value_counts = (
            df_merged.loc[df_merged["result_name"].str.contains("DD", na=False)]
            .groupby(["result_name", "is_pdd"])
            .agg(
                {
                    "pid": "nunique",
                    "sample_date": ["count", "min", "max"],
                }
            )
        )
        self.logger.info(spdd_value_counts)

    def describe_grouping_structures(
        self,
        df_merged: pd.DataFrame,
        patients: pd.DataFrame,
    ) -> None:
        """
        Describe grouping structures of the microbiology data and report an overview
        of the number of samples per description group and groupings, for description
        groups with more than 10k samples and groupings with more than 100 samples.

        Also, report counts and date range per hospital for SDD and PDD samples.

        :param df_merged: pandas DataFrame with merged microbiology data
        :param patients: pandas DataFrame with patient selection
        :return:
        """

        # Report an overview of the number of samples per description group and
        # groupings, for description groups with more than 10k samples and
        # groupings with more than 100 samples.

        mg_sdg_vc = df_merged["sample_description_groups"].value_counts(dropna=False)

        self.logger.info("Sample description groups value counts:")
        self.logger.info(mg_sdg_vc)

        mg_sdg_vc_a_10k = mg_sdg_vc[mg_sdg_vc > 10_000]
        mg_sdg_vc_a_10k_keys = mg_sdg_vc_a_10k.keys()

        mask_mic_group = df_merged["sample_description_groups"].isin(
            mg_sdg_vc_a_10k_keys
        )

        mic_group_filtered = df_merged.loc[mask_mic_group]
        mic_group_filtered_grouped = mic_group_filtered.groupby(
            ["sample_description_groups"]
        )
        mgfg_vc = mic_group_filtered_grouped["groupings"].value_counts(dropna=False)

        mgfg_vc_a_100 = mgfg_vc[mgfg_vc > 100]
        self.logger.debug(
            "Microbe groupings (neg/enterobact./candida) with"
            " more than 100 samples for sample description groups"
            " (blood/sputum/etc) with more than 10k samples:"
        )
        self.logger.debug(mgfg_vc_a_100)

        self.logger.debug(f"{df_merged.groupings.value_counts() =}")

        mg_rn_vc = df_merged["result_name"].value_counts(dropna=False)
        self.logger.debug(
            f"mic_group.result_name.value_counts(dropna=False) =" f" {mg_rn_vc}"
        )

        a = pd.merge(df_merged, patients, on=["pid"])

        a["sample_date"] = pd.to_datetime(a["sample_date"])
        a["adm_icu_adm"] = pd.to_datetime(a["adm_icu_adm"])
        a["adm_icu_dis"] = pd.to_datetime(a["adm_icu_dis"])

        a = a.loc[
            (a["sample_date"] >= a["adm_icu_adm"])
            & (a["sample_date"] <= a["adm_icu_dis"])
        ].copy()

        counts_and_daterange_per_hospital = (
            a.loc[a["result_name"].str.contains("|".join(["SDD", "PDD"]), na=False)]
            .groupby(["adm_hosp_loc", "result_name"])
            .agg(
                {
                    "pid": "nunique",
                    "sample_date": ["count", "min", "max"],
                }
            )
        )
        self.logger.info("Counts and date range per hospital:")
        self.logger.info(counts_and_daterange_per_hospital)

    def add_grouping_structures(self, df_merged: pd.DataFrame) -> pd.DataFrame:
        """
        Add grouping structures to the microbiology data. If the grouping structures
        are not yet defined, create a file with the grouping structures and raise
        an error. If the grouping structures are defined, merge them with the
        microbiology data. If there are still records without grouping structures,
        create a file with the missing grouping structures and raise an error.

        :param df_merged: pandas DataFrame with merged microbiology data
        :return: pandas DataFrame with the same contents and column groupings
        """

        microbe_groups_path = os.path.join(
            self.config.directory("definitions"),
            "microbe_groupings.csv",
        )

        group_by_columns = [
            "result_overall",
            "microbe_gram",
            "microbe_catcustom",
            "microbe_spotf",
        ]

        if os.path.isfile(microbe_groups_path):
            microbe_groups = pd.read_csv(
                os.path.join(
                    self.config.directory("definitions"), "microbe_groupings.csv"
                )
            )

            microbe_groups.info()

            microbe_groups_columns = [
                "result_overall",
                "microbe_gram",
                "microbe_spotf",
                "groupings",
                "pathogenicity",
                "gram",
                "shape",
            ]

            microbe_groups = microbe_groups[microbe_groups_columns].drop_duplicates()

            merge_on_columns = [
                "result_overall",
                "microbe_gram",
                "microbe_spotf",
            ]

            self.logger.debug(f"{df_merged.shape = }")
            self.logger.debug(f"{microbe_groups.shape = }")
            mic_group = df_merged.merge(
                right=microbe_groups,
                on=merge_on_columns,
                how="left",
            )
            self.logger.debug(f"{mic_group.shape = }")

            if mic_group.groupings.isna().sum() > 0:
                missing_groups = (
                    mic_group.loc[mic_group["groupings"].isna()]
                    .groupby(group_by_columns, dropna=False)["id"]
                    .count()
                )

                file_path_name = os.path.join(
                    self.config.directory("definitions"),
                    "microbe_groupings_missing.csv",
                )
                missing_groups.to_csv(file_path_name)

                raise ValueError(
                    f"Missing groupings ({mic_group.groupings.isna().sum()}):"
                    f" see {file_path_name}"
                )
            else:
                self.logger.info("All groupings are present")
                self.logger.info("Describing grouping contents:")

                # %%
                mic_group_value_counts = mic_group.groupby(["sample_description"])[
                    "groupings"
                ].value_counts(dropna=False)
                self.logger.debug(f"{mic_group_value_counts.shape = }")

                mic_group_value_counts_above_500 = mic_group_value_counts[
                    mic_group_value_counts > 500
                ]
                self.logger.debug(f"{mic_group_value_counts_above_500.shape = }")
                self.logger.debug(f"{mic_group_value_counts_above_500 = }")

                _cfg = self.config.settings.microbiology
                rd = _cfg.sample_description_value_rename_dict

                mic_group["sample_description_groups"] = mic_group[
                    "sample_description"
                ].map(rd)

                return mic_group

        else:
            microbe_groups = (
                df_merged.groupby(group_by_columns, dropna=False)["id"]
                .count()
                .reset_index()
            )
            microbe_groups["groupings"] = None

            microbe_groups.to_csv(microbe_groups_path)

            raise FileNotFoundError(
                f"No file existed yet, created new {microbe_groups_path}."
                f"Label this file with the correct groupings and rerun the script."
            )

    def merge_data(
        self, df_side: pd.DataFrame, df_labtrain: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge the two deliveries of microbiology data. The first delivery is the
        side-loaded data, the second delivery is the labtrain data.

        :param df_side:
        :param df_labtrain:
        :return:
        """
        self.logger.debug("Running merge_data()")

        uuid_columns = [
            "pid",
            "sample_id",
            "isolate_number",
            "sample_date",
            "result_code",
            "result_overall",
        ]
        self.logger.debug(f"{uuid_columns = }")

        # Compare the size of the two separate data deliveries
        self.logger.info("First delivery: ")
        self.logger.info(f"{df_side.shape = }")
        self.logger.info(f"{df_side.drop_duplicates().shape = }")
        self.logger.info("Labtrain delivery: ")
        self.logger.info(f"{df_labtrain.shape = }")
        self.logger.info(f"{df_labtrain.drop_duplicates().shape = }")

        df_side["id"] = create_unique_identifier(
            df=df_side,
            columns=uuid_columns,
        )
        df_labtrain["id"] = create_unique_identifier(
            df=df_labtrain,
            columns=uuid_columns,
        )

        df_side_id_unique = df_side["id"].unique()
        df_labtrain_id_unique = df_labtrain["id"].unique()

        self.logger.debug(
            "Records in first delivery which are NOT in the second" " delivery:"
        )
        self.logger.debug(
            f"{df_side.loc[~df_side['id'].isin(df_labtrain_id_unique)].shape = }"
        )

        self.logger.debug("Records in first delivery which are ALSO in the second:")
        self.logger.debug(
            f"{df_side.loc[df_side['id'].isin(df_labtrain_id_unique)].shape = }"
        )

        self.logger.debug("Records in second delivery which are NOT in the first:")
        self.logger.debug(
            f"{df_labtrain.loc[~df_labtrain['id'].isin(df_side_id_unique)].shape = }"
        )

        self.logger.debug("Records in second delivery which are ALSO in the first:")
        self.logger.debug(
            f"{df_labtrain.loc[df_labtrain['id'].isin(df_side_id_unique)].shape = }"
        )

        # Show the various custom grouping categories for records which are available
        # both deliveries
        self.logger.info("First delivery (custom grouping categories): ")
        self.logger.info(
            df_side.loc[df_side["id"].isin(df_labtrain_id_unique)]["microbe_catcustom"]
            .value_counts(dropna=False)
            .head(5)
        )
        self.logger.info("\nLabtrain delivery (custom grouping categories): ")
        self.logger.info(
            df_labtrain.loc[df_labtrain["id"].isin(df_side_id_unique)][
                "microbe_catcustom"
            ]
            .value_counts(dropna=False)
            .head(5)
        )

        # Let mic be the concatenated dataframe of the two deliveries, giving priority
        # to the df_labtrain delivery by removing records which are also in the
        # df_side delivery from the df_side delivery.
        mic = pd.concat(
            [
                df_side.loc[~df_side["id"].isin(df_labtrain_id_unique)],
                df_labtrain[[x for x in df_side]],
            ]
        )

        # %%
        # remove records without results
        self.logger.info("Removing records without results from mic dataframe")
        self.logger.info(f"{mic.shape = }")
        mic = mic.dropna(
            subset=["result_overall", "result_text", "microbe_spotf"], how="all"
        )
        self.logger.info(f"{mic.shape = }")

        return mic

    def load_data_sideloaded(self, patient_ids: np.ndarray) -> pd.DataFrame:
        self.logger.debug("Running load_data()")
        df_side_filepath = os.path.join(
            self.config.directory("side_loaded"),
            self.config.settings.microbiology.df_side_loaded_filename,
        )
        side_loaded = pd.read_csv(df_side_filepath, sep="\t").rename(
            columns=self.config.settings.microbiology.df_column_rename_dict
        )
        self.logger.debug(f"{side_loaded.shape = }")
        self.logger.debug("Reducing dataframe to only inlude included patients")
        side_loaded = side_loaded[side_loaded["pid"].isin(patient_ids)]
        self.logger.debug(f"{side_loaded.shape = }")
        self.logger.debug("Converting date columns to datetime")
        side_loaded["sample_date"] = pd.to_datetime(
            side_loaded["sample_date"], format="%Y-%m-%d"
        )
        self.logger.debug(td.utils.summary(side_loaded))

        self.logger.info(
            side_loaded.groupby(
                "department",
                dropna=False,
            )["sample_date"]
            .agg(["min", "max", "count"])
            .sort_values(
                ["count", "department"],
                ascending=[False, True],
            )
        )

        return side_loaded

    def load_data_labtrain(self, patient_ids: np.ndarray) -> pd.DataFrame:
        self.logger.debug("Running load_data()")
        df_labtrain_filepath = os.path.join(
            self.config.directory("raw"),
            self.config.settings.microbiology.df_labtrain_filename,
        )
        df_labtrain = td.load(df_labtrain_filepath).rename(
            columns=self.config.settings.microbiology.df_column_rename_dict
        )
        self.logger.debug(f"{df_labtrain.shape = }")
        self.logger.debug("Reducing dataframe to only inlude included patients")
        df_labtrain = df_labtrain[df_labtrain["pid"].isin(patient_ids)]
        self.logger.debug(f"{df_labtrain.shape = }")

        self.logger.debug("Converting date columns to datetime")
        df_labtrain["sample_date"] = pd.to_datetime(
            df_labtrain["sample_date"], format="%d-%m-%Y"
        )
        self.logger.debug(td.utils.summary(df_labtrain))

        self.logger.info(
            df_labtrain.groupby(
                "department",
                dropna=False,
            )["sample_date"]
            .agg(["min", "max", "count"])
            .sort_values(
                ["count", "department"],
                ascending=[False, True],
            )
        )

        return df_labtrain

    def load_patients(self) -> pd.DataFrame:
        return td.load(
            os.path.join(
                self.config.directory("processed"),
                self.config.settings.patient_selection.processed_filename,
            ),
            hash_type=["sha512"],
        )

    def save_data(self, df_merged: pd.DataFrame) -> None:
        self.logger.debug("Running save_data()")
        self.logger.info(f"{df_merged.shape = }")
        self.logger.info(f"{df_merged.columns = }")
        self.logger.debug(td.utils.summary(df_merged))
        td.dump(
            df_merged,
            os.path.join(self.config.directory("processed"), "0_mic.pkl.gz"),
            hash_type=["sha512"],
        )


if __name__ == "__main__":
    _config = Config(root="C:\\TADAM\\projects\\abstop")
    _mic_processor = MicrobiologyPreprocessor(_config)
    _mic_processor.run()
