import glob
import logging
import os

import numpy as np
import pandas as pd
import seaborn as sns
import tadam as td  # Personal repo
from matplotlib import pyplot as plt

from abstop.config import Config

logger = logging.getLogger(__name__)


class PatientSelector:
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
        self.tables = self.load_tables()
        self.admissions = self.get_admissions()

    def run(self) -> pd.DataFrame:
        self.logger.debug("Running PatientSelector.run()")
        return self.select_patients()

    def select_patients(self) -> pd.DataFrame:
        self.logger.debug("Running PatientSelector.select_patients()")
        self.logger.info("Selecting patients")

        patients: pd.DataFrame = self.get_patients()
        hospital_admissions: pd.DataFrame = self.get_hospital_admissions()
        icu_admissions: pd.DataFrame = self.get_icu_admissions()

        self.logger.debug(
            "Creating abpat dataframe from icu_admissions, hospital_admissions, and "
            "patients dataframes"
        )
        self.logger.debug(f"{icu_admissions.shape = }")
        self.logger.debug(td.utils.summary(icu_admissions))
        self.logger.debug(f"{hospital_admissions.shape = }")
        self.logger.debug(td.utils.summary(hospital_admissions))
        self.logger.debug(f"{patients.shape = }")
        self.logger.debug(td.utils.summary(patients))

        abpat: pd.DataFrame = pd.merge(
            icu_admissions,
            hospital_admissions,
            on="pid",
            how="left",
        ).merge(patients, on="pid", how="left")

        self.logger.debug(f"{abpat.shape = }")
        self.logger.debug(td.utils.summary(abpat))

        abpat = abpat.loc[
            (abpat["adm_icu_adm"] >= abpat["adm_hosp_adm"])
            & (abpat["adm_icu_adm"] <= abpat["adm_hosp_dis"])
        ].copy()

        self.logger.debug(f"{abpat.shape = }")
        self.logger.debug(td.utils.summary(abpat))

        for col in [
            "adm_icu_adm",
            "adm_icu_dis",
            "adm_hosp_adm",
            "adm_hosp_dis",
            "dod",
        ]:
            self.logger.debug(f"Converting {col} to datetime")
            abpat[col] = pd.to_datetime(abpat[col])

        self.logger.debug(f"{abpat.shape = }")
        self.logger.debug(td.utils.summary(abpat))

        self.logger.info("Summary of the selected patients: ")
        self.logger.debug("td.utils.summary(abpat) = ")
        self.logger.info(f"\n{td.utils.summary(abpat)}")

        self.logger.info(
            f"\nWe are starting with {patients.shape[0]} patients, "
            f"over {hospital_admissions.shape[0]} hospital admissions and "
            f"{icu_admissions.shape[0]} ICU admissions. We have then excluded "
            f"{(abpat['adm_hosp_age'] < 18).sum()} ICU admissions due to age < 18."
        )

        abpat = abpat.loc[abpat["adm_hosp_age"] >= 18].copy()

        self.logger.debug(f"{abpat.shape = }")
        self.logger.debug(td.utils.summary(abpat))

        self.log_admission_duration_statistics(abpat=abpat, level="debug")

        abpat_duration = abpat["adm_icu_dis"] - abpat["adm_icu_adm"]
        n_icu_admissions_under_26h = (abpat_duration <= pd.Timedelta(26.4, "h")).sum()
        abpat_duration_over_26h = abpat.loc[abpat_duration > pd.Timedelta(26.4, "h")]
        n_icu_admissions_over_26h = abpat_duration_over_26h.shape[0]
        n_unique_hosp_admissions_over_26h = abpat_duration_over_26h[
            "adm_hosp_adm_id"
        ].nunique()
        n_unique_patients_over_26h = abpat_duration_over_26h["pid"].nunique()

        self.logger.info(
            f"\nWe will select only the admissions with at least 24 hours in the ICU "
            f"as we want to filter out short-term stays for cardio-thoracic surgery "
            f"patients, as well as erroneous registrations.\nThis filters out "
            f"{n_icu_admissions_under_26h} ICU admissions, leaving us with "
            f"{n_icu_admissions_over_26h} ICU admissions for analysis over "
            f"{n_unique_hosp_admissions_over_26h} hospital admissions and "
            f"{n_unique_patients_over_26h} patients."
        )

        self.log_admission_duration_statistics(abpat=abpat, level="debug")

        self.plot_icu_admissions(
            abpat=abpat, file_name="0__patient_selector__icu_hosp_los__all.png"
        )

        self.logger.info("Removing ICU admissions with LOS < 26.4 hours")
        self.logger.debug(f"{abpat.shape = }")
        abpat = abpat.loc[
            abpat["adm_icu_dis"] - abpat["adm_icu_adm"] > pd.Timedelta(26.4, "h")
        ].copy()
        self.logger.debug(f"{abpat.shape = }")

        self.log_admission_duration_statistics(abpat=abpat, level="info")

        self.plot_icu_admissions(
            abpat=abpat, file_name="0__patient_selector__icu_hosp_los__26h.png"
        )

        abpat = self.add_bmi_to_abpat(abpat=abpat)

        abpat = self.calculate_derived_weights(abpat=abpat)

        files_to_dump = {
            "patients": patients,
            "hospital_admissions": hospital_admissions,
            "icu_admissions": icu_admissions,
            "abpat": abpat,
        }

        self.dump_files(
            files_to_dump=files_to_dump,
            hash_type=["sha512"],
        )

        return abpat

    def calculate_derived_weights(self, abpat: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the derived weights for the patients in the abpat dataframe.

        Adds columns for predicted_weight, and adjusted weight.

        :param abpat: pd.DataFrame containing columns for sex, length, and weight
        :return: pd.DataFrame
        """

        abpat["predicted_weight"] = abpat.apply(
            lambda x: self.predicted_body_weight(x["is_sex__male"], x["length"]), axis=1
        )

        abpat["adjusted_weight"] = abpat.apply(
            lambda x: self.adjusted_body_weight(x["predicted_weight"], x["weight"]),
            axis=1,
        )
        return abpat

    def predicted_body_weight(self, is_male: bool, length: float) -> float:
        """
        Calculate the predicted body weight

        Returns predicted body weight in kg as float
        """

        if length is None:
            return np.nan
        if length < 120:
            return np.nan

        base_weight = 0.91 * (length - 152.4) + 45.5
        if is_male is None:
            # estimate an average weight
            return base_weight + 2.25
        if is_male:
            return base_weight + 4.5
        else:
            return base_weight

    def adjusted_body_weight(
        self, predicted_body_weight: float, actual_body_weight: float
    ) -> float:
        if actual_body_weight is None:
            return np.nan
        if predicted_body_weight is None:
            return np.nan
        adjustment = actual_body_weight - predicted_body_weight
        return predicted_body_weight + (0.4 * adjustment)

    def dump_files(
        self, files_to_dump: dict[str, pd.DataFrame], hash_type: list[str]
    ) -> None:
        for filename, obj in files_to_dump.items():
            file_path = os.path.join(
                self.config.directory("processed"), f"0_{filename}.pkl.gz"
            )
            self.logger.debug(f"Dumping {filename} to {file_path}")

            td.dump(obj, file_path, hash_type=hash_type)
            hash_read = td.utils.read_hash(path=file_path, hash_type=hash_type)
            for h_type, h_value in hash_read.items():
                self.logger.info(f"{h_type} hash of {filename} = {h_value}")

    def log_admission_duration_statistics(
        self, abpat: pd.DataFrame, level: str
    ) -> None:
        mask_hosp_amc = abpat["adm_hosp_loc"] == "amc"
        abpat_amc = abpat.loc[mask_hosp_amc]
        n_admissions_hosp_amc = abpat_amc["adm_hosp_adm_id"].nunique()
        abpat_amc_first_adm = abpat_amc["adm_icu_adm"].min()
        abpat_amc_last_adm = abpat_amc["adm_icu_adm"].max()

        mask_hosp_vumc = abpat["adm_hosp_loc"] == "vumc"
        abpat_vumc = abpat.loc[mask_hosp_vumc]
        n_admissions_hosp_vumc = abpat_vumc["adm_hosp_adm_id"].nunique()
        abpat_vumc_first_adm = abpat_vumc["adm_icu_adm"].min()
        abpat_vumc_last_adm = abpat_vumc["adm_icu_adm"].max()

        # TODO: filter out patients with multiple ICU admissions
        los_hosp = abpat["adm_hosp_dis"] - abpat["adm_hosp_adm"]
        los_hosp_description = los_hosp.describe()
        los_hosp_description_grouped = (
            los_hosp.groupby(abpat["adm_hosp_loc"]).describe().T
        )

        los_icu = abpat["adm_icu_dis"] - abpat["adm_icu_adm"]
        los_icu_description = los_icu.describe()
        los_icu_description_grouped = (
            los_icu.groupby(abpat["adm_hosp_loc"]).describe().T
        )

        msg = (
            f"Divided per hospital we have:\n"
            f"AMC: {n_admissions_hosp_amc} hospital admissions and "
            f"{mask_hosp_amc.sum()} ICU admissions between {abpat_amc_first_adm} and "
            f"{abpat_amc_last_adm}.\n"
            f"VUmc: {n_admissions_hosp_vumc} hospital admissions and "
            f"{mask_hosp_vumc.sum()} ICU admissions between {abpat_vumc_first_adm} and "
            f"{abpat_vumc_last_adm}.\n\n"
            f"The average length of stay in the hospital is: \n"
            f"{los_hosp_description}\n\n"
            f"The average length of stay in the hospital, stratified by hospital is:\n"
            f"{los_hosp_description_grouped}\n\n"
            f"The average length of stay in the ICU is: \n"
            f"{los_icu_description}\n\n"
            f"The average length of stay in the ICU, stratified by hospital is:\n"
            f"{los_icu_description_grouped}\n\n"
        )

        match level:
            case "debug":
                self.logger.debug(msg)
            case "info":
                self.logger.info(msg)
            case _:
                raise ValueError(f"Unknown level: {level}")

    def add_bmi_to_abpat(self, abpat: pd.DataFrame) -> pd.DataFrame:
        meting_algemeen = td.load(self.tables["MetingAlgemeenICU"])
        meting_vitale = td.load(self.tables["MetingVitaleGegevens"])

        bmi = (
            meting_algemeen[["Pseudo_id", "MeetMoment", "BMIberekend"]]
            .dropna()
            .rename(
                columns={
                    "Pseudo_id": "pid",
                    "MeetMoment": "bmi_date",
                    "BMIberekend": "bmi",
                }
            )
        )
        bmi = bmi.loc[bmi["pid"].isin(abpat.pid.unique())]

        self.logger.info(
            f"""
            We have a total of {bmi.shape[0]} BMI measurements over 
            {bmi['pid'].nunique()} patients, indicating missingness for at least 
            {abpat['pid'].nunique() - bmi['pid'].nunique()} patients.
            """
        )

        length = (
            meting_vitale[["Pseudo_id", "MeetMoment", "VitaleGegevensBMIberekening"]]
            .dropna()
            .rename(
                columns={
                    "Pseudo_id": "pid",
                    "MeetMoment": "length_date",
                    "VitaleGegevensBMIberekening": "bmi",
                }
            )
        )
        length = length.loc[length["pid"].isin(abpat.pid.unique())].dropna()

        self.logger.info(
            f"""
            We have a total of {length.shape[0]} length measurements over
             {length['pid'].nunique()} patients, indicating missingness for at least 
             {abpat['pid'].nunique() - length['pid'].nunique()} patients.
            """
        )

        vitals = (
            meting_vitale[
                [
                    "Pseudo_id",
                    "MeetMoment",
                    "VitaleGegevensLengte",
                    "VitaleGegevensGewicht",
                    "VitaleGegevensBMIberekening",
                ]
            ]
            .dropna(subset=["VitaleGegevensLengte", "VitaleGegevensGewicht"], how="all")
            .sort_values(["Pseudo_id", "MeetMoment"])
            .rename(
                columns={
                    "Pseudo_id": "pid",
                    "MeetMoment": "measured_date",
                    "VitaleGegevensLengte": "length",
                    "VitaleGegevensGewicht": "weight",
                    "VitaleGegevensBMIberekening": "bmi",
                }
            )
        )
        vitals["measured_date"] = pd.to_datetime(vitals["measured_date"])

        abpat = pd.merge_asof(
            abpat.sort_values("adm_hosp_adm"),
            vitals.sort_values("measured_date").dropna(
                subset=["length", "weight"], how="any"
            ),
            left_on="adm_hosp_adm",
            right_on="measured_date",
            by="pid",
            tolerance=pd.Timedelta(365 * 24 * 3, "h"),
            direction="nearest",
        )
        abpat["bmi"] = abpat["bmi"].fillna(
            abpat["weight"] / (abpat["length"] / 100) ** 2
        )

        bmi["bmi_date"] = pd.to_datetime(bmi["bmi_date"])
        abpat = pd.merge_asof(
            abpat.sort_values("adm_hosp_adm"),
            bmi.sort_values("bmi_date"),
            left_on="adm_hosp_adm",
            right_on="bmi_date",
            by="pid",
            tolerance=pd.Timedelta(365 * 24, "h"),
            direction="nearest",
        )

        abpat["bmi"] = abpat["bmi_x"].fillna(abpat["bmi_y"])
        abpat["bmi"].notna().mean()

        abpat["length"] = abpat["length"].fillna(
            (abpat["weight"] / abpat["bmi"]) ** 0.5 * 100
        )
        abpat["weight"] = abpat["weight"].fillna(
            (abpat["length"] / 100**2) * abpat["bmi"]
        )
        abpat = abpat.drop(
            columns=[
                x for x in ["bmi_x", "bmi_y", "bmi_date", "measured_date"] if x in abpat
            ]
        ).copy()

        self.logger.info(
            f"""
            We have a total of {abpat.shape[0]} patients with {abpat['pid'].nunique()} 
            unique patients.
            
            The average age is {abpat['adm_hosp_age'].mean():.2f} years, with a standard
             deviation of {abpat['adm_hosp_age'].std():.2f} years. The youngest patient 
             is {abpat['adm_hosp_age'].min():.2f} years old, and the oldest patient is 
             {abpat['adm_hosp_age'].max():.2f} years old. The median age is 
             {abpat['adm_hosp_age'].median():.2f} years.

            The average BMI is {abpat['bmi'].mean():.2f}, with a standard deviation of 
            {abpat['bmi'].std():.2f}. The lowest BMI is {abpat['bmi'].min():.2f}, and 
            the highest BMI is {abpat['bmi'].max():.2f}. The median BMI is 
            {abpat['bmi'].median():.2f}. Missingness is 
            {abpat['bmi'].isna().mean():.2%}.

            The average length is {abpat['length'].mean():.2f} cm, with a standard 
            deviation of {abpat['length'].std():.2f} cm. The shortest patient is 
            {abpat['length'].min():.2f} cm, and the tallest patient is 
            {abpat['length'].max():.2f} cm. The median length is 
            {abpat['length'].median():.2f} cm. Missingness is 
            {abpat['length'].isna().mean():.2%}.

            The average weight is {abpat['weight'].mean():.2f} kg, with a standard 
            deviation of {abpat['weight'].std():.2f} kg. The lightest patient is 
            {abpat['weight'].min():.2f} kg, and the heaviest patient is 
            {abpat['weight'].max():.2f} kg. The median weight is 
            {abpat['weight'].median():.2f} kg. Missingness is 
            {abpat['weight'].isna().mean():.2%}.
            """
        )

        self.logger.debug(abpat.describe().T)

        return abpat

    def load_tables(self) -> dict[str, str]:
        table_list = glob.glob(os.path.join(self.config.directory("raw"), "*"))
        return {
            name: path
            for name, path in zip(
                [p.split("_gz\\")[-1].split(".")[0] for p in table_list], table_list
            )
        }

    def get_patients(self) -> pd.DataFrame:
        patients = td.load(self.tables.get("Patient"))
        patients = patients.rename(
            columns=self.config.settings.patient_selection.admission_table_rename_dict
        )

        rd = self.config.settings.patient_selection.admission_table_rename_dict.values()

        patients_columns = [x for x in patients.columns if x in rd]
        patients = patients[patients_columns]
        patients_rename_dict = {
            "sex": {
                "Man": "male",
                "Vrouw": "female",
            },
        }

        for col in ["dod"]:
            patients[col] = pd.to_datetime(patients[col])

        for col in patients:
            if col in patients_rename_dict.keys():
                patients[col] = (
                    patients[col].map(patients_rename_dict[col]).fillna(patients[col])
                )
                self.logger.debug(col)
                self.logger.debug(
                    [
                        x
                        for x in patients[col].unique()
                        if x not in patients_rename_dict[col].values()
                    ],
                )

        patients["is_sex__male"] = patients["sex"].map({"male": True, "female": False})
        patients["is_sex__female"] = patients["sex"].map({"male": False, "female":
            True})
        patients["is_sex__unknown"] = patients["sex"].isna()

        patients = patients[["pid", "dod", "is_sex__male", "is_sex__female",
                             "is_sex__unknown"]].copy()
        patients.info()
        return patients

    def get_admissions(self) -> pd.DataFrame:
        admissions: pd.DataFrame = td.load(self.tables.get("Opnametraject"))
        admissions = admissions.rename(
            columns=self.config.settings.patient_selection.admission_table_rename_dict
        ).copy()
        return admissions

    def get_icu_admissions(self) -> pd.DataFrame:
        self.logger.info(
            """
            We will create an ICU admissions table which transforms the IC and MC 
            trajectories into one single ICU admission.
            """
        )

        ic_mc_trajectory = td.load(self.tables.get("OpnameDeeltraject"))
        icu_admissions = (
            ic_mc_trajectory.groupby(
                [
                    "Pseudo_id",
                    "ICMCtrajectId",
                ]
            )
            .agg(
                {
                    "StartDatumTijd": "min",
                    "EindDatumTijd": "max",
                }
            )
            .reset_index()
            .rename(
                columns=self.config.settings.patient_selection.admission_table_rename_dict
            )
        )
        print(icu_admissions.info())
        return icu_admissions

    def get_hospital_admissions(self) -> pd.DataFrame:
        rd = self.config.settings.patient_selection.admission_table_rename_dict

        hospital_columns = [x for x in self.admissions.columns if x in rd.values()]
        hospital_admissions = self.admissions[hospital_columns].copy()

        category_dict: dict = (
            self.config.settings.patient_selection.categorical_columns_value_rename_dict
        )

        # Assert that all columns are in the category_dict
        for col in hospital_admissions:
            if col in category_dict.keys():
                hospital_admissions[col] = (
                    hospital_admissions[col]
                    .map(category_dict[col])
                    .fillna(hospital_admissions[col])
                )
                print(
                    col,
                    [
                        x
                        for x in hospital_admissions[col].unique()
                        if x not in category_dict[col].values()
                    ],
                )

        hospital_admissions["adm_hosp_adm_id"] = hospital_admissions.reset_index().index
        print(hospital_admissions.info())

        return hospital_admissions

    def plot_icu_admissions(
        self, abpat: pd.DataFrame, file_name: str | os.PathLike
    ) -> None:
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        sns.histplot(
            (abpat["adm_icu_dis"] - abpat["adm_icu_adm"]) / pd.Timedelta(1, "h") / 24,
            ax=ax[0],
        )
        sns.histplot(
            (abpat["adm_hosp_dis"] - abpat["adm_hosp_adm"]) / pd.Timedelta(1, "h") / 24,
            ax=ax[1],
        )
        ax[0].set_title("ICU - Length of Stay")
        ax[1].set_title("Hospital - Length of Stay")
        ax[0].set_xlabel("Days")
        ax[1].set_xlabel("Days")
        ax[0].set_ylabel("Count")
        ax[1].set_ylabel("Count")

        plt.savefig(os.path.join(self.config.directory("figures"), file_name))
        plt.close()


if __name__ == "__main__":
    patient_selector = PatientSelector(
        config=Config(root="C:\\TADAM\\projects\\abstop")
    )
    patient_selector.run()
