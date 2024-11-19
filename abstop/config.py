"""Specification of project concepts used when filtering and transforming data

This file contains the following classes:
    Config
    PatientSelection
    Microbiology
    AntibioticSelection
    DataLoader
    Settings
"""

import datetime
import os
import shutil
from dataclasses import dataclass

import numpy as np
import pandas as pd


class Config:
    """Configuration of the project.

    This class contains the following methods:
        __init__
        _create_directory_dict
        directory
    """

    def __init__(self, root: str, experiment_name=None):
        self.DIR_ROOT = os.path.expandvars(root)

        if experiment_name is None or experiment_name == "" or not isinstance(experiment_name, str):
            experiment_name = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%dT%H%M%SZ")

        self.experiment_name = experiment_name

        self.optuna_db_uri = f"postgresql+psycopg2://abstop:abstop@CZC122B5GK.workspace1.local:5432/abstop"

        self.directories: dict[str, str] = self._create_directory_dict(root, experiment_name)
        for key, value in self.directories.items():
            os.makedirs(value, exist_ok=True)
        self.settings = Settings()
        self.config_path = os.path.abspath(__file__)
        try:
            shutil.copy(self.config_path, os.path.join(self.directory('results'), 'config.py'))
        except shutil.SameFileError:
            # handle exception when running the autoprognosis pipeline
            pass
        self.optuna_db_uri = f"sqlite:///{os.path.join(self.directory('models'), 'optuna.db')}".replace(os.sep, "/")

    def copy_config_file(self):
        destination = os.path.join(self.directory("results"), 'config.py')
        shutil.copy2(self.config_path, destination)


    @staticmethod
    def _create_directory_dict(root: str, experiment_name=None) -> dict[str, str]:
        _root = os.path.expandvars(root)

        return {
            "root": _root,
            "data": os.path.join(_root, "data"),
            "raw": os.path.join(_root, "data", "raw_export_gz"),
            "processed": os.path.join(_root, "data", "processed"),
            "side_loaded": os.path.join(_root, "data", "side_loaded"),
            "definitions": os.path.join(_root, "abstop", "definitions"),
            "results": os.path.join(_root, "abstop", "results", experiment_name),
            "logs": os.path.join(_root, "abstop", "results", experiment_name, "logs"),
            "figures": os.path.join(_root, "abstop", "results", experiment_name, "figures"),
            "models": os.path.join(_root, "abstop", "results", experiment_name, "models"),
        }

    def directory(self, name: str) -> str:
        """Get the path of a directory in the data directory.

        :param name: Name of the directory.
        :return: Path to the directory.
        """
        _return_name = self.directories.get(name, None)
        if _return_name is None:
            raise ValueError(
                f"Directory name must be one of {self.directories}, not" f" {name}."
            )
        return _return_name

    atc_to_name = {
        'A07AA': 'SDD',
        'C01CA03': 'NORADRENALIN',
        'C02AC01': 'CLONIDINE',
        'J01AA02': 'DOXYCYCLIN',
        'J01AA08': 'MINOCYCLIN',
        'J01AA12': 'TIGECYCLIN',
        'J01CA04': 'AMOXICILLIN',
        'J01CE01': 'PENICILLIN',
        'J01CE05': 'FENETICILLIN',
        'J01CF05': 'FLUCOXACILLIN',
        'J01CR02': 'AMOX+B-LACTAM',
        'J01CR05': 'PIPERACILLIN TAZOBACTAM',
        'J01DB01': 'CEFALEXIN',
        'J01DB03': 'CEFALOTIN',
        'J01DB04': 'CEFAZOLIN',
        'J01DC02': 'CEFUROXIME',
        'J01DC03': 'CEFAMANDOL',
        'J01DD01': 'CEFOTAXIM',
        'J01DD02': 'CEFTAZIDIM',
        'J01DD04': 'CEFTRIAXONE',
        'J01DD14': 'CEFTIBUTEN',
        'J01DD52': 'CEFTAZIDIM AVIBACTAM',
        'J01DH02': 'MEROPENEM',
        'J01DH03': 'ERTAPENEM',
        'J01DH51': 'IMIPENEM CILASTATINE',
        'J01DI02': 'CEFTAROLIN',
        'J01DI54': 'CEFTOLOZANE TAZOBACTAM',
        'J01EA01': 'TRIMETHOPRIM',
        'J01EC02': 'SULFADIAZIN',
        'J01EE01': 'COTRIMOXAZOLE',
        'J01FA09': 'CLARITROMYCIN',
        'J01FA10': 'AZITROMYCIN',
        'J01FF01': 'CLINDAMYCIN',
        'J01GB01': 'TOBRAMYCIN',
        'J01GB03': 'GENTAMICIN',
        'J01GB06': 'AMIKACIN',
        'J01MA02': 'CIPROFLOXACIN',
        'J01MA06': 'NORFLOXACIN',
        'J01MA12': 'LEVOFLOXACIN',
        'J01MA14': 'MOXIFLOXACIN',
        'J01XA02': 'TEICOPLANIN',
        'J01XB01': 'COLISTIN',
        'J01XC01': 'FUSIDIC ACID',
        'J01XD01': 'METRONIDAZOL',
        'J01XE01': 'NITROFURANTOIN',
        'J01XX01': 'FOSFOMYCIN',
        'J01XX08': 'LINEZOLID',
        'J01XX09': 'DAPTOMYCIN',
        'J01XA01': 'VANCOMYCIN',
        'J01ZA01': 'VANCOMYCIN',
        'J04AM05': 'RIFAMPICIN',
        'N01AX10': 'PROPOFOL',
        'N03AE01': 'CLONAZEPAM',
        'N05AD01': 'HALOPERIDOL',
        'N05AH04': 'QUETIAPINE',
        'N05BA01': 'DIAZEPAM',
        'N05BA04': 'OXAZEPAM',
        'N05BA06': 'LORAZEPAM',
        'N05BA08': 'BROMAZEPAM',
        'N05BA12': 'ALPRAZOLAM',
        'N05CD01': 'FLURAZEPAM',
        'N05CD02': 'NITRAZEPAM',
        'N05CD06': 'LORMETAZEPAM',
        'N05CD07': 'TEMAZEPAM',
        'N05CD08': 'MIDAZOLAM',
        'N05CD09': 'BROTIZOLAM',
        'N05CD11': 'LOPRAZOLAM',
        'N05CF01': 'ZOPICLON',
        'N05CF02': 'ZOLPIDEM',
        'N05CM18': 'DEXMEDETOMIDINE',
    }


@dataclass
class PatientSelection:
    """Specification of patient selection.

    This class is used within the Settings class and the PatientSelector class.

    """

    admission_table_rename_dict = {
        "Pseudo_id": "pid",
        "OpnametrajectId": "adm_id",
        # 'OpnameType': 'adm_type', # not relevant for this analysis
        #  (Klinische opname, lots of missing)
        "OpnameBron": "adm_source",
        "OpnameDiagnose": "adm_diag",
        "LeeftijdInJarenOpMomentOpname": "adm_hosp_age",
        "OpnameMoment": "adm_hosp_adm",
        "OntslagMoment": "adm_hosp_dis",
        "ICMCTrajectStartDatumTijd": "adm_icu_adm",
        "ICMCTrajectEindDatumTijd": "adm_icu_dis",
        "OpnameStartDatumTijd": "adm_icu_hosp_adm",
        "OpnameEindDatumTijd": "adm_icu_hosp_dis",
        "ZiekenhuisLocatie": "adm_hosp_loc",
        "OperatiefOfBeschouwend": "adm_hosp_oper",
        "ICMCtrajectId": "adm_icu_adm_id",
        "ICMCTrajectType": "adm_icu_type",
        "ICMCTrajectBron": "adm_icu_source",
        "StartDatumTijd": "adm_icu_adm",
        "EindDatumTijd": "adm_icu_dis",
        "Geslacht": "sex",
        "Overlijdensdatumtijd": "dod",
        "Opnamewijze": "adm_hosp_route",
        "OpnameSpecialismeAfkorting": "adm_hosp_specialism",
        "OpnameSubspecialismeAfkorting": "adm_hosp_subspecialism",
        "OpnamePatientKlasse": "adm_patient_class",
        "IsOpnameStartSEH": "adm_hosp_start_seh_flag",
    }

    categorical_columns_value_rename_dict = {
        "adm_hosp_loc": {
            "Locatie VUmc": "vumc",
            "Locatie AMC": "amc",
            "Erasmus MC": "erasmus",
        },
        "adm_hosp_oper": {
            "Operatief": "surgical",
            "Beschouwend": "medical",
        },
        "adm_hosp_route": {
            "gepland": "elective",
            "ongepland": "unplanned",
            "Niet geregistreerd": "unknown",
            "informatie niet beschikbaar": "unknown",
            "onvrijwillige opname": "unknown",
        },
        "adm_hosp_start_seh_flag": {
            "Ja": True,
            "Nee": False,
        },
    }

    processed_filename = "0_abpat.pkl.gz"


@dataclass
class Microbiology:
    """Specification of microbiology data.

    This class is used within the Settings class and the MicrobiologyPreprocessor class.
    """

    groupings_filename: str | os.PathLike = "microbe_groupings.csv"
    df_side_loaded_filename: str | os.PathLike = "MicrobiologieData_gelabeld.tsv"
    df_labtrain_filename: str | os.PathLike = "LabTrainMMI.csv.pkl.gz"

    df_column_rename_dict = {
        "monsternummer": "sample_id",
        "afnamedatum": "sample_date",
        "patientID_pseudo": "pid",
        "databron": "mic_data_source",
        "vakgroep": "group_code",
        "afdeling": "department",
        "bepaling_code": "result_code",
        "bepaling_naam": "result_name",
        "materiaal_description": "sample_description",
        "materiaal_catCustom": "sample_catcustom",
        "kweekdoel": "culture_reason",
        "uitslag_overall": "result_overall",
        "uitslag_generieketekst": "result_text",
        "isolaatnummer": "isolate_number",
        "microbe_spotf": "microbe_spotf",
        "microbe_ID": "microbe_id",
        "microbe_kingdom": "microbe_kingdom",
        "microbe_phylum": "microbe_phylum",
        "microbe_order": "microbe_order",
        "microbe_family": "microbe_family",
        "microbe_genus": "microbe_genus",
        "microbe_species": "microbe_species",
        "microbe_short": "microbe_short",
        "microbe_gram": "microbe_gram",
        "microbe_catCustom": "microbe_catcustom",
        "ABstop_onderzoek": "abstop_project",
        "Pseudo_id": "pid",
    }

    sample_description_value_rename_dict = {
        "Bloedkweek": "blood",
        "Sputum": "sputum",
        "Anus": "anus",
        "Urine": "urine",
        "uitstrijk Keel": "swab_throat",
        "Uitstrijk-Keel": "swab_throat",
        "Uitstrijk-Pus": "swab_pus",
        "Drainvocht/-pus": "drain_pus",
        "Cathether/Draintip": "tip",
        "Broncho-alveolaire lavage": "sputum",
        "Tip Catheter/Tube": "tip",
        "Biopt": "biopsy",
        "Punctaat": "punctate",
        "uitstrijk Wond": "swab_pus",
        "Overig": "other",
        "Bronchusspoelsel": "sputum",
        "Ascitesvocht": "ascites",
        "Liquor": "liquor",
        "Uitstrijk-Neus": "swab_nose",
        "Pleurapunctaat": "punctate_pleural",
        "uitstrijk Neus": "swab_nose",
        "Faeces": "faeces",
        "uitstrijk Huid": "drop",
        "Lichaamsvreemd gerelateerd materiaal": "drop",
        "Gal": "gall",
        "Tracheaspoelsel": "sputum",
        "Mondholte": "mouth",
        "Uitstrijk-Cervix": "drop",
        "Uitstrijk Overig": "swab_other",
        "Uitstrijk Insteekopening": "swab_insertion",
        "Pericardvocht": "pericard_fluid",
        "Vagina/vulva": "drop",
        "Gewricht (kunstgewricht)": "drop",
        "Gewrichtspunctaat": "drop",
        "Peritoneaalvocht CAPD": "drop",
        "Uitstrijk-Oor": "drop",
        "Blaasjesvocht": "vesicular_fluid",
        "Uitstrijk-Overig": "swab_other",
        "Nasopharynx wat": "swab_nose",
        "Oog": "drop",
        "Oksel uitstrijk": "drop",
        "Urethra": "drop",
        "Placenta/vliezen": "drop",
        "uitstrijk Lies": "swab_groin",
        "Uitstrijk Rectovaginaal": "drop",
        "Aanhoestwat": "swab",
        "Navel": "drop",
        "Uitstrijk-Insteekopening": "swab_insertion",
        "Bloedkweek PED flesje": "blood",
        "Beenmerg": "drop",
        "Sinus": "drop",
        "Sectiemateriaal weefsel": "drop",
        "Uitstrijk-Ulcus": "swab_pus",
    }

    hematology_department_dict = {
        "VUMC HEMATOLOGIE": "hematology",
        "VUMC 3B HEM": "hematology",
        "AMC POLI HEMATOLOGIE": "hematology",
        "VUMC Polikliniek hematologie": "hematology",
        "VUMC DAGBEHANDELING HEMATOLOGIE": "hematology",
    }


@dataclass
class AntibioticSelection:
    """Specification of antibiotic selection.

    This class is used within the Settings class and the AntibioticSelector class.
    """

    ab_raw_filename: str | os.PathLike = "MedicatieToediening.csv.pkl.gz"
    ab_processed_filename: str | os.PathLike = "1_medication_antibiotics.pkl.gz"
    ab_med_freq_filename: str | os.PathLike = "1_medication_frequencies.csv"
    ab_med_definitions_filename: str | os.PathLike = "medication_selection.csv"
    mic_processed_filename: str | os.PathLike = "0_mic.pkl.gz"
    lab_raw_filename: str | os.PathLike = "Labuitslag.csv.pkl.gz"

    cotrim_dose_dict = {
        "COTRIMOXAZOL 96 INFOPL CONC 16/80MG/ML AMP 5ML": 480,
        "COTRIMOXAZOL 480 TABLET 80/400MG": 480,
        "COTRIMOXAZOL 960 TABLET 160/800MG": 960,
        "COTRIMOXAZOL 48 SUSP ORAAL 8/40MG/ML": 480,
        "COTRIMOXAZOL KORTLOPEND INFUUS >480 MG EN <=960 MG IN NACL": 480,
    }

    tdm_search_items: tuple = (
        "vanco",
        "genta",
    )

    lab_column_rename_dict = {
        "Pseudo_id": "pid",
        "Bepaling": "test_name",
        "UitslagNumeriek": "result_numeric",
        "UitslagOpmerking": "result_comment",
        "UitslagDatumtijd": "result_timestamp",
        "MateriaalAfnameDatumTijd": "sample_timestamp",
    }

    tdm_test_name_dict = {
        "VANCOMYCIN": "VANCOMYCIN",
        "Vancomycine": "VANCOMYCIN",
        "GENTAMICIN": "GENTAMICIN",
        "Gentamicine": "GENTAMICIN",
        "Vancomycine vrije fractie": "drop",
    }

    tdm_positive_values = {
        "VANCOMYCIN": 20,
        "GENTAMICIN": 1,
    }

    series_time_limit = 26.4  # hours

    ab_column_rename_dict = {
        "Pseudo_id": "pid",
        "VoorschriftId": "prescription_id",
        "ToedieningsDatumTijd": "timestamp",
        # 'ToedieningsDatum': 'ToedieningsDatum',
        # 'ToedieningsTijd': 'ToedieningsTijd',
        "ToedieningsStatus": "status",
        "Inloopsnelheid": "rate",
        "InloopsnelheidEenheid": "rate_unit",
        "ToegediendeHoeveelheid": "dose",
        "ToegediendeHoeveelheidEenheid": "dose_unit",
        "ToedieningsRoute": "route",
        # 'ToedieningsRouteCode': 'ToedieningsRouteCode',
        "MedicatieArtikelNaam": "original_name",
        # 'MedicatieArtikelCode': 'MedicatieArtikelCode',
        # 'IngredienttypeNaam': 'IngredienttypeNaam',
        # 'IngredienttypeCode': 'IngredienttypeCode',
        "ATCKlasseCode": "atc",
        # 'ATCKlasse': 'ATCKlasse',
        "FarmaceutischeKlasse": "class_pharmaceutical",
        # 'FarmaceutischeSubklasse': 'FarmaceutischeSubklasse',
        "TherapeutischeKlasse": "class_therapeutic",
        # 'WerkplekCode': 'WerkplekCode',
        # 'WerkplekAfkorting': 'WerkplekAfkorting',
        # "Werkplek": "Werkplek",
        # 'Ziekenhuislocatie': 'Ziekenhuislocatie',
        "ZiekenhuislocatieCode": "hospital_location",
        "ToedieningsReden": "status_description",
        # 'ToedieningsRedenCode': 'ToedieningsRedenCode',
        "MedicatieOpmerking": "medication_comment",
        # 'Bron': 'Bron',
        # 'DCMVerverstDatumTijd': 'DCMVerverstDatumTijd',
        # 'uitgifteDT': 'uitgifteDT',
    }

    ab_column_type_dict = {
        "pid": "int",
        "prescription_id": "int",
        "timestamp": "datetime64[ns]",
        "status": "category",
        "rate": "float",
        "rate_unit": "category",
        "dose": "float",
        "dose_unit": "category",
        "route": "category",
        "original_name": "category",
        "atc": "category",
        "class_pharmaceutical": "category",
        "class_therapeutic": "category",
        "hospital_location": "category",
        "status_description": "category",
        "medication_comment": "category",
    }

    ab_column_final_type_dict = {
        "class_pharmaceutical": "category",
        "class_therapeutic": "category",
        "original_atc": "category",
        "original_name": "category",
        "route": "category",
        "pid": "int",
        "prescription_id": "int",
        "timestamp": "datetime64[ns]",
        "status": "category",
        "rate": "float",
        "rate_unit": "category",
        "dose": "float",
        "dose_unit": "category",
        "Werkplek": "category",
        "hospital_location": "category",
        "status_description": "category",
        "medication_comment": "category",
        "is_sdd": "bool",
        "atc": "category",
        "name": "category",
        "is_prophylaxis": "bool",
        "is_pdd": "bool",
        "timestamp_diff_prev": "timedelta64[ns]",
        "timestamp_diff_next": "timedelta64[ns]",
        "timestamp_diff_prev_more_than_26h": "bool",
        "atc_sub_series_id": "int",
        "atc_series_id": "int",
        "record_id": "int",
        "tdm_flag": "int",
        "is_last_in_atc_group": "bool",
        "timestamp_diff_next_more_than_26h": "bool",
        "tdm_flag_change": "bool",
        "timestamp_diff_prev_tdm": "bool",
        "tdm_sub_series_id": "int",
        "tdm_series_id": "int",
        "tdm_series_id_size": "int",
        "dose_hourly": "float",
        "tdm_series_id_record_number": "int",
        "timestamp_tdm_adjusted": "datetime64[ns]",
        "ab_series_id": "int",
    }

    columns_to_clean = [
        "class_pharmaceutical",
        "class_therapeutic",
        "atc",
        "original_name",
    ]

    columns_for_medication_frequency = [
        "class_pharmaceutical",
        "class_therapeutic",
        "atc",
        "original_name",
        "route",
    ]

    freq_columns_to_add = [
        "include",  # 0 | 1
        "is_sdd",  # 0 | 1
        "target_atc",  # J01DB04 etc.
        "target_name",  # CEFAZOLIN
        "target_group",  # unused
        "check",  # label for manual check
        "comment",  # comment for manual check
    ]

    medication_definitions_merge_cols = [
        "class_pharmaceutical",
        "class_therapeutic",
        "atc",
        "original_name",
        "route",
    ]

    definition_rename_dict = {
        "atc": "original_atc",
        "name": "original_name",
        "target_name": "name",
        "target_atc": "atc",
    }

    definition_drop_columns = {
        "include",
        "target_group",
    }

    ab_status_dict = {
        "Toegediend": "keep",
        "Nieuwe zak": "keep",
        "Wijziging inloopsnelheid": "keep",
        "Zie alternatief": "drop",  # all nan
        "MTR-vasthouden": "drop",  # all nan
        "MTR-vasthouden ongedaan maken": "drop",  # all nan
        "Gestopt": "keep",
        "Gemist": "drop",
        "Geannuleerde invoer": "drop",
        "Gepland": "drop",
        "Opnieuw gestart": "keep",
        "Automatisch vastgehouden": "drop",
        "Toegediend door patient": "keep",
        "Voeding gegeven": "keep",
        "Gepauzeerd": "keep",
        "Toegediend door familielid": "keep",
        "Anesthesie controle voor bloed toediening": "drop",
        "Gestopt door anesthesie": "keep",
        "Continueren huidige zak/spuit": "keep",
        "Overname OK": "keep",
        "Gegeven tijdens downtime": "keep",
        "Pleister aangebracht": "keep",
        "Met verlof": "drop",
        "Terug van verlof": "drop",
        "Gestart tijdens downtime": "keep",
        "Bolus": "keep",
        "Tijdelijk opgeslagen": "keep",
        "Toegediend tijdens verlof": "drop",
        "Overdracht": "keep",
        "Pleister verwijderd": "drop",
        "Gegeven door patiënt": "keep",
        "Voeding gegeven door familielid": "keep",
        "Deel toediening": "keep",
        "Bolus vanuit spuitenpomp": "keep",
        "Meegegeven aan patiënt": "keep",
    }


@dataclass
class DataLoader:
    """Specification of data loading.

    This class is used within the Settings class and the Measurements class.
    """

    patients = {
        "filename": "0_abpat.pkl.gz",
        "path": "processed",
    }

    merge_medication_with_definitions = {
        "merge_kwargs": {
            "on": AntibioticSelection().medication_definitions_merge_cols,
            "how": "left",
        },
        "rename": {
            "atc": "original_atc",
            "name": "original_name",
            "target_name": "name",
            "target_atc": "atc",
            "target_group": "group",
        },
        "" "drop": {},
        "retype": {},
    }

    medication = {
        "filename": "MedicatieToediening.csv.pkl.gz",
        "rename": AntibioticSelection().ab_column_rename_dict,
        "retype": AntibioticSelection().ab_column_type_dict,
        "keep": {
            "status": {
                k
                for k, v in AntibioticSelection().ab_status_dict.items()
                if v == "keep"
            }
        },
        "drop_nan_any": {"pid", "timestamp", "original_name"},
        "merge_keep": {
            "filename": "medication_selection.csv",
            "on": AntibioticSelection().medication_definitions_merge_cols,
            "keep": "is_medication",
            "add_columns": [
                "concentration",
                "dose_per_unit",
                "target_atc",
                "target_name",
                "target_group",
            ],
        },
        "merge_patients": {
            "on": "pid",
            "data_timestamp": "timestamp",
            "patients_lower": "adm_icu_adm",
            "patients_upper": "adm_icu_dis",
        },
        "allowed_categories": {
            "NORADRENALIN": {
                "dose_unit": [
                    "microgr/kg/min",
                    "microgr/uur",
                    #'mg/ml',
                    "microgr/min",
                    "microgr/kg/uur",
                    #'mg',
                    "microgr/kg/24uur",
                    #'ml/uur',
                    "microgr",
                    "microgr/24uur",
                    "mg/uur",
                ]
            },
        },
        "convert_weight_based": {
            "predicted_weight": [
                "NORADRENALIN",
            ],
            "adjusted_weight": [
                "PROPOFOL",
            ],
        },
        "continuous_records": {
            "names": [
                "PROPOFOL INJ.EMULSIE WWSP 1000MG=50ML",
                "SPUITENPOMP",
            ],
        },
        "remove_rate_based": {
            "NORADRENALIN": {
                "min": 0.11,
                # "max": 80,
            },
            "PROPOFOL": {
                "min": 0.11,
                # "max": 0.5,
            },
        },
        "concentration_correction_upper_limits": {
            "NORADRENALIN": {
                "min": 0.001,
                "max": 0.5,
                "bins": {
                    0: 0,
                    0.06: 0.02,
                    0.15: 0.1,
                    0.25: 0.2,
                },
            },
        },
        "unit_doses": [
            # "mg",
            # "microgr/kg/min",
            # "microgr/uur",
            # "ml/uur",
            # "mg/kg/uur",
            # "microgr",
            # "ml",
            # "g",
            # "mg/uur",
            # "IE/uur",
            # "microgr/kg/uur",
            # "IE",
            # "microgr/ml",
            "sachet",
            # "mmol/uur",
            # "microgr/min",
            # "mg/ml",
            # "mg/24uur",
            "tablet",
            # "ml/24uur",
            # "IE/kg/uur",
            # "mmol",
            "druppel",
            # "g/24uur",
            "pleister",
            # "mg/kg",
            "stuk",
            "zetpil",
            # "IE/24uur",
            # "mg/min",
            # "IE/ml",
            "capsule",
            "spray",
            "klysma",
            # "microgr/24uur",
            # "ml/dag",
            "inhalatie",
            # "mmol/24uur",
            # "mg/kg/24uur",
            "flacon",
            # "microgr/kg/24uur",
            "dragee",
            # "IE/dag",
            # "mmol/ml",
            # "IE/min",
            "mgI",
            # "cm",
            # "ml/min",
            # "mg/kg/dag",
            # "mg/dag",
            # "milj eenheden",
            # "ml/kg/uur",
            # "E",
            # "milj IE/24uur",
            # "applicator",
            # "mg/kg/min",
            # "nanogram/kg/min",
            # "g/uur",
            "fles",
            # "mmol/dag",
            # "g/dag",
            "dosis",
            "maatlepel",
            "mg/m2",
            # "IE/kg/min",
            # "mmol/min",
            # "g/100 ml",
            # "l/uur",
            # "g/min",
            "container",
            # "mg/m2/dag",
            # "l/dag",
            # "mmol/kg/uur",
            # "g N",
            "ampul",
            # "microgr/dag",
            # "microgr/do",
            # "milj eenheden/kg/dag",
            # "mg/g",
            # "milj eenheden/dag",
            # "ml/kg/24uur",
            "film",
            # "mmol/kg",
            # "microgr/kg/dag",
            # "microgr/kg",
            "tabletje",
            # "mcg",
            "zak - niet gebruiken",
            "tube",
            "wwsp",
            # "l",
        ],
        "ml_doses": ["ml", "ml/uur", "ml/hr"],
        "benzo_equivalents": {
            ## Benzo Equivalence Calculation
            ## Source: https://www.benzo.org.uk/bzequiv.htm |
            ## https://deprescribe.web.unc.edu/wp-content/uploads/sites/20194/2020/04/Benzo-Equivalency-Table_UNC.pdf
            # factor: the equivalent amount of a drug to give in contrast to the
            # amount given as diazepam. E.g. if 10mg diazepam is given, this is
            # equivalent to 0.5mg alprazolam, therfore, the conversion factor is:
            # 0.5/10=0.05. To calculate from one to another, we can use the factors
            # in a chained equation:
            # 1mg lorazepam to oxazepam: 1mg / (1/10) * (20/10) = 20mg oxazepam
            "ALPRAZOLAM": {
                "factor": 1 / 10,  # 0.5 - 1
            },
            "BROMAZEPAM": {
                "factor": 5 / 10,  # 5-6
            },
            "BROTIZOLAM": {
                "factor": 0.25 / 10,  # 20-40
            },
            "CHLORDIAZEPOXIDE": {
                "factor": 20 / 10,
            },
            "CLOBAZAM": {
                "factor": 20 / 10,
            },
            "CLONAZEPAM": {
                "factor": 1 / 10,  # 0.5 - 1
            },
            "CLORAZEPATE": {
                "factor": 15 / 10,
            },
            "DIAZEPAM": {
                "factor": 10 / 10,
            },
            "FLURAZEPAM": {
                "factor": 15 / 10,  # 15-30
            },
            "LOPRAZOLAM": {
                "factor": 1 / 10,
            },
            "LORAZEPAM": {
                "factor": 1 / 10,  # 1-2
            },
            "LORMETAZEPAM": {
                "factor": 1 / 10,
            },
            "MIDAZOLAM": {
                "factor": 2.7 / 10,  # 2-4
            },
            "NITRAZEPAM": {
                "factor": 10 / 10,
            },
            "OXAZEPAM": {
                "factor": 30 / 10,  # 15-30
            },
            "TEMAZEPAM": {
                "factor": 20 / 10,  # 20-30
            },
            "ZOLPIDEM": {
                "factor": 20 / 10,
            },
            "ZOPICLON": {
                "factor": 15 / 10,
            },
        },
        "sort_post_processing": {
            "by": ["pid", "timestamp", "dose"],
            "ascending": [True, False, False],
        },
        "transform_overlapping_to_discrete_series": {
            "timestamp_start": "timestamp_start",
            "timestamp_end": "timestamp_end",
            "value": "dose",
            "group_by": ["pid", "target_name"],
            "keep": "sum",
        },
        "join_adjoining_records": {
            "timestamp_start": "timestamp_start",
            "timestamp_end": "timestamp_end",
            "values": ["dose"],
            "group_by": ["pid", "target_name"],
        },
        "impute_missing_as_zero": {
            "group_by": ["pid"],
            "data_start": "timestamp_start",
            "data_end": "timestamp_end",
            "data_var": "target_name",
            "data_val": "dose",
            "merge_on": ["pid"],
            "windows_start": "adm_icu_adm",
            "windows_end": "adm_icu_dis",
            "impute_window": "combined",
            "discrete_method": "sum",
        },
    }

    microbiology = {
        "filename": "0_mic.pkl.gz",
        "path": "processed",
        "rename": {
            "pid": "pid",
            "sample_date": "sample_date",
            "microbe_gram": "mic__gram",
            "groupings": "mic__group",
            "pathogenicity": "mic__pathgen",
            "is_sdd": "mic__sdd",
            "is_pdd": "mic__pdd",
            "is_icu": "mic__icu",
            "is_positive": "is_positive",
            "is_blood": "is_blood",
            "is_urine": "is_urine",
            "is_sputum": "is_sputum",
            "is_tip": "is_tip",
            "sdd_positive": "mic__sdd_positive",
        },
        "retype": {
            "pid": "int16",
            "sample_date": "datetime64[ns]",
            "mic__gram": "category",
            "mic__group": "category",
            "mic__pathgen": "category",
            "mic__sdd": "bool",
            "mic__pdd": "bool",
            "mic__icu": "bool",
            "is_positive": "bool",
            "is_blood": "bool",
            "is_urine": "bool",
            "is_sputum": "bool",
            "is_tip": "bool",
            "mic__sdd_positive": "bool",
        },
        "drop_nan_any": {"pid", "sample_date"},
        "drop_nan_all": {"mic__group", "is_positive", "mic__gram"},
        "remap": {
            "mic__gram": {
                "Gram-positive": "positive",
                "Gram-negative": "negative",
            },
        },
        "sort": {
            "by": ["pid", "sample_date"],
            "ascending": [True, True],
        },
        "set_timestamp_column": {
            "method": "fillna",
            "columns": ["sample_date"],
            "return_column_name": "timestamp",
            "adjustments": {
                "sample_date": pd.Timedelta(48, "h"),
            },
        },
        "combine_columns": {
            "mic__is_positive__blood": {
                "method": "bool_and",
                "columns": ["is_positive", "is_blood"],
                "drop": False,
            },
            "mic__is_positive__urine": {
                "method": "bool_and",
                "columns": ["is_positive", "is_urine"],
                "drop": False,
            },
            "mic__is_positive__sputum": {
                "method": "bool_and",
                "columns": ["is_positive", "is_sputum"],
                "drop": False,
            },
            "mic__is_positive__tip": {
                "method": "bool_and",
                "columns": ["is_positive", "is_tip"],
                "drop": False,
            },
        },
        "filter_on_combined_columns": {
            # this was created to filter is_positive__xx, but that filtered out all
            # negative results as well!
            "any": ["is_blood", "is_urine", "is_sputum", "is_tip"],
        },
        "melt": {
            "id_vars": ["pid", "timestamp"],
            "subset": [
                "mic__gram",
                "mic__group",
                "mic__pathgen",
                "mic__sdd_positive",
                "mic__is_positive__blood",
                "mic__is_positive__urine",
                "mic__is_positive__sputum",
                "mic__is_positive__tip",
                "is_blood",
                "is_sputum",
                "is_tip",
                "is_urine",
            ],
            "drop_nan_after_melt": True,
        },
        "post_processing": {
            "rename": {
                "variable": {
                    "is_blood": "mic__is_blood",
                    "is_sputum": "mic__is_sputum",
                    "is_tip": "mic__is_tip",
                    "is_urine": "mic__is_urine",
                },
            },
            "retype": {
                "pid": "int16",
                "timestamp_start": "datetime64[ns]",
                "timestamp_end": "datetime64[ns]",
                "variable": "category",
                "value": "object",
            },
            "sort_post_processing": {
                "by": ["pid", "timestamp", "variable", "value"],
                "ascending": [True, False, True, False],
            },
            "add_timestamps": {
                "rename": {
                    "timestamp": "timestamp_start",
                },
                "adjustments": {
                    "timestamp_end": {
                        "method": "add",
                        "column": "timestamp_start",
                        "value": pd.Timedelta(24, "h"),
                    },
                },
            },
        },
    }

    lab = {
        "filename": "Labuitslag.csv.pkl.gz",
        "rename": {
            "Pseudo_id": "pid",
            "Bepaling": "test_name",
            "BepalingCode": "test_code",
            "UitslagNumeriek": "result_numeric",
            "UitslagEenheid": "result_unit",
            "UitslagOpmerking": "result_comment",
            "UitslagDatumtijd": "result_timestamp",
            "MateriaalAfnameDatumTijd": "sample_timestamp",
            "afnamelocatie": "sample_location",
        },
        "retype": {
            "pid": "int16",
            "test_name": "category",
            "test_code": "category",
            "result_numeric": "float32",
            "result_comment": "category",
            "result_timestamp": "datetime64[ns]",
            "sample_timestamp": "datetime64[ns]",
        },
        "drop_nan_any": {
            "pid",
        },
        "drop_nan_all": {"result_numeric", "result_timestamp"},
        "remap": {
            "sample_location": {
                "Arterieel": "arterial",
                "Veneus": "venous",
                "Gemengd veneus": "venous__mixed",
                "Mixed Veneus": "venous__mixed",
                "{<GMO}": "drop",
                "Vragen om afname informatie": "unknown",
                "{<VVE}": "drop",
                "Capillair": "capillary",
                "Onbekend": "unknown",
                "Na Filter": "post-filter",
                "Arterielijn": "arterial",
                "{<TWM}": "drop",
                "Capillair vinger": "capillary__finger",
                "MBO": "unknown",
                "{<VIM}": "drop",
                "Voor Filter": "pre-filter",
                "{<VEN}": "venous",
                "{<GMA}": "insufficient_material",
                "M.B.O": "drop",
                "{<VMV}": "unknown",
                "{<GES}": "drop",
                "{<VMA}": "unknown",
                "Navel arterieel": "drop",
                "Navel veneus": "drop",
                "Nevirapine:": "drop",
                "{<ART}": "arterial",
                "{<ZOP}": "unknown",
                "Na-Filter": "post-filter",
                "gem.ven": "venous__mixed",
                "Centraal veneus": "venous__central",
                "Lumbaalpunctie": "drop",
                "arteriel": "arterial",
                "gem veneus": "venous__mixed",
                "{<MVG}": "drop",
                "verdunning?": "unknown",
                "{<MTWM}": "drop",
                "gemend veneus": "venous__mixed",
                "{<GEMVEN}": "venous__mixed",
                "Vragen om afname informatie;Arterieel": "arterial",
                "veneuse": "venous",
                "Efavirenz:": "drop",
                "Arterieel;Veneus": "unknown",
                "{<NTB}": "drop",
                "Arterielijn;Veneus": "unknown",
                "{<ONB}": "unknown",
                "Drain Wond": "drop",
                "{<TWM} ?": "drop",
                "Drain Thorax": "drop",
                "Navelstreng Arterieel": "drop",
                "Navelstreng Veneus": "drop",
                "Bloed": "unknown",
                "Keel": "drop",
                "Lopinavir:": "drop",
                "M.B.O.": "unknown",
                "Pleuravocht, Links, Locatie: ...": "drop",
            }
        },
        "keep": {
            "sample_location": [
                "arterial",
                "venous",
                "venous__mixed",
                "venous__central",
                "capillary",
                "capillary__finger",
                "pre-filter",
                "post-filter",
                pd.NA,
                np.nan,
            ],
        },
        "merge_keep": {
            "filename": "lab_selection.csv",
            "rename": {
                "Bepaling": "test_name",
                "BepalingCode": "test_code",
            },
            "on": {"test_name", "test_code"},
            "keep": "include",
            "add_columns": [
                "group",
            ],
        },
        "unit_conversion": {
            "column_unit": "result_unit",
            "column_value": "result_numeric",
            "factors": {
                "kPa": 7.500062,
            },
            "rename_units": {
                "kPa": "mmHg",
            },
        },
        "set_timestamp_column": {
            "method": "fillna",
            "adjustments": {"sample_timestamp": pd.Timedelta(6, "h")},
            "columns": ["result_timestamp", "sample_timestamp"],
            "return_column_name": "timestamp",
        },
        "adjust_variables": {
            "concat": {
                "filters": [
                    {
                        "group": ["PH", "PO2", "PCO2", "BICARBONATE"],
                    }
                ],
                "sep": "__",
                "columns": ["group", "sample_location"],
                "target_column": "group",
            }
        },
        "filter_values": {
            "mappers": {
                "group": {
                    "BILIRUBIN": {"min": 0, "max": 1000},
                    "CRP": {"min": 0, "max": 1000},
                    "GENTAMICIN": {"min": 0, "max": 1000},
                    "GLUCOSE": {"min": 0, "max": 100},
                    "HB": {"min": 0, "max": 20},
                    "KREAT": {"min": 0, "max": 2000},
                    "LACTATE": {"min": 0, "max": 50},
                    "LEUKOCYTES": {"min": 0, "max": 300},
                    "PCO2__arterial": {"min": 1, "max": 250},
                    "PCO2__capillary": {"min": 1, "max": 250},
                    "PCO2__nan": {"min": 1, "max": 250},
                    "PCO2__venous": {"min": 1, "max": 250},
                    "PCO2__venous__mixed": {"min": 1, "max": 250},
                    "PH__arterial": {"min": 6, "max": 9},
                    "PH__capillary": {"min": 6, "max": 9},
                    "PH__nan": {"min": 6, "max": 9},
                    "PH__venous": {"min": 6, "max": 9},
                    "PH__venous__mixed": {"min": 6, "max": 9},
                    "BICARBONATE__arterial": {"min": 1, "max": 100},
                    "BICARBONATE__capillary": {"min": 1, "max": 100},
                    "BICARBONATE__nan": {"min": 1, "max": 100},
                    "BICARBONATE__venous": {"min": 1, "max": 100},
                    "BICARBONATE__venous__mixed": {"min": 1, "max": 100},
                    "PO2__arterial": {"min": 10, "max": 700},
                    "PO2__capillary": {"min": 10, "max": 700},
                    "PO2__nan": {"min": 10, "max": 700},
                    "PO2__venous": {"min": 10, "max": 700},
                    "PO2__venous__mixed": {"min": 10, "max": 700},
                    "POTASSIUM": {"min": 0, "max": 20},
                    "SODIUM": {"min": 0, "max": 200},
                    "TROMBOCYTES": {"min": 0, "max": 1000},
                    "VANCOMYCIN": {"min": 0, "max": 200},
                    "CHLORIDE": {"min": 0, "max": 200},
                }
            },
            "value_column": "result_numeric",
        },
        "return": {
            "rename": {
                "group": "variable",
                "result_numeric": "value",
            },
            "columns": ["pid", "timestamp", "variable", "value"],
        },
    }

    hemodynamics = {
        "filename": "MetingCirculatieHemodynamiek.csv.pkl.gz",
        "rename": {
            "Pseudo_id": "pid",
            "MeetMoment": "timestamp",
            "HFmonitor": "heart_rate_ecg",
            "Polsmonitor": "heart_rate_pulse",
            "ABPsd": "abp_sd_0",
            "ABPm": "abp_m_0",
            "ABPsd2": "abp_sd_1",
            "ABPm2": "abp_m_1",
            "NIBPsd": "nibp_sd_0",
            "NIBPm": "nibp_m_0",
        },
        "drop_nan_any": {"pid", "timestamp"},
        "retype": {
            "pid": "int16",
            "timestamp": "datetime64[ns]",
            "heart_rate_ecg": "float32",
            "heart_rate_pulse": "float32",
            "abp_sd_0": "object",
            "abp_m_0": "float32",
            "abp_sd_1": "object",
            "abp_m_1": "float32",
            "nibp_sd_0": "object",
            "nibp_m_0": "float32",
        },
        "split_columns": {
            "abp_sd_0": ["abp_sys_0", "abp_dia_0"],
            "abp_sd_1": ["abp_sys_1", "abp_dia_1"],
            "nibp_sd_0": ["nibp_sys_0", "nibp_dia_0"],
        },
        "filter_columns": {
            "abp_sys_0": {"min": 20, "max": 290},
            "abp_dia_0": {"min": 20, "max": 200},
            "abp_m_0": {"min": 20, "max": 290},
            "abp_sys_1": {"min": 20, "max": 290},
            "abp_dia_1": {"min": 20, "max": 200},
            "abp_m_1": {"min": 20, "max": 290},
            "nibp_sys_0": {"min": 20, "max": 290},
            "nibp_dia_0": {"min": 20, "max": 200},
            "nibp_m_0": {"min": 20, "max": 290},
            "heart_rate_ecg": {"min": 10, "max": 300},
            "heart_rate_pulse": {"min": 10, "max": 300},
        },
        "combine_columns": {
            "abp_sys": {
                "method": "max",
                "columns": ["abp_sys_0", "abp_sys_1"],
            },
            "abp_dia": {
                "method": "max",
                "columns": ["abp_dia_0", "abp_dia_1"],
            },
            "abp_mean": {
                "method": "max",
                "columns": ["abp_m_0", "abp_m_1"],
            },
            "bp_sys": {
                "method": "max",
                "columns": ["abp_sys", "nibp_sys_0"],
            },
            "bp_dia": {
                "method": "max",
                "columns": ["abp_dia", "nibp_dia_0"],
            },
            "bp_mean": {
                "method": "max",
                "columns": ["abp_mean", "nibp_m_0"],
            },
            "heart_rate": {
                "method": "fillna",
                "columns": ["heart_rate_ecg", "heart_rate_pulse"],
            },
        },
        "melt": {
            "id_vars": ["pid", "timestamp"],
        },
        "drop_nan_after_melt": True,
    }

    respiratory_ventilator = {
        "filename": "MetingBeademingsMeetwaarden.csv.pkl.gz",
        "rename": {
            "Pseudo_id": "pid",
            "MeetMoment": "timestamp",
            "SpO2vent": "spo2_vent",
            "FiO2": "fio2_vent",
            "AFtotaal": "resp_rate_total",
            "AMVexp": "amv_exp",
        },
        "drop_nan_any": {"pid", "timestamp"},
        "retype": {
            "pid": "int16",
            "timestamp": "datetime64[ns]",
            "spo2_vent": "float32",
            "fio2_vent": "float32",
            "resp_rate_total": "float32",
            "amv_exp": "float32",
        },
        "filter_columns": {
            "spo2_vent": {"min": 0, "max": 100},
            "fio2_vent": {"min": 0, "max": 100},
            "resp_rate_total": {"min": 0, "max": 100},
            "amv_exp": {"min": 0, "max": 100},
        },
    }

    respiratory_default = {
        "filename": "MetingRespiratieAlgemeen.csv.pkl.gz",
        "rename": {
            "Pseudo_id": "pid",
            "MeetMoment": "timestamp",
            "AFMonitor": "resp_rate_monitor",
            "SpO2": "spo2",
            "O2LperMin": "o2_flow",
            "O2Toedieningssysteem": "o2_device",
            "FiO2Percentage": "fio2_other",
            "NebulizerFlowO2LperMin": "o2_flow_nebulizer",
        },
        "remap": {
            "o2_device": {
                "Zuurstofbril": "nasal_canula",
                "Geen": "none",
                "Kunstneus": "swedish_nose",
                "Spreekklep": "speaking_valve",
                "Highflow systeem": "highflow",
                "Zuurstofmasker": "mask",
                "anders (opmerking)": "other",
                "Non-rebreathing masker": "non-rebreathing_mask",
                "Nebulizer + zuurstofbril": "nebulizer",
                "Zuurstofslang nasaal": "nasal_canula",
                "Nebulizer": "nebulizer",
                "CPAP masker / Boussignac": "cpap",
                "Aangeblazen": "other",
                "Vernevelmasker": "nebulizer",
                "Ballon-masker": "bag_mask",
                "T-stuk": "other",
                "Prongs": "other",
            },
        },
        "drop_nan_any": {"pid", "timestamp"},
        "retype": {
            "pid": "int16",
            "timestamp": "datetime64[ns]",
            "resp_rate_monitor": "float32",
            "spo2": "float32",
            "o2_flow": "float32",
            "o2_device": "float32",
            "fio2_other": "float32",
            "o2_flow_nebulizer": "float32",
        },
        "filter_columns": {
            "resp_rate_monitor": {"min": 0, "max": 100},
            "spo2": {"min": 0, "max": 100},
            "o2_flow": {"min": 0, "max": 100},
            "fio2_other": {"min": 0, "max": 100},
        },
        "encode_columns": {
            "o2_device": {
                "method": "map",
                "values": {
                    "nasal_canula": 1,
                    "none": 2,
                    "swedish_nose": 3,
                    "speaking_valve": 4,
                    "highflow": 5,
                    "mask": 6,
                    "other": 7,
                    "non-rebreathing_mask": 8,
                    "nebulizer": 9,
                    "cpap": 10,
                    "bag_mask": 11,
                },
            },
        },
    }

    respiratory_merge = {
        "on": ["pid", "timestamp"],
        "combine_columns": {
            "resp_rate": {
                "method": "fillna",
                "columns": ["resp_rate_total", "resp_rate_monitor"],
            },
            "fio2": {
                "method": "fillna",
                "columns": ["fio2_vent", "fio2_other"],
            },
            "spo2": {
                "method": "fillna",
                "columns": ["spo2_vent", "spo2"],
            },
            "o2_flow": {
                "method": "fillna",
                "columns": ["o2_flow", "o2_flow_nebulizer"],
            },
        },
        "drop_nan_all": {"resp_rate", "fio2", "spo2", "o2_flow"},
        "melt": {
            "id_vars": ["pid", "timestamp"],
        },
        "drop_nan_after_melt": True,
    }

    neurology = {
        "filename": "MetingNeurologieGlasgowComaScale.csv.pkl.gz",
        "rename": {
            "Pseudo_id": "pid",
            "MeetMoment": "timestamp",
            "GCSTotaalScore": "gcs_total",
        },
        "drop_nan_any": {"pid", "timestamp"},
        "retype": {
            "pid": "int16",
            "timestamp": "datetime64[ns]",
            "gcs_total": "float32",
        },
        "filter_columns": {
            "gcs_total": {"min": 0, "max": 15},
        },
        "melt": {
            "id_vars": ["pid", "timestamp"],
        },
        "drop_nan_after_melt": True,
    }

    temperature = {
        "filename": "MetingCirculatieTemperatuur.csv.pkl.gz",
        "rename": {
            "Pseudo_id": "pid",
            "MeetMoment": "timestamp",
            "TBlaas": "temp_bladder",
            "TSlokdarm": "temp_esophagus",
            "TRectaal": "temp_rectal",
            "TBloed": "temp_blood",
            "T1": "temp_1",
            "TOor": "temp_ear",
            "TAxillair": "temp_axillary",
            "THuid": "temp_skin",
        },
        "drop_nan_any": {"pid", "timestamp"},
        "retype": {
            "pid": "int16",
            "timestamp": "datetime64[ns]",
            "temp_bladder": "float32",
            "temp_esophagus": "float32",
            "temp_rectal": "float32",
            "temp_blood": "float32",
            "temp_1": "float32",
            "temp_ear": "float32",
            "temp_axillary": "float32",
            "temp_skin": "float32",
        },
        "filter_columns": {
            "temp_bladder": {"min": 25, "max": 43},
            "temp_esophagus": {"min": 25, "max": 43},
            "temp_rectal": {"min": 25, "max": 43},
            "temp_blood": {"min": 25, "max": 43},
            "temp_1": {"min": 25, "max": 43},
            "temp_ear": {"min": 25, "max": 43},
            "temp_axillary": {"min": 25, "max": 43},
            "temp_skin": {"min": 25, "max": 43},
        },
        "combine_columns": {
            "temperature": {
                "method": "fillna",
                "columns": [
                    "temp_bladder",
                    "temp_esophagus",
                    "temp_rectal",
                    "temp_blood",
                    "temp_1",
                    "temp_ear",
                    "temp_axillary",
                    "temp_skin",
                ],
            },
        },
        "melt": {
            "id_vars": ["pid", "timestamp"],
        },
        "drop_nan_after_melt": True,
    }

    urine_production = {
        "filename": "MetingVochtbalansUit.csv.pkl.gz",
        "rename": {
            "Pseudo_id": "pid",
            "MeetMoment": "timestamp",
            "Waarde": "value",
            "Eenheid": "unit",
            "VochtbalansUitType": "variable",
        },
        "retype": {
            "pid": "int16",
            "timestamp": "datetime64[ns]",
            "variable": "category",
            "value": "float32",
            "unit": "category",
        },
        "filter_columns": {
            "value": {"min": 0, "max": 2000},
        },
        "drop_nan_any": {"pid", "timestamp", "value"},
        "remap": {
            "variable": {
                "Urine/ Urinedrains (ml)": "urine",
                # 'Thoraxdrain(s) (ml)',
                # 'Faeces',
                # 'Overige drains (ml)',
                # 'Liquordrain(s) (ml)',
                # 'G.I drains/ sondes/ braken (ml)',
                # 'Bloed.',
                # 'Overig',
            }
        },
        "keep": {
            "variable": [
                "urine",
            ],
        },
        "sort": {
            "by": ["pid", "timestamp", "value"],
            "ascending": [True, False, False],
        },
        "duplicates": {
            "subset": ["pid", "timestamp", "variable"],
            "keep": "first",  # combined with sort value keeps max value
        },
        "transform_single_to_range": {
            "timestamp": "timestamp",
            "group_by": ["pid", "variable"],
            "direction": "backward",
            "max_duration": pd.Timedelta(6, "h"),
            "fill_duration": pd.Timedelta(6, "h"),
        },
        "calculate_rate": {
            "value": "value",
            "start": "timestamp_start",
            "stop": "timestamp_end",
            "duration_unit": pd.Timedelta(1, "h"),
        },
        "impute_missing_as_zero": {
            "group_by": ["pid"],
            "data_start": "timestamp_start",
            "data_end": "timestamp_end",
            "data_var": "variable",
            "data_val": "rate",
            "merge_on": ["pid"],
            "windows_start": "adm_icu_adm",
            "windows_end": "adm_icu_dis",
        },
        "join_adjoining_records": {
            "timestamp_start": "timestamp_start",
            "timestamp_end": "timestamp_end",
            "values": ["rate"],
            "group_by": ["pid", "variable"],
        },
    }

    singles = {
        "retype": {
            "pid": "int16",
            "timestamp_start": "datetime64[ns]",
            "timestamp_end": "datetime64[ns]",
            "variable": "category",
            "value": "object",
        },
        "sort_post_processing": {
            "by": ["pid", "timestamp", "variable", "value"],
            "ascending": [True, False, True, False],
        },
        "transform_single_to_range": {
            "timestamp": "timestamp",
            "group_by": ["pid"],
            "direction": "forward",
            "max_duration": pd.Timedelta(8, "h"),
            "fill_duration": pd.Timedelta(8, "h"),
        },
        "transform_overlapping_to_discrete_series": {
            "timestamp_start": "timestamp_start",
            "timestamp_end": "timestamp_end",
            "value": "value",
            "group_by": ["pid", "variable"],
            "keep": "last",
        },
        "join_adjoining_records": {
            "timestamp_start": "timestamp_start",
            "timestamp_end": "timestamp_end",
            "values": ["value"],
            "group_by": ["pid", "variable"],
        },
    }


@dataclass
class Measurements:
    """Specify the data loading and processing for the measurements data.

    The data is loaded from the raw data files and processed into a single table.
    """

    files = {
        "medication": "MedicatieToediening.csv.pkl.gz",
        "lab": "Labuitslag.csv.pkl.gz",
        "patients": "0_abpat.pkl.gz",
        "medication_definitions": "medication_selection.csv",
        "output": "2_measurements.pkl.gz",
    }

    data_loader = DataLoader()


@dataclass
class Aggregator:
    """Specify the data loading and processing for the measurements data.

    This class is used to aggregate the measurements data into a single table.
    """

    files = {
        "measurements": ("processed", Measurements.files["output"]),
        "events": ("processed", "2_events.pkl.gz"),
        "output": ("processed", "3_aggregated.pkl.gz"),
    }

    windows = {
        "event_columns": ["pid", "stop", "start"],
        "rename_columns": {"stop": "event_timestamp", "start": "windows__start_exact"},
        "create_columns": {
            "windows__event_m1d": {
                "source": "event_timestamp",
                "offset": -pd.Timedelta(1, "d"),
            },
            "windows__event_m2d": {
                "source": "event_timestamp",
                "offset": -pd.Timedelta(2, "d"),
            },
            "windows__event_m3d": {
                "source": "event_timestamp",
                "offset": -pd.Timedelta(3, "d"),
            },
            "windows__start_m1d": {
                "source": "windows__start_exact",
                "offset": -pd.Timedelta(1, "d"),
            },
            "windows__start_p1d": {
                "source": "windows__start_exact",
                "offset": +pd.Timedelta(1, "d"),
            },
            "windows__start_p3d": {
                "source": "windows__start_exact",
                "offset": +pd.Timedelta(3, "d"),
            }
        },
    }

    variables = [
        "AMV_EXP",
        "BENZO",
        "BP_DIA",
        "BP_MEAN",
        "BP_SYS",
        "CLONIDINE",
        "DEXMEDETOMIDINE",
        "FIO2",
        "GCS_TOTAL",
        "HALOPERIDOL",
        "HEART_RATE",
        "MIC__GRAM",
        "MIC__GROUP",
        "MIC__PATHGEN",
        "MIC__SDD_POSITIVE",
        "MIC__IS_POSITIVE__BLOOD",
        "MIC__IS_POSITIVE__SPUTUM",
        "MIC__IS_POSITIVE__TIP",
        "MIC__IS_POSITIVE__URINE",
        "MIC__IS_BLOOD",
        "MIC__IS_SPUTUM",
        "MIC__IS_TIP",
        "MIC__IS_URINE",
        "NORADRENALIN",
        "O2_DEVICE",
        "PCO2__ARTERIAL",
        "PCO2__CAPILLARY",
        "PCO2__NAN",
        "PCO2__POST-FILTER",
        "PCO2__PRE-FILTER",
        "PCO2__VENOUS",
        "PCO2__VENOUS__MIXED",
        "PO2__ARTERIAL",
        "PO2__CAPILLARY",
        "PO2__NAN",
        "PO2__POST-FILTER",
        "PO2__PRE-FILTER",
        "PO2__VENOUS",
        "PO2__VENOUS__MIXED",
        "LACTATE",
        "CRP",
        "LEUKOCYTES",
        "TROMBOCYTES",
        "TEMPERATURE",
        "RESP_RATE",
        "PROPOFOL",
        "QUETIAPINE",
        "RESP_RATE",
        "URINE",
        "KREAT",
        "SPO2",
        "PAO2_FIO2_RATIO",
    ]

    M1D_VARIABLES = [
        "AMV_EXP",
        "BENZO",
        "BP_DIA",
        "BP_MEAN",
        "BP_SYS",
        "CLONIDINE",
        "DEXMEDETOMIDINE",
        "FIO2",
        "GCS_TOTAL",
        "HALOPERIDOL",
        "HEART_RATE",
        "RESP_RATE",
        "CRP",
        "TROMBOCYTES",
        "LACTATE",
        "LEUKOCYTES",
        "NORADRENALIN",
        "TEMPERATURE",
        "PCO2__ARTERIAL",
        "PO2__ARTERIAL",
        "PROPOFOL",
        "QUETIAPINE",
        "URINE",
        "SODIUM",
        "CHLORIDE",
        "BICARBONATE__ARTERIAL",
        "PH__ARTERIAL",
        "KREAT",
        "SPO2",
        "PAO2_FIO2_RATIO",
    ]

    M3D_VARIABLES = [
        "AMV_EXP",
        "BENZO",
        "BP_DIA",
        "BP_MEAN",
        "BP_SYS",
        "CLONIDINE",
        "DEXMEDETOMIDINE",
        "FIO2",
        "GCS_TOTAL",
        "HALOPERIDOL",
        "HEART_RATE",
        "RESP_RATE",
        "CRP",
        "TROMBOCYTES",
        "LACTATE",
        "LEUKOCYTES",
        "TEMPERATURE",
        "NORADRENALIN",
        "PCO2__ARTERIAL",
        "PO2__ARTERIAL",
        "PROPOFOL",
        "QUETIAPINE",
        "RESP_RATE",
        "URINE",
        "SODIUM",
        "CHLORIDE",
        "BICARBONATE__ARTERIAL",
        "PH__ARTERIAL",
        "KREAT",
        "SPO2",
        "PAO2_FIO2_RATIO",
    ]

    features = [
        {
            "variables": [
                "MIC__GROUP",
                "MIC__PATHGEN",
                "MIC__GRAM",
            ],
            "window": "EVENT_M1D",
            "agg_func": "OHE",
            "merge_kwargs": {
                "on": ["pid"],
                "windows_start": "windows__event_m1d",
                "windows_end": "event_timestamp",
                "measurements_start": "timestamp_start",
                "measurements_end": "timestamp_end",
                "group_by": ["pid", "window_id", "variable"],
                "variable_id": "variable",
                "value": "value",
                "agg_func": "ohe",
                "map_columns": False,
            },
        },
        {
            "variables": [
                "MIC__GROUP",
                "MIC__PATHGEN",
                "MIC__GRAM",
            ],
            "window": "EVENT_M3D",
            "agg_func": "OHE",
            "merge_kwargs": {
                "on": ["pid"],
                "windows_start": "windows__event_m3d",
                "windows_end": "event_timestamp",
                "measurements_start": "timestamp_start",
                "measurements_end": "timestamp_end",
                "group_by": ["pid", "window_id", "variable"],
                "variable_id": "variable",
                "value": "value",
                "agg_func": "ohe",
                "map_columns": False,
            },
        },
        {
            "variables": [
                "MIC__GROUP",
                "MIC__PATHGEN",
                "MIC__GRAM",
            ],
            "window": "START_M1D_P1D",
            "agg_func": "OHE",
            "merge_kwargs": {
                "on": ["pid"],
                "windows_start": "windows__start_m1d",
                "windows_end": "windows__start_p1d",
                "measurements_start": "timestamp_start",
                "measurements_end": "timestamp_end",
                "group_by": ["pid", "window_id", "variable"],
                "variable_id": "variable",
                "value": "value",
                "agg_func": "ohe",
                "map_columns": False,
            },
        },
        {
            "variables": [
                "MIC__GROUP",
                "MIC__PATHGEN",
                "MIC__GRAM",
            ],
            "window": "START_M1D_P3D",
            "agg_func": "OHE",
            "merge_kwargs": {
                "on": ["pid"],
                "windows_start": "windows__start_m1d",
                "windows_end": "windows__start_p3d",
                "measurements_start": "timestamp_start",
                "measurements_end": "timestamp_end",
                "group_by": ["pid", "window_id", "variable"],
                "variable_id": "variable",
                "value": "value",
                "agg_func": "ohe",
                "map_columns": False,
            },
        },
        {  # M2D-MEAN
            "variables": [
                "NORADRENALIN",
                "PROPOFOL",
                "CLONIDINE",
                "DEXMEDETOMIDINE",
                "HALOPERIDOL",
                "QUETIAPINE",
                "BENZO",
                "URINE",
            ],
            "window": "EVENT_M2D",
            "agg_func": "MEAN",
            "merge_kwargs": {
                "on": ["pid"],
                "windows_start": "windows__start_m1d",
                "windows_end": "windows__start_p1d",
                "measurements_start": "timestamp_start",
                "measurements_end": "timestamp_end",
                "group_by": ["pid", "window_id", "variable"],
                "variable_id": "variable",
                "value": "value",
                "agg_func": "mean",
                "map_columns": False,
            },
        },
        {  # M1D-ANY
            "variables": [
                "MIC__IS_POSITIVE__BLOOD",
                "MIC__IS_POSITIVE__URINE",
                "MIC__IS_POSITIVE__SPUTUM",
                "MIC__IS_POSITIVE__TIP",
                "MIC__IS_BLOOD",
                "MIC__IS_URINE",
                "MIC__IS_SPUTUM",
                "MIC__IS_TIP",
                "MIC__SDD_POSITIVE",
            ],
            "window": "EVENT_M1D",
            "agg_func": "ANY",
            "merge_kwargs": {
                "on": ["pid"],
                "windows_start": "windows__event_m1d",
                "windows_end": "event_timestamp",
                "measurements_start": "timestamp_start",
                "measurements_end": "timestamp_end",
                "group_by": ["pid", "window_id", "variable"],
                "variable_id": "variable",
                "value": "value",
                "agg_func": "any",
                "map_columns": False,
            },
        },
        {  # M3D-ANY
            "variables": [
                "MIC__IS_POSITIVE__BLOOD",
                "MIC__IS_POSITIVE__URINE",
                "MIC__IS_POSITIVE__SPUTUM",
                "MIC__IS_POSITIVE__TIP",
                "MIC__IS_BLOOD",
                "MIC__IS_URINE",
                "MIC__IS_SPUTUM",
                "MIC__IS_TIP",
                "MIC__SDD_POSITIVE",
            ],
            "window": "EVENT_M3D",
            "agg_func": "ANY",
            "merge_kwargs": {
                "on": ["pid"],
                "windows_start": "windows__event_m3d",
                "windows_end": "event_timestamp",
                "measurements_start": "timestamp_start",
                "measurements_end": "timestamp_end",
                "group_by": ["pid", "window_id", "variable"],
                "variable_id": "variable",
                "value": "value",
                "agg_func": "any",
                "map_columns": False,
            },
        },
        {  # START-M1D-P1D-ANY
            "variables": [
                "MIC__IS_POSITIVE__BLOOD",
                "MIC__IS_POSITIVE__URINE",
                "MIC__IS_POSITIVE__SPUTUM",
                "MIC__IS_POSITIVE__TIP",
                "MIC__IS_BLOOD",
                "MIC__IS_URINE",
                "MIC__IS_SPUTUM",
                "MIC__IS_TIP",
                "MIC__SDD_POSITIVE",
            ],
            "window": "START_M1D_P1D",
            "agg_func": "ANY",
            "merge_kwargs": {
                "on": ["pid"],
                "windows_start": "windows__start_m1d",
                "windows_end": "windows__start_p1d",
                "measurements_start": "timestamp_start",
                "measurements_end": "timestamp_end",
                "group_by": ["pid", "window_id", "variable"],
                "variable_id": "variable",
                "value": "value",
                "agg_func": "any",
                "map_columns": False,
            },
        },
        {  # START-M1D-P3D-ANY
            "variables": [
                "MIC__IS_POSITIVE__BLOOD",
                "MIC__IS_POSITIVE__URINE",
                "MIC__IS_POSITIVE__SPUTUM",
                "MIC__IS_POSITIVE__TIP",
                "MIC__IS_BLOOD",
                "MIC__IS_URINE",
                "MIC__IS_SPUTUM",
                "MIC__IS_TIP",
                "MIC__SDD_POSITIVE",
            ],
            "window": "START_M1D_P3D",
            "agg_func": "ANY",
            "merge_kwargs": {
                "on": ["pid"],
                "windows_start": "windows__start_m1d",
                "windows_end": "windows__start_p3d",
                "measurements_start": "timestamp_start",
                "measurements_end": "timestamp_end",
                "group_by": ["pid", "window_id", "variable"],
                "variable_id": "variable",
                "value": "value",
                "agg_func": "any",
                "map_columns": False,
            },
        },
        {  # M1D-MEAN
            "variables": M1D_VARIABLES,
            "window": "EVENT_M1D",
            "agg_func": "MEAN",
            "merge_kwargs": {
                "on": ["pid"],
                "windows_start": "windows__event_m1d",
                "windows_end": "event_timestamp",
                "measurements_start": "timestamp_start",
                "measurements_end": "timestamp_end",
                "group_by": ["pid", "window_id", "variable"],
                "variable_id": "variable",
                "value": "value",
                "agg_func": "mean",
                "map_columns": False,
            },
        },
        {  # M1D-LAST
            "variables": M1D_VARIABLES,
            "window": "EVENT_M1D",
            "agg_func": "LAST",
            "merge_kwargs": {
                "on": ["pid"],
                "windows_start": "windows__event_m1d",
                "windows_end": "event_timestamp",
                "measurements_start": "timestamp_start",
                "measurements_end": "timestamp_end",
                "group_by": ["pid", "window_id", "variable"],
                "variable_id": "variable",
                "value": "value",
                "agg_func": "last",
                "map_columns": False,
            },
        },
        {  # M1D-MIN
            "variables": M1D_VARIABLES,
            "window": "EVENT_M1D",
            "agg_func": "MIN",
            "merge_kwargs": {
                "on": ["pid"],
                "windows_start": "windows__event_m1d",
                "windows_end": "event_timestamp",
                "measurements_start": "timestamp_start",
                "measurements_end": "timestamp_end",
                "group_by": ["pid", "window_id", "variable"],
                "variable_id": "variable",
                "value": "value",
                "agg_func": "min",
                "map_columns": False,
            },
        },
        {  # M1D-MAX
            "variables": M1D_VARIABLES,
            "window": "EVENT_M1D",
            "agg_func": "MAX",
            "merge_kwargs": {
                "on": ["pid"],
                "windows_start": "windows__event_m1d",
                "windows_end": "event_timestamp",
                "measurements_start": "timestamp_start",
                "measurements_end": "timestamp_end",
                "group_by": ["pid", "window_id", "variable"],
                "variable_id": "variable",
                "value": "value",
                "agg_func": "max",
                "map_columns": False,
            },
        },
        {  # M1D-TREND
            "variables": M1D_VARIABLES,
            "window": "EVENT_M1D",
            "agg_func": "TREND",
            "merge_kwargs": {
                "on": ["pid"],
                "windows_start": "windows__event_m1d",
                "windows_end": "event_timestamp",
                "measurements_start": "timestamp_start",
                "measurements_end": "timestamp_end",
                "group_by": ["pid", "window_id", "variable"],
                "variable_id": "variable",
                "value": "value",
                "agg_func": "trend",
                "map_columns": False,
            },
        },
        {  # P1D-MEAN
            "variables": list(set(M1D_VARIABLES+M3D_VARIABLES)),
            "window": "START_P1D",
            "agg_func": "MEAN",
            "merge_kwargs": {
                "on": ["pid"],
                "windows_start": "windows__start_exact",
                "windows_end": "windows__start_p1d",
                "measurements_start": "timestamp_start",
                "measurements_end": "timestamp_end",
                "group_by": ["pid", "window_id", "variable"],
                "variable_id": "variable",
                "value": "value",
                "agg_func": "mean",
                "map_columns": False,
            },
        },
        {  # P1D-LAST
            "variables": list(set(M1D_VARIABLES+M3D_VARIABLES)),
            "window": "START_P1D",
            "agg_func": "LAST",
            "merge_kwargs": {
                "on": ["pid"],
                "windows_start": "windows__start_exact",
                "windows_end": "windows__start_p1d",
                "measurements_start": "timestamp_start",
                "measurements_end": "timestamp_end",
                "group_by": ["pid", "window_id", "variable"],
                "variable_id": "variable",
                "value": "value",
                "agg_func": "last",
                "map_columns": False,
            },
        },
        {  # P1D-MIN
            "variables": list(set(M1D_VARIABLES+M3D_VARIABLES)),
            "window": "START_P1D",
            "agg_func": "MIN",
            "merge_kwargs": {
                "on": ["pid"],
                "windows_start": "windows__start_exact",
                "windows_end": "windows__start_p1d",
                "measurements_start": "timestamp_start",
                "measurements_end": "timestamp_end",
                "group_by": ["pid", "window_id", "variable"],
                "variable_id": "variable",
                "value": "value",
                "agg_func": "min",
                "map_columns": False,
            },
        },
        {  # P1D-MAX
            "variables": list(set(M1D_VARIABLES+M3D_VARIABLES)),
            "window": "START_P1D",
            "agg_func": "MAX",
            "merge_kwargs": {
                "on": ["pid"],
                "windows_start": "windows__start_exact",
                "windows_end": "windows__start_p1d",
                "measurements_start": "timestamp_start",
                "measurements_end": "timestamp_end",
                "group_by": ["pid", "window_id", "variable"],
                "variable_id": "variable",
                "value": "value",
                "agg_func": "max",
                "map_columns": False,
            },
        },
        {  # P1D-TREND
            "variables": list(set(M1D_VARIABLES+M3D_VARIABLES)),
            "window": "START_P1D",
            "agg_func": "TREND",
            "merge_kwargs": {
                "on": ["pid"],
                "windows_start": "windows__start_exact",
                "windows_end": "windows__start_p1d",
                "measurements_start": "timestamp_start",
                "measurements_end": "timestamp_end",
                "group_by": ["pid", "window_id", "variable"],
                "variable_id": "variable",
                "value": "value",
                "agg_func": "trend",
                "map_columns": False,
            },
        },
        {  # M3D-MEAN
            "variables": M3D_VARIABLES,
            "window": "EVENT_M3D",
            "agg_func": "MEAN",
            "merge_kwargs": {
                "on": ["pid"],
                "windows_start": "windows__event_m3d",
                "windows_end": "event_timestamp",
                "measurements_start": "timestamp_start",
                "measurements_end": "timestamp_end",
                "group_by": ["pid", "window_id", "variable"],
                "variable_id": "variable",
                "value": "value",
                "agg_func": "mean",
                "map_columns": False,
            },
        },
        {  # M3D-LAST
            "variables": M3D_VARIABLES,
            "window": "EVENT_M3D",
            "agg_func": "LAST",
            "merge_kwargs": {
                "on": ["pid"],
                "windows_start": "windows__event_m3d",
                "windows_end": "event_timestamp",
                "measurements_start": "timestamp_start",
                "measurements_end": "timestamp_end",
                "group_by": ["pid", "window_id", "variable"],
                "variable_id": "variable",
                "value": "value",
                "agg_func": "last",
                "map_columns": False,
            },
        },
        {  # M3D-MIN
            "variables": M3D_VARIABLES,
            "window": "EVENT_M3D",
            "agg_func": "MIN",
            "merge_kwargs": {
                "on": ["pid"],
                "windows_start": "windows__event_m3d",
                "windows_end": "event_timestamp",
                "measurements_start": "timestamp_start",
                "measurements_end": "timestamp_end",
                "group_by": ["pid", "window_id", "variable"],
                "variable_id": "variable",
                "value": "value",
                "agg_func": "min",
                "map_columns": False,
            },
        },
        {  # M3D-MAX
            "variables": M3D_VARIABLES,
            "window": "EVENT_M3D",
            "agg_func": "MAX",
            "merge_kwargs": {
                "on": ["pid"],
                "windows_start": "windows__event_m3d",
                "windows_end": "event_timestamp",
                "measurements_start": "timestamp_start",
                "measurements_end": "timestamp_end",
                "group_by": ["pid", "window_id", "variable"],
                "variable_id": "variable",
                "value": "value",
                "agg_func": "max",
                "map_columns": False,
            },
        },
        {  # M3D-TREND
            "variables": M3D_VARIABLES,
            "window": "EVENT_M3D",
            "agg_func": "TREND",
            "merge_kwargs": {
                "on": ["pid"],
                "windows_start": "windows__event_m3d",
                "windows_end": "event_timestamp",
                "measurements_start": "timestamp_start",
                "measurements_end": "timestamp_end",
                "group_by": ["pid", "window_id", "variable"],
                "variable_id": "variable",
                "value": "value",
                "agg_func": "trend",
                "map_columns": False,
            },
        },
    ]

    post_features = [
        {
            "features": [
                "CLONIDINE__EVENT_M3D__MEAN",
                "DEXMEDETOMIDINE__EVENT_M3D__MEAN",
                "HALOPERIDOL__EVENT_M3D__MEAN",
                "QUETIAPINE__EVENT_M3D__MEAN",
            ],
            "method": "bool_or_above_threshold",
            "threshold": 0,
            "target_feature": "SEDATION__EVENT_M3D__ANY",
        },
        {
            "features": [
                "CLONIDINE__START_P1D__MEAN",
                "DEXMEDETOMIDINE__START_P1D__MEAN",
                "HALOPERIDOL__START_P1D__MEAN",
                "QUETIAPINE__START_P1D__MEAN",
            ],
            "method": "bool_or_above_threshold",
            "threshold": 0,
            "target_feature": "SEDATION__START_P1D__ANY",
        },
        {
            "features": [
                "NORADRENALIN__EVENT_M3D__MAX",
            ],
            "method": "bool_or_above_threshold",
            "threshold": 0.1,
            "target_feature": "NORADRENALIN__EVENT_M3D__ANY",
        },
        {
            "features": [
                "CLONIDINE__EVENT_M3D__MEAN",
                "CLONIDINE__EVENT_M1D__MEAN",
            ],
            "method": "proportional_difference",
            "target_feature": "CLONIDINE__EVENT_M3D_M1D__TREND",
        },
        {
            "features": [
                "DEXMEDETOMIDINE__EVENT_M3D__MEAN",
                "DEXMEDETOMIDINE__EVENT_M1D__MEAN",
            ],
            "method": "proportional_difference",
            "target_feature": "DEXMEDETOMIDINE__EVENT_M3D_M1D__TREND",
        },
        {
            "features": [
                "HALOPERIDOL__EVENT_M3D__MEAN",
                "HALOPERIDOL__EVENT_M1D__MEAN",
            ],
            "method": "proportional_difference",
            "target_feature": "HALOPERIDOL__EVENT_M3D_M1D__TREND",
        },
        {
            "features": [
                "QUETIAPINE__EVENT_M3D__MEAN",
                "QUETIAPINE__EVENT_M1D__MEAN",
            ],
            "method": "proportional_difference",
            "target_feature": "QUETIAPINE__EVENT_M3D_M1D__TREND",
        },
        {
            "features": [
                "CLONIDINE__EVENT_M3D_M1D__TREND",
                "DEXMEDETOMIDINE__EVENT_M3D_M1D__TREND",
                "HALOPERIDOL__EVENT_M3D_M1D__TREND",
                "QUETIAPINE__EVENT_M3D_M1D__TREND",
            ],
            "method": "bool_or_above_threshold",
            "threshold": 0.5,
            "target_feature": "SEDATION__EVENT_M3D_M1D__ANY_INCREASE_ABOVE_50p",
        },
    ]


@dataclass
class EventsCreator:
    """Specify the data loading and processing for the events data.

    This class is used to create the events data from the raw data files.
    """

    patients_filename: str | os.PathLike = "0_abpat.pkl.gz"
    events_filename: str | os.PathLike = "2_events.pkl.gz"
    antibiotics_filename: str | os.PathLike = "1_medication_antibiotics.pkl.gz"

    abpat_type_dict = {
        "adm_hosp_loc": "category",
        "adm_hosp_oper": "category",
        "adm_hosp_route": "category",
        "adm_hosp_specialism": "category",
        "adm_hosp_subspecialism": "category",
    }

    sdd_iv_meds = [
        "J01DD01",  # CEFOTAXIM
        "J01DD04",  # CEFTRIAXON
    ]

    pass


@dataclass
class Featurizer:
    files = {
        "aggregated": ("processed", "3_aggregated.pkl.gz"),
        "output": ("processed", "4_features.pkl.gz"),
    }

    transformations = [
        {
            "source_column": "atc_last_24h",
            "target_column": "atc_last_24h__groups",
            "method": "get_atc_groups",
        },
        {
            "source_column": "atc_first_24h",
            "target_column": "atc_first_24h__groups",
            "method": "get_atc_groups",
        },

        {
            "source_column": "outcome_next_atc",
            "target_column": "outcome_next_atc__groups",
            "method": "get_atc_groups",
        },
        {
            "source_column": "atc_last_24h",
            "target_column": "atc_last_24h__g2",
            "method": "get_atc_g2",
        },
        {
            "source_column": "atc_first_24h",
            "target_column": "atc_first_24h__g2",
            "method": "get_atc_g2",
        },
        {
            "source_column": "outcome_next_atc",
            "target_column": "outcome_next_atc__g2",
            "method": "get_atc_g2",
        },

    ]

    ohe_features = [
                # 'atc_first',
                'atc_first_24h',
                'atc_first_24h__groups',
                # 'atc_last',
                'atc_last_24h',
                "atc_last_24h__groups",
                'atc_overall',
                'outcome_next_atc',
                "outcome_next_atc__groups",
                "adm_hosp_loc",
                "adm_hosp_oper",
                "adm_hosp_route",
                "adm_hosp_specialism",
                "adm_hosp_subspecialism",
                "atc_last_24h__g2",
                "atc_first_24h__g2",
                "outcome_next_atc__g2",
    ]

    time_differences = [
        {
            "name": "outcome__los_icu__days",
            "fun": lambda df_start, df_end: (df_end - df_start) / pd.Timedelta(1, 'd'),
            "kwargs": {
                "df_start": "adm_icu_adm",
                "df_end": "adm_icu_dis",
            },
        },
        {
            "name": "outcome__los_hosp__days",
            "fun": lambda df_start, df_end: (df_end - df_start) / pd.Timedelta(1, 'd'),
            "kwargs": {
                "df_start": "adm_hosp_adm",
                "df_end": "adm_hosp_dis",
            },
        },
        {
            "name": "ab_duration_full__days",
            "fun": lambda df_duration: df_duration / pd.Timedelta(1, 'd'),
            "kwargs": {
                "df_duration": "ab_duration_full",
            }
        },
        {
            "name": "ab_duration_shortest_in_last_24h__days",
            "fun": lambda df_duration: df_duration / pd.Timedelta(1, 'd'),
            "kwargs": {
                "df_duration": "ab_duration_shortest_in_last_24h",
            }
        },
        {
            "name": "outcome_timedelta_restart__days",
            "fun": lambda df_duration: df_duration / pd.Timedelta(1, 'd'),
            "kwargs": {
                "df_duration": "outcome_timedelta_restart",
            }
        },
    ]

    g2_rename = {
        "A0": "Tetracyclines",
        "C0": "Amoxicillin,\n incl. B-lactam",
        "C1": "Penicillins",
        "D0": "Cephalosporins",
        "D1": "Carbapenems",
        "E0": "Cotrimoxazole",
        "F0": "Macrolides &\nLincosamides",
        "G0": "Aminoglycosides",
        "M0": "Fluoroquinolones",
        "X0": "Vancomycin",
        "X1": "Metronidazol",
        "X2": "Others",
        "_": "Stopped",
    }

    atc_groups = {  # base-rate 0.1943
        "A0": [  # Tetracylines
            "J01AA02",  # DOXYCYCLIN                   33; 0.242
            "J01AA08",  # MINOCYCLIN                    1; 0
            "J01AA12",  # TIGECYCLIN                    2; 0
        ],
        "C0": [  # Amoxicillin, B-lactam
            "J01CA04",  # AMOXICILLIN                 125; 0.184
            "J01CR02",  # AMOX+B-LACTAM               257; 0.113
        ],
        "C1": [  # Penicillin/Fluclox/Nitrofurantoin
            "J01CE01",  # PENICILLIN                   175; 0.426
            "J01CE05",  # FENETICILLIN                   3; 0.667
            "J01CF05",  # FLUCOXACILLIN                343; 0.379
        ],
        "D0": [  # Cephalosporins
            "J01DB04",  # CEFAZOLIN                    29 #1; 0.310
            "J01DC02",  # CEFUROXIME                    9 #2; 0.0
            "J01DD01",  # CEFOTAXIM                   678 #3; 0.245
            "J01DD02",  # CEFTAZIDIM                  339 #3; 0.189
            "J01DD04",  # CEFTRIAXONE                 674 #3; 0.095
        ],
        "D1": [  # Carbapenems
            "J01DH02",  # MEROPENEM                   260; 0.200
            "J01DH51",  # IMIPENEM CILASTATINE          3; 0.0
        ],
        "E0": [  # Cotrimoxazole
            "J01EE01",  # COTRIMOXAZOLE               162; 0.142
        ],
        "F0": [  # Macrolides & Lincosamides # No streptogramins in this group
            "J01FA09",  # CLARITROMYCIN                 2; 0.500
            "J01FA10",  # AZITROMYCIN                  11; 0.545
            "J01FF01",  # CLINDAMYCIN                  25; 0.120
        ],
        "G0": [  # Aminoglycosides
            "J01GB03",  # GENTAMICIN                   46; 0.130
        ],
        "M0": [  # Ciprofloxacin
            "J01MA02",  # CIPROFLOXACIN               554; 0.157
            "J01MA06",  # NORFLOXACIN                   1; 0.0
            "J01MA12",  # LEVOFLOXACIN                 11; 0.2727
            "J01MA14",  # MOXIFLOXACIN                 26; 0.231
        ],
        "X0": [  # Vancomycin
            "J01XA01",  # VANCOMYCIN                  644; 0.118
            "J01ZA01",  # VANCOMYCIN; after previous processing
        ],
        "X1": [  # Metronidazol
            "J01XD01",  # METRONIDAZOL                406; 0.195
        ],
        "X2": [  # Others
            "J01XA02",  # TEICOPLANIN                   9; 0.0
            "J01XB01",  # COLISTIN                      7; 0.0
            "J01XX08",  # LINEZOLID                     5; 0.0
            "J01CR05",  # PIPERACILLIN TAZOBACTAM       3; 0.0
        ],
    }

    atc_to_group = {}
    for k, v in atc_groups.items():
        for v1 in v:
            atc_to_group[v1] = k



@dataclass
class Archive:
    """Specify the data loading and processing for the events data.

    This class is used to create the events data from the raw data files.
    """

    exclusions = [
        "archive_",  # Do not archive archives self
        "Uitgifte_Stoppen",  # Raw uncompressed data already backed up in gz format
        "raw_export_gz",  # Raw uncompressed data files already backed up in gz format
        "abs_venv",  # Virtual environment
    ]
    archive_name = (
        f"archive_abstop_" f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz"
    )


features = {
    "base": {
        "numeric": [
            "adm_hosp_age",
            "bmi",
            "ab_duration_shortest_in_last_24h",
        ],
        "bool": [
            "is_sex__male",
            "adm_hosp_oper__medical__ohe",
            "is_sdd_within_14d_before_stop",
            "start_on_icu",
            # "adm_hosp_loc__amc__ohe",
        ]
    },
    "start": {
        "numeric": [

        ],
        "bool": [
            # "mic__is_positive__blood__start_m1d_p3d__any",
            # "mic__group_hospital_pathogens__start_m1d_p3d__ohe",
            # "atc_last_24h__groups__j01c__ohe",
        ],
    },
    "last_24h": {
        "numeric": [
            "heart_rate__event_m1d__last",
            "heart_rate__event_m1d__trend",
            "resp_rate__event_m1d__last",
            "resp_rate__event_m1d__trend",
            "temperature__event_m1d__max",
            "temperature__event_m1d__trend",
            "bp_sys__event_m1d__min",
            # "bp_sys__event_m1d__max",

            "ph__arterial__event_m1d__min",
            "ph__arterial__event_m1d__max",
            "kreat__event_m1d__last",
            # "noradrenalin__event_m1d__max",
            # "noradrenalin__event_m1d__trend",
        ],
        "bool": [
            # 'atc_last_24h__groups__j01a__ohe',
            # 'atc_last_24h__groups__j01c__ohe',
            # 'atc_last_24h__groups__j01d__ohe',
            # 'atc_last_24h__groups__j01e__ohe',
            # 'atc_last_24h__groups__j01f__ohe',
            # 'atc_last_24h__groups__j01g__ohe',
            # 'atc_last_24h__groups__j01m__ohe',
            # 'atc_last_24h__groups__j01x__ohe',
            # 'atc_last_24h__groups__j01z__ohe',
            'atc_last_24h__g2__a0__ohe',
            'atc_last_24h__g2__c0__ohe',
            'atc_last_24h__g2__c1__ohe',
            'atc_last_24h__g2__d0__ohe',
            'atc_last_24h__g2__d1__ohe',
            'atc_last_24h__g2__e0__ohe',
            'atc_last_24h__g2__f0__ohe',
            'atc_last_24h__g2__g0__ohe',
            'atc_last_24h__g2__m0__ohe',
            'atc_last_24h__g2__x0__ohe',
            'atc_last_24h__g2__x1__ohe',
            'atc_last_24h__g2__x2__ohe',
        ],
    },
    "last_72h": {
        "numeric": [
            "crp__event_m3d__last",
            "crp__event_m3d__trend",
            "lactate__event_m3d__last",
            "lactate__event_m3d__trend",
            "leukocytes__event_m3d__last",
            "leukocytes__event_m3d__trend",
            "trombocytes__event_m3d__last",
            "trombocytes__event_m3d__trend",
        ],
        "bool": [
            "sedation__event_m3d__any",
            "sedation__event_m3d_m1d__any_increase_above_50p",
            "mic__is_positive__tip__event_m3d__any",
            "mic__is_positive__sputum__event_m3d__any",
            "mic__is_positive__blood__event_m3d__any",
            "mic__is_positive__urine__event_m3d__any",

            "mic__pathgen_high__event_m3d__ohe",
            "mic__pathgen_medium__event_m3d__ohe",
            "mic__pathgen_low__event_m3d__ohe",
            "mic__pathgen_other__event_m3d__ohe",
            "mic__pathgen_none__event_m3d__ohe",

            # "mic__group_cns__event_m3d__ohe",
            # "mic__group_hospital_pathogens__event_m3d__ohe",
            # 'mic__group_community_pathogens__event_m3d__ohe',
            # "mic__group_negative__event_m3d__ohe",
            # 'mic__group_enterobacterales__event_m3d__ohe',
            # 'mic__group_viridans__event_m3d__ohe',
            # 'mic__group_other__event_m3d__ohe',
            # 'mic__group_candida__event_m3d__ohe',
            # 'mic__group_entercoc_aerococ__event_m3d__ohe',
            # 'mic__group_non_fermenters__event_m3d__ohe',
            # # 'mic__group_drop__event_m3d__ohe',
            # 'mic__group_coryneform_or_cutibacterium_or_bacillus_like__event_m3d__ohe',
            # 'mic__group_anaerobes_excl_cutibacterium__event_m3d__ohe',
            # 'mic__group_acinetobacter_and_aeromonas__event_m3d__ohe',
            # 'mic__group_hacek_or_nutrient_variant_streptococci__event_m3d__ohe',
            
            # SDD
            "mic__sdd_positive__event_m3d__any",
        ],
    },
}

outcomes = ["outcome_restart_in_72h_on_icu"]

descriptives = {
    "numeric": [
        "ab_duration_full__days",
        'outcome_timedelta_restart',
        'outcome_timedelta_readmission',
        'outcome_timedelta_mortality',
        'outcome_safely_stopped',
        'outcome_timedelta_previous_series_stop',
        'outcome__los_icu__days',
        'outcome__los_hosp__days',
        'outcome_timedelta_restart__days',
        'heart_rate__start_p1d__last',
         'resp_rate__start_p1d__last',
         'temperature__start_p1d__max',
         'lactate__start_p1d__last',
         'crp__start_p1d__last',

        "leukocytes__event_m3d__last",
        "leukocytes__event_m3d__trend",
        "trombocytes__event_m3d__last",
        "trombocytes__event_m3d__trend",

    ],
    "bool": [
        "is_sex__female",
        'sedation__start_p1d__any',
        'mic__gram_positive__event_m3d__ohe',
        'mic__gram_negative__event_m3d__ohe',
        'mic__group_candida__event_m3d__ohe',

            "mic__group_cns__event_m3d__ohe",
            "mic__group_hospital_pathogens__event_m3d__ohe",
            'mic__group_community_pathogens__event_m3d__ohe',
            "mic__group_negative__event_m3d__ohe",
            'mic__group_enterobacterales__event_m3d__ohe',
            'mic__group_viridans__event_m3d__ohe',
            'mic__group_other__event_m3d__ohe',
            'mic__group_candida__event_m3d__ohe',
            'mic__group_entercoc_aerococ__event_m3d__ohe',
            'mic__group_non_fermenters__event_m3d__ohe',
            # 'mic__group_drop__event_m3d__ohe',
            'mic__group_coryneform_or_cutibacterium_or_bacillus_like__event_m3d__ohe',
            'mic__group_anaerobes_excl_cutibacterium__event_m3d__ohe',
            'mic__group_acinetobacter_and_aeromonas__event_m3d__ohe',
            'mic__group_hacek_or_nutrient_variant_streptococci__event_m3d__ohe',
            # 'mic__group_moraxella_neisseria_non__event_m3d__ohe',


        "mic__is_blood__event_m3d__any",
        "mic__is_sputum__event_m3d__any",
        "mic__is_tip__event_m3d__any",
        "mic__is_urine__event_m3d__any",


        "mic__is_blood__start_m1d_p1d__any",
        "mic__is_sputum__start_m1d_p1d__any",
        "mic__is_tip__start_m1d_p1d__any",
        "mic__is_urine__start_m1d_p1d__any",

        "mic__is_blood__start_m1d_p3d__any",
        "mic__is_sputum__start_m1d_p3d__any",
        "mic__is_tip__start_m1d_p3d__any",
        "mic__is_urine__start_m1d_p3d__any",

        'mic__is_positive__sputum__start_m1d_p3d__any',
        'mic__is_positive__blood__start_m1d_p3d__any',
        'mic__is_positive__tip__start_m1d_p3d__any',
        'mic__is_positive__urine__start_m1d_p3d__any',

        'mic__gram_positive__start_m1d_p3d__ohe',
        'mic__gram_negative__start_m1d_p3d__ohe',
        'mic__group_negative__start_m1d_p3d__ohe',
        'mic__group_enterobacterales__start_m1d_p3d__ohe',
        'mic__group_cns__start_m1d_p3d__ohe',
        'mic__group_other__start_m1d_p3d__ohe',
        'mic__group_entercoc_aerococ__start_m1d_p3d__ohe',
        'mic__group_candida__start_m1d_p3d__ohe',
        'mic__group_hospital_pathogens__start_m1d_p3d__ohe',
        'mic__group_community_pathogens__start_m1d_p3d__ohe',
        'mic__group_coryneform_or_cutibacterium_or_bacillus_like__start_m1d_p3d__ohe',
        'mic__group_acinetobacter_and_aeromonas__start_m1d_p3d__ohe',
        'mic__group_non_fermenters__start_m1d_p3d__ohe',
        'mic__group_viridans__start_m1d_p3d__ohe',
        'mic__group_moraxella_neisseria_non__start_m1d_p3d__ohe',
        'mic__group_anaerobes_excl_cutibacterium__start_m1d_p3d__ohe',

        

        'mic__gram_positive__start_m1d_p1d__ohe',
        'mic__gram_negative__start_m1d_p1d__ohe',
        'mic__group_negative__start_m1d_p1d__ohe',
        'mic__group_enterobacterales__start_m1d_p1d__ohe',
        'mic__group_cns__start_m1d_p1d__ohe',
        'mic__group_other__start_m1d_p1d__ohe',
        'mic__group_entercoc_aerococ__start_m1d_p1d__ohe',
        'mic__group_candida__start_m1d_p1d__ohe',
        'mic__group_hospital_pathogens__start_m1d_p1d__ohe',
        'mic__group_community_pathogens__start_m1d_p1d__ohe',
        'mic__group_coryneform_or_cutibacterium_or_bacillus_like__start_m1d_p1d__ohe',
        'mic__group_acinetobacter_and_aeromonas__start_m1d_p1d__ohe',
        'mic__group_non_fermenters__start_m1d_p1d__ohe',
        'mic__group_viridans__start_m1d_p1d__ohe',
        'mic__group_moraxella_neisseria_non__start_m1d_p1d__ohe',
        'mic__group_anaerobes_excl_cutibacterium__start_m1d_p1d__ohe',
        'mic__is_positive__sputum__start_m1d_p1d__any',
        'mic__is_positive__blood__start_m1d_p1d__any',
        'mic__is_positive__tip__start_m1d_p1d__any',
        'mic__is_positive__urine__start_m1d_p1d__any',

        "mic__sdd_positive__event_m3d__any",
        "mic__sdd_positive__start_m1d_p3d__any",
        "mic__sdd_positive__start_m1d_p1d__any",

        "mic__pathgen_high__event_m3d__ohe",
        "mic__pathgen_medium__event_m3d__ohe",
        "mic__pathgen_low__event_m3d__ohe",
        "mic__pathgen_other__event_m3d__ohe",
        "mic__pathgen_none__event_m3d__ohe",

        "mic__pathgen_high__start_m1d_p1d__ohe",
        "mic__pathgen_medium__start_m1d_p1d__ohe",
        "mic__pathgen_low__start_m1d_p1d__ohe",
        "mic__pathgen_other__start_m1d_p1d__ohe",
        "mic__pathgen_none__start_m1d_p1d__ohe",

        "mic__pathgen_high__start_m1d_p3d__ohe",
        "mic__pathgen_medium__start_m1d_p3d__ohe",
        "mic__pathgen_low__start_m1d_p3d__ohe",
        "mic__pathgen_other__start_m1d_p3d__ohe",
        "mic__pathgen_none__start_m1d_p3d__ohe",

        'outcome_is_primary_series',
        'adm_hosp_loc__amc__ohe',
        'adm_hosp_loc__vumc__ohe',
        "adm_hosp_oper__medical__ohe",
        "adm_hosp_oper__surgical__ohe",
        "adm_hosp_route__elective__ohe",
        "adm_hosp_route__unplanned__ohe",
        "adm_hosp_route__unknown__ohe",
        'adm_hosp_specialism__8400__ohe',
         'adm_hosp_specialism__car__ohe',
         'adm_hosp_specialism__chi__ohe',
         'adm_hosp_specialism__ctc__ohe',
         'adm_hosp_specialism__int__ohe',
         'adm_hosp_specialism__kin__ohe',
         'adm_hosp_specialism__kno__ohe',
         'adm_hosp_specialism__lon__ohe',
         'adm_hosp_specialism__mdl__ohe',
         'adm_hosp_specialism__mka__ohe',
         'adm_hosp_specialism__nch__ohe',
         'adm_hosp_specialism__neu__ohe',
         'adm_hosp_specialism__oog__ohe',
         'adm_hosp_specialism__ort__ohe',
         'adm_hosp_specialism__pch__ohe',
         'adm_hosp_specialism__psy__ohe',
         'adm_hosp_specialism__rad__ohe',
         'adm_hosp_specialism__reu__ohe',
         'adm_hosp_specialism__rth__ohe',
         'adm_hosp_specialism__seh__ohe',
         'adm_hosp_specialism__uro__ohe',
         'adm_hosp_specialism__vro__ohe',
        'adm_hosp_subspecialism__card__ohe',
         'adm_hosp_subspecialism__chge__ohe',
         'adm_hosp_subspecialism__chir__ohe',
         'adm_hosp_subspecialism__chlo__ohe',
         'adm_hosp_subspecialism__conc__ohe',
         'adm_hosp_subspecialism__ctch__ohe',
         'adm_hosp_subspecialism__ctck__ohe',
         'adm_hosp_subspecialism__ctcv__ohe',
         'adm_hosp_subspecialism__ctra__ohe',
         'adm_hosp_subspecialism__cvat__ohe',
         'adm_hosp_subspecialism__ic__ohe',
         'adm_hosp_subspecialism__iend__ohe',
         'adm_hosp_subspecialism__ihem__ohe',
         'adm_hosp_subspecialism__iinf__ohe',
         'adm_hosp_subspecialism__inef__ohe',
         'adm_hosp_subspecialism__init__ohe',
         'adm_hosp_subspecialism__inta__ohe',
         'adm_hosp_subspecialism__ionc__ohe',
         'adm_hosp_subspecialism__ioud__ohe',
         'adm_hosp_subspecialism__ivas__ohe',
         'adm_hosp_subspecialism__khem__ohe',
         'adm_hosp_subspecialism__knoh__ohe',
         'adm_hosp_subspecialism__long__ohe',
         'adm_hosp_subspecialism__mdlz__ohe',
         'adm_hosp_subspecialism__mkaa__ohe',
         'adm_hosp_subspecialism__nchi__ohe',
         'adm_hosp_subspecialism__nchv__ohe',
         'adm_hosp_subspecialism__neuv__ohe',
         'adm_hosp_subspecialism__oogh__ohe',
         'adm_hosp_subspecialism__orth__ohe',
         'adm_hosp_subspecialism__pchi__ohe',
         'adm_hosp_subspecialism__psyc__ohe',
         'adm_hosp_subspecialism__reum__ohe',
         'adm_hosp_subspecialism__rint__ohe',
         'adm_hosp_subspecialism__rthe__ohe',
         'adm_hosp_subspecialism__sehl__ohe',
         'adm_hosp_subspecialism__urov__ohe',
         'adm_hosp_subspecialism__vgyn__ohe',
         'adm_hosp_subspecialism__vobs__ohe',
        'outcome_on_icu_restart',
        'outcome_restart_in_72h',
        'outcome_restart_in_72h_on_icu',
        'outcome_restart_same_antibiotic',
        'outcome_restart_same_antibiotic_group',
        'outcome_next_atc__groups__j01a__ohe',
        'outcome_next_atc__groups__j01c__ohe',
        'outcome_next_atc__groups__j01d__ohe',
        'outcome_next_atc__groups__j01e__ohe',
        'outcome_next_atc__groups__j01f__ohe',
        'outcome_next_atc__groups__j01g__ohe',
        'outcome_next_atc__groups__j01m__ohe',
        'outcome_next_atc__groups__j01x__ohe',
        'outcome_next_atc__groups__j01z__ohe',
        'outcome_next_atc__groups__none__ohe',

        'outcome_restart_same_antibiotic_g2',
        'outcome_next_atc__g2__a0__ohe',
        'outcome_next_atc__g2__c0__ohe',
        'outcome_next_atc__g2__c1__ohe',
        'outcome_next_atc__g2__d0__ohe',
        'outcome_next_atc__g2__d1__ohe',
        'outcome_next_atc__g2__e0__ohe',
        'outcome_next_atc__g2__f0__ohe',
        'outcome_next_atc__g2__g0__ohe',
        'outcome_next_atc__g2__m0__ohe',
        'outcome_next_atc__g2__x0__ohe',
        'outcome_next_atc__g2__x1__ohe',
        # 'outcome_next_atc__g2__x2__ohe',

        "atc_first_24h__groups__j01a__ohe",
        'atc_first_24h__groups__j01c__ohe',
        'atc_first_24h__groups__j01d__ohe',
        'atc_first_24h__groups__j01e__ohe',
        'atc_first_24h__groups__j01f__ohe',
        'atc_first_24h__groups__j01g__ohe',
        'atc_first_24h__groups__j01m__ohe',
        'atc_first_24h__groups__j01x__ohe',
        'atc_first_24h__groups__j01z__ohe',

        'atc_last_24h__g2__a0__ohe',
        'atc_last_24h__g2__c0__ohe',
        'atc_last_24h__g2__c1__ohe',
        'atc_last_24h__g2__d0__ohe',
        'atc_last_24h__g2__d1__ohe',
        'atc_last_24h__g2__e0__ohe',
        'atc_last_24h__g2__f0__ohe',
        'atc_last_24h__g2__g0__ohe',
        'atc_last_24h__g2__m0__ohe',
        'atc_last_24h__g2__x0__ohe',
        'atc_last_24h__g2__x1__ohe',
        'atc_last_24h__g2__x2__ohe',
        'atc_first_24h__g2__a0__ohe',
        'atc_first_24h__g2__c0__ohe',
        'atc_first_24h__g2__c1__ohe',
        'atc_first_24h__g2__d0__ohe',
        'atc_first_24h__g2__d1__ohe',
        'atc_first_24h__g2__e0__ohe',
        'atc_first_24h__g2__f0__ohe',
        'atc_first_24h__g2__g0__ohe',
        'atc_first_24h__g2__m0__ohe',
        'atc_first_24h__g2__x0__ohe',
        'atc_first_24h__g2__x1__ohe',
        'atc_first_24h__g2__x2__ohe',

        'outcome_readmission_in_72h',
        'outcome_mortality_in_72h',
        'outcome_mortality_adm_30d',
        'outcome_mortality_adm_90d',
    ],
}


@dataclass
class Columns:
    features_base_numeric = features["base"]["numeric"]
    features_base_bool = features["base"]["bool"]
    features_base = features_base_numeric + features_base_bool

    features_start_numeric = features["start"]["numeric"]
    features_start_bool = features["start"]["bool"]
    features_start = list(set(features_start_numeric + features_start_bool))

    features_last24h_numeric = features["last_24h"]["numeric"]
    features_last24h_bool = features["last_24h"]["bool"]
    features_last24h = list(set(features_last24h_numeric + features_last24h_bool))

    features_last72h_numeric = features["last_72h"]["numeric"]
    features_last72h_bool = features["last_72h"]["bool"]
    features_last72h = list(set(features_last72h_numeric + features_last72h_bool))

    features_numeric = list(set(features_base_numeric + features_start_numeric + features_last24h_numeric + features_last72h_numeric))
    features_bool = list(set(features_base_bool + features_start_bool + features_last24h_bool + features_last72h_bool))

    outcomes = outcomes
    group_by = ["pid"]

    features = list(set(features_base + features_start + features_last24h + features_last72h))
    model_columns = list(set(group_by + features + outcomes))

    descriptives_numeric = descriptives["numeric"]
    descriptives_bool = descriptives["bool"]
    descriptives_full = list(set(descriptives_numeric + descriptives_bool))
    full = list(set(features + outcomes + descriptives_full))


    table1 = {
        'adm_hosp_age': 'Age, years',
        "Sex": "Sex",
            'is_sex__male': '- Male',
            "is_sex__female": "- Female",
        'bmi': 'BMI',

        "Hospital": "Hospital",
            'adm_hosp_loc__amc__ohe': '- AMC',
            'adm_hosp_loc__vumc__ohe': '- VUMC',
        "Admission, type": "Admission, type",
             'adm_hosp_oper__medical__ohe': '- Medical',
             'adm_hosp_oper__surgical__ohe': '- Surgical',
        "Admission, route": "Admission, route",
             'adm_hosp_route__elective__ohe': '- Elective',
             'adm_hosp_route__unknown__ohe': '- Unknown__REMOVE',
             'adm_hosp_route__unplanned__ohe': '- Unplanned',
        "Admission, specialism": "Admission, specialism",
            'adm_hosp_specialism__chi__ohe': '- Surgery',
            'adm_hosp_specialism__int__ohe': '- Internal medicine',
            'adm_hosp_specialism__lon__ohe': '- Pulmonology',
            'adm_hosp_specialism__neu__ohe': '- Neurology',
            'adm_hosp_specialism__car__ohe': '- Cardiology',
            'adm_hosp_specialism__seh__ohe': '- Emergency medicine',
            'adm_hosp_specialism__ctc__ohe': '- Cardiothoracic surgery',
            'adm_hosp_specialism__nch__ohe': '- Neurosurgery',
            'adm_hosp_specialism__8400__ohe': 'adm_hosp_specialism__8400__ohe__ADD_SPECIALISM_TO_OTHER',
            'adm_hosp_specialism__mdl__ohe': '- Gastroenterology',
             'adm_hosp_specialism__kin__ohe': '- Other__ADD_SPECIALISM_TO_OTHER',
            "adm_hosp_specialism__missing": "- Missing",
             'adm_hosp_specialism__kno__ohe': 'adm_hosp_specialism__kno__ohe__ADD_SPECIALISM_TO_OTHER',
             'adm_hosp_specialism__mka__ohe': 'adm_hosp_specialism__mka__ohe__ADD_SPECIALISM_TO_OTHER',
             'adm_hosp_specialism__oog__ohe': 'adm_hosp_specialism__oog__ohe__ADD_SPECIALISM_TO_OTHER',
             'adm_hosp_specialism__ort__ohe': 'adm_hosp_specialism__ort__ohe__ADD_SPECIALISM_TO_OTHER',
             'adm_hosp_specialism__pch__ohe': 'adm_hosp_specialism__pch__ohe__ADD_SPECIALISM_TO_OTHER',
             'adm_hosp_specialism__psy__ohe': 'adm_hosp_specialism__psy__ohe__ADD_SPECIALISM_TO_OTHER',
             'adm_hosp_specialism__rad__ohe': 'adm_hosp_specialism__rad__ohe__ADD_SPECIALISM_TO_OTHER',
             'adm_hosp_specialism__reu__ohe': 'adm_hosp_specialism__reu__ohe__ADD_SPECIALISM_TO_OTHER',
             'adm_hosp_specialism__rth__ohe': 'adm_hosp_specialism__rth__ohe__ADD_SPECIALISM_TO_OTHER',
             'adm_hosp_specialism__uro__ohe': 'adm_hosp_specialism__uro__ohe__ADD_SPECIALISM_TO_OTHER',
             'adm_hosp_specialism__vro__ohe': 'adm_hosp_specialism__vro__ohe__ADD_SPECIALISM_TO_OTHER',

         "ATC group, last 24h": "ATC group, last 24h",
             'atc_last_24h__groups__j01a__ohe': '- Tetracyclines',
             'atc_last_24h__groups__j01c__ohe': '- Beta-lactam, penicillins',
             'atc_last_24h__groups__j01d__ohe': '- Other beta-lactam',
             'atc_last_24h__groups__j01e__ohe': '- Sulfonamides, trimethoprim',
             'atc_last_24h__groups__j01f__ohe': '- Macrolides, lincosamides & streptogramins',
             'atc_last_24h__groups__j01g__ohe': '- Aminoglycosides',
             'atc_last_24h__groups__j01m__ohe': '- Quinolones',
             'atc_last_24h__groups__j01x__ohe': '- Other antibacterials',
             'atc_last_24h__groups__j01z__ohe': '- Vancomycin',

        "Antibiotics, last 24h": "Antibiotics, last 24h",
            'atc_last_24h__g2__a0__ohe': "- Tetracyclines",
            'atc_last_24h__g2__c0__ohe': "- Amoxicillin, incl. B-lactam",
            'atc_last_24h__g2__c1__ohe': "- Penicillins",
            'atc_last_24h__g2__d0__ohe': "- Cephalosporins",
            'atc_last_24h__g2__d1__ohe': "- Carbapenems",
            'atc_last_24h__g2__e0__ohe': "- Cotrimoxazole",
            'atc_last_24h__g2__f0__ohe': "- Macrolides & Lincosamides",
            'atc_last_24h__g2__g0__ohe': "- Aminoglycosides",
            'atc_last_24h__g2__m0__ohe': "- Fluoroquinolones",
            'atc_last_24h__g2__x0__ohe': "- Vancomycin",
            'atc_last_24h__g2__x1__ohe': "- Metronidazol",
            'atc_last_24h__g2__x2__ohe': "- Others",

        "Antibiotics, first 24h": "Antibiotics, first 24h",
            'atc_first_24h__g2__a0__ohe': "- Tetracyclines",
            'atc_first_24h__g2__c0__ohe': "- Amoxicillin, incl. B-lactam",
            'atc_first_24h__g2__c1__ohe': "- Penicillins",
            'atc_first_24h__g2__d0__ohe': "- Cephalosporins, excl. Ceftriaxone",
            'atc_first_24h__g2__d2__ohe': "- Ceftriaxone",
            'atc_first_24h__g2__d1__ohe': "- Carbapenems",
            'atc_first_24h__g2__e0__ohe': "- Cotrimoxazole",
            'atc_first_24h__g2__f0__ohe': "- Macrolides & Lincosamides",
            'atc_first_24h__g2__g0__ohe': "- Aminoglycosides",
            'atc_first_24h__g2__m0__ohe': "- Fluoroquinolones",
            'atc_first_24h__g2__x0__ohe': "- Vancomycin",
            'atc_first_24h__g2__x1__ohe': "- Metronidazol",
            'atc_first_24h__g2__x2__ohe': "- Others",

        'start_on_icu': 'Antibiotics, started on ICU',
        'events_per_patient': "Stopped antibiotics, events per patient",
        'ab_duration_shortest_in_last_24h': 'Antibiotics, shortest treatment, days',
        "ab_duration_full__days": "Antibiotics, full treatment, days",
        'is_sdd_within_14d_before_stop': 'SDD given, <14 days',

        "Microbiology, culture results, -24h, +24h": "Microbiology, culture results, at start",
            'mic__group_cns__start_m1d_p1d__ohe': '- CNS',
            'mic__group_hospital_pathogens__start_m1d_p1d__ohe': "- Hospital pathogens",
            'mic__group_community_pathogens__start_m1d_p1d__ohe': "- Community pathogens",
            'mic__group_negative__start_m1d_p1d__ohe': '- Negative',
            "mic__group__missing__start_m1d_p1d__ohe": "- Missing",  # changed
            'mic__group_other__start_m1d_p1d__ohe': '- Other__ADD_TO_OTHER_M1D_P1D',
            'mic__group_enterobacterales__start_m1d_p1d__ohe': '__REMOVE__ADD_TO_OTHER_M1D_P1D',
            'mic__group_entercoc_aerococ__start_m1d_p1d__ohe': '__REMOVE__ADD_TO_OTHER_M1D_P1D',
            'mic__group_candida__start_m1d_p1d__ohe': "- Candida",
            'mic__group_coryneform_or_cutibacterium_or_bacillus_like__start_m1d_p1d__ohe': "__REMOVE__ADD_TO_OTHER_M1D_P1D",
            'mic__group_acinetobacter_and_aeromonas__start_m1d_p1d__ohe': "__REMOVE__ADD_TO_OTHER_M1D_P1D",
            'mic__group_non_fermenters__start_m1d_p1d__ohe': "__REMOVE__ADD_TO_OTHER_M1D_P1D",
            'mic__group_viridans__start_m1d_p1d__ohe': "__REMOVE__ADD_TO_OTHER_M1D_P1D",
            'mic__group_moraxella_neisseria_non__start_m1d_p1d__ohe': "__REMOVE__ADD_TO_OTHER_M1D_P1D",
            'mic__group_anaerobes_excl_cutibacterium__start_m1d_p1d__ohe': "__REMOVE__ADD_TO_OTHER_M1D_P1D",

        "Microbiology, culture pathogenicity, -24h, +24h": "Microbiology, culture pathogenicity, at start",
            "mic__pathgen_high__start_m1d_p1d__ohe": "- high",
            "mic__pathgen_medium__start_m1d_p1d__ohe": "- medium",
            "mic__pathgen_low__start_m1d_p1d__ohe": "- low",
            "mic__pathgen_other__start_m1d_p1d__ohe": "- others",
            "mic__pathgen_none__start_m1d_p1d__ohe": "- none",

        "Microbiology, Gram stain, -24h, +24h": "Microbiology, Gram stain, at start",
            'mic__gram_positive__start_m1d_p1d__ohe': "- Positive",
            'mic__gram_negative__start_m1d_p1d__ohe': "- Negative",
        "Microbiology, positive, -24h, +24h": "Microbiology, positive, at start",
            'mic__is_positive__blood__start_m1d_p1d__any': '- Blood',
            'mic__is_positive__sputum__start_m1d_p1d__any': '- Sputum',
            'mic__is_positive__tip__start_m1d_p1d__any': '- Tip',
            'mic__is_positive__urine__start_m1d_p1d__any': '- Urine',

        "Microbiology, culture site, -24h, +24h": "Microbiology, culture site, at start",
            "mic__is_blood__start_m1d_p1d__any": "- Blood",
            "mic__is_sputum__start_m1d_p1d__any": "- Sputum",
            "mic__is_tip__start_m1d_p1d__any": "- Tip",
            "mic__is_urine__start_m1d_p1d__any": "- Urine",

        "Microbiology, culture results, at_start": "Microbiology, culture results, at start (M1DP3D)",
            'mic__group_cns__start_m1d_p3d__ohe': '- CNS',
            'mic__group_hospital_pathogens__start_m1d_p3d__ohe': "- Hospital Pathogens",
            'mic__group_community_pathogens__start_m1d_p3d__ohe': "- Community Pathogens",
            'mic__group_negative__start_m1d_p3d__ohe': '- Negative',
            "mic__group__missing__start_m1d_p3d__ohe": "- Missing",  # changed
            'mic__group_other__start_m1d_p3d__ohe': '- Other__ADD_TO_OTHER_M1D_P3D',
            'mic__group_enterobacterales__start_m1d_p3d__ohe': '__REMOVE__ADD_TO_OTHER_M1D_P3D',
            'mic__group_entercoc_aerococ__start_m1d_p3d__ohe': '__REMOVE__ADD_TO_OTHER_M1D_P3D',
            'mic__group_candida__start_m1d_p3d__ohe': "- Candida",
            'mic__group_coryneform_or_cutibacterium_or_bacillus_like__start_m1d_p3d__ohe': "__REMOVE__ADD_TO_OTHER_M1D_P3D",
            'mic__group_acinetobacter_and_aeromonas__start_m1d_p3d__ohe': "__REMOVE__ADD_TO_OTHER_M1D_P3D",
            'mic__group_non_fermenters__start_m1d_p3d__ohe': "__REMOVE__ADD_TO_OTHER_M1D_P3D",
            'mic__group_viridans__start_m1d_p3d__ohe': "__REMOVE__ADD_TO_OTHER_M1D_P3D",
            'mic__group_moraxella_neisseria_non__start_m1d_p3d__ohe': "__REMOVE__ADD_TO_OTHER_M1D_P3D",
            'mic__group_anaerobes_excl_cutibacterium__start_m1d_p3d__ohe': "__REMOVE__ADD_TO_OTHER_M1D_P3D",

        "Microbiology, culture pathogenicity, at_start": "Microbiology, culture pathogenicity, (M1DP3D)",
            "mic__pathgen_high__start_m1d_p3d__ohe": "- high",
            "mic__pathgen_medium__start_m1d_p3d__ohe": "- medium",
            "mic__pathgen_low__start_m1d_p3d__ohe": "- low",
            "mic__pathgen_other__start_m1d_p3d__ohe": "- others",
            "mic__pathgen_none__start_m1d_p3d__ohe": "- none",

        "Microbiology, Gram stain, at_start": "Microbiology, Gram stain, at start (M1DP3D)",
            'mic__gram_positive__start_m1d_p3d__ohe': "- Positive",
            'mic__gram_negative__start_m1d_p3d__ohe': "- Negative",
        "Microbiology, positive, at_start": "Microbiology, positive, at start (M1DP3D)",
            'mic__is_positive__blood__start_m1d_p3d__any': '- Blood',
            'mic__is_positive__sputum__start_m1d_p3d__any': '- Sputum',
            'mic__is_positive__tip__start_m1d_p3d__any': '- Tip',
            'mic__is_positive__urine__start_m1d_p3d__any': '- Urine',

        "Microbiology, culture site, at_start": "Microbiology, culture site, at start (M1DP3D)",
            "mic__is_blood__start_m1d_p3d__any": "- Blood",
            "mic__is_sputum__start_m1d_p3d__any": "- Sputum",
            "mic__is_tip__start_m1d_p3d__any": "- Tip",
            "mic__is_urine__start_m1d_p3d__any": "- Urine",


        "Microbiology, culture results, last 72h": "Microbiology, culture results, last 72h",
            'mic__group_cns__event_m3d__ohe': '- CNS',
            'mic__group_hospital_pathogens__event_m3d__ohe': '- Hospital pathogens',
            'mic__group_community_pathogens__event_m3d__ohe': "- Community pathogens",
            'mic__group_negative__event_m3d__ohe': '- Negative',
            "mic__group__missing__event_m3d__ohe": "- Missing", # changed
            'mic__group_candida__event_m3d__ohe': '- Candida', # __REMOVE__ADD_TO_OTHER_M3D
            'mic__group_enterobacterales__event_m3d__ohe': '__REMOVE__ADD_TO_OTHER_M3D',
            'mic__group_entercoc_aerococ__event_m3d__ohe': '__REMOVE__ADD_TO_OTHER_M3D',
            'mic__group_other__event_m3d__ohe': '- Other__ADD_TO_OTHER_M3D',
            'mic__group_viridans__event_m3d__ohe': '__REMOVE__ADD_TO_OTHER_M3D',
            'mic__group_anaerobes_excl_cutibacterium__event_m3d__ohe': '__REMOVE__ADD_TO_OTHER_M3D',
            'mic__group_coryneform_or_cutibacterium_or_bacillus_like__event_m3d__ohe': '__REMOVE__ADD_TO_OTHER_M3D',
            'mic__group_non_fermenters__event_m3d__ohe': '__REMOVE__ADD_TO_OTHER_M3D',
            'mic__group_acinetobacter_and_aeromonas__event_m3d__ohe': '__REMOVE__ADD_TO_OTHER_M3D',
            # 'mic__group_moraxella_neisseria_non__event_m3d__ohe': '__REMOVE',
            'mic__group_hacek_or_nutrient_variant_streptococci__event_m3d__ohe': '__REMOVE__ADD_TO_OTHER_M3D',

        "Microbiology, culture pathogenicity, last 72h": "Microbiology, culture pathogenicity, last 72h",
            "mic__pathgen_low__event_m3d__ohe": "- low",
            "mic__pathgen_medium__event_m3d__ohe": "- medium",
            "mic__pathgen_high__event_m3d__ohe": "- high",
            "mic__pathgen_other__event_m3d__ohe": "- others",
            "mic__pathgen_none__event_m3d__ohe": "- none",

        "Microbiology, positive": "Microbiology, positive, last 72h",
            'mic__is_positive__blood__event_m3d__any': '- Blood',
            'mic__is_positive__sputum__event_m3d__any': '- Sputum',
            'mic__is_positive__tip__event_m3d__any': '- Tip',
            'mic__is_positive__urine__event_m3d__any': '- Urine',

        "Microbiology, culture site, last 72h": "Microbiology, culture site, last 72h",
            "mic__is_blood__event_m3d__any": "- Blood",
            "mic__is_sputum__event_m3d__any": "- Sputum",
            "mic__is_tip__event_m3d__any": "- Tip",
            "mic__is_urine__event_m3d__any": "- Urine",

        "Microbiology, Gram stain, last 72h": "Microbiology, Gram stain, last 72h",
            'mic__gram_positive__event_m3d__ohe': "- Positive",
            'mic__gram_negative__event_m3d__ohe': "- Negative",

        "mic__sdd_positive__event_m3d__any": "Microbiology, SDD positive, last 72h",
        "mic__sdd_positive__start_m1d_p3d__any": "Microbiology, SDD positive, at start (M1DP3)",
        "mic__sdd_positive__start_m1d_p1d__any": "Microbiology, SDD positive, at start",

        'resp_rate__event_m1d__last': 'Respiratory rate, last 24h, last',
        'resp_rate__event_m1d__trend': '- Trend',
        'heart_rate__event_m1d__last': 'Heart rate, last 24h, last',
        'heart_rate__event_m1d__trend': '- Trend',

        'temperature__event_m1d__max': 'Temperature, last 24, max',
        'temperature__event_m1d__trend': '- Trend',

        'sedation__event_m3d__any': 'Sedation, last 72h, any',
        'sedation__event_m3d_m1d__any_increase_above_50p': 'Sedation, last 72h, any, increase >50%',

        'crp__event_m3d__last': 'CRP, last 72h, last',
        'crp__event_m3d__trend': '- Trend',
        "leukocytes__event_m3d__last": "Leukocytes, last 72h, last",
        "leukocytes__event_m3d__trend": "Leukocytes, last 72h, trend",
        "trombocytes__event_m3d__last": "Thrombocytes, last 72h, last",
        "trombocytes__event_m3d__trend": "Thrombocytes, last 72h, trend",
        'lactate__event_m3d__last': 'Lactate, last 72h, last',
        'lactate__event_m3d__trend': '- Trend',

        'outcome__los_icu__days': 'Length of Stay, ICU, days',
        'outcome__los_hosp__days': 'Length of Stay, Hospital, days',
        'outcome_is_primary_series': 'Antibiotics, primary series',
        'outcome_restart_in_72h_on_icu': 'Restart, 72h, on ICU',
        'outcome_on_icu_restart': 'Restart, on ICU',
        'outcome_restart_in_72h': 'Restart, 72h',
        'outcome_timedelta_restart__days': 'Restart, time to restart, days',
        'outcome_readmission_in_72h': 'Readmission, 72h',
        'outcome_mortality_in_72h': 'Mortality, 72h',
        'outcome_mortality_adm_30d': 'Mortality, 30d',
        'outcome_mortality_adm_90d': 'Mortality, 90d',

        "ATC group, restarted antibiotic": "ATC group, restarted antibiotic",
             'outcome_next_atc__groups__j01a__ohe': '- Tetracyclines',
             'outcome_next_atc__groups__j01c__ohe': '- Beta-lactam, penicillins',
             'outcome_next_atc__groups__j01d__ohe': '- Other beta-lactam',
             'outcome_next_atc__groups__j01e__ohe': '- Sulfonamides, trimethoprim',
             'outcome_next_atc__groups__j01f__ohe': '- Macrolides, lincosamides & streptogramins',
             'outcome_next_atc__groups__j01g__ohe': '- Aminoglycosides',
             'outcome_next_atc__groups__j01m__ohe': '- Quinolones',
             'outcome_next_atc__groups__j01x__ohe': '- Other antibacterials',
             'outcome_next_atc__groups__j01z__ohe': '- Vancomycin',
             'outcome_next_atc__groups__none__ohe': '- None',

        "Antibiotics group, restarted antibiotic": "Antibiotics group, restarted antibiotic",
                'outcome_next_atc__g2__a0__ohe': "- Tetracyclines",
                'outcome_next_atc__g2__c0__ohe': "- Amoxicillin, incl. B-lactam",
                'outcome_next_atc__g2__c1__ohe': "- Penicillins",
                'outcome_next_atc__g2__d0__ohe': "- Cephalosporins",
                'outcome_next_atc__g2__d1__ohe': "- Carbapenems",
                'outcome_next_atc__g2__e0__ohe': "- Cotrimoxazole",
                'outcome_next_atc__g2__f0__ohe': "- Macrolides & Lincosamides",
                'outcome_next_atc__g2__g0__ohe': "- Aminoglycosides",
                'outcome_next_atc__g2__m0__ohe': "- Fluoroquinolones",
                'outcome_next_atc__g2__x0__ohe': "- Vancomycin",
                'outcome_next_atc__g2__x1__ohe': "- Metronidazol",
                'outcome_next_atc__g2__x2__ohe': "- Others",

        'outcome_restart_same_antibiotic': 'outcome_restart_same_antibiotic__REMOVE', # needs to be interpreted with on ICU and within 72h to be meaningful
        'outcome_restart_same_antibiotic_group': 'outcome_restart_same_antibiotic_group__REMOVE',
        'outcome_safely_stopped': 'outcome_safely_stopped__REMOVE',
        'outcome_timedelta_mortality': 'outcome_timedelta_mortality__REMOVE',
        'outcome_timedelta_previous_series_stop': 'outcome_timedelta_previous_series_stop__REMOVE',
        'outcome_timedelta_readmission': 'outcome_timedelta_readmission__REMOVE',
        'outcome_timedelta_restart': 'outcome_timedelta_restart__REMOVE',
        'adm_hosp_subspecialism__card__ohe': "__REMOVE",
        'adm_hosp_subspecialism__chge__ohe': "__REMOVE",
        'adm_hosp_subspecialism__chir__ohe': "__REMOVE",
        'adm_hosp_subspecialism__chlo__ohe': "__REMOVE",
        'adm_hosp_subspecialism__conc__ohe': "__REMOVE",
        'adm_hosp_subspecialism__ctch__ohe': "__REMOVE",
        'adm_hosp_subspecialism__ctck__ohe': "__REMOVE",
        'adm_hosp_subspecialism__ctcv__ohe': "__REMOVE",
        'adm_hosp_subspecialism__ctra__ohe': "__REMOVE",
        'adm_hosp_subspecialism__cvat__ohe': "__REMOVE",
        'adm_hosp_subspecialism__ic__ohe': "__REMOVE",
        'adm_hosp_subspecialism__iend__ohe': "__REMOVE",
        'adm_hosp_subspecialism__ihem__ohe': "__REMOVE",
        'adm_hosp_subspecialism__iinf__ohe': "__REMOVE",
        'adm_hosp_subspecialism__inef__ohe': "__REMOVE",
        'adm_hosp_subspecialism__init__ohe': "__REMOVE",
        'adm_hosp_subspecialism__inta__ohe': "__REMOVE",
        'adm_hosp_subspecialism__ionc__ohe': "__REMOVE",
        'adm_hosp_subspecialism__ioud__ohe': "__REMOVE",
        'adm_hosp_subspecialism__ivas__ohe': "__REMOVE",
        'adm_hosp_subspecialism__khem__ohe': "__REMOVE",
        'adm_hosp_subspecialism__knoh__ohe': "__REMOVE",
        'adm_hosp_subspecialism__long__ohe': "__REMOVE",
        'adm_hosp_subspecialism__mdlz__ohe': "__REMOVE",
        'adm_hosp_subspecialism__mkaa__ohe': "__REMOVE",
        'adm_hosp_subspecialism__nchi__ohe': "__REMOVE",
        'adm_hosp_subspecialism__nchv__ohe': "__REMOVE",
        'adm_hosp_subspecialism__neuv__ohe': "__REMOVE",
        'adm_hosp_subspecialism__oogh__ohe': "__REMOVE",
        'adm_hosp_subspecialism__orth__ohe': "__REMOVE",
        'adm_hosp_subspecialism__pchi__ohe': "__REMOVE",
        'adm_hosp_subspecialism__psyc__ohe': "__REMOVE",
        'adm_hosp_subspecialism__reum__ohe': "__REMOVE",
        'adm_hosp_subspecialism__rint__ohe': "__REMOVE",
        'adm_hosp_subspecialism__rthe__ohe': "__REMOVE",
        'adm_hosp_subspecialism__sehl__ohe': "__REMOVE",
        'adm_hosp_subspecialism__urov__ohe': "__REMOVE",
        'adm_hosp_subspecialism__vgyn__ohe': "__REMOVE",
        'adm_hosp_subspecialism__vobs__ohe': "__REMOVE",
        "pid": "__REMOVE",


    }

    plot_names = {
        'adm_hosp_age': 'Age, years',
        'is_sex__male': 'Sex, Male',
        "is_sex__female": "Sex, Female",
        'bmi': 'BMI',
        'adm_hosp_loc__amc__ohe': 'Hospital, AMC',
        'adm_hosp_loc__vumc__ohe': 'Hospital, VUmc',
        'adm_hosp_oper__medical__ohe': 'Admission type, Medical',
        'adm_hosp_oper__surgical__ohe': 'Admission type, Surgical',
        'adm_hosp_route__elective__ohe': 'Admission route, Elective',
        'adm_hosp_route__unknown__ohe': 'Admission route, Unknown',
        'adm_hosp_route__unplanned__ohe': 'Admission route, Unplanned',

        "ATC group, last 24h": "ATC group, last 24h",
        'atc_last_24h__groups__j01a__ohe': 'Antibiotics, last 24h, Tetracyclines',
        'atc_last_24h__groups__j01c__ohe': 'Antibiotics, last 24h, Beta-lactam, penicillins',
        'atc_last_24h__groups__j01d__ohe': 'Antibiotics, last 24h, Other beta-lactam',
        'atc_last_24h__groups__j01e__ohe': 'Antibiotics, last 24h,\nSulfonamides, trimethoprim',
        'atc_last_24h__groups__j01f__ohe': 'Antibiotics, last 24h,\nMacrolides, lincosamides & streptogramins',
        'atc_last_24h__groups__j01g__ohe': 'Antibiotics, last 24h, Aminoglycosides',
        'atc_last_24h__groups__j01m__ohe': 'Antibiotics, last 24h, Quinolones',
        'atc_last_24h__groups__j01x__ohe': 'Antibiotics, last 24h, Other antibacterials',
        'atc_last_24h__groups__j01z__ohe': 'Antibiotics, last 24h, Vancomycin',

        "Antibiotics, last 24h": "Antibiotics, last 24h",
        'atc_last_24h__g2__a0__ohe': "Antibiotics, last 24h, Tetracyclines",
        'atc_last_24h__g2__c0__ohe': "Antibiotics, last 24h,\nAmoxicillin, incl. B-lactam",
        'atc_last_24h__g2__c1__ohe': "Antibiotics, last 24h, Penicillins",
        'atc_last_24h__g2__d0__ohe': "Antibiotics, last 24h, Cephalosporins",
        'atc_last_24h__g2__d1__ohe': "Antibiotics, last 24h, Carbapenems",
        'atc_last_24h__g2__e0__ohe': "Antibiotics, last 24h, Cotrimoxazole",
        'atc_last_24h__g2__f0__ohe': "Antibiotics, last 24h,\nMacrolides & Lincosamides",
        'atc_last_24h__g2__g0__ohe': "Antibiotics, last 24h, Aminoglycosides",
        'atc_last_24h__g2__m0__ohe': "Antibiotics, last 24h, Fluoroquinolones",
        'atc_last_24h__g2__x0__ohe': "Antibiotics, last 24h, Vancomycin",
        'atc_last_24h__g2__x1__ohe': "Antibiotics, last 24h, Metronidazol",
        'atc_last_24h__g2__x2__ohe': "Antibiotics, last 24h, Others",

        'start_on_icu': 'Antibiotics, started on ICU',
        'events_per_patient': "Stopped antibiotics, events per patient",
        'ab_duration_shortest_in_last_24h': 'Antibiotics, shortest treatment, days',
        'is_sdd_within_14d_before_stop': 'SDD given, <14 days',

        "Microbiology, culture results, last 72h": "Microbiology, culture results, last 72h",
        'mic__group_cns__event_m3d__ohe': 'Culture positive group, last 72h, CNS',
        'mic__group_hospital_pathogens__event_m3d__ohe': 'Culture positive group, last 72h, Hospital pathogens',
        'mic__group_community_pathogens__event_m3d__ohe': "Culture positive group, last 72h, Community pathogens",
        'mic__group_negative__event_m3d__ohe': 'Culture negative, last 72h',

        "Microbiology, positive": "Microbiology, positive, last 72h",
        'mic__is_positive__blood__event_m3d__any': 'Culture positive group, last 72h, Blood',
        'mic__is_positive__sputum__event_m3d__any': 'Culture positive group, last 72h, Sputum',
        'mic__is_positive__tip__event_m3d__any': 'Culture positive group, last 72h, Tip',
        'mic__is_positive__urine__event_m3d__any': 'Culture positive group, last 72h, Urine',

        'resp_rate__event_m1d__last': 'Respiratory rate, last 24h, last',
        'resp_rate__event_m1d__trend': 'Respiratory rate, last 24h, trend',
        'heart_rate__event_m1d__last': 'Heart rate, last 24h, last',
        'heart_rate__event_m1d__trend': 'Heart rate, last 24h, trend',

        'temperature__event_m1d__max': 'Temperature, last 24, max',
        'temperature__event_m1d__trend': 'Temperature, last 24, trend',

        'sedation__event_m3d__any': 'Sedation, last 72h, any',
        'sedation__event_m3d_m1d__any_increase_above_50p': 'Sedation, last 72h, any, increase >50%',

        'crp__event_m3d__last': 'CRP, last 72h, last',
        'crp__event_m3d__trend': 'CRP, last 72h, trend',
        "leukocytes__event_m3d__last": "Leukocytes, last 72h, last",
        "leukocytes__event_m3d__trend": "Leukocytes, last 72h, trend",
        "trombocytes__event_m3d__last": "Thrombocytes, last 72h, last",
        "trombocytes__event_m3d__trend": "Thrombocytes, last 72h, trend",
        'lactate__event_m3d__last': 'Lactate, last 72h, last',
        'lactate__event_m3d__trend': 'Lactate, last 72h, trend',

        'kreat__event_m1d__last': 'Creatinine, last 24h, last',
        'bp_sys__event_m1d__min': 'Blood pressure, systolic, last 24h, min',
        'ph__arterial__event_m1d__min': 'pH, arterial, last 24h, min',
        'ph__arterial__event_m1d__max': 'pH, arterial, last 24h, max',

        'mic__group_enterobacterales__event_m3d__ohe': "Culture positive group, last 72h, Enterobacterales",
        'mic__group_viridans__event_m3d__ohe': "Culture positive group, last 72h, Viridans streptococci",
        'mic__group_other__event_m3d__ohe': "Culture positive group, last 72h, Other",
        'mic__group_candida__event_m3d__ohe': "Culture positive group, last 72h, Candida",
        'mic__group_entercoc_aerococ__event_m3d__ohe': "Culture positive group, last 72h, Enterococ/Aerococ",
        'mic__group_non_fermenters__event_m3d__ohe': "Culture positive group, last 72h, Non-fermenters",
        'mic__group_coryneform_or_cutibacterium_or_bacillus_like__event_m3d__ohe': "Culture positive group, last 72h,\n Coryneform/Cutibacterium/Bacillus",
        'mic__group_anaerobes_excl_cutibacterium__event_m3d__ohe': "Culture positive group, last 72h,\n Anaerobes (excl. cutibacterium)",
        'mic__group_acinetobacter_and_aeromonas__event_m3d__ohe': "Culture positive group, last 72h,\n Acinetobacter/Aeromonas",
        'mic__group_hacek_or_nutrient_variant_streptococci__event_m3d__ohe': "Culture positive group, last 72h,\n HACEK group",


        "mic__pathgen_high__event_m3d__ohe": "Culture pathogenicity, last 72h, High",
        "mic__pathgen_medium__event_m3d__ohe": "Culture pathogenicity, last 72h, Medium",
        "mic__pathgen_low__event_m3d__ohe": "Culture pathogenicity, last 72h, Low",
        "mic__pathgen_other__event_m3d__ohe": "Culture pathogenicity, last 72h, Others",
        "mic__pathgen_none__event_m3d__ohe": "Culture pathogenicity, last 72h, None",

        "mic__sdd_positive__event_m3d__any": "Microbiology, SDD positive, last 72h",
    }

@dataclass
class FeatureSelector:

    files = {
        "features": ("processed", "4_features.pkl.gz"),
        "output": ("processed", "5_features_selected.pkl.gz"),
        "corr_features": ("results", "corr_feat_x_feat.csv"),
        "corr_outcomes": ("results", "corr_feat_x_outcome.csv"),
        "pairplot": ("figures", "dist_feat_x_outcome.png"),
        "missing_per_patient": ("results", "missing_per_patient.csv"),
        "missing_per_feature": ("results", "missing_per_feature.csv"),
        "describe_per_feature": ("results", "describe_per_feature.csv"),
        "describe_per_feature_per_outcome": ("results", "describe_per_feature_per_outcome.csv"),
        "describe_per_feature_per_outcome_per_hospital": ("results", "describe_per_feature_per_outcome_per_hospital.csv"),
    }

    c = Columns()

    limits = {
        # col: {"min": 0.05, "max": 0.95}
        "clip": {},
        "drop": {
            'crp__event_m3d__trend': {"min": -500, "max": 500},
            'crp__start_p1d__trend': {"min": -500, "max": 500},
            'leukocytes__event_m3d__trend': {"min": -50, "max": 50},
            'trombocytes__event_m3d__trend': {"min": -500, "max": 500},
        },
    }


@dataclass
class ModelTrainer:

    seed = 42

    files = {
        "model_data": ("processed", "model_data.csv"),
        "output": ("processed", "6_model.pkl.gz"),
    }

    models = [
        # {
        #     "model": "SVM",
        #     "grid": {
        #         # Use Optuna
        #     },
        # },
        {
            "model": "LogisticRegression",
            "grid": [
                {
                    "C": [0.1, 1, 10, 100, 500, 1000], # 100 optimal?
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear"],
                    "class_weight": ["balanced"],
                    "random_state": [seed]
                },
                {
                    "C": [0.1, 1, 10, 100, 500, 1000], # 100 optimal?
                    "penalty": ["l2", None],
                    "solver": ["lbfgs", "newton-cholesky"],
                    "class_weight": ["balanced"],
                    "random_state": [seed]
                },

            ],
        },
        {
            "model": "LGBMClassifier",
            "grid": {
                "learning_rate": [
                    0.01,
                    # 0.1,
                    # 1,
                ],
                'n_estimators': [  # num_iterations; earlystopping callback
                    # 50,
                    100,
                    # 150, 200, 250, 300, 350, 400,
                    200,
                    300,
                    # 450,
                    # 500, 550, 600, 650, 700, 750, 800, 850,
                    # 600,
                    # 900,
                    # 1200,
                    ],
                # "max_bin": [10], [5, 10, 20],
                "num_leaves": [32],
                'max_depth': [-1],
                "scale_pos_weight": [True],
                "min_child_samples": [
                    # 20,
                    # 50,
                    # 100,
                    150,
                    200,
                    250,
                    # 300,
                    # 350,
                    # 400,
                    # 450,
                    # 500, 550,
                ],
                "random_state": [seed],
                "n_jobs": [-1],
                "device": ["gpu"],
                "bagging_freq": [1, 2, 5, 10],
                "bagging_fraction": [0.4, 0.6, 0.8, 1],
                "feature_fraction": [1],
            },
        },
    ]

    sensitivity_analyses = {
        "eickelberg": {
            "features": [
                'adm_hosp_age',
                'is_sex__male',
                'bmi',
                'adm_hosp_oper__medical__ohe',
                'start_on_icu',
                'is_sdd_within_14d_before_stop',

                # 'heart_rate__start_p1d__trend',  # change
                'heart_rate__start_p1d__last',  # change
                'resp_rate__start_p1d__last',  # change
                # 'resp_rate__start_p1d__trend',  # change
                # 'temperature__start_p1d__trend',  # change
                'temperature__start_p1d__max',  # change

                'lactate__start_p1d__last',  # change
                # 'lactate__event_m3d__trend',  # change
                'crp__start_p1d__last',  # change
                # 'crp__event_m3d__trend',  # change

                'sedation__start_p1d__any',  # change
                # 'sedation__event_m3d_m1d__any_increase_above_50p',  # change

                'atc_first_24h__groups__j01c__ohe',  # switch to first 24h
                'atc_first_24h__groups__j01d__ohe',  # switch
                'atc_first_24h__groups__j01x__ohe',  # switch
                'atc_first_24h__groups__j01z__ohe',  # switch
                'atc_first_24h__groups__j01g__ohe',  # switch
                'atc_first_24h__groups__j01a__ohe',  # switch
                'atc_first_24h__groups__j01e__ohe',  # switch
                'atc_first_24h__groups__j01m__ohe',  # switch
                'atc_first_24h__groups__j01f__ohe',  # switch
            ],
            "outcome": "eickelberg",
            "group": "pid",
        }
    }

@dataclass
class Settings:
    """Specify all settings classes for a single entry point.

    This class is used to load all the individual processing configuration points.
    """

    patient_selection = PatientSelection()
    antibiotic_selection = AntibioticSelection()
    microbiology = Microbiology()
    measurements = Measurements()
    aggregator = Aggregator()
    events_creator = EventsCreator()
    archive = Archive()
    featurizer = Featurizer()
    feature_selector = FeatureSelector()
    model_trainer = ModelTrainer()
