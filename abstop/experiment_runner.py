import logging
import os

from abstop.archiving.project_archiver import ProjectArchiver
from abstop.config import Config
from abstop.preprocessing.aggregator import Aggregator
from abstop.preprocessing.antibiotics import AntibioticsPreprocessor
from abstop.preprocessing.events import EventsCreator
from abstop.preprocessing.measurements import MeasurementsProcessor
from abstop.preprocessing.microbiology import MicrobiologyPreprocessor
from abstop.preprocessing.patient_selector import PatientSelector
from abstop.preprocessing.featurizer import Featurizer
from abstop.processing.feature_selector import FeatureSelector
from abstop.processing.model_trainer import ModelTrainer

logger = logging.getLogger("abstop")


def run_experiment(config: Config) -> None:
    """Run the experiment."""

    PatientSelector(config=config).run()
    MicrobiologyPreprocessor(config=config).run()
    AntibioticsPreprocessor(config=config).run()
    EventsCreator(config=config).run()
    MeasurementsProcessor(config=config).run()
    Aggregator(config=config).run()
    Featurizer(config=config).run()
    FeatureSelector(config=config).run()
    ModelTrainer(config=config).run()
    ProjectArchiver(config=config).archive()


if __name__ == "__main__":
    config = Config(root="C:\\TADAM\\projects\\abstop", experiment_name="fb2rc1")

    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(
        os.path.join(config.directory("logs"), "experiment_runner.log")
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

    run_experiment(config=config)
