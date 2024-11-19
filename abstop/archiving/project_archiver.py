import logging
import os
import tarfile
from typing import Any

import tadam as td

from abstop.config import Config


class ProjectArchiver:
    def __init__(self, config: Config):
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.logger.debug(f"from {__name__} instantiate {self.__class__.__name__}")
        self.config = config

    def archive(self) -> None:
        root_dir = self.config.directory("root")
        archive_dir = os.path.join(root_dir, "archive")
        os.makedirs(archive_dir, exist_ok=True)
        archive_file = os.path.join(
            archive_dir, self.config.settings.archive.archive_name
        )

        def exclude_function(tarinfo: Any) -> Any:
            for term in self.config.settings.archive.exclusions:
                if term in tarinfo.name:
                    self.logger.debug(f"Excluding {tarinfo.name}")
                    return None
            return tarinfo

        with tarfile.open(archive_file, "w:gz") as tar:
            tar.add(
                root_dir,
                arcname=os.path.basename(root_dir),
                filter=exclude_function,
            )
        self.logger.info(f"Archived project to {archive_file}")
        td.utils.calculate_hash(archive_file, hash_type=["sha512"], save=True)


if __name__ == "__main__":
    config = Config(root="C:\\TADAM\\projects\\abstop")
    ProjectArchiver(config=config).archive()
