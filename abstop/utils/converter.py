import logging
from dataclasses import dataclass

import numpy as np


class Converter:
    def __init__(self) -> None:
        self.factors = Factors()
        self.logger = logging.getLogger("abstop.utils.converter.Converter")

    def convert(
        self, value: float, source: str | None = None, target: str | None = None
    ) -> float:
        factor = self.get_factor(source=source, target=target)
        return value * factor

    def get_factor(self, source: str | None = None, target: str | None = None) -> float:
        if source and target:
            factor_source = self.factors.get(source)
            factor_target = self.factors.get(target)
            if factor_source and factor_target:
                return factor_source / factor_target
            else:
                return np.nan
        else:
            return np.nan


@dataclass
class Factors:
    logger = logging.getLogger("abstop.utils.converter.Factors")
    factors = {
        "mcg": 1e-3,
        "mg": 1,
        "g": 10e3,
        "kg": 10e6,
        "mg/hr": 1,
        "mg/uur": 1,
        "mcg/min": 1e-3 * 60,
        "microgr/kg/min": 1e-3 * 60,
        "microgr/kg/uur": 1e-3 * 1,
        "microgr/min": 1e-3 * 60,
        "microgr/uur": 1e-3 * 1,
        "microgr/kg/24uur": 1e-3 * (1 / 24),
        "microgr/24uur": 1e-3 * (1 / 24),
    }

    compatibility = {
        "mcg": "mass",
        "mg": "mass",
        "g": "mass",
        "kg": "mass",
        "mg/hr": "mass/time",
        "mcg/min": "mass/time",
        "microgr/kg/min": "mass/time",  # NOTE: kg are lost here
        "microgr/kg/uur": "mass/time",  # NOTE: kg are lost here
        "microgr/min": "mass/time",
        "microgr/uur": "mass/time",
        "microgr/kg/24uur": "mass/time",  # NOTE: kg are lost here
        "microgr/24uur": "mass/time",
    }

    def get(self, unit: str | None) -> float:
        if unit is None:
            return np.nan
        if unit not in self.factors:
            return np.nan
        return self.factors.get(unit, np.nan)
