"""
lts — Linguistic Time Series Forecasting based on Hedge Algebras.

Bài báo:
  Nguyen Duy Hieu, Nguyen Cat Ho, Vu Nhu Lan (2020).
  "Enrollment Forecasting Based on Linguistic Time Series."
  Journal of Computer Science and Cybernetics, V.36, N.2.
  DOI: 10.15625/1813-9663/36/2/14396

Sử dụng nhanh:
--------------
>>> from lts import ExperimentConfig, ExperimentRunner
>>> config = ExperimentConfig.paper_table4()
>>> output = ExperimentRunner(config).run(include_baselines=True)
>>> print(output.summary())

>>> from lts.experiments.paper_experiments import print_all_results
>>> print_all_results()
"""

from lts.core.hedge_algebras import HAParams, HedgeAlgebra
from lts.core.sqm_formulas import sqm_closed_form_7, PAPER_WORDS_7
from lts.models.base import BaseForecaster, ForecastResult
from lts.models.lts_model import LTSModel
from lts.models.lts_variations_model import LTSVariationsModel
from lts.models.chen1996 import Chen1996
from lts.models.song_chissom1993 import SongChissom1993
from lts.metrics.measures import ForecastMetrics
from lts.config.experiment_config import ExperimentConfig
from lts.experiments.runner import ExperimentRunner, ExperimentOutput
from lts.data.loader import DataLoader, Dataset
from lts.data.transforms import DataTransformer

__version__ = "0.1.0"

__all__ = [
    # Core
    "HAParams",
    "HedgeAlgebra",
    "sqm_closed_form_7",
    "PAPER_WORDS_7",
    # Models
    "BaseForecaster",
    "ForecastResult",
    "LTSModel",
    "LTSVariationsModel",
    "Chen1996",
    "SongChissom1993",
    # Metrics
    "ForecastMetrics",
    # Config & Experiments
    "ExperimentConfig",
    "ExperimentRunner",
    "ExperimentOutput",
    # Data
    "DataLoader",
    "Dataset",
    "DataTransformer",
]
