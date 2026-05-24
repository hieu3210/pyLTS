"""
lts — Linguistic Time Series Forecasting based on Hedge Algebras.

Bài báo:
  Nguyen Duy Hieu, Nguyen Cat Ho, Vu Nhu Lan (2020). LTS (2020).
  Nguyen Duy Hieu (2021). HO-LTS — High-Order LTS.
  Nguyen Duy Hieu (2022). LTS-PSO — PSO Parameter Optimization.
  Nguyen Duy Hieu (2023). CO-LTS — Co-Optimization.

Sử dụng nhanh:
--------------
>>> from lts import ExperimentConfig, ExperimentRunner
>>> config = ExperimentConfig.paper_table4()
>>> output = ExperimentRunner(config).run(include_baselines=True)
>>> print(output.summary())

>>> from lts import HOLTSModel, HAParams, DataLoader
>>> ds = DataLoader.bundled("alabama")
>>> m = HOLTSModel(HAParams(theta=0.527, alpha=0.412), order=2, specificity=2)
>>> m.fit(ds.values, ds.lb, ds.ub)
>>> print(m.predict()[:3])
"""

from lts.core.hedge_algebras import HAParams, HedgeAlgebra
from lts.core.sqm_formulas import sqm_closed_form_7, PAPER_WORDS_7
from lts.models.base import BaseForecaster, ForecastResult
from lts.models.lts_model import LTSModel
from lts.models.lts_variations_model import LTSVariationsModel
from lts.models.ho_lts import HOLTSModel
from lts.models.lts_pso import LTSPSOModel
from lts.models.co_lts import COLTSModel, COLTSConfig
from lts.models.chen1996 import Chen1996
from lts.models.song_chissom1993 import SongChissom1993
from lts.metrics.measures import ForecastMetrics
from lts.config.experiment_config import ExperimentConfig
from lts.experiments.runner import ExperimentRunner, ExperimentOutput
from lts.data.loader import DataLoader, Dataset
from lts.data.transforms import DataTransformer
from lts.optimization.pso import PSO, PSOConfig

__version__ = "0.2.0"

__all__ = [
    # Core
    "HAParams",
    "HedgeAlgebra",
    "sqm_closed_form_7",
    "PAPER_WORDS_7",
    # Models (2020)
    "BaseForecaster",
    "ForecastResult",
    "LTSModel",
    "LTSVariationsModel",
    "Chen1996",
    "SongChissom1993",
    # Models (2021–2023)
    "HOLTSModel",
    "LTSPSOModel",
    "COLTSModel",
    "COLTSConfig",
    # Optimization
    "PSO",
    "PSOConfig",
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
