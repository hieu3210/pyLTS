"""Placeholder model wrappers for NEW project.

These are minimal classes / interfaces. Implementations will follow after
we study the PDF describing the target forecasting methods.
"""
from typing import List, Optional, Tuple
import importlib.util
import os
import sys

try:
    import numpy as np
except Exception:
    np = None

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    ARIMA = None

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:
    ExponentialSmoothing = None

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
except Exception:
    RandomForestRegressor = None
    LinearRegression = None


class BaseModel:
    def fit(self, train_series: List[float]):
        raise NotImplementedError()

    def predict(self, horizon: int) -> List[float]:
        raise NotImplementedError()


def _load_lts_modules():
    # Load HAs and LTS classes from the repository LTS folder
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    lts_root = os.path.join(repo_root, 'LTS')
    ha_path = os.path.join(lts_root, 'HAs', '__init__.py')
    lts_path = os.path.join(lts_root, 'LTS', '__init__.py')
    if not os.path.exists(ha_path) or not os.path.exists(lts_path):
        return None, None
    spec_ha = importlib.util.spec_from_file_location('HAs', ha_path)
    mod_ha = importlib.util.module_from_spec(spec_ha)
    sys.modules['HAs'] = mod_ha
    spec_ha.loader.exec_module(mod_ha)  # type: ignore

    spec_lts = importlib.util.spec_from_file_location('LTS_pkg', lts_path)
    mod_lts = importlib.util.module_from_spec(spec_lts)
    sys.modules['LTS_pkg'] = mod_lts
    spec_lts.loader.exec_module(mod_lts)  # type: ignore

    return mod_ha, mod_lts


class HAWrapLTS(BaseModel):
    """Wrapper that uses the repository's LTS implementation.

    This wrapper dynamically loads the `HAs` and `LTS` modules from the
    repository and uses them to produce forecasts that match the original
    implementation in `LTS/`.
    """

    def __init__(self, theta: float = 0.57, alpha: float = 0.49, k: int = 3):
        self.theta = float(theta)
        self.alpha = float(alpha)
        self.k = int(k)
        self.model = None
        self.ha_mod = None
        self.lts_mod = None

    def fit(self, train_series: List[float]):
        ha_mod, lts_mod = _load_lts_modules()
        if ha_mod is None or lts_mod is None:
            raise RuntimeError('Could not load LTS modules from repository')
        self.ha_mod = ha_mod
        self.lts_mod = lts_mod
        HedgeAlgebra = ha_mod.HedgeAlgebras
        ha = HedgeAlgebra(self.theta, self.alpha)
        words = ha.get_words(self.k)
        # instantiate repository LTS class
        LTSClass = lts_mod.LTS
        # repository LTS expects parameters: order, repeat, data, lb, ub, words, theta, alpha
        # here we store training series; lb/ub will be inferred from data
        lb = min(train_series)
        ub = max(train_series)
        self.model = LTSClass(1, True, list(train_series), lb, ub, words, self.theta, self.alpha)

    def predict(self, horizon: int) -> List[float]:
        if self.model is None:
            raise RuntimeError('Model not fitted')
        # repository LTS produces results for in-sample forecasting; for multi-step horizon
        # we will return the last `horizon` values from model.results if available
        results = list(self.model.results)
        if len(results) == 0:
            return [0.0] * horizon
        # if results shorter than horizon, pad with last value
        out = results[-horizon:]
        if len(out) < horizon:
            out = ([out[-1]] * (horizon - len(out))) + out
        return [float(x) for x in out]


class ARIMAModel(BaseModel):
    def __init__(self, order: Tuple[int, int, int] = (1, 0, 0)):
        self.order = order
        self.model_fit = None

    def fit(self, train_series: List[float]):
        if ARIMA is None:
            raise RuntimeError('statsmodels.ARIMA is not available')
        self.model_fit = ARIMA(train_series, order=self.order).fit()

    def predict(self, horizon: int) -> List[float]:
        if self.model_fit is None:
            raise RuntimeError('Model not fitted')
        preds = self.model_fit.forecast(steps=horizon)
        return [float(x) for x in preds]


class ETSModel(BaseModel):
    def __init__(self, seasonal: Optional[str] = None, seasonal_periods: Optional[int] = None):
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model_fit = None

    def fit(self, train_series: List[float]):
        if ExponentialSmoothing is None:
            raise RuntimeError('statsmodels ExponentialSmoothing not available')
        self.model_fit = ExponentialSmoothing(train_series, seasonal=self.seasonal, seasonal_periods=self.seasonal_periods).fit()

    def predict(self, horizon: int) -> List[float]:
        if self.model_fit is None:
            raise RuntimeError('Model not fitted')
        preds = self.model_fit.forecast(steps=horizon)
        return [float(x) for x in preds]


class LagMLModel(BaseModel):
    """Simple lag-feature model using RandomForest or LinearRegression."""

    def __init__(self, lags: int = 5):
        self.lags = int(lags)
        self.model = None

    def _make_features(self, series: List[float]):
        X = []
        y = []
        for i in range(self.lags, len(series)):
            X.append(series[i - self.lags:i])
            y.append(series[i])
        return X, y

    def fit(self, train_series: List[float]):
        X, y = self._make_features(train_series)
        if len(X) == 0:
            raise RuntimeError('Not enough data for lag features')
        if RandomForestRegressor is not None:
            self.model = RandomForestRegressor(n_estimators=100)
        elif LinearRegression is not None:
            self.model = LinearRegression()
        else:
            raise RuntimeError('No sklearn regressors available')
        self.model.fit(X, y)

    def predict(self, horizon: int) -> List[float]:
        if self.model is None:
            raise RuntimeError('Model not fitted')
        # iterative multi-step forecasting
        last_window = list(self._last_window)
        out = []
        for _ in range(horizon):
            pred = float(self.model.predict([last_window])[0])
            out.append(pred)
            last_window = last_window[1:] + [pred]
        return out

    def fit(self, train_series: List[float]):
        X, y = self._make_features(train_series)
        if len(X) == 0:
            raise RuntimeError('Not enough data for lag features')
        if RandomForestRegressor is not None:
            self.model = RandomForestRegressor(n_estimators=100)
        elif LinearRegression is not None:
            self.model = LinearRegression()
        else:
            raise RuntimeError('No sklearn regressors available')
        self.model.fit(X, y)
        self._last_window = train_series[-self.lags:]

