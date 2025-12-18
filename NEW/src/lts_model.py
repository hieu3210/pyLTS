from typing import List, Tuple

class HALinguisticLTS:
    """Simplified HA-inspired linguistic time series forecaster.

    This implementation follows the principles in the extracted paper:
    - Partition numeric range into intervals (linguistic words)
    - Map historical values to interval indices
    - Build linguistic logical relationships (LLRs) as transitions
    - Forecast iteratively using rules (single successor -> use midpoint,
      multiple successors -> average of midpoints, no successor -> use own midpoint)
    """

    def __init__(self, n_intervals: int = 5):
        self.n_intervals = n_intervals
        self.intervals: List[Tuple[float, float]] = []
        self.midpoints: List[float] = []
        self.transitions = {}  # mapping from interval idx -> list of successor idx

    def fit(self, series: List[float]):
        if len(series) == 0:
            raise ValueError('Empty series')
        lo, hi = min(series), max(series)
        # if flat series, expand small range
        if hi == lo:
            hi = lo + 1.0
        width = (hi - lo) / self.n_intervals
        self.intervals = []
        self.midpoints = []
        for i in range(self.n_intervals):
            a = lo + i * width
            b = lo + (i + 1) * width
            self.intervals.append((a, b))
            self.midpoints.append((a + b) / 2.0)
        # map values to indices
        idxs = [self._value_to_idx(v) for v in series]
        # build transitions
        self.transitions = {i: [] for i in range(self.n_intervals)}
        for a, b in zip(idxs[:-1], idxs[1:]):
            self.transitions[a].append(b)

    def _value_to_idx(self, v: float) -> int:
        for i, (a, b) in enumerate(self.intervals):
            # include rightmost endpoint in last interval
            if i == len(self.intervals) - 1:
                if a <= v <= b:
                    return i
            else:
                if a <= v < b:
                    return i
        # fallback
        return max(0, min(len(self.intervals) - 1, int((v - self.intervals[0][0]) / (self.intervals[0][1] - self.intervals[0][0]))))

    def forecast(self, last_value: float, horizon: int = 1) -> List[float]:
        preds = []
        current = last_value
        for _ in range(horizon):
            idx = self._value_to_idx(current)
            succ = self.transitions.get(idx, [])
            if len(succ) == 0:
                # rule (3): use midpoint of own interval
                val = self.midpoints[idx]
            elif len(succ) == 1:
                # rule (1): use midpoint of single successor
                val = self.midpoints[succ[0]]
            else:
                # rule (2): average midpoints of successors
                val = sum(self.midpoints[s] for s in succ) / len(succ)
            preds.append(val)
            current = val
        return preds
