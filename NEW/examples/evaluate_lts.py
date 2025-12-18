from pathlib import Path
import sys
from statistics import mean

# make sure package can be imported
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from NEW.src.lts_model import HALinguisticLTS


def load_series(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    series = []
    for ln in lines:
        try:
            series.append(float(ln))
        except Exception:
            for part in ln.replace(',', ' ').split():
                try:
                    series.append(float(part))
                except Exception:
                    continue
    return series


def mae(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def rolling_origin_eval(series, initial_train=12, horizon=3, step=1, n_intervals=6):
    pers_errors = []
    lts_errors = []
    for start in range(initial_train, len(series) - horizon + 1, step):
        train = series[:start]
        test = series[start:start + horizon]
        # persistence
        last = train[-1]
        pers_pred = [last] * horizon
        # lts
        model = HALinguisticLTS(n_intervals=n_intervals)
        model.fit(train)
        lts_pred = model.forecast(last, horizon=horizon)
        pers_errors.append(mae(pers_pred, test))
        lts_errors.append(mae(lts_pred, test))
    return mean(pers_errors) if pers_errors else None, mean(lts_errors) if lts_errors else None


def main():
    data_path = Path(__file__).resolve().parents[2] / 'NEW' / 'Datasets' / 'alabama.txt'
    if not data_path.exists():
        print('Dataset not found:', data_path)
        return
    series = load_series(data_path)
    print('Series length:', len(series))
    pers_mae, lts_mae = rolling_origin_eval(series, initial_train=12, horizon=3, step=1)
    print('\nEvaluation (MAE) over rolling-origin:')
    print('Persistence MAE:', pers_mae)
    print('HA-LTS MAE:', lts_mae)


if __name__ == '__main__':
    main()
