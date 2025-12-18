from pathlib import Path
import sys

# ensure repo root is on sys.path so we can import NEW.src.lts_model
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from NEW.src.lts_model import HALinguisticLTS


def load_series(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    # attempt to parse floats, ignore non-numeric
    series = []
    for ln in lines:
        try:
            series.append(float(ln))
        except Exception:
            # try splitting
            for part in ln.replace(',', ' ').split():
                try:
                    series.append(float(part))
                except Exception:
                    continue
    return series


def main():
    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / 'NEW' / 'Datasets' / 'alabama.txt'
    if not data_path.exists():
        print('Dataset not found:', data_path)
        return
    series = load_series(data_path)
    print('Loaded series length =', len(series))
    model = HALinguisticLTS(n_intervals=6)
    model.fit(series)
    last = series[-1]
    horizon = 5
    preds = model.forecast(last, horizon=horizon)
    print('\nForecasts (next', horizon, 'values):')
    for i, p in enumerate(preds, 1):
        print(f'{i}: {p:.4f}')


if __name__ == '__main__':
    main()
