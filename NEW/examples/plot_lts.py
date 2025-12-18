from pathlib import Path
import sys
import matplotlib.pyplot as plt
from statistics import mean

# ensure repo root is importable
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


def main():
    data_path = repo_root / 'NEW' / 'Datasets' / 'alabama.txt'
    if not data_path.exists():
        print('Dataset not found:', data_path)
        return
    series = load_series(data_path)
    horizon = 5
    train = series[:-horizon]
    test = series[-horizon:]
    last = train[-1]

    # persistence baseline
    pers_pred = [last] * horizon
    pers_err = mae(pers_pred, test)

    # HA-LTS
    model = HALinguisticLTS(n_intervals=6)
    model.fit(train)
    lts_pred = model.forecast(last, horizon=horizon)
    lts_err = mae(lts_pred, test)

    print('Persistence MAE:', pers_err)
    print('HA-LTS MAE:', lts_err)

    # plot
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = list(range(len(series)))
    ax.plot(xs, series, '-o', label='historical', color='C0')

    # forecast x positions
    fx = list(range(len(train), len(train) + horizon))
    ax.plot(fx, lts_pred, '-s', label='HA-LTS forecast', color='C1')
    ax.plot(fx, pers_pred, '--', label='persistence', color='C2')
    ax.axvline(len(train)-0.5, color='k', alpha=0.3)
    ax.set_title('Alabama series: HA-LTS forecast vs persistence')
    ax.legend()
    ax.set_xlabel('time index')
    ax.set_ylabel('value')

    out_dir = repo_root / 'NEW' / 'docs' / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'lts_forecast.png'
    fig.tight_layout()
    fig.savefig(out_file)
    print('Saved plot to', out_file)


if __name__ == '__main__':
    main()
