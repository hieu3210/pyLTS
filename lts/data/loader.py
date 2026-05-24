"""Tải dataset từ nhiều nguồn khác nhau."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Dataset:
    """Dataset đã tải, sẵn sàng dùng cho forecasting.

    Attributes
    ----------
    name : str
        Tên dataset.
    values : list[float]
        Chuỗi giá trị thời gian.
    lb : float
        Cận dưới của universe of discourse.
    ub : float
        Cận trên của universe of discourse.
    metadata : dict
        Thông tin bổ sung (tuỳ chọn).
    """

    name: str
    values: list[float]
    lb: float
    ub: float
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:
        return (
            f"Dataset(name='{self.name}', n={len(self.values)}, "
            f"lb={self.lb}, ub={self.ub})"
        )


class DataLoader:
    """Tải dataset từ nhiều định dạng khác nhau.

    Định dạng hỗ trợ:
    - Legacy .txt (3 dòng): dòng 1 là giá trị phân tách bằng dấu phẩy,
      dòng 2 là lb, dòng 3 là ub.
    - CSV: chỉ định cột, auto-detect cột số đầu tiên.
    - Inline: list[float] cùng lb/ub tường minh.
    - Bundled: dataset đi kèm package (alabama, ...).
    """

    BUNDLED_DIR = Path(__file__).parent / "datasets"

    @classmethod
    def from_txt(cls, path: str | Path, name: str = "") -> Dataset:
        """Tải từ định dạng legacy 3 dòng .txt."""
        path = Path(path)
        with path.open() as f:
            values = list(map(float, f.readline().strip().split(",")))
            lb = float(f.readline().strip())
            ub = float(f.readline().strip())
        return Dataset(name=name or path.stem, values=values, lb=lb, ub=ub)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        column: str | None = None,
        lb: float | None = None,
        ub: float | None = None,
        name: str = "",
    ) -> Dataset:
        """Tải từ file CSV.

        Parameters
        ----------
        path : str | Path
            Đường dẫn file CSV.
        column : str | None
            Tên cột cần dùng; nếu None thì auto-detect cột số đầu tiên.
        lb, ub : float | None
            Cận dưới/trên; nếu None thì tự tính từ data.
        name : str
            Tên dataset.
        """
        try:
            import csv as csv_mod

            path = Path(path)
            rows: list[dict] = []
            with path.open(newline="") as f:
                reader = csv_mod.DictReader(f)
                headers = reader.fieldnames or []
                for row in reader:
                    rows.append(row)

            if not rows:
                raise ValueError(f"File CSV rỗng: {path}")

            if column is None:
                for h in headers:
                    try:
                        float(rows[0][h])
                        column = h
                        break
                    except (ValueError, TypeError):
                        continue
                if column is None:
                    raise ValueError("Không tìm được cột số trong CSV.")

            values = [float(r[column]) for r in rows]

        except ImportError:
            raise ImportError("Cần cài đặt thư viện 'csv' (standard library).")

        actual_lb = lb if lb is not None else min(values)
        actual_ub = ub if ub is not None else max(values)
        return Dataset(
            name=name or Path(path).stem,
            values=values,
            lb=actual_lb,
            ub=actual_ub,
        )

    @classmethod
    def from_list(
        cls,
        values: list[float],
        lb: float,
        ub: float,
        name: str = "inline",
    ) -> Dataset:
        """Tạo dataset từ list đã có sẵn."""
        return Dataset(name=name, values=list(values), lb=lb, ub=ub)

    @classmethod
    def bundled(cls, name: str) -> Dataset:
        """Tải dataset đi kèm package theo tên (vd: 'alabama').

        Raises
        ------
        FileNotFoundError
            Nếu dataset không tồn tại.
        """
        path = cls.BUNDLED_DIR / f"{name}.txt"
        if not path.exists():
            available = [p.stem for p in cls.BUNDLED_DIR.glob("*.txt")]
            raise FileNotFoundError(
                f"Dataset '{name}' không tồn tại. "
                f"Các dataset có sẵn: {available}"
            )
        return cls.from_txt(path, name=name)
