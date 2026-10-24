"""Đọc config.yaml và cho phép ghi đè từ dòng lệnh.

Vì sao có file này: trước đây tham số nằm rải ở ba chỗ - giá trị mặc định
trong dataclass (`PipelineConfig`, `MultiQueryConfig`, `AugmentConfig`), cờ
dòng lệnh của `run_full_submission.py`, và hằng số viết cứng trong các script
`scripts/`. Muốn biết một lần chạy thật sự dùng số nào thì phải đọc cả ba.

Giờ `config.yaml` là nguồn sự thật duy nhất. Dataclass vẫn giữ nguyên mặc
định của chúng để code cũ chạy được, nhưng mọi đường vào đều nạp YAML trước.

    cfg = load_config()                      # đọc config.yaml
    cfg = load_config(overrides=["rerank.max_length=1024"])
    cfg.get("retrieval.chunk.branch_top_k")  # {'vi': 35, 'en': 20, 'zh': 10}
"""
import copy
import os
from typing import Any, Dict, Iterable, Optional

import yaml

DEFAULT_PATH = "config.yaml"


def _coerce(text: str) -> Any:
    """Chuỗi từ dòng lệnh -> kiểu Python. Dùng chính bộ phân tích của YAML để
    `true`, `null`, `8000`, `{vi: 40}` đều hiểu giống hệt trong file."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


class Config:
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, path: str) -> Any:
        sentinel = object()
        value = self.get(path, sentinel)
        if value is sentinel:
            raise KeyError(f"thiếu khoá {path!r} trong config.yaml")
        return value

    def set(self, path: str, value: Any) -> None:
        parts = path.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def section(self, path: str) -> Dict[str, Any]:
        value = self.get(path, {})
        return copy.deepcopy(value) if isinstance(value, dict) else {}

    def as_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        return f"Config({sorted(self._data)})"


def load_config(path: Optional[str] = None,
                overrides: Optional[Iterable[str]] = None) -> Config:
    """Nạp YAML rồi áp các ghi đè dạng `a.b.c=giá trị`."""
    path = path or os.environ.get("VIBIOMIR_CONFIG") or DEFAULT_PATH
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    cfg = Config(data)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"ghi đè phải có dạng khoa.con=giá_trị, gặp {item!r}")
        key, raw = item.split("=", 1)
        cfg.set(key.strip(), _coerce(raw.strip()))
    return cfg


def add_config_args(parser) -> None:
    """Gắn `--config` và `--set` vào một ArgumentParser có sẵn."""
    parser.add_argument("--config", default=None,
                        help=f"đường dẫn file cấu hình (mặc định {DEFAULT_PATH})")
    parser.add_argument("--set", action="append", default=[], metavar="KHOA=GIATRI",
                        dest="overrides",
                        help="ghi đè một khoá, lặp lại được: --set rerank.max_length=1024")


def config_from_args(args) -> Config:
    return load_config(getattr(args, "config", None), getattr(args, "overrides", None))
