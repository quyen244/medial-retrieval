"""Lightweight, dependency-free per-query tracing.

No external collector (no OpenTelemetry) - this runs locally and is meant to
be inspected as JSON or summarized into mlflow metrics, per the requirement.

Usage:
    trace = Trace(query_id=42, query_text="...")
    with trace.span("retrieval.vi", params={"top_k": 50}, n_in=1) as span:
        candidates = retriever.retrieve(...)
        span.n_out = len(candidates)
    trace.finish()
    trace.to_json()                 # dump for inspection
    trace.mlflow_metrics()          # flat dict for mlflow.log_metrics
"""
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Span:
    name: str
    start_ts: float
    end_ts: Optional[float] = None
    duration_ms: Optional[float] = None
    n_in: Optional[int] = None
    n_out: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _SpanHandle:
    """Context-manager handle returned by Trace.span(); lets the caller set
    n_out (and n_in, if not known up front) while the span is open."""

    def __init__(self, span: Span):
        self._span = span

    @property
    def n_in(self):
        return self._span.n_in

    @n_in.setter
    def n_in(self, value):
        self._span.n_in = value

    @property
    def n_out(self):
        return self._span.n_out

    @n_out.setter
    def n_out(self, value):
        self._span.n_out = value


class Trace:
    """One trace per query: an ordered list of stage spans."""

    def __init__(self, query_id: Any, query_text: str):
        self.query_id = query_id
        self.query_text = query_text
        self.start_ts = time.perf_counter()
        self.end_ts: Optional[float] = None
        self.spans: List[Span] = []

    @contextmanager
    def span(self, name: str, params: Optional[Dict[str, Any]] = None, n_in: Optional[int] = None):
        s = Span(name=name, start_ts=time.perf_counter(), params=params or {}, n_in=n_in)
        handle = _SpanHandle(s)
        try:
            yield handle
        except Exception as e:
            s.error = f"{type(e).__name__}: {e}"
            raise
        finally:
            s.end_ts = time.perf_counter()
            s.duration_ms = (s.end_ts - s.start_ts) * 1000.0
            self.spans.append(s)

    def finish(self):
        self.end_ts = time.perf_counter()

    @property
    def total_duration_ms(self) -> float:
        end = self.end_ts if self.end_ts is not None else time.perf_counter()
        return (end - self.start_ts) * 1000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "spans": [
                {**s.to_dict(), "duration_ms": round(s.duration_ms, 2) if s.duration_ms is not None else None}
                for s in self.spans
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def mlflow_metrics(self) -> Dict[str, float]:
        """Flatten spans into a metric dict suitable for mlflow.log_metrics().

        Keys are "<stage>.duration_ms", "<stage>.n_in", "<stage>.n_out".
        """
        metrics: Dict[str, float] = {"total.duration_ms": round(self.total_duration_ms, 2)}
        for s in self.spans:
            safe_name = s.name.replace(" ", "_")
            if s.duration_ms is not None:
                metrics[f"{safe_name}.duration_ms"] = round(s.duration_ms, 2)
            if s.n_in is not None:
                metrics[f"{safe_name}.n_in"] = float(s.n_in)
            if s.n_out is not None:
                metrics[f"{safe_name}.n_out"] = float(s.n_out)
        return metrics
