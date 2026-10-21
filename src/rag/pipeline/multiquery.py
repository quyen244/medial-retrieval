"""Truy xuất nhiều lượt cho một câu hỏi, rồi hợp tập và rerank một lần.

Đọc `augmented.jsonl` do pha augment sinh ra (src/rag/augment). Nhánh ngắn có
một unit nên chạy y như baseline; nhánh dài có 3 sub-query cộng câu gốc, mỗi
unit một lượt truy xuất riêng.

Hai điểm then chốt của thiết kế, cả hai đều dễ làm sai:

1. **Rerank bằng câu GỐC, không phải chuỗi đã augment.** Re-ranker được train
   trên câu người thật viết. Đưa chuỗi ghép đầy thuật ngữ đa ngữ vào là đẩy nó
   ra ngoài phân phối train. Chuỗi augment chỉ để TÌM ứng viên, không để CHẤM.

2. **Hợp tập rồi mới rerank, không rerank từng lượt rồi mới hợp.** Rerank
   từng lượt thì điểm của hai lượt khác nhau không so được với nhau, hợp lại
   là so nhầm. Rerank một lần trên cả tập cho mọi ứng viên cùng một thước đo.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.rag.pipeline.models import Candidate, QueryItem
from src.rag.pipeline.pipeline import PipelineResult, RetrievalPipeline
from src.rag.pipeline.tracer import Trace


@dataclass
class MultiQueryConfig:
    augmented_file: str = "data/augment/augmented.jsonl"

    # Cắt bớt tập hợp trước khi rerank. None = không cắt. Mặc định không cắt
    # vì cắt theo điểm fusion của các lượt khác nhau là so nhầm thước đo (xem
    # ghi chú 2 ở đầu file) - muốn giảm chi phí rerank thì giảm fusion_top_k.
    max_rerank_candidates: Optional[int] = None

    # Query không có trong augmented.jsonl thì chạy baseline một lượt với câu
    # gốc, chứ không bỏ query: bài nộp thiếu một id là bị loại cả bài.
    fallback_to_baseline: bool = True

    # Chuỗi đưa cho re-ranker. Mặc định là câu GỐC, không phải chuỗi augment,
    # vì dữ liệu train re-ranker (judged.jsonl, trường `query`) là câu hỏi
    # tiếng Việt tự nhiên chưa augment - train bằng câu tự nhiên rồi chạy
    # bằng chuỗi ghép đa ngữ là lệch phân phối train/test.
    #
    # Bật cờ này để đo hướng ngược lại. Lưu ý nó chỉ thật sự nhất quán nếu
    # tập train cũng được augment rồi train lại re-ranker; bật không mà vẫn
    # dùng checkpoint train trên câu tự nhiên thì đang đo một hệ lệch.
    rerank_with_augmented: bool = False

    # "concat": mỗi unit là MỘT chuỗi ghép (câu + thuật ngữ + targets), hợp tập
    # rồi rerank - cách của B/B'.
    # "lines": mỗi dòng ngắn là một lượt truy xuất riêng (câu gốc, rồi từng
    # target vi; en; zh), gộp các lượt bằng RRF rồi mới rerank - biến thể C.
    # Không nhồi mọi thứ vào một vector nên dense không bị loãng.
    # C TẠM ĐÓNG BĂNG (tới 14 lượt tìm mỗi query, quá tốn); không nằm trong kế hoạch nộp.
    mode: str = "concat"
    # Chỉ dùng ở mode "lines": số dòng tối đa mỗi query (chặn chi phí truy
    # xuất) và số ứng viên giữ lại sau RRF để đưa cho reranker.
    max_lines: int = 14
    lines_rrf_k: int = 60
    lines_top_k: int = 60

    # Chỉ dùng với selection="branch-rerank", mode "concat", query có phân rã.
    # Mỗi sub-query rerank 65 ứng viên của chính nó bằng câu sub-query tự nhiên,
    # giữ sub_rerank_top_k vào pool chung; top slots_per_sub của nó được giữ chỗ
    # trong bài nộp nếu hạng theo câu gốc (trong 50 chunk) <= slot_max_rank.
    # Câu gốc (unit raw) không qua bước này, giữ đủ ứng viên.
    sub_rerank: bool = True
    # Phân rã nhưng KHÔNG augment: mỗi unit tìm bằng câu trơn (sub-query tự nhiên,
    # câu gốc) thay vì chuỗi ghép thuật ngữ. Câu ngắn khi đó tìm y như bài A.
    plain_units: bool = False
    sub_rerank_top_k: int = 20
    # None: sub-query giữ sub_rerank_top_k chunk điểm cao nhất, không xét ngôn
    # ngữ. Có giá trị (vd {"vi": 12, "en": 5, "zh": 3}): giữ đúng số đó mỗi ngôn
    # ngữ, tổng thay cho sub_rerank_top_k. Chỗ giữ vẫn là top slots_per_sub chung.
    sub_branch_keep: Optional[Dict[str, int]] = None
    # Ghi lại MỌI ứng viên của mọi unit kèm điểm câu gốc và điểm sub-query
    # (MultiQueryResult.dump), để scripts/simulate_subquota.py lắp offline các
    # kiểu chia top-k / giữ chỗ / số chunk nộp mà không chạy GPU lại. Tốn thêm
    # một lượt chấm cho ứng viên sub-query bị cắt (~150 cặp mỗi câu dài).
    dump_all: bool = False
    slots_per_sub: int = 3
    slot_max_rank: int = 30

    def to_dict(self) -> Dict[str, Any]:
        return {
            "augmented_file": self.augmented_file,
            "max_rerank_candidates": self.max_rerank_candidates,
            "fallback_to_baseline": self.fallback_to_baseline,
            "rerank_with_augmented": self.rerank_with_augmented,
            "mode": self.mode,
            "max_lines": self.max_lines,
            "lines_rrf_k": self.lines_rrf_k,
            "lines_top_k": self.lines_top_k,
            "sub_rerank": self.sub_rerank,
            "plain_units": self.plain_units,
            "sub_rerank_top_k": self.sub_rerank_top_k,
            "sub_branch_keep": self.sub_branch_keep,
            "slots_per_sub": self.slots_per_sub,
            "slot_max_rank": self.slot_max_rank,
        }


def load_augmented(path: str) -> Dict[int, Dict[str, Any]]:
    """qid -> bản ghi AugmentedQuery đã ghi ra jsonl."""
    out: Dict[int, Dict[str, Any]] = {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Không thấy {p}. Chạy `python -m src.rag.augment.run_augment` trước."
        )
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                out[int(row["qid"])] = row
    return out


def _retrieval_text(u: Dict[str, Any], plain: bool) -> str:
    """plain=True: câu trơn (`text`: sub-query tự nhiên, hoặc câu gốc) - phân rã
    nhưng KHÔNG ghép thuật ngữ / mục tiêu augment."""
    if plain:
        return (u.get("text") or "").strip()
    return (u.get("augmented") or u.get("text") or "").strip()


def unit_texts(record: Dict[str, Any], plain: bool = False) -> List[str]:
    """Chuỗi đem đi truy xuất của từng unit, theo đúng thứ tự đã ghi.

    `augmented` rỗng (unit raw, hoặc augment hỏng) thì tụt về `text`.
    """
    texts: List[str] = []
    for u in record.get("units", []):
        t = _retrieval_text(u, plain)
        if t:
            texts.append(t)
    return texts


def unit_specs(record: Dict[str, Any], plain: bool = False) -> List[tuple]:
    """(kind, chuỗi truy xuất, câu tự nhiên) của từng unit, cùng thứ tự và cùng
    điều kiện bỏ unit rỗng như unit_texts(). Câu tự nhiên (`text`) là thứ đưa
    cho reranker khi rerank theo sub-query: reranker được train trên câu tự
    nhiên, không phải chuỗi ghép thuật ngữ."""
    specs: List[tuple] = []
    for u in record.get("units", []):
        t = _retrieval_text(u, plain)
        if t:
            specs.append((u.get("kind", ""), t, (u.get("text") or t).strip()))
    return specs


def select_with_slots(kept: List[Candidate], slot_ids: List[List[str]], top_k: int,
                      slots_per_sub: int, max_rank: int) -> tuple:
    """Chọn top_k chunk nộp từ `kept` (đã xếp theo điểm câu gốc).

    Mỗi sub-query giữ tối đa `slots_per_sub` chỗ, xét đúng `slots_per_sub` chunk
    đầu của nó (theo điểm sub-query): chunk được giữ chỗ nếu nằm trong `kept` và
    có hạng câu gốc <= max_rank. Phần còn lại lấp theo hạng câu gốc. Kết quả xếp
    lại theo điểm câu gốc. Trả (danh sách nộp, số chunk được đưa vào nhờ giữ chỗ
    mà theo điểm thuần thì đã nằm ngoài top_k)."""
    rank = {str(c.chunk_id): i for i, c in enumerate(kept, 1)}
    by_id = {str(c.chunk_id): c for c in kept}
    chosen: Dict[str, Candidate] = {}
    for ids in slot_ids:
        for cid in ids[:slots_per_sub]:
            if cid in rank and rank[cid] <= max_rank and cid not in chosen:
                chosen[cid] = by_id[cid]
    n_rescued = sum(1 for cid in chosen if rank[cid] > top_k)
    for c in kept:
        if len(chosen) >= top_k:
            break
        chosen.setdefault(str(c.chunk_id), c)
    for c in chosen.values():
        c.reserved_slot = False
    reserved = set()
    for ids in slot_ids:
        for cid in ids[:slots_per_sub]:
            if cid in rank and rank[cid] <= max_rank:
                reserved.add(cid)
    for cid in reserved:
        chosen[cid].reserved_slot = True
    final = sorted(chosen.values(), key=lambda c: c.rerank_score, reverse=True)
    return final[:top_k], n_rescued


def line_texts(record: Dict[str, Any], max_lines: int) -> List[str]:
    """Các dòng ngắn của biến thể C: câu gốc trước, rồi từng retrieval target
    (vi; en; zh) theo thứ tự xuất hiện, bỏ trùng. Câu gốc đứng đầu để luôn có
    mặt dù bị cắt bởi `max_lines`."""
    lines: List[str] = []
    seen = set()

    def add(t: str) -> None:
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            lines.append(t)

    add(record.get("original", ""))
    for u in record.get("units", []):
        for tg in u.get("targets", []):
            forms: List[str] = []
            for k in ("vi", "en", "zh"):
                v = (tg.get(k) or "").strip()
                if v and v not in forms:
                    forms.append(v)
            add("; ".join(forms))
    return lines[:max_lines]


def rrf_union(per_line: List[List[Candidate]], k: int, top_k: int) -> tuple:
    """Gộp các lượt bằng RRF theo vị trí trong từng danh sách. Trả (ứng viên
    đã xếp, số lượt tìm ra mỗi chunk). Dùng thứ hạng chứ không dùng điểm vì
    điểm của hai lượt khác nhau không so được với nhau."""
    merged: Dict[str, Candidate] = {}
    score: Dict[str, float] = {}
    hits: Dict[str, int] = {}
    for candidates in per_line:
        for rank, c in enumerate(candidates, 1):
            key = str(c.chunk_id)
            score[key] = score.get(key, 0.0) + 1.0 / (k + rank)
            hits[key] = hits.get(key, 0) + 1
            merged.setdefault(key, c)
    for key, c in merged.items():
        c.fusion_score = score[key]
    ranked = sorted(merged.values(), key=lambda c: c.fusion_score, reverse=True)
    return ranked[:top_k], hits


def union_candidates(per_unit: List[List[Candidate]]) -> tuple:
    """Hợp các danh sách, khử trùng theo chunk_id.

    Giữ bản gặp đầu tiên: mọi bản sao có cùng `content` nên chọn bản nào cũng
    như nhau với re-ranker, và giữ-bản-đầu cho kết quả cố định giữa các lần
    chạy. Trả thêm số lượt tìm ra mỗi chunk để đo xem phân rã có thực sự mở
    rộng vùng phủ hay ba sub-query chỉ lặp lại nhau.
    """
    merged: Dict[str, Candidate] = {}
    hits: Dict[str, int] = {}
    for candidates in per_unit:
        for c in candidates:
            key = str(c.chunk_id)
            hits[key] = hits.get(key, 0) + 1
            merged.setdefault(key, c)
    return list(merged.values()), hits


@dataclass
class MultiQueryResult:
    """Kết quả một câu hỏi, kèm số liệu riêng của nhánh nhiều truy vấn."""
    result: PipelineResult
    n_unit: int = 0
    n_union: int = 0
    n_overlap: int = 0  # số chunk được từ 2 lượt trở lên tìm ra
    used_fallback: bool = False
    stats: Dict[str, Any] = field(default_factory=dict)
    dump: Optional[Dict[str, Any]] = None


class MultiQueryPipeline:
    """Bọc ngoài RetrievalPipeline. Không thay thế nó: mọi thành phần (router,
    retriever, fusion, reranker) vẫn là của pipeline gốc, chỉ khác ở chỗ phần
    truy xuất chạy nhiều lượt trước khi rerank."""

    def __init__(self, pipeline: RetrievalPipeline,
                 augmented: Dict[int, Dict[str, Any]],
                 config: Optional[MultiQueryConfig] = None):
        self.pipeline = pipeline
        self.augmented = augmented
        self.config = config or MultiQueryConfig()

    def component_params(self) -> Dict[str, Any]:
        params = self.pipeline.component_params()
        params["multiquery"] = self.config.to_dict()
        return params

    def _run_branch_rerank(self, query: QueryItem, record: Dict[str, Any], texts: List[str],
                           lines_mode: bool, per_unit: List[List[Candidate]],
                           unit_branches: List[Dict[str, List[Candidate]]],
                           branch_results: Dict[str, List[Candidate]],
                           trace: Trace) -> MultiQueryResult:
        """Luồng branch-rerank cho nhiều unit (xem docs: luồng truy xuất query phân rã).

        1. Mỗi sub-query rerank ứng viên của chính nó bằng câu sub-query, giữ
           sub_rerank_top_k; top slots_per_sub ghi lại để giữ chỗ.
        2. Gom theo nhánh: top-k của sub-query + toàn bộ ứng viên các unit khác.
        3. Rerank một lần bằng câu gốc, giữ branch_keep (vi 26 / en 16 / zh 8).
        4. Chọn rerank_top_k chunk nộp có giữ chỗ cho từng sub-query.
        """
        cfg = self.config
        specs = unit_specs(record, cfg.plain_units) if not lines_mode else []
        use_sub = (cfg.sub_rerank and not lines_mode
                   and any(kind == "sub" for kind, _, _ in specs))

        by_branch: Dict[str, List[Candidate]] = {}
        slot_ids: List[List[str]] = []
        # chunk_id -> {"sub1": điểm, ...}: điểm theo sub-query, giữ riêng vì bước
        # rerank câu gốc phía sau ghi rerank_score lên chính các object này.
        sub_scores: Dict[str, Dict[str, float]] = {}
        subs: List[tuple] = []  # (câu sub-query tự nhiên, ứng viên của nó)
        for i, branches in enumerate(unit_branches):
            kind, _, natural = specs[i] if use_sub else ("", "", "")
            if kind != "sub":
                for lang, cands in branches.items():
                    by_branch.setdefault(lang, []).extend(cands)
                continue
            seen: Dict[str, Candidate] = {}
            for cands in branches.values():
                for c in cands:
                    seen.setdefault(str(c.chunk_id), c)
            subs.append((natural, list(seen.values())))

        if subs:
            # Chấm ứng viên của MỌI sub-query trong một lần gọi reranker (~195 cặp)
            # thay vì ba lần ~65 cặp: GPU chạy lô đầy hơn, điểm từng cặp không đổi.
            scorer = getattr(self.pipeline.reranker, "score_pairs", None)
            with trace.span("rerank.sub", params=self.pipeline.reranker.get_params(),
                            n_in=sum(len(cs) for _, cs in subs)) as span:
                if scorer is not None:
                    flat = scorer([(nat, c.content) for nat, cs in subs for c in cs])
                    groups, k = [], 0
                    for nat, cs in subs:
                        groups.append(flat[k:k + len(cs)])
                        k += len(cs)
                else:
                    groups = []
                    for nat, cs in subs:
                        self.pipeline.reranker.rerank(QueryItem(id=query.id, text=nat), cs, top_k=len(cs))
                        groups.append([c.rerank_score for c in cs])
                span.n_out = len(subs)
            for n, ((nat, cs), sc) in enumerate(zip(subs, groups), 1):
                key = f"sub{n}"
                for c, s in zip(cs, sc):
                    c.rerank_score = float(s)
                    sub_scores.setdefault(str(c.chunk_id), {})[key] = float(s)
                # Chấm hết ứng viên để lưu đủ điểm, rồi mới cắt top-k.
                ranked = sorted(cs, key=lambda c: c.rerank_score, reverse=True)
                slot_ids.append([str(c.chunk_id) for c in ranked[: cfg.slots_per_sub]])
                if cfg.sub_branch_keep:
                    left = dict(cfg.sub_branch_keep)
                    scored = []
                    for c in ranked:
                        if left.get(c.branch, 0) > 0:
                            left[c.branch] -= 1
                            scored.append(c)
                else:
                    scored = ranked[: cfg.sub_rerank_top_k]
                for c in scored:
                    by_branch.setdefault(c.branch, []).append(c)

        rerank_query = query
        if cfg.rerank_with_augmented:
            rerank_query = QueryItem(id=query.id, text=texts[0], metadata=dict(query.metadata))
        kept, reranked = self.pipeline.rerank_by_branch(rerank_query, by_branch, trace)
        for c in kept:
            c.sub_scores = dict(sub_scores.get(str(c.chunk_id), {}))

        n_rescued = 0
        if use_sub and slot_ids:
            reranked, n_rescued = select_with_slots(
                kept, slot_ids, self.pipeline.config.rerank_top_k,
                cfg.slots_per_sub, cfg.slot_max_rank,
            )

        dump = self._dump(query, rerank_query, specs if use_sub else [], unit_branches,
                          by_branch, sub_scores) if cfg.dump_all else None

        _, hits = union_candidates(per_unit)
        trace.finish()
        return MultiQueryResult(
            dump=dump,
            result=PipelineResult(query=query, branch_results=branch_results,
                                  fused=kept, reranked=reranked, trace=trace),
            n_unit=len(texts),
            n_union=sum(len(v) for v in by_branch.values()),
            n_overlap=sum(1 for v in hits.values() if v > 1),
            stats={"is_long": bool(record.get("is_long")), "sub_rerank": use_sub,
                   "n_slot_rescued": n_rescued},
        )

    def _dump(self, query: QueryItem, rerank_query: QueryItem, specs: List[tuple],
              unit_branches: List[Dict[str, List[Candidate]]],
              by_branch: Dict[str, List[Candidate]],
              sub_scores: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Mọi ứng viên của mọi unit (theo đúng thứ tự truy xuất từng nhánh) kèm
        điểm câu gốc và điểm sub-query. Xem MultiQueryConfig.dump_all."""
        # Điểm câu gốc của pool: object ĐẦU TIÊN mỗi chunk_id trong by_branch là
        # object rerank_by_branch đã chấm (nó gom pool bằng đúng vòng lặp này).
        raw: Dict[str, float] = {}
        for cands in by_branch.values():
            for c in cands:
                raw.setdefault(str(c.chunk_id), c.rerank_score)
        chunks: Dict[str, Dict[str, Any]] = {}
        units = []
        for i, branches in enumerate(unit_branches):
            units.append({"kind": specs[i][0] if specs else "",
                          "branches": {lang: [str(c.chunk_id) for c in cs] for lang, cs in branches.items()}})
            for cs in branches.values():
                for c in cs:
                    chunks.setdefault(str(c.chunk_id), {"doc_id": str(c.doc_id), "branch": c.branch,
                                                        "language": c.language, "content": c.content})
        missing = [cid for cid in chunks if cid not in raw]
        if missing:
            scores = self.pipeline.reranker.score_pairs(
                [(rerank_query.text, chunks[cid]["content"]) for cid in missing])
            raw.update({cid: float(s) for cid, s in zip(missing, scores)})
        for cid, ch in chunks.items():
            ch["raw"] = raw[cid]
            ch["sub"] = dict(sub_scores.get(cid, {}))
        return {"id": int(query.id), "units": units, "chunks": chunks}

    def run(self, query: QueryItem, parallel_retrieval: bool = True) -> MultiQueryResult:
        record = self.augmented.get(int(query.id))
        lines_mode = self.config.mode == "lines"
        if lines_mode:
            texts = line_texts(record, self.config.max_lines) if record else []
        else:
            texts = unit_texts(record, self.config.plain_units) if record else []

        if not texts:
            if not self.config.fallback_to_baseline:
                raise KeyError(f"query {query.id} không có trong augmented.jsonl")
            return MultiQueryResult(
                result=self.pipeline.run(query, parallel_retrieval),
                n_unit=1, used_fallback=True,
            )

        trace = Trace(query_id=query.id, query_text=query.text)
        per_unit: List[List[Candidate]] = []
        branch_results: Dict[str, List[Candidate]] = {}
        # selection="branch-rerank": ứng viên của từng unit, giữ theo nhánh ngôn ngữ.
        unit_branches: List[Dict[str, List[Candidate]]] = []

        # Các lượt chạy tuần tự: mỗi lượt đã tự chạy song song 3 nhánh ngôn
        # ngữ rồi. Cho cả 4 lượt chạy song song nữa là nhân bốn số truy vấn
        # đồng thời vào Qdrant và số lần encode trên GPU - đáng làm nhưng phải
        # đo trước, không bật mặc định.
        # Các unit chỉ gọi Qdrant (vector đã encode sẵn) nên chạy song song được;
        # ex.map giữ đúng thứ tự unit.
        def one_unit(i_text):
            i, text = i_text
            unit_query = QueryItem(id=query.id, text=text, metadata=dict(query.metadata))
            return self.pipeline.retrieve_and_fuse(unit_query, trace, parallel_retrieval,
                                                   span_prefix=f"unit{i}")

        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(max_workers=min(4, len(texts))) as ex:
            unit_out = list(ex.map(one_unit, enumerate(texts)))
        for i, (branches, fused) in enumerate(unit_out):
            per_unit.append(fused)
            unit_branches.append(branches)
            # Giữ kết quả nhánh của lượt ĐẦU cho PipelineResult: đó là lượt
            # của câu hỏi chính, các lượt sau là mở rộng.
            if i == 0:
                branch_results = branches

        if self.pipeline.config.selection == "branch-rerank":
            return self._run_branch_rerank(query, record, texts, lines_mode,
                                           per_unit, unit_branches, branch_results, trace)

        if lines_mode:
            union, hits = rrf_union(per_unit, self.config.lines_rrf_k, self.config.lines_top_k)
        else:
            union, hits = union_candidates(per_unit)
        cap = self.config.max_rerank_candidates
        if cap and len(union) > cap:
            union = union[:cap]

        # Mặc định rerank bằng CÂU GỐC. Khi bật rerank_with_augmented thì dùng
        # chuỗi của unit ĐẦU: unit đầu là câu hỏi chính (nhánh ngắn) hoặc
        # sub-query thứ nhất (nhánh dài), và nó chứa đủ khối thuật ngữ dùng
        # chung. Không ghép cả 4 unit lại: chuỗi sẽ dài gấp bốn và lặp khối
        # thuật ngữ bốn lần.
        rerank_query = query
        if self.config.rerank_with_augmented:
            rerank_query = QueryItem(id=query.id, text=texts[0],
                                     metadata=dict(query.metadata))

        with trace.span("rerank.union", params=self.pipeline.reranker.get_params(),
                        n_in=len(union)) as span:
            reranked = self.pipeline.reranker.rerank(
                rerank_query, union, top_k=self.pipeline.config.rerank_top_k
            )
            span.n_out = len(reranked)

        trace.finish()
        return MultiQueryResult(
            result=PipelineResult(
                query=query,
                branch_results=branch_results,
                fused=union,
                reranked=reranked,
                trace=trace,
            ),
            n_unit=len(texts),
            n_union=len(union),
            n_overlap=sum(1 for v in hits.values() if v > 1),
            stats={
                "n_per_unit": [len(c) for c in per_unit],
                "is_long": bool(record.get("is_long")),
            },
        )
