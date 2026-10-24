"""Sửa chính tả câu hỏi gốc tiếng Việt, offline, cache ra jsonl.

Bước ĐẦU của nhánh query: chạy trước trích thực thể (augment) và dịch. Lỗi
chính tả lan theo dây chuyền - "tiểu đau và dắt" bị trích thành thực thể "dắt"
rồi dịch thành dyspareunia - nên phải sửa ở gốc.

Mỗi câu một lời gọi LLM (prompts/query_spelling_correction.txt). Few-shot cố
định ở prompts/query_spelling_correction_fewshots.jsonl: cùng LOẠI lỗi với các
case đã thấy nhưng là câu khác, để smoke trên chính các case đó vẫn đo được.

Câu gốc KHÔNG bị thay thế ở hạ nguồn: bản sửa là một biến thể riêng, vì model
có thể "sửa" nhầm thuật ngữ hiếm.

    python -m scripts.correct_queries --sample 20 --out data/spelling/smoke20.jsonl --show
    python -m scripts.correct_queries                       # cả 1.200 câu, chạy tiếp được
"""
import argparse
import difflib
import json
import os
import random
import re
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.augment.config import AugmentConfig
from src.rag.augment.steps import json_object, make_chat
from src.rag.labeling import prompts as prompt_store

QUERIES = "data/queries/real_queries.jsonl"
AUGMENTED = "data/augment_v2_full/augmented.jsonl"
PROMPT = "query_spelling_correction"
FEWSHOTS = "prompts/query_spelling_correction_fewshots.jsonl"
DOC_TEXTS = "data/doc_texts.jsonl"
CORPUS_CACHE = "data/spelling/corpus_vi_norm.txt"

_TOKEN = re.compile(r"\S+")
_WORD = re.compile(r"\w+", re.U)
# Tách cả khoảng trắng để dựng lại câu đúng nguyên dạng (giữ xuống dòng).
_PIECE = re.compile(r"\s+|\S+")
_NONLETTER = re.compile(r"[\W_]+", re.U)

# Viết tắt kiểu chat mà luật "dãy con" không bắt được ("oy" không phải dãy
# con của "roi"). Khoá đã bỏ dấu, viết thường.
ABBREV = {"oy": "rồi", "ko": "không", "k": "không", "kh": "không", "dc": "được", "đc": "được",
          "j": "gì", "bt": "bình thường", "xn": "xét nghiệm", "trc": "trước", "e": "em",
          "vs": "với", "ntn": "như thế nào", "bs": "bác sĩ", "sd": "sử dụng", "mk": "mình"}


def nfc(s):
    return unicodedata.normalize("NFC", s or "")


def bare(s):
    """Thường hoá, bỏ dấu, bỏ ký tự không phải chữ/số: "Trước," -> "truoc"."""
    s = unicodedata.normalize("NFD", s.lower()).replace("đ", "d").replace("đ", "d")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").replace("đ", "d")
    return _NONLETTER.sub("", s)


def _subseq(a, b):
    it = iter(b)
    return all(c in it for c in a)


def edit_kind(old, new):
    """Loại của một chỗ sửa, hoặc None nếu không thuộc loại nào được chấp nhận.

    Không dùng ngưỡng độ giống cho cả câu: nó phụ thuộc độ dài câu, cùng một
    chỗ đoán bừa ("lang thận" -> "thận ứ nước") ra 0.86 ở câu 42 ký tự nhưng
    ~0.97 ở câu 300 ký tự (đo trên smoke 30 câu).
    """
    # Theo TỪ, đã tách dấu câu: "dắt.Em" và "rắt. Em" đều là hai từ.
    ow, nw = _WORD.findall(old), _WORD.findall(new)
    if bare(old) == bare(new):
        return "format"            # hoa/thường, dấu câu, khoảng trắng, dạng Unicode
    if not ow or not nw:
        return None                # chèn hoặc xoá chữ
    if len(ow) == 1 and ABBREV.get(bare(old)) == " ".join(nw).lower():
        return "abbrev"
    ob, nb = bare(old), bare(new)
    # Bung viết tắt: chữ cũ là dãy con của chữ mới, cùng chữ cái đầu, và không
    # phải trường hợp chèn thêm từ quanh một từ giữ nguyên ("bệnh." -> "bệnh không.").
    if (len(ow) <= 2 and len(ob) < len(nb) and ob[:1] == nb[:1] and _subseq(ob, nb)
            and not {bare(w) for w in ow} <= {bare(w) for w in nw}):
        return "abbrev"
    # Lỗi gõ: cùng số từ, từng cặp từ hoặc giống hệt hoặc gần giống.
    if len(ow) == len(nw) and all(
            bare(o) == bare(n) or difflib.SequenceMatcher(a=bare(o), b=bare(n)).ratio() >= 0.5
            for o, n in zip(ow, nw)):
        return "typo"
    return None


class CorpusCheck:
    """Đối chiếu một chỗ sửa với tần suất cụm từ trong corpus tiếng Việt.

    Luật edit_kind() chỉ nhìn độ giống ký tự nên không tách được lỗi gõ thật
    ("1 lầm với" -> "lần") khỏi việc thay một từ ĐÚNG bằng từ khác ("vôi tuyến
    tùng" -> "tụy"): hai cặp giống nhau 0.67 và 0.57. Corpus phân biệt được:
    người viết thật có dùng cụm cũ hay không.

    Corpus = phần vi của data/doc_texts.jsonl, viết thường, NFC, mọi ký tự
    không phải chữ/số thành một khoảng trắng, mỗi doc một dòng có đệm khoảng
    trắng hai đầu - để đếm " a b c " là khớp đúng ranh giới từ. Cache ra
    CORPUS_CACHE.
    """

    MIN_NEW = 3     # cụm mới phải xuất hiện ít nhất chừng này lần
    RATIO = 3       # và gấp chừng này lần cụm cũ

    def __init__(self, path=DOC_TEXTS, cache=CORPUS_CACHE):
        if not os.path.exists(cache):
            t0 = time.time()
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            with open(path, encoding="utf-8") as f, open(cache, "w", encoding="utf-8") as out:
                for line in f:
                    r = json.loads(line)
                    if r.get("language") == "vi":
                        out.write(" " + _NONLETTER.sub(" ", nfc(r["text"]).lower()) + " \n")
            print(f"[spelling] dựng corpus đếm {cache} trong {time.time() - t0:.0f}s")
        with open(cache, encoding="utf-8") as f:
            self.text = f.read()
        self._counts = {}
        self._lock = threading.Lock()

    def count(self, words):
        key = " ".join(words)
        with self._lock:
            if key not in self._counts:
                self._counts[key] = self.text.count(f" {key} ")
            return self._counts[key]

    def verify(self, old_w, new_w, left_old, right_old, left_new, right_new):
        """-> (nhận?, mô tả bằng chứng). Ngữ cảnh cụm mới lấy từ câu LLM trả về:
        câu gõ không dấu thì ngữ cảnh gốc cũng không dấu, ghép với chữ đã
        thêm dấu sẽ không bao giờ khớp corpus."""
        windows = [(left_old[-1:] + old_w + right_old[:1], left_new[-1:] + new_w + right_new[:1]),
                   (left_old[-2:] + old_w, left_new[-2:] + new_w)]
        for wo, wn in windows:
            co, cn = self.count(wo), self.count(wn)
            if co or cn:
                break
        ok = cn >= self.MIN_NEW and cn >= self.RATIO * co
        return ok, f"'{' '.join(wo)}'={co} vs '{' '.join(wn)}'={cn}"


def _letters_changed(old, new):
    """Chữ (kể cả dấu thanh) thay đổi, không chỉ hoa/thường hay dấu câu."""
    norm = lambda s: _NONLETTER.sub("", nfc(s).lower())
    return norm(old) != norm(new)


def _words(s):
    return [w.lower() for w in _WORD.findall(nfc(s))]


def apply_rule(orig, fixed, corpus=None):
    """Giữ từng chỗ sửa hợp lệ, trả lại chữ gốc ở chỗ không hợp lệ.
    corpus: CorpusCheck để kiểm thêm chỗ sửa loại typo/đổi dấu; None = bỏ qua.
    -> (câu cuối, [chỗ nhận], [chỗ trả về])."""
    a, b = _PIECE.findall(orig), _PIECE.findall(fixed)
    # Gộp các chỗ sửa chỉ cách nhau khoảng trắng thành MỘT đơn vị: difflib cắt
    # "lang thận" -> "thận ứ nước" làm hai mảnh, xét riêng thì mảnh đầu lọt qua
    # luật lỗi gõ và câu ra "thận thận".
    groups = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        is_gap = op == "equal" and not "".join(a[i1:i2]).strip()
        if groups and groups[-1][0] == "edit" and (op != "equal" or is_gap):
            groups[-1][2], groups[-1][4] = i2, j2
        else:
            groups.append(["edit" if op != "equal" else "equal", i1, i2, j1, j2])
    out, kept, reverted = [], [], []
    for op, i1, i2, j1, j2 in groups:
        old, new = "".join(a[i1:i2]), "".join(b[j1:j2])
        if op == "equal" or not (old.strip() or new.strip()):
            out.append(new if op != "equal" else old)
            continue
        kind = edit_kind(old.strip(), new.strip())
        label = f"{old.strip() or '∅'} -> {new.strip() or '∅'}"
        if corpus and kind in ("typo", "format") and _letters_changed(old, new):
            ok, evidence = corpus.verify(_words(old), _words(new),
                                         _words("".join(a[:i1])), _words("".join(a[i2:])),
                                         _words("".join(b[:j1])), _words("".join(b[j2:])))
            label = f"{label} {{{evidence}}}"
            if not ok:
                kind = None
        if kind:
            out.append(new)
            kept.append(f"{label} [{kind}]")
        else:
            out.append(old)
            reverted.append(label)
    return "".join(out), kept, reverted


def few_shots_text(path=FEWSHOTS):
    shots = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    lines = []
    for s in shots:
        lines.append(f"Input: {s['query']}")
        lines.append("Output: " + json.dumps({"corrected_query": s["corrected_query"]}, ensure_ascii=False))
    return "\n".join(lines)


def build_prompt(template, query, shots_text):
    return prompt_store.fill(template, query=query, few_shots=shots_text)


def word_edits(before, after):
    """Các chỗ sửa theo từ: ["dắt -> rắt", ...]."""
    a, b = _TOKEN.findall(before), _TOKEN.findall(after)
    edits = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op != "equal":
            edits.append(f"{' '.join(a[i1:i2]) or '∅'} -> {' '.join(b[j1:j2]) or '∅'}")
    return edits


def check(before, after):
    """Cờ hỏng định dạng. Sửa quá tay do apply_rule() xử lý theo từng chỗ sửa."""
    flags = []
    if not after:
        return ["empty"]
    if re.search(r"[一-鿿]", after):
        flags.append("lang_change")
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/spelling/corrected_queries.jsonl")
    ap.add_argument("--sample", type=int, default=0,
                    help="N câu ngẫu nhiên nửa ngắn nửa dài; seed 7 = đúng 20 câu smoke của nhánh dịch")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ids", default=None, help="danh sách id, ví dụ 358,57,129")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--llm-url", default=None)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-corpus-check", action="store_true",
                    help="tắt bước đối chiếu tần suất cụm từ trong corpus (CorpusCheck)")
    args = ap.parse_args()

    cfg = AugmentConfig()
    if args.llm_url:
        cfg.llm_url = args.llm_url
    chat = make_chat(cfg)
    template = prompt_store.load(PROMPT)
    shots_text = few_shots_text()
    corpus = None if args.no_corpus_check else CorpusCheck()
    queries = [json.loads(l) for l in open(QUERIES, encoding="utf-8")]

    if args.ids:
        want = {int(x) for x in args.ids.split(",")}
        queries = [q for q in queries if int(q["id"]) in want]
    elif args.sample:
        # Cùng cách chọn với scripts/translate_queries.py để so được hai bước trên cùng câu.
        augmented = {int(json.loads(l)["qid"]): json.loads(l) for l in open(AUGMENTED, encoding="utf-8")}
        rnd = random.Random(args.seed)
        long_ = [q for q in queries if augmented.get(int(q["id"]), {}).get("is_long")]
        short = [q for q in queries if not augmented.get(int(q["id"]), {}).get("is_long")]
        half = args.sample // 2
        queries = rnd.sample(short, args.sample - half) + rnd.sample(long_, half)

    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["id"] for l in open(args.out, encoding="utf-8")}
    todo = [q for q in queries if q["id"] not in done]
    print(f"[spelling] {len(queries)} câu, đã có {len(done)}, còn {len(todo)} | {cfg.llm_model} @ {cfg.llm_url}")

    def one(q):
        t = time.time()
        # NFC trước mọi thứ: 4/1.200 câu lưu dạng tổ hợp (NFD), diff sẽ báo
        # "ngày -> ngày" dù nhìn giống hệt, và token sparse không khớp tài liệu.
        src = nfc(q["query"])
        raw = chat(build_prompt(template, src, shots_text), args.max_tokens)
        obj = json_object(raw) or {}
        llm = nfc((obj.get("corrected_query") or "").strip())
        flags = check(src, llm) if obj else ["parse_fail"]
        final, kept, reverted = apply_rule(src, llm, corpus) if llm and not flags else (src, [], [])
        if reverted:
            flags.append("reverted_edit")
        return {"id": q["id"], "query": q["query"], "corrected_query": final,
                "changed": final != q["query"], "nfc_changed": src != q["query"],
                "llm_query": llm, "edits_kept": kept, "edits_reverted": reverted,
                "flags": flags, "latency_s": round(time.time() - t, 2),
                "raw": None if obj else raw}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0, n, n_changed, n_flag = time.time(), 0, 0, 0
    with open(args.out, "a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for r in ex.map(one, todo):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
            n_changed += r["changed"]
            n_flag += bool(r["flags"])
            if args.show:
                if not (r["edits_kept"] or r["edits_reverted"] or r["flags"]):
                    continue
                print(f"\n--- id={r['id']} {r['flags'] or ''}")
                print(f"GỐC: {r['query']}")
                if r["edits_kept"]:
                    print(f"  nhận   : {' | '.join(r['edits_kept'])}")
                if r["edits_reverted"]:
                    print(f"  TRẢ VỀ : {' | '.join(r['edits_reverted'])}")
            elif n % 100 == 0:
                print(f"  {n}/{len(todo)} | sửa {n_changed} | gắn cờ {n_flag}", flush=True)

    print(f"\n[spelling] xong {n} câu trong {time.time() - t0:.0f}s | sửa {n_changed} | "
          f"gắn cờ {n_flag} -> {args.out}")


if __name__ == "__main__":
    main()
