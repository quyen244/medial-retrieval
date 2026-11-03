"""Build a self-contained, offline relevance-labeling page for one query."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL = ROOT / "data/T28_rrf_bge_q3e/w0.6_qc_vi90_bge_pool.jsonl"
COMPARE_POOL = ROOT / "data/T28_rrf_bge_q3e/w0.6_qc_vi90_bge_q3etext2048_pool.jsonl"
LONG_TEXTS = ROOT / "data/doc_texts_8k.jsonl"
RAW_INDEX = ROOT / "data/raw_documents_index.json"


def find_query(pool: Path, query_id: int) -> dict:
    with pool.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if int(row["id"]) == query_id:
                return row
    raise ValueError(f"Query {query_id} not found in {pool}")


def load_long_texts(doc_ids: set[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    with LONG_TEXTS.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            doc_id = str(row["doc_id"])
            if doc_id in doc_ids:
                found[doc_id] = row.get("text") or ""
                if len(found) == len(doc_ids):
                    break
    return found


def load_qdrant_texts(doc_ids: set[str]) -> dict[str, str]:
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    client = QdrantClient(url="http://127.0.0.1:6333", timeout=30)
    found: dict[str, str] = {}
    ids = sorted(doc_ids, key=int)
    for start in range(0, len(ids), 100):
        batch = [int(doc_id) for doc_id in ids[start:start + 100]]
        points, _ = client.scroll(
            "medical_docs",
            scroll_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchAny(any=batch))]),
            limit=len(batch) * 2,
            with_payload=["doc_id", "text"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            found[str(payload["doc_id"])] = payload.get("text") or ""
    missing = sorted(doc_id for doc_id in doc_ids if not found.get(doc_id))
    if missing:
        raise ValueError(f"{len(missing)} docs missing Qdrant text, e.g. {missing[:5]}")
    return found


def build_data(pool: Path, query_id: int, compare_pool: Path | None,
               qdrant_texts: dict[str, str] | None = None,
               long_texts: dict[str, str] | None = None,
               raw_index: dict | None = None,
               nav_ids: list[int] | None = None) -> dict:
    row = find_query(pool, query_id)
    candidates = row["candidates"][:300]
    if len(candidates) != 300:
        raise ValueError(f"Query {query_id} has {len(candidates)} candidates, expected 300")
    if len({str(c["doc_id"]) for c in candidates}) != 300:
        raise ValueError("Candidate doc IDs are not unique")
    if any(candidates[i]["rerank_score"] < candidates[i + 1]["rerank_score"]
           for i in range(299)):
        raise ValueError("Candidates are not ordered by descending rerank score")

    doc_ids = {str(c["doc_id"]) for c in candidates}
    compare = None
    compare_rank = {}
    if compare_pool and compare_pool.is_file():
        try:
            compare = find_query(compare_pool, query_id)
        except ValueError:
            compare = None
        if compare:
            other = compare["candidates"][:300]
            if len(other) != 300 or {str(c["doc_id"]) for c in other} != doc_ids:
                raise ValueError("The 2048 pool does not contain the same 300 doc IDs")
            compare_rank = {str(c["doc_id"]): (rank, c["rerank_score"])
                            for rank, c in enumerate(other, 1)}
    if qdrant_texts is None:
        qdrant_texts = load_qdrant_texts(doc_ids)
    if long_texts is None:
        long_texts = load_long_texts(doc_ids) if compare else {}
    missing_qdrant = sorted(doc_id for doc_id in doc_ids if not qdrant_texts.get(doc_id))
    if missing_qdrant:
        raise ValueError(f"{len(missing_qdrant)} docs missing Qdrant text, e.g. {missing_qdrant[:5]}")
    if compare:
        missing = sorted(doc_id for doc_id in doc_ids if not long_texts.get(doc_id))
        if missing:
            raise ValueError(f"{len(missing)} docs missing long text, e.g. {missing[:5]}")
    if raw_index is None:
        raw_index = json.loads(RAW_INDEX.read_text(encoding="utf-8"))
    docs = []
    for rank, candidate in enumerate(candidates, 1):
        doc_id = str(candidate["doc_id"])
        domain = raw_index.get(doc_id, "")
        path = ROOT / "data/raw_documents" / domain / f"{doc_id}.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        source = raw.get("source") or {}
        content = raw.get("content") or {}
        docs.append({
            "rank": rank,
            "doc_id": doc_id,
            "score": candidate["rerank_score"],
            "rank_2048": compare_rank.get(doc_id, (None, None))[0],
            "score_2048": compare_rank.get(doc_id, (None, None))[1],
            "language": candidate.get("language") or source.get("language_category") or "?",
            "title": source.get("title") or candidate.get("title") or f"Doc {doc_id}",
            "domain": source.get("domain") or domain,
            "url": source.get("final_url") or source.get("requested_url") or "",
            "bge_text_1024": qdrant_texts[doc_id],
            "bge_text_2048": long_texts.get(doc_id, ""),
            "body": content.get("markdown") or "",
        })

    fingerprint = hashlib.sha256(
        (str(pool.relative_to(ROOT)) + "|" + str(query_id) + "|" +
         ",".join(d["doc_id"] for d in docs)).encode("utf-8")
    ).hexdigest()[:12]
    return {
        "query_id": query_id,
        "query": row["query"],
        "pool": str(pool.relative_to(ROOT)).replace("\\", "/"),
        "reranker": row.get("rr_model") or "BAAI/bge-reranker-v2-m3",
        "compare_pool": str(compare_pool.relative_to(ROOT)).replace("\\", "/") if compare else None,
        "compare_rr_len": compare.get("rr_len") if compare else None,
        "text_source_1024": "Qdrant medical_docs.payload.text",
        "text_source_2048": "data/doc_texts_8k.jsonl" if compare else None,
        "fingerprint": fingerprint,
        "nav_ids": nav_ids or [query_id],
        "docs": docs,
    }


HTML = r'''<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Đánh nhãn doc - ViBioMIR</title>
<style>
:root { color-scheme: light; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background:#f4f6f7; color:#17242b; }
* { box-sizing: border-box; }
body { margin:0; }
button, input, textarea, select { font:inherit; }
button { cursor:pointer; }
.top { padding:18px 24px; background:#102b36; color:#fff; }
.topline { display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap; }
h1 { margin:0; font-size:21px; letter-spacing:-.03em; }
.sub { color:#bdd2d7; font-size:13px; margin-top:4px; }
.actions { display:flex; gap:8px; flex-wrap:wrap; }
.actions button, .actions label { border:1px solid #7e9da5; background:transparent; color:#fff; border-radius:7px; padding:7px 11px; font-size:13px; cursor:pointer; }
.actions button:hover, .actions label:hover { background:#254954; }
.querynav { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-top:12px; }
.querynav a, .querynav select { border:1px solid #7e9da5; background:#173944; color:#fff; border-radius:7px; padding:6px 9px; font-size:13px; text-decoration:none; }
.querynav a:hover { background:#254954; }
.querynav a[aria-disabled="true"] { opacity:.45; pointer-events:none; }
.query { max-width:1400px; margin:14px auto 0; padding:15px 17px; background:#23414b; border-radius:9px; line-height:1.52; }
.query strong { display:block; color:#9dd8ca; margin-bottom:5px; font-size:12px; text-transform:uppercase; letter-spacing:.08em; }
.query .original { color:#c3d7db; margin-top:8px; font-size:12px; }
.wrap { max-width:1600px; margin:auto; padding:16px; display:grid; grid-template-columns:390px minmax(0,1fr); gap:16px; min-height:calc(100vh - 190px); }
.panel { background:#fff; border:1px solid #dce4e7; border-radius:10px; min-width:0; }
.left { display:flex; flex-direction:column; height:calc(100vh - 220px); min-height:520px; position:sticky; top:16px; }
.controls { padding:14px; border-bottom:1px solid #e3e9eb; }
.stats { display:flex; gap:7px; flex-wrap:wrap; margin-bottom:10px; font-size:12px; }
.stat { padding:5px 7px; background:#edf3f4; border-radius:5px; }
.stat.good { background:#dcefe4; color:#15593a; }
.stat.bad { background:#fbe8e4; color:#a74431; }
.controls select, .controls input { width:100%; padding:8px 10px; background:#fff; border:1px solid #bac9cf; border-radius:6px; margin:4px 0; }
.controlrow { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
.hint { margin:7px 0 0; color:#577078; font-size:12px; line-height:1.4; }
.list { overflow:auto; flex:1; }
.item { display:block; width:100%; text-align:left; border:0; border-bottom:1px solid #e8eef0; padding:11px 13px; background:#fff; color:inherit; }
.item:hover { background:#f4f8f8; }
.item.active { background:#e8f4f1; box-shadow:inset 3px 0 #187d6d; }
.itemtop { display:flex; justify-content:space-between; align-items:center; gap:8px; color:#5c737b; font-size:11px; }
.itemtitle { font-weight:650; font-size:13px; line-height:1.35; margin:5px 0; }
.badge { font-weight:700; border-radius:4px; padding:2px 5px; white-space:nowrap; }
.badge.relevant { color:#17603e; background:#d8efe2; }
.badge.irrelevant { color:#a34130; background:#fbe5df; }
.badge.unsure { color:#765d12; background:#fff1bf; }
.badge.empty { color:#61747b; background:#edf1f2; }
.pager { display:flex; justify-content:space-between; align-items:center; gap:8px; padding:10px 12px; border-top:1px solid #e3e9eb; font-size:12px; }
.pager button { border:1px solid #c5d2d5; background:#fff; border-radius:5px; padding:5px 9px; }
.pager button:disabled { opacity:.4; cursor:default; }
.right { padding:20px; min-height:620px; }
.dochead { display:flex; justify-content:space-between; gap:12px; align-items:flex-start; }
.dochead h2 { font-size:22px; line-height:1.3; margin:4px 0 7px; letter-spacing:-.025em; }
.meta { color:#5a727a; font-size:12px; line-height:1.5; }
.meta a { color:#126e87; word-break:break-all; }
.labelrow { display:flex; gap:8px; flex-wrap:wrap; margin:18px 0 12px; }
.labelrow button { border:1px solid transparent; border-radius:7px; padding:9px 15px; font-weight:700; }
.labelrow .yes { background:#dbefe4; color:#165a3b; }
.labelrow .no { background:#fae7e2; color:#a14331; }
.labelrow .maybe { background:#fff1c8; color:#725a12; }
.labelrow button.chosen { outline:2px solid currentColor; outline-offset:1px; }
.labelrow .clear { background:#eef2f3; color:#3e5962; }
.note { width:100%; min-height:54px; resize:vertical; padding:9px; border:1px solid #cbd7db; border-radius:6px; }
.texts { display:grid; grid-template-columns:minmax(0, .8fr) minmax(0, 1.2fr); gap:14px; margin-top:18px; }
.textcol { min-width:0; }
.textcol h3 { font-size:13px; margin:0; }
.texthead { display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:7px; }
.copy { border:1px solid #c9d5d9; background:#fff; color:#2c4650; border-radius:6px; padding:3px 10px; font-size:12px; cursor:pointer; white-space:nowrap; }
.copy:hover { background:#eef3f5; } .copy:disabled { opacity:.45; cursor:default; }
.textcol .origin { color:#6b7d83; font-size:11px; margin-bottom:7px; }
.textbody { border:1px solid #dce4e7; background:#fafcfc; border-radius:7px; padding:12px; white-space:pre-wrap; overflow:auto; height:calc(100vh - 470px); min-height:330px; font-size:13px; line-height:1.6; overflow-wrap:anywhere; }
.notice { color:#506a72; font-size:12px; line-height:1.45; margin-top:12px; }
.emptymsg { padding:24px; color:#667c84; }
@media (max-width:950px) { .wrap { grid-template-columns:1fr; } .left { position:static; height:430px; min-height:0; } .texts { grid-template-columns:1fr; } .textbody { height:300px; } }
</style>
</head>
<body>
<header class="top">
 <div class="topline"><div><h1>Đánh nhãn tài liệu · Query <span id="qid"></span></h1><div class="sub">300 doc theo thứ tự điểm BGE rerank · làm từng đợt, dừng lúc nào cũng được</div></div>
 <div class="actions"><button id="export" type="button">Xuất nhãn JSON</button><label for="import">Nhập nhãn JSON</label><input id="import" type="file" accept="application/json,.json" hidden></div></div>
 <nav class="querynav" aria-label="Chuyển query"><a id="prev-query" href="#">← Query trước</a><select id="query-switch" aria-label="Chọn query"></select><a id="next-query" href="#">Query tiếp →</a></nav>
 <div class="query"><strong>Query dùng để rerank</strong><div id="query"></div><div class="original" id="query-note"></div></div>
</header>
<main class="wrap">
 <aside class="panel left"><div class="controls">
  <div class="stats"><span class="stat good" id="yes-count"></span><span class="stat bad" id="no-count"></span><span class="stat" id="other-count"></span><span class="stat" id="sample-count"></span></div>
  <select id="ranking" aria-label="Thứ tự rerank"><option value="2048">BGE 2048 · text dài (dùng để gán)</option><option value="1024">BGE 1024 · text cũ</option></select>
  <div class="controlrow"><select id="scope" aria-label="Phạm vi"><option value="sample">Mẫu nhanh 30</option><option value="all">Tất cả 300</option></select><select id="status" aria-label="Lọc nhãn"><option value="all">Mọi trạng thái</option><option value="empty">Chưa gán nhãn</option><option value="relevant">Đúng</option><option value="irrelevant">Sai</option><option value="unsure">Một nửa</option></select></div>
  <input id="search" type="search" placeholder="Tìm theo tên hoặc doc ID" aria-label="Tìm tài liệu">
  <div class="hint">Đổi thứ tự 1024/2048 không làm mất nhãn: cả hai dùng cùng 300 doc. Mẫu nhanh rải đều theo hạng; chuyển sang toàn bộ khi cần.</div>
 </div><div class="list" id="list"></div><div class="pager"><button id="prev" type="button">← Trước</button><span id="page"></span><button id="next" type="button">Tiếp →</button></div></aside>
 <section class="panel right" id="detail"><div class="emptymsg">Chọn một tài liệu để bắt đầu.</div></section>
</main>
<script type="application/json" id="dataset">__DATA__</script>
<script>
(() => {
 const data = JSON.parse(document.getElementById('dataset').textContent);
 const docs = data.docs;
 const sampleRanks = new Set([1,2,3,4,5,6,7,8,9,10,21,30,40,50,60,70,80,90,100,120,170,185,200,215,230,245,260,275,290,300]);
 const key = `vibiomir-labels:${data.fingerprint}`;
 const statuses = { relevant:'Đúng', irrelevant:'Sai', unsure:'Một nửa' };
 let labels = {};
 try { labels = JSON.parse(localStorage.getItem(key) || '{}'); } catch { labels = {}; }
 let selected = null, page = 0;
 const $ = id => document.getElementById(id);
 $('qid').textContent = data.query_id;
 $('query').textContent = data.query;
 $('query-note').textContent = `Nguồn: ${data.pool} · ${data.reranker}`;
 const navIds = data.nav_ids || [data.query_id];
 for (const id of navIds) {
   const option = document.createElement('option'); option.value = id; option.textContent = `Query ${id}`;
   $('query-switch').append(option);
 }
 $('query-switch').value = String(data.query_id);
 $('query-switch').addEventListener('change', e => { location.href = `query_${e.target.value}.html`; });
 const navIndex = navIds.indexOf(data.query_id);
 for (const [element, target] of [[$('prev-query'), navIds[navIndex - 1]], [$('next-query'), navIds[navIndex + 1]]]) {
   if (target) element.href = `query_${target}.html`;
   else { element.removeAttribute('href'); element.setAttribute('aria-disabled', 'true'); }
 }
 if (!data.compare_pool) { $('ranking').value='1024'; $('ranking').disabled=true; }
 const rankOf = d => $('ranking').value==='2048' && d.rank_2048 ? d.rank_2048 : d.rank;
 const scoreOf = d => $('ranking').value==='2048' && d.score_2048 != null ? d.score_2048 : d.score;
 const getLabel = d => labels[d.doc_id] || {};
 const visible = () => {
   const scope = $('scope').value, status = $('status').value, term = $('search').value.trim().toLocaleLowerCase();
   return docs.filter(d => (scope === 'all' || sampleRanks.has(rankOf(d))) &&
     (status === 'all' || (getLabel(d).label || 'empty') === status) &&
     (!term || d.title.toLocaleLowerCase().includes(term) || d.doc_id.includes(term)))
     .sort((a,b)=>rankOf(a)-rankOf(b));
 };
 function save() {
   try { localStorage.setItem(key, JSON.stringify(labels)); }
   catch { alert('Trình duyệt không lưu được nhãn. Hãy dùng nút Xuất nhãn JSON.'); }
 }
 function renderCounts() {
   const counts = {relevant:0,irrelevant:0,unsure:0};
   docs.forEach(d => { const s = getLabel(d).label; if (s in counts) counts[s]++; });
   $('yes-count').textContent = `Đúng ${counts.relevant}`;
   $('no-count').textContent = `Sai ${counts.irrelevant}`;
   $('other-count').textContent = `Một nửa ${counts.unsure} · Đã gán ${counts.relevant+counts.irrelevant+counts.unsure}/300`;
   $('sample-count').textContent = `Mẫu ${docs.filter(d => sampleRanks.has(rankOf(d)) && getLabel(d).label).length}/30`;
 }
 function renderList() {
   const rows = visible(), size = $('scope').value === 'sample' ? 10 : 20;
   const pages = Math.max(1, Math.ceil(rows.length / size));
   page = Math.min(page, pages - 1);
   $('list').replaceChildren();
   for (const d of rows.slice(page * size, (page + 1) * size)) {
     const button = document.createElement('button'); button.className = 'item' + (selected === d.doc_id ? ' active' : ''); button.type = 'button';
     const top = document.createElement('div'); top.className = 'itemtop';
     const info = document.createElement('span'); info.textContent = `#${rankOf(d)} · ${d.doc_id} · ${d.language} · BGE ${scoreOf(d).toFixed(4)}`;
     const badge = document.createElement('span'); const state = getLabel(d).label || 'empty'; badge.className = `badge ${state}`; badge.textContent = statuses[state] || 'Chưa gán';
     top.append(info,badge);
     const title = document.createElement('div'); title.className = 'itemtitle'; title.textContent = d.title;
     button.append(top,title); button.addEventListener('click', () => select(d.doc_id)); $('list').append(button);
   }
   if (!rows.length) { const empty = document.createElement('div'); empty.className='emptymsg'; empty.textContent='Không có doc trong bộ lọc này.'; $('list').append(empty); }
   $('page').textContent = `${rows.length} doc · trang ${page+1}/${pages}`;
   $('prev').disabled = page === 0; $('next').disabled = page >= pages-1;
 }
 function select(docId) {
   selected = docId; const d = docs.find(x => x.doc_id === docId); if (!d) return;
   renderList(); const detail = $('detail'); detail.replaceChildren();
   const head = document.createElement('div'); head.className='dochead';
   const h = document.createElement('div');
   const meta = document.createElement('div'); meta.className='meta'; meta.textContent=`#${rankOf(d)}/300 · ID ${d.doc_id} · ${d.language} · ${d.domain} · BGE ${scoreOf(d).toFixed(6)} · hạng 1024: ${d.rank}${d.rank_2048 ? ` · hạng 2048: ${d.rank_2048}` : ''}`;
   const title = document.createElement('h2'); title.textContent=d.title; h.append(meta,title);
   if (/^https?:\/\//i.test(d.url)) { const a=document.createElement('a'); a.href=d.url; a.target='_blank'; a.rel='noopener noreferrer'; a.textContent='Mở trang gốc ↗'; meta.append(' · ',a); }
   head.append(h); detail.append(head);
   const row=document.createElement('div'); row.className='labelrow';
   for (const [value,text,cls] of [['relevant','1 · Đúng','yes'],['irrelevant','2 · Sai','no'],['unsure','3 · Một nửa','maybe'],['','Bỏ nhãn','clear']]) {
     const b=document.createElement('button'); b.type='button'; b.className=cls + (getLabel(d).label === value && value ? ' chosen' : ''); b.textContent=text;
     b.addEventListener('click',()=>setLabel(d,value)); row.append(b);
   }
   detail.append(row);
   const note=document.createElement('textarea'); note.className='note'; note.placeholder='Ghi chú ngắn, tùy chọn'; note.value=getLabel(d).note || '';
   note.addEventListener('input',()=>{ labels[d.doc_id]={...getLabel(d),note:note.value,updated_at:new Date().toISOString()}; save(); }); detail.append(note);
   const texts=document.createElement('div'); texts.className='texts';
   const isLong = $('ranking').value === '2048' && data.compare_pool;
   const bgeText = isLong ? d.bge_text_2048 : d.bge_text_1024;
   const textSource = isLong ? data.text_source_2048 : data.text_source_1024;
   for (const [heading,origin,value] of [['BGE text đã dùng để rerank',`Nguồn: ${textSource}`,bgeText],['Thân bài đầy đủ','Markdown gốc đã crawl',d.body]]) {
     const col=document.createElement('section'); col.className='textcol'; const h3=document.createElement('h3'); h3.textContent=heading; const src=document.createElement('div'); src.className='origin'; src.textContent=origin;
     const head=document.createElement('div'); head.className='texthead';
     const copy=document.createElement('button'); copy.type='button'; copy.className='copy'; copy.textContent='Copy'; copy.disabled=!value;
     copy.addEventListener('click',async()=>{
       try { await navigator.clipboard.writeText(value); }
       catch { const ta=document.createElement('textarea'); ta.value=value; document.body.append(ta); ta.select(); document.execCommand('copy'); ta.remove(); }
       copy.textContent='Đã copy ✓'; setTimeout(()=>{copy.textContent='Copy';},1500); copy.blur();
     });
     head.append(h3,copy);
     const body=document.createElement('div'); body.className='textbody'; body.textContent=value || 'Không tìm thấy văn bản trong dữ liệu tại chỗ.';
     col.append(head,src,body); texts.append(col);
   }
   detail.append(texts);
   const notice=document.createElement('div'); notice.className='notice'; notice.textContent='Phím 1 = Đúng, 2 = Sai, 3 = Một nửa; ←/→ = chuyển doc trong danh sách đang xem. Có thể dừng sau 10 hoặc 20 doc rồi xuất JSON.'; detail.append(notice);
 }
 function setLabel(d,value) {
   const existing=getLabel(d);
   if (value) labels[d.doc_id]={...existing,label:value,updated_at:new Date().toISOString()};
   else if (existing.note) labels[d.doc_id]={note:existing.note,updated_at:new Date().toISOString()};
   else delete labels[d.doc_id];
   save(); renderCounts(); renderList(); select(d.doc_id);
   if (value) move(1);
 }
 function move(step) {
   const rows=visible(); const i=rows.findIndex(d=>d.doc_id===selected); const n=i+step;
   if (n<0 || n>=rows.length) return;
   const size=$('scope').value==='sample'?10:20; page=Math.floor(n/size); select(rows[n].doc_id);
 }
 function exportLabels() {
   const records=docs.filter(d=>getLabel(d).label || getLabel(d).note).map(d=>({rank_1024:d.rank,rank_2048:d.rank_2048,doc_id:d.doc_id,label:getLabel(d).label || null,note:getLabel(d).note || '',updated_at:getLabel(d).updated_at || null}));
   const payload={query_id:data.query_id,query:data.query,pool:data.pool,compare_pool:data.compare_pool,fingerprint:data.fingerprint,exported_at:new Date().toISOString(),labels:records};
   const blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}); const url=URL.createObjectURL(blob); const a=document.createElement('a');
   a.href=url; a.download=`labels_query_${data.query_id}_${data.fingerprint}.json`; a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
 }
 async function importLabels(file) {
   try {
     const payload=JSON.parse(await file.text());
     if (payload.query_id!==data.query_id || payload.fingerprint!==data.fingerprint || !Array.isArray(payload.labels)) throw Error('File nhãn không khớp query và thứ tự doc của trang này.');
     const ids=new Set(docs.map(d=>d.doc_id)); const next={...labels};
     for (const r of payload.labels) {
       if (!ids.has(String(r.doc_id)) || (r.label && !(r.label in statuses))) throw Error('File có doc ID hoặc nhãn không hợp lệ.');
       next[String(r.doc_id)]={label:r.label || undefined,note:String(r.note || ''),updated_at:r.updated_at || new Date().toISOString()};
     }
     labels=next; save(); renderCounts(); renderList(); if(selected) select(selected);
   } catch(e) { alert(e.message || 'Không đọc được file nhãn.'); }
 }
 $('ranking').addEventListener('change',()=>{page=0;renderCounts();renderList();const first=visible()[0];if(first)select(first.doc_id);});
 $('scope').addEventListener('change',()=>{page=0;renderList();}); $('status').addEventListener('change',()=>{page=0;renderList();}); $('search').addEventListener('input',()=>{page=0;renderList();});
 $('prev').addEventListener('click',()=>{page--;renderList();}); $('next').addEventListener('click',()=>{page++;renderList();});
 $('export').addEventListener('click',exportLabels); $('import').addEventListener('change',e=>{if(e.target.files[0]) importLabels(e.target.files[0]);e.target.value='';});
 document.addEventListener('keydown',e=>{
   if (!selected || ['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)) return;
   const d=docs.find(x=>x.doc_id===selected);
   if(e.key==='1') setLabel(d,'relevant'); else if(e.key==='2') setLabel(d,'irrelevant'); else if(e.key==='3') setLabel(d,'unsure');
   else if(e.key==='ArrowRight') move(1); else if(e.key==='ArrowLeft') move(-1); else return;
   e.preventDefault();
 });
 renderCounts(); renderList(); const first=visible()[0]; if(first) select(first.doc_id);
})();
</script>
</body>
</html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-id", type=int, default=1)
    parser.add_argument("--through-query-id", type=int,
                        help="Build every query from --query-id through this ID, sharing one text load")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--compare-pool", type=Path, default=COMPARE_POOL)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    pool = args.pool.resolve()
    end_id = args.through_query_id or args.query_id
    if end_id < args.query_id:
        parser.error("--through-query-id must be >= --query-id")
    ids = list(range(args.query_id, end_id + 1))
    if args.out and len(ids) > 1:
        parser.error("--out cannot be used with a query range")
    compare_pool = args.compare_pool.resolve()
    shared = {}
    if len(ids) > 1:
        all_doc_ids = {str(c["doc_id"])
                       for query_id in ids
                       for c in find_query(pool, query_id)["candidates"][:300]}
        shared = {
            "qdrant_texts": load_qdrant_texts(all_doc_ids),
            "long_texts": load_long_texts(all_doc_ids),
            "raw_index": json.loads(RAW_INDEX.read_text(encoding="utf-8")),
        }
    for query_id in ids:
        data = build_data(pool, query_id, compare_pool, nav_ids=ids, **shared)
        destination = (args.out or ROOT / f"artifact/labeling/query_{query_id}.html").resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        embedded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
        destination.write_text(HTML.replace("__DATA__", embedded), encoding="utf-8")
        missing_text = sum(not d["bge_text_1024"] or (data["compare_pool"] and not d["bge_text_2048"])
                           for d in data["docs"])
        missing_body = sum(not d["body"] for d in data["docs"])
        print(f"{destination} | 300 docs | {destination.stat().st_size / 1e6:.2f} MB | "
              f"missing rerank text {missing_text} | missing body {missing_body}", flush=True)


if __name__ == "__main__":
    main()
