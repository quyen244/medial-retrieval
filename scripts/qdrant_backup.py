"""Sao lưu / khôi phục một collection Qdrant qua snapshot, thành file mang về máy khác được.

    # Trên server đang có dữ liệu:
    python -m scripts.qdrant_backup backup  --collection medical_documents --out-dir data/backups

    # Trên máy mới (Qdrant đã chạy, bundle đã copy sang):
    python -m scripts.qdrant_backup restore --bundle data/backups/medical_documents-20261021-180000

    python -m scripts.qdrant_backup list    # xem snapshot đang nằm trên server Qdrant

Một bundle là một thư mục gồm:
  <collection>.snapshot    snapshot Qdrant (vector dense + sparse + payload + index)
  ledger.sqlite            ledger ingest cùng collection (nếu có), để ingest chạy tiếp được
                           sau khi restore mà không embed lại doc đã có
  manifest.json            số point, sha256, kích thước, phiên bản Qdrant lúc backup

Snapshot được tạo và tải theo kiểu stream (không nạp cả file vào RAM). Tải xong có
kiểm tra sha256, và restore đối chiếu lại số point.

Ghi chú: backup lúc đang ingest vẫn được (snapshot là ảnh chụp tại một thời điểm),
nhưng ledger chép sau snapshot nên có thể hơi mới hơn dữ liệu. Muốn khớp tuyệt đối thì
dừng ingest trước khi backup; ingest chạy lại vẫn an toàn vì upsert theo id.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

DEFAULT_URL = "http://localhost:6333"
LEDGER_DIR = Path("data/ingest_progress")
CHUNK = 8 * 1024 * 1024


def client(url: str, api_key: Optional[str]) -> httpx.Client:
    headers = {"api-key": api_key} if api_key else {}
    # Tạo snapshot và upload có thể mất rất lâu: không đặt timeout đọc.
    return httpx.Client(base_url=url.rstrip("/"), headers=headers,
                        timeout=httpx.Timeout(30.0, read=None, write=None))


def check(r: httpx.Response) -> dict:
    if r.status_code >= 400:
        raise SystemExit(f"Qdrant trả {r.status_code}: {r.text[:500]}")
    return r.json()


def points_count(c: httpx.Client, collection: str) -> int:
    r = c.post(f"/collections/{collection}/points/count", json={"exact": True})
    return check(r)["result"]["count"]


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def cmd_backup(args) -> int:
    c = client(args.url, args.api_key)
    info = check(c.get(f"/collections/{args.collection}"))["result"]
    version = c.get("/").json().get("version")
    n_points = points_count(c, args.collection)
    print(f"Collection {args.collection}: {n_points} point, qdrant {version}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    bundle = Path(args.out_dir) / f"{args.collection}-{stamp}"
    bundle.mkdir(parents=True, exist_ok=True)

    print("Tạo snapshot trên server (lâu với collection lớn)...")
    t0 = time.time()
    snap = check(c.post(f"/collections/{args.collection}/snapshots"))["result"]
    name = snap["name"]
    print(f"  {name} ({human(snap.get('size', 0))}) sau {time.time() - t0:.0f}s")

    dest = bundle / f"{args.collection}.snapshot"
    sha = hashlib.sha256()
    done = 0
    total = snap.get("size") or 0
    t0 = time.time()
    with c.stream("GET", f"/collections/{args.collection}/snapshots/{name}") as r:
        if r.status_code >= 400:
            raise SystemExit(f"Tải snapshot lỗi {r.status_code}")
        total = int(r.headers.get("content-length", total) or 0)
        with dest.open("wb") as f:
            for chunk in r.iter_bytes(CHUNK):
                f.write(chunk)
                sha.update(chunk)
                done += len(chunk)
                el = max(time.time() - t0, 1e-6)
                pct = f"{100 * done / total:5.1f}%" if total else "  ?  "
                eta = f"ETA {(total - done) / (done / el):.0f}s" if total and done else ""
                print(f"\r  tải {human(done)}/{human(total)} {pct} {human(done / el)}/s {eta}   ",
                      end="", flush=True)
    print()

    if total and done != total:
        raise SystemExit(f"Tải thiếu: {done}/{total} byte. Bundle không dùng được: {bundle}")

    ledger_src = LEDGER_DIR / f"{args.collection}.sqlite"
    has_ledger = ledger_src.exists()
    if has_ledger:
        shutil.copy2(ledger_src, bundle / "ledger.sqlite")
    else:
        print(f"  (không thấy ledger {ledger_src}, bỏ qua)")

    manifest = {
        "collection": args.collection,
        "points_count": n_points,
        "snapshot_file": dest.name,
        "snapshot_name_on_server": name,
        "size_bytes": done,
        "sha256": sha.hexdigest(),
        "qdrant_version": version,
        "has_ledger": has_ledger,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vectors": info["config"]["params"].get("vectors"),
        "sparse_vectors": info["config"]["params"].get("sparse_vectors"),
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.keep_remote:
        check(c.delete(f"/collections/{args.collection}/snapshots/{name}"))
        print("  đã xoá snapshot tạm trên server (dùng --keep-remote để giữ)")

    print(f"\nXong: {bundle}  ({human(done)}, {n_points} point)")
    print(f"Mang về máy: scp -r <server>:{bundle.resolve()} .")
    return 0


def cmd_restore(args) -> int:
    bundle = Path(args.bundle)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    collection = args.collection or manifest["collection"]
    snap_path = bundle / manifest["snapshot_file"]

    print(f"Bundle: {manifest['points_count']} point, tạo {manifest['created_at']}, "
          f"qdrant {manifest['qdrant_version']}")

    if not args.skip_checksum:
        print("Kiểm tra sha256...")
        sha = hashlib.sha256()
        with snap_path.open("rb") as f:
            while chunk := f.read(CHUNK):
                sha.update(chunk)
        if sha.hexdigest() != manifest["sha256"]:
            raise SystemExit("sha256 không khớp: file snapshot hỏng hoặc copy dở. Dừng.")
        print("  khớp")

    c = client(args.url, args.api_key)
    existing = {x["name"] for x in check(c.get("/collections"))["result"]["collections"]}
    if collection in existing:
        if not args.overwrite:
            raise SystemExit(f"Collection '{collection}' đã tồn tại trên server. "
                             f"Thêm --overwrite nếu muốn ghi đè (snapshot sẽ thay toàn bộ nội dung).")
        print(f"  --overwrite: ghi đè collection {collection}")

    print(f"Upload {human(snap_path.stat().st_size)} lên Qdrant (lâu với file lớn)...")
    t0 = time.time()
    with snap_path.open("rb") as f:
        r = c.post(f"/collections/{collection}/snapshots/upload",
                   params={"priority": "snapshot", "wait": "true"},
                   files={"snapshot": (snap_path.name, f, "application/octet-stream")})
    check(r)
    print(f"  xong sau {time.time() - t0:.0f}s")

    got = points_count(c, collection)
    ok = got == manifest["points_count"]
    print(f"Số point: {got} (bundle ghi {manifest['points_count']}) -> {'KHỚP' if ok else 'LỆCH'}")

    ledger = bundle / "ledger.sqlite"
    if manifest.get("has_ledger") and ledger.exists():
        target = LEDGER_DIR / f"{collection}.sqlite"
        if target.exists() and not args.overwrite:
            print(f"  ledger {target} đã có, giữ nguyên (thêm --overwrite để thay)")
        else:
            LEDGER_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ledger, target)
            print(f"  đã khôi phục ledger -> {target}")
    return 0 if ok else 2


def cmd_list(args) -> int:
    c = client(args.url, args.api_key)
    for col in check(c.get("/collections"))["result"]["collections"]:
        snaps = check(c.get(f"/collections/{col['name']}/snapshots"))["result"]
        print(f"{col['name']}: {len(snaps)} snapshot")
        for s in snaps:
            print(f"    {s['name']}  {human(s.get('size', 0))}  {s.get('creation_time', '')}")
    return 0


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--api-key", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backup", help="tạo snapshot và tải về thành bundle")
    b.add_argument("--collection", default="medical_documents")
    b.add_argument("--out-dir", default="data/backups")
    b.add_argument("--keep-remote", action="store_true", help="không xoá snapshot tạm trên server")
    b.set_defaults(fn=cmd_backup)

    r = sub.add_parser("restore", help="nạp bundle vào Qdrant")
    r.add_argument("--bundle", required=True)
    r.add_argument("--collection", default=None, help="tên collection đích (mặc định: tên trong bundle)")
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("--skip-checksum", action="store_true")
    r.set_defaults(fn=cmd_restore)

    l = sub.add_parser("list", help="liệt kê snapshot trên server")
    l.set_defaults(fn=cmd_list)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
