---
name: setup-gpu-env
description: Dựng môi trường chạy medial-retrieval trên một máy GPU mới (không docker, Ubuntu 22.04, RTX 50-series/Blackwell) - venv, vLLM, Qdrant binary, SSH GitHub. Dùng khi chuyển sang server thuê mới, hoặc vLLM/Qdrant không lên được.
---

# Dựng môi trường GPU cho medial-retrieval

Đã chạy thật trên: Ubuntu 22.04 (glibc 2.35), RTX 5060 Ti 16GB (sm_120), Python 3.10,
không docker, không nvcc. Kết quả: Qdrant :6333 + vLLM :8000 (Qwen3-8B-AWQ) chạy được.

## 0. Kiểm máy trước

```bash
nvidia-smi; python3 --version; which docker gcc nvcc; ldd --version | head -1
ls data/raw_documents/*/ | head   # corpus có thật không? (data/ bị gitignore)
```

**Corpus không đi theo git.** `data/raw_documents/{domain}/{doc_id}.json` phải chép từ máy crawl.
Thư mục domain rỗng = chưa chép, ingest/smoke sẽ không chạy được.

## 1. Python venv + vLLM

```bash
python3 -m venv .venv
.venv/bin/pip install -U pip wheel setuptools
.venv/bin/pip install vllm          # kéo theo torch cu13 hỗ trợ Blackwell, mất vài phút
.venv/bin/pip install -r requirements.txt FlagEmbedding mlflow requests scikit-learn pytest
.venv/bin/pip check                 # phải là "No broken requirements found"
```

Cài vLLM TRƯỚC để nó chọn torch; các package sau không đụng tới torch.
`requirements.txt` thiếu `FlagEmbedding`, `mlflow`, `requests`, `scikit-learn` dù code import chúng.

## 2. gcc (bắt buộc cho vLLM)

```bash
apt-get update -qq && apt-get install -y build-essential python3.10-dev
```

Thiếu thì vLLM chết với `InductorError: Failed to find C compiler`.

## 3. Qdrant không docker

Binary mặc định (`-gnu`) cần glibc 2.38 -> lỗi `GLIBC_2.38 not found` trên Ubuntu 22.04.
Dùng bản **musl** (static):

```bash
mkdir -p /root/qdrant/{storage,snapshots,musl} && cd /root/qdrant
curl -sL https://github.com/qdrant/qdrant/releases/download/v1.19.2/qdrant-x86_64-unknown-linux-musl.tar.gz | tar xz -C musl
cat > musl/config.yaml <<'EOF'
storage:
  storage_path: /root/qdrant/storage
  snapshots_path: /root/qdrant/snapshots
service:
  http_port: 6333
  grpc_port: 6334
telemetry_disabled: true
EOF
cd musl && nohup ./qdrant --config-path config.yaml > /root/qdrant/qdrant.log 2>&1 &
curl -s localhost:6333/collections
```

Có docker thì dùng `docker compose up -d` như bình thường (`docker-compose.yaml`).

## 4. vLLM

```bash
PATH=$PWD/.venv/bin:$PATH nohup bash scripts/serve_vllm.sh > logs/vllm.log 2>&1 &
until curl -sf localhost:8000/v1/models; do sleep 5; done
```

Lần đầu tải ~5.5GB weights từ HF, rồi compile ~1 phút.

`scripts/serve_vllm.sh` đã vá 2 lỗi của vLLM 0.31 (đừng revert):
- `--disable-log-requests` đã bị gỡ -> dùng `--no-enable-log-requests`.
- `VLLM_USE_FLASHINFER_SAMPLER=0`: sampler FlashInfer JIT bằng nvcc, máy không có nvcc thì
  `Could not find nvcc and default cuda_home='/usr/local/cuda'`. Attention vẫn là FlashAttention 2
  (FA3 chỉ Hopper). Không cần cài CUDA toolkit.

Chiếm ~12.9GB/16GB VRAM, còn ~3.3GB cho bge-m3. Muốn đo tốc độ ingest sạch thì tắt vLLM (`pkill -f "vllm serve"`).

## 5. SSH tới GitHub

Key để ở `/root/tmp` (không commit). Config gốc dùng đường dẫn Windows (`C:\Users\...\.ssh\`),
phải sửa thành `/root/.ssh/...`:

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
cp /root/tmp/id_ed25519_* /root/tmp/known_hosts ~/.ssh/ && chmod 600 ~/.ssh/id_ed25519_quyen ~/.ssh/id_ed25519_quyen_dep
sed -E 's#C:\\Users\\Admin\\\.ssh\\#/root/.ssh/#' /root/tmp/config > ~/.ssh/config && chmod 600 ~/.ssh/config
ssh -T git@quyen244        # "Hi quyen244! You've successfully authenticated"
git remote set-url origin git@quyen244:quyen244/medial-retrieval.git   # origin đã có sẵn dạng https
```

Không in nội dung private key ra terminal/log.

## 6. Kiểm nhanh

```bash
curl -s localhost:6333 ; curl -s localhost:8000/v1/models
ps aux | grep -E "[v]llm serve|[q]drant"; nvidia-smi
```

## Lưu ý khi chạy các lệnh sau đó

- Luôn chạy bằng `.venv/bin/python -m ...` từ thư mục gốc repo.
- `run_generate` không có `--limit`, dùng `--n-chunks N`. `run_augment` và `run_full_submission` có `--limit`.
- `ingest_documents.py`: mặc định fp32, `--encode-batch-size 4` là số đo trên GTX 1660 Ti. Trên card có
  Tensor Core phải đo lại với `--fp16` và batch lớn hơn, đừng mang số cũ sang.
- Thứ tự ưu tiên ingest (TODO.md): vinmec, medlatec, suckhoedoisong, nhathuoclongchau, bachmai, và
  PubMed (en) bắt buộc.
