# BEV Diversity Pipeline - 사용 예시

## 시나리오 1: 처음부터 시작 (이미지 → 임베딩 → Vendi Score)

```bash
# 1. 설정 파일 수정
vim bev_diversity_config.yaml
# model_path와 input_dir 경로 수정

# 2. 실행 (임베딩도 저장)
python -m bev_diversity -c bev_diversity_config.yaml --save-embeddings

# 3. 결과 확인
cat output/vendi_scores.json
```

**출력**:
- `output/embeddings.npy` - 임베딩 파일 (나중에 재사용 가능)
- `output/vendi_scores.json` - Vendi Score 결과

---

## 시나리오 2: 저장된 임베딩으로 다시 계산

이미 임베딩을 추출했다면, 다른 커널이나 q 값으로 빠르게 재실험할 수 있습니다.

```bash
# 1. embedding-only 설정 파일 생성
cat > config_test.yaml << EOF
embedding_path: "./output/embeddings.npy"

vendi:
  q_values: [0.1, 0.5, 1.0, 2.0]
  include_infinity: true
  kernel: "rbf"  # 다른 커널 시도
  rbf_gamma: 0.5

output_dir: "./output_rbf"
EOF

# 2. 빠르게 실행 (모델 로딩 없음!)
python -m bev_diversity -c config_test.yaml

# 3. 결과 비교
diff output/vendi_scores.json output_rbf/vendi_scores.json
```

**장점**: 모델 로딩 없이 1-2초 안에 완료!

---

## 시나리오 3: 여러 커널 비교

```bash
# cosine 커널
cat > config_cosine.yaml << EOF
embedding_path: "./output/embeddings.npy"
vendi:
  kernel: "cosine"
output_dir: "./results_cosine"
EOF

# RBF 커널
cat > config_rbf.yaml << EOF
embedding_path: "./output/embeddings.npy"
vendi:
  kernel: "rbf"
  rbf_gamma: 1.0
output_dir: "./results_rbf"
EOF

# linear 커널
cat > config_linear.yaml << EOF
embedding_path: "./output/embeddings.npy"
vendi:
  kernel: "linear"
output_dir: "./results_linear"
EOF

# 모두 실행
for config in config_*.yaml; do
    python -m bev_diversity -c $config
done

# 결과 비교
echo "Cosine:" && jq '.vendi_scores' results_cosine/vendi_scores.json
echo "RBF:" && jq '.vendi_scores' results_rbf/vendi_scores.json
echo "Linear:" && jq '.vendi_scores' results_linear/vendi_scores.json
```

---

## 시나리오 4: Python 스크립트에서 사용

```python
#!/usr/bin/env python3
"""여러 임베딩 파일에 대해 Vendi Score 계산"""

from pathlib import Path
from bev_diversity.metrics.vendi_family import VendiScoreFamily
from bev_diversity.utils import load_embeddings

# 여러 임베딩 파일 처리
embedding_files = [
    "dataset1/embeddings.npy",
    "dataset2/embeddings.npy",
    "dataset3/embeddings.npy",
]

vendi = VendiScoreFamily(kernel="cosine")

for emb_file in embedding_files:
    print(f"\nProcessing {emb_file}...")

    # 임베딩 로드
    embeddings = load_embeddings(emb_file)

    # Vendi Score 계산
    scores = vendi.compute_all(
        embeddings=embeddings,
        q_values=[0.1, 0.5, 1.0, 2.0],
        include_infinity=True
    )

    # 결과 출력
    print(f"  VS (q=1.0): {scores['VS_q=1.0']:.2f}")
    print(f"  VS (q=inf): {scores['VS_q=inf']:.2f}")
    print(f"  Ratio: {scores['VS_q=inf'] / scores['VS_q=0.1']:.2%}")
```

---

## 시나리오 5: 대용량 데이터셋

GPU 메모리가 부족한 경우:

```yaml
# config_low_memory.yaml
model:
  model_path: "/path/to/model"
  torch_dtype: "float16"  # bfloat16 대신 float16 사용
  device: "cuda"

embedding:
  batch_size: 1  # 배치 크기 줄이기

input_dir: "/path/to/large_dataset"
output_dir: "./output"
```

```bash
python -m bev_diversity -c config_low_memory.yaml --save-embeddings
```

---

## 시나리오 6: 배치 처리

```bash
#!/bin/bash
# 여러 디렉토리의 이미지를 순차적으로 처리

for dir in dataset1 dataset2 dataset3; do
    echo "Processing $dir..."

    # 설정 파일 생성
    cat > config_$dir.yaml << EOF
model:
  model_path: "/path/to/model"
embedding:
  batch_size: 4
input_dir: "./$dir"
output_dir: "./output_$dir"
vendi:
  q_values: [0.1, 1.0, 2.0]
  kernel: "cosine"
EOF

    # 실행
    python -m bev_diversity -c config_$dir.yaml --save-embeddings

    echo "Done: $dir"
    echo "---"
done

# 결과 요약
echo "Summary:"
for dir in dataset1 dataset2 dataset3; do
    echo "$dir:"
    jq '.vendi_scores["VS_q=1.0"]' output_$dir/vendi_scores.json
done
```

---

## 팁

### 1. NPZ 파일 사용 (압축)

```python
import numpy as np

# 저장
embeddings = np.random.randn(10000, 1024).astype(np.float32)
np.savez_compressed("embeddings.npz", embeddings=embeddings)

# 로드 (자동으로 지원됨)
# config에서 embedding_path: "embeddings.npz" 사용 가능
```

### 2. 빠른 프로토타이핑

```bash
# 작은 샘플로 먼저 테스트
python -m bev_diversity -c config.yaml -v --save-embeddings

# 임베딩만 재사용하며 실험
python -m bev_diversity -c config_embedding_only.yaml
```

### 3. 결과 비교

```python
import json

# 두 결과 로드
with open("output1/vendi_scores.json") as f:
    scores1 = json.load(f)["vendi_scores"]

with open("output2/vendi_scores.json") as f:
    scores2 = json.load(f)["vendi_scores"]

# 비교
for key in scores1:
    diff = scores1[key] - scores2[key]
    print(f"{key}: {scores1[key]:.2f} vs {scores2[key]:.2f} (diff: {diff:+.2f})")
```
