# BEV Diversity Pipeline - 사용 가이드

InternVL3.5-8B 모델을 사용하여 프레임 이미지로부터 visual embedding을 추출하고, 논문 "Cousins Of The Vendi Score"의 확장된 Vendi Score Family를 계산하는 파이프라인입니다.

## 설치

### 의존성 설치

```bash
pip install pyyaml tqdm scipy numpy torch transformers Pillow scikit-learn
```

## 사용 방법

### 두 가지 모드

BEV Diversity Pipeline은 두 가지 모드로 동작합니다:

1. **EXTRACTION 모드**: 이미지에서 임베딩을 추출하고 Vendi Score 계산
2. **EMBEDDING-ONLY 모드**: 이미 추출된 임베딩으로부터 Vendi Score만 계산

---

## 모드 1: EXTRACTION 모드 (이미지 → 임베딩 → Vendi Score)

### 1. 설정 파일 준비

`bev_diversity_config.yaml` 파일을 수정하여 설정을 지정합니다:

```yaml
model:
  # 필수: 로컬에 저장된 InternVL3.5-8B 모델 경로
  model_path: "/path/to/InternVL3_5-8B"
  torch_dtype: "bfloat16"
  device: "cuda"

embedding:
  batch_size: 4
  normalize_embeddings: true

vendi:
  q_values: [0.1, 0.5, 1.0, 2.0]
  include_infinity: true
  kernel: "cosine"

# 필수: 프레임 이미지가 있는 루트 디렉토리
input_dir: "/path/to/frames"

output_dir: "./output"
```

### 2. 실행

```bash
# 기본 실행
python -m bev_diversity -c bev_diversity_config.yaml

# 상세 출력 모드
python -m bev_diversity -c bev_diversity_config.yaml -v

# 임베딩도 함께 저장
python -m bev_diversity -c bev_diversity_config.yaml --save-embeddings

# 출력 디렉토리 변경
python -m bev_diversity -c bev_diversity_config.yaml -o ./my_results
```

### 3. 출력

실행하면 다음과 같은 출력을 볼 수 있습니다:

```
================================================================================
BEV Diversity Analysis
Video Diversity Assessment using Vendi Score Family
================================================================================

Loading configuration from bev_diversity_config.yaml...
Configuration loaded successfully

Initializing embedder...
Loading InternVL model from /path/to/InternVL3_5-8B...
Model loaded successfully on cuda

Extracting embeddings from images...
Discovering images in /path/to/frames...
Found 1,234 images
Extracting embeddings... 100%|████████████████████| 1234/1234

Computing Vendi Score Family...

================================================================================
Results
================================================================================

Images Processed: 1,234
Kernel: cosine

Vendi Score Family:
--------------------------------------------------------------------------------
  VS (q= 0.1):    156.32  [Most sensitive to rare items]
  VS (q= 0.5):     89.47
  VS (q= 1.0):     45.23  [Original Vendi Score - Shannon entropy]
  VS (q= 2.0):     28.91
  VS (q= inf):     12.45  [Most sensitive to common items/duplicates]

--------------------------------------------------------------------------------

Interpretation:
  - Higher scores indicate greater diversity
  - Scores represent "effective number of unique scenarios"
  - VS_inf being much lower than VS_0.1 (7.97%) suggests significant duplicates

Results saved to: ./output/vendi_scores.json
================================================================================
```

### 4. 임베딩 저장 (선택)

나중에 재사용하기 위해 임베딩을 저장할 수 있습니다:

```bash
python -m bev_diversity -c bev_diversity_config.yaml --save-embeddings
```

이렇게 하면 `output/embeddings.npy` 파일에 임베딩이 저장됩니다.

### 5. 결과 파일

결과는 JSON 형식으로 저장됩니다 (`output/vendi_scores.json`):

```json
{
  "vendi_scores": {
    "VS_q=0.1": 156.32,
    "VS_q=0.5": 89.47,
    "VS_q=1.0": 45.23,
    "VS_q=2.0": 28.91,
    "VS_q=inf": 12.45
  },
  "metadata": {
    "num_images": 1234,
    "embedding_dim": 1024,
    "config_file": "bev_diversity_config.yaml"
  }
}
```

---

## 모드 2: EMBEDDING-ONLY 모드 (임베딩 → Vendi Score만 계산)

이미 추출된 임베딩이 있다면, 모델 로딩 없이 바로 Vendi Score를 계산할 수 있습니다.

### 1. 설정 파일 준비

`bev_diversity_config_embedding_only.yaml` 파일을 생성합니다:

```yaml
# 필수: 임베딩 파일 경로 (.npy 또는 .npz)
embedding_path: "/path/to/embeddings.npy"

vendi:
  q_values: [0.1, 0.5, 1.0, 2.0]
  include_infinity: true
  kernel: "cosine"

output_dir: "./output"
```

### 2. 실행

```bash
python -m bev_diversity -c bev_diversity_config_embedding_only.yaml
```

### 3. 출력

```
================================================================================
BEV Diversity Analysis
Video Diversity Assessment using Vendi Score Family
================================================================================

Loading configuration from bev_diversity_config_embedding_only.yaml...
Configuration loaded successfully

Running in EMBEDDING-ONLY mode
Loading pre-computed embeddings from /path/to/embeddings.npy...
Loaded 1,234 embeddings (dimension: 1024)

Computing Vendi Score Family...

================================================================================
Results
================================================================================
...
```

### 장점

- ✅ **빠른 실행**: 모델 로딩 없이 즉시 실행
- ✅ **GPU 불필요**: CPU만으로 Vendi Score 계산 가능
- ✅ **재사용성**: 한 번 추출한 임베딩으로 여러 번 실험 가능

---

## Python API 사용

CLI 대신 Python 코드에서 직접 사용할 수도 있습니다:

### EXTRACTION 모드

```python
from pathlib import Path
from bev_diversity import load_config, ImageEmbedder, VendiScoreFamily

# 설정 로드
config = load_config("bev_diversity_config.yaml")

# 임베딩 추출
embedder = ImageEmbedder(
    model_config=config.model,
    embedding_config=config.embedding
)

embeddings = embedder.extract_from_directory(
    root_dir=Path(config.input_dir),
    extensions=config.image_extensions
)

# Vendi Score 계산
vendi = VendiScoreFamily(
    kernel=config.vendi.kernel,
    rbf_gamma=config.vendi.rbf_gamma
)

scores = vendi.compute_all(
    embeddings=embeddings,
    q_values=config.vendi.q_values,
    include_infinity=config.vendi.include_infinity
)

print(scores)
# {'VS_q=0.1': 156.32, 'VS_q=0.5': 89.47, 'VS_q=1.0': 45.23, ...}
```

### EMBEDDING-ONLY 모드

```python
from bev_diversity import load_config, VendiScoreFamily
from bev_diversity.utils import load_embeddings

# 설정 로드
config = load_config("bev_diversity_config_embedding_only.yaml")

# 임베딩 로드
embeddings = load_embeddings(config.embedding_path)

# Vendi Score 계산
vendi = VendiScoreFamily(kernel=config.vendi.kernel)
scores = vendi.compute_all(
    embeddings=embeddings,
    q_values=config.vendi.q_values,
    include_infinity=True
)

print(scores)
```

## Vendi Score Family 이해하기

### Order q의 의미

- **q = 0.1, 0.5**: 희귀한 항목에 민감 → 데이터셋에 드문 시나리오가 있는지 확인
- **q = 1.0**: 원본 Vendi Score (Shannon 엔트로피) → 전반적인 다양성
- **q = 2.0, ∞**: 흔한 항목/중복에 민감 → 중복이나 과대표현된 시나리오 탐지

### 수식

- **q = 1**: `VS_1 = exp(-Σ λ_i log(λ_i))`
- **q ≠ 1, ∞**: `VS_q = exp((1/(1-q)) * log(Σ λ_i^q))`
- **q = ∞**: `VS_∞ = 1/max(λ_i)`

여기서 λ_i는 정규화된 유사도 행렬 K/n의 고유값입니다.

### 점수 해석

- **점수 값**: "효과적인 고유 시나리오 수"를 나타냄
- **높은 점수**: 다양성이 높음
- **단조성**: VS_∞ ≤ VS_2 ≤ VS_1 ≤ VS_0.5 ≤ VS_0.1
- **VS_∞ << VS_0.1**: 중복이나 매우 유사한 항목이 많음을 시사

## 커널 옵션

### Cosine Kernel (권장)
```yaml
vendi:
  kernel: "cosine"
```
- 정규화된 내적 유사도
- 임베딩 방향만 고려 (크기 무시)
- 대부분의 경우 권장

### RBF Kernel
```yaml
vendi:
  kernel: "rbf"
  rbf_gamma: 1.0
```
- Gaussian 커널: `exp(-gamma * ||x - x'||^2)`
- gamma가 클수록 더 국소적인 유사도

### Linear Kernel
```yaml
vendi:
  kernel: "linear"
```
- 단순 내적: `x @ x'`
- 임베딩 크기도 함께 고려

## 참고 논문

Amey P. Pasarkar and Adji Bousso Dieng. "Cousins Of The Vendi Score: A Family Of Similarity-Based Diversity Metrics For Science And Machine Learning." AISTATS 2024.

## 문제 해결

### CUDA Out of Memory
- `batch_size`를 줄이세요 (예: 4 → 2 → 1)
- `torch_dtype`을 "float16"으로 변경하세요

### 모델 로딩 오류
- `model_path`가 정확한지 확인하세요
- 모델 디렉토리에 필요한 파일이 모두 있는지 확인하세요
- `trust_remote_code: true`가 설정되어 있는지 확인하세요

### 이미지를 찾을 수 없음
- `input_dir` 경로가 정확한지 확인하세요
- `image_extensions`에 필요한 확장자가 모두 포함되어 있는지 확인하세요
