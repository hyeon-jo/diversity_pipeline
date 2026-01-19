# BEV Diversity Pipeline

InternVL3.5-8B 모델과 Vendi Score Family를 사용한 영상 다양성 분석 파이프라인

## 특징

- ✅ **로컬 모델 로딩**: 로컬에 저장된 InternVL3.5-8B 모델 사용
- ✅ **자동 이미지 발견**: 디렉토리에서 재귀적으로 이미지 검색
- ✅ **Progress Bar**: 이미지 발견 및 임베딩 추출 진행상황 표시
- ✅ **확장된 Vendi Score**: 논문 "Cousins Of The Vendi Score"의 q-order 지원
- ✅ **YAML 설정**: 간편한 설정 파일 관리
- ✅ **여러 커널 지원**: Cosine, RBF, Linear 커널

## 설치

```bash
# 의존성 설치
pip install -r bev_diversity_requirements.txt

# 또는 개별 설치
pip install pyyaml tqdm scipy numpy torch transformers Pillow scikit-learn
```

## 빠른 시작

### 두 가지 모드

1. **EXTRACTION 모드**: 이미지에서 임베딩 추출 + Vendi Score 계산
2. **EMBEDDING-ONLY 모드**: 이미 추출된 임베딩에서 Vendi Score만 계산

### 모드 1: EXTRACTION (이미지 → 임베딩 → Vendi Score)

#### 1. 설정 파일 수정

`bev_diversity_config.yaml` 파일에서 다음 항목을 수정하세요:

```yaml
model:
  model_path: "/path/to/InternVL3_5-8B"  # 실제 모델 경로로 변경

input_dir: "/path/to/frames"  # 프레임 이미지 경로로 변경
```

#### 2. 실행

```bash
python -m bev_diversity -c bev_diversity_config.yaml

# 임베딩도 저장 (나중에 재사용)
python -m bev_diversity -c bev_diversity_config.yaml --save-embeddings
```

#### 3. 결과 확인

결과는 `output/vendi_scores.json`에 저장됩니다:

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
    "embedding_dim": 1024
  }
}
```

### 모드 2: EMBEDDING-ONLY (임베딩 → Vendi Score만 계산)

이미 추출된 임베딩이 있다면 모델 로딩 없이 바로 실행 가능합니다.

#### 1. 설정 파일 생성

`bev_diversity_config_embedding_only.yaml`:

```yaml
embedding_path: "/path/to/embeddings.npy"  # 임베딩 파일 경로

vendi:
  q_values: [0.1, 0.5, 1.0, 2.0]
  include_infinity: true
  kernel: "cosine"

output_dir: "./output"
```

#### 2. 실행

```bash
python -m bev_diversity -c bev_diversity_config_embedding_only.yaml
```

**장점**:
- ✅ 모델 로딩 불필요 (빠른 실행)
- ✅ GPU 불필요 (CPU만으로 실행 가능)
- ✅ 여러 번 재실험 가능

---

## 디렉토리 구조

```
bev_diversity/
├── __init__.py              # 공개 API
├── __main__.py              # CLI 진입점
├── config.py                # YAML 설정 로딩 (두 가지 모드 지원)
├── cli.py                   # CLI 구현 (모드 자동 감지)
├── utils.py                 # 유틸리티 (임베딩 로딩/저장)
├── embedder/
│   ├── __init__.py
│   ├── model.py             # InternVL 모델 로더
│   └── extractor.py         # 임베딩 추출기
└── metrics/
    ├── __init__.py
    ├── similarity.py        # 유사도 행렬 계산
    └── vendi_family.py      # Vendi Score Family
```

## 주요 파일

- [bev_diversity_config.yaml](bev_diversity_config.yaml) - 설정 파일 예시
- [bev_diversity_requirements.txt](bev_diversity_requirements.txt) - 의존성 목록
- [BEV_DIVERSITY_USAGE.md](BEV_DIVERSITY_USAGE.md) - 상세 사용 가이드
- [test_bev_diversity.py](test_bev_diversity.py) - 단위 테스트

## Vendi Score Family 이해

### Order q의 의미

| q 값 | 의미 | 용도 |
|------|------|------|
| 0.1, 0.5 | 희귀 항목에 민감 | 드문 시나리오 탐지 |
| 1.0 | 원본 Vendi Score | 전반적 다양성 |
| 2.0, ∞ | 흔한 항목에 민감 | 중복/과대표현 탐지 |

### 수식

```
VS_q = exp((1/(1-q)) * log(Σ λ_i^q))
```

여기서 λ_i는 정규화된 유사도 행렬의 고유값

### 해석

- **높은 점수**: 다양성 높음
- **낮은 점수**: 중복이 많거나 다양성 낮음
- **VS_∞ << VS_0.1**: 중복된 항목이 많음을 시사

## Python API 예시

```python
from pathlib import Path
from bev_diversity import load_config, ImageEmbedder, VendiScoreFamily

# 설정 로드
config = load_config("bev_diversity_config.yaml")

# 임베딩 추출
embedder = ImageEmbedder(config.model, config.embedding)
embeddings = embedder.extract_from_directory(
    Path(config.input_dir),
    config.image_extensions
)

# Vendi Score 계산
vendi = VendiScoreFamily(kernel=config.vendi.kernel)
scores = vendi.compute_all(
    embeddings,
    config.vendi.q_values,
    include_infinity=True
)

print(scores)
```

## CLI 옵션

```bash
# EXTRACTION 모드: 이미지에서 임베딩 추출 + Vendi Score 계산
python -m bev_diversity -c config.yaml

# EMBEDDING-ONLY 모드: 임베딩 파일에서 Vendi Score만 계산
python -m bev_diversity -c config_embedding_only.yaml

# 상세 출력
python -m bev_diversity -c config.yaml -v

# 임베딩도 저장 (나중에 재사용)
python -m bev_diversity -c config.yaml --save-embeddings

# 출력 디렉토리 변경
python -m bev_diversity -c config.yaml -o ./results
```

## 테스트

```bash
# 단위 테스트 실행 (numpy 등이 설치되어 있어야 함)
python test_bev_diversity.py
```

## 논문 참조

Amey P. Pasarkar and Adji Bousso Dieng. "Cousins Of The Vendi Score: A Family Of Similarity-Based Diversity Metrics For Science And Machine Learning." AISTATS 2024.

## 라이선스

이 코드는 기존 diversity_pipeline 프로젝트의 일부입니다.

## 문의

문제가 있으시면 이슈를 등록해주세요.
