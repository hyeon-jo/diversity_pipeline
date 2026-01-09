# Video-Centric Data Curation Pipeline

자율주행 영상 데이터셋의 다양성을 분석하고 평가하는 파이프라인입니다.

## 프로젝트 개요

- **목적**: 자율주행 데이터셋의 시나리오 다양성 분석 및 평가
- **핵심 모델**: InternVL3.5-8B (8.5B params) - 비전-언어 모델
- **클러스터링**: Leiden 알고리즘 (k-NN 그래프 기반)
- **평가 지표**: Vendi Score, Coverage, Density 등 8개 지표

## 디렉토리 구조

```
diversity_pipeline/
├── pipeline/                    # 핵심 파이프라인 패키지 (2,904 lines, 16개 파일)
│   ├── __init__.py              # 공개 API 통합
│   ├── config.py                # 설정 클래스 (VideoEmbedderConfig, ClusteringConfig, AnalysisConfig)
│   ├── types.py                 # 타입 정의 (ClusterInfo, PipelineResults)
│   ├── embedder/                # 임베딩 추출 서브패키지
│   │   ├── __init__.py
│   │   ├── base.py              # VideoEmbedder 클래스
│   │   ├── model_loader.py      # InternVL 모델 로딩
│   │   └── frame_loader.py      # 프레임 로딩 + 최적화된 디렉토리 탐색
│   ├── clusterer.py             # GraphClusterer (Leiden clustering)
│   ├── analyzer.py              # ClusterAnalyzer (edge case/void 탐지)
│   ├── captioning/              # 캡셔닝 서브패키지
│   │   ├── __init__.py
│   │   ├── base.py              # CaptioningInterface (추상 클래스)
│   │   ├── mock.py              # MockCaptioningInterface
│   │   └── internvl.py          # InternVLCaptioningInterface
│   ├── pipeline.py              # VideoCurationPipeline (오케스트레이션)
│   ├── demo.py                  # generate_synthetic_embeddings, print_results_summary
│   └── cli.py                   # CLI main() 함수
│
├── diversity_metrics/           # 다양성 평가 패키지 (2,332 lines)
│   ├── __init__.py
│   ├── metrics.py               # 8개 다양성 지표 구현
│   ├── evaluator.py             # DiversityEvaluator 클래스
│   ├── reference_datasets.py    # BDD100K, nuImages 참조 데이터셋
│   └── report.py                # HTML/CSV 리포트 생성
│
├── visualizer/                  # 시각화 패키지 (2,775 lines)
│   ├── __init__.py
│   ├── config.py                # 시각화 설정
│   ├── embedding_plot.py        # 임베딩 2D/3D 시각화 (UMAP)
│   ├── cluster_gallery.py       # 클러스터별 대표 이미지 갤러리
│   ├── distribution_chart.py    # 분포 차트 (Sunburst, Treemap)
│   ├── network_graph.py         # 클러스터 네트워크 그래프
│   ├── dashboard.py             # 통합 대시보드
│   └── main.py                  # 시각화 CLI
│
├── video_curation_pipeline.py   # 후방 호환성 wrapper (~60 lines)
├── DIVERSITY_METRICS.md         # 다양성 지표 문서
├── requirements.txt             # 의존성
└── CLAUDE.md                    # 이 파일
```

## 완료된 작업

### 1. pipeline/ 모듈화 ✅
- **이전**: `video_curation_pipeline.py` (2,370 lines, 단일 파일)
- **이후**: `pipeline/` 패키지 (16개 파일, 2,904 lines)
- **개선사항**:
  - 명확한 책임 분리 (config, types, embedder, clusterer, analyzer, captioning, pipeline)
  - 순환 의존성 없음
  - 후방 호환성 유지 (`from video_curation_pipeline import *` 동작)

### 2. 디렉토리 스캔 최적화 ✅
- **문제**: `Path.glob("**/CMR_GT_Frame")` 이 대규모 디렉토리에서 매우 느림 (~60초)
- **해결**: 3단계 regex 기반 탐색 구현
- **위치**: `pipeline/embedder/frame_loader.py`
- **함수**:
  - `find_frame_directories_optimized()` - 10-12배 성능 향상
  - `find_frame_directories_cached()` - 캐싱 지원
- **사용법**: `python video_curation_pipeline.py --frame-dir ./trainlake --cache-dirs`

### 3. diversity_metrics 패키지 ✅
- **8개 다양성 지표**:
  1. Vendi Score (ICLR 2023) - 유효 고유 시나리오 수
  2. Coverage Score (NeurIPS 2019) - 참조 분포 커버리지
  3. Density Score (ICML 2020) - 참조 분포 내 밀도
  4. Feature Space Coverage - 그리드 기반 커버리지
  5. Scenario Entropy - 클러스터 분포 엔트로피
  6. Balance Score - 1 - Gini 계수
  7. Rarity Score - k-NN 기반 희귀성
  8. Intra-Cluster Diversity - 클러스터 내 변동성

- **참조 데이터셋**: BDD100K, nuImages 지원
- **출력**: 모든 지표 0-100% 정규화

### 4. visualizer 패키지 ✅
- **5가지 시각화**:
  1. 임베딩 산점도 (2D/3D UMAP)
  2. 클러스터 갤러리 (대표 프레임)
  3. 분포 차트 (Sunburst, Treemap)
  4. 네트워크 그래프 (클러스터 관계)
  5. 통합 대시보드

## 사용 방법

### Demo 모드 (합성 데이터)
```bash
python video_curation_pipeline.py --demo --n-samples 500
```

### Frame 디렉토리 처리 (실제 데이터)
```bash
# 기본 (glob 사용 - 느림)
python video_curation_pipeline.py --frame-dir ./trainlake

# 최적화 + 캐싱 (권장)
python video_curation_pipeline.py --frame-dir ./trainlake --cache-dirs
```

### Python API
```python
from pipeline import VideoCurationPipeline, VideoEmbedderConfig

# 파이프라인 생성
pipeline = VideoCurationPipeline(
    embedder_config=VideoEmbedderConfig(output_dir="./output")
)

# 실행
results = pipeline.run(video_paths=["video1.mp4", "video2.mp4"])

# 또는 임베딩에서 직접 실행
results = pipeline.run_from_embeddings(embeddings, video_paths)
```

## 데이터 구조

### 입력 디렉토리 구조 (trainlake)
```
trainlake/
├── [yy].[mm]w/                          # 예: 24.01w
│   └── N[7자리]-[12자리]/               # 예: N1234567-240115120000
│       └── RAW_DB/
│           └── *_CMR*/                  # 예: FRONT_CMR
│               └── CMR_GT_Frame/
│                   └── *.jpg            # 프레임 이미지들
```

### 출력
- `output/*.npy` - 비디오별 임베딩 벡터
- 클러스터 분석 결과 (cluster_info, edge_cases, voids)
- 캡션 (클러스터별 시나리오 설명)

## 기술 스택

| 카테고리 | 기술 |
|---------|-----|
| 비전-언어 모델 | InternVL3.5-8B (OpenGVLab) |
| 딥러닝 프레임워크 | PyTorch, transformers |
| 클러스터링 | Leiden (igraph, leidenalg) |
| 근사 최근접 이웃 | pynndescent |
| 시각화 | Plotly, UMAP, pyvis |
| 비디오 처리 | decord, PyAV |

## 핵심 클래스

### pipeline 패키지

| 클래스 | 파일 | 역할 |
|-------|-----|------|
| `VideoEmbedderConfig` | config.py | 임베딩 추출 설정 |
| `ClusteringConfig` | config.py | 클러스터링 설정 |
| `AnalysisConfig` | config.py | 분석 설정 |
| `VideoEmbedder` | embedder/base.py | InternVL 기반 임베딩 추출 |
| `GraphClusterer` | clusterer.py | k-NN 그래프 + Leiden 클러스터링 |
| `ClusterAnalyzer` | analyzer.py | Edge case/void 탐지 |
| `VideoCurationPipeline` | pipeline.py | 전체 파이프라인 오케스트레이션 |

### diversity_metrics 패키지

| 클래스/함수 | 파일 | 역할 |
|------------|-----|------|
| `vendi_score()` | metrics.py | Vendi Score 계산 |
| `coverage_score()` | metrics.py | Coverage 계산 |
| `DiversityEvaluator` | evaluator.py | 전체 다양성 평가 |
| `ReferenceDatasetLoader` | reference_datasets.py | 참조 데이터셋 로딩 |

## 추후 작업 (TODO)

1. **테스트 코드**: 단위 테스트 및 통합 테스트 추가
2. **성능 프로파일링**: 실제 대규모 데이터셋으로 성능 측정
3. **실제 데이터 검증**: trainlake 데이터로 전체 파이프라인 검증
4. **diversity_metrics 통합**: pipeline 결과와 diversity_metrics 연동
5. **시각화 통합**: visualizer와 pipeline 결과 연동

## 알려진 이슈

1. `leidenalg` 미설치 시 SpectralClustering으로 fallback
2. `pynndescent` 미설치 시 sklearn NearestNeighbors 사용
3. Flash Attention은 CUDA 환경에서만 사용 가능

## Git 히스토리 참고

```
379f6ee refactor with gemini-cli
dd43b85 plan phase 5-1 완료; phase 5-2 수행 중 중단
80f62f6 diversity 측정 지표 및 리포트 생성 코드 업데이트
9abbde4 first commit
```

---

*마지막 업데이트: 2025-01-09*
