# --frame-dir 디렉토리 탐색 및 임베딩 추출 성능 개선 계획

## 문제점 1: 디렉토리 탐색이 느림

`--frame-dir` 인자 사용 시 디렉토리 탐색이 너무 오래 걸림 (~60초)

## 해결 방안 1: 최적화를 기본값으로 변경

**기본 동작을 최적화된 방식으로 변경** (사용자 승인 완료)

---

## 문제점 2: 이미 추출된 임베딩을 재추출함 (치명적!)

**현재 상태**: 임베딩 파일이 이미 존재해도 무조건 프레임 로딩 + 임베딩 재추출 + 덮어쓰기

### 현재 동작 (이미 하나씩 저장 중 ✅)
- `extract_embedding_from_frame_dir()`에서 매번 `_save_embedding()` 호출
- 개별 .npy 파일로 즉시 저장 → 중단 시에도 이미 처리된 것은 보존됨

### 문제점 (Skip 로직 없음 ❌)
- `extract_embeddings_from_frame_dirs()`에서 기존 .npy 파일 확인 안 함
- 매번 모든 frame_dir을 처리 (프레임 로딩 + InternVL 추론)
- **3.8만개 비디오, 37시간 작업이 중단되면 처음부터 재시작**

### 실제 사용자 시나리오
```bash
# 첫 실행: 38,000개 비디오 처리 (37시간)
python video_curation_pipeline.py --frame-dir ./trainlake
# → 20시간 진행 후 중단 (20,000개 완료)

# 재실행: 38,000개 다시 재추출 (37시간) <- 치명적!
# 이미 완료된 20,000개도 다시 처리
python video_curation_pipeline.py --frame-dir ./trainlake
```

## 해결 방안 2: Skip 로직 추가

기존 임베딩 파일이 있으면 로드만 하고 재추출 건너뛰기

---

## 검증 완료: 프레임 임베딩 평균화 ✅

**발견**: .npy 파일이 17KB = 단일 4096차원 벡터만 저장됨

### 데이터 흐름 분석

```
프레임 디렉토리 (예: 160개 JPG)
    ↓
load_frames_for_internvl()
    ├─ 균등 샘플링: 16개 프레임 선택
    └─ 반환: (16, 3, 448, 448) ✅
    ↓
extract_vision_features()
    ├─ InternVL 모델 처리
    └─ 반환: vit_embeds (shape 불명확)
    ↓
평균화: mean(dim=(0, 1))
    └─ (4096,) 단일 벡터 ❌
    ↓
저장: .npy 파일 (17KB)
```

### 코드 위치: [base.py:222-224](pipeline/embedder/base.py#L222-L224)

```python
# Mean pooling over all patches and frames
# vit_embeds shape: (num_frames, num_patches, hidden_dim)
embedding = vit_embeds.mean(dim=(0, 1))  # (hidden_dim,)
```

### 잠재적 문제

| 항목 | 현재 동작 | 우려사항 |
|------|-----------|----------|
| 프레임 수 | 16개 로드 ✅ | 모두 평균화되어 시간 정보 손실? |
| 최종 임베딩 | 단일 4096차원 벡터 | 각 프레임의 고유 특징 희석? |
| 파일 크기 | 17KB | 정상 (4096 × 4bytes) |

### 검증 필요

1. **vit_embeds의 실제 shape**
   - 예상: `(16, num_patches, 4096)`
   - 실제: 확인 필요

2. **프레임별 임베딩 차이**
   - 각 프레임이 다른 특징을 가지는지
   - 아니면 모두 동일한 값인지

3. **평균화 방식의 적절성**
   - 의도: 비디오 전체를 단일 벡터로 표현
   - 확인 필요: 이것이 시나리오 다양성 측정에 적합한가?

### 진단 코드 추가 계획

**위치**: [base.py:220-225](pipeline/embedder/base.py#L220-L225)

```python
# 디버그 로깅 추가
logger.info(f"DEBUG - vit_embeds shape: {vit_embeds.shape}")
logger.info(f"DEBUG - vit_embeds dtype: {vit_embeds.dtype}")

# 각 프레임별 임베딩 확인
if len(vit_embeds.shape) >= 2:
    frame_embeddings = vit_embeds.mean(dim=1)  # 패치만 평균
    logger.info(f"DEBUG - Frame embeddings shape: {frame_embeddings.shape}")
    logger.info(f"DEBUG - Frame 0 first 5 values: {frame_embeddings[0][:5]}")
    if len(frame_embeddings) > 1:
        logger.info(f"DEBUG - Frame -1 first 5 values: {frame_embeddings[-1][:5]}")

# Mean pooling over all patches and frames
embedding = vit_embeds.mean(dim=(0, 1))  # (hidden_dim,)
logger.info(f"DEBUG - Final embedding shape: {embedding.shape}")
```

**검증 결과**:
- ✅ 16개 프레임 모두 로드 및 처리
- ✅ `mean(dim=(0,1))`로 단일 벡터 생성 (의도된 설계)
- ✅ 비디오 전체를 하나의 시나리오로 표현하는 것이 적합하다고 판단
- ✅ 17KB 파일 크기 정상 (4096 × 4bytes)

---

## 수정 대상 파일

| 파일 | 수정 내용 |
|------|----------|
| [pipeline/cli.py](pipeline/cli.py) | ✅ --no-cache-dirs 변경 완료, ⏳ --force-recompute 플래그 추가 예정 |
| [pipeline/embedder/base.py](pipeline/embedder/base.py) | ⏳ skip_existing 파라미터 및 로직 추가 예정 |

---

## 구현 세부사항

### 작업 1: 디렉토리 탐색 최적화 (✅ 완료)

#### 1-1. CLI 인자 변경 ([cli.py:44-48](pipeline/cli.py#L44-L48))

```python
# Before: --cache-dirs
# After: --no-cache-dirs (✅ 완료)
```

#### 1-2. 분기 로직 변경 ([cli.py:113-128](pipeline/cli.py#L113-L128))

```python
# Before: if args.cache_dirs
# After: if not args.no_cache_dirs (✅ 완료)
```

**결과**: 디렉토리 탐색 ~60초 → ~5-10초 (첫 실행), <1초 (이후)

---

### 작업 2: 임베딩 Skip 로직 추가 (사용자 승인 완료)

#### 2-1. extract_embeddings_from_frame_dirs() 수정

**위치**: [pipeline/embedder/base.py:261-314](pipeline/embedder/base.py#L261-L314)

**현재 로직**:
```python
for i, frame_dir in enumerate(frame_dirs):
    embedding, video_name = self.extract_embedding_from_frame_dir(
        frame_dir, save_embedding=save_embeddings
    )
    embeddings.append(embedding)
    video_names.append(video_name)
```

**개선 로직**:
```python
for i, frame_dir in enumerate(frame_dirs):
    video_name = self.extract_video_name_from_frame_dir(frame_dir)
    embedding_path = self._get_embedding_path(video_name)

    if skip_existing and embedding_path.exists():
        # 기존 임베딩 로드
        embedding = np.load(embedding_path)
        logger.info(f"Loaded existing embedding: {embedding_path.name}")
    else:
        # 새로 추출
        embedding, video_name = self.extract_embedding_from_frame_dir(
            frame_dir, save_embedding=save_embeddings
        )

    embeddings.append(embedding)
    video_names.append(video_name)
```

#### 2-2. _get_embedding_path() 헬퍼 메서드 추가

**위치**: [pipeline/embedder/base.py](pipeline/embedder/base.py) (새 메서드)

```python
def _get_embedding_path(self, video_name: str) -> Path:
    """Get the expected embedding file path for a video name."""
    output_dir = Path(self.config.output_dir)
    embedding_filename = video_name.replace(".mp4", ".npy")
    return output_dir / embedding_filename
```

#### 2-3. CLI 플래그 추가

**위치**: [pipeline/cli.py](pipeline/cli.py) (argparse 영역)

```python
parser.add_argument(
    "--skip-existing",
    action="store_true",
    default=True,
    help="Skip videos with existing embedding files (default: True)"
)
parser.add_argument(
    "--force-recompute",
    action="store_true",
    help="Force recompute all embeddings even if files exist"
)
```

**사용법**:
```bash
# 기본: 기존 임베딩 스킵
python video_curation_pipeline.py --frame-dir ./trainlake

# 강제 재추출
python video_curation_pipeline.py --frame-dir ./trainlake --force-recompute
```

#### 2-4. CLI에서 플래그 전달

**위치**: [pipeline/cli.py:137-141](pipeline/cli.py#L137-L141)

```python
# Before
embeddings, video_names, failed_indices = pipeline.embedder.extract_embeddings_from_frame_dirs(
    frame_dirs,
    save_embeddings=True,
    show_progress=True
)

# After
skip_existing = not args.force_recompute  # --force-recompute가 있으면 skip_existing=False
embeddings, video_names, failed_indices = pipeline.embedder.extract_embeddings_from_frame_dirs(
    frame_dirs,
    save_embeddings=True,
    show_progress=True,
    skip_existing=skip_existing
)
```

---

## 최종 예상 결과

### 작업 1 (✅ 완료)
| 사용 방식 | Before | After |
|----------|--------|-------|
| `--frame-dir ./trainlake` | ~60초 | ~5-10초 (첫 실행), <1초 (이후) |

### 작업 2 (⏳ 계획) - 중요도: 매우 높음 (37시간 작업 보호)
| 시나리오 | Before | After |
|----------|--------|-------|
| 첫 실행 (38,000개) | 37시간 | 37시간 |
| 중단 후 재실행 (20,000개 완료 상태) | 37시간 (처음부터) | ~18.5시간 (18,000개만) |
| 두 번째 실행 (모두 존재) | 37시간 | <5분 (로드만) |
| 일부만 존재 (19,000개 신규) | 37시간 | ~18.5시간 (19,000개만) |

---

## 테스트 계획

### 테스트 환경 준비

```bash
# 테스트용 작은 데이터셋 준비
mkdir -p test_trainlake/24.01w/N1234567-240115120000/RAW_DB/FRONT_CMR/CMR_GT_Frame
mkdir -p test_trainlake/24.01w/N1234568-240115130000/RAW_DB/FRONT_CMR/CMR_GT_Frame
mkdir -p test_trainlake/24.02w/N1234569-240215120000/RAW_DB/REAR_CMR/CMR_GT_Frame

# 더미 이미지 생성 (각 디렉토리에 5개씩)
# Python으로 생성하거나 실제 이미지 복사
```

### 작업 1 테스트: 디렉토리 탐색 최적화

#### 테스트 1-1: 기본 동작 확인 (최적화 활성화)

```bash
# 실행
time python video_curation_pipeline.py --frame-dir ./test_trainlake --skip-clustering

# 예상 출력
# - "Use optimized cached search (default)" 메시지
# - 진행률 표시: "Scanning... N directories checked | Current: ..."
# - 캐시 파일 생성: test_trainlake/.frame_dirs_cache.txt
# - 탐색 시간: <1초
```

**검증 항목**:
- [ ] 캐시 파일 생성 확인: `ls test_trainlake/.frame_dirs_cache.txt`
- [ ] 3개 디렉토리 발견 확인: "Found 3 frame directories to process"
- [ ] 탐색 시간 <1초 확인

#### 테스트 1-2: 캐시 재사용 확인

```bash
# 두 번째 실행 (캐시 사용)
time python video_curation_pipeline.py --frame-dir ./test_trainlake --skip-clustering

# 예상 출력
# - 즉시 로드 (<0.1초)
# - "Found 3 frame directories to process"
```

**검증 항목**:
- [ ] 탐색 시간 <0.1초 확인 (즉시 로드)
- [ ] 동일한 3개 디렉토리 발견

#### 테스트 1-3: --no-cache-dirs 플래그 확인

```bash
# glob 방식 사용
time python video_curation_pipeline.py --frame-dir ./test_trainlake --no-cache-dirs --skip-clustering

# 예상 출력
# - "Use standard glob search (slower, for compatibility)" 메시지
# - glob 패턴 사용: "**/CMR_GT_Frame"
```

**검증 항목**:
- [ ] glob 사용 확인 (로그 메시지)
- [ ] 캐시 파일 미생성 또는 미사용

---

### 작업 2 테스트: 임베딩 Skip 로직

#### 테스트 2-1: 첫 실행 (임베딩 생성)

```bash
# output 디렉토리 초기화
rm -rf output_test
mkdir output_test

# 첫 실행
time python video_curation_pipeline.py \
    --frame-dir ./test_trainlake \
    --output-dir ./output_test \
    --skip-clustering

# 예상 출력
# - "Extracting embeddings from frames: 100%|██████| 3/3"
# - "Successful: 3"
# - 각 비디오별 .npy 파일 생성
```

**검증 항목**:
- [ ] 3개 .npy 파일 생성: `ls output_test/*.npy | wc -l` → 3
- [ ] 파일명 패턴 확인: `N1234567-240115120000.npy` 등
- [ ] 파일 크기 > 0 확인: `du -sh output_test/*.npy`

#### 테스트 2-2: 재실행 (Skip 동작 확인)

```bash
# 동일 명령 재실행
time python video_curation_pipeline.py \
    --frame-dir ./test_trainlake \
    --output-dir ./output_test \
    --skip-clustering

# 예상 출력
# - "Loaded existing embedding: N1234567-240115120000.npy" (로그)
# - 처리 시간 대폭 감소 (프레임 로딩/추론 없음)
# - "Successful: 3"
```

**검증 항목**:
- [ ] 로그에 "Loaded existing embedding" 메시지 확인
- [ ] 처리 시간 < 첫 실행의 10% 확인
- [ ] .npy 파일 수정 시간 변경 없음: `stat output_test/*.npy`

#### 테스트 2-3: 일부 삭제 후 재실행

```bash
# 한 파일 삭제
rm output_test/N1234567-240115120000.npy

# 재실행
time python video_curation_pipeline.py \
    --frame-dir ./test_trainlake \
    --output-dir ./output_test \
    --skip-clustering

# 예상 출력
# - 2개는 "Loaded existing embedding"
# - 1개는 새로 추출
# - "Successful: 3"
```

**검증 항목**:
- [ ] 삭제된 파일만 재생성 확인
- [ ] 기존 2개 파일 수정 시간 변경 없음
- [ ] 신규 1개 파일 생성 확인

#### 테스트 2-4: --force-recompute 플래그 확인

```bash
# 강제 재추출
time python video_curation_pipeline.py \
    --frame-dir ./test_trainlake \
    --output-dir ./output_test \
    --skip-clustering \
    --force-recompute

# 예상 출력
# - Skip 로직 비활성화
# - 모든 임베딩 재추출
# - 파일 덮어쓰기
```

**검증 항목**:
- [ ] "Loaded existing embedding" 메시지 없음
- [ ] 모든 .npy 파일 수정 시간 갱신
- [ ] 처리 시간 = 첫 실행과 유사

---

### 통합 테스트: 실제 시나리오

#### 시나리오 1: 중단 후 재개

```bash
# 1단계: 일부 처리 후 중단 시뮬레이션
# - 3개 중 1개만 임베딩 생성하고 중단 (Ctrl+C)

# 2단계: 재실행
python video_curation_pipeline.py \
    --frame-dir ./test_trainlake \
    --output-dir ./output_test \
    --skip-clustering

# 예상 동작
# - 1개는 skip (기존 파일)
# - 2개는 새로 추출
# - 최종 3개 모두 완성
```

**검증 항목**:
- [ ] 기존 1개 파일 재사용 확인
- [ ] 신규 2개 파일 생성 확인
- [ ] 최종 결과 정상 (3개 임베딩)

#### 시나리오 2: 대규모 데이터셋 시뮬레이션

```bash
# 100개 디렉토리 생성 스크립트
for i in {1..100}; do
    week=$(printf "24.%02dw" $((i % 12 + 1)))
    video=$(printf "N%07d-%012d" $i $((240000000000 + i)))
    mkdir -p "test_large/$week/$video/RAW_DB/FRONT_CMR/CMR_GT_Frame"
    # 더미 이미지 1개씩 추가
done

# 첫 실행 (시간 측정)
time python video_curation_pipeline.py \
    --frame-dir ./test_large \
    --output-dir ./output_large \
    --skip-clustering

# 재실행 (skip 효과 측정)
time python video_curation_pipeline.py \
    --frame-dir ./test_large \
    --output-dir ./output_large \
    --skip-clustering
```

**검증 항목**:
- [ ] 첫 실행 vs 재실행 시간 비교 (>10배 차이 예상)
- [ ] 100개 .npy 파일 생성 확인
- [ ] 메모리 사용량 정상 확인

---

### 성능 측정 스크립트

#### measure_performance.sh

```bash
#!/bin/bash

echo "=== Performance Test ==="
echo "Test Date: $(date)"
echo ""

# 테스트 환경 정보
echo "--- Environment ---"
echo "Python: $(python --version)"
echo "Directory count: $(find test_trainlake -type d -name 'CMR_GT_Frame' | wc -l)"
echo ""

# 캐시 초기화
rm -f test_trainlake/.frame_dirs_cache.txt
rm -rf output_test

# 테스트 1: 디렉토리 탐색 (최적화)
echo "--- Test 1: Optimized Directory Scan ---"
/usr/bin/time -v python video_curation_pipeline.py \
    --frame-dir ./test_trainlake \
    --output-dir ./output_test \
    --skip-clustering 2>&1 | grep "Elapsed\|Maximum"

# 테스트 2: 디렉토리 탐색 (캐시 재사용)
echo ""
echo "--- Test 2: Cached Directory Scan ---"
/usr/bin/time -v python video_curation_pipeline.py \
    --frame-dir ./test_trainlake \
    --output-dir ./output_test \
    --skip-clustering 2>&1 | grep "Elapsed\|Maximum"

# 테스트 3: 임베딩 Skip
echo ""
echo "--- Test 3: Embedding Skip (All Exist) ---"
/usr/bin/time -v python video_curation_pipeline.py \
    --frame-dir ./test_trainlake \
    --output-dir ./output_test \
    --skip-clustering 2>&1 | grep "Elapsed\|Maximum"

# 테스트 4: 강제 재추출
echo ""
echo "--- Test 4: Force Recompute ---"
/usr/bin/time -v python video_curation_pipeline.py \
    --frame-dir ./test_trainlake \
    --output-dir ./output_test \
    --skip-clustering \
    --force-recompute 2>&1 | grep "Elapsed\|Maximum"

echo ""
echo "=== Test Complete ==="
```

**사용법**:
```bash
chmod +x measure_performance.sh
./measure_performance.sh > performance_results.txt
```

---

### 자동 검증 스크립트

#### validate_implementation.py

```python
#!/usr/bin/env python3
"""Validate the implementation of performance improvements."""

import os
import sys
from pathlib import Path
import time

def test_cache_file_creation():
    """Test 1: 캐시 파일 생성 확인"""
    cache_file = Path("test_trainlake/.frame_dirs_cache.txt")
    assert cache_file.exists(), "Cache file not created"

    lines = cache_file.read_text().strip().split("\n")
    assert len(lines) == 3, f"Expected 3 directories, found {len(lines)}"
    print("✓ Test 1 passed: Cache file created with 3 entries")

def test_embedding_files():
    """Test 2: 임베딩 파일 생성 확인"""
    output_dir = Path("output_test")
    npy_files = list(output_dir.glob("*.npy"))

    assert len(npy_files) == 3, f"Expected 3 .npy files, found {len(npy_files)}"

    for npy_file in npy_files:
        assert npy_file.stat().st_size > 0, f"Empty file: {npy_file}"

    print("✓ Test 2 passed: 3 embedding files created")

def test_skip_logic():
    """Test 3: Skip 로직 확인 (파일 수정 시간 체크)"""
    output_dir = Path("output_test")
    npy_files = list(output_dir.glob("*.npy"))

    # 수정 시간 기록
    mtimes_before = {f.name: f.stat().st_mtime for f in npy_files}

    # 재실행 (skip 예상)
    time.sleep(1)  # 시간 차이 확보
    os.system("python video_curation_pipeline.py --frame-dir ./test_trainlake --output-dir ./output_test --skip-clustering > /dev/null 2>&1")

    # 수정 시간 재확인
    mtimes_after = {f.name: f.stat().st_mtime for f in output_dir.glob("*.npy")}

    assert mtimes_before == mtimes_after, "Files were modified (skip logic failed)"
    print("✓ Test 3 passed: Skip logic working (files not modified)")

def test_partial_recompute():
    """Test 4: 일부 파일 삭제 후 재생성"""
    output_dir = Path("output_test")
    npy_files = sorted(output_dir.glob("*.npy"))

    # 첫 파일 삭제
    deleted_file = npy_files[0]
    deleted_name = deleted_file.name
    deleted_file.unlink()

    # 재실행
    os.system("python video_curation_pipeline.py --frame-dir ./test_trainlake --output-dir ./output_test --skip-clustering > /dev/null 2>&1")

    # 삭제된 파일 재생성 확인
    assert (output_dir / deleted_name).exists(), f"Deleted file not recreated: {deleted_name}"

    # 총 3개 유지 확인
    assert len(list(output_dir.glob("*.npy"))) == 3, "Expected 3 files after partial recompute"
    print("✓ Test 4 passed: Partial recompute working")

if __name__ == "__main__":
    try:
        test_cache_file_creation()
        test_embedding_files()
        test_skip_logic()
        test_partial_recompute()

        print("\n✓ All tests passed!")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)
```

**사용법**:
```bash
chmod +x validate_implementation.py
./validate_implementation.py
```

---

## 테스트 실행 순서

1. **환경 준비**: 테스트용 디렉토리 생성
2. **작업 1 테스트**: 디렉토리 탐색 최적화 검증
3. **작업 2 구현**: Skip 로직 추가
4. **작업 2 테스트**: Skip 로직 검증
5. **통합 테스트**: 실제 시나리오 테스트
6. **성능 측정**: measure_performance.sh 실행
7. **자동 검증**: validate_implementation.py 실행

## 성공 기준

- [ ] 모든 단위 테스트 통과
- [ ] 통합 테스트 통과
- [ ] 성능 개선 확인:
  - 디렉토리 탐색: >10배 향상 (캐시 사용 시)
  - 임베딩 재실행: >50배 향상 (모두 skip 시)
- [ ] 자동 검증 스크립트 통과
