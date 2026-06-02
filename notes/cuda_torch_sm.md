# CUDA 버전 × `sm_xx` × PyTorch 버전 관계 정리

## 1. 큰 그림 — 3개의 독립적인 차원

GPU에서 PyTorch를 돌리기까지 얽혀 있는 버전 정보는 사실 **3개의 독립 축**으로 나뉜다.

| 축 | 결정 주체 | 예시 |
|---|---|---|
| **PyTorch 버전** | 라이브러리 자체 (API, 기능) | `2.7.0`, `2.8.0`, `2.11.0` |
| **CUDA 빌드 (`cuXXX`)** | wheel을 컴파일할 때 쓴 CUDA Toolkit | `+cu118`, `+cu126`, `+cu128` |
| **GPU 아키텍처 (`sm_xx`)** | 물리 GPU의 Compute Capability | `sm_80` (A100), `sm_90` (H100), `sm_120` (Blackwell) |

이 셋이 모두 맞아야 GPU 커널이 실행된다. 어긋나면 `CUDA error: no kernel image is available for execution on the device` 가 뜬다.

---

## 2. 각 차원 자세히

### 2-1. PyTorch 버전 (`2.7.0`, `2.8.0` ...)
- 라이브러리 코드 자체의 버전. API 변경, 새 모델 지원, 버그 fix 등이 들어간다.
- transformers, peft 등 다른 패키지가 요구하는 호환 범위가 있다 → 함부로 올리면 다른 의존성이 깨질 수 있음.

### 2-2. CUDA 빌드 태그 (`+cu126`, `+cu128`)
- "이 wheel을 CUDA Toolkit 몇 버전으로 컴파일했냐"의 표시.
- 같은 `torch==2.7.0` 도 빌드는 여러 개:
  - `torch-2.7.0+cu118`
  - `torch-2.7.0+cu126`
  - `torch-2.7.0+cu128`
- 빌드별로 wheel 안에 들어가는 **GPU 커널 코드**(PTX/SASS)가 달라진다.

### 2-3. GPU `sm_xx` (Compute Capability)
- GPU 하드웨어 세대의 ID. `nvidia-smi`로 모델 확인 후 검색 (또는 `torch.cuda.get_device_capability()`).

| 세대 | sm | 대표 GPU |
|---|---|---|
| Pascal | sm_60, sm_61 | GTX 10 |
| Volta | sm_70 | V100 |
| Turing | sm_75 | RTX 20, T4 |
| Ampere | sm_80, sm_86 | A100, RTX 30 |
| Ada Lovelace | sm_89 | RTX 40 |
| Hopper | sm_90 | H100 |
| **Blackwell** | **sm_100, sm_120** | **B100, RTX 50 / RTX PRO Blackwell** |

---

## 3. CUDA Toolkit 버전이 어떤 `sm_xx`를 지원하나

CUDA Toolkit의 nvcc(컴파일러)는 **빌드 시점에 알려진 아키텍처**까지만 코드를 생성할 수 있다. 그래서:

| CUDA Toolkit | 추가 지원되는 `sm_xx` |
|---|---|
| CUDA 10.x | sm_75 (Turing) |
| CUDA 11.0~11.7 | sm_80 (Ampere) |
| CUDA 11.8 | sm_89 (Ada) |
| CUDA 12.0~12.3 | sm_90 (Hopper) |
| CUDA 12.4~12.6 | sm_90 까지 |
| **CUDA 12.8** | **sm_100, sm_120 (Blackwell)** |
| CUDA 12.9+ | 동일 + 일부 fix |

즉 **Blackwell(sm_120)를 쓰려면 CUDA 12.8 이상으로 빌드된 wheel이 필요**하다.

---

## 4. torch × CUDA 빌드 호환 매트릭스 (대략)

| torch | cu118 | cu121 | cu124 | cu126 | cu128 |
|---|---|---|---|---|---|
| 2.4.x | ✅ | ✅ | ✅ | ❌ | ❌ |
| 2.5.x | ✅ | ✅ | ✅ | ❌ | ❌ |
| 2.6.x | ✅ | — | ✅ | ✅ | ❌ |
| **2.7.x** | ✅ | — | ✅ | ✅ | **✅** |
| 2.8.x | ❌ | — | — | ✅ | ✅ |
| 2.9.x~ | ❌ | — | — | ✅ | ✅ |

(빈 칸/❌는 공식 wheel 미제공.)

조회: `pip index versions torch --index-url https://download.pytorch.org/whl/cu128`

---

## 5. 시스템의 다른 CUDA 요소들과의 관계

흔히 헷갈리는 4개를 정리:

| 요소 | 역할 | 어디서 확인 |
|---|---|---|
| **NVIDIA 드라이버** | 커널 모드 드라이버. 이 시스템이 **최대 어떤 CUDA 런타임 버전까지 돌릴 수 있는지** 결정 (backward-compatible) | `nvidia-smi` 상단의 `CUDA Version: 13.0` 같은 표시 |
| **시스템 nvcc / CUDA Toolkit** | 시스템에 깔린 CUDA 컴파일러 (직접 C++/CUDA 컴파일할 때만 사용) | `nvcc --version` |
| **wheel에 포함된 CUDA 런타임** | torch wheel 설치 시 함께 깔리는 `nvidia-cuda-runtime-cu12`, `nvidia-cudnn-cu12` 등 | `pip list \| grep nvidia` |
| **torch가 빌드된 CUDA 버전** | wheel 빌드 시 쓴 nvcc 버전. wheel 이름의 `+cu128`에 대응 | `torch.version.cuda`, `torch.__version__` |

**핵심 규칙:**
1. 시스템 nvcc 버전과 torch wheel의 cu 버전은 **다를 수 있다** (torch는 wheel에 자기 런타임을 포함).
2. **드라이버 ≥ wheel CUDA 버전**이기만 하면 된다. (예: 드라이버 580 → CUDA 13 지원 → cu128 wheel 잘 돔)
3. wheel의 `sm_xx` 지원 범위는 **wheel을 빌드한 CUDA Toolkit 버전**에 의해 결정된다 (4번 항목).

---

## 6. 실전: 내 환경에 맞는 torch 고르기

```bash
# 1) GPU compute capability 확인
nvidia-smi --query-gpu=name,compute_cap --format=csv

# 2) 드라이버가 어디까지 지원하는지
nvidia-smi | head -3   # "CUDA Version: 13.0" 같은 줄

# 3) 현재 torch가 컴파일된 sm 목록
python -c "import torch; print(torch.cuda.get_arch_list())"

# 4) 필요한 cu 빌드 결정
#    GPU가 sm_120(Blackwell)이면 → cu128 이상
#    GPU가 sm_90(H100)이면      → cu124+ 면 OK
#    GPU가 sm_86(RTX 30)이면    → cu118+ 다 OK

# 5) 동일 torch 버전을 cu128 빌드로 갈아끼우기 (다른 패키지 호환 유지)
pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.0 torchvision==0.22.0
```

---

## 7. 에러 트러블슈팅

### `CUDA error: no kernel image is available for execution on the device`
→ **GPU의 `sm_xx`가 torch wheel의 arch list에 없다.** wheel을 더 높은 cu 빌드로 교체.

확인:
```python
import torch
print("device cap:", torch.cuda.get_device_capability())   # 예: (12, 0) → sm_120
print("wheel archs:", torch.cuda.get_arch_list())          # sm_120 없으면 문제
```

### `UserWarning: ... with CUDA capability sm_XXX is not compatible ...`
→ 같은 원인. 경고만 뜨고 첫 커널 실행 전까지는 안 죽지만, 결국 위 에러로 이어진다.

### `RuntimeError: CUDA driver version is insufficient for CUDA runtime version`
→ **드라이버가 너무 낮다.** 드라이버를 업데이트하거나, 더 낮은 cu 빌드 wheel을 쓴다.

---

## 8. 이번 케이스 (2026-06-01 기록)

| 항목 | 값 |
|---|---|
| GPU | RTX PRO 5000 Blackwell |
| Compute capability | `sm_120` |
| 드라이버 | 580.95.05 (CUDA 13.0 까지) |
| 시스템 nvcc | 12.1 (사용 안 함) |
| **문제 발생 시 torch** | `2.7.0+cu126` → arch list `[sm_50..sm_90]` (sm_120 없음) |
| **해결 후 torch** | `2.7.0+cu128` → arch list `[sm_75, sm_80, sm_86, sm_90, sm_100, sm_120, compute_120]` ✅ |

해결 명령:
```bash
pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.0 torchvision==0.22.0
```

같은 torch 버전(2.7.0)으로 cu 빌드만 바꿔서 transformers/peft/accelerate 등 의존성은 그대로 유지됨.
