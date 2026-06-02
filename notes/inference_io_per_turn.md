# LaDeCo Inference — turn별 Model Input / Output 정리

`llava/infer/infer.py` 의 inference 루프가 한 sample에 대해 어떤 입력을 모델에 넣고 무엇을 받는지, **5 turn 각각**에 대해 정리한 문서입니다.

---

## 전체 개요

각 sample(=디자인 1개)마다:

```
for turn_id in 0..4:
    1. dataset.__getitem__(idx, end_layer_index=turn_id, gpt_dict, images, new_images)
       → input_ids, image tensor stack 을 빌드
    2. model.generate(input_ids, images=...) 호출 1회
    3. 출력 string → predictions 누적, render() 호출 → {num}_{id}_{turn_id}.png 저장
    4. 그 PNG를 next turn의 `current canvas state: <image>` 자리로 매핑
```

- **5 turn = 5번의 `model.generate()` 호출** (start/end_layer_index 범위 밖이면 GT 사용, generate 안 함)
- 모든 turn은 **같은 input_ids를 새로 build** 하고, **prefix는 동일하게 유지**되며 turn마다 conversation 뒷부분이 길어짐 (autoregressive multi-turn).
- 같은 sample 안에서 모델 weights, tokenizer, image processor는 그대로 — turn 간 변하는 건 conversation 길이와 `<image>` 토큰에 매핑되는 실제 이미지 텐서들뿐.

---

## Generation hyper-params (모든 turn 공통)

`infer.py:206-216`

| 인자 | 값 | 의미 |
|---|---|---|
| `do_sample` | `True` | 확률적 샘플링 |
| `temperature` | `0.7` (`eval_args.temperature`) | 분포 sharpness |
| `top_p` | `0.95` | nucleus sampling |
| `num_return_sequences` | `1` (assert로 강제) | 1 sample만 |
| `pad_token_id` | `tokenizer.eos_token_id` | 패딩 토큰 |
| `max_length` | `5000` | 새로 생성할 토큰 한도 (prompt 포함 길이 한도가 아님 → 사실상 매우 큼) |

---

## Per-turn 입력/출력 — 8 elements 예시 (`5a2fb0ccd8141396fe9b527f`)

이 디자인은 element 8개, role 분포 = Background 2, Underlay 0, Logo/Image 2, Text 2, Embellishment 2.

### 공통 헤더 (모든 turn의 prompt 맨 앞)

turn 0 의 첫 human message에서 시작해 turn N까지 conversation이 누적됩니다. 첫 message는 디자인 전체 컨텍스트(canvas 크기 + element enumerate)를 한 번에 줍니다:

```
A poster of canvas width 1080px, canvas height 1080px.
element 0: <image>,
element 1: <image>,
element 2: <image>,
element 3: <image>,
element 4: "Happy Hanukkah",
element 5: "May the light be in your heart all year long",
element 6: <image>,
element 7: <image>
Please predict step by step according to the semantics of the elements.
After each prediction, there will be an intermediate rendering result as a reference to better make the next prediction.

Now predict the background elements: element 0: <image>, element 1: <image>
```

→ 6개의 `<image>` 토큰 위치에 element 이미지가 매핑됨 (text element 4, 5는 텍스트로 그대로 들어가서 이미지 자리 없음).
→ 마지막에 다시 background element만 별도로 enumerate.

---

### Turn 0 — Background

#### Input
- **Text prompt** = 위 공통 헤더만 (= `conversations[0]["value"]`).
- **Image tensors** (input_ids의 `<image>` 토큰 순서대로):
  | slot | 파일 |
  |---|---|
  | 0 | `5a2fb0ccd.../ele_1.png` (Background) |
  | 1 | `5a2fb0ccd.../ele_0.png` (Background) |
  | 2 | `5a2fb0ccd.../ele_3.png` (Logo/Image) |
  | 3 | `5a2fb0ccd.../ele_2.png` (Logo/Image) |
  | 4 | `5a2fb0ccd.../ele_7.png` (Embellishment) |
  | 5 | `5a2fb0ccd.../ele_6.png` (Embellishment) |
  | 6 | `5a2fb0ccd.../ele_1.png` (Background, 재등장 — "Now predict the background" 부분) |
  | 7 | `5a2fb0ccd.../ele_0.png` (Background, 재등장) |

  text element 4, 5의 자리에는 `<image>`가 없으므로 image 텐서가 없음. (test.json의 `image` 리스트 인덱스 0–7 에 대응.)

- **이미지 전처리** (`infer.py:99-111`):
  1. `Image.open(image_folder + path)` 로 로딩
  2. `white_rgb_convert(img)`: RGBA → 흰색 배경 위에 paste → RGB
  3. `expand2square(img, mean_color)`: 짧은 변에 CLIP image_mean 색으로 padding → 정사각
  4. `processor.preprocess(...)`: CLIP-L/14-336용 resize/normalize → `(3, 336, 336)` 텐서
  5. 실패 시 `torch.zeros(3, 336, 336)` placeholder

#### Output (예시)
```json
{"index": 0, "left": -4, "top": -4, "width": 1087, "height": 1087}
{"index": 1, "left": 0, "top": 0, "width": 1080, "height": 1080}
```
- Background element 2개의 layout JSON이 공백으로 join되어 출력
- 각 JSON에는 `index, left, top, width, height` 만. text가 아니므로 font 관련 속성 없음.

#### Turn 종료 후
- `gpt_dict[0] = "{...} {...}"` (위 출력 그대로 저장)
- `render(...)` 호출 → 전체 캔버스에 element 0, 1을 위 layout으로 그려서 `{num}_{id}_0.png` 저장
- `new_images = {layer_image_list[0]: <방금 저장한 png 경로>}` 로 turn 1 입력 준비

---

### Turn 1 — Underlay

#### Input
- **Text prompt** = 공통 헤더 + turn 0의 모델 출력 + turn 1 질문

  ```
  [공통 헤더 ... ]

  ASSISTANT: {"index": 0, ...} {"index": 1, ...}

  USER: current canvas state: <image>. Now predict the underlay elements: null
  ```

  - `ASSISTANT` 자리에는 모델이 turn 0에서 직접 뱉은 string (gpt_dict[0]). teacher-forcing 모드(layer가 generate 범위 밖)면 test.json의 GT 사용.
  - `null` ← 이 디자인에 underlay element가 0개라서 `create_dataset.py`가 그렇게 채움.

- **Image tensors** (turn 0 대비 변동):
  | slot | 파일 | 변경? |
  |---|---|---|
  | 0–7 | (turn 0과 동일) | 그대로 |
  | **8** | **`{num}_{id}_0.png` (방금 저장한 pred 렌더)** | **NEW — turn 0 결과** |

  - `new_images = {8: <pred 0번 PNG 경로>}` 로 슬롯 8 의 텐서만 교체됨 (`infer.py:113-118`).
  - 학습 시 test.json에는 이 슬롯에 `5a2fb0ccd.../layer_0.png` (GT) 가 매핑돼 있지만, inference는 자기가 만든 PNG로 덮어씀.

#### Output
```
{}
```
- Underlay element 0개라서 빈 JSON 1개만 출력.
- (만약 underlay가 있었다면 layer 0과 동일하게 N개의 JSON이 join돼 나옴.)

#### Turn 종료 후
- render() 다시 호출 → turn 0 + 1 누적 → `{num}_{id}_1.png` 저장 (이 디자인은 underlay 없어서 turn 0과 거의 같은 이미지)
- `new_images = {layer_image_list[1]: <{num}_{id}_1.png>}` 로 turn 2 준비

---

### Turn 2 — Logo/Image

#### Input
- **Text prompt** = 공통 헤더 + turn 0, 1 출력 + turn 2 질문

  ```
  [공통 헤더]
  ASSISTANT: {turn 0 출력}
  USER: current canvas state: <image>. Now predict the underlay elements: null
  ASSISTANT: {}
  USER: current canvas state: <image>. Now predict the logo/image elements: element 2: <image>, element 3: <image>
  ```

  → Logo/Image 질문에는 해당 layer의 element들이 다시 한 번 enumerate됨.

- **Image tensors**:
  | slot | 파일 | 변경? |
  |---|---|---|
  | 0–7 | 초기 element 이미지들 | 그대로 |
  | 8 | `{num}_{id}_0.png` (turn 0 결과) | turn 1에서 교체된 채로 유지 |
  | **9** | **`{num}_{id}_1.png` (turn 1 결과)** | **NEW** |
  | **10** | **`5a2fb0ccd.../ele_3.png`** (Logo/Image) | **NEW — "Now predict logo/image" 부분** |
  | **11** | **`5a2fb0ccd.../ele_2.png`** (Logo/Image) | **NEW** |

  - test.json의 `image` 리스트 인덱스 10, 11 은 이 layer의 element들 (재등장).
  - 슬롯 9는 layer_image (현 캔버스 상태), 10–11은 raw element ↔ 이건 turn 시작 시 이미 `images` 리스트에 들어 있던 것.

#### Output
```json
{"index": 2, "left": 394, "top": 217, "width": 293, "height": 287}
{"index": 3, "left": 136, "top": 136, "width": 807, "height": 807}
```
- Logo/Image element 2개의 layout. text가 아니라서 geometry-only.

#### Turn 종료 후
- render() → `{num}_{id}_2.png` 저장
- `new_images = {layer_image_list[2]: ...}` 로 turn 3 준비

---

### Turn 3 — Text

#### Input
- Text prompt = 누적된 4 turn + turn 3 질문:
  ```
  ...
  USER: current canvas state: <image>. Now predict the text elements: element 4: "Happy Hanukkah", element 5: "May the light be in your heart all year long"
  ```
  - Text element는 텍스트 그대로 들어감, `<image>` 토큰 없음.

- Image tensors:
  | slot | 파일 |
  |---|---|
  | 0–9 | (이전 turn에서 채워진 그대로) |
  | **12** | **`{num}_{id}_2.png` (turn 2 결과)** ← NEW |

  → text element는 이미지가 없으니 새 image 슬롯은 캔버스 상태 1개만 추가됨.

#### Output (Text는 다른 layer와 다르게 추가 속성을 많이 가짐)
```json
{"index": 4, "left": 170, "top": 543, "width": 741, "height": 234, "angle": 0, "font": "Mate", "font_size": 114, "color": [209, 182, 43], "text_align": "center", "capitalize": "true", "letter_spacing": 0.0, "line_height": 1.0}
{"index": 5, "left": 251, "top": 784, "width": 575, "height": 119, "angle": 0, "font": "Parisienne", "font_size": 47, "color": [147, 149, 152], ...}
```

`create_dataset.py`의 schema가 text 에 대해서 자동으로 추가 필드를 요구합니다:
- `angle` (회전각도)
- `font` (family 이름)
- `font_size`
- `color` (RGB 0–255)
- `text_align` ("center" | "left" | "right")
- `capitalize` ("true" | "false")
- `letter_spacing`
- `line_height`

---

### Turn 4 — Embellishment

#### Input
- Text prompt:
  ```
  ...
  USER: current canvas state: <image>. Now predict the embellishment elements: element 6: <image>, element 7: <image>
  ```

- Image tensors:
  | slot | 파일 |
  |---|---|
  | 0–12 | (이전 turn까지 채워진 그대로) |
  | **13** | **`{num}_{id}_3.png` (turn 3 결과)** ← NEW |
  | **14** | **`ele_7.png` (Embellishment)** ← NEW |
  | **15** | **`ele_6.png` (Embellishment)** ← NEW |

#### Output
```json
{"index": 6, "left": 675, "top": 193, "width": 96, "height": 265}
{"index": 7, "left": 305, "top": 193, "width": 96, "height": 265}
```

#### Turn 종료 후
- render() → `{num}_{id}_4.png` 저장 — **최종 디자인**.
- `pred.jsonl` 에 sample 1개의 `data` 한 줄 기록.

---

## 핵심 코드 포인트

### Image 슬롯 vs `<image>` 토큰 매핑
`tokenizer_image_token(prompt, tokenizer, ...)` 가 prompt string의 `<image>` 토큰을 토크나이저 차원에서 placeholder ID로 바꾸고, model이 forward할 때 `images=[...]` 리스트의 i번째 텐서를 i번째 `<image>` 위치에 끼웁니다.

- input_ids에서 `<image>` 토큰의 등장 횟수 = `images` 리스트 길이.
- `test.json["image"]` 의 순서가 곧 `<image>` 등장 순서.

### Conversation slicing
`infer.py:124-131`:
```python
for sentence in sources["conversations"]:
    if sentence["from"] == "gpt" and turn_id in gpt_dict:
        sentence["value"] = gpt_dict[turn_id]   # 모델 출력으로 덮어쓰기
    conv.append_message(sentence["from"], sentence["value"])
    if sentence["from"] == "gpt" and turn_id == end_layer_index:
        break                                   # 현 turn까지만 prompt에 포함
    if sentence["from"] == "gpt":
        turn_id += 1
```

→ `gpt_dict` 에 채워진 turn은 **모델이 직접 뱉었거나** (정상 generation) **GT로 강제** (layer가 [start, end] 범위 밖) 된 것.

### Image 슬롯 갱신
`infer.py:113-118`, `infer.py:240`:
```python
for k, v in new_images.items():   # 다음 turn entry에서
    images[k] = processor.preprocess(Image.open(v))[0]
# ...
new_images = {layer_image_list[turn_id]: f"{num}_{id}_{turn_id}.png"}
```

→ `layer_image_list` 는 첫 `__getitem__` 호출에서 `image_list` 중 `"layer_"` 포함된 path들의 인덱스를 기록한 리스트. test.json에서 미리 layer_0~3.png 자리가 정해져 있으니, 그 슬롯들을 inference 결과로 덮어씀.

---

## start/end_layer_index 의 의미

| 옵션 | 동작 |
|---|---|
| `--start 0 --end 4` (default) | 5개 layer 전부 모델이 생성 — full inference |
| `--start 2 --end 4` | layer 0, 1 은 GT 사용, layer 2, 3, 4 만 generate — "lower layers given" controlled 평가 |
| `--start 0 --end 1` | layer 2, 3, 4 는 GT 사용, layer 0, 1 만 generate — "upper layers given" |

GT 사용 turn은 `model.generate()` 가 호출되지 않고 `gpt_dict[turn_id] = gts[turn_id]` 로 바로 채워짐 (`infer.py:189-191`). render는 그대로 호출되므로 PNG는 5장 다 나옴.

---

## 한 sample 처리 시간

- generate() 1회당 수십 토큰 ~ 수백 토큰 출력 (element 수에 비례)
- 5 turn × `model.generate()` = sample 1개당 generate 5회
- 16개 이미지 텐서가 vision encoder를 통과 (한 번에 모두 인코딩, 캐시 가능)
- 8 element 디자인 기준 약 30초~수 분 (GPU 사양에 따라)

`pred.jsonl` 에는 sample마다 1줄씩 기록되므로, 진행 도중 중단해도 그때까지 완료된 sample은 남습니다.
