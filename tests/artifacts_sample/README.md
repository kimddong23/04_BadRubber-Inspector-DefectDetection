# Sample output of artifact_saver

이 디렉토리는 `defect_detection.utils.artifact_saver.save_batch_artifacts` 의 실제 실행 결과 중
**대표 이미지 1건**을 레포에 커밋해 둔 것입니다.

## 원본 입력
- 이미지: `BR-B_4_20260410_000001_797.jpg`
- 촬영: BR-B 라인, 카메라 4번, 2026-04-10
- 해상도: 2432 × 2048
- 원본 위치 (사내 NAS): `/백업/LG_Chemistry_Site/BR-B/2026-04-10/...CAM4/`

## 실행 환경
- Google Colab, **NVIDIA L4** GPU (Ada Lovelace, sm_89)
- PyTorch 2.5.1+cu121 / CUDA 12.1 / cuDNN 9.1.0
- Python 3.12.13

## 생성 파일

### artifact_saver 가 직접 만드는 파일 (본 PR 의 레포 기능)

| 파일 | 내용 | 크기 |
| --- | --- | --- |
| `heatmap.jpg` | AnomalyCLIP output 의 JET 컬러맵 시각화 | 348 KB |
| `heatmap.npy` | raw float32 anomaly map (post-resize + foreground mask 적용, 원본 해상도) | 19 MB |
| `metadata.json` | 크롭 좌표 · classification · segmentation 좌표/클래스 전체 | 58 KB |
| `crops/*.jpg` | anomaly region 별 크롭 이미지 (앞 5장만 샘플로 포함) | 75 KB |

`main.py` 플래그 대응:
- `--save-artifacts` → 디렉토리 + `heatmap.jpg` + `metadata.json` + `crops/`
- `--save-heatmap-npy` → `heatmap.npy` 추가

### `intermediates/` — Colab 디버깅 훅 전용 (레포 코드에 없음)

AnomalyCLIP 모델 내부 중간값을 확인하려고 Colab 노트북에서 `AnomalyCLIPInference.infer`
를 monkey-patch 해 캡처한 것. 본 레포 코드의 `Detector` / `artifact_saver` 는 이 파일들을
생성하지 않습니다 — 레포 기능 (artifact_saver) + 외부 디버깅 훅으로 역할이 나뉜 구조입니다.

| 파일 | shape / dtype | 설명 |
| --- | --- | --- |
| `intermediates/anomaly_map_pre_resize.npy` | (512, 512) float32 | resize 전 raw anomaly map. 본 샘플은 **max = 1.341** (1.0 초과) |
| `intermediates/image_features.npy` | (768,) float32 | CLIP ViT-L/14 image embedding |
| `intermediates/patch_features_layer00..03.npy` | (1297, 768) float32 × 4 layers | Deep attention layer 별 patch feature |

## 전체 실행 결과 (참고)

- 이미지 132장 전수 처리 성공 (입력 폴더: CAM4)
- 총 1,717 crops (이미지당 평균 13.01, min 4, max 22)
- classification pass / ng : 1,413 / 304
- 자세한 수치는 PR 본문 참조
