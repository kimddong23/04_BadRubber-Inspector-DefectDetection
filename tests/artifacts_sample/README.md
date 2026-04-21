# Sample output of `python main.py --save-artifacts`

이 디렉토리는 본 PR 이 추가한 `save_batch_artifacts` 기능의 실제 실행 결과 중
**대표 이미지 1개** 를 레포에 커밋해 둔 것입니다.

## 원본 입력
- 이미지: `BR-B_4_20260410_000001_797.jpg`
- 촬영: BR-B 라인, 카메라 4번, 2026-04-10
- 해상도: 2432 × 2048
- 위치 (사내 NAS): `/백업/LG_Chemistry_Site/BR-B/2026-04-10/.../CAM4/`

## 실행 환경
- Google Colab, NVIDIA L4 GPU (Ada Lovelace, sm_89)
- PyTorch 2.5.1+cu121 / CUDA 12.1 / cuDNN 9.1.0
- Python 3.12.13

## 생성 파일
- `heatmap.jpg` — AnomalyCLIP output 의 JET 컬러맵 시각화
- `metadata.json` — 크롭 좌표 · 분류 · 세그멘테이션 정보 전부 포함
- `crops/*.jpg` — anomaly region 별 크롭 이미지 (앞 5장만 샘플로 포함)

## 실제 전체 실행 결과 (참고)
- 총 132 이미지 전부 처리 성공
- 총 1,717 crops (이미지당 평균 13.01)
- classification pass/ng: 1,413 / 304
- 자세한 수치는 PR 본문 "실측 결과" 섹션 참조
