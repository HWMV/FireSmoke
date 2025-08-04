# 데이터 준비 및 전처리

AI Hub 화재 감지 데이터셋을 프로젝트에 활용하기 위한 전처리 도구들입니다.

## 파일 구성

- `aihub_dataset_guide.md`: AI Hub 데이터셋 상세 분석 및 활용 가이드
- `convert_aihub.py`: AI Hub 형식을 YOLO 형식으로 변환
- `validate_dataset.py`: 변환된 데이터셋 검증 및 시각화

## 사용 순서

### 1. AI Hub 데이터 다운로드
```bash
# AI Hub 포털에서 화재 감지 데이터셋 다운로드 및 압축 해제
# (한국인만 가능, 승인 필요)
```

### 2. 데이터 형식 변환
```bash
python data_preparation/convert_aihub.py \
  --input_json /path/to/aihub/annotations.json \
  --input_images /path/to/aihub/images/ \
  --output_dir data/fire_smoke/
```

### 3. 데이터셋 검증
```bash
# 전체 검증
python data_preparation/validate_dataset.py \
  --data_dir data/fire_smoke/ \
  --check_labels

# 샘플 시각화
python data_preparation/validate_dataset.py \
  --data_dir data/fire_smoke/ \
  --visualize_samples 10 \
  --split train
```

## 주요 기능

### convert_aihub.py
- COCO 형식 → YOLO 형식 변환
- 클래스 매핑 (AI Hub → 프로젝트)
- 자동 train/val/test 분할 (8:1:1)
- 데이터셋 설정 파일 생성

### validate_dataset.py
- 폴더 구조 검증
- 라벨 형식 검증
- 이미지-라벨 쌍 매칭 확인
- 클래스 분포 분석
- 샘플 시각화

## 클래스 매핑

| AI Hub ID | AI Hub 클래스 | 프로젝트 클래스 | 매핑 ID |
|-----------|---------------|-----------------|---------|
| 1 | fire | fire | 0 |
| 2 | smoke | smoke | 1 |
| 3 | wall_clean | wall_clean | 2 |
| 4 | wall_dirty | wall_dirty | 3 |
| 5 | wall_damaged | wall_damaged | 4 |

## 데이터셋 통계

- **총 이미지**: 1,700,000장
- **해상도**: 1920×1280
- **어노테이션**: 바운딩 박스 + 폴리곤
- **클래스**: 5개 (화재, 연기, 벽 상태 3종)

## 주의사항

1. **용량**: 대용량 데이터셋으로 충분한 저장공간 필요
2. **권한**: AI Hub 가입 및 데이터셋 승인 필요
3. **형식**: JSON 어노테이션을 YOLO 형식으로 변환 필수
4. **검증**: 변환 후 반드시 데이터 품질 검증 수행