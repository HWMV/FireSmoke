# AI Hub 화재 감지 데이터셋 활용 가이드

## 데이터셋 개요

### 구성 및 크기
- **총 이미지 수**: 1,700,000장
  - 화재 장면 이미지: 740,000장
  - 유사 장면 이미지: 660,000장  
  - 무관한 장면 이미지: 300,000장
- **바운딩 박스**: 1,998,784개
- **폴리곤 어노테이션**: 140,000개+

### 데이터 형식
- **이미지 형식**: JPG
- **해상도**: 1920×1280
- **어노테이션 형식**: JSON
- **메타데이터**: 촬영일, 위치, 저작권 정보, DPI 등

## 데이터 다운로드 방법

### 1. 계정 준비
```bash
# AI Hub 회원가입 필요 (한국인만 가능)
# 데이터셋 승인 신청 후 다운로드 가능
```

### 2. 다운로드 옵션
- **웹 다운로드**: AI Hub 포털에서 직접 다운로드
- **API 다운로드**: 프로그래매틱 다운로드
- **오프라인 접근**: K-ICT 빅데이터센터 방문

### 3. 파일 병합 (Linux 환경)
```bash
# 분할된 파일들을 병합
cat fire_detection_part* > fire_detection_complete.zip
unzip fire_detection_complete.zip
```

## 데이터 구조 분석

### 어노테이션 구조 (JSON)
```json
{
  "images": [
    {
      "id": 1,
      "file_name": "fire_001.jpg",
      "width": 1920,
      "height": 1280,
      "date_captured": "2023-XX-XX"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "segmentation": [[x1, y1, x2, y2, ...]],
      "area": 12345,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "fire"},
    {"id": 2, "name": "smoke"}
  ]
}
```

### 클래스 정보 (01-10 카테고리)
- **화재 관련**: 불, 연기, 화염 등
- **건물 외벽**: 깨끗한 벽, 오염된 벽, 손상된 벽 등

## 데이터 전처리 및 활용 방법

### 1. 데이터 변환 스크립트 작성
```python
# AI Hub 형식을 YOLO 형식으로 변환하는 스크립트 필요
def convert_aihub_to_yolo(json_file, output_dir):
    # JSON 파싱
    # 바운딩 박스 좌표 변환 (COCO → YOLO)
    # 클래스 매핑
    pass
```

### 2. 우리 프로젝트 적용 방안

#### 클래스 매핑
```python
# AI Hub 클래스 → 우리 프로젝트 클래스
AIHUB_TO_PROJECT_MAPPING = {
    1: 0,  # fire → fire
    2: 1,  # smoke → smoke
    3: 2,  # wall_clean → wall_clean
    4: 3,  # wall_dirty → wall_dirty
    5: 4,  # wall_damaged → wall_damaged
}
```

#### 데이터 분할
```python
# 학습/검증/테스트 분할 (8:1:1)
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
```

### 3. 성능 벤치마크
- **기준 모델**: YOLACT + ResNet50
- **기준 성능**: mAP 49-50%
- **목표**: YOLOv5-nano + CBAM으로 동등 이상 성능

## 실제 활용을 위한 단계별 가이드

### Step 1: 데이터 획득
```bash
# 1. AI Hub 가입 및 데이터셋 신청
# 2. 승인 후 다운로드
# 3. 파일 압축 해제
```

### Step 2: 데이터 변환
```bash
python data_preparation/convert_aihub.py \
  --input_json /path/to/aihub/annotations.json \
  --input_images /path/to/aihub/images/ \
  --output_dir data/fire_smoke/ \
  --format yolo
```

### Step 3: 데이터 검증
```bash
python data_preparation/validate_dataset.py \
  --data_dir data/fire_smoke/ \
  --check_labels \
  --visualize_samples 100
```

### Step 4: 학습 실행
```bash
python scripts/train/train.py \
  --config configs/model_config.yaml \
  --data data/fire_smoke/
```

## 주의사항 및 제한사항

### 법적 제한사항
- **사용자 제한**: 한국인만 사용 가능
- **용도 제한**: 연구 및 서비스 플랫폼 기술 개발용
- **재배포 금지**: 원본 데이터의 재배포 불가

### 기술적 고려사항
- **용량**: 대용량 데이터셋으로 충분한 저장공간 필요
- **전처리**: JSON → YOLO 형식 변환 필수
- **품질**: 일부 어노테이션 품질 검증 필요

### 성능 최적화 팁
1. **데이터 증강**: 기존 증강 외에 도메인 특화 증강 적용
2. **클래스 밸런싱**: 클래스별 데이터 불균형 해결
3. **하드 네거티브 마이닝**: 어려운 샘플 위주 학습
4. **앙상블**: 다중 모델 조합으로 성능 향상

## 예상 성능 및 효과

### 기대 효과
- **정확도 향상**: 대용량 실제 데이터로 일반화 성능 개선
- **실용성**: 실제 화재 상황 데이터로 현실 적용성 높음
- **벤치마크**: 표준 데이터셋으로 성능 비교 가능

### 성능 목표
- **mAP@0.5**: 55%+ (기준 모델 대비 향상)
- **FPS**: 30+ (젯슨 나노에서)
- **모델 크기**: <10MB (경량화 유지)