# 하이퍼파라미터 조정 가이드

## 📋 설정 파일 위치
모든 하이퍼파라미터는 `configs/model_config.yaml` 파일에서 관리됩니다.

## 🎯 주요 하이퍼파라미터

### 모델 구조
```yaml
model:
  backbone: yolov5n          # 백본 모델 (yolov5n, yolov5s 등)
  input_size: 640            # 입력 이미지 크기
  num_classes: 5             # 클래스 수
  
  head:
    attention_type: CBAM     # Attention 타입 (CBAM, SE, ECA)
    reduction_ratio: 16      # Channel attention reduction ratio
    kernel_size: 7           # Spatial attention kernel size
  
  depth_multiple: 0.33       # 모델 깊이 배수
  width_multiple: 0.25       # 모델 너비 배수
```

### 학습 설정
```yaml
training:
  batch_size: 16             # 배치 크기 (Mac M3 Max: 32-64 권장)
  epochs: 100                # 에포크 수
  learning_rate: 0.01        # 초기 학습률
  momentum: 0.937            # SGD 모멘텀
  weight_decay: 0.0005       # 가중치 감쇠
  warmup_epochs: 3           # 워밍업 에포크
```

### 데이터 증강
```yaml
training:
  augmentation:
    hsv_h: 0.015             # 색조 변화
    hsv_s: 0.7               # 채도 변화
    hsv_v: 0.4               # 명도 변화
    degrees: 0.0             # 회전 각도
    translate: 0.1           # 이동 비율
    scale: 0.5               # 크기 변화
    shear: 0.0               # 전단 변형
    perspective: 0.0         # 원근 변환
    flipud: 0.0              # 상하 뒤집기
    fliplr: 0.5              # 좌우 뒤집기
    mosaic: 1.0              # 모자이크 증강
    mixup: 0.0               # 믹스업 증강
```

### 손실 함수 가중치
```yaml
training:
  loss:
    box: 0.05                # 바운딩 박스 손실 가중치
    cls: 0.5                 # 분류 손실 가중치
    obj: 1.0                 # 객체성 손실 가중치
```

## 🚀 Mac M3 Max 최적화 권장 설정

### 고성능 설정 (64GB RAM 활용)
```yaml
training:
  batch_size: 64             # 큰 배치 크기로 안정적 학습
  epochs: 200                # 충분한 학습
  learning_rate: 0.01        # 큰 배치에 맞는 학습률
  
  # 강한 데이터 증강으로 과적합 방지
  augmentation:
    hsv_h: 0.02
    hsv_s: 0.8
    hsv_v: 0.5
    fliplr: 0.5
    mosaic: 1.0
    mixup: 0.15              # 믹스업 추가
```

### 빠른 실험 설정
```yaml
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.02        # 높은 학습률로 빠른 수렴
  
model:
  input_size: 416            # 작은 입력 크기로 속도 향상
```

## 📊 WandB 모니터링 설정

```yaml
wandb:
  enabled: true
  project: "FireSmoke"
  entity: "hyunwoo220"       # 여러분의 WandB 사용자명

experiment_name: "exp_001"   # 실험별 고유 이름
```

## 🎛️ 하이퍼파라미터 튜닝 전략

### 1단계: 기본 설정 확인
```bash
# 작은 설정으로 빠른 테스트
python scripts/train/train.py --config configs/model_config.yaml
```

### 2단계: 배치 크기 최적화
```yaml
# configs/model_config.yaml에서 순차적으로 테스트
batch_size: 16  → 32 → 64 → 128
```

### 3단계: 학습률 스케줄링
```yaml
# 다양한 학습률 테스트
learning_rate: 0.001, 0.01, 0.1
```

### 4단계: 모델 복잡도 조정
```yaml
# 성능 vs 속도 트레이드오프
width_multiple: 0.25 → 0.5 → 0.75
depth_multiple: 0.33 → 0.67 → 1.0
```

## 🔧 실시간 하이퍼파라미터 수정

### 방법 1: 설정 파일 직접 수정
```bash
# configs/model_config.yaml 파일을 에디터로 열어서 수정
vim configs/model_config.yaml
```

### 방법 2: 명령행 인수로 오버라이드 (향후 구현 예정)
```bash
python scripts/train/train.py \
  --config configs/model_config.yaml \
  --batch_size 32 \
  --learning_rate 0.02
```

### 방법 3: 여러 설정 파일 사용
```bash
# 실험별 설정 파일 생성
cp configs/model_config.yaml configs/experiment_001.yaml
cp configs/model_config.yaml configs/experiment_002.yaml

# 각각 다른 설정으로 실험
python scripts/train/train.py --config configs/experiment_001.yaml
python scripts/train/train.py --config configs/experiment_002.yaml
```

## 📈 성능 최적화 팁

### 메모리 최적화
```yaml
# 64GB RAM 활용
training:
  batch_size: 64             # 큰 배치로 학습 안정성 향상
  num_workers: 8             # 데이터 로딩 병렬화
```

### MPS 가속 최적화
```yaml
model:
  input_size: 640            # MPS에 최적화된 크기
training:
  batch_size: 32             # MPS 메모리 효율적 배치 크기
```

### 학습 안정성
```yaml
training:
  warmup_epochs: 5           # 충분한 워밍업
  weight_decay: 0.0001       # 적절한 정규화
```

## 🎯 실험 추천 시나리오

### 실험 1: 기본 성능 확인
```yaml
experiment_name: "baseline"
training:
  batch_size: 16
  epochs: 50
  learning_rate: 0.01
```

### 실험 2: 대용량 배치
```yaml
experiment_name: "large_batch"
training:
  batch_size: 64
  epochs: 100
  learning_rate: 0.02
```

### 실험 3: 강한 증강
```yaml
experiment_name: "heavy_aug"
training:
  augmentation:
    hsv_s: 0.9
    hsv_v: 0.6
    mixup: 0.2
```

### 실험 4: 큰 모델
```yaml
experiment_name: "large_model"
model:
  width_multiple: 0.5
  depth_multiple: 0.67
```

이제 `configs/model_config.yaml` 파일만 수정하면 모든 하이퍼파라미터를 쉽게 조정할 수 있습니다!