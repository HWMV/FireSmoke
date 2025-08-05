# 화재/연기 및 건물 외벽 오염 감지 시스템

YOLOv5-nano backbone에 CBAM Attention 모듈을 적용한 경량화 객체 감지 시스템입니다.
젯슨 나노에서 실시간으로 동작 가능하도록 최적화되었습니다.

## 주요 특징

- **경량화 모델**: YOLOv5-nano (width: 0.25, depth: 0.33) 기반
- **Attention 메커니즘**: CBAM (Channel & Spatial Attention) 적용으로 도메인 특화 성능 향상
- **다중 클래스 감지**: 
  - 화재 (fire)
  - 연기 (smoke)
  - 깨끗한 벽 (wall_clean)
  - 오염된 벽 (wall_dirty)
  - 손상된 벽 (wall_damaged)
- **젯슨 나노 최적화**: TensorRT 지원, FP16 연산 가능

## 설치 방법

```bash
# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

## 프로젝트 구조

```
FireSmokeDetection/
├── configs/
│   └── model_config.yaml      # 모델 설정 파일
├── data/                      # 데이터셋 디렉토리
│   ├── fire_smoke/
│   └── wall_contamination/
├── models/
│   ├── backbone/              # YOLOv5 백본
│   ├── heads/                 # Attention 헤드
│   └── utils/                 # 유틸리티 (dataset, loss)
├── scripts/
│   ├── train/                 # 학습 스크립트
│   └── evaluate/              # 평가 스크립트
├── outputs/                   # 출력 디렉토리
│   ├── checkpoints/
│   ├── logs/
│   └── visualizations/
└── demo.py                    # 데모 스크립트
```

## 사용 방법

### 1. 데모 실행

```bash
# 전체 데모 실행
python demo.py --create-data --train-demo --inference-demo

# 개별 데모
python demo.py --create-data     # 샘플 데이터 생성
python demo.py --train-demo      # 학습 모니터링 데모
python demo.py --inference-demo  # 추론 데모
```

### 2. 모델 학습

```bash
# 학습 시작
python scripts/train/train.py --config configs/model_config.yaml

# TensorBoard로 학습 모니터링
tensorboard --logdir outputs/logs
```

### 3. 모델 평가

```bash
# 테스트셋 평가
python scripts/evaluate/evaluate.py --model outputs/checkpoints/best.pth
```

## 모델 아키텍처

### Backbone: YOLOv5-nano
- 젯슨 나노에 최적화된 경량 버전
- Width multiplier: 0.25
- Depth multiplier: 0.33

### Custom Head with Attention
- CBAM (Convolutional Block Attention Module) 적용
- Channel Attention + Spatial Attention
- 도메인 특화 feature 학습을 통한 성능 향상

## 학습 설정

- Batch size: 16
- Epochs: 100
- Learning rate: 0.01
- Optimizer: SGD with momentum
- Data augmentation: 
  - Random resize crop
  - Horizontal flip
  - Color jitter
  - Gaussian noise

## 성능 지표

- 실시간 처리: 젯슨 나노에서 30+ FPS
- 높은 정확도: 도메인 특화 데이터로 튜닝
- 경량화: 모델 크기 < 10MB

## 데이터셋

**AI Hub 화재 감지 데이터셋**: https://aihub.or.kr/aihubdata/data/view.do?pageIndex=3&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%ED%99%94%EC%9E%AC&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=176

## 모니터링 및 시각화

- **시각화**: Gradio UI로 간단한 인터페이스 제공 예정
- **모니터링**: WandB 설정으로 실시간 학습 모니터링 예정
- **TensorBoard**: 기본 학습 로그 시각화

## 향후 계획

- [ ] Gradio 기반 웹 UI 개발
- [ ] WandB 모니터링 시스템 구축
- [ ] TensorRT 변환 및 최적화 (젯슨 나노 적용)
- [ ] INT8 양자화 지원
- [ ] 실시간 비디오 스트림 처리

## 라이센스

본 프로젝트는 연구 및 개발 목적으로 제작되었습니다.


## 후속 작업