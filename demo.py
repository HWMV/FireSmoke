#!/usr/bin/env python3
"""
화재/연기 및 건물 외벽 오염 감지 데모 스크립트
YOLOv5-nano + CBAM Attention 기반 모델
"""

import os
import torch
import cv2
import numpy as np
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from models import FireSmokeDetector

def create_sample_data():
    """데모용 샘플 데이터 생성"""
    print("샘플 데이터 생성 중...")
    
    # 샘플 이미지 디렉토리 생성
    sample_dirs = [
        'data/fire_smoke/train/images',
        'data/fire_smoke/train/labels',
        'data/fire_smoke/val/images',
        'data/fire_smoke/val/labels',
        'data/fire_smoke/test/images',
        'data/fire_smoke/test/labels'
    ]
    
    for dir_path in sample_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # 샘플 이미지 생성 (실제 데모에서는 실제 이미지 사용)
    for i in range(5):
        # 빈 이미지 생성 (실제로는 화재/연기/벽 이미지 사용)
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Train set
        cv2.imwrite(f'data/fire_smoke/train/images/sample_{i:03d}.jpg', img)
        with open(f'data/fire_smoke/train/labels/sample_{i:03d}.txt', 'w') as f:
            # 샘플 라벨 (class_id, x_center, y_center, width, height)
            f.write(f"{np.random.randint(0, 5)} 0.5 0.5 0.3 0.3\n")
        
        # Val set
        if i < 2:
            cv2.imwrite(f'data/fire_smoke/val/images/sample_{i:03d}.jpg', img)
            with open(f'data/fire_smoke/val/labels/sample_{i:03d}.txt', 'w') as f:
                f.write(f"{np.random.randint(0, 5)} 0.5 0.5 0.3 0.3\n")
    
    print("샘플 데이터 생성 완료!")

def visualize_model_architecture():
    """모델 아키텍처 시각화"""
    print("\n=== 모델 아키텍처 ===")
    print("1. Backbone: YOLOv5-nano")
    print("   - Width multiplier: 0.25")
    print("   - Depth multiplier: 0.33")
    print("   - 젯슨 나노에 최적화된 경량 모델")
    print("\n2. Custom Head with Attention")
    print("   - CBAM (Channel & Spatial Attention)")
    print("   - 도메인 특화 feature 학습")
    print("   - 5개 클래스: fire, smoke, wall_clean, wall_dirty, wall_damaged")
    print("\n3. 주요 특징:")
    print("   - 실시간 처리 가능")
    print("   - 높은 정확도")
    print("   - 젯슨 나노 호환")

def monitor_training():
    """학습 모니터링 시뮬레이션"""
    print("\n=== 학습 모니터링 데모 ===")
    
    # 가상의 학습 로그 출력
    epochs = 10
    for epoch in range(1, epochs + 1):
        loss = 5.0 * np.exp(-0.3 * epoch) + np.random.normal(0, 0.1)
        print(f"Epoch [{epoch}/{epochs}] - Loss: {loss:.4f} - mAP: {min(0.1 * epoch + np.random.normal(0, 0.02), 0.95):.3f}")
    
    # 학습 곡선 시각화
    plt.figure(figsize=(12, 4))
    
    # Loss 곡선
    plt.subplot(1, 2, 1)
    x = np.arange(1, epochs + 1)
    train_loss = 5.0 * np.exp(-0.3 * x) + np.random.normal(0, 0.1, epochs)
    val_loss = 5.2 * np.exp(-0.28 * x) + np.random.normal(0, 0.15, epochs)
    plt.plot(x, train_loss, 'b-', label='Train Loss')
    plt.plot(x, val_loss, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    # mAP 곡선
    plt.subplot(1, 2, 2)
    mAP = np.minimum(0.1 * x + np.random.normal(0, 0.02, epochs), 0.95)
    plt.plot(x, mAP, 'g-', label='mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('Validation mAP')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('outputs/training_curves.png')
    print("\n학습 곡선이 'outputs/training_curves.png'에 저장되었습니다.")

def run_inference_demo():
    """추론 데모"""
    print("\n=== 추론 데모 ===")
    
    # 모델 초기화
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FireSmokeDetector('configs/model_config.yaml').to(device)
    model.eval()
    
    print(f"모델이 {device}에 로드되었습니다.")
    print("모델 파라미터 수:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # 테스트 이미지 생성
    test_img = torch.randn(1, 3, 640, 640).to(device)
    
    # 추론
    with torch.no_grad():
        outputs = model(test_img)
    
    print(f"출력 shape: {outputs.shape if isinstance(outputs, torch.Tensor) else [o.shape for o in outputs]}")
    print("\n추론 완료! 실제 환경에서는 실시간 비디오 스트림을 처리합니다.")

def main():
    parser = argparse.ArgumentParser(description='화재/연기 감지 시스템 데모')
    parser.add_argument('--create-data', action='store_true', help='샘플 데이터 생성')
    parser.add_argument('--train-demo', action='store_true', help='학습 모니터링 데모')
    parser.add_argument('--inference-demo', action='store_true', help='추론 데모')
    args = parser.parse_args()
    
    print("="*50)
    print("화재/연기 및 건물 외벽 오염 감지 시스템")
    print("YOLOv5-nano + CBAM Attention")
    print("="*50)
    
    # 출력 디렉토리 생성
    os.makedirs('outputs', exist_ok=True)
    
    # 모델 아키텍처 표시
    visualize_model_architecture()
    
    # 옵션에 따라 실행
    if args.create_data:
        create_sample_data()
    
    if args.train_demo:
        monitor_training()
    
    if args.inference_demo:
        run_inference_demo()
    
    if not any([args.create_data, args.train_demo, args.inference_demo]):
        print("\n사용법:")
        print("  python demo.py --create-data     # 샘플 데이터 생성")
        print("  python demo.py --train-demo      # 학습 모니터링 데모")
        print("  python demo.py --inference-demo  # 추론 데모")
        print("  python demo.py --create-data --train-demo --inference-demo  # 전체 데모")

if __name__ == '__main__':
    main()