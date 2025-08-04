#!/usr/bin/env python3
"""
빠른 하이퍼파라미터 튜닝 스크립트
Mac M3 Max에 최적화된 설정으로 빠른 실험을 수행합니다.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_quick_experiments():
    """Mac M3 Max에 최적화된 빠른 실험 설정들"""
    
    base_config_path = 'configs/model_config.yaml'
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # 실험 설정들
    experiments = [
        {
            'name': 'baseline_m3max',
            'description': 'Mac M3 Max 기본 설정',
            'params': {
                'training.batch_size': 32,
                'training.learning_rate': 0.01,
                'training.epochs': 50,
                'model.input_size': 640,
                'experiment_name': 'baseline_m3max'
            }
        },
        {
            'name': 'large_batch_m3max',
            'description': '큰 배치 크기로 안정적 학습',
            'params': {
                'training.batch_size': 64,
                'training.learning_rate': 0.02,
                'training.epochs': 100,
                'model.input_size': 640,
                'experiment_name': 'large_batch_m3max'
            }
        },
        {
            'name': 'fast_experiment',
            'description': '빠른 실험용 작은 해상도',
            'params': {
                'training.batch_size': 64,
                'training.learning_rate': 0.02,
                'training.epochs': 30,
                'model.input_size': 416,
                'experiment_name': 'fast_experiment'
            }
        },
        {
            'name': 'high_resolution',
            'description': '고해상도 실험',
            'params': {
                'training.batch_size': 16,
                'training.learning_rate': 0.01,
                'training.epochs': 100,
                'model.input_size': 832,
                'experiment_name': 'high_resolution'
            }
        },
        {
            'name': 'attention_tuned',
            'description': 'Attention 모듈 최적화',
            'params': {
                'training.batch_size': 32,
                'training.learning_rate': 0.015,
                'training.epochs': 80,
                'model.head.reduction_ratio': 8,
                'model.head.kernel_size': 11,
                'experiment_name': 'attention_tuned'
            }
        }
    ]
    
    # 실험 설정 파일들 생성
    configs_dir = Path('configs/experiments')
    configs_dir.mkdir(exist_ok=True)
    
    print("🚀 Mac M3 Max 최적화 실험 설정 생성")
    print("=" * 50)
    
    for exp in experiments:
        # 설정 복사 및 수정
        config = base_config.copy()
        
        # 파라미터 적용
        for key_path, value in exp['params'].items():
            keys = key_path.split('.')
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value
        
        # WandB 그룹 설정
        if 'wandb' in config:
            config['wandb']['group'] = 'm3max_optimization'
            config['wandb']['name'] = exp['name']
        
        # 설정 파일 저장
        config_path = configs_dir / f"{exp['name']}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ {exp['name']}: {exp['description']}")
        print(f"   설정 파일: {config_path}")
        print(f"   주요 파라미터: {exp['params']}")
        print()
    
    return experiments, configs_dir

def generate_run_script(experiments, configs_dir):
    """실험 실행 스크립트 생성"""
    
    script_content = f"""#!/bin/bash
# Mac M3 Max 최적화 실험 자동 실행 스크립트

echo "🔥 Mac M3 Max 최적화 실험 시작"
echo "================================="

# 가상환경 활성화
source venv/bin/activate

# 각 실험 순차 실행
"""
    
    for i, exp in enumerate(experiments, 1):
        script_content += f"""
echo ""
echo "🚀 실험 {i}/{len(experiments)}: {exp['name']}"
echo "설명: {exp['description']}"
echo "시작 시간: $(date)"

python scripts/train/train.py \\
  --config {configs_dir}/{exp['name']}.yaml

echo "완료 시간: $(date)"
echo "체크포인트: outputs/checkpoints/"
echo "로그: outputs/logs/"
"""
    
    script_content += """
echo ""
echo "🎉 모든 실험 완료!"
echo "결과 확인:"
echo "  - TensorBoard: tensorboard --logdir outputs/logs"
echo "  - WandB: https://wandb.ai/hyunwoo220/FireSmoke"
"""
    
    # 스크립트 파일 저장
    script_path = Path('run_m3max_experiments.sh')
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # 실행 권한 부여
    os.chmod(script_path, 0o755)
    
    return script_path

def main():
    parser = argparse.ArgumentParser(description='Mac M3 Max 최적화 실험 설정 생성')
    parser.add_argument('--create-configs', action='store_true', 
                       help='실험 설정 파일들만 생성')
    parser.add_argument('--create-script', action='store_true',
                       help='실행 스크립트만 생성')
    
    args = parser.parse_args()
    
    # 실험 설정 생성
    experiments, configs_dir = create_quick_experiments()
    
    if args.create_configs:
        print("✅ 실험 설정 파일 생성 완료")
        return
    
    # 실행 스크립트 생성
    script_path = generate_run_script(experiments, configs_dir)
    
    if args.create_script:
        print(f"✅ 실행 스크립트 생성: {script_path}")
        return
    
    print("🎯 Mac M3 Max 최적화 실험 준비 완료!")
    print(f"📁 설정 파일들: {configs_dir}/")
    print(f"🚀 실행 스크립트: {script_path}")
    print("")
    print("실행 방법:")
    print("1. 전체 자동 실행:")
    print(f"   ./{script_path}")
    print("")
    print("2. 개별 실험 실행:")
    for exp in experiments:
        print(f"   python scripts/train/train.py --config {configs_dir}/{exp['name']}.yaml")
    print("")
    print("3. 하이퍼파라미터 자동 튜닝:")
    print("   python scripts/hyperparameter_tuning.py --experiments 10")

if __name__ == '__main__':
    main()