#!/usr/bin/env python3
"""
하이퍼파라미터 자동 튜닝 스크립트
다양한 하이퍼파라미터 조합을 자동으로 실험하여 최적 설정을 찾습니다.
"""

import os
import sys
import yaml
import argparse
import itertools
import subprocess
import time
from pathlib import Path
from datetime import datetime
import json

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class HyperparameterTuner:
    def __init__(self, base_config_path, output_dir='experiments'):
        self.base_config_path = base_config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 기본 설정 로드
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        # 실험 결과 저장용
        self.results = []
        
    def create_experiment_config(self, params, experiment_name):
        """실험용 설정 파일 생성"""
        config = self.base_config.copy()
        
        # 파라미터 적용
        for key_path, value in params.items():
            keys = key_path.split('.')
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value
        
        # 실험 이름 설정
        config['experiment_name'] = experiment_name
        
        # WandB 설정 (그룹으로 묶기)
        if 'wandb' in config:
            config['wandb']['group'] = f"tuning_{datetime.now().strftime('%Y%m%d_%H%M')}"
            config['wandb']['name'] = experiment_name
        
        # 설정 파일 저장
        config_path = self.output_dir / f"{experiment_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return config_path
    
    def run_experiment(self, config_path, experiment_name):
        """단일 실험 실행"""
        print(f"\n🚀 실험 시작: {experiment_name}")
        print(f"설정 파일: {config_path}")
        
        # 학습 스크립트 실행
        cmd = [
            'python', 'scripts/train/train.py',
            '--config', str(config_path)
        ]
        
        start_time = time.time()
        
        try:
            # 로그 파일 경로
            log_file = self.output_dir / f"{experiment_name}.log"
            
            with open(log_file, 'w') as f:
                process = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=7200  # 2시간 타임아웃
                )
            
            success = process.returncode == 0
            runtime = time.time() - start_time
            
            if success:
                print(f"✅ 실험 완료: {experiment_name} ({runtime:.1f}초)")
            else:
                print(f"❌ 실험 실패: {experiment_name}")
            
            return success, runtime
            
        except subprocess.TimeoutExpired:
            print(f"⏰ 실험 타임아웃: {experiment_name}")
            return False, time.time() - start_time
        except Exception as e:
            print(f"❌ 실험 오류: {experiment_name} - {e}")
            return False, time.time() - start_time
    
    def extract_results(self, experiment_name):
        """실험 결과 추출"""
        # 체크포인트에서 최고 성능 확인
        checkpoint_dir = Path('outputs/checkpoints')
        best_checkpoint = checkpoint_dir / 'best.pth'
        
        if best_checkpoint.exists():
            # 체크포인트에서 메트릭 추출
            try:
                import torch
                checkpoint = torch.load(best_checkpoint, map_location='cpu')
                best_loss = checkpoint.get('best_loss', float('inf'))
                epoch = checkpoint.get('epoch', 0)
                
                return {
                    'best_loss': float(best_loss),
                    'epochs_trained': int(epoch),
                    'checkpoint_exists': True
                }
            except Exception as e:
                print(f"체크포인트 로드 실패: {e}")
                return {
                    'best_loss': float('inf'),
                    'epochs_trained': 0,
                    'checkpoint_exists': False
                }
        else:
            return {
                'best_loss': float('inf'),
                'epochs_trained': 0,
                'checkpoint_exists': False
            }
    
    def grid_search(self, param_grid, max_experiments=None):
        """그리드 서치 수행"""
        print("🔍 그리드 서치 시작")
        
        # 파라미터 조합 생성
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))
        
        if max_experiments and len(combinations) > max_experiments:
            import random
            combinations = random.sample(combinations, max_experiments)
            print(f"📊 {len(combinations)}개 조합을 랜덤 샘플링")
        
        print(f"📊 총 {len(combinations)}개 실험 예정")
        
        for i, combination in enumerate(combinations, 1):
            # 파라미터 딕셔너리 생성
            params = dict(zip(keys, combination))
            experiment_name = f"exp_{i:03d}_{int(time.time())}"
            
            print(f"\n{'='*50}")
            print(f"실험 {i}/{len(combinations)}: {experiment_name}")
            print(f"파라미터: {params}")
            
            # 실험 설정 생성
            config_path = self.create_experiment_config(params, experiment_name)
            
            # 실험 실행
            success, runtime = self.run_experiment(config_path, experiment_name)
            
            # 결과 추출
            if success:
                metrics = self.extract_results(experiment_name)
            else:
                metrics = {
                    'best_loss': float('inf'),
                    'epochs_trained': 0,
                    'checkpoint_exists': False
                }
            
            # 결과 저장
            result = {
                'experiment_name': experiment_name,
                'parameters': params,
                'success': success,
                'runtime': runtime,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            self.results.append(result)
            
            # 중간 결과 저장
            self.save_results()
            
            print(f"결과: Loss={metrics['best_loss']:.4f}, "
                  f"Epochs={metrics['epochs_trained']}, "
                  f"Runtime={runtime:.1f}s")
    
    def save_results(self):
        """결과를 JSON 파일로 저장"""
        results_file = self.output_dir / 'tuning_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"📊 결과 저장: {results_file}")
    
    def print_summary(self):
        """실험 결과 요약 출력"""
        if not self.results:
            print("❌ 실험 결과가 없습니다.")
            return
        
        print(f"\n{'='*60}")
        print("🏆 하이퍼파라미터 튜닝 결과 요약")
        print(f"{'='*60}")
        
        # 성공한 실험만 필터링
        successful_results = [r for r in self.results if r['success']]
        
        if not successful_results:
            print("❌ 성공한 실험이 없습니다.")
            return
        
        # 최고 성능 실험 찾기
        best_result = min(successful_results, key=lambda x: x['metrics']['best_loss'])
        
        print(f"\n🥇 최고 성능 실험:")
        print(f"   실험명: {best_result['experiment_name']}")
        print(f"   Loss: {best_result['metrics']['best_loss']:.4f}")
        print(f"   파라미터:")
        for key, value in best_result['parameters'].items():
            print(f"     {key}: {value}")
        
        # 상위 3개 실험
        top_3 = sorted(successful_results, key=lambda x: x['metrics']['best_loss'])[:3]
        
        print(f"\n🏆 상위 3개 실험:")
        for i, result in enumerate(top_3, 1):
            print(f"   {i}. {result['experiment_name']} - "
                  f"Loss: {result['metrics']['best_loss']:.4f}")
        
        # 전체 통계
        losses = [r['metrics']['best_loss'] for r in successful_results 
                 if r['metrics']['best_loss'] != float('inf')]
        
        if losses:
            print(f"\n📊 전체 통계:")
            print(f"   성공한 실험: {len(successful_results)}/{len(self.results)}")
            print(f"   평균 Loss: {sum(losses)/len(losses):.4f}")
            print(f"   최고 Loss: {min(losses):.4f}")
            print(f"   최저 Loss: {max(losses):.4f}")
        
        # 최적 설정 파일 생성
        best_config_path = self.output_dir / 'best_config.yaml'
        best_config = self.base_config.copy()
        
        # 최적 파라미터 적용
        for key_path, value in best_result['parameters'].items():
            keys = key_path.split('.')
            current = best_config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value
        
        best_config['experiment_name'] = 'best_tuned_model'
        
        with open(best_config_path, 'w') as f:
            yaml.dump(best_config, f, default_flow_style=False)
        
        print(f"\n💾 최적 설정 저장: {best_config_path}")

def load_tuning_ranges(ranges_file):
    """튜닝 범위 파일 로드"""
    if ranges_file and Path(ranges_file).exists():
        with open(ranges_file, 'r') as f:
            return yaml.safe_load(f)
    else:
        # 기본 튜닝 범위
        return {
            'training.batch_size': [16, 32, 64],
            'training.learning_rate': [0.001, 0.01, 0.02],
            'model.head.reduction_ratio': [8, 16, 32],
            'training.weight_decay': [0.0001, 0.0005, 0.001]
        }

def main():
    parser = argparse.ArgumentParser(description='하이퍼파라미터 자동 튜닝')
    parser.add_argument('--base_config', default='configs/model_config.yaml', 
                       help='기본 설정 파일')
    parser.add_argument('--param_ranges', default=None,
                       help='파라미터 범위 파일 (YAML)')
    parser.add_argument('--experiments', type=int, default=10,
                       help='최대 실험 횟수')
    parser.add_argument('--output_dir', default='experiments',
                       help='실험 결과 저장 디렉토리')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 튜너 초기화
    tuner = HyperparameterTuner(args.base_config, args.output_dir)
    
    # 파라미터 범위 로드
    param_ranges = load_tuning_ranges(args.param_ranges)
    
    print("🎯 하이퍼파라미터 자동 튜닝 시작")
    print(f"기본 설정: {args.base_config}")
    print(f"최대 실험: {args.experiments}회")
    print(f"파라미터 범위: {param_ranges}")
    
    # 그리드 서치 실행
    tuner.grid_search(param_ranges, args.experiments)
    
    # 결과 요약
    tuner.print_summary()
    
    print(f"\n🎉 하이퍼파라미터 튜닝 완료!")
    print(f"📁 결과 저장: {args.output_dir}/")

if __name__ == '__main__':
    main()