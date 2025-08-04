#!/usr/bin/env python3
"""
YOLO 형식 데이터셋 검증 및 시각화 스크립트
"""

import os
import argparse
import random
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import Counter

class DatasetValidator:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.class_names = ['fire', 'smoke', 'wall_clean', 'wall_dirty', 'wall_damaged']
        self.colors = ['red', 'orange', 'green', 'yellow', 'purple']
        
    def validate_structure(self):
        """데이터셋 폴더 구조 검증"""
        print("=== Dataset Structure Validation ===")
        
        required_dirs = [
            'train/images', 'train/labels',
            'val/images', 'val/labels',
            'test/images', 'test/labels'
        ]
        
        all_valid = True
        for dir_name in required_dirs:
            dir_path = self.data_dir / dir_name
            if dir_path.exists():
                count = len(list(dir_path.glob('*')))
                print(f"✓ {dir_name}: {count} files")
            else:
                print(f"✗ {dir_name}: Not found")
                all_valid = False
        
        return all_valid
    
    def validate_labels(self):
        """라벨 파일 형식 검증"""
        print("\n=== Label Format Validation ===")
        
        issues = []
        class_counts = Counter()
        
        for split in ['train', 'val', 'test']:
            labels_dir = self.data_dir / split / 'labels'
            if not labels_dir.exists():
                continue
                
            label_files = list(labels_dir.glob('*.txt'))
            print(f"\n{split.upper()} set: {len(label_files)} label files")
            
            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    for line_num, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:
                            continue
                            
                        parts = line.split()
                        if len(parts) != 5:
                            issues.append(f"{label_file}:{line_num} - Invalid format")
                            continue
                        
                        try:
                            class_id = int(parts[0])
                            bbox = [float(x) for x in parts[1:]]
                            
                            # 클래스 ID 검증
                            if not (0 <= class_id < len(self.class_names)):
                                issues.append(f"{label_file}:{line_num} - Invalid class ID: {class_id}")
                            else:
                                class_counts[class_id] += 1
                            
                            # 바운딩 박스 좌표 검증
                            if not all(0 <= coord <= 1 for coord in bbox):
                                issues.append(f"{label_file}:{line_num} - Invalid bbox coordinates")
                                
                        except ValueError:
                            issues.append(f"{label_file}:{line_num} - Invalid number format")
                            
                except Exception as e:
                    issues.append(f"{label_file} - Error reading file: {e}")
        
        # 클래스 분포 출력
        print("\nClass Distribution:")
        for class_id, count in sorted(class_counts.items()):
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Unknown({class_id})"
            print(f"  {class_name}: {count}")
        
        # 이슈 출력
        if issues:
            print(f"\nFound {len(issues)} issues:")
            for issue in issues[:10]:  # 처음 10개만 출력
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more issues")
        else:
            print("\n✓ All labels are valid!")
        
        return len(issues) == 0, class_counts
    
    def check_image_label_pairs(self):
        """이미지-라벨 쌍 매칭 검증"""
        print("\n=== Image-Label Pair Validation ===")
        
        issues = []
        
        for split in ['train', 'val', 'test']:
            images_dir = self.data_dir / split / 'images'
            labels_dir = self.data_dir / split / 'labels'
            
            if not (images_dir.exists() and labels_dir.exists()):
                continue
            
            image_files = set(f.stem for f in images_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png'])
            label_files = set(f.stem for f in labels_dir.glob('*.txt'))
            
            # 매칭되지 않는 파일들 찾기
            images_without_labels = image_files - label_files
            labels_without_images = label_files - image_files
            
            print(f"\n{split.upper()} set:")
            print(f"  Images: {len(image_files)}")
            print(f"  Labels: {len(label_files)}")
            print(f"  Matched pairs: {len(image_files & label_files)}")
            
            if images_without_labels:
                print(f"  Images without labels: {len(images_without_labels)}")
                issues.extend([f"{split}/images/{name}" for name in list(images_without_labels)[:5]])
            
            if labels_without_images:
                print(f"  Labels without images: {len(labels_without_images)}")
                issues.extend([f"{split}/labels/{name}.txt" for name in list(labels_without_images)[:5]])
        
        if issues:
            print(f"\nFound {len(issues)} mismatched files (showing first few):")
            for issue in issues:
                print(f"  - {issue}")
        
        return len(issues) == 0
    
    def visualize_samples(self, num_samples=10, split='train'):
        """샘플 이미지와 어노테이션 시각화"""
        print(f"\n=== Visualizing {num_samples} samples from {split} set ===")
        
        images_dir = self.data_dir / split / 'images'
        labels_dir = self.data_dir / split / 'labels'
        
        if not (images_dir.exists() and labels_dir.exists()):
            print(f"❌ {split} directories not found")
            return
        
        # 이미지 파일 목록 가져오기
        image_files = list(images_dir.glob('*'))
        image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        if len(image_files) == 0:
            print("❌ No image files found")
            return
        
        # 랜덤 샘플 선택
        sample_files = random.sample(image_files, min(num_samples, len(image_files)))
        
        # 시각화
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for idx, img_file in enumerate(sample_files):
            if idx >= 10:
                break
                
            # 이미지 로드
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            # 라벨 로드
            label_file = labels_dir / f"{img_file.stem}.txt"
            
            ax = axes[idx]
            ax.imshow(img)
            ax.set_title(f"{img_file.name}", fontsize=8)
            ax.axis('off')
            
            if label_file.exists():
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        
                        # YOLO → 픽셀 좌표 변환
                        x1 = (cx - bw/2) * w
                        y1 = (cy - bh/2) * h
                        box_w = bw * w
                        box_h = bh * h
                        
                        # 바운딩 박스 그리기
                        color = self.colors[class_id % len(self.colors)]
                        rect = patches.Rectangle((x1, y1), box_w, box_h, 
                                               linewidth=2, edgecolor=color, facecolor='none')
                        ax.add_patch(rect)
                        
                        # 클래스 라벨 추가
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class{class_id}"
                        ax.text(x1, y1-5, class_name, fontsize=8, color=color, 
                               bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8))
                        
                    except (ValueError, IndexError):
                        continue
        
        # 사용하지 않는 subplot 숨기기
        for idx in range(len(sample_files), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # 저장
        output_path = self.data_dir / f"sample_visualization_{split}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved: {output_path}")
        
        plt.show()
    
    def generate_report(self):
        """전체 검증 보고서 생성"""
        print("=" * 60)
        print("DATASET VALIDATION REPORT")
        print("=" * 60)
        
        # 구조 검증
        structure_valid = self.validate_structure()
        
        # 라벨 검증
        labels_valid, class_counts = self.validate_labels()
        
        # 이미지-라벨 쌍 검증
        pairs_valid = self.check_image_label_pairs()
        
        # 종합 결과
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Dataset structure: {'✓ PASS' if structure_valid else '❌ FAIL'}")
        print(f"Label format: {'✓ PASS' if labels_valid else '❌ FAIL'}")
        print(f"Image-label pairs: {'✓ PASS' if pairs_valid else '❌ FAIL'}")
        
        overall_valid = structure_valid and labels_valid and pairs_valid
        print(f"\nOverall: {'✓ DATASET IS VALID' if overall_valid else '❌ DATASET HAS ISSUES'}")
        
        return overall_valid

def main():
    parser = argparse.ArgumentParser(description='Validate YOLO dataset')
    parser.add_argument('--data_dir', required=True, help='Path to YOLO dataset directory')
    parser.add_argument('--check_labels', action='store_true', help='Validate label format')
    parser.add_argument('--visualize_samples', type=int, default=0, help='Number of samples to visualize')
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test'], help='Split to visualize')
    
    args = parser.parse_args()
    
    validator = DatasetValidator(args.data_dir)
    
    if args.check_labels or args.visualize_samples == 0:
        validator.generate_report()
    
    if args.visualize_samples > 0:
        validator.visualize_samples(args.visualize_samples, args.split)

if __name__ == '__main__':
    main()