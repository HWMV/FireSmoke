#!/usr/bin/env python3
"""
AI Hub 화재 감지 데이터셋을 YOLO 형식으로 변환하는 스크립트
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

class AIHubToYOLOConverter:
    def __init__(self, input_json, input_images, output_dir):
        self.input_json = input_json
        self.input_images = Path(input_images)
        self.output_dir = Path(output_dir)
        
        # 클래스 매핑 (AI Hub → 프로젝트)
        self.class_mapping = {
            1: 0,  # fire → fire
            2: 1,  # smoke → smoke
            3: 2,  # wall_clean → wall_clean
            4: 3,  # wall_dirty → wall_dirty
            5: 4,  # wall_damaged → wall_damaged
            6: 2,  # 기타 벽 관련 → wall_clean
            7: 3,  # 기타 오염 → wall_dirty
            8: 4,  # 기타 손상 → wall_damaged
            9: 0,  # 기타 화재 → fire
            10: 1, # 기타 연기 → smoke
        }
        
        self.class_names = ['fire', 'smoke', 'wall_clean', 'wall_dirty', 'wall_damaged']
        
        # 출력 디렉토리 생성
        self.create_output_dirs()
    
    def create_output_dirs(self):
        """출력 디렉토리 구조 생성"""
        dirs = [
            'train/images', 'train/labels',
            'val/images', 'val/labels', 
            'test/images', 'test/labels'
        ]
        
        for dir_name in dirs:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    def load_annotations(self):
        """AI Hub JSON 어노테이션 로드"""
        print("Loading AI Hub annotations...")
        with open(self.input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def convert_bbox_format(self, bbox, img_width, img_height):
        """COCO bbox를 YOLO 형식으로 변환"""
        x, y, w, h = bbox
        
        # YOLO 형식: center_x, center_y, width, height (모두 정규화)
        center_x = (x + w / 2) / img_width
        center_y = (y + h / 2) / img_height
        norm_w = w / img_width
        norm_h = h / img_height
        
        return [center_x, center_y, norm_w, norm_h]
    
    def split_dataset(self, image_ids, train_ratio=0.8, val_ratio=0.1):
        """데이터셋을 train/val/test로 분할"""
        np.random.shuffle(image_ids)
        
        total = len(image_ids)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        
        train_ids = image_ids[:train_end]
        val_ids = image_ids[train_end:val_end]
        test_ids = image_ids[val_end:]
        
        return train_ids, val_ids, test_ids
    
    def convert_single_image(self, image_info, annotations, split='train'):
        """단일 이미지와 어노테이션 변환"""
        image_id = image_info['id']
        filename = image_info['file_name']
        img_width = image_info['width']
        img_height = image_info['height']
        
        # 이미지 복사
        src_img_path = self.input_images / filename
        dst_img_path = self.output_dir / split / 'images' / filename
        
        if src_img_path.exists():
            shutil.copy2(src_img_path, dst_img_path)
        else:
            print(f"Warning: Image not found: {src_img_path}")
            return False
        
        # 라벨 파일 생성
        label_filename = filename.rsplit('.', 1)[0] + '.txt'
        label_path = self.output_dir / split / 'labels' / label_filename
        
        yolo_annotations = []
        for ann in annotations:
            if ann['image_id'] == image_id:
                category_id = ann['category_id']
                
                # 클래스 매핑
                if category_id in self.class_mapping:
                    mapped_class = self.class_mapping[category_id]
                    
                    # 바운딩 박스 변환
                    bbox = ann['bbox']
                    yolo_bbox = self.convert_bbox_format(bbox, img_width, img_height)
                    
                    # YOLO 형식으로 저장
                    yolo_line = f"{mapped_class} {' '.join(map(str, yolo_bbox))}"
                    yolo_annotations.append(yolo_line)
        
        # 라벨 파일 저장
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
        
        return True
    
    def convert(self):
        """전체 변환 프로세스 실행"""
        # 어노테이션 로드
        data = self.load_annotations()
        
        images = data['images']
        annotations = data['annotations']
        
        print(f"Total images: {len(images)}")
        print(f"Total annotations: {len(annotations)}")
        
        # 이미지 ID 리스트 생성
        image_ids = [img['id'] for img in images]
        
        # 데이터셋 분할
        train_ids, val_ids, test_ids = self.split_dataset(image_ids)
        
        print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
        
        # 이미지별 변환
        splits = {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        }
        
        for split, ids in splits.items():
            print(f"\nConverting {split} set...")
            success_count = 0
            
            for img_info in tqdm(images):
                if img_info['id'] in ids:
                    if self.convert_single_image(img_info, annotations, split):
                        success_count += 1
            
            print(f"{split} conversion completed: {success_count}/{len(ids)} images")
        
        # 데이터셋 설정 파일 생성
        self.create_dataset_yaml()
        
        print("\nConversion completed successfully!")
    
    def create_dataset_yaml(self):
        """데이터셋 설정 YAML 파일 생성"""
        yaml_content = f"""# AI Hub Fire Detection Dataset
path: {self.output_dir.absolute()}
train: train/images
val: val/images
test: test/images

# Classes
nc: {len(self.class_names)}
names: {self.class_names}
"""
        
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"Dataset YAML created: {yaml_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert AI Hub dataset to YOLO format')
    parser.add_argument('--input_json', required=True, help='Path to AI Hub annotations JSON')
    parser.add_argument('--input_images', required=True, help='Path to AI Hub images directory')
    parser.add_argument('--output_dir', required=True, help='Output directory for YOLO dataset')
    parser.add_argument('--train_ratio', default=0.8, type=float, help='Training set ratio')
    parser.add_argument('--val_ratio', default=0.1, type=float, help='Validation set ratio')
    
    args = parser.parse_args()
    
    # 변환기 초기화 및 실행
    converter = AIHubToYOLOConverter(
        args.input_json,
        args.input_images, 
        args.output_dir
    )
    
    converter.convert()

if __name__ == '__main__':
    main()