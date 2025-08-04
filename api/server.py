#!/usr/bin/env python3
"""
화재/연기 감지 API 서버
FastAPI 기반 RESTful API
"""

import os
import sys
import io
import torch
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Dict, Any
import base64
import time

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import FireSmokeDetector
from models.utils import FireSmokeDataset

app = FastAPI(
    title="Fire & Smoke Detection API",
    description="YOLOv5-nano + CBAM Attention 기반 화재/연기 및 건물 외벽 오염 감지 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
model = None
device = None
class_names = ['fire', 'smoke', 'wall_clean', 'wall_dirty', 'wall_damaged']
colors = [(255, 0, 0), (255, 165, 0), (0, 255, 0), (255, 255, 0), (128, 0, 128)]

def load_model():
    """모델 로드"""
    global model, device
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device} (CUDA)")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: {device} (Apple Silicon MPS)")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device} (CPU)")
    
    # 모델 초기화
    model = FireSmokeDetector('configs/model_config.yaml').to(device)
    
    # 체크포인트 로드 (있는 경우)
    checkpoint_path = 'outputs/checkpoints/best.pth'
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            print("Using randomly initialized model")
    else:
        print("No checkpoint found, using randomly initialized model")
    
    model.eval()
    return model

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    load_model()
    print("Fire & Smoke Detection API server started!")

def preprocess_image(image: Image.Image, img_size: int = 640):
    """이미지 전처리"""
    # RGB로 변환
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 원본 크기 저장
    orig_width, orig_height = image.size
    
    # 리사이즈
    image = image.resize((img_size, img_size))
    
    # numpy 배열로 변환
    img_array = np.array(image) / 255.0
    
    # 정규화 (ImageNet 평균/표준편차)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # 텐서로 변환 (CHW 형식)
    img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor, orig_width, orig_height

def postprocess_detections(outputs, conf_threshold=0.25, iou_threshold=0.45, orig_width=640, orig_height=640):
    """검출 결과 후처리"""
    detections = []
    
    # 모델 출력이 리스트인 경우 (훈련 모드)
    if isinstance(outputs, list):
        # 추론 모드로 변환 필요 - 실제로는 모델을 eval 모드로 설정해야 함
        return detections
    
    # NMS 적용하지 않은 원시 출력 처리
    for detection in outputs[0]:  # 배치 크기 1 가정
        conf = detection[4].item()
        
        if conf > conf_threshold:
            # 바운딩 박스 좌표 (중심점 기준)
            cx, cy, w, h = detection[:4].tolist()
            
            # 원본 이미지 크기로 변환
            cx = cx * orig_width / 640
            cy = cy * orig_height / 640
            w = w * orig_width / 640
            h = h * orig_height / 640
            
            # 좌상단 좌표로 변환
            x1 = max(0, cx - w/2)
            y1 = max(0, cy - h/2)
            x2 = min(orig_width, cx + w/2)
            y2 = min(orig_height, cy + h/2)
            
            # 클래스 예측
            class_scores = detection[5:].tolist()
            class_id = np.argmax(class_scores)
            class_conf = class_scores[class_id]
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': conf,
                'class_id': class_id,
                'class_name': class_names[class_id],
                'class_confidence': class_conf
            })
    
    return detections

def draw_detections(image: Image.Image, detections: List[Dict]):
    """검출 결과를 이미지에 그리기"""
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        class_id = det['class_id']
        confidence = det['confidence']
        class_name = det['class_name']
        
        # 바운딩 박스 그리기
        color = colors[class_id % len(colors)]
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
        
        # 라벨 그리기
        label = f"{class_name}: {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_cv, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(img_cv, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # PIL 이미지로 변환
    result_image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    return result_image

@app.get("/")
async def root():
    """API 상태 확인"""
    return {
        "message": "Fire & Smoke Detection API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    }

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_ready": model is not None
    }

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """이미지에서 객체 감지"""
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # 이미지 로드
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # 전처리
        img_tensor, orig_width, orig_height = preprocess_image(image)
        img_tensor = img_tensor.to(device)
        
        # 추론
        start_time = time.time()
        with torch.no_grad():
            outputs = model(img_tensor)
        inference_time = time.time() - start_time
        
        # 후처리
        detections = postprocess_detections(outputs, orig_width=orig_width, orig_height=orig_height)
        
        # 결과 반환
        return {
            "success": True,
            "inference_time": round(inference_time, 4),
            "detections": detections,
            "image_size": [orig_width, orig_height],
            "num_detections": len(detections)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/detect_with_visualization")
async def detect_with_visualization(file: UploadFile = File(...)):
    """이미지에서 객체 감지 + 시각화"""
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # 이미지 로드
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # 전처리
        img_tensor, orig_width, orig_height = preprocess_image(image)
        img_tensor = img_tensor.to(device)
        
        # 추론
        start_time = time.time()
        with torch.no_grad():
            outputs = model(img_tensor)
        inference_time = time.time() - start_time
        
        # 후처리
        detections = postprocess_detections(outputs, orig_width=orig_width, orig_height=orig_height)
        
        # 시각화
        result_image = draw_detections(image, detections)
        
        # 이미지를 base64로 인코딩
        buffer = io.BytesIO()
        result_image.save(buffer, format='JPEG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # 결과 반환
        return {
            "success": True,
            "inference_time": round(inference_time, 4),
            "detections": detections,
            "image_size": [orig_width, orig_height],
            "num_detections": len(detections),
            "result_image": f"data:image/jpeg;base64,{img_base64}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """모델 정보 조회"""
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # 모델 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_name": "FireSmokeDetector",
        "backbone": "YOLOv5-nano",
        "attention": "CBAM",
        "classes": class_names,
        "num_classes": len(class_names),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "device": str(device)
    }

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )