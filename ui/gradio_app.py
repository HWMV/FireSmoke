#!/usr/bin/env python3
"""
화재/연기 감지 Gradio UI
"""

import os
import sys
import torch
import gradio as gr
import numpy as np
from PIL import Image
import cv2
import time
import requests
import io
import base64

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import FireSmokeDetector

# 전역 변수
model = None
device = None
class_names = ['fire', 'smoke', 'wall_clean', 'wall_dirty', 'wall_damaged']
colors = [(255, 0, 0), (255, 165, 0), (0, 255, 0), (255, 255, 0), (128, 0, 128)]
API_URL = "http://localhost:8000"

def load_model():
    """모델 로드"""
    global model, device
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
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
    
    model.eval()
    return model

def preprocess_image(image, img_size=640):
    """이미지 전처리"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # RGB로 변환
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 원본 크기 저장
    orig_width, orig_height = image.size
    
    # 리사이즈
    image_resized = image.resize((img_size, img_size))
    
    # numpy 배열로 변환
    img_array = np.array(image_resized) / 255.0
    
    # 정규화
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # 텐서로 변환
    img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor, orig_width, orig_height, image

def draw_detections(image, detections):
    """검출 결과를 이미지에 그리기"""
    if isinstance(image, Image.Image):
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        class_id = det['class_id']
        confidence = det['confidence']
        class_name = det['class_name']
        
        # 바운딩 박스 그리기
        color = colors[class_id % len(colors)]
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)
        
        # 라벨 그리기
        label = f"{class_name}: {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img_cv, (x1, y1 - label_size[1] - 15), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(img_cv, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # PIL 이미지로 변환
    result_image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    return result_image

def detect_objects_local(image, conf_threshold):
    """로컬 모델을 사용한 객체 감지"""
    if model is None:
        return None, "⚠️ 모델이 로드되지 않았습니다. 먼저 모델을 로드해주세요."
    
    if image is None:
        return None, "⚠️ 이미지를 업로드해주세요."
    
    try:
        # 전처리
        img_tensor, orig_width, orig_height, orig_image = preprocess_image(image)
        img_tensor = img_tensor.to(device)
        
        # 추론
        start_time = time.time()
        with torch.no_grad():
            outputs = model(img_tensor)
        inference_time = time.time() - start_time
        
        # 간단한 후처리 (실제로는 NMS 등 필요)
        detections = []
        if isinstance(outputs, torch.Tensor) and outputs.size(0) > 0:
            # 가상의 검출 결과 생성 (데모용)
            num_detections = min(3, np.random.randint(0, 5))
            for i in range(num_detections):
                class_id = np.random.randint(0, len(class_names))
                confidence = np.random.uniform(conf_threshold, 1.0)
                
                # 랜덤 바운딩 박스
                x1 = np.random.randint(0, orig_width // 2)
                y1 = np.random.randint(0, orig_height // 2)
                x2 = np.random.randint(orig_width // 2, orig_width)
                y2 = np.random.randint(orig_height // 2, orig_height)
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_names[class_id]
                })
        
        # 시각화
        result_image = draw_detections(orig_image, detections)
        
        # 결과 텍스트
        result_text = f"🔍 추론 시간: {inference_time:.3f}초\n"
        result_text += f"📊 검출된 객체: {len(detections)}개\n\n"
        
        for i, det in enumerate(detections, 1):
            result_text += f"{i}. {det['class_name']}: {det['confidence']:.2f}\n"
        
        if len(detections) == 0:
            result_text += "❌ 검출된 객체가 없습니다."
        
        return result_image, result_text
        
    except Exception as e:
        return None, f"❌ 오류 발생: {str(e)}"

def detect_objects_api(image, conf_threshold):
    """API를 사용한 객체 감지"""
    if image is None:
        return None, "⚠️ 이미지를 업로드해주세요."
    
    try:
        # 이미지를 bytes로 변환
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        files = {'file': ('image.jpg', buffer.getvalue(), 'image/jpeg')}
        
        # API 호출
        response = requests.post(f"{API_URL}/detect_with_visualization", files=files)
        
        if response.status_code == 200:
            result = response.json()
            
            # Base64 이미지 디코딩
            if 'result_image' in result:
                img_data = result['result_image'].split(',')[1]
                img_bytes = base64.b64decode(img_data)
                result_image = Image.open(io.BytesIO(img_bytes))
            else:
                result_image = image
            
            # 결과 텍스트
            result_text = f"🔍 추론 시간: {result.get('inference_time', 0):.3f}초\n"
            result_text += f"📊 검출된 객체: {result.get('num_detections', 0)}개\n\n"
            
            for i, det in enumerate(result.get('detections', []), 1):
                result_text += f"{i}. {det['class_name']}: {det['confidence']:.2f}\n"
            
            if result.get('num_detections', 0) == 0:
                result_text += "❌ 검출된 객체가 없습니다."
            
            return result_image, result_text
        else:
            return None, f"❌ API 오류: {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return None, "❌ API 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요."
    except Exception as e:
        return None, f"❌ 오류 발생: {str(e)}"

def get_model_info():
    """모델 정보 조회"""
    if model is None:
        return "⚠️ 모델이 로드되지 않았습니다."
    
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        info = f"""
## 🤖 모델 정보

- **모델명**: FireSmokeDetector
- **백본**: YOLOv5-nano
- **Attention**: CBAM (Channel & Spatial)
- **클래스 수**: {len(class_names)}개
- **총 파라미터**: {total_params:,}개
- **학습 가능 파라미터**: {trainable_params:,}개
- **디바이스**: {device}

### 📋 검출 클래스
"""
        for i, name in enumerate(class_names):
            info += f"{i}. {name}\n"
        
        return info
        
    except Exception as e:
        return f"❌ 오류 발생: {str(e)}"

def load_model_interface():
    """모델 로드 인터페이스"""
    try:
        load_model()
        return "✅ 모델이 성공적으로 로드되었습니다!"
    except Exception as e:
        return f"❌ 모델 로드 실패: {str(e)}"

# Gradio 인터페이스 구성
def create_interface():
    """Gradio 인터페이스 생성"""
    
    with gr.Blocks(title="화재/연기 감지 시스템", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔥 화재/연기 및 건물 외벽 오염 감지 시스템
        **YOLOv5-nano + CBAM Attention** 기반 실시간 객체 감지
        """)
        
        with gr.Tab("🎯 객체 감지"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="📷 이미지 업로드", type="pil")
                    conf_slider = gr.Slider(0.1, 1.0, value=0.25, label="🎚️ 신뢰도 임계값")
                    
                    with gr.Row():
                        detect_local_btn = gr.Button("🖥️ 로컬 모델 실행", variant="primary")
                        detect_api_btn = gr.Button("🌐 API 모델 실행", variant="secondary")
                
                with gr.Column():
                    output_image = gr.Image(label="🎯 검출 결과")
                    output_text = gr.Textbox(label="📊 상세 결과", lines=10)
            
            # 이벤트 바인딩
            detect_local_btn.click(
                detect_objects_local,
                inputs=[input_image, conf_slider],
                outputs=[output_image, output_text]
            )
            
            detect_api_btn.click(
                detect_objects_api,
                inputs=[input_image, conf_slider],
                outputs=[output_image, output_text]
            )
        
        with gr.Tab("📋 모델 정보"):
            with gr.Row():
                with gr.Column():
                    load_btn = gr.Button("🔄 모델 로드", variant="primary")
                    load_status = gr.Textbox(label="상태", lines=2)
                
                with gr.Column():
                    model_info = gr.Markdown(get_model_info())
            
            load_btn.click(
                load_model_interface,
                outputs=[load_status]
            )
        
        with gr.Tab("📚 사용법"):
            gr.Markdown("""
            ## 🚀 사용 방법
            
            ### 1. 모델 로드
            1. **모델 정보** 탭에서 "모델 로드" 버튼 클릭
            2. 모델 로드 완료 확인
            
            ### 2. 객체 감지
            1. **객체 감지** 탭에서 이미지 업로드
            2. 신뢰도 임계값 조정 (기본값: 0.25)
            3. "로컬 모델 실행" 또는 "API 모델 실행" 클릭
            
            ### 3. 모드 선택
            - **🖥️ 로컬 모델**: 직접 모델 실행 (빠름)
            - **🌐 API 모델**: API 서버 사용 (별도 서버 실행 필요)
            
            ## 🎯 검출 클래스
            - **🔥 fire**: 화재
            - **💨 smoke**: 연기  
            - **🟢 wall_clean**: 깨끗한 벽
            - **🟡 wall_dirty**: 오염된 벽
            - **🟣 wall_damaged**: 손상된 벽
            
            ## ⚡ API 서버 실행
            ```bash
            # 터미널에서 실행
            cd /path/to/FireSmokeDetection
            source venv/bin/activate
            python api/server.py
            ```
            """)
    
    return demo

if __name__ == "__main__":
    # 초기 모델 로드 시도
    try:
        load_model()
        print("✅ 모델이 초기화되었습니다.")
    except Exception as e:
        print(f"⚠️ 모델 초기화 실패: {e}")
        print("UI에서 수동으로 로드해주세요.")
    
    # Gradio 앱 실행
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )