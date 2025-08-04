#!/bin/bash
# 화재/연기 감지 시스템 실행 스크립트

echo "🔥 화재/연기 감지 시스템 데모 실행"
echo "================================="

# 가상환경 활성화
if [ -d "venv" ]; then
    echo "📦 가상환경 활성화..."
    source venv/bin/activate
else
    echo "⚠️ 가상환경이 없습니다. python -m venv venv 실행 후 다시 시도하세요."
    exit 1
fi

# 필요한 패키지 설치
echo "📋 패키지 설치 확인..."
pip install -q fastapi uvicorn python-multipart gradio requests

# 출력 디렉토리 생성
mkdir -p outputs/{checkpoints,logs,visualizations}

echo ""
echo "🚀 실행 옵션을 선택하세요:"
echo "1) 기본 데모 실행"
echo "2) API 서버 실행 (포트 8000)"
echo "3) Gradio UI 실행 (포트 7860)"
echo "4) 학습 스크립트 실행"
echo "5) 전체 시스템 실행 (API + UI)"

read -p "선택 (1-5): " choice

case $choice in
    1)
        echo "🎯 기본 데모 실행..."
        python demo.py --create-data --train-demo --inference-demo
        ;;
    2)
        echo "🌐 API 서버 실행 중... (Ctrl+C로 종료)"
        echo "📍 API 문서: http://localhost:8000/docs"
        python api/server.py
        ;;
    3)
        echo "🖥️ Gradio UI 실행 중... (Ctrl+C로 종료)"
        echo "📍 UI 주소: http://localhost:7860"
        python ui/gradio_app.py
        ;;
    4)
        echo "📚 학습 스크립트 실행..."
        python scripts/train/train.py --config configs/model_config.yaml
        ;;
    5)
        echo "🎪 전체 시스템 실행..."
        echo "API 서버와 Gradio UI를 동시에 실행합니다."
        echo "터미널 2개를 열어서 각각 실행하세요:"
        echo ""
        echo "터미널 1: python api/server.py"
        echo "터미널 2: python ui/gradio_app.py"
        echo ""
        echo "또는 백그라운드 실행:"
        python api/server.py &
        API_PID=$!
        sleep 3
        python ui/gradio_app.py &
        UI_PID=$!
        
        echo "🌐 API 서버: http://localhost:8000"
        echo "🖥️ Gradio UI: http://localhost:7860"
        echo ""
        echo "종료하려면 Ctrl+C를 누르세요..."
        
        # 종료 신호 처리
        trap "echo '🛑 시스템 종료 중...'; kill $API_PID $UI_PID 2>/dev/null; exit" INT
        wait
        ;;
    *)
        echo "❌ 잘못된 선택입니다."
        exit 1
        ;;
esac