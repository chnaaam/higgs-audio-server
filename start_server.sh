#!/bin/bash

# Higgs Audio v2 服务启动脚本

echo "=== Higgs Audio v2 REST API Server ==="
echo "Starting server with default configuration..."

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed"
    exit 1
fi

# 检查依赖是否安装
if ! python3 -c "import fastapi, uvicorn" &> /dev/null; then
    echo "Installing required dependencies..."
    pip install -r requirements.txt
fi

# 设置默认参数
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}
MODEL_PATH=${MODEL_PATH:-"bosonai/higgs-audio-v2-generation-3B-base"}
AUDIO_TOKENIZER_PATH=${AUDIO_TOKENIZER_PATH:-"bosonai/higgs-audio-v2-tokenizer"}
DEVICE=${DEVICE:-"auto"}

echo "Configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Model: $MODEL_PATH"
echo "  Audio Tokenizer: $AUDIO_TOKENIZER_PATH"
echo "  Device: $DEVICE"
echo ""

# 启动服务
python3 main.py \
    --host "$HOST" \
    --port "$PORT" \
    --model-path "$MODEL_PATH" \
    --audio-tokenizer-path "$AUDIO_TOKENIZER_PATH" \
    ${DEVICE:+--device "$DEVICE"}