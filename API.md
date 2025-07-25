# Higgs Audio v2 REST API 文档

## 概述
Higgs Audio v2 REST API 提供强大的文本转语音服务，支持基础TTS生成、零样本语音克隆、多说话人对话生成等功能。

## 启动服务

### 基础启动
```bash
python main.py --host 0.0.0.0 --port 8000
```

### 自定义配置
```bash
python main.py \
  --host 0.0.0.0 \
  --port 8000 \
  --model-path "bosonai/higgs-audio-v2-generation-3B-base" \
  --audio-tokenizer-path "bosonai/higgs-audio-v2-tokenizer" \
  --device cuda
```

## API 接口

### 1. 健康检查
**GET** `/health`

检查服务状态和模型加载情况。

**响应示例：**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "voices_available": 15
}
```

### 2. 获取可用声音列表
**GET** `/voices`

获取所有可用的参考声音列表。

**响应示例：**
```json
[
  {
    "name": "belinda",
    "description": "A warm, friendly female voice"
  },
  {
    "name": "chadwick", 
    "description": "A deep, authoritative male voice"
  }
]
```

### 3. 基础文本转语音
**POST** `/generate`

基础的文本转语音生成，模型自动选择合适的声音。

**请求参数：**
```json
{
  "text": "Hello, this is a sample text to convert to speech.",
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 50,
  "max_new_tokens": 1024,
  "seed": 42,
  "scene_prompt": "Audio is recorded from a quiet room."
}
```

**响应示例：**
```json
{
  "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhY...",
  "text": "Hello, this is a sample text to convert to speech.",
  "sampling_rate": 24000,
  "duration": 3.5,
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 128,
    "total_tokens": 173
  }
}
```

### 4. 零样本语音克隆
**POST** `/generate/voice-clone`

使用参考语音进行零样本语音克隆。

**请求参数：**
```json
{
  "text": "I want to clone this voice to say something new.",
  "reference_voice": "belinda",
  "temperature": 0.3,
  "top_p": 0.95,
  "max_new_tokens": 1024
}
```

或者使用上传的音频：
```json
{
  "text": "I want to clone this voice to say something new.",
  "reference_audio": "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhY...",
  "temperature": 0.3
}
```

### 5. 多说话人对话生成
**POST** `/generate/multi-speaker`

生成多说话人对话，支持指定不同说话人的声音。

**请求参数：**
```json
{
  "text": "[SPEAKER0] Hello there! How are you today?\n[SPEAKER1] I'm doing great, thanks for asking!",
  "reference_voices": ["belinda", "chadwick"],
  "temperature": 0.7,
  "max_new_tokens": 2048,
  "scene_prompt": "A friendly conversation in a coffee shop."
}
```

### 6. 上传参考声音
**POST** `/upload-voice`

上传新的参考声音文件。

**Form Data：**
- `file`: 音频文件 (WAV, MP3, FLAC)
- `name`: 声音名称 (可选，默认使用文件名)
- `description`: 声音描述 (可选)

## 请求参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| text | string | - | 要转换的文本内容 |
| temperature | float | 0.7 | 生成随机性，0.0-2.0 |
| top_p | float | 0.95 | 核采样参数，0.0-1.0 |
| top_k | int | 50 | Top-k采样参数 |
| max_new_tokens | int | 1024 | 最大生成token数 |
| seed | int | null | 随机种子，用于可复现生成 |
| scene_prompt | string | null | 场景描述，提供语音生成的上下文 |

## 多说话人文本格式

对于多说话人对话，请使用以下格式：

```
[SPEAKER0] 第一个说话人的文本内容
[SPEAKER1] 第二个说话人的文本内容
[SPEAKER0] 第一个说话人的回应
```

## 音频格式

- 输入音频：支持 WAV, MP3, FLAC 格式
- 输出音频：WAV 格式，24kHz 采样率
- 编码：Base64 编码的音频数据

## 错误处理

API 使用标准 HTTP 状态码：

- `200`: 请求成功
- `400`: 请求参数错误
- `500`: 服务器内部错误
- `503`: 服务不可用（模型未加载）

**错误响应示例：**
```json
{
  "detail": "Reference voice 'unknown_voice' not found. Available: ['belinda', 'chadwick', 'mabel']"
}
```

## 使用示例

### Python 客户端示例

```python
import requests
import base64

# 基础TTS
response = requests.post('http://localhost:8000/generate', json={
    'text': 'Hello, world!',
    'temperature': 0.7
})
result = response.json()

# 保存音频
with open('output.wav', 'wb') as f:
    f.write(base64.b64decode(result['audio_base64']))

# 语音克隆
response = requests.post('http://localhost:8000/generate/voice-clone', json={
    'text': 'This is a cloned voice speaking.',
    'reference_voice': 'belinda'
})
```

### cURL 示例

```bash
# 基础TTS
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello, world!", "temperature": 0.7}'

# 检查健康状态
curl -X GET "http://localhost:8000/health"

# 获取可用声音
curl -X GET "http://localhost:8000/voices"
```

## 性能建议

- 建议使用GPU（至少24GB显存）以获得最佳性能
- 对于生产环境，建议调整 `max_new_tokens` 参数以控制生成长度
- 使用较低的 `temperature` 值（0.3-0.7）可以获得更稳定的输出
- 批量请求时建议设置合适的超时时间