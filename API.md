# Higgs Audio v2 REST API 문서

## 개요
Higgs Audio v2 REST API는 강력한 텍스트 음성 변환 서비스를 제공하며, 기본 TTS 생성, 제로샷 음성 복제, 다중 화자 대화 생성 등의 기능을 지원합니다.

## 서비스 시작

### 기본 시작

```bash
python main.py --host 0.0.0.0 --port 8000
```

### 사용자 정의 설정

```bash
python main.py \
  --host 0.0.0.0 \
  --port 8000 \
  --model-path "bosonai/higgs-audio-v2-generation-3B-base" \
  --audio-tokenizer-path "bosonai/higgs-audio-v2-tokenizer" \
  --device cuda
```

## API 인터페이스

### 1. 상태 확인
**GET** `/health`

서비스 상태와 모델 로딩 상황을 확인합니다.

**응답 예시:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "voices_available": 15
}
```

### 2. 사용 가능한 음성 목록 조회
**GET** `/voices`

사용 가능한 모든 참조 음성 목록을 조회합니다.

**응답 예시:**
```json
[
  {
    "name": "belinda",
    "description": "따뜻하고 친근한 여성 음성"
  },
  {
    "name": "chadwick", 
    "description": "깊고 권위있는 남성 음성"
  }
]
```

### 3. 기본 텍스트 음성 변환
**POST** `/generate`

기본적인 텍스트 음성 변환 생성으로, 모델이 자동으로 적절한 음성을 선택합니다.

**요청 매개변수:**
```json
{
  "text": "안녕하세요, 이것은 음성으로 변환할 샘플 텍스트입니다.",
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 50,
  "max_new_tokens": 1024,
  "seed": 42,
  "scene_prompt": "조용한 방에서 녹음된 오디오입니다."
}
```

**응답 예시:**
```json
{
  "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhY...",
  "text": "안녕하세요, 이것은 음성으로 변환할 샘플 텍스트입니다.",
  "sampling_rate": 24000,
  "duration": 3.5,
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 128,
    "total_tokens": 173
  }
}
```

### 4. 제로샷 음성 복제
**POST** `/generate/voice-clone`

참조 음성을 사용하여 제로샷 음성 복제를 수행합니다.

**요청 매개변수:**
```json
{
  "text": "이 음성을 복제하여 새로운 말을 하게 하고 싶습니다.",
  "reference_voice": "belinda",
  "temperature": 0.3,
  "top_p": 0.95,
  "max_new_tokens": 1024
}
```

또는 업로드된 오디오 사용:
```json
{
  "text": "이 음성을 복제하여 새로운 말을 하게 하고 싶습니다.",
  "reference_audio": "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhY...",
  "temperature": 0.3
}
```

### 5. 다중 화자 대화 생성
**POST** `/generate/multi-speaker`

다중 화자 대화를 생성하며, 각기 다른 화자의 음성을 지정할 수 있습니다.

**요청 매개변수:**
```json
{
  "text": "[SPEAKER0] 안녕하세요! 오늘 어떻게 지내세요?\n[SPEAKER1] 잘 지내고 있어요, 물어봐 주셔서 감사해요!",
  "reference_voices": ["belinda", "chadwick"],
  "temperature": 0.7,
  "max_new_tokens": 2048,
  "scene_prompt": "카페에서의 친근한 대화."
}
```

### 6. 참조 음성 업로드
**POST** `/upload-voice`

새로운 참조 음성 파일을 업로드합니다.

**Form Data:**
- `file`: 오디오 파일 (WAV, MP3, FLAC)
- `name`: 음성 이름 (선택사항, 기본값은 파일명 사용)
- `description`: 음성 설명 (선택사항)

## 요청 매개변수 설명

| 매개변수 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| text | string | - | 변환할 텍스트 내용 |
| temperature | float | 0.7 | 생성 임의성, 0.0-2.0 |
| top_p | float | 0.95 | 핵 샘플링 매개변수, 0.0-1.0 |
| top_k | int | 50 | Top-k 샘플링 매개변수 |
| max_new_tokens | int | 1024 | 최대 생성 토큰 수 |
| seed | int | null | 재현 가능한 생성을 위한 랜덤 시드 |
| scene_prompt | string | null | 음성 생성의 맥락을 제공하는 장면 설명 |

## 다중 화자 텍스트 형식

다중 화자 대화의 경우 다음 형식을 사용하세요:

```
[SPEAKER0] 첫 번째 화자의 텍스트 내용
[SPEAKER1] 두 번째 화자의 텍스트 내용
[SPEAKER0] 첫 번째 화자의 응답
```

## 오디오 형식
- 입력 오디오: WAV, MP3, FLAC 형식 지원
- 출력 오디오: WAV 형식, 24kHz 샘플링 레이트
- 인코딩: Base64 인코딩된 오디오 데이터

## 오류 처리

API는 표준 HTTP 상태 코드를 사용합니다:
- `200`: 요청 성공
- `400`: 요청 매개변수 오류
- `500`: 서버 내부 오류
- `503`: 서비스 사용 불가 (모델 미로딩)

**오류 응답 예시:**
```json
{
  "detail": "참조 음성 'unknown_voice'를 찾을 수 없습니다. 사용 가능한 음성: ['belinda', 'chadwick', 'mabel']"
}
```

## 사용 예제

### Python 클라이언트 예제

```python
import requests
import base64

# 기본 TTS
response = requests.post('http://localhost:8000/generate', json={
    'text': '안녕, 세상아!',
    'temperature': 0.7
})
result = response.json()

# 오디오 저장
with open('output.wav', 'wb') as f:
    f.write(base64.b64decode(result['audio_base64']))

# 음성 복제
response = requests.post('http://localhost:8000/generate/voice-clone', json={
    'text': '이것은 복제된 음성이 말하고 있습니다.',
    'reference_voice': 'belinda'
})
```

### cURL 예제

```bash
# 기본 TTS
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"text": "안녕, 세상아!", "temperature": 0.7}'

# 상태 확인
curl -X GET "http://localhost:8000/health"

# 사용 가능한 음성 조회
curl -X GET "http://localhost:8000/voices"
```

## 성능 권장사항
- 최적의 성능을 위해 GPU 사용을 권장합니다 (최소 24GB VRAM)
- 프로덕션 환경에서는 생성 길이 제어를 위해 `max_new_tokens` 매개변수를 조정하는 것을 권장합니다
- 더 안정적인 출력을 위해 낮은 `temperature` 값 (0.3-0.7)을 사용하세요
- 배치 요청 시 적절한 타임아웃 시간을 설정하는 것을 권장합니다
