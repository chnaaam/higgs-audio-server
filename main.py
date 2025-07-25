#!/usr/bin/env python3
"""
Higgs Audio v2 REST API Server

This server provides RESTful API endpoints for Higgs Audio v2 text-to-speech generation,
including basic TTS, voice cloning, and multi-speaker dialog generation.

Usage:
    python main.py --host 0.0.0.0 --port 8000

Endpoints:
    POST /generate              - Basic text-to-speech generation  
    POST /generate/voice-clone  - Zero-shot voice cloning
    POST /generate/multi-speaker - Multi-speaker dialog generation
    GET /health                 - Health check
    GET /voices                 - List available reference voices
"""

import os
import base64
import asyncio
import tempfile
from typing import Optional, List, Dict, Any
from pathlib import Path
import uuid

import torch
import torchaudio
import librosa
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import click
from loguru import logger

from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent, TextContent


# Configuration
DEFAULT_MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
DEFAULT_AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
DEFAULT_SCENE_PROMPT = "Audio is recorded from a quiet room."

# Global variables
serve_engine: Optional[HiggsAudioServeEngine] = None
voice_prompts_dir = Path(__file__).parent / "examples" / "voice_prompts"


# Request/Response Models
class GenerateRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature for generation")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: Optional[int] = Field(50, ge=1, description="Top-k sampling parameter")
    max_new_tokens: int = Field(1024, ge=1, le=4096, description="Maximum new tokens to generate")
    seed: Optional[int] = Field(None, description="Random seed for reproducible generation")
    scene_prompt: Optional[str] = Field(None, description="Scene description for context")


class VoiceCloneRequest(GenerateRequest):
    reference_audio: Optional[str] = Field(None, description="Base64 encoded reference audio (WAV format)")
    reference_voice: Optional[str] = Field(None, description="Name of built-in reference voice")


class MultiSpeakerRequest(GenerateRequest):
    reference_voices: Optional[List[str]] = Field(None, description="List of reference voice names for speakers")
    reference_audios: Optional[List[str]] = Field(None, description="List of base64 encoded reference audios")


class GenerateResponse(BaseModel):
    audio_base64: str = Field(..., description="Generated audio in base64 encoded WAV format")
    text: str = Field(..., description="Generated or input text")
    sampling_rate: int = Field(..., description="Audio sampling rate")
    duration: float = Field(..., description="Audio duration in seconds")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool
    device: str
    voices_available: int


class VoiceInfo(BaseModel):
    name: str
    description: Optional[str] = None


# Utility functions
def audio_to_base64(audio_array: np.ndarray, sample_rate: int) -> str:
    """Convert numpy audio array to base64 encoded WAV."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        torchaudio.save(tmp_file.name, torch.from_numpy(audio_array)[None, :], sample_rate)
        with open(tmp_file.name, "rb") as f:
            audio_data = f.read()
        os.unlink(tmp_file.name)
    return base64.b64encode(audio_data).decode()


def base64_to_audio(audio_base64: str) -> tuple[np.ndarray, int]:
    """Convert base64 encoded audio to numpy array."""
    audio_data = base64.b64decode(audio_base64)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(audio_data)
        tmp_file.flush()
        audio_array, sample_rate = librosa.load(tmp_file.name, sr=None)
        os.unlink(tmp_file.name)
    return audio_array, sample_rate


def get_available_voices() -> List[VoiceInfo]:
    """Get list of available reference voices."""
    voices = []
    if voice_prompts_dir.exists():
        for wav_file in voice_prompts_dir.glob("*.wav"):
            name = wav_file.stem
            txt_file = voice_prompts_dir / f"{name}.txt"
            description = None
            if txt_file.exists():
                try:
                    description = txt_file.read_text(encoding="utf-8").strip()
                except Exception:
                    pass
            voices.append(VoiceInfo(name=name, description=description))
    return voices


def build_system_message(scene_prompt: Optional[str] = None, 
                        reference_voices: Optional[List[str]] = None) -> str:
    """Build system message with scene description."""
    base_message = "Generate audio following instruction."
    
    if scene_prompt or reference_voices:
        scene_desc_parts = []
        if scene_prompt:
            scene_desc_parts.append(scene_prompt)
        if reference_voices and len(reference_voices) > 1:
            # Multi-speaker scenario
            for i, voice in enumerate(reference_voices):
                scene_desc_parts.append(f"SPEAKER{i}: Use the provided reference voice")
        
        if scene_desc_parts:
            scene_desc = "\n".join(scene_desc_parts)
            return f"{base_message}\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>"
    
    return base_message


# FastAPI app
app = FastAPI(
    title="Higgs Audio v2 API",
    description="REST API for Higgs Audio v2 text-to-speech generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global serve_engine
    logger.info("Starting Higgs Audio v2 API server...")
    
    # Get model paths from environment variables or use defaults
    model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    audio_tokenizer_path = os.getenv("AUDIO_TOKENIZER_PATH", DEFAULT_AUDIO_TOKENIZER_PATH)
    device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        serve_engine = HiggsAudioServeEngine(
            model_name_or_path=model_path,
            audio_tokenizer_name_or_path=audio_tokenizer_path,
            device=device,
        )
        logger.info(f"Model loaded successfully on {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        model_loaded=serve_engine is not None,
        device=serve_engine.device if serve_engine else "unknown",
        voices_available=len(get_available_voices())
    )


@app.get("/voices", response_model=List[VoiceInfo])
async def list_voices():
    """Get list of available reference voices."""
    return get_available_voices()


@app.post("/generate", response_model=GenerateResponse)
async def generate_speech(request: GenerateRequest):
    """Basic text-to-speech generation."""
    if not serve_engine:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Build system message
        system_prompt = build_system_message(
            scene_prompt=request.scene_prompt or DEFAULT_SCENE_PROMPT
        )
        
        # Create ChatML sample
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=request.text),
        ]
        chat_ml_sample = ChatMLSample(messages=messages)
        
        # Generate audio
        logger.info(f"Generating speech for text: {request.text[:100]}...")
        output: HiggsAudioResponse = serve_engine.generate(
            chat_ml_sample=chat_ml_sample,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            seed=request.seed,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        )
        
        if output.audio is None:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
        
        # Convert audio to base64
        audio_base64 = audio_to_base64(output.audio, output.sampling_rate)
        duration = len(output.audio) / output.sampling_rate
        
        return GenerateResponse(
            audio_base64=audio_base64,
            text=output.generated_text or request.text,
            sampling_rate=output.sampling_rate,
            duration=duration,
            usage=output.usage or {}
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/voice-clone", response_model=GenerateResponse)
async def generate_voice_clone(request: VoiceCloneRequest):
    """Zero-shot voice cloning generation."""
    if not serve_engine:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare audio content
        audio_content = None
        reference_text = "Please use this voice as reference for cloning."
        
        if request.reference_audio:
            # Use uploaded base64 audio  
            audio_content = AudioContent(raw_audio=request.reference_audio, audio_url="placeholder")
        elif request.reference_voice:
            # Use built-in reference voice
            voice_file = voice_prompts_dir / f"{request.reference_voice}.wav"
            if not voice_file.exists():
                available_voices = [v.name for v in get_available_voices()]
                raise HTTPException(
                    status_code=400, 
                    detail=f"Reference voice '{request.reference_voice}' not found. Available: {available_voices}"
                )
            # For built-in voices, only set audio_url (don't set raw_audio)
            audio_content = AudioContent(audio_url=str(voice_file))
            
            # For built-in voices, try to get the reference text
            txt_file = voice_prompts_dir / f"{request.reference_voice}.txt"
            if txt_file.exists():
                try:
                    reference_text = txt_file.read_text(encoding="utf-8").strip()
                except Exception:
                    pass
        else:
            raise HTTPException(status_code=400, detail="Either reference_audio or reference_voice must be provided")
        
        # Build system message
        system_prompt = build_system_message(
            scene_prompt=request.scene_prompt or DEFAULT_SCENE_PROMPT
        )
        
        # Create ChatML sample with reference audio (following the correct pattern)
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=reference_text),
            Message(role="assistant", content=audio_content),
            Message(role="user", content=request.text),
        ]
        chat_ml_sample = ChatMLSample(messages=messages)
        
        # Generate audio
        logger.info(f"Generating voice clone for text: {request.text[:100]}...")
        output: HiggsAudioResponse = serve_engine.generate(
            chat_ml_sample=chat_ml_sample,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            seed=request.seed,
            force_audio_gen=True,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        )
        
        if output.audio is None:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
        
        # Convert audio to base64
        audio_base64 = audio_to_base64(output.audio, output.sampling_rate)
        duration = len(output.audio) / output.sampling_rate
        
        return GenerateResponse(
            audio_base64=audio_base64,
            text=output.generated_text or request.text,
            sampling_rate=output.sampling_rate,
            duration=duration,
            usage=output.usage or {}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice clone generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/multi-speaker", response_model=GenerateResponse)
async def generate_multi_speaker(request: MultiSpeakerRequest):
    """Multi-speaker dialog generation."""
    if not serve_engine:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare audio contents for speakers
        audio_contents = []
        if request.reference_voices:
            for voice_name in request.reference_voices:
                voice_file = voice_prompts_dir / f"{voice_name}.wav"
                if not voice_file.exists():
                    available_voices = [v.name for v in get_available_voices()]
                    raise HTTPException(
                        status_code=400,
                        detail=f"Reference voice '{voice_name}' not found. Available: {available_voices}"
                    )
                audio_contents.append(AudioContent(audio_url=str(voice_file)))
        elif request.reference_audios:
            for audio_b64 in request.reference_audios:
                audio_contents.append(AudioContent(raw_audio=audio_b64, audio_url="placeholder"))
        
        # Build system message for multi-speaker
        system_prompt = build_system_message(
            scene_prompt=request.scene_prompt or DEFAULT_SCENE_PROMPT,
            reference_voices=request.reference_voices or [f"speaker_{i}" for i in range(len(request.reference_audios or []))]
        )
        
        # Create ChatML sample
        messages = [Message(role="system", content=system_prompt)]
        
        # Add reference voices as examples if provided
        if audio_contents:
            for i, audio_content in enumerate(audio_contents):
                messages.extend([
                    Message(role="user", content=f"[SPEAKER{i}] Reference voice sample."),
                    Message(role="assistant", content=audio_content),
                ])
        
        # Add the actual generation request
        messages.append(Message(role="user", content=request.text))
        
        chat_ml_sample = ChatMLSample(messages=messages)
        
        # Generate audio
        logger.info(f"Generating multi-speaker dialog for text: {request.text[:100]}...")
        output: HiggsAudioResponse = serve_engine.generate(
            chat_ml_sample=chat_ml_sample,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            seed=request.seed,
            force_audio_gen=True,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        )
        
        if output.audio is None:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
        
        # Convert audio to base64
        audio_base64 = audio_to_base64(output.audio, output.sampling_rate)
        duration = len(output.audio) / output.sampling_rate
        
        return GenerateResponse(
            audio_base64=audio_base64,
            text=output.generated_text or request.text,
            sampling_rate=output.sampling_rate,
            duration=duration,
            usage=output.usage or {}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multi-speaker generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-voice")
async def upload_reference_voice(
    file: UploadFile = File(...),
    name: str = "",
    description: str = ""
):
    """Upload a new reference voice."""
    if not file.filename.endswith(('.wav', '.mp3', '.flac')):
        raise HTTPException(status_code=400, detail="Only audio files (.wav, .mp3, .flac) are allowed")
    
    try:
        # Use filename as name if not provided
        voice_name = name or Path(file.filename).stem
        
        # Save the uploaded audio file
        voice_file = voice_prompts_dir / f"{voice_name}.wav"
        
        # Read and convert audio to WAV format
        audio_data = await file.read()
        with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix) as tmp_file:
            tmp_file.write(audio_data)
            tmp_file.flush()
            
            # Load and resave as WAV
            audio_array, sr = librosa.load(tmp_file.name, sr=24000)  # Resample to 24kHz
            torchaudio.save(str(voice_file), torch.from_numpy(audio_array)[None, :], 24000)
        
        # Save description if provided
        if description:
            desc_file = voice_prompts_dir / f"{voice_name}.txt"
            desc_file.write_text(description, encoding="utf-8")
        
        return {"message": f"Voice '{voice_name}' uploaded successfully"}
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@click.command()
@click.option("--host", default="127.0.0.1", help="Host to bind the server")
@click.option("--port", default=8000, type=int, help="Port to bind the server")
@click.option("--workers", default=1, type=int, help="Number of worker processes")
@click.option("--model-path", default=DEFAULT_MODEL_PATH, help="Path to the Higgs Audio model")
@click.option("--audio-tokenizer-path", default=DEFAULT_AUDIO_TOKENIZER_PATH, help="Path to the audio tokenizer")
@click.option("--device", default=None, help="Device to run the model (cuda/cpu)")
def main(host, port, workers, model_path, audio_tokenizer_path, device):
    """Start the Higgs Audio v2 REST API server."""
    # Set environment variables for the startup event
    os.environ["MODEL_PATH"] = model_path
    os.environ["AUDIO_TOKENIZER_PATH"] = audio_tokenizer_path
    if device:
        os.environ["DEVICE"] = device
    
    # Ensure voice prompts directory exists
    voice_prompts_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Audio Tokenizer: {audio_tokenizer_path}")
    logger.info(f"Device: {device or 'auto-detect'}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()