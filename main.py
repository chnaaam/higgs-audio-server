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
import yaml
import re
import jieba
import langid

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
voice_profiles = {}

# Load voice profiles
def load_voice_profiles():
    """Load voice profiles from YAML file."""
    global voice_profiles
    profile_file = voice_prompts_dir / "profile.yaml"
    if profile_file.exists():
        try:
            with open(profile_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                voice_profiles = data.get('profiles', {})
                logger.info(f"Loaded {len(voice_profiles)} voice profiles")
        except Exception as e:
            logger.error(f"Failed to load voice profiles: {e}")
            voice_profiles = {}
    else:
        voice_profiles = {}

# Text chunking utilities
def normalize_chinese_punctuation(text):
    """Convert Chinese punctuation to English equivalents."""
    chinese_to_english_punct = {
        "，": ", ", "。": ".", "：": ":", "；": ";", "？": "?", "！": "!",
        "（": "(", "）": ")", "【": "[", "】": "]", "《": "<", "》": ">",
        """: '"', """: '"', "'": "'", "'": "'", "、": ",", "—": "-",
        "…": "...", "·": ".", "「": '"', "」": '"', "『": '"', "』": '"',
    }
    for zh_punct, en_punct in chinese_to_english_punct.items():
        text = text.replace(zh_punct, en_punct)
    return text

def prepare_chunk_text(text, chunk_method: Optional[str] = None, chunk_max_word_num: int = 100, chunk_max_num_turns: int = 1):
    """Chunk text into smaller pieces."""
    if chunk_method is None:
        return [text]
    
    if chunk_method == "speaker":
        # Split by speaker tags
        speaker_pattern = r'(\[SPEAKER\d+\][^\[]*?)(?=\[SPEAKER\d+\]|$)'
        chunks = re.findall(speaker_pattern, text, re.DOTALL)
        if not chunks:
            return [text]
        
        # Group chunks by turns
        grouped_chunks = []
        current_group = []
        turn_count = 0
        
        for chunk in chunks:
            current_group.append(chunk.strip())
            if len(current_group) >= 2:  # One turn = both speakers speak
                turn_count += 1
                if turn_count >= chunk_max_num_turns:
                    grouped_chunks.append(''.join(current_group))
                    current_group = []
                    turn_count = 0
        
        if current_group:
            grouped_chunks.append(''.join(current_group))
        
        return grouped_chunks if grouped_chunks else [text]
    
    elif chunk_method == "word":
        # Split by words
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= chunk_max_word_num:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [text]
    
    return [text]


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
    voice_profile: Optional[str] = Field(None, description="Text description of voice characteristics")
    cross_lingual: bool = Field(False, description="Enable cross-lingual voice cloning")


class MultiSpeakerRequest(GenerateRequest):
    reference_voices: Optional[List[str]] = Field(None, description="List of reference voice names for speakers")
    reference_audios: Optional[List[str]] = Field(None, description="List of base64 encoded reference audios")
    voice_profiles: Optional[List[str]] = Field(None, description="List of voice profile descriptions")
    chunk_method: Optional[str] = Field(None, description="Chunking method: 'speaker' or 'word'")
    chunk_max_word_num: int = Field(100, description="Max words per chunk when using word chunking")
    chunk_max_num_turns: int = Field(1, description="Max conversation turns per chunk when using speaker chunking")


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
    type: str = Field(default="audio", description="Type: 'audio' or 'profile'")

class ExperimentalRequest(GenerateRequest):
    mode: str = Field(..., description="Experimental mode: 'humming' or 'bgm'")
    reference_audio: Optional[str] = Field(None, description="Base64 encoded reference audio")
    reference_voice: Optional[str] = Field(None, description="Name of built-in reference voice")

class LongFormRequest(GenerateRequest):
    reference_audio: Optional[str] = Field(None, description="Base64 encoded reference audio")
    reference_voice: Optional[str] = Field(None, description="Name of built-in reference voice")
    chunk_method: str = Field("word", description="Chunking method: 'word' or 'sentence'")
    chunk_max_word_num: int = Field(100, description="Max words per chunk")
    generation_chunk_buffer_size: int = Field(2, description="Buffer size for chunk generation")


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
    
    # Add audio-based voices
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
            voices.append(VoiceInfo(name=name, description=description, type="audio"))
    
    # Add profile-based voices
    global voice_profiles
    for profile_name, profile_desc in voice_profiles.items():
        voices.append(VoiceInfo(
            name=f"profile:{profile_name}", 
            description=profile_desc, 
            type="profile"
        ))
    
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
    
    # Load voice profiles
    load_voice_profiles()
    
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
    """Zero-shot voice cloning generation with enhanced features."""
    if not serve_engine:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Handle voice reference with enhanced support
        audio_content, reference_text = handle_voice_reference(
            request.reference_voice, request.reference_audio, request.voice_profile
        )
        
        # Handle cross-lingual cloning
        text_to_generate = request.text
        if request.cross_lingual:
            # For cross-lingual, we might want to adjust the scene prompt
            scene_prompt = request.scene_prompt or "Audio is recorded from a quiet room."
        else:
            scene_prompt = request.scene_prompt or DEFAULT_SCENE_PROMPT
        
        # Build system message
        system_prompt = build_system_message(scene_prompt=scene_prompt)
        
        # Create ChatML sample with reference audio
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=reference_text),
            Message(role="assistant", content=audio_content),
            Message(role="user", content=text_to_generate),
        ]
        chat_ml_sample = ChatMLSample(messages=messages)
        
        # Adjust temperature for cross-lingual (lower for better stability)
        temperature = request.temperature
        if request.cross_lingual and temperature > 0.5:
            temperature = 0.3
            logger.info("Adjusted temperature to 0.3 for cross-lingual generation")
        
        # Generate audio
        logger.info(f"Generating voice clone for text: {request.text[:100]}...")
        if request.cross_lingual:
            logger.info("Using cross-lingual mode")
            
        output: HiggsAudioResponse = serve_engine.generate(
            chat_ml_sample=chat_ml_sample,
            max_new_tokens=request.max_new_tokens,
            temperature=temperature,
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
    """Enhanced multi-speaker dialog generation with chunking support."""
    if not serve_engine:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Normalize and potentially chunk the text
        text = normalize_chinese_punctuation(request.text)
        
        # Handle chunking if requested
        if request.chunk_method:
            chunks = prepare_chunk_text(
                text,
                chunk_method=request.chunk_method,
                chunk_max_word_num=request.chunk_max_word_num,
                chunk_max_num_turns=request.chunk_max_num_turns
            )
            logger.info(f"Processing multi-speaker text in {len(chunks)} chunks")
        else:
            chunks = [text]
        
        # Prepare audio contents for speakers
        audio_contents = []
        
        if request.reference_voices:
            for voice_name in request.reference_voices:
                if voice_name.startswith("profile:"):
                    # Handle profile-based voice
                    profile_name = voice_name[8:]
                    global voice_profiles
                    if profile_name not in voice_profiles:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Voice profile '{profile_name}' not found. Available: {list(voice_profiles.keys())}"
                        )
                    profile_desc = voice_profiles[profile_name]
                    audio_contents.append(create_voice_profile_audio_content(profile_desc))
                else:
                    # Handle regular audio file
                    voice_file = voice_prompts_dir / f"{voice_name}.wav"
                    if not voice_file.exists():
                        available_voices = [v.name for v in get_available_voices() if v.type == "audio"]
                        raise HTTPException(
                            status_code=400,
                            detail=f"Reference voice '{voice_name}' not found. Available: {available_voices}"
                        )
                    audio_contents.append(AudioContent(audio_url=str(voice_file)))
        elif request.reference_audios:
            for audio_b64 in request.reference_audios:
                audio_contents.append(AudioContent(raw_audio=audio_b64, audio_url="placeholder"))
        elif request.voice_profiles:
            for profile_desc in request.voice_profiles:
                audio_contents.append(create_voice_profile_audio_content(profile_desc))
        
        # Process all chunks
        all_audio = []
        
        for i, chunk_text in enumerate(chunks):
            logger.info(f"Processing multi-speaker chunk {i+1}/{len(chunks)}")
            
            # Build system message for multi-speaker
            reference_voice_names = (
                request.reference_voices or 
                request.voice_profiles or 
                [f"speaker_{j}" for j in range(len(request.reference_audios or []))]
            )
            
            system_prompt = build_system_message(
                scene_prompt=request.scene_prompt or DEFAULT_SCENE_PROMPT,
                reference_voices=reference_voice_names
            )
            
            # Create ChatML sample for this chunk
            messages = [Message(role="system", content=system_prompt)]
            
            # Add reference voices as examples if provided
            if audio_contents:
                for j, audio_content in enumerate(audio_contents):
                    reference_text = f"[SPEAKER{j}] Reference voice sample."
                    if request.voice_profiles and j < len(request.voice_profiles):
                        reference_text = f"[SPEAKER{j}] Voice with characteristics: {request.voice_profiles[j]}"
                    messages.extend([
                        Message(role="user", content=reference_text),
                        Message(role="assistant", content=audio_content),
                    ])
            
            # Add the actual generation request
            messages.append(Message(role="user", content=chunk_text))
            
            chat_ml_sample = ChatMLSample(messages=messages)
            
            # Generate audio for this chunk
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
                raise HTTPException(status_code=500, detail=f"Failed to generate audio for chunk {i+1}")
            
            all_audio.append(output.audio)
        
        # Concatenate all chunks if multiple
        if len(all_audio) > 1:
            final_audio = np.concatenate(all_audio)
        else:
            final_audio = all_audio[0]
        
        # Convert audio to base64
        audio_base64 = audio_to_base64(final_audio, output.sampling_rate)
        duration = len(final_audio) / output.sampling_rate
        
        return GenerateResponse(
            audio_base64=audio_base64,
            text=output.generated_text or text,
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


def create_voice_profile_audio_content(profile_desc: str) -> AudioContent:
    """Create an AudioContent for voice profile description."""
    # This is a placeholder implementation. In a real system, you might want to 
    # convert the text description to a special token or handle it differently
    return AudioContent(raw_audio="", audio_url=f"profile:{profile_desc}")


def handle_voice_reference(reference_voice: str, reference_audio: str, voice_profile: str):
    """Handle different types of voice references."""
    audio_content = None
    reference_text = "Please use this voice as reference."
    
    if reference_audio:
        # Use uploaded base64 audio
        audio_content = AudioContent(raw_audio=reference_audio, audio_url="placeholder")
    elif reference_voice:
        if reference_voice.startswith("profile:"):
            # Handle profile-based voice
            profile_name = reference_voice[8:]  # Remove "profile:" prefix
            global voice_profiles
            if profile_name not in voice_profiles:
                raise HTTPException(
                    status_code=400,
                    detail=f"Voice profile '{profile_name}' not found. Available: {list(voice_profiles.keys())}"
                )
            # Create a special audio content for profile
            profile_desc = voice_profiles[profile_name]
            reference_text = f"Please use a voice with the following characteristics: {profile_desc}"
            audio_content = create_voice_profile_audio_content(profile_desc)
        else:
            # Use built-in reference voice
            voice_file = voice_prompts_dir / f"{reference_voice}.wav"
            if not voice_file.exists():
                available_voices = [v.name for v in get_available_voices() if v.type == "audio"]
                raise HTTPException(
                    status_code=400,
                    detail=f"Reference voice '{reference_voice}' not found. Available: {available_voices}"
                )
            audio_content = AudioContent(audio_url=str(voice_file))
            
            # Try to get reference text
            txt_file = voice_prompts_dir / f"{reference_voice}.txt"
            if txt_file.exists():
                try:
                    reference_text = txt_file.read_text(encoding="utf-8").strip()
                except Exception:
                    pass
    elif voice_profile:
        # Use text-based voice description
        reference_text = f"Please use a voice with the following characteristics: {voice_profile}"
        audio_content = create_voice_profile_audio_content(voice_profile)
    else:
        raise HTTPException(status_code=400, detail="Must provide reference_audio, reference_voice, or voice_profile")
    
    return audio_content, reference_text


@app.post("/generate/experimental", response_model=GenerateResponse)
async def generate_experimental(request: ExperimentalRequest):
    """Generate experimental audio with special effects (humming, BGM, etc.)."""
    if not serve_engine:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Handle different experimental modes
        if request.mode == "humming":
            # For humming, we wrap the text with special tags
            enhanced_text = f"Are you asking if I can hum a tune? Of course I can! [humming start] {request.text} [humming end] See?"
        elif request.mode == "bgm":
            # For background music, wrap with music tags
            enhanced_text = f"[music start] {request.text} [music end]"
        else:
            raise HTTPException(status_code=400, detail="Unsupported experimental mode. Use 'humming' or 'bgm'")
        
        # Handle voice reference
        audio_content, reference_text = handle_voice_reference(
            request.reference_voice, request.reference_audio, None
        )
        
        # Build system message
        system_prompt = build_system_message(
            scene_prompt=request.scene_prompt or DEFAULT_SCENE_PROMPT
        )
        
        # Create ChatML sample
        if audio_content:
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=reference_text),
                Message(role="assistant", content=audio_content),
                Message(role="user", content=enhanced_text),
            ]
        else:
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=enhanced_text),
            ]
        
        chat_ml_sample = ChatMLSample(messages=messages)
        
        # Generate audio
        logger.info(f"Generating experimental audio ({request.mode}) for text: {request.text[:100]}...")
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
            text=output.generated_text or enhanced_text,
            sampling_rate=output.sampling_rate,
            duration=duration,
            usage=output.usage or {}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Experimental generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/long-form", response_model=GenerateResponse)
async def generate_long_form(request: LongFormRequest):
    """Generate long-form audio with chunking support."""
    if not serve_engine:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Normalize text
        text = normalize_chinese_punctuation(request.text)
        
        # Chunk the text
        chunks = prepare_chunk_text(
            text, 
            chunk_method=request.chunk_method,
            chunk_max_word_num=request.chunk_max_word_num
        )
        
        logger.info(f"Processing long-form text in {len(chunks)} chunks")
        
        # Handle voice reference
        audio_content, reference_text = handle_voice_reference(
            request.reference_voice, request.reference_audio, None
        )
        
        # Process chunks
        all_audio = []
        previous_audio_content = audio_content  # Keep reference for continuity
        
        for i, chunk_text in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Build system message
            system_prompt = build_system_message(
                scene_prompt=request.scene_prompt or DEFAULT_SCENE_PROMPT
            )
            
            # Create messages for this chunk
            messages = [Message(role="system", content=system_prompt)]
            
            # Add reference audio if available
            if previous_audio_content:
                messages.extend([
                    Message(role="user", content=reference_text),
                    Message(role="assistant", content=previous_audio_content),
                ])
            
            # Add current chunk
            messages.append(Message(role="user", content=chunk_text))
            
            chat_ml_sample = ChatMLSample(messages=messages)
            
            # Generate audio for this chunk
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
                raise HTTPException(status_code=500, detail=f"Failed to generate audio for chunk {i+1}")
            
            all_audio.append(output.audio)
            
            # Update previous audio for continuity (use last generated audio as reference)
            if len(all_audio) >= request.generation_chunk_buffer_size:
                # Use recent audio as context for next chunk
                recent_audio = np.concatenate(all_audio[-request.generation_chunk_buffer_size:])
                audio_base64 = audio_to_base64(recent_audio, output.sampling_rate)
                previous_audio_content = AudioContent(raw_audio=audio_base64, audio_url="previous_chunk")
        
        # Concatenate all audio chunks
        final_audio = np.concatenate(all_audio)
        
        # Convert final audio to base64
        audio_base64 = audio_to_base64(final_audio, output.sampling_rate)
        duration = len(final_audio) / output.sampling_rate
        
        return GenerateResponse(
            audio_base64=audio_base64,
            text=text,
            sampling_rate=output.sampling_rate,
            duration=duration,
            usage=output.usage or {}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Long-form generation error: {e}")
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