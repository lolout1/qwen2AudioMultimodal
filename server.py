import os
import io
import hashlib
import torch
import librosa
import requests
import aiofiles
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

app = FastAPI(title="Qwen2-Audio Inference API")

# --- Configuration and Paths ---
BASE_DIR = os.getcwd()  # Current working directory
MODEL_PATH = os.path.join(BASE_DIR, "Qwen2-Audio-7B-Instruct")  # Local model clone
AUDIO_SAVE_DIR = os.path.join(BASE_DIR, "audio_files")
os.makedirs(AUDIO_SAVE_DIR, exist_ok=True)

# Set offline mode externally (e.g., export HF_HUB_OFFLINE=1)
SAMPLING_RATE = 16000  # Force resampling to 16 kHz

# Force model onto one GPU (cuda:0) if available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def debug_print(msg: str):
    print(f"[DEBUG] {msg}")

# --- Model Loading ---
def load_model_and_processor():
    debug_print("Loading model and processor from local files...")
    try:
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch_dtype,
            device_map={"": "cuda:0"} if torch.cuda.is_available() else None,
            trust_remote_code=True,
            local_files_only=True
        )
    except Exception as e:
        debug_print(f"Error loading model: {e}")
        raise e
    model.tie_weights()  # Ensure weights are tied
    try:
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True
        )
    except Exception as e:
        debug_print(f"Error loading processor: {e}")
        raise e
    debug_print("Model and processor loaded successfully.")
    return model, processor

model, processor = load_model_and_processor()

# Global cache to avoid reprocessing identical audio files
cached_audio_hash = None
cached_audio_data = None

def slice_prompt_from_generation(generated_ids, prompt_len: int):
    debug_print(f"Slicing out {prompt_len} tokens from generated output.")
    return generated_ids[:, prompt_len:]

async def save_upload_file(upload_file: UploadFile, destination: str):
    debug_print(f"Saving uploaded file to {destination}...")
    try:
        async with aiofiles.open(destination, 'wb') as out_file:
            content = await upload_file.read()
            await out_file.write(content)
        debug_print("File saved successfully.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {e}")

def load_audio_from_disk(file_path: str) -> list:
    debug_print(f"Loading audio from disk: {file_path} at sr={SAMPLING_RATE}...")
    try:
        audio_data, _ = librosa.load(file_path, sr=SAMPLING_RATE)
        debug_print(f"Audio loaded: {len(audio_data)} samples.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading audio from disk: {e}")
    return audio_data

def load_audio_from_bytes(audio_bytes: bytes) -> list:
    debug_print("Loading audio from bytes with sr=" + str(SAMPLING_RATE))
    try:
        audio_data, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLING_RATE)
        debug_print(f"Audio loaded from bytes: {len(audio_data)} samples.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing audio bytes: {e}")
    return audio_data

def load_audio_from_url(audio_url: str) -> list:
    debug_print(f"Downloading audio from URL: {audio_url}...")
    try:
        r = requests.get(audio_url)
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading audio from URL: {e}")
    return load_audio_from_bytes(r.content)

def compute_md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()

# --- Conversation Template Helper ---
def build_conversation(audio_placeholder: str, prompt: str):
    """
    Build a chat-style conversation. For local files, we use a placeholder in the audio field.
    """
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_placeholder},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    return conversation

# --- Transcription Function using Chat Template ---
def transcribe_conversation(audio_data, prompt: str) -> str:
    # Use a placeholder for local audio; this value is ignored if audios parameter is provided.
    audio_placeholder = "LOCAL_AUDIO"
    conversation = build_conversation(audio_placeholder, prompt)
    debug_print(f"Conversation: {conversation}")
    try:
        chat_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        debug_print("Chat prompt created via apply_chat_template.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error applying chat template: {e}")
    
    try:
        inputs = processor(
            text=chat_prompt,
            audios=audio_data,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
            padding=True
        )
        debug_print("Inputs prepared successfully.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing inputs: {e}")
    
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device)
    
    try:
        debug_print("Starting generation with sample parameters...")
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=60000,
            min_new_tokens=32,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            top_k=50
        )
        prompt_length = inputs.input_ids.shape[1]
        generated_ids = slice_prompt_from_generation(generated_ids, prompt_length)
        transcription = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        debug_print("Generation complete.")
        return transcription
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during generation: {e}")

# --- Web Demo Endpoint ---
@app.get("/")
def read_root():
    return HTMLResponse(
        """
        <html>
          <head>
            <title>Qwen2-Audio-Instruct Demo</title>
          </head>
          <body>
            <h1>Qwen2-Audio-Instruct Demo</h1>
            <p>Upload an audio file (.wav, .mp3, .flac) and provide a text prompt to initiate a chat-style transcription.</p>
            <form action="/transcribe" method="post" enctype="multipart/form-data">
              <label for="audio">Audio File:</label>
              <input type="file" id="audio" name="audio" accept=".wav,.mp3,.flac" required><br><br>
              <label for="prompt">Text Prompt:</label>
              <input type="text" id="prompt" name="prompt" value="What is happening in this audio?"><br><br>
              <input type="submit" value="Transcribe">
            </form>
          </body>
        </html>
        """
    )

@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    prompt: str = Form("What is happening in this audio?")
):
    global cached_audio_hash, cached_audio_data
    debug_print("Received file upload for transcription.")
    
    # Save the uploaded file
    save_path = os.path.join(AUDIO_SAVE_DIR, audio.filename)
    await save_upload_file(audio, save_path)
    debug_print(f"Uploaded file saved at {save_path}")
    
    # Read file bytes from disk for caching
    try:
        with open(save_path, "rb") as f:
            file_bytes = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading saved file: {e}")
    
    file_hash = compute_md5(file_bytes)
    debug_print(f"Computed MD5 hash: {file_hash}")
    if file_hash == cached_audio_hash and cached_audio_data is not None:
        debug_print("Using cached audio data.")
        audio_data = cached_audio_data
    else:
        debug_print("Processing new audio file.")
        audio_data = load_audio_from_disk(save_path)
        cached_audio_hash = file_hash
        cached_audio_data = audio_data

    transcription = transcribe_conversation(audio_data, prompt)
    return {"transcription": transcription}

# --- JSON API Endpoint ---
class TranscriptionRequest(BaseModel):
    prompt: str = "What is happening in this audio?"
    audio_url: str = None  # Optional URL to an audio file

@app.post("/api/transcribe")
def api_transcribe(request: TranscriptionRequest):
    debug_print("Received API transcription request.")
    audio_data = None
    if request.audio_url:
        audio_data = load_audio_from_url(request.audio_url)
        debug_print("Audio data loaded from URL.")
    else:
        debug_print("No audio URL provided; using text-only input.")

    if audio_data is not None:
        conversation = build_conversation("REMOTE_AUDIO", request.prompt)
        try:
            chat_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            inputs = processor(
                text=chat_prompt,
                audios=audio_data,
                sampling_rate=SAMPLING_RATE,
                return_tensors="pt",
                padding=True
            )
            debug_print("Inputs prepared successfully with audio.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error preparing inputs with audio: {e}")
    else:
        try:
            inputs = processor(text=request.prompt, return_tensors="pt")
            debug_print("Text-only inputs prepared successfully.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error preparing text-only inputs: {e}")

    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device)
    
    try:
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=60000,
            min_new_tokens=32,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            top_k=50
        )
        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = slice_prompt_from_generation(generated_ids, prompt_length)
        output_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        debug_print("API generation complete.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during generation: {e}")
    
    return {"response": output_text}

