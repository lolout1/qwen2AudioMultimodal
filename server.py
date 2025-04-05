import os
import io
import torch
import librosa
import requests
import aiofiles
from hashlib import md5
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# Set offline mode externally:
#   export HF_HUB_OFFLINE=1

app = FastAPI(title="Qwen2-Audio Inference API")

# Directory to save uploaded audio files
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
AUDIO_SAVE_DIR = os.path.join(BASE_DIR, "audio_files")
os.makedirs(AUDIO_SAVE_DIR, exist_ok=True)

# Path to your local clone of Qwen2-Audio-7B-Instruct (downloaded via Git LFS)
MODEL_PATH = os.path.join(BASE_DIR, "Qwen2-Audio-7B-Instruct")

# Force resampling to 16 kHz
SAMPLING_RATE = 16000

# Use one GPU: Force model to load on cuda:0 by setting a manual device map.
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch_dtype,
    device_map={"": "cuda:0"} if torch.cuda.is_available() else None,
    trust_remote_code=True,
    local_files_only=True
)
# Tie weights to suppress warnings
model.tie_weights()

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True
)

# Global cache for identical audio (optional)
cached_audio_hash = None
cached_audio_data = None

def slice_prompt_from_generation(generated_ids, input_ids_length):
    """Remove the prompt portion from the generated output."""
    return generated_ids[:, input_ids_length:]

async def save_upload_file(upload_file: UploadFile, destination: str):
    """Save an uploaded file asynchronously to disk."""
    try:
        async with aiofiles.open(destination, 'wb') as out_file:
            content = await upload_file.read()
            await out_file.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {e}")

def load_audio_from_disk(file_path: str) -> torch.Tensor:
    """Load an audio file from disk and resample it to SAMPLING_RATE."""
    try:
        audio_data, _ = librosa.load(file_path, sr=SAMPLING_RATE)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading audio from disk: {e}")
    return audio_data

def load_audio_from_url(audio_url: str) -> torch.Tensor:
    """Download an audio file from a URL and load/resample it."""
    try:
        r = requests.get(audio_url)
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading audio from URL: {e}")
    try:
        audio_data, _ = librosa.load(io.BytesIO(r.content), sr=SAMPLING_RATE)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing downloaded audio: {e}")
    return audio_data

# ----- Web Demo Endpoint -----
@app.get("/")
def read_root():
    """Return an HTML form for audio file upload and text prompt."""
    return HTMLResponse(
        """
        <html>
          <head>
            <title>Qwen2-Audio 7B Instruct Demo</title>
          </head>
          <body>
            <h1>Qwen2-Audio 7B Instruct Demo</h1>
            <p>Upload an audio file (.wav, .mp3, .flac) and optionally provide a text prompt.</p>
            <form action="/transcribe" method="post" enctype="multipart/form-data">
              <label for="audio">Audio File:</label>
              <input type="file" id="audio" name="audio" accept=".wav,.mp3,.flac" required><br><br>
              <label for="prompt">Text Prompt (optional):</label>
              <input type="text" id="prompt" name="prompt" value="Generate the caption in English:"><br><br>
              <input type="submit" value="Transcribe">
            </form>
          </body>
        </html>
        """
    )

@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    prompt: str = Form("Generate the caption in English:")
):
    """
    Endpoint to process an uploaded audio file.
    Saves the file to disk, loads & resamples it to 16 kHz, waits until generation is complete,
    and returns the transcription.
    """
    global cached_audio_hash, cached_audio_data

    # Save uploaded file to disk in the designated directory
    save_path = os.path.join(AUDIO_SAVE_DIR, audio.filename)
    await save_upload_file(audio, save_path)

    # Read file bytes from disk (for caching)
    try:
        with open(save_path, "rb") as f:
            file_bytes = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading saved file: {e}")

    new_hash = md5(file_bytes).hexdigest()
    if new_hash == cached_audio_hash and cached_audio_data is not None:
        audio_data = cached_audio_data
    else:
        cached_audio_hash = new_hash
        audio_data = load_audio_from_disk(save_path)
        cached_audio_data = audio_data

    # Build full prompt with required tokens
    full_prompt = f"<|audio_bos|><|AUDIO|><|audio_eos|> {prompt}"
    try:
        inputs = processor(
            text=full_prompt,
            audios=audio_data,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
            padding=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing inputs: {e}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device)

    try:
        # Generate output with max_new_tokens=60000 and wait for completion
        generate_ids = model.generate(**inputs, max_new_tokens=60000)
        # Remove the prompt portion from the output tokens
        generate_ids = slice_prompt_from_generation(generate_ids, inputs.input_ids.shape[1])
        transcription = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during generation: {e}")

    return {"transcription": transcription}

# ----- JSON API Endpoint -----
class TranscriptionRequest(BaseModel):
    prompt: str = "Generate the caption in English:"
    audio_url: str = None  # Optional URL to an audio file (.wav, .mp3, .flac)

@app.post("/api/transcribe")
def api_transcribe(request: TranscriptionRequest):
    """
    JSON endpoint that accepts a text prompt and an optional audio URL.
    Downloads the audio if provided, resamples to 16 kHz, waits until full generation is complete,
    and returns the transcription.
    """
    audio_data = None
    if request.audio_url:
        audio_data = load_audio_from_url(request.audio_url)

    if audio_data is not None:
        full_prompt = f"<|audio_bos|><|AUDIO|><|audio_eos|> {request.prompt}"
        try:
            inputs = processor(
                text=full_prompt,
                audios=audio_data,
                sampling_rate=SAMPLING_RATE,
                return_tensors="pt",
                padding=True
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error preparing inputs with audio: {e}")
    else:
        try:
            inputs = processor(text=request.prompt, return_tensors="pt")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error preparing text-only inputs: {e}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device)

    try:
        generate_ids = model.generate(**inputs, max_new_tokens=60000)
        generate_ids = slice_prompt_from_generation(generate_ids, inputs["input_ids"].shape[1])
        output_text = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during generation: {e}")

    return {"response": output_text}

