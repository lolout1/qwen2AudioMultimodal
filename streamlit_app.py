import os
import io
import hashlib
import streamlit as st
import torch
import librosa
import requests
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# --- Offline & Single-GPU Setup ---
# Ensure you've downloaded your model files (via Git LFS) into the directory below.
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "QwenTestYapper", "qwen2_audio_demo"))
MODEL_PATH = os.path.join(BASE_DIR, "Qwen2-Audio-7B-Instruct")
# For offline/firewalled environments, set the environment variable HF_HUB_OFFLINE=1 before launching.

# Force resampling to 16 kHz.
SAMPLING_RATE = 16000

# Force the model to use one GPU: we'll load the model onto cuda:0 if available.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def debug_print(msg: str):
    print(f"[DEBUG] {msg}")

@st.cache_resource(show_spinner=False)
def load_model_and_processor():
    debug_print("Loading model and processor from local files...")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_dtype,
        device_map={"": "cuda:0"} if torch.cuda.is_available() else None,
        trust_remote_code=True,
        local_files_only=True
    )
    model.tie_weights()
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True
    )
    debug_print("Model and processor loaded successfully.")
    return model, processor

model, processor = load_model_and_processor()

# --- File Saving & Audio Loading Utilities ---
AUDIO_SAVE_DIR = os.path.join(BASE_DIR, "audio_files")
os.makedirs(AUDIO_SAVE_DIR, exist_ok=True)
debug_print(f"Audio files will be saved to: {AUDIO_SAVE_DIR}")

def compute_md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()

def load_audio_from_bytes(audio_bytes: bytes) -> list:
    debug_print("Loading audio from bytes with sampling rate " + str(SAMPLING_RATE))
    try:
        audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLING_RATE)
        debug_print(f"Audio loaded successfully, shape: {len(audio_data)} samples, sr: {sr}")
        return audio_data
    except Exception as e:
        debug_print(f"Error in load_audio_from_bytes: {e}")
        st.error(f"Error processing audio: {e}")
        return None

def load_audio_from_url(audio_url: str) -> list:
    debug_print(f"Downloading audio from URL: {audio_url}")
    try:
        r = requests.get(audio_url)
        r.raise_for_status()
    except Exception as e:
        debug_print(f"Error downloading audio: {e}")
        st.error(f"Error downloading audio: {e}")
        return None
    return load_audio_from_bytes(r.content)

def save_uploaded_file(upload_file, dest_path: str):
    debug_print(f"Saving uploaded file to: {dest_path}")
    try:
        with open(dest_path, "wb") as f:
            f.write(upload_file.getvalue())
        debug_print("File saved successfully.")
    except Exception as e:
        debug_print(f"Error saving file: {e}")
        st.error(f"Error saving file: {e}")

def slice_prompt_from_generation(generated_ids, prompt_len: int):
    debug_print(f"Slicing prompt tokens (length {prompt_len}) from generation output.")
    return generated_ids[:, prompt_len:]

# Global cache to avoid reprocessing identical audio files
cached_audio_hash = None
cached_audio_data = None

def transcribe(audio_data, prompt: str) -> str:
    full_prompt = f"<|audio_bos|><|AUDIO|><|audio_eos|> {prompt}"
    debug_print(f"Full prompt: {full_prompt}")
    try:
        inputs = processor(
            text=full_prompt,
            audios=audio_data,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
            padding=True
        )
        debug_print("Inputs prepared successfully.")
    except Exception as e:
        debug_print(f"Error preparing inputs: {e}")
        st.error(f"Error preparing inputs: {e}")
        return ""
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device)
    try:
        debug_print("Generating transcription with max_new_tokens=60000...")
        generated_ids = model.generate(**inputs, max_new_tokens=60000)
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
        debug_print(f"Error during generation: {e}")
        st.error(f"Error during generation: {e}")
        return ""

# --- Streamlit UI ---
st.title("Qwen2-Audio-Instruct Transcription")

st.markdown("### Choose Input Method")
input_method = st.radio("Select input method", ("Upload Audio File", "Provide Audio URL"))

audio_data = None

if input_method == "Upload Audio File":
    uploaded_file = st.file_uploader("Choose an audio file (.wav, .mp3, .flac)", type=["wav", "mp3", "flac"])
    if uploaded_file is not None:
        save_path = os.path.join(AUDIO_SAVE_DIR, uploaded_file.name)
        save_uploaded_file(uploaded_file, save_path)
        debug_print(f"Uploaded file saved as: {save_path}")
        file_bytes = uploaded_file.getvalue()
        file_hash = compute_md5(file_bytes)
        debug_print(f"Computed MD5 hash: {file_hash}")
        global cached_audio_hash, cached_audio_data
        if file_hash == cached_audio_hash and cached_audio_data is not None:
            debug_print("Using cached audio data.")
            audio_data = cached_audio_data
        else:
            debug_print("Processing new audio file...")
            audio_data = load_audio_from_bytes(file_bytes)
            cached_audio_hash = file_hash
            cached_audio_data = audio_data
elif input_method == "Provide Audio URL":
    audio_url = st.text_input("Enter audio URL", "")
    if audio_url:
        audio_data = load_audio_from_url(audio_url)

prompt = st.text_input("Enter text prompt", "Generate the caption in English:")

if st.button("Transcribe"):
    if audio_data is None:
        st.error("No audio data provided.")
    else:
        st.info("Transcribing... please wait until complete.")
        transcription = transcribe(audio_data, prompt)
        st.subheader("Transcription:")
        st.write(transcription)

