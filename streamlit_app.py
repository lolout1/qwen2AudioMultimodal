import streamlit as st
import torch
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

@st.cache(allow_output_mutation=True)
def load_model(model_path):
    # Use torch.float16 if CUDA is available and you prefer FP16 inference
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor

# Replace with your local path to the cloned qwen2-audio-7b-instruct model
MODEL_PATH = "./Qwen2-Audio-7B-Instruct"
model, processor = load_model(MODEL_PATH)

st.title("Qwen2-Audio 7B Instruct Demo")
st.markdown("Upload an audio file and provide an optional text prompt to generate a response.")

# Audio file uploader (Streamlit will automatically create a temporary file)
audio_file = st.file_uploader("Upload Audio File", type=["wav", "flac", "mp3"])
text_prompt = st.text_input("Optional Text Prompt", value="Generate the caption in English:")

if st.button("Run Inference"):
    if audio_file is None:
        st.error("Please upload an audio file to proceed.")
    else:
        # Load audio file with librosa using the sampling rate from the processor's feature extractor
        sampling_rate = processor.feature_extractor.sampling_rate
        try:
            # librosa.load accepts a file-like object
            audio, _ = librosa.load(audio_file, sr=sampling_rate)
        except Exception as e:
            st.error(f"Error loading audio: {e}")
            st.stop()
        
        # Create a prompt combining the audio and the optional text.
        # The prompt format may need to match what the model expects.
        prompt = f"<|audio_bos|><|AUDIO|><|audio_eos|> {text_prompt}"
        
        # Process inputs using the processor
        inputs = processor(text=prompt, audios=audio, return_tensors="pt", padding=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        st.info("Generating response...")
        with st.spinner("Processing..."):
            generate_ids = model.generate(**inputs, max_length=256)
            # Remove special tokens and clean up the output
            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        st.success("Response generated!")
        st.write("**Model Output:**")
        st.write(response)

