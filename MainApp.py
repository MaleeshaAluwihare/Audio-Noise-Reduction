import numpy as np
import wave
import torch
import streamlit as st
from tempfile import NamedTemporaryFile
from df.enhance import enhance, init_df
import os
import soundfile as sf

# # Environment variables for DeepFilterNet2
# os.environ["DF_CHECKPOINT"] = "DeepFilterNet2/checkpoints/model_96.ckpt.best"
# os.environ["DF_CONFIG"] = "DeepFilterNet2/config.ini"

# # Initialize DeepFilterNet2 model and state
# model, df_state, _ = init_df()

from df import config
from df.enhance import enhance, init_df, load_audio, save_audio
from df.io import resample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, df, _ = init_df("./DeepFilterNet2", config_allow_defaults=True)
model = model.to(device=device).eval()

# Streamlit UI
st.title("Audio Noise Reduction")
st.write("Upload an audio file to reduce noise.")

# Audio file upload
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

# Temporary file to save the output
output_file = NamedTemporaryFile(delete=False, suffix=".wav")

if uploaded_file:
    try:
        # Read the uploaded audio file
        st.write("Processing uploaded audio file...")
        audio_data, samplerate = sf.read(uploaded_file)
        print(f"Uploaded audio shape: {audio_data.shape}, Sample rate: {samplerate}")

        # Ensure audio is mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)  # Convert to mono
            print("Converted to mono.")

        # Normalize the audio data to match DeepFilterNet's input expectations
        audio_tensor = torch.tensor(audio_data * 32767, dtype=torch.float32).unsqueeze(0)

        # Process the audio using DeepFilterNet
        enhanced_audio = enhance(model=model, audio=audio_tensor, df_state=df).squeeze().numpy()
        print(f"Enhanced audio shape: {enhanced_audio.shape}")

        # Save the enhanced audio
        with wave.open(output_file.name, "wb") as wf:
            wf.setnchannels(1)  # Mono audio
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(samplerate)
            wf.writeframes(enhanced_audio.astype(np.int16).tobytes())

        # Provide the enhanced audio for download and playback
        st.write("Noise reduction complete.")
        st.audio(output_file.name, format="audio/wav")
        st.download_button("Download Enhanced Audio", data=open(output_file.name, "rb"), file_name="enhanced_audio.wav")
    except Exception as e:
        st.error("An error occurred while processing the audio file.")
        print(f"Processing error: {e}")