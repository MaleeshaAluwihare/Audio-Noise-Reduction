import streamlit as st
from df.enhance import enhance, init_df, load_audio
from io import BytesIO
import torchaudio

# Initialize DeepFilterNet model
model, df_state, _ = init_df()

# Streamlit App
st.title("Audio Noise Removal")
st.write("Upload an audio file to remove background noise.")

uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav", start_time=0)
    st.write("Processing your audio...")

    audio, _ = load_audio(uploaded_file, sr=df_state.sr())

    enhanced_audio = enhance(model, df_state, audio)

    # Ensure enhanced_audio is 2D
    if len(enhanced_audio.shape) == 1:
        enhanced_audio = enhanced_audio.unsqueeze(0)

    buffer = BytesIO()
    torchaudio.save(buffer, enhanced_audio, sample_rate=df_state.sr(), format="wav")
    buffer.seek(0)

    st.audio(buffer, format="audio/wav")
    st.download_button(
        label="Download Enhanced Audio",
        data=buffer,
        file_name="enhanced_audio.wav",
        mime="audio/wav",
    )
