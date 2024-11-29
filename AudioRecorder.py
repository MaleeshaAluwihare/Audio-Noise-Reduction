import numpy as np
import wave
import sounddevice as sd
import streamlit as st
from df.enhance import enhance, init_df
from tempfile import NamedTemporaryFile

# Initialize DeepFilterNet model and state
model, df_state, _ = init_df()

# Audio parameters
RATE = 16000
CHANNELS = 1
CHUNK = 1024

# Buffer to accumulate enhanced audio
enhanced_audio = []

# Streamlit UI setup
st.title("Real-Time Audio Noise Reduction")
st.write("Click the button below to start recording and denoising the audio.")

# Temporary file for saving output audio
output_file = NamedTemporaryFile(delete=False, suffix=".wav")

# Streamlit button to start recording
start_button = st.button("Start Recording")

# Function to handle the audio callback
def audio_callback(indata, frames, time, status):
    """Callback function for processing audio in real-time."""
    if status:
        print(f"Stream status: {status}")
    
    # Convert input audio to numpy array (from float32 to int16)
    audio_chunk = indata[:, 0]
    audio_chunk = (audio_chunk * 32767).astype(np.int16)

    # Enhance the audio using DeepFilterNet (real-time denoising)
    enhanced_chunk = enhance(model, df_state, audio_chunk)

    # Accumulate the enhanced chunk to the buffer
    enhanced_audio.append(enhanced_chunk)

# If the user clicks "Start Recording"
if start_button:
    st.write("Recording started... Press 'Stop Recording' to stop.")
    
    try:
        with sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype="float32", callback=audio_callback, blocksize=CHUNK):
            st.button("Stop Recording", key="stop_button")
    except KeyboardInterrupt:
        st.write("\nRecording Stopped")
    
    finally:
        # Concatenate the enhanced audio buffer into a single numpy array
        final_enhanced_audio = np.concatenate(enhanced_audio)

        # Save the enhanced audio to the temporary file
        with wave.open(output_file.name, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 2 bytes for 16-bit audio
            wf.setframerate(RATE)
            wf.writeframes(final_enhanced_audio.tobytes())

        st.write(f"Enhanced audio saved to {output_file.name}")

        # Provide download link for the enhanced audio file
        st.audio(output_file.name, format="audio/wav")
        st.download_button("Download Enhanced Audio", output_file.name)

else:
    st.write("Press 'Start Recording' to begin.")
