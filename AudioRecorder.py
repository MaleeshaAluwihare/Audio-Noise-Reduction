import numpy as np
import wave
import torch
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
# Initialize session state for recording if it does not exist
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
    
# Streamlit button to start recording
start_button = st.button("Start Recording")

# Flag to stop recording
stop_button = st.button("Stop Recording", key="stop_button")

# Function to handle the audio callback
def audio_callback(indata, frames, time, status):
    """Callback function for processing audio in real-time."""
    if status:
        print(f"Stream status: {status}")
    
    # Convert input audio to numpy array (from float32 to int16)
    audio_chunk = indata[:, 0]
    audio_chunk = (audio_chunk * 32767).astype(np.int16)

    # Convert audio chunk to tensor, ensuring it's 2D: (1, N)
    audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32)
    audio_tensor = audio_tensor.unsqueeze(0)  # Shape: (1, N)

    # Enhance the audio using DeepFilterNet (real-time denoising)
    enhanced_chunk = enhance(model, df_state, audio_tensor)

    # Accumulate the enhanced chunk to the buffer
    enhanced_audio.append(enhanced_chunk)


# If the user clicks "Start Recording"
if start_button and not st.session_state.is_recording:
    st.session_state.is_recording = True
    st.write("Recording started... Press 'Stop Recording' to stop.")
    
    try:
        with sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype="float32", callback=audio_callback, blocksize=CHUNK):
            # Run the stream until the stop button is pressed
            while st.session_state.is_recording:
                sd.sleep(1000)
                
    except KeyboardInterrupt:
        st.write("\nRecording Stopped")
    
elif stop_button and st.session_state.is_recording:
    # Stop the recording when the stop button is clicked
    st.session_state.is_recording = False
    st.write("Recording stopped. Processing the enhanced audio...")

    # Check if we have any enhanced audio to save
    if enhanced_audio:
        # Concatenate the enhanced audio buffer into a single numpy array
        try:
            final_enhanced_audio = np.concatenate(enhanced_audio)
        except ValueError:
            st.write("Error: No valid audio data was collected.")
            final_enhanced_audio = None

        if final_enhanced_audio is not None:
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
        st.write("No audio was recorded.")