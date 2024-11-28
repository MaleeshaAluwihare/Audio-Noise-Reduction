import sounddevice as sd
import numpy as np
import wave
from DeepFilterNet.DeepFilterNet.df.deepfilternet import DeepFilterNet

#initialize deepFilterNet
model_path = "./DeepFilterNet/models/DeepFilterNet2_extracted/checkpoints"
deep_filter_net = DeepFilterNet(model_path=model_path)

#Audio parameters
RATE = 16000
CHANNELS = 1
CHUNK = 1024
OUTPUT_FILE = "filtered_output.wav"

#Prepare to save the output
wf = wave.open(OUTPUT_FILE,"wb")
wf.setnchannels(CHANNELS)
wf.setsampwidth(2)
wf.setframerate(RATE)

def audio_callback(indata, frames, time, status):
    """
    callback function for processing audio in real-time
    """
    if status:
        print(f"Stream status: {status}")
    print("Processing audio...")

    #convert input audio to numpy array
    audio_chunck = indata[:,0]
    audio_chunck = (audio_chunck * 32767).astype(np.int16)

    #Apply DeepFilterNet noise reduction
    enhanced_chunk = deep_filter_net.process(audio_chunck)

    #write filtered audio to the wave file
    wf.writeframes(enhanced_chunk.tobytes())

print("Recording and filtering in real-time... Press Ctrl+c to stop")
try:
    #start the audio stram
    with sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype="float32", callback=audio_callback, blocksize=CHUNK):
        while True:
            sd.sleep(1000)

except KeyboardInterrupt:
    print("\nRecording Stopped")

finally:
    wf.close()