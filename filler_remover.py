import os
import sys
import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from utils import adjust_pauses_for_hf_pipeline_output

from moviepy.editor import VideoFileClip

def vid_to_audio(video_path, audio_path="extracted_audio.wav"):
    # Using 'with' ensures the file is closed and unlocked properly on Windows
    with VideoFileClip(video_path) as video:
        if video.audio is not None:
            video.audio.write_audiofile(audio_path)
        else:
            raise ValueError("This video file does not contain an audio track.")
    
    return audio_path


def cut_out_ums(response, video_path):
    cuts_to_make = []
    for chunk in response["chunks"]:
        if "[" in chunk['text']:
            
            print("\nremove word (y)?")
            print(chunk)
            q = input()
            if q == "y":
                cuts_to_make.insert(0, chunk["timestamp"])

    clip = VideoFileClip(video_path)
    if hasattr(clip, 'rotation') and clip.rotation in (90, 270):
        clip = clip.resize(clip.size[::-1])
        clip.rotation = 0

    modified_clip = clip
    for start, end in cuts_to_make:
        modified_clip = modified_clip.cutout(start, end)

    modified_clip.write_videofile(
        "output_cutout.mp4",
        ffmpeg_params=['-aspect', '9:16'] 
    )

print("Running uh the program...")

# Set up device and precision
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "nyrahealth/CrisperWhisper"

# Load the model
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch_dtype, 
    low_cpu_mem_usage=True, 
    use_safetensors=True
)
model.to(device)

# Load the processor
processor = AutoProcessor.from_pretrained(model_id)

# Initialize the pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps='word',
    torch_dtype=torch_dtype,
    device=device,
)

vid_path = input("\nInput the location of the video file: ")
# --- Process Custom Video ---
audio_path = "extracted_audio.wav" 
vid_to_audio(vid_path, audio_path)

# librosa bypasses the buggy huggingface file reader. 
# sr=16000 forces the exact sample rate Whisper expects.
audio_array, sampling_rate = librosa.load(audio_path, sr=16000)

audio_duration = len(audio_array) / sampling_rate
# print(f"Loaded {audio_duration:.2f} seconds of audio.")
# print(f"Transcribing on {device}...")

# Pass the raw audio array and sampling rate directly into the pipeline!
inputs = {"array": audio_array, "sampling_rate": sampling_rate}
hf_pipeline_output = pipe(inputs)

# Adjust pauses
crisper_whisper_result = adjust_pauses_for_hf_pipeline_output(hf_pipeline_output)

print("\n--- DONE Transcribing ---")
print(crisper_whisper_result)

cut_out_ums(crisper_whisper_result, "test.mp4")
