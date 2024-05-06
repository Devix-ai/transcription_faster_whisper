import os
import sys
from faster_whisper import WhisperModel
import yt_dlp
import subprocess
#import torch
import shutil
import numpy as np
from IPython.display import display, Markdown, YouTubeVideo
import requests
from urllib.parse import urlsplit
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from pathlib import Path


 # **Model selection** 

#|  Size    | Parameters | English-only model | Multilingual model | Required VRAM   | Relative speed |
#|:------:  |:----------:|:------------------:|:------------------:|:---------------:|:--------------:|
#|  tiny    |    39 M    |     `tiny.en`      |       `tiny`       |     ~0.8 GB     |      ~32x      |
#|  base    |    74 M    |     `base.en`      |       `base`       |     ~1.0 GB     |      ~16x      |
#| small    |   244 M    |     `small.en`     |      `small`       |     ~1.4 GB     |      ~6x       |
#| medium   |   769 M    |    `medium.en`     |      `medium`      |     ~2.7 GB     |      ~2x       |
#| large-v1 |   1550 M   |        N/A         |      `large-v1`    |     ~4.3 GB     |       1x       |
#| large-v2 |   1550 M   |        N/A         |      `large-v2`    |     ~4.3 GB     |       1x       |


model_size = 'medium'      # Options ['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2']
device_type = "cuda"         # Options ['cuda', 'cpu']
compute_type = "float16"     # Options ['float16', 'int8_float16', 'int8']


# **Video selection** 📺

Type = "Youtube video or playlist"  # Options ['Youtube video or playlist', 'Local', 'Direct download']

# **Youtube video or playlist**
URL = "https://www.youtube.com/watch?v=MPC-3ENpvFw" 


#  **Local video**
video_path = "C:/Users/DESKTOP/youtube_transcription/VID-20210607-WA0034.mp4" 

# **Direct Download**
ddl_url = "https://video-aajtak.tosshub.com/aajtak/video/2024_05/06_may_24_at_nand_gopal_vo_mm_1024_512.mp4" 


model = WhisperModel(model_size, device=device_type, compute_type=compute_type)

video_path_local_list = []

if Type == "Youtube video or playlist":

    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'outtmpl': '%(id)s.%(ext)s',
        # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
        'postprocessors': [{  # Extract audio using ffmpeg
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download([URL])
        list_video_info = [ydl.extract_info(URL, download=False)]

    for video_info in list_video_info:
        video_path_local_list.append(Path(f"{video_info['id']}.mp3"))

elif Type == "Local":
    video_path = Path(video_path)
    if video_path.is_dir():
        for video_file in video_path.glob("**/*"):
            if video_file.is_file():
                display(Markdown(f"**{str(video_file)} selected for processing.**"))
                video_path_local_list.append(video_file)
            elif video_file.is_dir():
                display(Markdown(f"**Subfolders not supported.**"))
            else:
                display(Markdown(f"**{str(video_file)} does not exist, skipping.**"))
    elif video_path.is_file():
        display(Markdown(f"**{str(video_path)} selected for processing.**"))
        video_path_local_list.append(video_path)
    else:
        display(Markdown(f"**{str(video_path)} does not exist.**"))

elif Type == "Direct download":
    print(f"⚠️ Please ensure this is a direct download link and is of a valid format")
    print(f"Attempting to download: {ddl_url}\n")

    response = requests.get(ddl_url)

    if response.status_code == 200:
        # Extract the filename from the URL
        filename = urlsplit(ddl_url).path.split("/")[-1]

        # Create the full path for the destination file in the current working directory
        destination_path = os.path.join(os.getcwd(), filename)

        # Save the file
        with open(destination_path, 'wb') as file:
            file.write(response.content)

        print(f"File downloaded successfully: {destination_path}")

        video_path_local = Path(".").resolve() / (filename)

        # print(f"Path local: {video_path_local}") # /content/video.mkv

        video_path_local_list.append(video_path_local)
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

else:
    raise(TypeError("Please select supported input type."))

for video_path_local in video_path_local_list:
    valid_suffixes = [".mp4", ".mkv", ".mov", ".avi", ".wmv", ".flv", ".webm", ".3gp", ".mpeg"]

    print(f"Processing video file {video_path_local} with ffmpeg..")

    if video_path_local.suffix in valid_suffixes:
        input_suffix = video_path_local.suffix
        video_path_local = video_path_local.with_suffix(".wav")
        result = subprocess.run(["ffmpeg", "-i", str(video_path_local.with_suffix(input_suffix)), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(video_path_local)])

def seconds_to_time_format(s):
    # Convert seconds to hours, minutes, seconds, and milliseconds
    hours = s // 3600
    s %= 3600
    minutes = s // 60
    s %= 60
    seconds = s // 1
    milliseconds = round((s % 1) * 1000)

    # Return the formatted string
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"

# Language
language = "en" #Options ["auto", "en", "zh", "ja", "fr", "de"]

initial_prompt = "Please translate this from Japanese to English." 
word_level_timestamps = False 
vad_filter = True 
vad_filter_min_silence_duration_ms = 50 
text_only = False #@param {type:"boolean"}


segments, info = model.transcribe(str(video_path_local), beam_size=5,
                                  language=None if language == "auto" else language,
                                  initial_prompt=initial_prompt,
                                  word_timestamps=word_level_timestamps,
                                  vad_filter=vad_filter,
                                  vad_parameters=dict(min_silence_duration_ms=vad_filter_min_silence_duration_ms))

display(Markdown(f"Detected language '{info.language}' with probability {info.language_probability}"))

ext_name = '.txt' if text_only else ".srt"
output_file_name = video_path_local.stem + ext_name
sentence_idx = 1
with open(output_file_name, 'w') as f:
  for segment in segments:
    if word_level_timestamps:
      for word in segment.words:
        ts_start = seconds_to_time_format(word.start)
        ts_end = seconds_to_time_format(word.end)
        print(f"[{ts_start} --> {ts_end}] {word.word}")
        if not text_only:
          f.write(f"{sentence_idx}\n")
          f.write(f"{ts_start} --> {ts_end}\n")
          f.write(f"{word.word}\n\n")
        else:
          f.write(f"{word.word}")
        f.write("\n")
        sentence_idx = sentence_idx + 1
    else:
      ts_start = seconds_to_time_format(segment.start)
      ts_end = seconds_to_time_format(segment.end)
      print(f"[{ts_start} --> {ts_end}] {segment.text}")
      if not text_only:
        f.write(f"{sentence_idx}\n")
        f.write(f"{ts_start} --> {ts_end}\n")
        f.write(f"{segment.text.strip()}\n\n")
      else:
        f.write(f"{segment.text.strip()}\n")
      sentence_idx = sentence_idx + 1
