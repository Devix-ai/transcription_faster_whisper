import os
import whisper
from whisper import load_model
import yt_dlp
import subprocess
import requests
from urllib.parse import urlsplit
from pathlib import Path
from IPython.display import display, Markdown
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# **Model selection**
model_size = 'large'  # Options ['tiny', 'base', 'small', 'medium', 'large']

# **Video selection**
Type = "Local"  # Options ['Youtube video or playlist', 'Local', 'Direct download']

# **Youtube video or playlist**
URL = "https://www.youtube.com/watch?v=9ez8lm9I26Y"

# **Local video**
video_path = "C:/Users/DESKTOP/youtube_transcription/MSpSw2cdYVY.mp4"

# **Direct Download**
ddl_url = "https://video-aajtak.tosshub.com/aajtak/video/2024_05/06_may_24_at_nand_gopal_vo_mm_1024_512.mp4"

# Load Whisper model

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model(model_size).to(device)

video_path_local_list = []

def download_youtube_video(url):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': '%(id)s.%(ext)s',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        video_info = ydl.extract_info(url, download=False)
        return Path(f"{video_info['id']}.mp4")

def download_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        filename = urlsplit(url).path.split("/")[-1]
        destination_path = os.path.join(os.getcwd(), filename)
        with open(destination_path, 'wb') as file:
            file.write(response.content)
        return Path(destination_path)
    else:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

if Type == "Youtube video or playlist":
    video_path_local_list.append(download_youtube_video(URL))

elif Type == "Local":
    video_path = Path(video_path)
    if video_path.is_file():
        video_path_local_list.append(video_path)
    else:
        raise FileNotFoundError(f"{str(video_path)} does not exist.")

elif Type == "Direct download":
    video_path_local_list.append(download_file(ddl_url))

else:
    raise TypeError("Please select a supported input type.")

def convert_video_to_audio(video_path):
    valid_suffixes = [".mp4", ".mkv", ".mov", ".avi", ".wmv", ".flv", ".webm", ".3gp", ".mpeg"]
    if video_path.suffix in valid_suffixes:
        audio_path = video_path.with_suffix(".mp3")
        subprocess.run(["ffmpeg", "-i", str(video_path), "-vn", "-acodec", "libmp3lame", "-q:a", "2", str(audio_path)])
        return audio_path
    else:
        raise ValueError(f"Unsupported video format: {video_path.suffix}")

def seconds_to_time_format(s):
    hours = s // 3600
    s %= 3600
    minutes = s // 60
    s %= 60
    seconds = s // 1
    milliseconds = round((s % 1) * 1000)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"

language = "en"
word_level_timestamps = False
vad_filter = True
vad_filter_min_silence_duration_ms = 50
text_only = True

for video_path_local in video_path_local_list:
    audio_path = convert_video_to_audio(video_path_local)
    result = model.transcribe(
        str(audio_path),
        language=language if language != "auto" else None,
    )
    
    segments = result['segments']
    print(result["text"])

    ext_name = '.txt' if text_only else ".srt"
    output_file_name = audio_path.stem + ext_name
    sentence_idx = 1
    with open(output_file_name, 'w', encoding='utf-8') as f:
        for segment in segments:
            try:
                if word_level_timestamps:
                    for word in segment['words']:
                        try:
                            ts_start = seconds_to_time_format(word['start'])
                            ts_end = seconds_to_time_format(word['end'])
                            print(f"[{ts_start} --> {ts_end}] {word['text']}")
                            if not text_only:
                                f.write(f"{sentence_idx}\n")
                                f.write(f"{ts_start} --> {ts_end}\n")
                                f.write(f"{word['word']}\n\n")
                            else:
                                f.write(f"{word['word']}\n")
                            sentence_idx += 1
                        except UnicodeEncodeError:
                            print(f"Skipping word with encoding issue: {word['text']}")
                            continue
                else:
                    ts_start = seconds_to_time_format(segment['start'])
                    ts_end = seconds_to_time_format(segment['end'])
                    print(f"[{ts_start} --> {ts_end}] {segment['text']}")
                    if not text_only:
                        f.write(f"{sentence_idx}\n")
                        f.write(f"{ts_start} --> {ts_end}\n")
                        f.write(f"{segment['text'].strip()}\n\n")
                    else:
                        f.write(f"{segment['text'].strip()}\n")
                    sentence_idx += 1
            except UnicodeEncodeError:
                print(f"Skipping segment with encoding issue: {segment['text']}")
                continue