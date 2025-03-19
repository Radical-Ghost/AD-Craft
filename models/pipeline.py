import yt_dlp
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import torch
import clip
import cv2
from PIL import Image
import whisper
import subprocess

# Ensure correct device (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Load the CLIP model and tokenizer
model, preprocess = clip.load("ViT-B/32", device=device)

def download_video(url, output_path="input_video.mp4"):
    options = {'format': 'best', 'outtmpl': output_path}
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([url])
    print(f"Video downloaded: {output_path}")

link = input("Enter a video link: ")

# Example usage
download_video(link)

def detect_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scenes = scene_manager.get_scene_list()
    print(f"Detected {len(scenes)} scenes")
    return scenes

# Example usage
scenes = detect_scenes("input_video.mp4")

def rank_scenes(video_path, scenes, keywords):
    scene_scores = []
    for i, (start, end) in enumerate(scenes):
        # Convert FrameTimecode to integers (frame numbers)
        start_frame = start.get_frames()
        end_frame = end.get_frames()

        # Capture the middle frame of each scene
        cap = cv2.VideoCapture(video_path)
        mid_frame = (start_frame + end_frame) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            continue

        # Preprocess the frame and compute similarity with keywords
        image = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
        text_inputs = clip.tokenize(keywords).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text_inputs)
            similarity = (image_features @ text_features.T).softmax(dim=-1)

        scene_scores.append((i, similarity.max().item()))

    # Sort scenes by relevance (highest similarity first)
    scene_scores.sort(key=lambda x: x[1], reverse=True)
    return scene_scores

# Example usage
keywords = ["ketchup", "product demo", "heinz"]
ranked_scenes = rank_scenes("input_video.mp4", scenes, keywords)
print(ranked_scenes[:5])  # Top 5 most relevant scenes

def transcribe_video(video_path):
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    print("Transcript: ", result['text'])
    return result['text']

# Example usage
transcribe_video("input_video.mp4")

def trim_video(input_path, output_path, start_time, duration):
    command = [
        'ffmpeg',
        '-i', input_path,
        '-ss', str(start_time),
        '-t', str(duration),
        '-c:v', 'libx264',
        '-c:a', 'aac',
        output_path
    ]
    subprocess.run(command)
    print(f"Trimmed video saved to: {output_path}")

# Example usage: Trim a 15-second highlight
trim_video("input_video.mp4", "highlight.mp4", 30, 15)
