import os
import shutil
import uuid
import yt_dlp
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import torch
import clip
import cv2
import subprocess
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoFileClip, concatenate_videoclips
import random
import streamlit as st

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def download_video(url, output_path=None):
    if output_path is None:
        output_path = f"temp/input_video_{uuid.uuid4().hex}.mp4"
    options = {'format': 'best', 'outtmpl': output_path}
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([url])
    return output_path

def detect_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    scenes = [(start.get_frames(), end.get_frames()) for start, end in scene_list]
    return scenes

def capture_frame(video_path, start_frame, end_frame):
    cap = cv2.VideoCapture(video_path)
    mid_frame = (start_frame + end_frame) // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def rank_scenes(video_path, scenes, keywords, batch_size=32, max_scenes=100):
    scene_scores = []
    text_inputs = clip.tokenize(keywords).to(device)
    scenes = scenes[:max_scenes]
    frames = [capture_frame(video_path, start, end) for start, end in scenes]
    frames = [frame for frame in frames if frame is not None]

    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        images = torch.stack([preprocess(Image.fromarray(frame)) for frame in batch]).to(device)
        with torch.no_grad():
            image_features = model.encode_image(images)
            text_features = model.encode_text(text_inputs)
            similarities = (image_features @ text_features.T).softmax(dim=-1)
        for j, similarity in enumerate(similarities):
            best_score = similarity.max().item()
            scene_idx = i + j
            start, end = scenes[scene_idx]
            duration = (end - start) / 30
            scene_scores.append((scene_idx, best_score, start, end, duration))

    scene_scores.sort(key=lambda x: x[1], reverse=True)
    return scene_scores

def select_scenes_for_duration_v1(ranked_scenes, target_length, fps, min_gap=30, min_scenes=10):
    """
    Variation 1: Prioritize most relevant scenes.
    """
    selected_scenes = []
    total_duration = 0.0

    ranked_scenes.sort(key=lambda x: x[1], reverse=True)  # Sort by relevance

    for scene in ranked_scenes:
        i, score, start, end, _ = scene
        duration = (end - start) / fps

        if selected_scenes:
            last_end_time = selected_scenes[-1][3] / fps
            current_start_time = start / fps
            if current_start_time - last_end_time < min_gap:
                continue

        selected_scenes.append((i, score, start, end, duration))
        total_duration += duration

        if total_duration >= target_length:
            break

    return selected_scenes

def select_scenes_for_duration_v3(ranked_scenes, target_length, fps, min_gap=30, min_scenes=10):
    """
    Variation 3: Randomly select scenes.
    """
    selected_scenes = []
    total_duration = 0.0

    random.shuffle(ranked_scenes)  # Shuffle scenes randomly

    for scene in ranked_scenes:
        i, score, start, end, _ = scene
        duration = (end - start) / fps

        if selected_scenes:
            last_end_time = selected_scenes[-1][3] / fps
            current_start_time = start / fps
            if current_start_time - last_end_time < min_gap:
                continue

        selected_scenes.append((i, score, start, end, duration))
        total_duration += duration

        if total_duration >= target_length:
            break

    return selected_scenes

def select_scenes_for_duration_v4(ranked_scenes, target_length, fps, min_gap=30, min_scenes=10):
    """
    Variation 4: Mix of relevance and duration.
    """
    selected_scenes = []
    total_duration = 0.0

    ranked_scenes.sort(key=lambda x: (x[1], x[4]), reverse=True)  # Sort by relevance and duration

    for scene in ranked_scenes:
        i, score, start, end, _ = scene
        duration = (end - start) / fps

        if selected_scenes:
            last_end_time = selected_scenes[-1][3] / fps
            current_start_time = start / fps
            if current_start_time - last_end_time < min_gap:
                continue

        selected_scenes.append((i, score, start, end, duration))
        total_duration += duration

        if total_duration >= target_length:
            break

    return selected_scenes

def select_scenes_for_duration_v5(ranked_scenes, target_length, fps, min_gap=30, min_scenes=10):
    """
    Variation 5: Prioritize scenes with the highest relevance and shortest duration.
    """
    selected_scenes = []
    total_duration = 0.0

    # Sort by relevance (descending) and duration (ascending)
    ranked_scenes.sort(key=lambda x: (x[1], -x[4]), reverse=True)

    for scene in ranked_scenes:
        i, score, start, end, _ = scene
        duration = (end - start) / fps

        if selected_scenes:
            last_end_time = selected_scenes[-1][3] / fps
            current_start_time = start / fps
            if current_start_time - last_end_time < min_gap:
                continue

        selected_scenes.append((i, score, start, end, duration))
        total_duration += duration

        if total_duration >= target_length:
            break

    return selected_scenes

def select_scenes_for_duration_v6(ranked_scenes, target_length, fps, min_gap=30, min_scenes=10):
    """
    Variation 6: Prioritize scenes with the highest relevance and longest duration.
    """
    selected_scenes = []
    total_duration = 0.0

    # Sort by relevance (descending) and duration (descending)
    ranked_scenes.sort(key=lambda x: (x[1], x[4]), reverse=True)

    for scene in ranked_scenes:
        i, score, start, end, _ = scene
        duration = (end - start) / fps

        if selected_scenes:
            last_end_time = selected_scenes[-1][3] / fps
            current_start_time = start / fps
            if current_start_time - last_end_time < min_gap:
                continue

        selected_scenes.append((i, score, start, end, duration))
        total_duration += duration

        if total_duration >= target_length:
            break

    return selected_scenes

def select_scenes_for_duration_v7(ranked_scenes, target_length, fps, min_gap=30, min_scenes=10):
    """
    Variation 7: Prioritize scenes with the highest relevance and random duration.
    """
    selected_scenes = []
    total_duration = 0.0

    # Sort by relevance (descending)
    ranked_scenes.sort(key=lambda x: x[1], reverse=True)

    # Shuffle scenes randomly
    random.shuffle(ranked_scenes)

    for scene in ranked_scenes:
        i, score, start, end, _ = scene
        duration = (end - start) / fps

        if selected_scenes:
            last_end_time = selected_scenes[-1][3] / fps
            current_start_time = start / fps
            if current_start_time - last_end_time < min_gap:
                continue

        selected_scenes.append((i, score, start, end, duration))
        total_duration += duration

        if total_duration >= target_length:
            break

    return selected_scenes


def trim_scene(input_video, output_path, start_frame, end_frame, fps):
    start_time = start_frame / fps
    end_time = end_frame / fps
    command = [
        'ffmpeg',
        '-i', input_video,
        '-ss', str(start_time),
        '-to', str(end_time),
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-y', output_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def merge_videos(video_list, output_path):
    with open("file_list.txt", "w") as f:
        for video in video_list:
            f.write(f"file '{video}'\n")
    command = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', 'file_list.txt',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-c:a', 'aac',
        output_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def create_ad(input_video, selected_scenes, output_ad_path):
    temp_folder = "temp_clips"
    os.makedirs(temp_folder, exist_ok=True)
    video_fps = get_video_fps(input_video)
    trimmed_videos = []
    for idx, (i, score, start_frame, end_frame, duration) in enumerate(selected_scenes):
        output_clip = os.path.join(temp_folder, f"clip_{idx + 1}.mp4")
        trim_scene(input_video, output_clip, start_frame, end_frame, video_fps)
        trimmed_videos.append(output_clip)
    merge_videos(trimmed_videos, output_ad_path)

def create_cta_image(cta_text, output_image_path, text_size=50, font_path="arial.ttf", font_color="#FFFFFF", font_weight=0, size=(1280, 720)):
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("Size must be a list or tuple of length 2 (width, height).")
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(font_path, text_size)
    except IOError:
        font = ImageFont.load_default()
    bbox = font.getbbox(cta_text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
    font_rgb = tuple(int(font_color[i:i+2], 16) for i in (1, 3, 5))
    if font_weight > 0:
        for x in range(-font_weight, font_weight + 1):
            for y in range(-font_weight, font_weight + 1):
                draw.text((text_position[0] + x, text_position[1] + y), cta_text, font=font, fill=(0, 0, 0, 255))
    draw.text(text_position, cta_text, font=font, fill=font_rgb + (255,))
    image.save(output_image_path, "PNG")

def create_cta_video(cta_image_path, output_video_path, duration=5):
    command = [
        'ffmpeg',
        '-loop', '1',
        '-i', cta_image_path,
        '-t', str(duration),
        '-vf', 'format=yuv420p',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-y', output_video_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def get_video_duration(video_path):
    command = [
        'ffprobe',
        '-i', video_path,
        '-show_entries', 'format=duration',
        '-v', 'quiet',
        '-of', 'csv=p=0'
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return float(result.stdout.strip())

def get_video_fps(video_path):
    command = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    output = subprocess.run(command, capture_output=True, text=True).stdout.strip()
    num, denom = map(int, output.split('/'))
    return num / denom

def apply_blur_and_mute(input_clip, blur_strength):
    output_blurred_path = input_clip.replace(".mp4", "_blurred.mp4")
    command = [
        "ffmpeg",
        "-i", input_clip,
        "-vf", f"boxblur={blur_strength}:1",
        "-an",
        "-c:v", "libx264",
        "-preset", "fast",
        "-y", output_blurred_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_blurred_path

def create_final_ad_with_blur_cta(input_video, selected_scenes, output_ad_path, cta_text, blur_strength, text_size, font_path, font_color, font_weight):
    temp_folder = "temp_clips"
    os.makedirs(temp_folder, exist_ok=True)
    blurred_clip = None
    other_clips = []
    video_fps = get_video_fps(input_video)
    for idx, (i, score, start_frame, end_frame, duration) in enumerate(selected_scenes):
        output_clip_path = os.path.join(temp_folder, f"clip_{idx + 1}.mp4")
        trim_scene(input_video, output_clip_path, start_frame, end_frame, video_fps)
        if blurred_clip is None and 1 <= duration <= 2:
            blurred_clip = apply_blur_and_mute(output_clip_path, blur_strength)
            cta_image_path = os.path.join(temp_folder, "cta_image.png")
            create_cta_image(cta_text, cta_image_path, text_size, font_path, font_color, font_weight)
            blurred_with_cta = os.path.join(temp_folder, "blurred_with_cta.mp4")
            overlay_cta_on_scene(blurred_clip, cta_image_path, blurred_with_cta)
            blurred_clip = VideoFileClip(blurred_with_cta)
        else:
            other_clips.append(VideoFileClip(output_clip_path))
    if blurred_clip is None:
        if selected_scenes:
            start_frame, end_frame = selected_scenes[0][2], selected_scenes[0][3]
            mid_frame = (start_frame + end_frame) // 2
            cta_start_frame = mid_frame
            cta_end_frame = min(cta_start_frame + int(video_fps), end_frame)
            cta_clip_path = os.path.join(temp_folder, "cta_clip.mp4")
            trim_scene(input_video, cta_clip_path, cta_start_frame, cta_end_frame, video_fps)
            blurred_clip = apply_blur_and_mute(cta_clip_path, blur_strength)
            cta_image_path = os.path.join(temp_folder, "cta_image.png")
            create_cta_image(cta_text, cta_image_path, text_size, font_path, font_color, font_weight)
            blurred_with_cta = os.path.join(temp_folder, "blurred_with_cta.mp4")
            overlay_cta_on_scene(blurred_clip, cta_image_path, blurred_with_cta)
            blurred_clip = VideoFileClip(blurred_with_cta)
        else:
            return
    final_ad = concatenate_videoclips(other_clips + [blurred_clip])
    final_ad.write_videofile(output_ad_path, codec='libx264')

def overlay_cta_on_scene(input_scene, cta_image, output_scene):
    try:
        command = [
            'ffmpeg',
            '-i', input_scene,
            '-i', cta_image,
            '-filter_complex', 'overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-c:a', 'aac',
            '-y', output_scene
        ]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        st.error(f"Error overlaying transparent CTA: {e}")

def producer(youtube_url, keywords, cta_text, blur_strength=20, text_size=50, font_path="arial.ttf", font_color="#FFFFFF", font_weight=0, ad_length=30):
    os.makedirs("temp", exist_ok=True)
    input_video = download_video(youtube_url)
    output_ads = []
    os.makedirs("output_ads", exist_ok=True)
    video_fps = get_video_fps(input_video)
    scenes = detect_scenes(input_video)[:300]
    if not scenes:
        shutil.rmtree("temp")
        return output_ads
    ranked_scenes = rank_scenes(input_video, scenes, keywords, batch_size=32, max_scenes=300)
    lst = [0, 2, 3, 4, 5, 6]
    for _ in range(4):
        i = random.choice(lst)
        lst.remove(i)
        if i == 0:
            selected_scenes = select_scenes_for_duration_v1(ranked_scenes, ad_length, video_fps)
        elif i == 2     :
            selected_scenes = select_scenes_for_duration_v3(ranked_scenes, ad_length, video_fps)
        elif i == 3:
            selected_scenes = select_scenes_for_duration_v4(ranked_scenes, ad_length, video_fps)
        elif i == 4:
            selected_scenes = select_scenes_for_duration_v5(ranked_scenes, ad_length, video_fps)
        elif i == 5:
            selected_scenes = select_scenes_for_duration_v6(ranked_scenes, ad_length, video_fps)
        elif i == 6:
            selected_scenes = select_scenes_for_duration_v7(ranked_scenes, ad_length, video_fps)
        if not selected_scenes:
            continue
        output_ad = f"output_ads/final_ad_with_blurred_cta_{uuid.uuid4().hex}.mp4"
        create_final_ad_with_blur_cta(input_video, selected_scenes, output_ad, cta_text, blur_strength, text_size, font_path, font_color, font_weight)
        output_ads.append(output_ad)
    return output_ads

# Streamlit App
st.title("AI Video-to-Ad Converter 🎥")

# Input fields
youtube_url = st.text_input("Enter YouTube Video URL:")
keywords = st.text_input("Enter keywords for highlight selection (comma-separated):").strip().split(",")
ad_length = st.number_input("Enter desired ad length (in seconds, e.g., 15, 30):", min_value=10, value=30)
cta_text = st.text_input("Enter your Call-to-Action (e.g., 'Buy Now!'):")
blur_strength = st.slider("Blur Strength:", min_value=1, max_value=50, value=20)
text_size = st.slider("CTA Text Size:", min_value=10, max_value=100, value=50)
font_path = st.text_input("Enter font path (e.g., 'arial.ttf'):", value="arial.ttf")
font_color = st.color_picker("Choose CTA Text Color:", value="#FFFFFF")
font_weight = st.slider("CTA Text Outline Weight:", min_value=0, max_value=10, value=0)

# Process video when the user clicks the button
if st.button("Generate Ad"):
    if youtube_url and keywords and cta_text:
        with st.spinner("Processing video..."):
            output_ads = producer(
                youtube_url=youtube_url,
                keywords=keywords,
                cta_text=cta_text,
                blur_strength=blur_strength,
                text_size=text_size,
                font_path=font_path,
                font_color=font_color,
                font_weight=font_weight,
                ad_length=ad_length
            )
        if output_ads:
            st.success("Ad generated successfully!")
            for ad in output_ads:
                st.video(ad)
        else:
            st.error("No scenes detected or no suitable scenes found.")
    else:
        st.warning("Please fill in all the required fields.")