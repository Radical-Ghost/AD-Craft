#!/usr/bin/env python
# coding: utf-8

# In[107]:


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

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# In[108]:


def download_video(url, output_path=None):
    if output_path is None:
        output_path = f"temp/input_video_{uuid.uuid4().hex}.mp4"
    options = {'format': 'best', 'outtmpl': output_path}
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([url])
    return output_path


# In[109]:


def detect_scenes(video_path):
    """
    Detects scenes from a video using SceneDetect.

    Args:
    - video_path: Path to the video file.

    Returns:
    - List of detected scenes (start, end frames).
    """
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())

    # Process the video to detect scenes
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()

    # Extract frame numbers from detected scenes
    scenes = [(start.get_frames(), end.get_frames()) for start, end in scene_list]
    print(f"📊 Detected {len(scenes)} scenes.")
    return scenes


# In[110]:


# Capture a single representative frame (middle frame of the scene)
def capture_frame(video_path, start_frame, end_frame):
    cap = cv2.VideoCapture(video_path)

    mid_frame = (start_frame + end_frame) // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ret, frame = cap.read()

    cap.release()
    return frame if ret else None

# Efficient scene ranking with optimized batching and sampling
def rank_scenes(video_path, scenes, keywords, batch_size=32, max_scenes=100):
    scene_scores = []
    text_inputs = clip.tokenize(keywords).to(device)

    # Process up to `max_scenes` for efficiency
    scenes = scenes[:max_scenes]

    # Capture frames in one pass (avoid reopening video repeatedly)
    frames = [capture_frame(video_path, start, end) for start, end in scenes]
    frames = [frame for frame in frames if frame is not None]

    # Batch frame evaluation for faster processing
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        images = torch.stack([preprocess(Image.fromarray(frame)) for frame in batch]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(images)
            text_features = model.encode_text(text_inputs)
            similarities = (image_features @ text_features.T).softmax(dim=-1)

        # Store the best keyword match for each frame
        for j, similarity in enumerate(similarities):
            best_score = similarity.max().item()
            scene_idx = i + j
            start, end = scenes[scene_idx]
            duration = (end - start) / 30  # Assuming 30 FPS
            scene_scores.append((scene_idx, best_score, start, end, duration))

    # Sort by relevance (highest score first)
    scene_scores.sort(key=lambda x: x[1], reverse=True)
    return scene_scores


# In[111]:


# def select_scenes_for_duration(ranked_scenes, target_length, fps, min_gap=30, min_scenes=10):
#     """
#     Select scenes to ensure the total duration is at least 10 seconds
#     and does not exceed the target length, while prioritizing relevant scenes.

#     Args:
#     - ranked_scenes: List of (index, score, start, end, duration).
#     - target_length: Desired ad length (in seconds).
#     - fps: Frames per second of the video.
#     - min_gap: Minimum gap between scenes (in seconds).
#     - min_scenes: Minimum number of scenes to select (default: 10).

#     Returns:
#     - List of selected scenes.
#     """
#     selected_scenes = []
#     total_duration = 0.0

#     print(f"🎯 Target Length: {target_length}s (Min Scenes: {min_scenes})")

#     # Sort scenes by relevance (highest score first)
#     ranked_scenes.sort(key=lambda x: x[1], reverse=True)  # Sort by score (x[1] is score)

#     # First, select the most relevant scenes
#     for scene in ranked_scenes:
#         i, score, start, end, _ = scene

#         # Calculate accurate scene duration
#         duration = (end - start) / fps

#         # Ensure time gap between scenes
#         if selected_scenes:
#             last_end_time = selected_scenes[-1][3] / fps
#             current_start_time = start / fps

#             if current_start_time - last_end_time < min_gap:
#                 continue  # Skip this scene if it's too close to the previous one

#         # Add the scene to the selection
#         selected_scenes.append((i, score, start, end, duration))
#         total_duration += duration

#         # Stop if we reach the target length
#         if total_duration >= target_length:
#             break

#     # If the total duration is still less than 10 seconds, extend the duration of the selected scenes
#     if total_duration < 10:
#         print("⚠️ Total duration is less than 10 seconds. Extending the duration of selected scenes.")
#         for idx, (i, score, start, end, duration) in enumerate(selected_scenes):
#             # Extend the scene by adding more frames (up to 5 seconds)
#             extended_end = min(end + int((10 - total_duration) * fps), end + int(5 * fps))
#             extended_duration = (extended_end - start) / fps
#             selected_scenes[idx] = (i, score, start, extended_end, extended_duration)
#             total_duration += (extended_duration - duration)

#             # Stop if we reach the minimum duration
#             if total_duration >= 10:
#                 break

#     # After meeting the minimum duration, continue adding scenes until the target length is reached
#     for scene in ranked_scenes:
#         if scene not in selected_scenes:
#             i, score, start, end, _ = scene

#             # Calculate accurate scene duration
#             duration = (end - start) / fps

#             # Ensure time gap between scenes
#             if selected_scenes:
#                 last_end_time = selected_scenes[-1][3] / fps
#                 current_start_time = start / fps

#                 if current_start_time - last_end_time < min_gap:
#                     continue  # Skip this scene if it's too close to the previous one

#             # Add the scene to the selection
#             if total_duration + duration <= target_length:
#                 selected_scenes.append((i, score, start, end, duration))
#                 total_duration += duration

#             # Stop if we reach the target length
#             if total_duration >= target_length:
#                 break

#     print(f"✅ Final Ad Duration: {total_duration:.2f}s (Target: {target_length}s)")
#     print(f"📈 Selected {len(selected_scenes)} scenes")
#     return selected_scenes


# In[112]:


def select_scenes_for_duration_v1(ranked_scenes, target_length, fps, min_gap=30, min_scenes=10):
    selected_scenes = []
    total_duration = 0.0

    ranked_scenes.sort(key=lambda x: x[1], reverse=True)  # Sort by relevance

    for scene in ranked_scenes:
        i, score, start, end, _ = scene
        duration = (int(end) - int(start)) / int(fps)  # Ensure all values are integers

        if selected_scenes:
            last_end_time = selected_scenes[-1][3] / int(fps)
            current_start_time = int(start) / int(fps)
            if current_start_time - last_end_time < int(min_gap):
                continue

        selected_scenes.append((i, score, start, end, duration))
        total_duration += duration

        if total_duration >= int(target_length):
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


# In[113]:


# def select_scenes_for_duration(ranked_scenes, target_length, fps, min_gap=30, min_scenes=10):
#     """
#     Select at least 10 scenes while ensuring the total duration is at least 10 seconds
#     and does not exceed the target length.

#     Args:
#     - ranked_scenes: List of (index, score, start, end, duration).
#     - target_length: Desired ad length (in seconds).
#     - fps: Frames per second of the video.
#     - min_gap: Minimum gap between scenes (in seconds).
#     - min_scenes: Minimum number of scenes to select (default: 10).

#     Returns:
#     - List of selected scenes.
#     """
#     selected_scenes = []
#     total_duration = 0.0

#     print(f"🎯 Target Length: {target_length}s (Min Scenes: {min_scenes})")

#     # Sort scenes by duration (shortest first) to maximize the number of scenes
#     ranked_scenes.sort(key=lambda x: x[4])  # Sort by duration (x[4] is duration)

#     # First, select at least 10 scenes regardless of duration
#     for scene in ranked_scenes:
#         i, score, start, end, _ = scene

#         # Calculate accurate scene duration
#         duration = (end - start) / fps

#         # Ensure time gap between scenes
#         if selected_scenes:
#             last_end_time = selected_scenes[-1][3] / fps
#             current_start_time = start / fps

#             if current_start_time - last_end_time < min_gap:
#                 continue  # Skip this scene if it's too close to the previous one

#         # Always select at least 10 scenes
#         if len(selected_scenes) < min_scenes:
#             selected_scenes.append((i, score, start, end, duration))
#             total_duration += duration
#             continue

#         # After 10 scenes, ensure we don't exceed target_length
#         if total_duration + duration <= target_length:
#             selected_scenes.append((i, score, start, end, duration))
#             total_duration += duration

#         # Stop if we reach both conditions
#         if len(selected_scenes) >= min_scenes and total_duration >= target_length:
#             break

#     # Ensure the total duration is at least 10 seconds
#     if total_duration < 10:
#         print("⚠️ Total duration is less than 10 seconds. Adding more scenes to meet the minimum duration.")
#         for scene in ranked_scenes:
#             if scene not in selected_scenes:
#                 i, score, start, end, _ = scene
#                 duration = (end - start) / fps

#                 # Ensure time gap between scenes
#                 if selected_scenes:
#                     last_end_time = selected_scenes[-1][3] / fps
#                     current_start_time = start / fps

#                     if current_start_time - last_end_time < min_gap:
#                         continue  # Skip this scene if it's too close to the previous one

#                 # Add the scene to increase the total duration
#                 selected_scenes.append((i, score, start, end, duration))
#                 total_duration += duration

#                 # Stop if we reach the minimum duration
#                 if total_duration >= 10:
#                     break

#     print(f"✅ Final Ad Duration: {total_duration:.2f}s (Target: {target_length}s)")
#     print(f"📈 Selected {len(selected_scenes)} scenes")
#     return selected_scenes


# In[114]:


def trim_scene(input_video, output_path, start_frame, end_frame, fps):
    """
    Trim a scene from the input video.

    Args:
    - input_video: Path to the original video.
    - output_video: Path to save the trimmed scene.
    - start_frame: Starting frame of the scene.
    - end_frame: Ending frame of the scene.
    - fps: Frames per second (default is 30).

    Returns:
    - None (saves the trimmed scene to output_video).
    """
    start_time = start_frame / fps
    end_time = end_frame / fps
    duration = end_time - start_time

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
    # print(f"🎥 Trimmed: {output_path} (Duration: {duration:.2f}s)")


# In[115]:


def merge_videos(video_list, output_path):
    """
    Merge multiple video clips into one.

    Args:
    - video_list: List of trimmed video paths.
    - output_path: Path to save the final merged video.

    Returns:
    - None (saves merged video to output_path).
    """
    # Create a temporary list file for FFmpeg input
    with open("file_list.txt", "w") as f:
        for video in video_list:
            f.write(f"file '{video}'\n")

    # FFmpeg command to concatenate videos
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
    # print(f"🎬 Final Ad Created: {output_path}")


# In[116]:


def create_ad(input_video, selected_scenes, output_ad_path):
    """
    Trim selected scenes and merge them into a final ad video.

    Args:
    - input_video: Path to the original video.
    - selected_scenes: List of selected scenes (start, end, duration).
    - output_ad_path: Path to save the final ad.

    Returns:
    - None (saves the final ad).
    """
    temp_folder = "temp_clips"
    os.makedirs(temp_folder, exist_ok=True)

    video_fps = get_video_fps(input_video)

    trimmed_videos = []
    for idx, (i, score, start_frame, end_frame, duration) in enumerate(selected_scenes):
        output_clip = os.path.join(temp_folder, f"clip_{idx + 1}.mp4")
        trim_scene(input_video, output_clip, start_frame, end_frame, video_fps)
        trimmed_videos.append(output_clip)

    # Merge the trimmed clips into a single ad
    merge_videos(trimmed_videos, output_ad_path)


# In[117]:


def create_cta_image(cta_text, output_image_path, text_size=50, font_path="arial.ttf", font_color="#FFFFFF", font_weight=0,  size=(1280, 720)):
    """
    Create a transparent CTA image with customizable font style, color, and dynamic font size.

    Args:
    - cta_text: The CTA message (e.g., "Buy Now!").
    - output_image_path: Path to save the transparent image.
    - text_size: Font size (integer).
    - font_path: Path to a TrueType (.ttf) font file (e.g., 'Roboto.ttf').
    - font_color: Font color in HEX format (e.g., '#FF0000' for red).
    - size: Tuple of image dimensions (width, height).
    - font_weight: Thickness of the font outline (default is 0).

    Returns:
    - None (saves the transparent CTA image).
    """
    # Ensure size is a tuple
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("Size must be a list or tuple of length 2 (width, height).")

    # Create a transparent RGBA image
    image = Image.new("RGBA", size, (0, 0, 0, 0)) 
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(font_path, text_size)
    except IOError:
        print(f"⚠️ Font not found: {font_path}. Falling back to default.")
        font = ImageFont.load_default()

    # Calculate text dimensions and center the CTA
    bbox = font.getbbox(cta_text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)

    # Convert HEX to RGBA
    font_rgb = tuple(int(font_color[i:i+2], 16) for i in (1, 3, 5))

    # Draw font outline if font_weight > 0
    if font_weight > 0:
        for x in range(-font_weight, font_weight + 1):
            for y in range(-font_weight, font_weight + 1):
                draw.text((text_position[0] + x, text_position[1] + y), cta_text, font=font, fill=(0, 0, 0, 255))

    # Draw the CTA text on the transparent image
    draw.text(text_position, cta_text, font=font, fill=font_rgb + (255,))  # (255) sets full opacity

    # Save the image
    image.save(output_image_path, "PNG")
    # print(f"✅ CTA Image Created: {output_image_path} with Font: {font_path}, Color: {font_color}, Size: {text_size}")


# In[118]:


def create_cta_video(cta_image_path, output_video_path, duration=5):
    """
    Convert a CTA image to a video.

    Args:
    - cta_image_path: Path to the input image.
    - output_video_path: Path to save the video.
    - duration: Length of the CTA video in seconds.

    Returns:
    - None (saves the video).
    """
    command = [
        'ffmpeg',
        '-loop', '1',                # Loop the image
        '-i', cta_image_path,         # Input image
        '-t', str(duration),          # Duration of the video (in seconds)
        '-vf', 'format=yuv420p',      # Ensure compatibility
        '-c:v', 'libx264',            # Consistent video codec
        '-pix_fmt', 'yuv420p',        # Force pixel format for compatibility
        '-y', output_video_path       # Overwrite if file exists
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # print(f"🎥 CTA Video Created: {output_video_path}")


# In[119]:


def append_cta_to_ad(ad_video, cta_video, output_final_ad):
    """
    Append the CTA video to the final ad.

    Args:
    - ad_video: Path to the original ad video.
    - cta_video: Path to the CTA video.
    - output_final_ad: Path to save the ad with CTA.

    Returns:
    - None (saves the merged video).
    """

    # Force re-encoding for compatibility
    command = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', 'merge_list.txt',
        '-c:v', 'libx264',            # Ensure both are encoded with H.264
        '-pix_fmt', 'yuv420p',        # Set a compatible pixel format
        '-preset', 'fast',
        '-c:a', 'aac',
        '-y', output_final_ad         # Overwrite output file if it exists
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # print(f"🎬 Final Ad with CTA Created: {output_final_ad}")


# In[120]:


# def create_final_ad_with_cta(input_video, selected_scenes, output_ad_path, cta_text, ):
#     """
#     Create the final ad by merging video clips and overlaying a customized CTA.

#     Args:
#     - input_video: Path to the original video.
#     - selected_scenes: List of scenes to include in the ad.
#     - output_ad_path: Path to save the final ad.
#     - cta_text: Call-to-action message.
#     - font_path: Path to the font file (.ttf).
#     - font_color: Font color in HEX (e.g., "#FFFFFF").
#     """
#     temp_folder = "temp_clips"
#     os.makedirs(temp_folder, exist_ok=True)

#     # Step 1: Trim the selected scenes
#     trimmed_videos = []
#     video_fps = get_video_fps(input_video)

#     for idx, (i, score, start_frame, end_frame, duration) in enumerate(selected_scenes):
#         output_clip = os.path.join(temp_folder, f"clip_{idx + 1}.mp4")
#         trim_scene(input_video, output_clip, start_frame, end_frame, video_fps)
#         trimmed_videos.append(output_clip)

#     # Step 2: Create the CTA image with custom font and color
#     cta_image_path = os.path.join(temp_folder, "cta_image.png")
#     create_cta_image(cta_text, cta_image_path, 

#     # Step 3: Merge the video clips
#     merged_ad_path = os.path.join(temp_folder, "ad_without_cta.mp4")
#     merge_videos(trimmed_videos, merged_ad_path)

#     # Step 4: Overlay CTA on the final scene
#     overlay_cta_on_scene(merged_ad_path, cta_image_path, output_ad_path)

#     print(f"🎬 Final ad with CTA saved: {output_ad_path}")


# In[121]:


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
    """
    Extracts the actual FPS from the video using FFmpeg.
    """
    import subprocess
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

import subprocess

def apply_blur_and_mute(input_clip, blur_strength):
    """
    Apply a blur effect and mute the audio of a video.

    Args:
    - input_clip: Path to the input video.
    - blur_strength: Intensity of the blur effect.

    Returns:
    - Path to the blurred and muted video.
    """
    output_blurred_path = input_clip.replace(".mp4", "_blurred.mp4")

    command = [
        "ffmpeg",
        "-i", input_clip,
        "-vf", f"boxblur={blur_strength}:1",
        "-an",  # Mute the audio
        "-c:v", "libx264",
        "-preset", "fast",
        "-y", output_blurred_path
    ]

    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # print(f"✅ Blurred and muted scene: {output_blurred_path}")
    return output_blurred_path



# In[122]:


def create_final_ad_with_blur_cta(input_video, selected_scenes, output_ad_path, cta_text, blur_strength, text_size, font_path, font_color, font_weight):
    """
    Create the final ad by blurring, muting, and overlaying the CTA on a 1-2 second scene.
    If no scene is detected for the CTA, trim 1 second from any scene and apply the CTA.

    Args:
    - input_video: Path to the original video.
    - selected_scenes: List of scenes (with frame start, end, and duration).
    - output_ad_path: Path to save the final ad.
    - cta_text: The call-to-action text to display.
    - blur_strength: Blur intensity (for the selected scene).
    - text_size: Font size for the CTA text.
    - font_path: Path to the font file (.ttf).
    - font_color: Font color in HEX format.
    """
    temp_folder = "temp_clips"
    os.makedirs(temp_folder, exist_ok=True)

    blurred_clip = None
    other_clips = []
    video_fps = get_video_fps(input_video)

    for idx, (i, score, start_frame, end_frame, duration) in enumerate(selected_scenes):
        output_clip_path = os.path.join(temp_folder, f"clip_{idx + 1}.mp4")
        trim_scene(input_video, output_clip_path, start_frame, end_frame, video_fps)

        # Select the first scene with a duration between 1 and 2 seconds
        if blurred_clip is None and 1 <= duration <= 2:
            blurred_clip = apply_blur_and_mute(output_clip_path, blur_strength)

            # Step 1: Create the CTA image
            cta_image_path = os.path.join(temp_folder, "cta_image.png")
            create_cta_image(cta_text, cta_image_path, text_size, font_path, font_color, font_weight)

            # Step 2: Overlay CTA on the blurred scene
            blurred_with_cta = os.path.join(temp_folder, "blurred_with_cta.mp4")
            overlay_cta_on_scene(blurred_clip, cta_image_path, blurred_with_cta)

            blurred_clip = VideoFileClip(blurred_with_cta)
        else:
            other_clips.append(VideoFileClip(output_clip_path))

    # If no scene was blurred, trim 1 second from any scene and apply the CTA
    if blurred_clip is None:
        print("❌ No scene between 1-2 seconds found. Trimming 1 second from a scene for the CTA.")
        
        # Take the first scene and trim 1 second from it
        if selected_scenes:
            start_frame, end_frame = selected_scenes[0][2], selected_scenes[0][3]
            mid_frame = (start_frame + end_frame) // 2
            cta_start_frame = mid_frame
            cta_end_frame = min(cta_start_frame + int(video_fps), end_frame)  # Trim 1 second

            # Trim the scene
            cta_clip_path = os.path.join(temp_folder, "cta_clip.mp4")
            trim_scene(input_video, cta_clip_path, cta_start_frame, cta_end_frame, video_fps)

            # Apply blur and mute
            blurred_clip = apply_blur_and_mute(cta_clip_path, blur_strength)

            # Step 1: Create the CTA image
            cta_image_path = os.path.join(temp_folder, "cta_image.png")
            create_cta_image(cta_text, cta_image_path, text_size, font_path, font_color, font_weight)

            # Step 2: Overlay CTA on the blurred scene
            blurred_with_cta = os.path.join(temp_folder, "blurred_with_cta.mp4")
            overlay_cta_on_scene(blurred_clip, cta_image_path, blurred_with_cta)

            blurred_clip = VideoFileClip(blurred_with_cta)
        else:
            print("❌ No scenes available. Exiting.")
            return

    # Step 3: Merge normal clips and blurred CTA scene
    final_ad = concatenate_videoclips(other_clips + [blurred_clip])

    # Step 4: Save the final ad
    final_ad.write_videofile(output_ad_path, codec='libx264')
    print(f"🎬 Final ad with CTA saved: {output_ad_path}")


# In[123]:


def overlay_cta_on_scene(input_scene, cta_image, output_scene):
    """
    Overlay a transparent CTA image onto a video scene.

    Args:
    - input_scene: Path to the input video scene (e.g., blurred video).
    - cta_image: Path to the transparent CTA image.
    - output_scene: Path to save the video with CTA overlay.

    Returns:
    - None (saves the output video).
    """
    try:
        command = [
            'ffmpeg',
            '-i', input_scene,                 # Input blurred video
            '-i', cta_image,                   # Transparent CTA image
            '-filter_complex', 'overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2',  # Center overlay
            '-c:v', 'libx264',                 # Ensure video compatibility
            '-preset', 'fast',
            '-c:a', 'aac',                     # Ensure audio compatibility
            '-y', output_scene                  # Output video
        ]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        # print(f"✅ Transparent CTA Overlay Applied: {output_scene}")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error overlaying transparent CTA: {e}")


# In[124]:


# def main():
#     print("🎥 Welcome to the AI Video-to-Ad Converter!\n")

#     # Step 1: Collect Inputs
#     # youtube_url = input("📹 Enter YouTube Video URL: ")
#     # keywords = input("🔍 Enter keywords for highlight selection (comma-separated): ").strip().split(",")
#     # ad_length = int(input("⏳ Enter desired ad length (in seconds, e.g., 15, 30): "))
#     # cta_text = input("📢 Enter your Call-to-Action (e.g., 'Buy Now!'): ")
#     # blur_strength = int(input("Enter Blur Strength: "))

#     youtube_url = "https://www.youtube.com/watch?v=Av5xTnxHOTs"
#     keywords = ['Spear', 'Sword', 'Geo', 'Dragon', 'Liyue', 'Shield', 'Zhongli']
#     ad_length = 30
#     cta_text = "Play it Now!!"
#     blur_strength = 20

#     input_video = "input_video.mp4"
#     output_ad = "final_ad_with_blurred_cta.mp4"

#     # Step 2: Download the YouTube Video
#     download_video(youtube_url, input_video)

#     # Step 3: Get FPS of the Video
#     video_fps = get_video_fps(input_video)

#     # Step 4: Detect and Rank Scenes
#     scenes = detect_scenes(input_video)[:300]
#     if not scenes:
#         print("❌ No scenes detected. Exiting.")
#         return

#     ranked_scenes = rank_scenes(input_video, scenes, keywords, batch_size=32, max_scenes=300)

#     # Step 5: Select Scenes to Meet Ad Length
#     selected_scenes = select_scenes_for_duration(ranked_scenes, ad_length, video_fps)

#     if not selected_scenes:
#         print("❌ No suitable scenes found. Try different keywords.")
#         return

#     print(f"✅ Selected {len(selected_scenes)} scenes for the ad.")

#     # Step 6: Create the Final Ad with Blurred CTA
#     create_final_ad_with_blur_cta(input_video, selected_scenes, output_ad, cta_text, blur_strength)
    

#     # Step 7: Verify the Final Output Length
#     final_duration = get_video_duration(output_ad)
#     print(f"🎬 Final Ad Length: {final_duration:.2f}s (Target: {ad_length}s)")
#     print(f"🎉 Ad successfully created: {output_ad}")

# if __name__ == "__main__":
#     main()


# In[ ]:


def producer(youtube_url, keywords, cta_text, blur_strength=20, text_size=50, font_path="arial.ttf", font_color="#FFFFFF", font_weight=0, ad_length=30):
    os.makedirs("temp", exist_ok=True)

    # Download the video and get the path
    input_video = download_video(youtube_url)
    output_ads = []

    # Ensure the output_ads folder exists
    os.makedirs("output_ads", exist_ok=True)

    # Process the video
    video_fps = get_video_fps(input_video)  # Pass the correct input_video path
    scenes = detect_scenes(input_video)[:300]

    if not scenes:
        shutil.rmtree("temp")  # Clean up temp folder
        return output_ads

    ranked_scenes = rank_scenes(input_video, scenes, keywords, batch_size=32, max_scenes=300)

    lst = [0, 2, 3, 4, 5, 6]
    # Generate 4 variations
    for _ in range(4):
        i = random.choice(lst)
        lst.remove(i)
        if i == 0:
            selected_scenes = select_scenes_for_duration_v1(ranked_scenes, ad_length, video_fps)
        elif i == 2:
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

def initializer(youtube_url, keywords, cta_text, text_size=50, font_path="Arial.ttf", font_color="#FFFFFF", font_weight=0):
    videos = (
        youtube_url,
        keywords,
        60,
        cta_text,
        text_size,
        font_path,
        font_color,
        font_weight
    )

    producer(
            youtube_url=videos[0],
            keywords=videos[1],
            cta_text=videos[3],
            text_size=videos[4],
            font_path=videos[5],
            font_color=videos[6],
            font_weight=videos[7]
    )