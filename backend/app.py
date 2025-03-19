from flask import Flask, request, jsonify, send_from_directory
import os
import shutil
import logging
from model import initializer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configure logging
logging.getLogger("yt_dlp").setLevel(logging.ERROR)  # Suppress yt-dlp logs
logging.getLogger("pyscenedetect").setLevel(logging.ERROR)  # Suppress pyscenedetect logs
logging.getLogger("moviepy").setLevel(logging.ERROR)  # Suppress moviepy logs
logging.getLogger("PIL").setLevel(logging.ERROR)  # Suppress PIL logs
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)  # Suppress ffmpeg logs

# Configure Flask logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output folder for generated ads
OUTPUT_ADS_FOLDER = os.path.join(os.getcwd(), 'output_ads')
TEMP_FOLDER = os.path.join(os.getcwd(), 'temp')
TEMP_CLIPS_FOLDER = os.path.join(os.getcwd(), 'temp_clips')
os.makedirs(OUTPUT_ADS_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(TEMP_CLIPS_FOLDER, exist_ok=True)

def clear_directories(directories):
    """Clear all files in the specified directories."""
    for directory in directories:
        if os.path.exists(directory):
            logger.info(f"Clearing directory: {directory}")
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Remove file or symbolic link
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Remove directory
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {str(e)}")

@app.route('/generate_ad', methods=['POST'])
def generate_ad():
    try:
        clear_directories([OUTPUT_ADS_FOLDER, TEMP_FOLDER, TEMP_CLIPS_FOLDER])
        # Extract data from the request
        data = request.json
        youtube_url = data.get('youtube_url')
        keywords = data.get('keywords')
        cta_text = data.get('cta_text')
        text_size = int(data.get('text_size', 50))  # Ensure text_size is an integer
        font_path = data.get('font_path', "arial.ttf")
        font_color = data.get('font_color', "#FFFFFF")
        font_weight = int(data.get('font_weight', 0))  # Ensure font_weight is an integer

        # Validate required fields
        if not youtube_url or not keywords or not cta_text:
            return jsonify({"status": "error", "message": "Missing required fields"}), 400

        # Call the producer function to generate ads
        output_files = initializer(youtube_url, keywords, cta_text, text_size, font_path, font_color, font_weight)

        if not output_files:
            logger.error("Ad generation failed: No output files generated")
            return jsonify({"status": "error", "message": "Ad generation failed"}), 500

        logger.info(f"Ads generated successfully: {output_files}")
        return jsonify({
            "status": "success",
            "message": "Ads generated successfully",
            "output_files": output_files
        })

    except Exception as e:
        logger.error(f"Error generating ad: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_videos', methods=['GET'])
def get_videos():
    try:
        # Ensure the output folder exists
        if not os.path.exists(OUTPUT_ADS_FOLDER):
            logger.error(f"Output folder does not exist: {OUTPUT_ADS_FOLDER}")
            return jsonify({'status': 'error', 'message': 'Output folder not found'}), 500

        # List all .mp4 files in the output folder
        videos = [f for f in os.listdir(OUTPUT_ADS_FOLDER) if f.endswith('.mp4')]
        logger.info(f"Videos found: {videos}")
        return jsonify({'status': 'success', 'videos': videos})
    except Exception as e:
        logger.error(f"Error fetching videos: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/output_ads/<filename>', methods=['GET'])
def serve_video(filename):
    """Serve video files from the output_ads folder."""
    try:
        return send_from_directory(OUTPUT_ADS_FOLDER, filename)
    except Exception as e:
        logger.error(f"Error serving file {filename}: {str(e)}")
        return jsonify({'status': 'error', 'message': 'File not found'}), 404

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5500))  # Use environment variable for port
    logger.info(f"Starting Flask server on port {port}")
    app.run(port=port)  