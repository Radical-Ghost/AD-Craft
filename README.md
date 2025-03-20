# 📹 AD-Craft: AI-Powered Video-to-Ad Converter

This project is an **AI-powered video-to-ad converter** that automatically converts long-form YouTube videos into **short, engaging ads**. Using advanced scene detection and semantic understanding, it selects **relevant scenes**, applies a **blurred CTA** (Call-to-Action), and generates an ad optimized for **Google Ads** and **YouTube Shorts**.

## 🧠 How It Works

1. **Video Download & Preprocessing**:

    - Downloads YouTube videos using `yt-dlp`.
    - Extracts key scenes with `pyscenedetect`.

2. **Scene Ranking**:

    - Ranks scenes based on **keyword relevance** using **CLIP**.
    - Ensures **at least 10 scenes** are selected, with a **maximum duration of 60 seconds**.

3. **CTA Integration**:

    - Identifies the **first 1-2 second scene**, applies a **blur effect**, and **mutes audio**.
    - Overlays a **customizable CTA** with dynamic **font size, color, weight**, and a **transparent background**.

4. **Ad Generation**:
    - Merges selected scenes and the **blurred CTA**.
    - Provides a **final ad** in a web-friendly format via a **Flask API**.

## 🚀 Features

-   **Fully Automated Pipeline**: End-to-end automation from **video input** to **ad output**.
-   **Dynamic CTA Customization**: Adjust **text size**, **color**, and **font weight**.
-   **Smart Scene Selection**: Prioritizes scenes that align with the **target message**.
-   **Fallback Handling**: Always produces an ad, even if no suitable scene is found.
-   **Fast & Scalable**: Optimized for **speed** with batch processing.
-   **API-Ready**: Flask-based API for **easy integration** into any platform.

**Request Parameters:**

-   `youtube_url`: YouTube video link
-   `keywords`: List of keywords for scene ranking
-   `ad_length`: Desired ad length (in seconds)
-   `cta_text`: Call-to-Action text
-   `font_size`: Font size for CTA
-   `font_path`: Path to the font file
-   `font_color`: Font color in HEX (e.g., #FFFFFF)
-   `font_weight`: Font weight (e.g., bold, normal)

## 📦 Installation

### Prerequisites

Ensure you have the following installed:

-   [Python 3.12+](https://www.python.org/downloads/)
-   [FFmpeg](https://www.ffmpeg.org/download.html)

### Clone the Repository

```bash
git clone https://github.com/Radical-Ghost/AD-Craft.git
cd video-to-ad
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install node Modules

```bash
npm install
```

## ▶️ Usage

Ensure you are in the root folder (Folder with the name Ad-Craft)

### Run the frontend

```bash
npm run dev
```

### Run the Flask API

```bash
cd backend
python app.py
```

## 🏆 What Makes This Unique?

-   **Adaptive Scene Selection**: Ensures **at least 10 scenes** while staying **within 60 seconds**.
-   **Blurred CTA Integration**: Enhances **engagement** by focusing attention on key moments.
-   **Error-Resilient**: Handles **edge cases** with fallback mechanisms.
-   **Flexible Output**: Fully customizable **CTA styling** and **video length**.
