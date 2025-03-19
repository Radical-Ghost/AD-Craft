import React, { useState, useEffect } from "react";
import "../App.css";

const OutputColumn = () => {
	const [videos, setVideos] = useState([]);

	// Fetch the list of videos from the backend
	const fetchVideos = async () => {
		try {
			const response = await fetch("http://127.0.0.1:5500/get_videos");
			const data = await response.json();
			if (data.status === "success") {
				setVideos(data.videos.slice(0, 4)); // Limit to 4 videos
			} else {
				console.error("Error fetching videos:", data.message);
			}
		} catch (error) {
			console.error("Error:", error);
		}
	};

	useEffect(() => {
		fetchVideos(); // Fetch videos on component mount
	}, []);

	return (
		<div className="column output-column">
			<h2>
				<i className="fas fa-tv"></i> 📺 Ad Previews
			</h2>
			<div className="preview-grid">
				{videos.length > 0 ? (
					videos.map((video, index) => (
						<div className="preview" key={index}>
							<video controls>
								<source
									src={`http://127.0.0.1:5500/output_ads/${video}`}
									type="video/mp4"
								/>
								Your browser does not support the video tag.
							</video>
						</div>
					))
				) : (
					<p>No videos available in the output folder.</p>
				)}
			</div>
		</div>
	);
};

export default OutputColumn;
