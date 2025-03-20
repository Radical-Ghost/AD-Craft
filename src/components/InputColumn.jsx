import React, { useState } from "react";
import "../App.css"; // Corrected import path

const InputColumn = () => {
	const [videoSource] = useState("youtube");
	const [formData, setFormData] = useState({
		"youtube-link": "",
		keywords: "",
		cta: "",
		"font-style": "Roboto",
		"font-color": "#ffffff",
		"font-size": 40,
		"font-width": 4,
	});

	const handleInputChange = (e) => {
		const { id, value } = e.target;
		setFormData({ ...formData, [id]: value });
	};

	const handleGenerateAd = async () => {
		const requestData = {
			youtube_url: formData["youtube-link"],
			keywords: formData.keywords.split(",").map((k) => k.trim()), // Ensure keywords are an array
			cta_text: formData.cta,
			text_size: parseInt(formData["font-size"], 10), // Ensure text_size is an integer
			font_color: formData["font-color"],
			font_weight: parseInt(formData["font-width"], 10), // Ensure font_weight is an integer
		};

		try {
			const response = await fetch("http://127.0.0.1:5500/generate_ad", {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify(requestData),
			});

			if (response.ok) {
				const result = await response.json();
				if (result.status === "success") {
					alert("Ad generated successfully!");
					console.log("Output Files:", result.output_files);
				} else {
					alert("Ad generation failed: " + result.message);
				}
			} else {
				alert("Failed to generate ad. Please try again.");
			}
		} catch (error) {
			console.error("Error:", error);
			alert("An error occurred while generating the ad.");
		}
	};

	return (
		<div className="column input-column">
			{videoSource === "youtube" && (
				<div className="input-group youtube-input">
					<label htmlFor="youtube-link">
						<i className="fab fa-youtube"></i>{" "}
						<strong>Link:</strong>
					</label>
					<input
						type="text"
						id="youtube-link"
						placeholder="Enter the link..."
						value={formData["youtube-link"]}
						onChange={handleInputChange}
					/>
				</div>
			)}

			{videoSource === "upload" && (
				<div className="input-group upload-input">
					<label htmlFor="video-upload">
						<i className="fas fa-upload"></i>{" "}
						<strong>Upload Video</strong>
					</label>
					<div className="drag-drop-area" id="drag-drop-area">
						<p>
							Drag and drop a video file here or click to upload.
						</p>
						<input
							type="file"
							id="video-upload"
							accept="video/mp4"
							style={{ display: "none" }}
						/>
					</div>
				</div>
			)}

			<div className="input-group">
				<label htmlFor="keywords">
					<i className="fas fa-key"></i> <strong>Keywords:</strong>
				</label>
				<textarea
					id="keywords"
					placeholder="e.g., marketing, ads, video"
					value={formData.keywords}
					onChange={handleInputChange}></textarea>
			</div>

			<div className="input-group">
				<label htmlFor="cta">
					<i className="fas fa-bullhorn"></i>{" "}
					<strong>CTA Text:</strong>
				</label>
				<input
					type="text"
					id="cta"
					placeholder="e.g., Subscribe Now!"
					value={formData.cta}
					onChange={handleInputChange}
				/>
			</div>

			<div className="input-row">
				<div className="input-group">
					<label htmlFor="font-style">
						<i className="fas fa-font"></i>{" "}
						<strong>Font Style:</strong>
					</label>
					<select
						id="font-style"
						value={formData["font-style"]}
						onChange={handleInputChange}>
						<option
							value="Roboto-Regular.ttf"
							style={{
								color: "black",
								backgroundColor: "transparent",
							}}>
							Roboto
						</option>
						<option
							value="Comic Sans MS.ttf"
							style={{
								color: "black",
								backgroundColor: "transparent",
							}}>
							Comic Sans MS
						</option>
						<option
							value="Times New Roman.ttf"
							style={{
								color: "black",
								backgroundColor: "transparent",
							}}>
							Times New Roman
						</option>
					</select>
				</div>
				<div className="input-group">
					<label htmlFor="font-color">
						<i className="fas fa-palette"></i>{" "}
						<strong>Font Color:</strong>
					</label>
					<input
						type="color"
						id="font-color"
						value={formData["font-color"]}
						onChange={handleInputChange}
					/>
				</div>
			</div>

			<div className="input-row">
				<div className="input-group">
					<label htmlFor="font-size">
						<i className="fas fa-text-height"></i>{" "}
						<strong>Font Size:</strong>
					</label>
					<input
						type="number"
						id="font-size"
						min="10"
						max="80"
						value={formData["font-size"]}
						placeholder="e.g., 16px"
						onChange={handleInputChange}
					/>
				</div>
				<div className="input-group">
					<label htmlFor="font-width">
						<i className="fas fa-text-width"></i>{" "}
						<strong>Font Width:</strong>
					</label>
					<input
						type="number"
						id="font-width"
						min="1"
						max="10"
						value={formData["font-width"]}
						placeholder="e.g., 4 (Normal)"
						onChange={handleInputChange}
					/>
				</div>
			</div>

			<button
				id="generate-ad-btn"
				className="btn"
				onClick={handleGenerateAd}>
				<i className="fas fa-rocket"></i> 🚀 Generate Ad
			</button>
		</div>
	);
};

export default InputColumn;
