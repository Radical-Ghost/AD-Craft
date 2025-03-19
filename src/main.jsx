import React from "react";
import { createRoot } from "react-dom/client"; // Import createRoot from react-dom/client
import App from "./App";
import "./App.css";

// Get the root element from your HTML file
const container = document.getElementById("root");

// Create a root
const root = createRoot(container);

// Render the App component into the root
root.render(
	<React.StrictMode>
		<App />
	</React.StrictMode>
);
