import React from "react";
import Background from "./components/Background";
import Title from "./components/Title";
import InputColumn from "./components/InputColumn";
import OutputColumn from "./components/OutputColumn";
import "./App.css";

function App() {
	return (
		<div className="content-container">
			<Background />
			<Title />
			<div className="two-column-layout">
				<InputColumn />
				<OutputColumn />
			</div>
		</div>
	);
}

export default App;
