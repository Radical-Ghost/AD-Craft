import React from 'react';
import '../App.css'; // Corrected import path

const Background = () => {
    return (
        <div className="gradient-background">
            <div className="gradient-sphere sphere-1"></div>
            <div className="gradient-sphere sphere-2"></div>
            <div className="gradient-sphere sphere-3"></div>
            <div className="glow"></div>
            <div className="grid-overlay"></div>
            <div className="noise-overlay"></div>
            <div id="particles-container"></div>
        </div>
    );
};

export default Background;