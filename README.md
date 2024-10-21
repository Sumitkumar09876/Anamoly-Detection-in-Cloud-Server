# Real-time Anomaly Detection in Cloud Servers

This project implements real-time anomaly detection using `IsolationForest` in a `Streamlit` app. The system monitors CPU, memory, and network activity metrics in cloud servers and detects anomalies in real-time.

## Features

- **Real-time Data Streaming & Anomaly Detection**: Detects anomalies in server metrics using the `IsolationForest` algorithm.
- **Pause/Resume Functionality**: Toggle the real-time simulation at any point.
- **Customizable Parameters**: Adjust the contamination level, simulation speed, and selected features (CPU usage, memory usage, network activity).
- **Live Plotting**: Real-time visualization of the server metrics and detected anomalies.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/real-time-anomaly-detection.git
   cd real-time-anomaly-detection
