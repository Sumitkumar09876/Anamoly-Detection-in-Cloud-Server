import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import time
import matplotlib.pyplot as plt

# Title of the app
st.title("Real-time Anomaly Detection in Cloud Servers")

# Initialize session state variables if not already initialized
if 'is_paused' not in st.session_state:
    st.session_state.is_paused = False
if 'start' not in st.session_state:
    st.session_state.start = 0  # Start index for simulation
if 'current_batch' not in st.session_state:
    st.session_state.current_batch = None  # To keep the last batch of data visible

# Move buttons right below the title
col1, col2 = st.columns([1, 1])

# Pause/Resume Button
with col1:
    if st.button("Pause/Resume Simulation"):
        st.session_state.is_paused = not st.session_state.is_paused  # Toggle pause state

# "Run" button to trigger real-time anomaly detection
with col2:
    if st.button("Start Real-time Anomaly Detection") or st.session_state.start > 0:
        st.session_state.start = 1  # Set start session state

# Sidebar for user input
st.sidebar.header("User Input Parameters")

# Set default contamination value for Isolation Forest
contamination = st.sidebar.slider("Set contamination level (anomalies proportion)", 0.01, 0.1, 0.02)

# Simulation speed control (slider)
simulation_speed = st.sidebar.slider("Set simulation speed (seconds per batch)", 0.1, 2.0, 1.0)

# Generate synthetic data
np.random.seed(42)
n_samples = 10000
time_values = np.arange(n_samples)
cpu_usage = np.sin(time_values * 0.01) + np.random.normal(0, 0.05, n_samples)
memory_usage = np.cos(time_values * 0.01) + np.random.normal(0, 0.05, n_samples)
network_activity = np.random.normal(0, 0.1, n_samples)
anomalies = np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])

cpu_usage += anomalies * (np.random.normal(0.8, 0.2, n_samples))
memory_usage += anomalies * (np.random.normal(0.8, 0.2, n_samples))
network_activity += anomalies * (np.random.normal(0.5, 0.3, n_samples))

# Create DataFrame
data = pd.DataFrame({
    'time': time_values,
    'cpu_usage': cpu_usage,
    'memory_usage': memory_usage,
    'network_activity': network_activity,
    'anomaly': anomalies
})

# Select features for anomaly detection
st.sidebar.subheader("Select Features for Detection")
features = st.sidebar.multiselect("Choose columns for detection", ['cpu_usage', 'memory_usage', 'network_activity'], default=['cpu_usage', 'memory_usage', 'network_activity'])

# Create a container to hold the plot and model results
placeholder = st.empty()

# Real-time anomaly detection logic
if st.session_state.start > 0:

    # Isolation Forest Model
    iso_forest = IsolationForest(contamination=contamination, random_state=42)

    # Variables to store incoming data batch-by-batch
    batch_size = 100   # New data batch size to simulate real-time streaming

    # Start real-time simulation (continue from where it was paused)
    for start in range(st.session_state.start, len(data), batch_size):
        # If paused, retain current state and stop updating
        if st.session_state.is_paused:
            st.warning("Simulation Paused. Click 'Pause/Resume Simulation' to continue.")
            st.session_state.start = start  # Save the current batch index
            break

        # Extract current batch of data
        current_batch = data.iloc[start:start + batch_size]
        st.session_state.current_batch = current_batch  # Keep the last batch visible

        # Select features and apply the model
        if len(features) > 0:
            X_batch = current_batch[features]
        else:
            st.warning("Please select at least one feature.")
            break

        # Fit model on the current batch
        iso_forest.fit(X_batch)

        # Make predictions on the batch
        y_pred_batch = iso_forest.predict(X_batch)
        y_pred_batch = [1 if p == -1 else 0 for p in y_pred_batch]  # Convert to binary (1 for anomaly, 0 for normal)

        # Append predicted anomalies to the batch
        current_batch['predicted_anomaly'] = y_pred_batch

        # Clear previous graph and plot the current batch
        placeholder.empty()  # Clear the previous plot
        with placeholder.container():
            # Plot the graph in a single line without batch count
            st.subheader("Real-time Anomaly Detection")
            
            # Create a real-time plot for selected features
            fig, ax = plt.subplots(figsize=(10, 6))
            for feature in features:
                ax.plot(current_batch['time'], current_batch[feature], label=feature.capitalize())
            
            # Plot anomalies in the current batch
            ax.scatter(current_batch['time'], current_batch[features[0]] * current_batch['predicted_anomaly'], color='red', label='Predicted Anomaly', marker='o')
            ax.legend()
            ax.set_xlabel("Time")
            ax.set_ylabel("Feature Values")
            ax.set_title("Real-time Anomaly Detection")
            st.pyplot(fig)

        # Pause for user-defined simulation speed
        time.sleep(simulation_speed)

        # Stop after displaying enough batches
        if start + batch_size >= len(data):
            st.session_state.start = 0  # Reset start position after completion
            break

    st.success("Real-time anomaly detection completed!")

# Show the last state of the batch if paused
if st.session_state.is_paused and st.session_state.current_batch is not None:
    with placeholder.container():
        st.subheader("Paused Data")
        fig, ax = plt.subplots(figsize=(10, 6))
        for feature in features:
            ax.plot(st.session_state.current_batch['time'], st.session_state.current_batch[feature], label=feature.capitalize())
        ax.scatter(st.session_state.current_batch['time'], st.session_state.current_batch[features[0]] * st.session_state.current_batch['predicted_anomaly'], color='red', label='Predicted Anomaly', marker='o')
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("Feature Values")
        ax.set_title("Real-time Anomaly Detection (Paused)")
        st.pyplot(fig)

        # Show the paused batch of data
        st.write(st.session_state.current_batch)
