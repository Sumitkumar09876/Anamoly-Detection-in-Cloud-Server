import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 10000

# Time periods
time = np.arange(n_samples)

# Generate synthetic data for CPU, Memory, and Network
cpu_usage = np.sin(time * 0.01) + np.random.normal(0, 0.05, n_samples)  # Regular sinusoidal pattern with noise
memory_usage = np.cos(time * 0.01) + np.random.normal(0, 0.05, n_samples)  # Cosine pattern with noise
network_activity = np.random.normal(0, 0.1, n_samples)  # Random normal activity

# Insert anomalies (random spikes)
anomalies = np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])
cpu_usage += anomalies * (np.random.normal(0.8, 0.2, n_samples))
memory_usage += anomalies * (np.random.normal(0.8, 0.2, n_samples))
network_activity += anomalies * (np.random.normal(0.5, 0.3, n_samples))

# Combine into a DataFrame
data = pd.DataFrame({
    'time': time,
    'cpu_usage': cpu_usage,
    'memory_usage': memory_usage,
    'network_activity': network_activity,
    'anomaly': anomalies
})

# Save dataset to CSV
data.to_csv('cloud_server_synthetic_data.csv', index=False)

# Plot a sample of the data
plt.figure(figsize=(10, 6))
plt.plot(data['time'][:500], data['cpu_usage'][:500], label='CPU Usage')
plt.plot(data['time'][:500], data['memory_usage'][:500], label='Memory Usage')
plt.plot(data['time'][:500], data['network_activity'][:500], label='Network Activity')
plt.scatter(data['time'][:500], data['cpu_usage'][:500] * data['anomaly'][:500], color='red', label='Anomaly', marker='x')
plt.legend()
plt.title('Cloud Server Synthetic Data (First 500 Samples)')
plt.xlabel('Time')
plt.ylabel('Usage / Activity')
plt.show()
