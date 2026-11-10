
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. Generate the Input Signal
# ============================================================

t = np.linspace(0, 4 * np.pi, 500)
signal = np.sin(t) + 0.5 * np.sin(2 * t)

window_size = 4
quantization_step = 0.1

# ============================================================
# 2. Create Sliding Window Dataset
# ============================================================

X, y = [], []

for i in range(len(signal) - window_size):
    X.append(signal[i:i + window_size])
    y.append(signal[i + window_size])

X = np.array(X)
y = np.array(y)

# ============================================================
# 3. Normalize Input Features
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 4. Initial Neural Network Training
# ============================================================

nn_predictor = MLPRegressor(
    hidden_layer_sizes=(16, 8),
    activation='tanh',
    solver='adam',
    max_iter=2000,
    random_state=42
)

nn_predictor.fit(X_scaled, y)

print("Initial Neural Network Training Completed!")

# ============================================================
# 5. DPCM Reconstruction Function
# ============================================================

def simulate_dpcm(model, scaler, signal, window_size, step,
                  collect_training_data=False):

    reconstructed = list(signal[:window_size])
    transmitted_errors = []

    reconstructed_windows = []
    target_samples = []

    for i in range(window_size, len(signal)):

        current_window = np.array(
            reconstructed[-window_size:]
        ).reshape(1, -1)

        # Normalize using the training scaler
        current_window_scaled = scaler.transform(current_window)

        # Predict the next sample
        predicted_val = model.predict(current_window_scaled)[0]

        actual_val = signal[i]

        # Calculate prediction error
        error = actual_val - predicted_val

        # Quantize prediction error
        quantized_index = np.round(error / step)
        quantized_error = quantized_index * step

        transmitted_errors.append(quantized_error)

        # Save reconstructed-history inputs for retraining
        if collect_training_data:
            reconstructed_windows.append(current_window.flatten())
            target_samples.append(actual_val)

        # Reconstruct sample
        reconstructed_val = predicted_val + quantized_error
        reconstructed.append(reconstructed_val)

    if collect_training_data:
        return (
            np.array(reconstructed),
            np.array(transmitted_errors),
            np.array(reconstructed_windows),
            np.array(target_samples)
        )

    return (
        np.array(reconstructed),
        np.array(transmitted_errors)
    )

# ============================================================
# 6. Initial DPCM Simulation
# ============================================================

(
    initial_reconstructed,
    initial_errors,
    X_reconstructed,
    y_reconstructed
) = simulate_dpcm(
    nn_predictor,
    scaler,
    signal,
    window_size,
    quantization_step,
    collect_training_data=True
)

initial_mse = np.mean(
    (signal[window_size:] -
     initial_reconstructed[window_size:]) ** 2
)

print(f"Initial Reconstruction MSE: {initial_mse:.6f}")

# ============================================================
# 7. Quantization-Aware Retraining
# ============================================================

# Combine original windows and reconstructed-history windows
X_combined = np.vstack((X, X_reconstructed))
y_combined = np.concatenate((y, y_reconstructed))

# Refit the scaler using both types of input windows
scaler_final = StandardScaler()
X_combined_scaled = scaler_final.fit_transform(X_combined)

# Train a new predictor on the combined dataset
nn_predictor_final = MLPRegressor(
    hidden_layer_sizes=(16, 8),
    activation='tanh',
    solver='adam',
    max_iter=2000,
    random_state=42
)

nn_predictor_final.fit(X_combined_scaled, y_combined)

print("Quantization-Aware Training Completed!")

# ============================================================
# 8. Final DPCM Simulation
# ============================================================

reconstructed_signal, transmitted_errors = simulate_dpcm(
    nn_predictor_final,
    scaler_final,
    signal,
    window_size,
    quantization_step
)

# ============================================================
# 9. Calculate Reconstruction Error
# ============================================================

mse = np.mean(
    (signal[window_size:] -
     reconstructed_signal[window_size:]) ** 2
)

rmse = np.sqrt(mse)

print("\n--- DPCM Performance ---")
print(f"Initial MSE: {initial_mse:.6f}")
print(f"Final MSE:   {mse:.6f}")
print(f"Final RMSE:  {rmse:.6f}")

# ============================================================
# 10. Plot Original vs Reconstructed Signal
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    signal,
    label="Original Signal",
    color="blue",
    linewidth=2
)

plt.plot(
    range(window_size, len(signal)),
    reconstructed_signal[window_size:],
    label="AI-Reconstructed Signal",
    color="red",
    linestyle="dashed",
    linewidth=1.8
)

plt.title("AI-Driven DPCM: Original vs Reconstructed Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.savefig(
    "dpcm_results.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nPlot saved as dpcm_results.png")

# ============================================================
# 11. Plot Reconstruction Error
# ============================================================

reconstruction_error = signal - reconstructed_signal

plt.figure(figsize=(12, 4))

plt.plot(
    reconstruction_error,
    color="purple",
    linewidth=1.5
)

plt.title("AI-Driven DPCM: Reconstruction Error")
plt.xlabel("Sample Index")
plt.ylabel("Error")
plt.grid(True)

plt.savefig(
    "dpcm_error.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Error plot saved as dpcm_error.png")
