import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --- Constants ---
INPUT_CSV_FILENAME = "magnetic_readings_complex_noisy_v4.csv"  # V4 data
OUTPUT_PLOT_FILENAME = "filtered_comparison_v4.png"  # V4 plot
OUTPUT_ZOOMED_PLOT_FILENAME = "filtered_zoomed_v4.png"  # V4 zoomed plot

# Columns to use from the CSV
COL_GROUND_TRUTH = "b_total_strength_gt"
COL_NOISY_SIGNAL = "b_total_strength_noisy"
COL_THROTTLE = "throttle"
COL_ROC_THROTTLE = "roc_throttle"  # New for V4 filter
COL_POS_X = "pos_x"


def load_data(filename=INPUT_CSV_FILENAME):
    """Loads the noisy magnetic data from the specified CSV file."""
    print(f"Loading data from {filename}...")
    try:
        data = pd.read_csv(filename)
        print(f"Data loaded successfully. Shape: {data.shape}")
        # Basic validation
        required_cols = [
            COL_GROUND_TRUTH,
            COL_NOISY_SIGNAL,
            COL_THROTTLE,
            COL_ROC_THROTTLE,
            COL_POS_X,
        ]  # Added COL_ROC_THROTTLE
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {', '.join(missing_cols)}. Available: {', '.join(data.columns)}"
            )
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found. Please run mag_sim_v2.py first.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def calculate_rmse(signal1, signal2):
    """Calculates the Root Mean Square Error between two signals."""
    return np.sqrt(np.mean((signal1 - signal2) ** 2))


def objective_function_v4(
    k_vector, noisy_signal, ground_truth_signal, throttle, throttle_sq, roc_throttle
):
    """
    Objective function for V4 to minimize for finding optimal K1, K2, K3.
    k_vector: [K1, K2, K3]
    """
    k1, k2, k3 = k_vector
    predicted_noise_scalar = (throttle * k1) + (throttle_sq * k2) + (roc_throttle * k3)
    estimated_signal = noisy_signal - predicted_noise_scalar
    return calculate_rmse(estimated_signal, ground_truth_signal)


def find_optimal_coefficients(
    noisy_signal, ground_truth_signal, throttle, throttle_sq, roc_throttle
):
    """
    Finds the optimal K_vector = [K1, K2, K3] using scipy.optimize.minimize.
    """
    print("Optimizing K1, K2, K3 coefficients to minimize RMSE...")
    initial_k_guess_vector = np.array(
        [1e-5, 1e-6, 1e-6]
    )  # Initial guess for [K1, K2, K3]

    result = minimize(
        objective_function_v4,
        initial_k_guess_vector,
        args=(noisy_signal, ground_truth_signal, throttle, throttle_sq, roc_throttle),
        method="L-BFGS-B",  # Suitable for multi-variable optimization, can also use 'Nelder-Mead'
    )

    if result.success:
        optimal_k_vector = result.x
        print(f"Optimization successful. Optimal K_vector = {optimal_k_vector}")
        return optimal_k_vector
    else:
        print(f"Optimization failed: {result.message}")
        print("Warning: Optimization failed. Results might not be optimal.")
        return initial_k_guess_vector  # Or handle error more gracefully


def apply_filter_v4(
    noisy_signal, throttle, throttle_sq, roc_throttle, k_optimal_vector
):
    """Applies the V4 filter using the optimal K_vector."""
    k1_opt, k2_opt, k3_opt = k_optimal_vector
    predicted_noise_scalar = (
        (throttle * k1_opt) + (throttle_sq * k2_opt) + (roc_throttle * k3_opt)
    )
    filtered_signal = noisy_signal - predicted_noise_scalar
    return filtered_signal


def plot_results(
    x_coords,
    ground_truth,
    noisy_signal,
    filtered_signal,
    k_optimal,
    rmse_before,
    rmse_after,
):
    """Plots the ground truth, noisy, and filtered signals for V4."""
    print(f"Plotting V4 results to {OUTPUT_PLOT_FILENAME}...")
    plt.figure(figsize=(15, 8))

    k1, k2, k3 = k_optimal  # k_optimal is now a vector for V4

    plt.plot(
        x_coords,
        ground_truth,
        "b--",
        label=f"Ground Truth ({COL_GROUND_TRUTH})",
        alpha=0.9,
    )
    plt.plot(
        x_coords,
        noisy_signal,
        "r-",
        label=f"Noisy Signal V4 ({COL_NOISY_SIGNAL})\nRMSE_before: {rmse_before:.6e}",
        alpha=0.6,
    )
    plt.plot(
        x_coords,
        filtered_signal,
        "k-",
        label=f"Filtered Signal V4 (K1={k1:.2e}, K2={k2:.2e}, K3={k3:.2e})\nRMSE_after: {rmse_after:.6e}",
        alpha=0.9,
        linewidth=1.5,
    )

    plt.title(
        "V4 Magnetic Signal Filtering: Ground Truth vs. Noisy vs. Filtered (3-Coeff Model)"
    )
    plt.xlabel("X Position (m)")
    plt.ylabel("Total Magnetic Field Strength (Tesla)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()

    try:
        plt.savefig(OUTPUT_PLOT_FILENAME)
        print(f"Plot saved successfully to {OUTPUT_PLOT_FILENAME}")
    except Exception as e:
        print(f"Error saving plot: {e}")


def plot_zoomed_results(x_coords, ground_truth, filtered_signal, k_optimal, rmse_after):
    """Plots a zoomed-in comparison of the ground truth and filtered signals for V4."""
    print(f"Plotting V4 zoomed results to {OUTPUT_ZOOMED_PLOT_FILENAME}...")
    plt.figure(figsize=(15, 8))

    k1, k2, k3 = k_optimal  # k_optimal is a vector for V4

    plt.plot(
        x_coords,
        ground_truth,
        "b--",
        label=f"Ground Truth ({COL_GROUND_TRUTH})",
        alpha=0.9,
    )
    plt.plot(
        x_coords,
        filtered_signal,
        "k-",
        label=f"Filtered Signal V4 (K1={k1:.2e}, K2={k2:.2e}, K3={k3:.2e})\nRMSE_after: {rmse_after:.6e}",
        alpha=0.9,
        linewidth=1.5,
    )

    # Auto-scale Y-axis to fit these two lines tightly
    min_val = min(np.min(ground_truth), np.min(filtered_signal))
    max_val = max(np.max(ground_truth), np.max(filtered_signal))
    padding_fraction = 0.05  # 5% padding
    data_range = max_val - min_val

    if data_range == 0:  # Handle case where signals are perfectly flat and identical
        padding = 0.1 * abs(max_val) if max_val != 0 else 0.1
    else:
        padding = data_range * padding_fraction

    plt.ylim(min_val - padding, max_val + padding)

    plt.title(
        "V4 Zoomed Comparison: Ground Truth vs. Filtered Signal (3-Coeff Model Investor Plot)"
    )
    plt.xlabel("X Position (m)")
    plt.ylabel("Total Magnetic Field Strength (Tesla)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()

    try:
        plt.savefig(OUTPUT_ZOOMED_PLOT_FILENAME)
        print(f"Zoomed plot saved successfully to {OUTPUT_ZOOMED_PLOT_FILENAME}")
    except Exception as e:
        print(f"Error saving zoomed plot: {e}")


def main():
    """Main function to run the filtering process."""
    data = load_data()
    if data is None:
        return

    ground_truth = data[COL_GROUND_TRUTH].values
    noisy_signal = data[COL_NOISY_SIGNAL].values
    throttle = data[COL_THROTTLE].values
    roc_throttle = data[COL_ROC_THROTTLE].values
    throttle_squared = throttle**2  # Calculate throttle_squared
    x_coords = data[COL_POS_X].values

    # 1. Quantify noise before filtering
    rmse_before_filtering = calculate_rmse(noisy_signal, ground_truth)
    print(
        f"RMSE (Noisy Signal V4 vs Ground Truth) BEFORE filtering: {rmse_before_filtering:.6e} Tesla"
    )

    # 2. Find optimal K1, K2, K3 coefficients
    optimal_coefficients = find_optimal_coefficients(
        noisy_signal, ground_truth, throttle, throttle_squared, roc_throttle
    )
    print(
        f"Recovered optimal coefficients: K1={optimal_coefficients[0]:.2e}, K2={optimal_coefficients[1]:.2e}, K3={optimal_coefficients[2]:.2e}"
    )

    # 3. Apply the V4 filter
    filtered_signal = apply_filter_v4(
        noisy_signal, throttle, throttle_squared, roc_throttle, optimal_coefficients
    )
    print("V4 Filter applied.")

    # 4. Quantify noise after filtering
    rmse_after_filtering = calculate_rmse(filtered_signal, ground_truth)
    print(
        f"RMSE (Filtered Signal V4 vs Ground Truth) AFTER filtering: {rmse_after_filtering:.6e} Tesla"
    )

    # 5. Visualize results
    plot_results(
        x_coords,
        ground_truth,
        noisy_signal,
        filtered_signal,
        optimal_coefficients,  # Pass the K_vector
        rmse_before_filtering,
        rmse_after_filtering,
    )
    # Also generate the zoomed plot
    plot_zoomed_results(
        x_coords,
        ground_truth,
        filtered_signal,
        optimal_coefficients,  # Pass the K_vector
        rmse_after_filtering,
    )

    print("\nFilter V4 processing complete.")
    if rmse_after_filtering < rmse_before_filtering:
        print(
            f"Filter V4 successfully reduced RMSE by {(rmse_before_filtering - rmse_after_filtering):.6e} (from {rmse_before_filtering:.6e} to {rmse_after_filtering:.6e})"
        )
    else:
        print("Warning: Filter did not improve RMSE. Check parameters or model.")


if __name__ == "__main__":
    main()
