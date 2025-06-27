import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --- Constants ---
INPUT_CSV_FILENAME = "magnetic_readings_noisy.csv"
OUTPUT_PLOT_FILENAME = "filtered_comparison_v3.png"

# Columns to use from the CSV
COL_GROUND_TRUTH = "b_total_strength_gt"
COL_NOISY_SIGNAL = "b_total_strength_noisy"
COL_THROTTLE = "throttle"
COL_POS_X = "pos_x"


def load_data(filename=INPUT_CSV_FILENAME):
    """Loads the noisy magnetic data from the specified CSV file."""
    print(f"Loading data from {filename}...")
    try:
        data = pd.read_csv(filename)
        print(f"Data loaded successfully. Shape: {data.shape}")
        # Basic validation
        required_cols = [COL_GROUND_TRUTH, COL_NOISY_SIGNAL, COL_THROTTLE, COL_POS_X]
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


def objective_function(k, noisy_signal, ground_truth_signal, throttle_signal):
    """
    Objective function to minimize for finding the optimal K.
    K is the coupling constant.
    """
    predicted_noise = throttle_signal * k
    estimated_signal = noisy_signal - predicted_noise
    return calculate_rmse(estimated_signal, ground_truth_signal)


def find_optimal_k(noisy_signal, ground_truth_signal, throttle_signal):
    """
    Finds the optimal K value using scipy.optimize.minimize.
    """
    print("Optimizing K to minimize RMSE...")
    initial_k_guess = 0.0  # Initial guess for K
    # Bounds for K can be helpful if we have an idea of its scale, but not strictly necessary for 'Nelder-Mead'
    # result = minimize(objective_function, initial_k_guess,
    #                   args=(noisy_signal, ground_truth_signal, throttle_signal),
    #                   method='Nelder-Mead') # Nelder-Mead is robust for simple problems
    # Using L-BFGS-B which can be more efficient for single variable and allows easy bounds if needed
    result = minimize(
        objective_function,
        initial_k_guess,
        args=(noisy_signal, ground_truth_signal, throttle_signal),
        method="L-BFGS-B",
    )

    if result.success:
        optimal_k = result.x[0]
        print(f"Optimization successful. Optimal K = {optimal_k:.6f}")
        return optimal_k
    else:
        print(f"Optimization failed: {result.message}")
        # Fallback or raise error
        # For simplicity, let's try a very basic scan if optimization fails (or just return guess)
        # This part can be made more robust.
        print("Warning: Optimization failed. Results might not be optimal.")
        return initial_k_guess  # Or handle error more gracefully


def apply_filter(noisy_signal, throttle_signal, k_optimal):
    """Applies the filter using the optimal K."""
    predicted_noise = throttle_signal * k_optimal
    filtered_signal = noisy_signal - predicted_noise
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
    """Plots the ground truth, noisy, and filtered signals."""
    print(f"Plotting results to {OUTPUT_PLOT_FILENAME}...")
    plt.figure(figsize=(15, 8))

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
        label=f"Noisy Signal ({COL_NOISY_SIGNAL})\nRMSE_before: {rmse_before:.6e}",
        alpha=0.6,
    )
    plt.plot(
        x_coords,
        filtered_signal,
        "k-",
        label=f"Filtered Signal (K={k_optimal:.4f})\nRMSE_after: {rmse_after:.6e}",
        alpha=0.9,
        linewidth=1.5,
    )

    plt.title("Magnetic Signal Filtering: Ground Truth vs. Noisy vs. Filtered")
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


def main():
    """Main function to run the filtering process."""
    data = load_data()
    if data is None:
        return

    ground_truth = data[COL_GROUND_TRUTH].values
    noisy_signal = data[COL_NOISY_SIGNAL].values
    throttle = data[COL_THROTTLE].values
    x_coords = data[COL_POS_X].values

    # 1. Quantify noise before filtering
    rmse_before_filtering = calculate_rmse(noisy_signal, ground_truth)
    print(
        f"RMSE (Noisy Signal vs Ground Truth) BEFORE filtering: {rmse_before_filtering:.6e} Tesla"
    )

    # 2. Find optimal K
    optimal_k = find_optimal_k(noisy_signal, ground_truth, throttle)

    # 3. Apply the filter
    filtered_signal = apply_filter(noisy_signal, throttle, optimal_k)
    print("Filter applied.")

    # 4. Quantify noise after filtering
    rmse_after_filtering = calculate_rmse(filtered_signal, ground_truth)
    print(
        f"RMSE (Filtered Signal vs Ground Truth) AFTER filtering: {rmse_after_filtering:.6e} Tesla"
    )

    # 5. Visualize results
    plot_results(
        x_coords,
        ground_truth,
        noisy_signal,
        filtered_signal,
        optimal_k,
        rmse_before_filtering,
        rmse_after_filtering,
    )

    print("\nFilter V3 processing complete.")
    if rmse_after_filtering < rmse_before_filtering:
        print(
            f"Filter successfully reduced RMSE by {(rmse_before_filtering - rmse_after_filtering):.6e} (from {rmse_before_filtering:.6e} to {rmse_after_filtering:.6e})"
        )
    else:
        print("Warning: Filter did not improve RMSE. Check parameters or model.")


if __name__ == "__main__":
    main()
