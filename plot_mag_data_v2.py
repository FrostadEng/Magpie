import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# V2 uses the noisy data file and produces a new plot file
INPUT_FILENAME = "magnetic_readings_noisy.csv"
OUTPUT_PLOT_FILENAME = "magnetic_field_map_v2.png"


def plot_magnetic_data_v2():
    """
    Reads noisy magnetic field data from the V2 CSV file and generates plots
    comparing the ground truth signal ('b_total_strength_gt') with the
    noisy signal ('b_total_strength_noisy').
    It also plots the simulated throttle value.
    """
    print(f"Plotting V2 data: Reading from {INPUT_FILENAME}...")
    try:
        data = pd.read_csv(INPUT_FILENAME)
    except FileNotFoundError:
        print(
            f"Error: Data file '{INPUT_FILENAME}' not found. "
            "Please run mag_sim_v2.py first to generate the data."
        )
        return
    except pd.errors.EmptyDataError:
        print(f"Error: Data file '{INPUT_FILENAME}' is empty.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return

    if data.empty:
        print("No data to plot.")
        return

    print(f"Data loaded successfully. {len(data)} readings found.")

    # Verify necessary columns are present
    required_columns = [
        "pos_x",
        "pos_y",
        "pos_z",
        "b_total_strength_gt",
        "b_total_strength_noisy",
        "throttle",
    ]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(
            f"Error: The CSV file '{INPUT_FILENAME}' is missing required columns: {', '.join(missing_columns)}"
        )
        print(f"Available columns are: {', '.join(data.columns)}")
        return

    x_coords = data["pos_x"]
    y_flight_pos = data["pos_y"].iloc[0] if not data["pos_y"].empty else "N/A"
    z_flight_alt = data["pos_z"].iloc[0] if not data["pos_z"].empty else "N/A"

    # Create a figure with two subplots, sharing the X-axis
    fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(
        f"Magnetic Anomaly Simulation V2: Ground Truth vs. Noisy Data\nFlight Path Y={y_flight_pos}m, Z={z_flight_alt}m",
        fontsize=16,
    )

    # --- Subplot 1: Ground Truth vs. Noisy Magnetic Field Strength ---
    axs[0].plot(
        x_coords,
        data["b_total_strength_gt"],
        label="Ground Truth Signal (B_gt total)",
        linestyle="--",
        color="blue",
        alpha=0.9,
    )
    axs[0].plot(
        x_coords,
        data["b_total_strength_noisy"],
        label="Noisy Signal (B_noisy total)",
        color="red",
        alpha=0.7,
    )
    axs[0].set_ylabel("Total Magnetic Field Strength (Tesla)")
    axs[0].set_title("Comparison of Magnetic Field Strengths")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, linestyle=":", alpha=0.7)

    # --- Subplot 2: Throttle Value ---
    axs[1].plot(
        x_coords, data["throttle"], label="Throttle Value", color="green", alpha=0.8
    )
    axs[1].set_xlabel("X Position (m) along flight path")
    axs[1].set_ylabel("Throttle (normalized)")
    axs[1].set_title("Simulated Throttle Value")
    axs[1].legend(loc="upper right")
    axs[1].grid(True, linestyle=":", alpha=0.7)
    axs[1].set_ylim(
        data["throttle"].min() - 0.1, data["throttle"].max() + 0.1
    )  # Auto-adjust Y limits slightly

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for suptitle

    print(f"Saving V2 plot to {OUTPUT_PLOT_FILENAME}...")
    try:
        plt.savefig(OUTPUT_PLOT_FILENAME)
        print(f"Plot successfully saved to {OUTPUT_PLOT_FILENAME}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    # plt.show() # Uncomment to display the plot directly during execution


if __name__ == "__main__":
    plot_magnetic_data_v2()
