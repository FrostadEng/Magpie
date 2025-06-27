import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

INPUT_FILENAME = "magnetic_readings.csv"
OUTPUT_PLOT_FILENAME = "magnetic_field_map.png"


def plot_magnetic_data():
    """
    Reads magnetic field data from a CSV file and generates a 2D color map
    of the total magnetic field strength along the flight path.
    """
    print(f"Reading data from {INPUT_FILENAME}...")
    try:
        data = pd.read_csv(INPUT_FILENAME)
    except FileNotFoundError:
        print(
            f"Error: Data file '{INPUT_FILENAME}' not found. "
            "Please run mag_sim_v1.py first to generate the data."
        )
        return
    except pd.errors.EmptyDataError:
        print(f"Error: Data file '{INPUT_FILENAME}' is empty.")
        return

    if data.empty:
        print("No data to plot.")
        return

    print(f"Data loaded successfully. {len(data)} readings found.")

    # Assuming flight path is primarily along the X-axis at a constant Y and Z.
    # We will create a 2D plot where X is the flight path direction,
    # and the color represents the magnetic field strength.
    # For this simple case, we can treat the flight path as a line.
    # If the flight path were a 2D grid (e.g., lawnmower pattern),
    # we would need to reshape the data accordingly.

    x_coords = data["pos_x"].unique()  # Should be sorted if generated linearly
    y_flight_pos = data["pos_y"].iloc[0]  # Assuming constant Y for the flight
    z_flight_alt = data["pos_z"].iloc[0]  # Assuming constant Z for the flight

    # We'll make a simple line plot colored by strength for this 1D flight path
    # If it was a 2D survey, we'd use imshow or pcolormesh more directly with 2D data.

    plt.figure(figsize=(12, 6))

    # Create a scatter plot where color indicates field strength
    # We use a colormap to show the strength.
    # The 'y' axis of this plot will just be a constant value, or we can plot strength vs x.

    # Plotting B_total_strength vs. pos_x
    # For a 2D heatmap of a 2D survey area, you would typically have:
    # x_coords = sorted(data['pos_x'].unique())
    # y_coords = sorted(data['pos_y'].unique())
    # strength_matrix = data.pivot_table(index='pos_y', columns='pos_x', values='b_total_strength').values
    # plt.imshow(strength_matrix, aspect='auto', origin='lower',
    #            extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()])

    # Since our flight path is a single line, a heatmap is less intuitive.
    # A colored line or scatter plot is more appropriate.
    # Let's do a scatter plot where the color of each point is the field strength.

    # Normalize the field strength for colormap
    norm = Normalize(
        vmin=data["b_total_strength"].min(), vmax=data["b_total_strength"].max()
    )
    cmap = plt.get_cmap("viridis")

    sc = plt.scatter(
        data["pos_x"],
        data["b_total_strength"],
        c=data["b_total_strength"],
        cmap=cmap,
        norm=norm,
        s=10,
    )

    plt.colorbar(sc, label="Total Magnetic Field Strength (Tesla)")
    plt.xlabel(
        f"X Position (m) along flight path (Y={y_flight_pos}m, Z={z_flight_alt}m)"
    )
    plt.ylabel("Total Magnetic Field Strength (Tesla)")
    plt.title("Magnetic Field Strength Along Flight Path")
    plt.grid(True)
    plt.tight_layout()

    print(f"Saving plot to {OUTPUT_PLOT_FILENAME}...")
    plt.savefig(OUTPUT_PLOT_FILENAME)
    print(f"Plot saved to {OUTPUT_PLOT_FILENAME}")
    # plt.show() # Uncomment to display the plot directly


if __name__ == "__main__":
    plot_magnetic_data()
