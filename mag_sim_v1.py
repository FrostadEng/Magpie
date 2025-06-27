import numpy as np
import csv

# --- Simulation Parameters ---
# Environment
SPACE_DIMS = np.array([100.0, 100.0, 50.0])  # meters (x, y, z)

# Dipole (Ground Truth Anomaly)
DIPOLE_POSITION = np.array([50.0, 50.0, 10.0])  # meters (x, y, z) - buried depth
DIPOLE_MOMENT_STRENGTH = 1000.0  # A*m^2 (Amperemeter squared)
DIPOLE_ORIENTATION_THETA = 0.0  # radians, angle from Z-axis (0 = vertical)
DIPOLE_ORIENTATION_PHI = (
    0.0  # radians, angle from X-axis in XY-plane (0 = aligned with X)
)

# Flight Path
FLIGHT_ALTITUDE = 25.0  # meters
FLIGHT_START_X = 0.0
FLIGHT_END_X = SPACE_DIMS[0]
FLIGHT_Y_POSITION = SPACE_DIMS[1] / 2  # Fly through the middle of the Y-axis
SAMPLING_INTERVAL = 0.1  # meters (10 cm)

# Output
OUTPUT_FILENAME = "magnetic_readings.csv"

# Physical constants
MU_0 = 4 * np.pi * 1e-7  # Permeability of free space (Tesla*meter/Ampere)


def get_dipole_moment_vector(strength, theta, phi):
    """
    Calculates the magnetic moment vector from spherical coordinates.
    theta: inclination from Z-axis (radians)
    phi: azimuth from X-axis in XY-plane (radians)
    """
    mx = strength * np.sin(theta) * np.cos(phi)
    my = strength * np.sin(theta) * np.sin(phi)
    mz = strength * np.cos(theta)
    return np.array([mx, my, mz])


def calculate_magnetic_field(dipole_pos, dipole_moment_vec, sensor_pos):
    """
    Calculates the magnetic field vector at a sensor position due to a magnetic dipole.
    Formula from: https://en.wikipedia.org/wiki/Magnetic_dipole#Field_of_a_static_magnetic_dipole
    """
    r_vec = sensor_pos - dipole_pos
    r_norm = np.linalg.norm(r_vec)

    if r_norm == 0:
        return np.array([0.0, 0.0, 0.0])  # Avoid division by zero at dipole location

    # Unit vector in the direction of r_vec
    r_hat = r_vec / r_norm

    # Magnetic field calculation
    # B = (mu_0 / (4 * pi * r^3)) * (3 * (m . r_hat) * r_hat - m)
    term1_scalar = 3 * np.dot(dipole_moment_vec, r_hat)
    term1_vector = term1_scalar * r_hat
    b_field = (MU_0 / (4 * np.pi * r_norm**3)) * (term1_vector - dipole_moment_vec)

    return b_field  # Returns Bx, By, Bz in Tesla


def simulate_flight_and_collect_data():
    """
    Simulates the sensor flight and collects magnetic field data.
    """
    print("Starting magnetic simulation...")

    dipole_m_vec = get_dipole_moment_vector(
        DIPOLE_MOMENT_STRENGTH, DIPOLE_ORIENTATION_THETA, DIPOLE_ORIENTATION_PHI
    )
    print(f"Dipole Moment Vector: {dipole_m_vec} A*m^2")

    flight_points_x = np.arange(FLIGHT_START_X, FLIGHT_END_X, SAMPLING_INTERVAL)
    num_points = len(flight_points_x)
    print(f"Simulating {num_points} points along the flight path...")

    readings = []

    for i, x_pos in enumerate(flight_points_x):
        sensor_position = np.array([x_pos, FLIGHT_Y_POSITION, FLIGHT_ALTITUDE])
        b_field_vector = calculate_magnetic_field(
            DIPOLE_POSITION, dipole_m_vec, sensor_position
        )
        readings.append(
            {
                "pos_x": sensor_position[0],
                "pos_y": sensor_position[1],
                "pos_z": sensor_position[2],
                "b_x": b_field_vector[0],
                "b_y": b_field_vector[1],
                "b_z": b_field_vector[2],
                "b_total_strength": np.linalg.norm(b_field_vector),
            }
        )
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{num_points} points...")

    print(
        f"Simulation complete. Saving {len(readings)} readings to {OUTPUT_FILENAME}..."
    )
    # Save data to CSV
    if readings:
        with open(OUTPUT_FILENAME, "w", newline="") as csvfile:
            fieldnames = readings[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(readings)
        print(f"Data successfully saved to {OUTPUT_FILENAME}")
    else:
        print("No readings generated.")

    print("Simulation finished.")


if __name__ == "__main__":
    simulate_flight_and_collect_data()
