import numpy as np
import csv
from scipy.spatial.transform import Rotation as R

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
OUTPUT_FILENAME = "magnetic_readings_complex_noisy_v4.csv"  # Changed for V4

# Physical constants
MU_0 = 4 * np.pi * 1e-7  # Permeability of free space (Tesla*meter/Ampere)

# --- V4 Noise Model Parameters ---

# 1. Dominant Motor Noise (Complex Model for V4)
# The motor noise is now a scalar value that will be added as a Z-component to the B-field
# *before* IMU noise and Diurnal noise are applied.
# Motor_Noise_Scalar = (Throttle * K1_ACTUAL) + (Throttle^2 * K2_ACTUAL) + (Rate_of_change_of_Throttle * K3_ACTUAL)
K1_ACTUAL = 1.0e-5  # Linear term coefficient (based on filter_v3's optimal K)
K2_ACTUAL = 0.5e-5  # Quadratic term coefficient
K3_ACTUAL = 0.2e-5  # Rate of change term coefficient

# Throttle simulation (parameters remain from V2)
THROTTLE_MIN = 0.5
THROTTLE_MAX = 1.0
THROTTLE_OSCILLATIONS = (
    2.0  # Number of full sine wave oscillations over the flight path
)

# Note: The old V2 motor dipole parameters (MOTOR_REL_POSITION, MOTOR_BASE_MOMENT_STRENGTH,
# MOTOR_DIPOLE_MOMENT_ORIENTATION_VECTOR) will NOT be used directly to calculate the primary motor noise in V4,
# as the new K1, K2, K3 model dictates the scalar motor noise value.
# These could be conceptually thought of as part of what K1_ACTUAL, etc. are modeling, or removed if truly redundant.
# For now, they will be unused by the new scalar noise model.

# 2. Sensor Orientation (IMU) Noise (parameters remain from V2)
# Small random errors in sensor's roll, pitch, yaw
# Assuming sensor's 'perfect' orientation is aligned with world axes (e.g., Z down if measuring gravity)
# For magnetic field, 'perfect' means its axes align with the (X, Y, Z) world frame.
IMU_ROLL_STD_DEV = np.deg2rad(0.5)  # Standard deviation of roll error in radians
IMU_PITCH_STD_DEV = np.deg2rad(0.5)  # Standard deviation of pitch error in radians
IMU_YAW_STD_DEV = np.deg2rad(1.0)  # Standard deviation of yaw error in radians

# 3. Environmental (Diurnal) Drift
# Slow change in Earth's background field. Modeled as a low-frequency sine wave.
DIURNAL_PERIOD_FLIGHTS = 0.5  # How many full sine wave periods of diurnal change occur over the flight duration.
# e.g., 0.5 means half a sine wave period.
DIURNAL_AMPLITUDE_XT = 1e-9  # Max change in Tesla for X component (e.g., 1 nT)
DIURNAL_AMPLITUDE_YT = 1e-9  # Max change in Tesla for Y component (e.g., 1 nT)
DIURNAL_AMPLITUDE_ZT = (
    5e-9  # Max change in Tesla for Z component (e.g., 5 nT, often dominant)
)


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
    print("Starting magnetic simulation (V2 with noise models)...")

    # Ground truth dipole
    gt_dipole_m_vec = get_dipole_moment_vector(
        DIPOLE_MOMENT_STRENGTH, DIPOLE_ORIENTATION_THETA, DIPOLE_ORIENTATION_PHI
    )
    print(f"Ground Truth Dipole Moment Vector: {gt_dipole_m_vec} A*m^2")

    flight_points_x = np.arange(FLIGHT_START_X, FLIGHT_END_X, SAMPLING_INTERVAL)
    flight_length = FLIGHT_END_X - FLIGHT_START_X
    num_points = len(flight_points_x)
    print(
        f"Simulating {num_points} points along the flight path (V4 Complex Motor Noise)..."
    )

    readings = []
    previous_throttle = (
        THROTTLE_MAX + THROTTLE_MIN
    ) / 2  # Initialize previous_throttle for RoC calc

    for i, x_pos in enumerate(flight_points_x):
        sensor_position = np.array([x_pos, FLIGHT_Y_POSITION, FLIGHT_ALTITUDE])

        # --- Calculate Ground Truth Magnetic Field (B_gt) ---
        b_gt_vector = calculate_magnetic_field(
            DIPOLE_POSITION, gt_dipole_m_vec, sensor_position
        )
        b_gt_total_strength = np.linalg.norm(b_gt_vector)

        # --- 1. V4 Complex Dominant Motor Noise ---
        # Simulate throttle value (sine wave)
        current_throttle = (THROTTLE_MAX - THROTTLE_MIN) / 2 * np.sin(
            2 * np.pi * THROTTLE_OSCILLATIONS * (x_pos - FLIGHT_START_X) / flight_length
        ) + (THROTTLE_MAX + THROTTLE_MIN) / 2

        throttle_squared = current_throttle**2

        if i == 0:  # For the first point, RoC is zero or based on initial assumption
            rate_of_change_of_throttle = 0.0
        else:
            # Simple RoC: (current - previous). K3_ACTUAL will absorb scaling by dt or dx.
            rate_of_change_of_throttle = current_throttle - previous_throttle

        motor_noise_scalar = (
            (current_throttle * K1_ACTUAL)
            + (throttle_squared * K2_ACTUAL)
            + (rate_of_change_of_throttle * K3_ACTUAL)
        )

        # This scalar noise is added as a Z-component vector
        b_motor_noise_contribution_vector = np.array([0.0, 0.0, motor_noise_scalar])

        # Update previous_throttle for the next iteration
        previous_throttle = current_throttle

        # The field that will be subjected to IMU noise is GT + this new motor noise contribution
        b_field_world_frame = b_gt_vector + b_motor_noise_contribution_vector
        # --- End of V4 Motor Noise ---

        # --- 2. Sensor Orientation (IMU) Noise ---
        # Simulate small random orientation errors
        roll_error = np.random.normal(0, IMU_ROLL_STD_DEV)
        pitch_error = np.random.normal(0, IMU_PITCH_STD_DEV)
        yaw_error = np.random.normal(0, IMU_YAW_STD_DEV)

        # Create a rotation object for this misorientation
        # Assuming 'xyz' extrinsic convention for roll, pitch, yaw.
        # This rotation transforms vectors from the SENSOR frame to the WORLD frame.
        # R_sensor_to_world = R.from_euler('xyz', [roll_error, pitch_error, yaw_error], degrees=False)
        # To find what the misoriented sensor measures, we need to transform the world-frame field
        # into the sensor's frame. This is achieved by R_world_to_sensor = R_sensor_to_world.inv()
        # So, B_sensor_frame = R_world_to_sensor.apply(B_world_frame)

        # Simpler: define rotation that transforms from WORLD to SENSOR frame
        # If sensor has roll, pitch, yaw error, it means its axes are rotated relative to world.
        # A positive roll means sensor's y-axis is tilted down.
        # The field B_world_frame, when measured by this tilted sensor, will appear rotated.
        # The transformation from world coordinates to sensor coordinates is R.from_euler('xyz', [-roll, -pitch, -yaw]).inv()
        # No, this is simpler: the field vector B_world is static. The sensor's axes are rotated.
        # So we project B_world onto the sensor's rotated axes.
        # If sensor axes are s_x, s_y, s_z, then B_measured_x = B_world . s_x etc.
        # This is equivalent to rotating B_world by the inverse of the sensor's orientation.

        # Let R be the rotation that aligns the world axes with the sensor's actual (noisy) axes.
        # If the sensor undergoes roll(r), pitch(p), yaw(y) with respect to the world frame.
        # A vector V_sensor in sensor coordinates is V_world = R_rpy * V_sensor in world coordinates.
        # We have B_world. We want B_sensor. So B_sensor = R_rpy.inv() * B_world.
        rotation_error = R.from_euler(
            "xyz", [roll_error, pitch_error, yaw_error], degrees=False
        )
        b_noisy_vector = rotation_error.inv().apply(b_field_world_frame)

        # This b_noisy_vector currently holds GT + Motor effects, as perceived by misoriented sensor.

        # --- 3. Environmental (Diurnal) Drift ---
        # Calculate the progress through the flight as a fraction (0 to 1)
        flight_progress = (
            (x_pos - FLIGHT_START_X) / flight_length if flight_length > 0 else 0
        )

        # Calculate diurnal drift for each component based on a sine wave
        # The sine wave completes DIURNAL_PERIOD_FLIGHTS cycles over the flight duration.
        diurnal_sine_value = np.sin(
            2 * np.pi * DIURNAL_PERIOD_FLIGHTS * flight_progress
        )

        b_diurnal_drift_x = DIURNAL_AMPLITUDE_XT * diurnal_sine_value
        b_diurnal_drift_y = DIURNAL_AMPLITUDE_YT * diurnal_sine_value
        b_diurnal_drift_z = DIURNAL_AMPLITUDE_ZT * diurnal_sine_value
        diurnal_drift_vector = np.array(
            [b_diurnal_drift_x, b_diurnal_drift_y, b_diurnal_drift_z]
        )

        # Add diurnal drift to the noisy vector
        b_noisy_vector += (
            diurnal_drift_vector  # b_noisy_vector was (GT+Motor) rotated by IMU error
        )

        # Final noisy magnetic field strength
        b_noisy_total_strength = np.linalg.norm(b_noisy_vector)

        readings.append(
            {
                "pos_x": sensor_position[0],
                "pos_y": sensor_position[1],
                "pos_z": sensor_position[2],
                "b_x_gt": b_gt_vector[0],  # Ground Truth
                "b_y_gt": b_gt_vector[1],
                "b_z_gt": b_gt_vector[2],
                "b_total_strength_gt": b_gt_total_strength,
                "throttle": current_throttle,  # Use current_throttle
                "roc_throttle": rate_of_change_of_throttle,  # Store RoC of throttle
                # Storing the scalar motor noise and its Z-component contribution for clarity in V4
                "motor_noise_scalar_v4": motor_noise_scalar,
                "b_motor_contrib_z_v4": b_motor_noise_contribution_vector[
                    2
                ],  # This is motor_noise_scalar
                "b_x_noisy": b_noisy_vector[
                    0
                ],  # Final noisy field after all V2/V4 noises
                "b_y_noisy": b_noisy_vector[1],
                "b_z_noisy": b_noisy_vector[2],
                "b_total_strength_noisy": b_noisy_total_strength,
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
