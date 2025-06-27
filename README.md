# Magnetic Anomaly Simulator (V1 and V2)

This project simulates a magnetic anomaly (e.g., a buried dipole) and a sensor's flight path to collect magnetic field data.
Version 2 (`mag_sim_v2.py`) enhances the simulation by incorporating realistic noise models.

## Project Structure

-   `mag_sim_v1.py`: The original simulation script. It defines a 3D space, a magnetic dipole, a flight path, calculates ideal magnetic field readings, and saves them to `magnetic_readings.csv`.
-   `plot_mag_data.py`: A script to read data from `magnetic_readings.csv` (V1 output) and create a plot of the ideal magnetic field strength.
-   `magnetic_readings.csv`: Output data file from `mag_sim_v1.py`.
-   `magnetic_field_map.png`: Output plot from `plot_mag_data.py`.

-   `mag_sim_v2.py`: The V2 simulation script. It builds upon V1 by adding:
    -   Dominant Motor Noise: Simulates magnetic interference from a drone motor as a dipole whose strength varies with throttle.
    -   Sensor Orientation (IMU) Noise: Adds random Gaussian noise to the sensor's roll, pitch, and yaw, affecting how it perceives the magnetic field.
    -   Environmental (Diurnal) Drift: Simulates slow changes in the Earth's background magnetic field using a low-frequency sine wave.
    It saves the ground truth and noisy readings to `magnetic_readings_noisy.csv`.
-   `plot_mag_data_v2.py`: A script to read data from `magnetic_readings_noisy.csv` (V2 output). It generates a comparative plot showing the ground truth signal, the noisy signal, and the simulated throttle value.
-   `magnetic_readings_noisy.csv`: Output data file from `mag_sim_v2.py`.
-   `magnetic_field_map_v2.png`: Output plot from `plot_mag_data_v2.py`.

-   `AGENTS.md`: Instructions for AI agents working with this codebase (e.g., code formatting).

## Prerequisites

-   Python 3.x
-   NumPy: `pip install numpy`
-   Pandas: `pip install pandas`
-   Matplotlib: `pip install matplotlib`
-   SciPy: `pip install scipy` (used in V2 for rotation calculations)

You can install all dependencies at once using:
`pip install numpy pandas matplotlib scipy black` (black is for formatting)

## How to Run

### Version 1 (Ideal Simulation)

1.  **Run the Simulation:**
    ```bash
    python mag_sim_v1.py
    ```
    This creates `magnetic_readings.csv`. Parameters can be adjusted in `mag_sim_v1.py`.

2.  **Plot the Data:**
    ```bash
    python plot_mag_data.py
    ```
    This creates `magnetic_field_map.png`.

### Version 2 (Simulation with Noise)

1.  **Run the Simulation:**
    Execute the `mag_sim_v2.py` script to generate magnetic field data with added noise.
    ```bash
    python mag_sim_v2.py
    ```
    This will create a file named `magnetic_readings_noisy.csv`.
    Simulation and noise parameters can be adjusted directly within the `mag_sim_v2.py` script.

2.  **Plot the Data:**
    After `mag_sim_v2.py` has generated `magnetic_readings_noisy.csv`, run `plot_mag_data_v2.py` to visualize the data.
    ```bash
    python plot_mag_data_v2.py
    ```
    This will create `magnetic_field_map_v2.png`, showing the ground truth signal, the noisy signal, and the throttle values.

## Customization

### `mag_sim_v1.py` / `mag_sim_v2.py` (Common Parameters)
-   **Dipole Parameters (Ground Truth Anomaly):**
    -   `DIPOLE_POSITION`: Coordinates (x, y, z) of the dipole in meters.
    -   `DIPOLE_MOMENT_STRENGTH`: Strength of the dipole in A*m^2.
    -   `DIPOLE_ORIENTATION_THETA`, `DIPOLE_ORIENTATION_PHI`: Orientation angles.
-   **Flight Path:**
    -   `FLIGHT_ALTITUDE`, `FLIGHT_START_X`, `FLIGHT_END_X`, `FLIGHT_Y_POSITION`, `SAMPLING_INTERVAL`.
-   **3D Space:**
    -   `SPACE_DIMS`.

### `mag_sim_v2.py` (Noise Model Parameters)
-   **Motor Noise:**
    -   `MOTOR_REL_POSITION`: Position of the motor dipole relative to the sensor.
    -   `MOTOR_BASE_MOMENT_STRENGTH`: Base strength of the motor dipole at full throttle.
    -   `MOTOR_DIPOLE_MOMENT_ORIENTATION_VECTOR`: Orientation of the motor's magnetic moment.
    -   `THROTTLE_MIN`, `THROTTLE_MAX`, `THROTTLE_OSCILLATIONS`: Parameters for simulating throttle changes.
-   **IMU Noise:**
    -   `IMU_ROLL_STD_DEV`, `IMU_PITCH_STD_DEV`, `IMU_YAW_STD_DEV`: Standard deviations for orientation errors (in radians).
-   **Diurnal Drift:**
    -   `DIURNAL_PERIOD_FLIGHTS`: Number of diurnal sine wave cycles over the flight duration.
    -   `DIURNAL_AMPLITUDE_XT`, `DIURNAL_AMPLITUDE_YT`, `DIURNAL_AMPLITUDE_ZT`: Max drift amplitudes in Tesla for X, Y, Z components.

## Output Files

### Version 1
-   `magnetic_readings.csv`:
    -   Columns: `pos_x`, `pos_y`, `pos_z`, `b_x`, `b_y`, `b_z`, `b_total_strength`.
-   `magnetic_field_map.png`: Plot of total magnetic field strength.

### Version 2
-   `magnetic_readings_noisy.csv`:
    -   `pos_x`, `pos_y`, `pos_z`: Sensor position.
    -   `b_x_gt`, `b_y_gt`, `b_z_gt`, `b_total_strength_gt`: Ground truth magnetic field.
    -   `throttle`: Simulated motor throttle value.
    -   `b_motor_x`, `b_motor_y`, `b_motor_z`: Magnetic field contribution from motor noise.
    -   `b_x_noisy`, `b_y_noisy`, `b_z_noisy`, `b_total_strength_noisy`: Final noisy magnetic field readings after all noise models applied.
-   `magnetic_field_map_v2.png`: A multi-panel plot showing:
    -   Comparison of ground truth vs. noisy total field strength.
    -   Simulated throttle value over the flight path.
