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

-   `filter_v3.py`: Script for V2 data, using a simple `Throttle * K` filter model.
-   `filtered_comparison_v3.png`, `filtered_zoomed_v3.png`: Output plots from `filter_v3.py`.

-   `mag_sim_v4.py`: The V4 simulation script. It enhances `mag_sim_v2.py` by generating a more complex motor noise signal based on a 3-coefficient model: `Motor_Noise_Scalar = (Throttle * K1_ACTUAL) + (Throttle^2 * K2_ACTUAL) + (Rate_of_change_of_Throttle * K3_ACTUAL)`. This scalar noise is injected as a Z-component into the magnetic field before IMU and diurnal effects. Output is `magnetic_readings_complex_noisy_v4.csv`.
-   `filter_v4.py`: The V4 filter script, designed to work with data from `mag_sim_v4.py`. It attempts to recover the K1, K2, K3 coefficients by optimizing a 3-term filter model against the complex noisy data.
-   `magnetic_readings_complex_noisy_v4.csv`: Output data from `mag_sim_v4.py`.
-   `filtered_comparison_v4.png`, `filtered_zoomed_v4.png`: Output plots from `filter_v4.py`.


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

### Phase 3: Filtering Noisy Data

1.  **Run the Filtering Script:**
    After generating `magnetic_readings_noisy.csv` using `mag_sim_v2.py`, you can run the adaptive filter.
    ```bash
    python filter_v3.py
    ```
    This script will:
    - Load `magnetic_readings_noisy.csv`.
    - Calculate the initial RMSE between the noisy total field strength and the ground truth total field strength.
    - Optimize a coupling constant `K` to model the motor noise based on the throttle signal.
    - Apply the filter: `Filtered_Signal = Noisy_Signal - (Throttle * K)`.
    - Calculate the final RMSE between the filtered total field strength and the ground truth.
    - Print both RMSE values to the console.
    - Generate `filtered_comparison_v3.png`, a plot showing the ground truth, noisy, and filtered signals.
    - Generate `filtered_zoomed_v3.png`, a plot focusing on just the ground truth and filtered signals with a tight Y-axis scale to highlight the filter's precision.

### Version 4 (Complex Noise Simulation and Advanced Filtering)

1.  **Run the V4 Simulation:**
    Execute `mag_sim_v4.py` to generate a dataset with more complex, throttle-dependent motor noise.
    ```bash
    python mag_sim_v4.py
    ```
    This creates `magnetic_readings_complex_noisy_v4.csv`. Key parameters (`K1_ACTUAL`, `K2_ACTUAL`, `K3_ACTUAL`) for noise generation are inside this script.

2.  **Run the V4 Filter:**
    After generating the V4 data, run `filter_v4.py`.
    ```bash
    python filter_v4.py
    ```
    This script will:
    - Load `magnetic_readings_complex_noisy_v4.csv`.
    - Attempt to find optimal K1, K2, K3 coefficients for its 3-term filter model.
    - Print initial and final RMSE, and the recovered K1, K2, K3 values.
    - Generate `filtered_comparison_v4.png` and `filtered_zoomed_v4.png`.

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

### `mag_sim_v4.py` (In addition to V2 parameters)
-   **Complex Motor Noise Coefficients (Ground Truth for Generation):**
    -   `K1_ACTUAL`, `K2_ACTUAL`, `K3_ACTUAL`: Coefficients used to generate the motor noise based on throttle, throttle^2, and rate-of-change of throttle.

### `filter_v3.py`
- The script uses `scipy.optimize.minimize` to find an optimal scalar coupling constant `K`.
- The primary data being filtered is the total magnetic field strength (`b_total_strength_noisy` against `b_total_strength_gt` using `throttle` as the reference noise).

### `filter_v4.py`
- Similar to `filter_v3.py`, but attempts to optimize a 3-coefficient vector `[K1, K2, K3]` for a noise model: `(Throttle * K1) + (Throttle^2 * K2) + (Rate_of_change_of_Throttle * K3)`.
- It processes data from `magnetic_readings_complex_noisy_v4.csv`.

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
    -   `b_motor_x`, `b_motor_y`, `b_motor_z`: Magnetic field contribution from motor noise (in V2).
    -   `b_x_noisy`, `b_y_noisy`, `b_z_noisy`, `b_total_strength_noisy`: Final noisy magnetic field readings after all noise models applied.
-   `magnetic_field_map_v2.png`: Plot for V2 data.
-   `filtered_comparison_v3.png`, `filtered_zoomed_v3.png`: Plots for V3 filter (acting on V2 data).

### Version 4
-   `magnetic_readings_complex_noisy_v4.csv`:
    -   Includes columns from V2's noisy CSV.
    -   Adds `roc_throttle` (rate of change of throttle).
    -   Motor noise component is now based on K1, K2, K3 model; `motor_noise_scalar_v4` and `b_motor_contrib_z_v4` reflect this.
-   `filtered_comparison_v4.png`, `filtered_zoomed_v4.png`: Plots for V4 filter, showing results of the 3-coefficient model. Recovered K1, K2, K3 values are typically in labels/titles.
