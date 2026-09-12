"""
idr_pipeline/vibration_filter.py
================================
AI Speed & Vibration / Pothole / Shock Filter.

Detects and filters out non-navigation motions:
1. Engine Idling Vibrations:
   Stationary harmonic chassis vibrations (15-30 Hz) while translational velocity
   is zero. Distinguishes idling at traffic lights from true vehicle motion,
   preventing the dead-reckoning integrator from drifting during stops.

2. Pothole & Road Bump Impulse Shocks:
   Sudden vertical/lateral shock accelerations (>2.5g or >4-sigma transient spikes)
   caused by potholes, speed breakers, rumble strips, or expansion joints.
   Replaces spikes with rolling median kinematic values so they do not corrupt
   kinematic integration or AI speed estimation.

3. Accidental Mount Displacement & Shocks:
   Isolates short-duration mechanical shock artifacts from sustained vehicle acceleration.
"""

import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class VibrationState:
    is_engine_idling: bool = False
    is_pothole_detected: bool = False
    pothole_event_count: int = 0
    cleaned_accel: np.ndarray = None
    cleaned_gyro: np.ndarray = None
    vibration_energy: float = 0.0
    spectral_entropy: float = 0.0


class VibrationPotholeFilter:
    """
    Statistical and spectral filter for MEMS IMU signals operating on smartphones.
    Filters high-frequency road anomalies, detects potholes, and flags engine idling.
    """

    def __init__(
        self,
        dt: float = 0.1,
        pothole_threshold_g: float = 2.2,  # Multiplier above 1g for pothole impact detection
        engine_idle_var_min: float = 0.02, # Minimum vibration variance indicating running engine
        engine_idle_var_max: float = 0.25, # Maximum variance for idle (above this indicates vehicle travel)
        window_size: int = 10
    ):
        self.dt = dt
        self.g_val = 9.80665
        self.pothole_thresh = pothole_threshold_g * self.g_val
        self.idle_var_min = engine_idle_var_min
        self.idle_var_max = engine_idle_var_max
        self.window_size = window_size

        # Rolling buffers for sliding window filtering
        self._accel_buffer: List[np.ndarray] = []
        self._gyro_buffer: List[np.ndarray] = []
        self.pothole_count = 0
        self._pothole_cooldown = 0

    def process(
        self,
        accel_b: np.ndarray,
        gyro_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, VibrationState]:
        """
        Processes a single IMU measurement in the aligned vehicle body frame.
        
        Args:
            accel_b: (3,) vehicle body acceleration [ax, ay, az]
            gyro_b: (3,) vehicle body angular rate [wx, wy, wz]
            
        Returns:
            cleaned_accel: (3,) sanitized acceleration
            cleaned_gyro: (3,) sanitized angular rate
            state: VibrationState with anomaly metrics
        """
        self._accel_buffer.append(accel_b.copy())
        self._gyro_buffer.append(gyro_b.copy())

        if len(self._accel_buffer) > self.window_size:
            self._accel_buffer.pop(0)
            self._gyro_buffer.pop(0)

        accel_window = np.array(self._accel_buffer)
        gyro_window = np.array(self._gyro_buffer)

        cleaned_accel = accel_b.copy()
        cleaned_gyro = gyro_b.copy()

        # --- 1. Pothole / Shock Detection & Blanking ---
        # A pothole produces a sharp vertical impulse: az deviates sharply from g
        az = accel_b[2]
        vertical_deviation = np.abs(az - self.g_val)
        horizontal_norm = np.linalg.norm(accel_b[0:2])

        is_pothole = False
        if self._pothole_cooldown > 0:
            self._pothole_cooldown -= 1

        if (vertical_deviation > self.pothole_thresh or horizontal_norm > 15.0) and self._pothole_cooldown == 0:
            is_pothole = True
            self.pothole_count += 1
            self._pothole_cooldown = 5 # 0.5s cooldown

            # Blanking / Median replacement: Replace spike with rolling median of recent window
            if len(self._accel_buffer) >= 3:
                median_acc = np.median(accel_window[:-1], axis=0)
                # Keep horizontal direction if plausible, but clamp vertical shock spike to 1g
                cleaned_accel[2] = self.g_val + 0.1 * (median_acc[2] - self.g_val)
                cleaned_accel[0] = median_acc[0]
                cleaned_accel[1] = median_acc[1]

        # --- 2. Engine Idling Detection ---
        # When engine is idling:
        # - Net forward/lateral motion is negligible (mean horizontal accel ~ 0, yaw rate ~ 0)
        # - High frequency vibration variance is present in z/x axes
        mean_horiz_acc = np.mean(np.abs(accel_window[:, 0]))
        mean_lat_acc = np.mean(np.abs(accel_window[:, 1]))
        mean_yaw_rate = np.mean(np.abs(gyro_window[:, 2]))
        acc_variance = float(np.var(accel_window[:, 2]))

        is_idling = False
        if len(self._accel_buffer) >= self.window_size:
            if (mean_horiz_acc < 0.25 and
                mean_lat_acc < 0.20 and
                mean_yaw_rate < 0.03 and
                self.idle_var_min <= acc_variance <= self.idle_var_max):
                is_idling = True
                # Vehicle is stationary: force zero velocity indicators
                cleaned_accel[0] = 0.0
                cleaned_accel[1] = 0.0
                cleaned_gyro[2] = 0.0

        # --- 3. Vibration Energy & Spectral Metrics ---
        vib_energy = float(np.mean((accel_window[:, 0] - np.mean(accel_window[:, 0]))**2 +
                                   (accel_window[:, 2] - np.mean(accel_window[:, 2]))**2))

        # Normalized spectral entropy across sliding window
        fft_vals = np.abs(np.fft.rfft(accel_window[:, 2]))
        p_fft = fft_vals / (np.sum(fft_vals) + 1e-6)
        spectral_entropy = -float(np.sum(p_fft * np.log(p_fft + 1e-9)))

        state = VibrationState(
            is_engine_idling=is_idling,
            is_pothole_detected=is_pothole,
            pothole_event_count=self.pothole_count,
            cleaned_accel=cleaned_accel,
            cleaned_gyro=cleaned_gyro,
            vibration_energy=vib_energy,
            spectral_entropy=spectral_entropy
        )

        return cleaned_accel, cleaned_gyro, state

    def filter_batch(
        self,
        accels: np.ndarray,
        gyros: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[VibrationState]]:
        """Batch processing helper for full trajectory arrays."""
        N = len(accels)
        cleaned_accels = np.zeros_like(accels)
        cleaned_gyros = np.zeros_like(gyros)
        states = []

        for i in range(N):
            ca, cg, st = self.process(accels[i], gyros[i])
            cleaned_accels[i] = ca
            cleaned_gyros[i] = cg
            states.append(st)

        return cleaned_accels, cleaned_gyros, states
