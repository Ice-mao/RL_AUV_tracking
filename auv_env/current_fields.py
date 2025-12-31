"""
Ocean Current Field Functions for AUV Tracking Environment

This module provides various ocean current field implementations that can be
used to simulate realistic underwater currents affecting the AUV.

All field functions follow the signature:
    field(location, time, **params) -> np.ndarray[3]

where location is [x, y, z] and time is simulation time in seconds.
"""

import numpy as np
from typing import Callable, Dict, Optional


def vortex_field(location: np.ndarray, time: float, **params) -> np.ndarray:
    """
    Circular vortex current around a center point with optional time-dependent rotation.

    Parameters
    ----------
    location : np.ndarray, shape (3,)
        Agent position [x, y, z] in global coordinates
    time : float
        Current simulation time in seconds
    **params : dict
        center : list[float], optional
            Vortex center position [x, y, z] (default: [0, 0, -10])
        strength : float, optional
            Tangential velocity magnitude at reference radius (default: 1.0 m/s)
        radius : float, optional
            Reference radius for velocity calculation (default: 5.0 m)
        angular_velocity : float, optional
            Rotation speed of vortex center (default: 0.0 rad/s)
        decay_rate : float, optional
            How fast velocity decays with distance (default: 1.0)

    Returns
    -------
    current_velocity : np.ndarray, shape (3,)
        Current velocity vector [dx, dy, dz] in m/s
    """
    center = np.array(params.get('center', [0.0, 0.0, -10.0]))
    strength = params.get('strength', 1.0)
    radius = params.get('radius', 5.0)
    angular_velocity = params.get('angular_velocity', 0.0)
    decay_rate = params.get('decay_rate', 1.0)

    # Calculate relative position
    rel_pos = location - center
    x, y, z = rel_pos

    # Only apply current if underwater (z < 0)
    if z > 0:
        return np.zeros(3)

    # Calculate horizontal distance
    r_horizontal_squared = x**2 + y**2 + 1e-6  # Avoid divide by zero
    r_horizontal = np.sqrt(r_horizontal_squared)

    # Tangential velocity (perpendicular to radial direction)
    velocity_magnitude = strength * (radius / (r_horizontal + radius)) ** decay_rate

    # Tangential direction: rotate radial vector by 90 degrees in xy-plane
    # Add time-dependent phase shift to create rotating vortex
    time_phase = angular_velocity * time
    cos_phase = np.cos(time_phase)
    sin_phase = np.sin(time_phase)

    # Base tangential velocity
    dx_base = -y / r_horizontal * velocity_magnitude
    dy_base = x / r_horizontal * velocity_magnitude

    # Apply rotation due to angular velocity
    dx = dx_base * cos_phase - dy_base * sin_phase
    dy = dx_base * sin_phase + dy_base * cos_phase

    # Vertical component (spiral motion with time variation)
    dz = 0.2 * strength * np.cos(0.1 * r_horizontal_squared + time_phase)

    return np.array([dx, dy, dz])


def uniform_field(location: np.ndarray, time: float, **params) -> np.ndarray:
    """
    Constant current in a fixed direction with optional time modulation.

    Parameters
    ----------
    location : np.ndarray, shape (3,)
        Agent position [x, y, z] in global coordinates
    time : float
        Current simulation time in seconds
    **params : dict
        velocity : list[float], optional
            Constant velocity vector [dx, dy, dz] (default: [0.5, 0.0, 0.0] m/s)
        time_modulation : bool, optional
            Enable sinusoidal time variation (default: False)
        modulation_period : float, optional
            Period for time variation in seconds (default: 10.0 s)
        modulation_amplitude : float, optional
            Amplitude multiplier for modulation (default: 0.5)

    Returns
    -------
    current_velocity : np.ndarray, shape (3,)
        Current velocity vector [dx, dy, dz] in m/s
    """
    velocity = np.array(params.get('velocity', [0.5, 0.0, 0.0]))
    time_modulation = params.get('time_modulation', False)
    modulation_period = params.get('modulation_period', 10.0)
    modulation_amplitude = params.get('modulation_amplitude', 0.5)

    # Only apply current if underwater (z < 0)
    if location[2] > 0:
        return np.zeros(3)

    if time_modulation:
        # Sinusoidal modulation: base + amplitude * sin(2π * t / period)
        modulation_factor = 1.0 + modulation_amplitude * np.sin(2 * np.pi * time / modulation_period)
        return velocity * modulation_factor
    else:
        return velocity.copy()


def random_field(location: np.ndarray, time: float, **params) -> np.ndarray:
    """
    Spatially-varying random turbulence using Perlin-like noise.

    Parameters
    ----------
    location : np.ndarray, shape (3,)
        Agent position [x, y, z] in global coordinates
    time : float
        Current simulation time in seconds
    **params : dict
        max_velocity : float, optional
            Maximum current speed (default: 1.0 m/s)
        spatial_scale : float, optional
            Spatial frequency of variation in meters (default: 10.0 m)
        temporal_scale : float, optional
            How fast field changes in time (default: 5.0 s)
        seed : int, optional
            Random seed for reproducibility (default: None)
        correlation_length : float, optional
            Spatial correlation length (default: 3.0 m)

    Returns
    -------
    current_velocity : np.ndarray, shape (3,)
        Current velocity vector [dx, dy, dz] in m/s
    """
    max_velocity = params.get('max_velocity', 1.0)
    spatial_scale = params.get('spatial_scale', 10.0)
    temporal_scale = params.get('temporal_scale', 5.0)
    seed = params.get('seed', None)
    correlation_length = params.get('correlation_length', 3.0)

    # Only apply current if underwater (z < 0)
    if location[2] > 0:
        return np.zeros(3)

    # Use hash-based pseudo-random noise for deterministic but spatially-varying currents
    if seed is not None:
        np.random.seed(seed)

    # Create spatiotemporal coordinates
    x_coord = location[0] / spatial_scale + time / temporal_scale
    y_coord = location[1] / spatial_scale + time / temporal_scale
    z_coord = location[2] / spatial_scale + time / temporal_scale

    # Generate pseudo-random values using sine waves (simple Perlin-like approach)
    # Each velocity component uses different frequencies
    dx = max_velocity * np.sin(x_coord * 2.3 + np.cos(y_coord * 1.7)) * np.cos(time / temporal_scale)
    dy = max_velocity * np.sin(y_coord * 1.9 + np.cos(z_coord * 2.1)) * np.cos(time / temporal_scale + 1.0)
    dz = max_velocity * 0.3 * np.sin(z_coord * 2.7 + np.cos(x_coord * 1.5)) * np.cos(time / temporal_scale + 2.0)

    # Apply spatial correlation (smooth out the noise)
    dist_factor = np.exp(-np.linalg.norm(location[:2]) / correlation_length)
    velocity = np.array([dx, dy, dz]) * dist_factor

    return velocity


def sine_wave_field(location: np.ndarray, time: float, **params) -> np.ndarray:
    """
    Oscillating wave-like currents propagating through space.

    Parameters
    ----------
    location : np.ndarray, shape (3,)
        Agent position [x, y, z] in global coordinates
    time : float
        Current simulation time in seconds
    **params : dict
        direction : list[float], optional
            Wave propagation direction [dx, dy, dz] (default: [1, 0, 0])
        amplitude : float, optional
            Current amplitude (default: 1.0 m/s)
        wavelength : float, optional
            Spatial wavelength in meters (default: 10.0 m)
        period : float, optional
            Temporal period in seconds (default: 5.0 s)
        phase : float, optional
            Initial phase offset in radians (default: 0.0)

    Returns
    -------
    current_velocity : np.ndarray, shape (3,)
        Current velocity vector [dx, dy, dz] in m/s
    """
    direction = np.array(params.get('direction', [1.0, 0.0, 0.0]))
    amplitude = params.get('amplitude', 1.0)
    wavelength = params.get('wavelength', 10.0)
    period = params.get('period', 5.0)
    phase = params.get('phase', 0.0)

    # Only apply current if underwater (z < 0)
    if location[2] > 0:
        return np.zeros(3)

    # Normalize direction
    direction_norm = direction / (np.linalg.norm(direction) + 1e-6)

    # Calculate wave phase: k·r - ωt + φ
    # where k = 2π/λ (wave number), ω = 2π/T (angular frequency)
    wave_number = 2 * np.pi / wavelength
    angular_frequency = 2 * np.pi / period

    # Project position onto wave direction
    position_along_wave = np.dot(location, direction_norm)
    wave_phase = wave_number * position_along_wave - angular_frequency * time + phase

    # Velocity oscillates in the direction of wave propagation
    velocity_magnitude = amplitude * np.sin(wave_phase)
    velocity = direction_norm * velocity_magnitude

    # Add perpendicular component for more realistic flow
    # Perpendicular direction in xy-plane
    if abs(direction_norm[0]) > 0.1 or abs(direction_norm[1]) > 0.1:
        perp_direction = np.array([-direction_norm[1], direction_norm[0], 0.0])
        perp_direction /= (np.linalg.norm(perp_direction) + 1e-6)
        perp_magnitude = amplitude * 0.3 * np.cos(wave_phase)
        velocity += perp_direction * perp_magnitude

    return velocity


# Registry of available current field functions
CURRENT_FIELD_REGISTRY: Dict[str, Callable] = {
    'vortex': vortex_field,
    'uniform': uniform_field,
    'random': random_field,
    'sine_wave': sine_wave_field,
}


def get_current_field(field_type: str) -> Callable:
    """
    Retrieve a current field implementation by name.

    Parameters
    ----------
    field_type : str
        Name of the field type ('vortex', 'uniform', 'random', 'sine_wave')

    Returns
    -------
    field_func : callable
        Current field function with signature (location, time, **params) -> velocity

    Raises
    ------
    ValueError
        If field_type is not recognized
    """
    if field_type not in CURRENT_FIELD_REGISTRY:
        available_types = ', '.join(CURRENT_FIELD_REGISTRY.keys())
        raise ValueError(
            f"Unknown current field type: '{field_type}'. "
            f"Available types: {available_types}"
        )
    return CURRENT_FIELD_REGISTRY[field_type]


def create_current_field(config: Dict) -> Optional[Callable]:
    """
    Create a current field function from configuration.

    This factory function returns a closure that captures the field parameters,
    providing a simple interface: current_field(location, time) -> velocity

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys:
        - 'type': str, field type name
        - 'params': dict, field-specific parameters

    Returns
    -------
    field_func : callable or None
        Function that takes (location, time) and returns velocity [dx, dy, dz].
        Returns None if type is not specified.

    Examples
    --------
    >>> config = {
    ...     'type': 'vortex',
    ...     'params': {'center': [0, 0, -10], 'strength': 2.0}
    ... }
    >>> current_field = create_current_field(config)
    >>> velocity = current_field(np.array([5, 5, -10]), 0.0)
    """
    field_type = config.get('type', None)
    if field_type is None:
        return None

    params = config.get('params', {})

    try:
        field_func = get_current_field(field_type)
    except ValueError as e:
        print(f"Warning: {e}")
        return None

    # Return a closure that captures params
    def current_field_closure(location: np.ndarray, time: float) -> np.ndarray:
        """Closure that applies the configured current field"""
        return field_func(location, time, **params)

    return current_field_closure
