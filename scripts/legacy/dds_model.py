"""
Title:     dds_model.py
Purpose:   Defines and fits the Dual Damped Sine (DDS) model for TMS–EEG TEP analysis.
           Includes single- and dual-component exponentially damped sinusoids, curve fitting,
           and ROI averaging utility for MNE Evoked objects.

Author:    Damian Jan (damian.jan@dejsl.com)
Affiliation:
           - Instituto Universitario de Neurociencias, Universidad de La Laguna, Spain
           - Dynamic Enterprise Junction Consulting S.L., Spain

Dependencies:
    - numpy
    - scipy (curve_fit)
    - MNE-Python (for roi_average)

Main functions:
    - fit_dds(...) : Fit one- or two-component DDS to 1D signal (e.g., ROI-averaged TEP)
    - roi_average(...) : Average Evoked data across selected EEG channels

Model:
    y(t) = A1·e^(−γ1·t)·sin(2π·f1·t + φ1) + A2·e^(−γ2·t)·sin(2π·f2·t + φ2) + offset
    Each component starts at t0 and is gated by a Heaviside step function.

Usage:
    This module is intended to be imported into a higher-level runner (e.g., run_dds_ds001849.py).
    Ensure input signals are preprocessed and time-locked to TMS onset.

License:   MIT License 

Citation:
    If you use this module, please cite the accompanying Journal of Neuroscience Methods paper:
    "A Dual Damped Sine (DDS) model for TMS–EEG: parametric validation against cosine similarity on OpenNeuro ds001849"
"""

import numpy as np
from scipy.optimize import curve_fit

def _fs(time_s):
    """Sampling rate inferred from time vector (Hz)."""
    return 1.0 / float(np.median(np.diff(time_s)))

def _heaviside(t):
    return (t >= 0).astype(float)

def _two_damped_sinusoids(t, A1, gamma1, f1, phi1, A2, gamma2, f2, phi2, offset, t0):
    """Sum of two exponentially damped sinusoids starting at t0 (s)."""
    tt = t - t0
    H = _heaviside(tt)
    y1 = A1 * np.exp(-gamma1 * tt) * np.sin(2*np.pi*f1*tt + phi1) * H
    y2 = A2 * np.exp(-gamma2 * tt) * np.sin(2*np.pi*f2*tt + phi2) * H
    return y1 + y2 + offset

def _one_damped_sinusoid(t, A1, gamma1, f1, phi1, offset, t0):
    tt = t - t0
    H = _heaviside(tt)
    y1 = A1 * np.exp(-gamma1 * tt) * np.sin(2*np.pi*f1*tt + phi1) * H
    return y1 + offset

def fit_dds(time_s, signal, tmin=0.005, tmax=0.100, two_components=True, t0=0.0, h_freq=100.0):
    """
    Fit damped sinusoid model(s) to a 1D TEP signal (e.g., ROI average).

    Parameters
    ----------
    time_s : array, seconds
    signal : array, microvolts (or arbitrary units)
    tmin, tmax : float, fit window relative to stimulus (s)
    two_components : bool, if False use single component
    t0 : float, onset in seconds (usually 0.0)
    h_freq : float, low-pass cutoff used in preprocessing (Hz) -> to cap f1/f2

    Returns
    -------
    params : dict with fitted parameters and fit-quality metrics (R2, RMSE)
    model_y : array, model evaluated on time_s
    """
    idx = (time_s >= tmin) & (time_s <= tmax)
    t = time_s[idx]
    y = signal[idx]
    if len(t) < 10:
        raise ValueError("Not enough points in the fit window.")

    # Sampling and dynamic frequency ceiling
    fs   = _fs(time_s)
    nyq  = 0.5 * fs
    # Máximo de búsqueda: limitado por el low-pass y por Nyquist
    fmax = float(min(0.95 * h_freq, 0.90 * nyq))
    # Seguridad por si fs/h_freq fueran raros
    if fmax <= 5.0:
        fmax = max(5.0, min(0.90 * nyq, h_freq))  # nunca >Nyquist

    # Initial guesses
    A_guess = float(np.nanmax(np.abs(y)) or 1e-6)
    f1_0, f2_0 = 10.0, min(30.0, 0.5 * fmax)   # seeds razonables
    g1_0, g2_0 = 60.0, 120.0                   # 1/s (≈ 5–17 ms y 8–17 ms)
    phi0 = 0.0
    offset0 = float(np.nanmedian(y))

    # Bounds (A ≥ 0, el signo lo resuelve la fase; φ ∈ [-π, π])
    bounds_single = (
        [0.0,        10.0,   4.0,   -np.pi, -5.0,  t0 - 0.002],
        [10*A_guess, 400.0,  fmax,   np.pi,  5.0,  t0 + 0.010],
    )
    bounds_double = (
        [0.0,        10.0,   4.0,   -np.pi,  0.0,       10.0,   4.0,   -np.pi, -5.0,  t0 - 0.002],
        [10*A_guess, 400.0,  fmax,   np.pi,  10*A_guess,400.0,  fmax,   np.pi,  5.0,  t0 + 0.010],
    )

    if two_components:
        p0 = [0.7*A_guess, g1_0, f1_0, phi0, 0.3*A_guess, g2_0, f2_0, phi0, offset0, t0]
        func = _two_damped_sinusoids
        bounds = bounds_double
        names = ["A1","gamma1","f1","phi1","A2","gamma2","f2","phi2","offset","t0"]
    else:
        p0 = [A_guess, g1_0, f1_0, phi0, offset0, t0]
        func = _one_damped_sinusoid
        bounds = bounds_single
        names = ["A1","gamma1","f1","phi1","offset","t0"]

    try:
        popt, pcov = curve_fit(func, t, y, p0=p0, bounds=bounds, maxfev=40000)
    except Exception as e:
        raise RuntimeError(f"curve_fit failed: {e}")

    y_hat = func(time_s, *popt)

    # Goodness of fit en la ventana del ajuste
    y_hat_win = func(t, *popt)
    ss_res = float(np.nansum((y - y_hat_win)**2))
    ss_tot = float(np.nansum((y - np.nanmean(y))**2))
    R2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    RMSE = float(np.sqrt(ss_res / (len(y) if len(y) else np.nan)))

    params = dict(zip(names, [float(v) for v in popt]))
    params.update({
        "R2": R2, "RMSE": RMSE, "two_components": bool(two_components),
        "tmin": float(tmin), "tmax": float(tmax),
        "fmax_used": float(fmax), "fs": float(fs), "h_freq": float(h_freq)
    })
    return params, y_hat


def roi_average(evoked, picks):
    """Return 1D ROI-averaged signal from an mne.Evoked (channels x time)."""
    import numpy as np
    data = evoked.copy().pick(picks).data
    return np.nanmean(data, axis=0)
