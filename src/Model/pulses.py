# pulse.py
# Created by Gurudev Dutt <gdutt@pitt.edu> on 7/28/25
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
from __future__ import annotations
from typing import List, Dict
import numpy as np
from abc import ABC, abstractmethod

class Pulse(ABC):
    """
    Abstract base class for hardware-agnostic waveform pulses.
    Subclasses implement `generate_samples()` to return a float array of envelope values.
    """
    def __init__(self, name: str, length: int, fixed_timing: bool = False):
        """
        :param name: Identifier for this pulse
        :param length: Number of samples in the pulse envelope
        :param fixed_timing: If True, this pulse's timing should not be adjusted during scans
        """
        self.name = name
        self.length = length
        self.fixed_timing = fixed_timing

    @abstractmethod
    def generate_samples(self) -> np.ndarray:
        """
        Generate the normalized envelope of the pulse as a 1D float array of length `self.length`.
        """
        pass


class GaussianPulse(Pulse):
    """
    Gaussian-shaped pulse envelope.
    """
    def __init__(self, name: str, length: int, sigma: float, amplitude: float = 1.0, fixed_timing: bool = False):
        super().__init__(name, length, fixed_timing)
        self.sigma = sigma
        self.amplitude = amplitude
        # Center the Gaussian at the midpoint
        self.center = (length - 1) / 2.0

    def generate_samples(self) -> np.ndarray:
        t = np.arange(self.length)
        envelope = self.amplitude * np.exp(-((t - self.center)**2) / (2 * self.sigma**2))
        return envelope.astype(float)


class SechPulse(Pulse):
    """
    Hyperbolic secant-shaped pulse envelope.
    """
    def __init__(self, name: str, length: int, width: float, amplitude: float = 1.0, fixed_timing: bool = False):
        super().__init__(name, length, fixed_timing)
        self.width = width
        self.amplitude = amplitude
        self.center = (length - 1) / 2.0

    def generate_samples(self) -> np.ndarray:
        t = np.arange(self.length) - self.center
        envelope = self.amplitude * (1.0 / np.cosh(t / self.width))
        return envelope.astype(float)


class LorentzianPulse(Pulse):
    """
    Lorentzian-shaped pulse envelope.
    """
    def __init__(self, name: str, length: int, gamma: float, amplitude: float = 1.0, fixed_timing: bool = False):
        super().__init__(name, length, fixed_timing)
        self.gamma = gamma
        self.amplitude = amplitude
        self.center = (length - 1) / 2.0

    def generate_samples(self) -> np.ndarray:
        t = np.arange(self.length)
        envelope = self.amplitude * (self.gamma**2) / ((t - self.center)**2 + self.gamma**2)
        return envelope.astype(float)


class SquarePulse(Pulse):
    """
    Constant-amplitude (square) pulse envelope.
    """
    def __init__(self, name: str, length: int, amplitude: float = 1.0, fixed_timing: bool = False):
        super().__init__(name, length, fixed_timing)
        self.amplitude = amplitude

    def generate_samples(self) -> np.ndarray:
        return np.full(self.length, self.amplitude, dtype=float)


class DataPulse(Pulse):
    """
    Pulse defined by external data file (e.g. CSV of time vs amplitude).
    Automatically resamples to `length`.
    """
    def __init__(self, name: str, length: int, filename: str):
        super().__init__(name, length)
        self.filename = filename

    def generate_samples(self) -> np.ndarray:
        # Load data, skipping the first row (header) and using comma delimiter
        data = np.loadtxt(self.filename, delimiter=',', skiprows=1)
        # assume data[:,0] = time, data[:,1] = amplitude
        times = data[:,0]
        amps = data[:,1]
        # resample uniformly across times
        resampled_times = np.linspace(times[0], times[-1], num=self.length)
        envelope = np.interp(resampled_times, times, amps)
        return envelope.astype(float)


class MarkerEvent:
    """
    Represents a digital marker for a given pulse window.
    """
    def __init__(self, name: str, length: int, on_index: int, off_index: int):
        self.name = name
        self.length = length
        self.on_index = on_index
        self.off_index = off_index

    def generate_markers(self) -> np.ndarray:
        """
        Returns a binary (0/1) array of length `self.length` marking the event window.
        """
        markers = np.zeros(self.length, dtype=int)
        markers[self.on_index:self.off_index] = 1
        return markers


# ======================================================================
# Frequency-swept (adiabatic) pulses for chirp / HS DEER pumping.
# These are FREQUENCY-MODULATED: the baseband is complex, z(t)=A(t)e^{i*phi(t)},
# so the I channel carries Re[z] and Q carries Im[z]. Each pulse emits ONE
# quadrature (quadrature='I' or 'Q'); place an I copy on the I channel and a Q
# copy on the Q channel with identical parameters. bandwidth/center_freq are
# BASEBAND frequencies in Hz (the LO sets the RF center of the sweep).
# ======================================================================

def _phase_from_frequency(freq_hz: np.ndarray, sample_rate: float) -> np.ndarray:
    """Instantaneous phase (rad): 2*pi * cumulative integral of f(t), referenced to pulse center."""
    dt = 1.0 / sample_rate
    phase = 2.0 * np.pi * np.cumsum(freq_hz) * dt
    phase -= phase[len(phase) // 2]      # global phase reference at the temporal center
    return phase


def _apply_quadrature(amp_env: np.ndarray, phase: np.ndarray, quadrature: str, phase0: float = 0.0) -> np.ndarray:
    q = str(quadrature).upper()
    if q == "I":
        return (amp_env * np.cos(phase + phase0)).astype(float)
    if q == "Q":
        return (amp_env * np.sin(phase + phase0)).astype(float)
    raise ValueError(f"quadrature must be 'I' or 'Q' (got {quadrature!r})")


class ChirpPulse(Pulse):
    """
    Linear frequency-swept ('chirp') adiabatic inversion pulse, one IQ quadrature.
        amplitude(t) = A0                (flat top, optional quarter-sine edges)
        f(t)         = center_freq - bandwidth/2 + bandwidth * t/Tp   (linear sweep)
        I = A0 cos(phi(t)),  Q = A0 sin(phi(t)),   phi = 2*pi * integral f dt
    """
    def __init__(self, name: str, length: int, sample_rate: float, bandwidth: float,
                 amplitude: float = 1.0, quadrature: str = "I", center_freq: float = 0.0,
                 edge_fraction: float = 0.0, phase0: float = 0.0, fixed_timing: bool = False):
        super().__init__(name, length, fixed_timing)
        self.sample_rate = float(sample_rate)
        self.bandwidth = float(bandwidth)
        self.amplitude = float(amplitude)
        self.quadrature = quadrature
        self.center_freq = float(center_freq)
        self.edge_fraction = float(edge_fraction)
        self.phase0 = float(phase0)

    def generate_samples(self) -> np.ndarray:
        n = self.length
        if n <= 0:
            return np.zeros(0, dtype=float)
        t = np.arange(n) / self.sample_rate
        Tp = n / self.sample_rate
        f = self.center_freq - self.bandwidth / 2.0 + self.bandwidth * (t / Tp)
        amp = np.full(n, self.amplitude, dtype=float)
        ne = int(self.edge_fraction * n)
        if ne > 0:
            ramp = np.sin(np.linspace(0.0, np.pi / 2.0, ne))
            amp[:ne] *= ramp
            amp[-ne:] *= ramp[::-1]
        phase = _phase_from_frequency(f, self.sample_rate)
        return _apply_quadrature(amp, phase, self.quadrature, self.phase0)


class HyperbolicSecantPulse(Pulse):
    """
    Symmetric hyperbolic-secant (sech/tanh, 'HS1') adiabatic inversion pulse, one IQ quadrature.
        x            = linspace(-1, 1, N)                (normalized time, 0 at center)
        amplitude(t) = A0 * sech(beta * x)
        f(t)         = center_freq + (bandwidth/2) * tanh(beta*x)/tanh(beta)
    beta = dimensionless truncation (sech(beta) is the edge amplitude; beta ~ 5.3 -> ~1%).
    bandwidth = full edge-to-edge sweep in Hz.
    """
    def __init__(self, name: str, length: int, sample_rate: float, bandwidth: float,
                 beta: float = 5.3, amplitude: float = 1.0, quadrature: str = "I",
                 center_freq: float = 0.0, phase0: float = 0.0, fixed_timing: bool = False):
        super().__init__(name, length, fixed_timing)
        self.sample_rate = float(sample_rate)
        self.bandwidth = float(bandwidth)
        self.beta = float(beta)
        self.amplitude = float(amplitude)
        self.quadrature = quadrature
        self.center_freq = float(center_freq)
        self.phase0 = float(phase0)

    def generate_samples(self) -> np.ndarray:
        n = self.length
        if n <= 0:
            return np.zeros(0, dtype=float)
        x = np.linspace(-1.0, 1.0, n)
        amp = self.amplitude / np.cosh(self.beta * x)
        f = self.center_freq + (self.bandwidth / 2.0) * np.tanh(self.beta * x) / np.tanh(self.beta)
        phase = _phase_from_frequency(f, self.sample_rate)
        return _apply_quadrature(amp, phase, self.quadrature, self.phase0)


class AsymmetricHyperbolicSecantPulse(Pulse):
    """
    Asymmetric hyperbolic-secant pulse 'HS{n_left, n_right}' (e.g. HS{1,6}), one IQ quadrature.
    Offset-independent-adiabaticity (OIA) construction: stretched-sech amplitude with different
    steepness on the two halves; frequency sweep proportional to the running integral of amp^2
    (so the pulse dwells longest where B1 is largest).
        x            = linspace(-1, 1, N)
        n(x)         = n_left for x < 0, else n_right
        amplitude(t) = A0 * sech(beta * |x|**n(x))
        f(t)         = center_freq + bandwidth * (G(x) - 0.5),  G = normalized cumulative integral of amp^2
    n_left == n_right -> symmetric HSn; n_left == n_right == 1 -> HS1. Jeschke et al.: HS{1,6} best for short DEER traces.
    """
    def __init__(self, name: str, length: int, sample_rate: float, bandwidth: float,
                 n_left: float = 1.0, n_right: float = 6.0, beta: float = 5.3,
                 amplitude: float = 1.0, quadrature: str = "I", center_freq: float = 0.0,
                 phase0: float = 0.0, fixed_timing: bool = False):
        super().__init__(name, length, fixed_timing)
        self.sample_rate = float(sample_rate)
        self.bandwidth = float(bandwidth)
        self.n_left = float(n_left)
        self.n_right = float(n_right)
        self.beta = float(beta)
        self.amplitude = float(amplitude)
        self.quadrature = quadrature
        self.center_freq = float(center_freq)
        self.phase0 = float(phase0)

    def generate_samples(self) -> np.ndarray:
        n = self.length
        if n <= 0:
            return np.zeros(0, dtype=float)
        x = np.linspace(-1.0, 1.0, n)
        order = np.where(x < 0.0, self.n_left, self.n_right)
        amp = self.amplitude / np.cosh(self.beta * np.abs(x) ** order)
        g = np.cumsum(amp ** 2)
        g = (g - g[0]) / (g[-1] - g[0])          # normalize running integral to [0, 1]
        f = self.center_freq + self.bandwidth * (g - 0.5)
        phase = _phase_from_frequency(f, self.sample_rate)
        return _apply_quadrature(amp, phase, self.quadrature, self.phase0)


def adiabaticity_Q(bandwidth_hz: float, duration_s: float, rabi_hz: float) -> float:
    """
    Adiabaticity parameter Q = Omega1^2 / |d(delta)/dt| at resonance (Appendix A). Need Q >> 1.
    For a linear sweep |d(delta)/dt| = 2*pi*bandwidth/duration. rabi_hz = Omega1/2pi = gamma*B1/2pi.
    """
    sweep_rate = 2.0 * np.pi * bandwidth_hz / duration_s
    omega1 = 2.0 * np.pi * rabi_hz
    return (omega1 ** 2) / sweep_rate