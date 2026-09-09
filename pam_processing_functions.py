"""
Signal-processing functions for spectrogram denoising and enhancement.

This module contains several complementary methods for improving the
signal-to-noise ratio (SNR) of acoustic spectrograms, including:

    - Time domain wavelet denoising
    - 2-D (frequency-time) wavelet denoising
    - Frequency-dependent spectrogram pre-whitening
    - Spectrogram pre-whitening
    - Iterative adaptive Wiener filtering in time-frequency domain
    - Sliding-window SVD background removal
    - Spectrogram resizing with energy conservation
    - A configurable refinement loop
    - Wavelet-based directional enhancement
    - Per-Channel Energy Normalization (PCEN)

Most functions operate on 2-D spectrograms with the convention:

    axis 0 -> frequency
    axis 1 -> time

Unless otherwise specified, spectrogram values are assumed to be
non-negative linear magnitude values rather than logarithmic
decibel values.
"""

# =============================================================================
# =============================================================================
# ================================= Libraries =================================
# =============================================================================
# =============================================================================

# ---- Third-party libraries
import numpy as np
import pywt
from maad.sound import median_equalizer
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import svd
from scipy.ndimage import (gaussian_filter, gaussian_filter1d)

# =============================================================================
# =============================================================================
# ================================= FUNCTIONS =================================
# =============================================================================
# =============================================================================

# =====================================================
# TIME-DOMAIN WAVELET DENOISING
# =====================================================
def wavelet_denoise_time(
    audio,
    level=None,
    threshold_scale=0.1,
):
    """
    Denoise a time-domain audio signal using wavelet thresholding.

    The signal is decomposed into approximation and detail coefficients
    using a discrete wavelet transform. Noise level is estimated
    independently at each decomposition scale using the Median Absolute
    Deviation (MAD). A scale-dependent threshold is then applied to the
    detail coefficients using soft thresholding.

    Parameters
    ----------
    audio : array-like
        Input time-domain audio signal.
    level : int, optional
        Wavelet decomposition level. If None, the maximum allowable
        decomposition level is used.
    threshold_scale : float, optional
        Scaling factor applied to the wavelet threshold at each
        decomposition level. Default is 0.1.

    Returns
    -------
    numpy.ndarray
        Denoised audio signal with the same length as the input.
    """
    
    # Wavelet used for the decomposition.
    wavelet_t = 'db10'
    wave = pywt.Wavelet(wavelet_t)

    # Determine the maximum decomposition level supported by the signal.
    max_level = pywt.dwt_max_level(len(audio), wave.dec_len)

    if level is None:
        level = max_level

    # Decompose the signal into approximation and detail coefficients.
    coeffs = pywt.wavedec(audio, wavelet_t, level=int(level))

    # Preserve the approximation coefficients without thresholding.
    coeffs_thresh = [coeffs[0]]

    # Apply scale-dependent soft thresholding to the detail coefficients.
    for c in coeffs[1:]:
        # Estimate the noise level using the median absolute deviation (MAD).
        sigma = np.median(np.abs(c)) / 0.6745

        # Compute a scale-dependent universal threshold.
        T = threshold_scale * sigma * np.sqrt(2 * np.log(len(c)))

        # Apply soft thresholding to suppress low-amplitude coefficients.
        coeffs_thresh.append(pywt.threshold(c, T, mode="soft"))

    # Reconstruct the denoised signal and restore the original signal length.
    return pywt.waverec(coeffs_thresh, wavelet_t)[:len(audio)]


# =============================================================================
# 2-D FREQUENCY-TIME WAVELET DENOISING
# =============================================================================

def wavelet_denoise_spectrogram(
    S,
    wavelet="db4",
    level=None,
    threshold_scale_ft=1.0,
    mode="soft",
    threshold_method="global",
):
    """
    Denoise a 2-D spectrogram using wavelet thresholding.

    The spectrogram is decomposed into approximation and detail
    coefficients using a 2-D discrete wavelet transform. Noise levels
    are estimated from the diagonal detail coefficients using the Median
    Absolute Deviation (MAD), and universal thresholding is applied to
    the detail coefficients.

    The threshold can be calculated using either a global or adaptive
    approach.

    Parameters
    ----------
    S : np.ndarray
        Input 2-D spectrogram with shape ``(frequency, time)``.

    wavelet : str, optional
        Name of the wavelet used for the decomposition.
        Default is ``"db4"`` (Daubechies-4).

    level : int or None, optional
        Number of wavelet decomposition levels.

        If ``None``, PyWavelets automatically selects an appropriate
        decomposition level.

    threshold_scale_ft : float, optional
        Scaling factor applied to the universal threshold to control
        the aggressiveness of denoising.

        Values greater than 1 result in stronger thresholding, while
        values below 1 result in weaker thresholding.

        Default is 1.0.

    mode : {"soft", "hard"}, optional
        Thresholding strategy used for the wavelet detail coefficients.

        ``"soft"`` shrinks coefficients toward zero and generally
        produces smoother results.

        ``"hard"`` sets coefficients below the threshold to zero while
        leaving larger coefficients unchanged.

        Default is ``"soft"``.

    threshold_method : {"global", "adaptive"}, optional
        Method used to calculate the wavelet threshold.

        ``"global"`` estimates the noise level from the finest-scale
        diagonal detail coefficients and applies the same threshold to
        all decomposition levels.

        ``"adaptive"`` estimates the noise level independently at each
        decomposition level and applies a scale-dependent threshold.

        Default is ``"global"``.

    Returns
    -------
    np.ndarray
        Denoised spectrogram with the same dimensions as the input
        spectrogram.

    Raises
    ------
    ValueError
        If ``threshold_method`` is not ``"global"`` or ``"adaptive"``.
    """

    # Input
    S = np.nan_to_num(S)

    if threshold_method not in {"global", "adaptive"}:
        raise ValueError(
            "threshold_method must be either 'global' or 'adaptive'."
        )

    # 2-D wavelet decomposition
    coeffs = pywt.wavedec2(S, wavelet=wavelet, level=level)

    cA = coeffs[0]
    detail_coeffs = coeffs[1:]

    # Global thersholding
    if threshold_method == "global":

        # Estimate the noise level from the finest-scale diagonal detail coefficients using MAD.
        _, _, cD_finest = detail_coeffs[-1]

        sigma = (np.median(np.abs(cD_finest)) / 0.6745)

        # Calculate the scaled universal threshold using the size of the complete spectrogram.
        T_global = (threshold_scale_ft * sigma * np.sqrt(2 * np.log(S.size)))

    # Wavelet thresholding
    new_details = []

    for cH, cV, cD in detail_coeffs:

        if threshold_method == "global":

            # Apply the same threshold to all decomposition levels.
            T = T_global

        else:

            # Estimate the noise level independently for the current decomposition level using MAD.
            sigma = (np.median(np.abs(cD)) / 0.6745)

            # Calculate the level-dependent scaled universal threshold.
            T = (threshold_scale_ft * sigma * np.sqrt(2 * np.log(cD.size)))

        # Apply thresholding to the horizontal, vertical, and diagonal detail coefficients.
        cH = pywt.threshold(cH, T, mode=mode)
        cV = pywt.threshold(cV, T, mode=mode)
        cD = pywt.threshold(cD, T, mode=mode)

        new_details.append((cH, cV, cD))

    # Wavelet reconstruction
    coeffs_new = [cA] + new_details
    S_denoised = pywt.waverec2(coeffs_new, wavelet)

    # Crop the reconstructed spectrogram to the original dimensions.
    return S_denoised[:S.shape[0], :S.shape[1]]


# =============================================================================
# FREQUENCY-DEPENDENT PRE-WHITENING
# =============================================================================

def pre_whitening(
    Sxx,
    noise_estimation="median_residual",
    smooth_sigma=2.0,
    noise_percentile=20,
    floor_percentile=10,
    eps=1e-12,
):
    """
    Perform frequency-dependent pre-whitening of a spectrogram.

    The spectrogram is normalized by an estimated frequency-dependent
    noise spectrum to reduce spectral coloration and produce a more
    spectrally uniform representation.

    Two approaches are available for estimating the frequency-dependent
    noise spectrum:

    1. Median-residual estimation
        Uses a median equalizer to estimate the non-noise component.
        The residual is then used to estimate the frequency-dependent
        noise spectrum.

    2. Percentile-based estimation
        Estimates the low-energy noise level independently for each
        frequency bin using a specified percentile.

    Parameters
    ----------
    Sxx : np.ndarray
        Input 2-D spectrogram with shape ``(frequency, time)``.

    noise_estimation : {"median_residual", "percentile"}, optional
        Approach used to estimate the frequency-dependent noise spectrum.

        ``"median_residual"`` uses a median equalizer to estimate the
        non-noise component. The residual between the input spectrogram
        and the estimated non-noise component is then used to estimate
        the frequency-dependent noise spectrum.

        ``"percentile"`` estimates the low-energy noise level
        independently for each frequency bin using ``noise_percentile``.

        Default is ``"median_residual"``.

    smooth_sigma : float, optional
        Standard deviation of the Gaussian smoothing filter applied
        along the frequency axis to the estimated noise spectrum.

        Default is 2.0.

    noise_percentile : float, optional
        Percentile used when ``noise_estimation="percentile"`` to
        estimate the low-energy noise level at each frequency bin.

        Default is 20.

    floor_percentile : float, optional
        Percentile used to define the minimum allowed noise level and
        prevent excessive whitening gains.

        Default is 10.

    eps : float, optional
        Small positive value used to prevent division by zero and other
        numerical problems.

        Default is ``1e-12``.

    Returns
    -------
    S_white : np.ndarray
        Frequency-whitened spectrogram.

    noise_psd : np.ndarray
        Estimated frequency-dependent noise spectrum before smoothing.

    noise_psd_smooth : np.ndarray
        Smoothed and stabilized frequency-dependent noise spectrum used
        for whitening.

    Raises
    ------
    ValueError
        If ``Sxx`` is not a 2-D array or if ``noise_estimation`` is not
        ``"median_residual"`` or ``"percentile"``.
    """

    # Input

    Sxx = np.asarray(Sxx, dtype=np.float64)

    if Sxx.ndim != 2:
        raise ValueError(
            "Sxx must be a 2-D array with shape (frequency, time)."
        )

    if noise_estimation not in {"median_residual", "percentile"}:
        raise ValueError(
            "noise_estimation must be either "
            "'median_residual' or 'percentile'."
        )

    # Frequency-dependent noise estimation

    if noise_estimation == "median_residual":

        # Estimate the non-noise component using the median equalizer.
        Sxx_nonoise = median_equalizer(Sxx)

        # Estimate the noise component from the residual.
        Sxx_noise = Sxx - Sxx_nonoise

        # Estimate the average noise level for each frequency bin.
        noise_psd = np.mean(Sxx_noise, axis=1)

    else:

        # Estimate the low-energy noise level for each frequency bin.
        noise_psd = np.percentile(Sxx, noise_percentile, axis=1)

    # Smooth the noise spectrum along the frequency axis.
    noise_psd_smooth = gaussian_filter1d(noise_psd, sigma=smooth_sigma)

    # Define a lower noise floor to prevent excessive whitening gains.
    floor = np.percentile(noise_psd_smooth, floor_percentile)

    # Apply the noise floor and numerical stability limit.
    noise_psd_smooth = np.maximum(noise_psd_smooth, floor)
    noise_psd_smooth = np.maximum(noise_psd_smooth, eps)

    # Apply frequency-dependent whitenng filter
    S_white = Sxx / noise_psd_smooth[:, None]

    # Restore global level
    original_mean = np.mean(Sxx)
    whitened_mean = np.mean(S_white)

    if whitened_mean > eps:
        S_white *= original_mean / whitened_mean

    return S_white, noise_psd, noise_psd_smooth


# =============================================================================
# ADAPTIVE TIME-FREQUENCY WIENER FILTER
# =============================================================================
def adaptive_tf_wiener(S):
    """
    Apply an adaptive time-frequency Wiener filter.

    Local background power is estimated from the lower-energy regions
    of the spectrogram and converted into a local SNR estimate. A
    Wiener-like gain is then applied to suppress low-SNR components.

    Parameters
    ----------
    S : np.ndarray
        Input spectrogram with shape ``(frequency, time)``.

    Returns
    -------
    np.ndarray
        Wiener-filtered spectrogram.
    """

    # Convert the spectrogram to power.
    P = S ** 2

    # Estimate local background power while limiting the influence of strong acoustic events.
    noise_tf = gaussian_filter(np.minimum(P, np.percentile(P, 50)), (2, 2), 0)

    # Estimate local SNR.
    snr = P / (noise_tf + 1e-12)

    # Convert SNR to a Wiener-like gain.
    G = snr / (snr + 1)

    # Retain a minimum gain to avoid removing weak acoustic components.
    G = np.clip(G, 0.05, 1.0)

    # Apply the adaptive gain.
    return S * G


# =============================================================================
# ITERATIVE ADAPTIVE TIME-FREQUENCY WIENER FILTER
# =============================================================================
def iterative_wiener(
    S,
    n_iter=2,
):
    """
    Apply the adaptive time-frequency Wiener filter iteratively.

    Parameters
    ----------
    S : np.ndarray
        Input spectrogram.

    n_iter : int, optional
        Number of Wiener-filter iterations. Default is 2.

    Returns
    -------
    np.ndarray
        Spectrogram after repeated Wiener filtering.
    """

    # Preserve the input array and iteratively apply the Wiener filter.
    S_out = S.copy()

    for _ in range(n_iter):
        S_out = adaptive_tf_wiener(S_out)

    return S_out


# =============================================================================
# SLIDING-WINDOW SVD BACKGROUND NOISE REMOVAL
# =============================================================================
def sliding_adaptive_svd(
    S,
    win=18,
    hop=8,
    energy_threshold=0.80,
    min_energy_ratio=0.40,
    max_energy_ratio=0.80,
    max_rank=3,
    temporal_persistence_threshold=0.60,
    eps=1e-12,
):
    """
    Remove persistent background structure using sliding-window SVD.

    SVD is applied independently to overlapping time windows. Dominant
    modes with sufficient energy and temporal persistence are treated
    as background-like components and adaptively subtracted.

    Parameters
    ----------
    S : np.ndarray
        Input 2-D spectrogram with shape ``(frequency, time)``.

    win : int, optional
        Number of time frames in each SVD window. Default is 18.

    hop : int, optional
        Number of frames between consecutive windows. Default is 8.

    energy_threshold : float, optional
        Cumulative singular-value energy used to determine the dominant
        modes. Default is 0.80.

    min_energy_ratio : float, optional
        Minimum first-mode energy ratio required for SVD subtraction.
        Default is 0.40.

    max_energy_ratio : float, optional
        First-mode energy ratio at which subtraction reaches maximum
        strength. Default is 0.80.

    max_rank : int, optional
        Maximum number of SVD modes removed. Default is 3.

    temporal_persistence_threshold : float, optional
        Minimum fraction of the window for which a mode must remain
        active to be considered persistent. Default is 0.60.

    eps : float, optional
        Numerical stability constant. Default is ``1e-12``.

    Returns
    -------
    np.ndarray
        Noise-reduced spectrogram with the same shape as the input.
    """

    # Convert to float64 for numerical stability during SVD.
    S = np.asarray(S, dtype=np.float64)

    if S.ndim != 2:
        raise ValueError("S must be a 2-D array.")

    _, T = S.shape

    if T == 0:
        return S.copy()

    # Set the window to the full time axis when win is None.
    if win is None:
        win = T
        hop = T

    win = min(int(win), T)

    if hop is None or hop <= 0:
        hop = win

    # Allocate arrays for overlap-add reconstruction.
    output = np.zeros_like(S)
    weight = np.zeros_like(S)

    # Determine the starting position of each time window.
    starts = list(range(0, max(T - win + 1, 1), hop))

    # Ensure that the final portion of the spectrogram is included.
    if starts[-1] + win < T:
        starts.append(T - win)

    # Process each overlapping time window.
    for t in starts:

        block = S[:, t:t + win]

        # Compute the SVD of the current time window.
        U, s, Vt = svd(block, full_matrices=False, check_finite=False,)

        if len(s) == 0:
            continue

        # Calculate the relative energy of each singular mode.
        s2 = s ** 2
        total_energy = np.sum(s2) + eps
        energy_ratio = s2 / total_energy

        first_ratio = energy_ratio[0]

        # Determine adaptive subtraction strength from the dominant-mode energy ratio.
        alpha = np.clip((first_ratio - min_energy_ratio) / (max_energy_ratio - min_energy_ratio), 0.0, 1.0,)

        if alpha <= 0:
            clean_block = block.copy()

        else:

            # Determine the number of dominant modes to consider.
            cumulative_energy = np.cumsum(energy_ratio)

            rank = (np.searchsorted(cumulative_energy, energy_threshold) + 1)

            rank = min(rank, max_rank,len(s))

            # Reconstruct modes that exhibit sufficient temporal persistence.
            background = np.zeros_like(block)

            for k in range(rank):

                temporal_component = np.abs(Vt[k])
                max_temporal = np.max(temporal_component)

                if max_temporal <= eps:
                    continue

                normalized_temporal = (temporal_component / max_temporal)

                persistence = np.mean(normalized_temporal > 0.30)

                if persistence >= temporal_persistence_threshold:
                    background += (s[k] * np.outer(U[:, k], Vt[k]))

            # Subtract the estimated persistent background.
            clean_block = block - alpha * background

            # Prevent negative spectrogram values.
            clean_block = np.maximum(clean_block, eps)

        # Accumulate overlapping processed windows.
        output[:, t:t + win] += clean_block
        weight[:, t:t + win] += 1.0

    # Average overlapping windows.
    output /= np.maximum(weight, 1.0)

    return output


# =============================================================================
# SPECTROGRAM RESIZING WITH ENERGY CONSERVATION
# =============================================================================
def resize_spectrogram(
    S,
    t_old,
    f_old,
    new_nt=None,
    new_nf=None,
    t_new=None,
    f_new=None,
    conserve_energy=True,
):
    """
    Resize a spectrogram using physical time/frequency coordinates.

    The spectrogram is interpolated onto a new time-frequency grid.
    Optionally, the result is scaled to approximately conserve the
    integrated spectrogram energy.

    Parameters
    ----------
    S : np.ndarray
        Input spectrogram with shape ``(frequency, time)``.

    t_old : np.ndarray
        Original time coordinates.

    f_old : np.ndarray
        Original frequency coordinates.

    new_nt : int or None, optional
        Number of samples in the new time grid.

    new_nf : int or None, optional
        Number of samples in the new frequency grid.

    t_new : np.ndarray or None, optional
        Explicit new time coordinates.

    f_new : np.ndarray or None, optional
        Explicit new frequency coordinates.

    conserve_energy : bool, optional
        If ``True``, approximately conserve integrated spectrogram
        energy after interpolation. Default is ``True``.

    Returns
    -------
    S_new : np.ndarray
        Resized spectrogram.

    t_new : np.ndarray
        New time coordinates.

    f_new : np.ndarray
        New frequency coordinates.
    """

    S = np.asarray(S)

    # Generate new coordinates when they are not explicitly provided.
    if t_new is None:
        t_new = np.linspace(t_old.min(), t_old.max(), new_nt)

    if f_new is None:
        f_new = np.linspace(f_old.min(), f_old.max(), new_nf)

    # Interpolate the spectrogram in physical frequency/time coordinates.
    interp = RegularGridInterpolator((f_old, t_old), S, bounds_error=False, fill_value=0.0)

    TT, FF = np.meshgrid(t_new, f_new)

    S_new = interp(np.stack([FF.ravel(), TT.ravel()], axis=-1)).reshape(len(f_new), len(t_new))

    # Correct the interpolated spectrogram to approximately conserve energy.
    if conserve_energy:
        df_old = np.mean(np.diff(f_old))
        dt_old = np.mean(np.diff(t_old))

        df_new = np.mean(np.diff(f_new))
        dt_new = np.mean(np.diff(t_new))

        energy_old = S.sum() * df_old * dt_old
        energy_new = S_new.sum() * df_new * dt_new

        if energy_new > 0:
            S_new *= energy_old / energy_new

    return (S_new, t_new, f_new)


# =============================================================================
# SPECTROGRAM REFINEMENT LOOP
# =============================================================================
def refinement_loop(
    S,
    k=2,
    perc=90,
    kernel_size=(0.8, 0.4),
    flag_svd=0,
    flag_S=0,
    flag_G=1,
):
    """
    Apply a configurable sequence of spectrogram refinement operations.

    The pipeline can independently apply SVD-based background removal,
    softplus compression, and Gaussian smoothing.

    Parameters
    ----------
    S : np.ndarray
        Input spectrogram.

    k : float, optional
        Scaling parameter for the softplus transformation. Default is 2.

    perc : float, optional
        Percentile used to determine the softplus threshold. Default is 90.

    kernel_size : tuple, optional
        Gaussian smoothing parameters for the frequency and time axes.
        Default is ``(0.8, 0.4)``.

    flag_svd : int, optional
        Enable SVD-based background removal. Default is 0.

    flag_S : int, optional
        Enable softplus compression. Default is 0.

    flag_G : int, optional
        Enable Gaussian smoothing. Default is 1.

    Returns
    -------
    np.ndarray
        Refined spectrogram.
    """

    # Apply SVD-based background removal.
    if flag_svd == 1:
        S_tmp = sliding_adaptive_svd(S)
    else:
        S_tmp = S

    # Apply percentile-based softplus compression.
    if flag_S == 1:
        thr = np.percentile(S_tmp, perc)
        S_tmp2 = (np.logaddexp(0, k * (S_tmp - thr)) / k)
    else:
        S_tmp2 = S_tmp

    # Apply Gaussian smoothing.
    if flag_G == 1:
        S_out = gaussian_filter(np.maximum(S_tmp2, 1e-12), kernel_size)
    else:
        S_out = S_tmp2

    return S_out


# =============================================================================
# PER-CHANNEL ENERGY NORMALIZATION (PCEN)
# =============================================================================
def pcen_custom(
    S,
    eps=1e-6,
    s=0.025,
    alpha=0.98,
    delta=2,
    r=0.5,
):
    """
    Apply custom Per-Channel Energy Normalization (PCEN).

    A temporally smoothed background level is estimated independently
    for each frequency channel and used for adaptive energy normalization
    and nonlinear dynamic-range compression.

    Parameters
    ----------
    S : np.ndarray
        Input spectrogram with shape ``(frequency, time)``.

    eps : float, optional
        Numerical stability constant. Default is ``1e-6``.

    s : float, optional
        Temporal smoothing coefficient for the background estimate.
        Default is 0.025.

    alpha : float, optional
        Strength of automatic gain control. Default is 0.98.

    delta : float, optional
        Bias added before nonlinear compression. Default is 2.

    r : float, optional
        Compression exponent. Default is 0.5.

    Returns
    -------
    np.ndarray
        PCEN-normalized spectrogram.
    """

    # Initialize the temporal background estimate.
    M = np.zeros_like(S)

    # Initialize the background with the first time frame.
    M[:, 0] = S[:, 0]

    # Update the background estimate independently for each frequency bin.
    for t in range(1, S.shape[1]):

        M[:, t] = ((1 - s) * M[:, t - 1] + s * S[:, t])

    # Normalize by the local background and apply nonlinear compression.
    pcen = ((S / (eps + M) ** alpha) + delta) ** r - delta ** r

    return pcen


# =============================================================================
# WAVELET SNR ENHANCEMENT
# =============================================================================

class WaveletSNREnhancer:
    """
    Wavelet-based directional spectrogram SNR enhancement.

    Representative wavelet patterns are learned from averaged frequency
    and time profiles and used for regularized FFT-domain deconvolution
    along both spectrogram axes. The directional results are then blended.
    """

    def __init__(
        self,
        wavelet="morl",
        scales=np.arange(1, 64),
        smooth_sigma=2,
        regularization=1e-3,
        blend_weight=0.8,
    ):
        """
        Initialize the wavelet SNR enhancer.

        Parameters
        ----------
        wavelet : str, optional
            Wavelet used for the continuous wavelet transform.
            Default is ``"morl"``.

        scales : np.ndarray, optional
            Wavelet scales used for the continuous wavelet transform.
            Default is ``np.arange(1, 64)``.

        smooth_sigma : float, optional
            Gaussian smoothing applied to the learned wavelet pattern.
            Default is 2.

        regularization : float, optional
            Regularization parameter for FFT-domain deconvolution.
            Default is ``1e-3``.

        blend_weight : float, optional
            Weight assigned to the frequency-direction enhancement.
            The time-direction weight is ``1 - blend_weight``.
            Default is 0.8.
        """

        self.wavelet = wavelet
        self.scales = scales
        self.smooth_sigma = smooth_sigma
        self.regularization = regularization
        self.blend_weight = blend_weight

    def enhance(self, S):
        """
        Apply wavelet-based directional SNR enhancement.

        Parameters
        ----------
        S : np.ndarray
            Input 2-D spectrogram with shape ``(frequency, time)``.

        Returns
        -------
        np.ndarray
            Enhanced spectrogram with non-negative values.
        """

        # Smooth the spectrogram before learning representative patterns.
        S_noise = gaussian_filter(S, (0.5, 0.5))

        # Learn representative patterns from the time and frequency profiles.
        wt = self._learn_wavelet(np.mean(S_noise, axis=0))

        wf = self._learn_wavelet(np.mean(S_noise, axis=1))

        # Apply directional enhancement along the frequency and time axes.
        Sf = self._directional_enhancement(S, wt, axis=0)

        St = self._directional_enhancement(S, wf, axis=1)

        # Blend the directional enhancement results.
        Sout = (self.blend_weight * Sf + (1 - self.blend_weight) * St)

        # Prevent negative spectrogram values.
        return np.maximum(Sout, 0)

    def _learn_wavelet(self, profile):
        """
        Learn a representative wavelet pattern from a 1-D profile.

        Parameters
        ----------
        profile : np.ndarray
            One-dimensional frequency or time profile.

        Returns
        -------
        np.ndarray
            Normalized and smoothed representative wavelet pattern.
        """

        # Compute the continuous wavelet transform.
        coeffs, _ = pywt.cwt(profile, self.scales, self.wavelet)

        # Calculate wavelet energy at each scale.
        energy = np.sum(np.abs(coeffs) ** 2, axis=1)

        # Select the most energetic wavelet pattern.
        w = coeffs[np.argmax(energy)]

        # Normalize the learned pattern.
        w /= (np.max(np.abs(w)) + 1e-8)

        # Smooth the pattern to reduce small-scale fluctuations.
        return gaussian_filter1d(w, self.smooth_sigma)

    def _fft_deconvolve(self, x, w):
        """
        Perform regularized FFT-domain deconvolution.

        Parameters
        ----------
        x : np.ndarray
            Input one-dimensional signal.

        w : np.ndarray
            Learned wavelet/template.

        Returns
        -------
        np.ndarray
            Deconvolved signal.
        """

        # Resample the learned wavelet to the signal length.
        w = np.interp(np.linspace(0, 1, len(x)), np.linspace(0, 1, len(w)), w)

        # Transform the signal and wavelet into the frequency domain.
        F = np.fft.fft(x)
        W = np.fft.fft(w)

        # Apply regularized inverse filtering.
        return np.real(np.fft.ifft(F * np.conj(W) / (np.abs(W) ** 2 + self.regularization)))

    def _directional_enhancement(
        self,
        S,
        wavelet,
        axis,
    ):
        """
        Apply enhancement independently along one spectrogram axis.

        Parameters
        ----------
        S : np.ndarray
            Input spectrogram.

        wavelet : np.ndarray
            Learned one-dimensional wavelet/template.

        axis : int
            Axis along which enhancement is performed.
            ``0`` processes frequency profiles.
            ``1`` processes time profiles.

        Returns
        -------
        np.ndarray
            Directionally enhanced spectrogram.
        """

        Sout = np.zeros_like(S)

        # Select the spectrogram profiles along the requested axis.
        if axis == 0:

            iterator = range(S.shape[0])
            getter = lambda i: S[i]
            setter = lambda i, v: Sout.__setitem__(i, v)

        else:

            iterator = range(S.shape[1])
            getter = lambda i: S[:, i]
            setter = lambda i, v: Sout.__setitem__((slice(None), i), v)

        # Enhance each one-dimensional profile independently.
        for i in iterator:

            signal = getter(i)

            enhanced = self._fft_deconvolve(signal, wavelet)

            # Estimate the relative strength of the enhancement.
            alpha = np.clip(np.mean(signal) / (np.std(signal) + 1e-8) / 5, 0, 1)

            # Blend the enhanced and original profiles.
            setter(i, alpha * enhanced + (1 - alpha) * signal)

        return Sout