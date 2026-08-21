"""
Spectrogram enhancement pipeline for marine mammal acoustic data.

This module provides a Ketos-compatible transformation that applies a
multi-stage signal-processing pipeline to spectrograms.

The processing pipeline consists of:

    1. Input validation and numerical sanitization
    2. Pre-whitening
    3. Refinement
    4. Iterative Wiener filtering
    5. Refinement
    6. Wavelet-based denoising
    7. Final refinement
    8. Numerical cleanup
    9. Optional resizing

The class is designed to operate on a Ketos-like representation object
that contains a ``data`` attribute holding a 2-D spectrogram.
"""

# ---- Third-party libraries
import numpy as np
from skimage.transform import resize

# ---- Custom modules
from pam_processing_functions import (
    pre_whitening,
    iterative_wiener,
    wavelet_denoise_spectrogram,
    refinement_loop,
    resize_spectrogram,
)

# =============================================================================
# =============================================================================
# ======================== Spectrogram SNR Enhancement ========================
# =============================================================================
# =============================================================================

class EnhanceSNR:
    """
    Enhance the signal-to-noise ratio of a spectrogram.

    This class implements a multi-stage spectrogram enhancement and
    denoising pipeline for marine mammal acoustic data.

    The class follows the callable-transform pattern used by Ketos, where
    an object containing a spectrogram in ``rep.data`` is passed to the
    transform, processed, and returned.

    Processing pipeline
    -------------------
    The spectrogram can be processed through several enhancement and
    denoising stages:

    1. SVD-based background removal
        Removes dominant background components using singular value
        decomposition. Enabled by ``flag_svd``.

    2. Softplus compression
        Compresses high-amplitude values to reduce the dynamic range of
        the spectrogram. The compression threshold is determined from
        ``perc`` and its scaling is controlled by ``k``. Enabled by
        ``flag_S``.

    3. Gaussian smoothing
        Applies Gaussian smoothing along the frequency and time axes
        using ``kernel_size``. Enabled by ``flag_G``.

    4. Adaptive Wiener filtering
        Suppresses noise using an iterative estimate of the signal and
        noise components. The number of iterations is controlled by
        ``n_iter``.

    5. Wavelet denoising
        Applies two-dimensional wavelet thresholding to suppress
        localized noise while preserving transient acoustic features.

    Refinement operations may be applied after the major processing
    stages to further improve the enhanced spectrogram.

    Parameters
    ----------
    k : float, optional
        Scaling parameter for the softplus compression.
        Default is 2.

    perc : float, optional
        Percentile used to determine the softplus compression threshold.
        Default is 90.

    kernel_size : tuple of float, optional
        Gaussian smoothing parameters along the frequency and time
        dimensions, respectively.
        Default is ``(0.8, 0.4)``.

    flag_svd : int, optional
        Controls SVD-based background removal.
        ``1`` enables SVD-based background removal and ``0`` disables it.
        Default is 1.

    flag_S : int, optional
        Controls softplus compression.
        ``1`` enables softplus compression and ``0`` disables it.
        Default is 1.

    flag_G : int, optional
        Controls Gaussian smoothing.
        ``1`` enables Gaussian smoothing and ``0`` disables it.
        Default is 1.

    n_iter : int, optional
        Number of adaptive Wiener-filter iterations.
        Default is 2.

    wavelet : str, optional
        Wavelet used for two-dimensional wavelet denoising.
        Default is ``"db4"``.

    wavelet_level : int or None, optional
        Number of wavelet decomposition levels. If ``None``, the
        decomposition level is selected automatically by PyWavelets.
        Default is ``None``.

    threshold_scale_ft : float, optional
        Scaling factor applied to the wavelet universal threshold to
        control the strength of wavelet denoising.
        Default is 1.0.

    wavelet_mode : {"soft", "hard"}, optional
        Thresholding mode used for wavelet denoising.
        Default is ``"soft"``.

    eps : float, optional
        Small positive value used to prevent numerical problems associated
        with zero or invalid values during processing.
        Default is ``1e-12``.

    resized_axis : tuple, optional
        Desired output spectrogram dimensions specified as
        ``(frequency, time)``.

        If either dimension is ``None``, resizing is skipped.
        Default is ``(None, None)``.

    Notes
    -----
    The input spectrogram is converted to ``float32`` and sanitized before
    processing. NaN and infinite values are replaced with zero, and values
    smaller than ``eps`` are clipped to ``eps``.

    The processed spectrogram is sanitized again before being returned.
    """

    def __init__(self, k=1.0, perc=50, thr_ft=0.7, kernel_size=(0.8, 0.4), n_iter=2, eps=1e-12, resized_axis=(None, None)):

        self.k = k
        self.perc = perc
        self.n_iter = n_iter
        self.eps = eps
        self.thr_ft = thr_ft
        self.kernel_size = kernel_size
        self.f = resized_axis[0] # frequency dimension
        self.t = resized_axis[1] # time dimension

    def __call__(self, rep):
        """
        Apply the complete enhancement pipeline to a spectrogram.

        Parameters
        ----------
        rep : object
            Ketos-like representation object containing a 2-D
            spectrogram in ``rep.data``.

        Returns
        -------
        object
            The same representation object with ``rep.data`` replaced
            by the enhanced spectrogram.

        Notes
        -----
        The processing is performed directly on the spectrogram data.

        The original ``rep`` object is preserved so that metadata such
        as frequency and time resolution can remain attached to the
        representation.
        """

        # 1. LOAD AND SANITIZE INPUT SPECTROGRAM
        Sxx = np.asarray(rep.data, dtype=np.float32)
        Sxx = np.nan_to_num(Sxx, nan=0.0, posinf=0.0, neginf=0.0) # Replace: NaN  -> 0, inf -> 0
        Sxx = np.maximum(Sxx, self.eps)  # Clip every value to at least eps.

        # 2. PRE-WHITENING - to reduces red noise  
        S_white_i, _, _ = pre_whitening(Sxx, noise_estimation="median_residual", smooth_sigma=2.0, noise_percentile=20, floor_percentile=10, eps=1e-12)
        S_white = refinement_loop(S_white_i, k=self.k, perc=self.perc, kernel_size=self.kernel_size, flag_svd=1, flag_S=0, flag_G=1)
 
        # 3. ITERATIVE WIENER FILTERING - to separate signal from background noise using an estimate of the local signal/noise structure.
        S_wiener_i = iterative_wiener(S_white, n_iter=self.n_iter)
        S_wiener = refinement_loop(S_wiener_i, k=self.k, perc=self.perc, kernel_size=self.kernel_size, flag_svd=0, flag_S=0, flag_G=1)

        # 4. WAVELET DENOISING - to remove additional noise while preserving localized acoustic structures.
        S_denoised_i = wavelet_denoise_spectrogram(S_wiener, wavelet="db4", level=None, threshold_scale_ft=self.thr_ft, mode="soft", threshold_method="global")
        S_denoised = refinement_loop(S_denoised_i, k=self.k, perc=self.perc, kernel_size=self.kernel_size, flag_svd=0, flag_S=0, flag_G=1)

        # 5. FINAL NUMERICAL CLEANUP
        S_denoised = np.nan_to_num(S_denoised, nan=0.0, posinf=0.0, neginf=0.0)
        S_denoised = np.maximum(S_denoised, self.eps)

        # 6. RESIZE OUTPUT SPECTROGRAM
        if self.f is not None and self.t is not None:
            rep.data = resize(S_denoised, (self.f, self.t), anti_aliasing=True, preserve_range=True).astype(np.float32)
        else:
            rep.data = S_denoised

        return rep
