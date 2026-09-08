# PAM Spectrogram Enhancement Pipeline

A Python-based spectrogram enhancement and denoising pipeline for **Passive Acoustic Monitoring (PAM)** data stored in a **Ketos HDF5 databases**.

The pipeline applies a multi-stage signal-processing workflow to each spectrogram, including:

* Pre-whitening - to reduces red noise  
* Iterative Wiener filtering - to enhance signal to noise ratio using an estimate of the local signal/noise structure
* Wavelet-based denoising - to remove white noise while preserving localized acoustic structures
* Optional spectrogram resizing

Each step (except resizing) is followed by a refinement loop to remove artefacts, including:
* SVD-based background removal
* Gaussian smoothing

The original HDF5 database is opened in **read-only mode** and is never modified. A new timestamped HDF5 database is created containing the processed spectrograms and the original metadata/non-data fields.

---

## Overview

The processing architecture consists of three Python modules:

```text
┌─────────────────────────────────────┐
│  **run_processing_pipeline.py**     │         
└─────────────────────────────────────┘
                   ▲
                   │
┌─────────────────────────────────────┐
│  **pam_processing_pipline.py**      │ 
└─────────────────────────────────────┘
                   ▲
                   │
┌─────────────────────────────────────┐
│  **pam_processing_functions.py**    │
└─────────────────────────────────────┘
```

### Module responsibilities

| Module                        | Purpose                                                         |
| ----------------------------- | --------------------------------------------------------------- |
| `run_processing_pipeline.py`  | Command-line interface and HDF5 database processing             |
| `pam_processing_pipline.py`   | Defines the `EnhanceSNR` transformation and processing sequence |
| `pam_processing_functions.py` | Implements the individual signal-processing functions           |

---

# Processing Pipeline

The pipeline automatically searches the input HDF5 database for a table containing the **Ketos-specific** `audio_repres` attribute.

```text
HDF5 Database
     │
     ├── Group
     │    └── Table
     │
     ├── Group
     │    └── Table
     │
     └── Ketos Spectrogram Table
              │
              └── audio_repres
                   ↑
              Ketos-specific attribute
```

The **first table** containing the `audio_repres` attribute is selected for processing. If no suitable table is found, the pipeline raises:

```text
Could not find a table with 'audio_repres' attribute.
```

### Input Sanitization

Each spectrogram is converted to `float32` and sanitized before processing:

```python
Sxx = np.asarray(rep.data, dtype=np.float32)

Sxx = np.nan_to_num(
    Sxx,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

Sxx = np.maximum(Sxx, 1e-12)
```

This prevents **NaN, infinite, and excessively small values** from propagating through the processing stages.

### Enhancement Stages

Each spectrogram then passes through the following stages:

```text
Input Spectrogram
       │
       ▼
┌──────────────────────┐
│ Input Sanitization   │
│ NaN / Inf handling   │
│ Minimum-value clamp  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Pre-whitening        │
│ Red-noise reduction  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Refinement            │
│ SVD + Gaussian        │
│ smoothing             │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Iterative Wiener      │
│ filtering             │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Refinement            │
│ Gaussian smoothing    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Wavelet Denoising     │
│ 2-D thresholding      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Refinement            │
│ Gaussian smoothing    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Numerical Cleanup     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Optional Resize       │
└──────────┬───────────┘
           │
           ▼
Enhanced Spectrogram
```

# Command-Line Usage

The main script is:

```text
run_processing_pipeline.py
```

Basic usage:

```bash
python run_processing_pipeline.py input.h5 output.h5
```

---

## Example

```bash
python run_processing_pipeline.py input.h5 output.h5 --k 1.0 --perc 50 --threshold_scale_ft 0.7 --kernel_size 0.8 0.4 --n_iter 2 --resize 240 240
```

The script will process every spectrogram in `input.h5`.

---

## Command-Line Arguments

| Option                    | Description                               |   Default |
| ------------------------- | ----------------------------------------- | --------: |
| `input_database`          | Input Ketos HDF5 database                 |         — |
| `output_database`         | Output enhanced HDF5 database             |         — |
| `--k`                     | Softplus compression scaling              |       `2` |
| `--perc`                  | Percentile for compression threshold      |      `90` |
| `--kernel_size FREQ TIME` | Gaussian smoothing size (frequency, time) | `0.8 0.4` |
| `--flag_svd`              | Enable SVD background removal (`1`/`0`)   |       `1` |
| `--flag_S`                | Enable softplus compression (`1`/`0`)     |       `1` |
| `--flag_G`                | Enable Gaussian smoothing (`1`/`0`)       |       `1` |
| `--n_iter`                | Number of Wiener-filter iterations        |       `2` |
| `--wavelet`               | Wavelet for 2-D denoising                 |     `db4` |
| `--wavelet_level`         | Wavelet decomposition levels              | Automatic |
| `--threshold_scale_ft`    | Wavelet threshold scaling                 |     `1.0` |
| `--wavelet_mode`          | Thresholding mode: `soft` or `hard`       |    `soft` |
| `--resize FREQ TIME`      | Output spectrogram dimensions             |      None |



> **Implementation note:** The current `EnhanceSNR.__call__()` uses `db4`, automatic level selection, `soft` thresholding, and `global` thresholding directly in the function call. The command-line `--wavelet`, `--wavelet_level`, and `--wavelet_mode` arguments are therefore not currently propagated to the class. If these options are intended to be configurable, the class interface should be updated accordingly.

> **Implementation note:** The current `EnhanceSNR.__init__()` shown in this repository does not accept `flag_svd`, `flag_S`, or `flag_G` as constructor arguments. These options are currently defined by the command-line parser but are not passed into `EnhanceSNR`. Their behavior should therefore be reviewed if they are intended to be user-configurable from the command line.

---

# Installation

It is recommended to create a virtual environment before installing the required packages listed in the requirements.txt file.

1. [Download](https://www.python.org/downloads/) and install `Python` if not already installed.
2.	Install virtualenv (if needed; on UNIX-based systems): `sudo apt install python3-venv` 
3.	Create a virtual environment: `python3 -m venv myenv` 
4.	Activate it: `source myenv/bin/activate` on UNIX-based systems OR `myenv\Scripts\activate` on Windows
5.	Install packages: `pip install -r requirements.txt`
6.	Deactivate: `deactivate`

#### Notes

- If a virtual environment is not required, install dependencies directly using:

```bash
pip install -r requirements.txt
```

- Python standard libraries are included with Python and do not require additional installation.


---


# Future Development

Potential improvements include:

* Parallel processing of independent spectrograms
* Configurable wavelet parameters
* Configurable refinement flags
* YAML/JSON configuration files
* Detailed processing logs
* Progress bars
* Unit tests
* Performance benchmarking
* Output validation
* Automatic database integrity checks
* Optional multiprocessing for large databases

---

# License

Add the appropriate project license here.

For example:

```text
This project is licensed under the MIT License.
See the LICENSE file for details.
```

---

# Author

**Farid Jedari-Eyvazi, PhD**

Senior Data Scientist / Machine Learning Engineer

---

# Citation

If this software is used in a publication, please cite the associated repository and/or publication:

```text
[Add citation information here]
```


