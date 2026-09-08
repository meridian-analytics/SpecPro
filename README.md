# PAM Spectrogram Enhancement Pipeline

A Python-based spectrogram enhancement and denoising pipeline for **Passive Acoustic Monitoring (PAM)** data stored in **Ketos-compatible HDF5 databases**.

The pipeline applies a multi-stage signal-processing workflow to each spectrogram, including:

* Pre-whitening
* SVD-based background removal
* Gaussian smoothing
* Iterative Wiener filtering
* Wavelet-based denoising
* Spectrogram refinement
* Numerical sanitization
* Optional spectrogram resizing

The original HDF5 database is opened in **read-only mode** and is never modified. A new timestamped HDF5 database is created containing the processed spectrograms and the original metadata/non-data fields.

---

## Overview

The processing architecture consists of three Python modules:

```text
┌─────────────────────────────────────┐
│  run_processing_pipeline.py         │
│                                     │
│  Command-line entry point           │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  pam_processing_pipline.py          │
│                                     │
│  EnhanceSNR transformation          │
│  Multi-stage processing workflow    │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  pam_processing_functions.py        │
│                                     │
│  Individual signal-processing       │
│  functions                          │
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

Each spectrogram passes through the following processing stages:

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
│ Reduce red noise     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Refinement           │
│ SVD + Gaussian       │
│ smoothing            │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Iterative Wiener     │
│ Filtering            │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Refinement           │
│ Gaussian smoothing   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Wavelet Denoising    │
│ 2-D wavelet          │
│ thresholding         │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Final Refinement     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Numerical Cleanup    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Optional Resize      │
└──────────┬───────────┘
           │
           ▼
Enhanced Spectrogram
```

---

# 1. Input Sanitization

Before processing, each spectrogram is converted to `float32`.

Invalid numerical values are handled as follows:

```python
Sxx = np.asarray(rep.data, dtype=np.float32)

Sxx = np.nan_to_num(
    Sxx,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

Sxx = np.maximum(Sxx, eps)
```

This ensures that NaN, infinite, and values below the numerical floor do not propagate through the processing pipeline.

The default numerical floor is:

```text
eps = 1e-12
```

---

# 2. Pre-whitening

The first signal-processing stage is **pre-whitening**, which is intended to reduce the influence of red/background noise.

The pipeline uses:

```python
pre_whitening(
    Sxx,
    noise_estimation="median_residual",
    smooth_sigma=2.0,
    noise_percentile=20,
    floor_percentile=10,
    eps=1e-12
)
```

The resulting whitened spectrogram is then passed to a refinement stage.

---

# 3. Spectrogram Refinement

The pipeline uses the `refinement_loop()` function at several stages.

For the first refinement stage:

```python
refinement_loop(
    S_white_i,
    k=self.k,
    perc=self.perc,
    kernel_size=self.kernel_size,
    flag_svd=1,
    flag_S=0,
    flag_G=1
)
```

This stage enables:

* SVD-based background removal
* Gaussian smoothing

while softplus compression is disabled in this particular stage.

The refinement function provides configurable processing components through:

```text
flag_svd
flag_S
flag_G
```

---

# 4. Iterative Wiener Filtering

The whitened spectrogram is processed using an iterative Wiener filter:

```python
S_wiener_i = iterative_wiener(
    S_white,
    n_iter=self.n_iter
)
```

The purpose of this stage is to suppress background noise using an iterative estimate of the local signal/noise structure.

The number of iterations is controlled by:

```text
n_iter
```

Default:

```text
n_iter = 2
```

A second refinement operation is then applied:

```python
S_wiener = refinement_loop(
    S_wiener_i,
    k=self.k,
    perc=self.perc,
    kernel_size=self.kernel_size,
    flag_svd=0,
    flag_S=0,
    flag_G=1
)
```

At this stage, Gaussian smoothing is enabled while SVD-based removal and softplus compression are disabled.

---

# 5. Wavelet Denoising

The next stage applies two-dimensional wavelet denoising:

```python
S_denoised_i = wavelet_denoise_spectrogram(
    S_wiener,
    wavelet="db4",
    level=None,
    threshold_scale_ft=self.thr_ft,
    mode="soft",
    threshold_method="global"
)
```

The default wavelet is:

```text
db4
```

Soft thresholding is used with a global threshold.

The denoising strength is controlled by:

```text
threshold_scale_ft
```

The default value is:

```text
threshold_scale_ft = 0.7
```

A final refinement operation is subsequently applied.

---

# 6. Final Numerical Cleanup

After processing, the spectrogram is sanitized again:

```python
S_denoised = np.nan_to_num(
    S_denoised,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

S_denoised = np.maximum(
    S_denoised,
    eps
)
```

This provides a final numerical safeguard before the spectrogram is written to the output database.

---

# 7. Optional Resizing

The processed spectrogram can optionally be resized using:

```python
skimage.transform.resize
```

For example:

```bash
--resize 240 240
```

produces spectrograms with:

```text
Frequency × Time
240 × 240
```

Resizing uses:

```python
resize(
    S_denoised,
    (frequency, time),
    anti_aliasing=True,
    preserve_range=True
)
```

If `--resize` is not specified, the original spectrogram dimensions are retained.

---

# `EnhanceSNR` Class

The main transformation is implemented in:

```text
pam_processing_pipline.py
```

The class follows a Ketos-compatible callable-transform pattern:

```python
transform = EnhanceSNR(...)

rep = transform(rep)
```

The representation object is expected to contain a 2-D spectrogram in:

```python
rep.data
```

The transformation modifies `rep.data` and returns the same representation object.

This allows metadata associated with the representation to remain attached to the object.

---

# Parameters

The `EnhanceSNR` transformation supports the following parameters.

| Parameter            |      Default | Description                                         |
| -------------------- | -----------: | --------------------------------------------------- |
| `k`                  |        `1.0` | Scaling parameter used by the refinement processing |
| `perc`               |         `50` | Percentile used by the refinement processing        |
| `kernel_size`        | `(0.8, 0.4)` | Frequency and time smoothing parameters             |
| `n_iter`             |          `2` | Number of iterative Wiener-filter iterations        |
| `threshold_scale_ft` |        `0.7` | Wavelet denoising threshold scaling                 |
| `eps`                |      `1e-12` | Numerical floor                                     |
| `resize`             |  `None None` | Optional output dimensions                          |

> The defaults above reflect the command-line processing script. The values should be kept consistent with the actual implementation when the code is updated.

---

# Command-Line Usage

The main script is:

```text
run_processing_pipeline.py
```

Basic usage:

```bash
python run_processing_pipeline.py input.h5 output.h5
```

The input database is read-only and the output is created as a new timestamped HDF5 file.

---

## Example

```bash
python run_processing_pipeline.py input.h5 output.h5 \
    --k 1.0 \
    --perc 50 \
    --threshold_scale_ft 0.7 \
    --kernel_size 0.8 0.4 \
    --n_iter 2 \
    --resize 240 240
```

The script will process every spectrogram in `input.h5`.

---

# Command-Line Arguments

## Input and Output

### `input_database`

Path to the input Ketos HDF5 database.

```bash
python run_processing_pipeline.py input.h5 output.h5
```

### `output_database`

Requested path/name for the output database.

The actual output filename receives a timestamp.

For example:

```text
output.h5
```

may result in:

```text
output_20260908_113025.h5
```

This prevents accidental overwriting of previous processing results.

---

## Enhancement Parameters

### `--k`

Scaling parameter for the refinement processing.

Default:

```text
2
```

Example:

```bash
--k 1.0
```

### `--perc`

Percentile parameter used by the refinement processing.

Default:

```text
90
```

Example:

```bash
--perc 50
```

### `--kernel_size`

Gaussian smoothing parameters:

```text
FREQUENCY TIME
```

Default:

```text
0.8 0.4
```

Example:

```bash
--kernel_size 0.8 0.4
```

---

## Refinement Flags

The command-line interface also exposes three processing flags.

### `--flag_svd`

Controls SVD-based background removal.

```text
0 = disabled
1 = enabled
```

Default:

```text
1
```

### `--flag_S`

Controls softplus compression.

```text
0 = disabled
1 = enabled
```

Default:

```text
1
```

### `--flag_G`

Controls Gaussian smoothing.

```text
0 = disabled
1 = enabled
```

Default:

```text
1
```

> **Implementation note:** The current `EnhanceSNR.__init__()` shown in this repository does not accept `flag_svd`, `flag_S`, or `flag_G` as constructor arguments. These options are currently defined by the command-line parser but are not passed into `EnhanceSNR`. Their behavior should therefore be reviewed if they are intended to be user-configurable from the command line.

---

# Wiener Filter

### `--n_iter`

Number of iterative Wiener-filter iterations.

Default:

```text
2
```

Example:

```bash
--n_iter 3
```

Higher values may increase processing time and alter the amount of iterative filtering.

---

# Wavelet Denoising

### `--wavelet`

Wavelet used for two-dimensional denoising.

Default:

```text
db4
```

Example:

```bash
--wavelet db4
```

### `--wavelet_level`

Number of wavelet decomposition levels.

If omitted, the wavelet decomposition level is selected automatically.

Example:

```bash
--wavelet_level 3
```

### `--threshold_scale_ft`

Scaling factor applied to the wavelet threshold.

Example:

```bash
--threshold_scale_ft 0.7
```

### `--wavelet_mode`

Wavelet thresholding mode:

```text
soft
hard
```

Default:

```text
soft
```

Example:

```bash
--wavelet_mode soft
```

> **Implementation note:** The current `EnhanceSNR.__call__()` uses `db4`, automatic level selection, `soft` thresholding, and `global` thresholding directly in the function call. The command-line `--wavelet`, `--wavelet_level`, and `--wavelet_mode` arguments are therefore not currently propagated to the class. If these options are intended to be configurable, the class interface should be updated accordingly.

---

# Spectrogram Resizing

### `--resize`

Specify the output dimensions as:

```text
FREQUENCY TIME
```

Example:

```bash
--resize 240 240
```

If omitted:

```text
--resize
```

the processed spectrogram retains its original dimensions.

---

# HDF5 Database Processing

The pipeline automatically searches the input HDF5 file for a table containing the:

```text
audio_repres
```

attribute.

This avoids relying on a hard-coded HDF5 table path.

Conceptually:

```text
HDF5 Database
     │
     ├── Group
     │    └── Table
     │
     ├── Group
     │    └── Table
     │
     └── Spectrogram Table
              │
              └── audio_repres
```

The first table containing the `audio_repres` attribute is selected for processing.

If no suitable table is found, the pipeline raises an error:

```text
Could not find a table with 'audio_repres' attribute.
```

---

# Metadata Preservation

The output database recreates the structure of the selected input table.

For each column:

```text
data
other columns
metadata fields
```

the `data` column is replaced by the processed spectrogram, while the remaining columns are copied from the original row.

For example:

```text
Input row
┌──────────┬──────────┬─────────────┐
│ data     │ label    │ timestamp   │
└────┬─────┴──────────┴─────────────┘
     │
     ▼
EnhanceSNR
     │
     ▼
Output row
┌───────────────┬──────────┬─────────────┐
│ enhanced data │ label    │ timestamp   │
└───────────────┴──────────┴─────────────┘
```

Table-level attributes are also copied to the output table.

---

# Memory and Processing Strategy

Spectrograms are processed **one row at a time**.

The processing loop is conceptually:

```python
for i in range(table_in.nrows):

    row_in = table_in[i]

    spectrogram = row_in["data"]

    enhanced = EnhanceSNR(spectrogram)

    row_out["data"] = enhanced

    row_out.append()
```

This avoids loading the entire spectrogram database into memory simultaneously.

The output table is periodically flushed to disk during processing.

---

# Output File Naming

The requested output filename is automatically timestamped.

For example:

```bash
python run_processing_pipeline.py \
    input.h5 \
    enhanced.h5
```

may produce:

```text
enhanced_20260908_113025.h5
```

The timestamp format is:

```text
YYYYMMDD_HHMMSS
```

---

# Repository Structure

Recommended repository structure:

```text
pam-spectrogram-processing/
│
├── run_processing_pipeline.py
├── pam_processing_pipline.py
├── pam_processing_functions.py
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── tests/
│   ├── test_processing_functions.py
│   └── test_enhance_snr.py
│
├── examples/
│   └── ...
│
└── data/
    └── README.md
```

Large HDF5 databases should generally not be committed directly to the Git repository.




---

# Installation

It is recommended to create a virtual environment before installing the required packages listed in the requirements.txt file.

1. [Download](https://www.python.org/downloads/) and install `Python`
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

# Complete Example

```bash
python run_processing_pipeline.py \
    data/input_database.h5 \
    data/enhanced_database.h5 \
    --k 1.0 \
    --perc 50 \
    --kernel_size 0.8 0.4 \
    --n_iter 2 \
    --threshold_scale_ft 0.7 \
    --resize 240 240
```

Processing progress is displayed in the terminal:

```text
EnhanceSNR configuration
------------------------
Input       : data/input_database.h5
Output      : data/enhanced_database.h5
k           : 1.0
perc        : 50.0
thr_ft      : 0.7
kernel_size : (0.8, 0.4)
n_iter      : 2
resize      : (240, 240)

Found input table: /data/table
Number of rows: 101820

Processing...

Processing 101820/101820

DONE
Enhanced database: ...
```

---

# Design Principles

The pipeline follows several design principles:

### 1. Preserve the original data

The input HDF5 file is opened in read-only mode.

### 2. Separate processing from database management

The signal-processing functions are contained in:

```text
pam_processing_functions.py
```

while the `EnhanceSNR` processing sequence is defined in:

```text
pam_processing_pipline.py
```

and database/CLI management is handled by:

```text
run_processing_pipeline.py
```

### 3. Process data incrementally

Spectrograms are processed independently, reducing the memory requirement for large databases.

### 4. Preserve metadata

Non-data columns and table-level attributes are copied to the output database.

### 5. Provide reproducible processing

Processing parameters are explicitly exposed through the command-line interface.

---

# Important Implementation Notes

The current code contains several command-line parameters that are defined in `run_processing_pipeline.py` but are not currently passed into `EnhanceSNR`.

Specifically:

```text
--flag_svd
--flag_S
--flag_G
--wavelet
--wavelet_level
--wavelet_mode
```

The current `EnhanceSNR` implementation internally uses:

```text
wavelet       = "db4"
wavelet_level = None
mode          = "soft"
threshold     = "global"
```

and the refinement flags are hard-coded within the processing stages.

If these parameters are intended to be configurable from the command line, the `EnhanceSNR` constructor and `__call__()` implementation should be updated so that the command-line values are propagated through the pipeline.

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


