"""
Apply EnhanceSNR post-processing to a Ketos HDF5 spectrogram database.

This script:

1. Reads an existing Ketos-style HDF5 database.
2. Locates the spectrogram table using its ``audio_repres`` attribute.
3. Creates a new HDF5 database with the same table structure.
4. Applies the ``EnhanceSNR`` transformation to every spectrogram.
5. Resizes the processed spectrograms to the requested dimensions.
6. Copies all metadata and non-data columns to the output database.
7. Uses multiprocessing to process spectrograms in parallel.
8. Displays a live progress bar with processing rate and ETA.
9. Records detailed processing logs.
10. Tracks and reports individual spectrogram processing errors.
11. Generates a CSV error report when failures occur.
12. Produces a final performance summary.

The input database is opened read-only and is never modified.

Example
-------
python enhance_database.py input.h5 output.h5 \
    --k 1.0 \
    --perc 50 \
    --threshold_scale_ft 0.7 \
    --kernel_size 0.8 0.4 \
    --n_iter 2 \
    --resize 240 240 \
    --workers 8

Output
------
The script creates timestamped output files:

    output_YYYYMMDD_HHMMSS.h5
    output_YYYYMMDD_HHMMSS.log
    output_YYYYMMDD_HHMMSS_errors.csv

The log contains configuration, progress information, processing
statistics, warnings, errors, and the final performance summary.
"""

# =============================================================================
# Libraries
# =============================================================================

# ---- Standard libraries
import argparse
import copy
import csv
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# ---- Third-party libraries
import numpy as np
import tables
from tqdm import tqdm

# ---- Custom modules
from pam_processing_pipline import EnhanceSNR


# =============================================================================
# Global worker variables
# =============================================================================

_worker_h5 = None
_worker_table = None
_worker_transform = None


# =============================================================================
# Representation object
# =============================================================================

class Rep:
    """Minimal representation required by EnhanceSNR."""

    pass


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Apply spectrogram SNR enhancement and denoising "
            "to a Ketos HDF5 spectrogram database."
        )
    )

    # -------------------------------------------------------------------------
    # Input / output
    # -------------------------------------------------------------------------

    parser.add_argument(
        "input_database",
        type=str,
        help="Path to the input HDF5 database.",
    )

    parser.add_argument(
        "output_database",
        type=str,
        help=(
            "Base path for the processed HDF5 database. "
            "A timestamp is automatically added."
        ),
    )

    # -------------------------------------------------------------------------
    # Spectrogram refinement
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--k",
        type=float,
        default=2,
        help="Scaling parameter for softplus compression. Default: 2.",
    )

    parser.add_argument(
        "--perc",
        type=float,
        default=90,
        help=(
            "Percentile used to determine the softplus compression "
            "threshold. Default: 90."
        ),
    )

    parser.add_argument(
        "--kernel_size",
        type=float,
        nargs=2,
        default=(0.8, 0.4),
        metavar=("FREQ", "TIME"),
        help=(
            "Gaussian smoothing parameters along frequency and time. "
            "Default: 0.8 0.4."
        ),
    )

    parser.add_argument(
        "--flag_svd",
        type=int,
        choices=[0, 1],
        default=0,
        help="Enable SVD-based background removal. Default: 0.",
    )

    parser.add_argument(
        "--flag_S",
        type=int,
        choices=[0, 1],
        default=0,
        help="Enable softplus compression. Default: 0.",
    )

    parser.add_argument(
        "--flag_G",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable Gaussian smoothing. Default: 1.",
    )

    # -------------------------------------------------------------------------
    # Wiener filter
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help=(
            "Number of adaptive Wiener-filter iterations. "
            "Default: 2."
        ),
    )

    # -------------------------------------------------------------------------
    # Wavelet denoising
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--wavelet",
        type=str,
        default="db4",
        help="Wavelet used for 2-D wavelet denoising. Default: db4.",
    )

    parser.add_argument(
        "--wavelet_level",
        type=int,
        default=None,
        help=(
            "Number of wavelet decomposition levels. "
            "If omitted, PyWavelets selects the level automatically."
        ),
    )

    parser.add_argument(
        "--wavelet_mode",
        type=str,
        choices=["soft", "hard"],
        default="soft",
        help="Wavelet thresholding mode. Default: soft.",
    )

    parser.add_argument(
        "--threshold_scale_ft",
        type=float,
        default=1.0,
        help=(
            "Scaling factor applied to the wavelet universal threshold. "
            "Default: 1.0."
        ),
    )

    # -------------------------------------------------------------------------
    # Spectrogram resizing
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=(None, None),
        metavar=("FREQ", "TIME"),
        help=(
            "Output spectrogram size as frequency x time. "
            "Default: None."
        ),
    )

    # -------------------------------------------------------------------------
    # Multiprocessing
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help=(
            "Number of worker processes used for spectrogram processing. "
            "Default: one fewer than the number of CPU cores."
        ),
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=1,
        help=(
            "Number of spectrogram indices sent to a worker at once. "
            "Default: 1. Larger values may improve throughput but reduce "
            "progress granularity."
        ),
    )

    # -------------------------------------------------------------------------
    # Logging / monitoring
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--flush_interval",
        type=int,
        default=100,
        help=(
            "Flush the output HDF5 table every N successfully processed "
            "spectrograms. Default: 100."
        ),
    )

    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level. Default: INFO.",
    )

    return parser.parse_args()


# =============================================================================
# Logging
# =============================================================================

def setup_logging(log_file, log_level="INFO"):
    """
    Configure logging to both console and a log file.

    Parameters
    ----------
    log_file : Path
        Path to the log file.
    log_level : str
        Logging level.
    """

    level = getattr(logging, log_level)

    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers so logging is not duplicated.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(processName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(
        log_file,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# =============================================================================
# Worker initialization
# =============================================================================

def _init_worker(input_database, table_path, transform_kwargs):
    """
    Initialize one multiprocessing worker.

    Each worker opens the input HDF5 database independently in read-only
    mode and creates its own EnhanceSNR transformation object.
    """

    global _worker_h5
    global _worker_table
    global _worker_transform

    _worker_h5 = tables.open_file(
        input_database,
        mode="r",
    )

    _worker_table = _worker_h5.get_node(table_path)

    _worker_transform = EnhanceSNR(
        **transform_kwargs
    )


# =============================================================================
# Worker processing
# =============================================================================

def _process_spectrogram(i):
    """
    Process one spectrogram.

    Errors are caught inside the worker so that one failed spectrogram
    does not terminate the entire multiprocessing job.

    Parameters
    ----------
    i : int
        Row index.

    Returns
    -------
    dict
        Processing result containing:

        index
        success
        data
        elapsed
        error_type
        error_message
        traceback
    """

    start_time = time.perf_counter()

    try:
        row_in = _worker_table[i]

        S = np.asarray(
            row_in["data"],
            dtype=np.float32,
        )

        rep = Rep()
        rep.data = S

        meta = _worker_table.attrs.audio_repres

        rep.freq_res = meta["freq_res"]
        rep.time_res = meta["time_res"]
        rep.freq_min = meta["freq_min"]

        rep = _worker_transform(rep)

        elapsed = time.perf_counter() - start_time

        return {
            "index": i,
            "success": True,
            "data": np.asarray(rep.data, dtype=np.float32),
            "elapsed": elapsed,
            "error_type": "",
            "error_message": "",
            "traceback": "",
        }

    except Exception as exc:

        elapsed = time.perf_counter() - start_time

        return {
            "index": i,
            "success": False,
            "data": None,
            "elapsed": elapsed,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }


# =============================================================================
# Worker cleanup
# =============================================================================

def _close_worker():
    """Close the worker's input HDF5 database."""

    global _worker_h5

    if _worker_h5 is not None:
        try:
            _worker_h5.close()
        except Exception:
            pass

        _worker_h5 = None


# =============================================================================
# Utility functions
# =============================================================================

def format_duration(seconds):
    """
    Convert seconds to a human-readable duration.
    """

    if seconds < 60:
        return f"{seconds:.2f} s"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60

    if minutes < 60:
        return f"{minutes} min {remaining_seconds:.1f} s"

    hours = int(minutes // 60)
    remaining_minutes = minutes % 60

    return (
        f"{hours} h "
        f"{remaining_minutes} min "
        f"{remaining_seconds:.1f} s"
    )


def write_error_report(error_file, errors):
    """
    Write processing errors to a CSV file.

    Parameters
    ----------
    error_file : Path
        Output CSV path.
    errors : list of dict
        Processing error records.
    """

    if not errors:
        return

    fieldnames = [
        "index",
        "elapsed",
        "error_type",
        "error_message",
        "traceback",
    ]

    with open(
        error_file,
        mode="w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for error in errors:
            writer.writerow(
                {
                    key: error.get(key, "")
                    for key in fieldnames
                }
            )


# =============================================================================
# Main
# =============================================================================

def main():
    """
    Run EnhanceSNR processing on the input HDF5 database.

    The input database is opened read-only and is never modified.

    The main process is responsible for writing the output HDF5 file,
    while worker processes independently read and process spectrograms.
    """

    # -------------------------------------------------------------------------
    # Parse arguments
    # -------------------------------------------------------------------------

    args = parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")

    if args.chunksize < 1:
        raise ValueError("--chunksize must be at least 1.")

    if args.flush_interval < 1:
        raise ValueError("--flush_interval must be at least 1.")

    # -------------------------------------------------------------------------
    # Create timestamped output names
    # -------------------------------------------------------------------------

    output_path = Path(args.output_database)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_database = (
        output_path.parent
        / f"{output_path.stem}_{timestamp}{output_path.suffix}"
    )

    log_file = (
        output_database.parent
        / f"{output_database.stem}.log"
    )

    error_file = (
        output_database.parent
        / f"{output_database.stem}_errors.csv"
    )

    # -------------------------------------------------------------------------
    # Configure logging
    # -------------------------------------------------------------------------

    setup_logging(
        log_file,
        args.log_level,
    )

    logger = logging.getLogger(__name__)

    run_start = time.perf_counter()

    logger.info("=" * 78)
    logger.info("EnhanceSNR HDF5 processing started")
    logger.info("=" * 78)

    # -------------------------------------------------------------------------
    # Log configuration
    # -------------------------------------------------------------------------

    logger.info("Input database       : %s", args.input_database)
    logger.info("Output database      : %s", output_database)
    logger.info("Log file             : %s", log_file)
    logger.info("Error report         : %s", error_file)

    logger.info("k                    : %s", args.k)
    logger.info("perc                 : %s", args.perc)
    logger.info(
        "threshold_scale_ft   : %s",
        args.threshold_scale_ft,
    )
    logger.info(
        "kernel_size          : %s",
        tuple(args.kernel_size),
    )
    logger.info("n_iter               : %s", args.n_iter)
    logger.info("resize               : %s", tuple(args.resize))
    logger.info("wavelet              : %s", args.wavelet)
    logger.info(
        "wavelet_level        : %s",
        args.wavelet_level,
    )
    logger.info(
        "wavelet_mode         : %s",
        args.wavelet_mode,
    )
    logger.info("flag_svd             : %s", args.flag_svd)
    logger.info("flag_S               : %s", args.flag_S)
    logger.info("flag_G               : %s", args.flag_G)
    logger.info("workers              : %s", args.workers)
    logger.info("chunksize            : %s", args.chunksize)
    logger.info(
        "flush_interval       : %s",
        args.flush_interval,
    )

    # -------------------------------------------------------------------------
    # Open input database
    # -------------------------------------------------------------------------

    with tables.open_file(
        args.input_database,
        mode="r",
    ) as h5_in:

        # ---------------------------------------------------------------------
        # Locate spectrogram table
        # ---------------------------------------------------------------------

        table_in = None

        for node in h5_in.walk_nodes(
            "/",
            classname="Table",
        ):

            if hasattr(node.attrs, "audio_repres"):
                table_in = node
                break

        if table_in is None:
            raise RuntimeError(
                "Could not find a table with "
                "'audio_repres' attribute."
            )

        nrows = table_in.nrows

        logger.info(
            "Input table          : %s",
            table_in._v_pathname,
        )

        logger.info(
            "Input rows           : %d",
            nrows,
        )

        # ---------------------------------------------------------------------
        # Determine input/output shapes
        # ---------------------------------------------------------------------

        old_description = table_in.description

        original_shape = (
            old_description
            ._v_colobjects["data"]
            .shape
        )

        if args.resize == (None, None):
            output_shape = original_shape
        else:
            output_shape = tuple(args.resize)

        logger.info(
            "Input spectrogram    : %s",
            original_shape,
        )

        logger.info(
            "Output spectrogram   : %s",
            output_shape,
        )

        # ---------------------------------------------------------------------
        # Create output database
        # ---------------------------------------------------------------------

        logger.info(
            "Creating output database..."
        )

        with tables.open_file(
            output_database,
            mode="w",
        ) as h5_out:

            # -------------------------------------------------------------
            # Re-create parent group structure
            # -------------------------------------------------------------

            parent_path = (
                table_in
                ._v_parent
                ._v_pathname
            )

            if parent_path == "/":

                parent = h5_out.root

            else:

                parent = h5_out.create_group(
                    "/",
                    parent_path.strip("/"),
                )

            # -------------------------------------------------------------
            # Build output table description
            # -------------------------------------------------------------

            new_columns = {}

            for name in old_description._v_names:

                old_col = (
                    old_description
                    ._v_colobjects[name]
                )

                if name == "data":

                    new_columns[name] = tables.Float32Col(
                        shape=output_shape,
                        pos=old_col._v_pos,
                    )

                else:

                    new_columns[name] = copy.copy(
                        old_col
                    )

            # -------------------------------------------------------------
            # Create output table
            # -------------------------------------------------------------

            table_out = h5_out.create_table(
                parent,
                table_in.name,
                description=new_columns,
                title=table_in.title,
            )

            # -------------------------------------------------------------
            # Copy table-level attributes
            # -------------------------------------------------------------

            attribute_errors = 0

            for attr_name in table_in.attrs._v_attrnames:

                try:

                    value = getattr(
                        table_in.attrs,
                        attr_name,
                    )

                    setattr(
                        table_out.attrs,
                        attr_name,
                        value,
                    )

                except Exception as exc:

                    attribute_errors += 1

                    logger.warning(
                        "Could not copy attribute '%s': %s",
                        attr_name,
                        exc,
                    )

            if attribute_errors:
                logger.warning(
                    "%d table attributes could not be copied.",
                    attribute_errors,
                )

            # -------------------------------------------------------------
            # Prepare transform configuration
            # -------------------------------------------------------------

            transform_kwargs = {
                "k": args.k,
                "perc": args.perc,
                "thr_ft": args.threshold_scale_ft,
                "kernel_size": tuple(args.kernel_size),
                "n_iter": args.n_iter,
                "resized_axis": tuple(args.resize),
                "wavelet": args.wavelet,
                "w_level": args.wavelet_level,
                "w_mode": args.wavelet_mode,
                "flag_svd": args.flag_svd,
                "flag_S": args.flag_S,
                "flag_G": args.flag_G,
            }

            # -------------------------------------------------------------
            # Processing statistics
            # -------------------------------------------------------------

            processed_count = 0
            failed_count = 0
            written_count = 0

            processing_times = []
            errors = []

            total_processing_time = 0.0

            # -------------------------------------------------------------
            # Start multiprocessing
            # -------------------------------------------------------------

            logger.info("=" * 78)
            logger.info(
                "Starting multiprocessing: %d workers",
                args.workers,
            )
            logger.info("=" * 78)

            processing_start = time.perf_counter()

            row_out = table_out.row

            with mp.Pool(
                processes=args.workers,
                initializer=_init_worker,
                initargs=(
                    args.input_database,
                    table_in._v_pathname,
                    transform_kwargs,
                ),
            ) as pool:

                iterator = pool.imap(
                    _process_spectrogram,
                    range(nrows),
                    chunksize=args.chunksize,
                )

                with tqdm(
                    iterator,
                    total=nrows,
                    desc="EnhanceSNR",
                    unit="spec",
                    dynamic_ncols=True,
                ) as progress:

                    for result in progress:

                        index = result["index"]

                        elapsed = result["elapsed"]

                        total_processing_time += elapsed

                        processing_times.append(
                            elapsed
                        )

                        row_in = table_in[index]

                        # -------------------------------------------------
                        # Successful processing
                        # -------------------------------------------------

                        if result["success"]:

                            processed_data = result["data"]

                            # Validate output shape.
                            if (
                                processed_data.shape
                                != output_shape
                            ):

                                error_message = (
                                    "Unexpected output shape: "
                                    f"{processed_data.shape}; "
                                    f"expected {output_shape}"
                                )

                                logger.error(
                                    "Row %d failed: %s",
                                    index,
                                    error_message,
                                )

                                failed_count += 1

                                error_record = {
                                    "index": index,
                                    "elapsed": elapsed,
                                    "error_type": "ShapeError",
                                    "error_message": error_message,
                                    "traceback": "",
                                }

                                errors.append(
                                    error_record
                                )

                                # Preserve original data.
                                processed_data = np.asarray(
                                    row_in["data"],
                                    dtype=np.float32,
                                )

                            else:

                                processed_count += 1

                        # -------------------------------------------------
                        # Failed processing
                        # -------------------------------------------------

                        else:

                            failed_count += 1

                            logger.error(
                                "Row %d failed after %.3f s: "
                                "%s: %s",
                                index,
                                elapsed,
                                result["error_type"],
                                result["error_message"],
                            )

                            logger.debug(
                                "Traceback for row %d:\n%s",
                                index,
                                result["traceback"],
                            )

                            errors.append(
                                {
                                    "index": index,
                                    "elapsed": elapsed,
                                    "error_type": (
                                        result["error_type"]
                                    ),
                                    "error_message": (
                                        result["error_message"]
                                    ),
                                    "traceback": (
                                        result["traceback"]
                                    ),
                                }
                            )

                            # Preserve the row in the output database
                            # by writing the original spectrogram.
                            processed_data = np.asarray(
                                row_in["data"],
                                dtype=np.float32,
                            )

                            # If the original shape differs from the
                            # requested output shape, the failed row
                            # cannot be copied directly.
                            if (
                                processed_data.shape
                                != output_shape
                            ):

                                logger.error(
                                    "Row %d original shape %s does "
                                    "not match output shape %s. "
                                    "Skipping row.",
                                    index,
                                    processed_data.shape,
                                    output_shape,
                                )

                                continue

                        # -------------------------------------------------
                        # Write row
                        # -------------------------------------------------

                        for name in old_description._v_names:

                            if name == "data":

                                row_out[name] = (
                                    processed_data
                                )

                            else:

                                row_out[name] = (
                                    row_in[name]
                                )

                        row_out.append()

                        written_count += 1

                        # -------------------------------------------------
                        # Periodic flush
                        # -------------------------------------------------

                        if (
                            written_count
                            % args.flush_interval
                            == 0
                        ):

                            table_out.flush()

                        # -------------------------------------------------
                        # Update progress information
                        # -------------------------------------------------

                        completed = (
                            processed_count
                            + failed_count
                        )

                        elapsed_total = (
                            time.perf_counter()
                            - processing_start
                        )

                        if elapsed_total > 0:

                            rate = (
                                completed
                                / elapsed_total
                            )

                            progress.set_postfix(
                                {
                                    "ok": processed_count,
                                    "err": failed_count,
                                    "rate": (
                                        f"{rate:.2f}/s"
                                    ),
                                }
                            )

            # -----------------------------------------------------------------
            # Final flush
            # -----------------------------------------------------------------

            table_out.flush()

            processing_elapsed = (
                time.perf_counter()
                - processing_start
            )

            logger.info(
                "Output table successfully flushed."
            )

            logger.info(
                "Output rows written: %d",
                table_out.nrows,
            )

    # =========================================================================
    # Write error report
    # =========================================================================

    if errors:

        write_error_report(
            error_file,
            errors,
        )

        logger.warning(
            "Error report written to: %s",
            error_file,
        )

    else:

        logger.info(
            "No spectrogram processing errors detected."
        )

        # Do not create an empty error file.

    # =========================================================================
    # Final performance statistics
    # =========================================================================

    total_elapsed = (
        time.perf_counter()
        - run_start
    )

    if processing_times:

        avg_time = np.mean(
            processing_times
        )

        median_time = np.median(
            processing_times
        )

        min_time = np.min(
            processing_times
        )

        max_time = np.max(
            processing_times
        )

    else:

        avg_time = 0.0
        median_time = 0.0
        min_time = 0.0
        max_time = 0.0

    if processing_elapsed > 0:

        throughput_sec = (
            (processed_count + failed_count)
            / processing_elapsed
        )

        throughput_min = (
            throughput_sec * 60.0
        )

    else:

        throughput_sec = 0.0
        throughput_min = 0.0

    success_rate = (
        100.0 * processed_count / nrows
        if nrows > 0
        else 0.0
    )

    failure_rate = (
        100.0 * failed_count / nrows
        if nrows > 0
        else 0.0
    )

    # =========================================================================
    # Final summary
    # =========================================================================

    logger.info("")
    logger.info("=" * 78)
    logger.info("FINAL PERFORMANCE SUMMARY")
    logger.info("=" * 78)

    logger.info(
        "Total input spectrograms : %d",
        nrows,
    )

    logger.info(
        "Successfully processed   : %d",
        processed_count,
    )

    logger.info(
        "Failed                   : %d",
        failed_count,
    )

    logger.info(
        "Rows written             : %d",
        written_count,
    )

    logger.info(
        "Success rate             : %.2f%%",
        success_rate,
    )

    logger.info(
        "Failure rate             : %.2f%%",
        failure_rate,
    )

    logger.info(
        "Processing time          : %s",
        format_duration(processing_elapsed),
    )

    logger.info(
        "Total runtime            : %s",
        format_duration(total_elapsed),
    )

    logger.info(
        "Average spectrogram time : %.4f s",
        avg_time,
    )

    logger.info(
        "Median spectrogram time  : %.4f s",
        median_time,
    )

    logger.info(
        "Minimum spectrogram time : %.4f s",
        min_time,
    )

    logger.info(
        "Maximum spectrogram time : %.4f s",
        max_time,
    )

    logger.info(
        "Throughput               : %.3f spectrograms/s",
        throughput_sec,
    )

    logger.info(
        "Throughput               : %.2f spectrograms/min",
        throughput_min,
    )

    if errors:

        logger.info(
            "Error report             : %s",
            error_file,
        )

    else:

        logger.info(
            "Error report             : None"
        )

    logger.info(
        "Output database          : %s",
        output_database,
    )

    logger.info(
        "Log file                 : %s",
        log_file,
    )

    logger.info("=" * 78)
    logger.info("EnhanceSNR processing completed.")
    logger.info("=" * 78)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":

    # Required for multiprocessing, especially on Windows/macOS.
    mp.freeze_support()

    main()