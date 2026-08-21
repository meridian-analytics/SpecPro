
"""
Apply EnhanceSNR post-processing to a Ketos HDF5 spectrogram database.

This script:

1. Reads an existing Ketos-style HDF5 database.
2. Locates the spectrogram table using its ``audio_repres`` attribute.
3. Creates a new HDF5 database with the same table structure.
4. Applies the ``EnhanceSNR`` transformation to every spectrogram.
5. Resizes the processed spectrograms to the requested dimensions.
6. Copies all metadata and non-data columns to the output database.

The input database is opened read-only and is never modified.

Example
-------
python enhance_database.py input.h5 output.h5 \
    --k 1.0 \
    --perc 50 \
    --threshold_scale_ft 0.7 \
    --kernel_size 0.8 0.4 \
    --n_iter 2 \
    --resize 240 240
"""

# =============================================================================
# =============================================================================
# ================================= Libraries =================================
# =============================================================================
# =============================================================================

# ---- Standard libraries
import argparse
import copy
from datetime import datetime
from pathlib import Path

# ---- Third-party libraries
import numpy as np
import tables

# ---- Custom modules
from pam_processing_pipline import EnhanceSNR


# =============================================================================
# =============================================================================
# ======================= COMMAND-LINE ARGUMENT PARSING =======================
# =============================================================================
# =============================================================================

def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Namespace containing input/output paths and spectrogram
        enhancement parameters.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Apply spectrogram SNR enhancement and denoising "
            "to a Ketos HDF5 spectrogram database."
        )
    )

    # =============================================================================
    # INPUT / OUTPUT DATABASES
    # =============================================================================

    parser.add_argument(
        "input_database",
        type=str,
        help="Path to the input HDF5 database.",
    )

    parser.add_argument(
        "output_database",
        type=str,
        help="Path where the enhanced HDF5 database will be created.",
    )

    # =============================================================================
    # SPECTROGRAM REFINEMENT
    # =============================================================================

    parser.add_argument(
        "--k",
        type=float,
        default=2,
        help=(
            "Scaling parameter for the softplus compression. "
            "Default: 2."
        ),
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
        default=1,
        help=(
            "Enable SVD-based background removal: "
            "1 = enabled, 0 = disabled. Default: 1."
        ),
    )

    parser.add_argument(
        "--flag_S",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Enable softplus compression: "
            "1 = enabled, 0 = disabled. Default: 1."
        ),
    )

    parser.add_argument(
        "--flag_G",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Enable Gaussian smoothing: "
            "1 = enabled, 0 = disabled. Default: 1."
        ),
    )

    # =============================================================================
    # WIENER FILTER
    # =============================================================================

    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help=(
            "Number of adaptive Wiener-filter iterations. "
            "Default: 2."
        ),
    )

    # =============================================================================
    # WAVELET DENOISING
    # =============================================================================

    parser.add_argument(
        "--wavelet",
        type=str,
        default="db4",
        help=(
            "Wavelet used for 2-D wavelet denoising. "
            "Default: db4."
        ),
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
        "--threshold_scale_ft",
        type=float,
        default=1.0,
        help=(
            "Scaling factor applied to the wavelet universal threshold "
            "to control denoising strength. Default: 1.0."
        ),
    )

    parser.add_argument(
        "--wavelet_mode",
        type=str,
        choices=["soft", "hard"],
        default="soft",
        help=(
            "Wavelet thresholding mode. Default: soft."
        ),
    )

    # =============================================================================
    # SPECTROGRAM RESIZING
    # =============================================================================

    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=(None, None),
        metavar=("FREQ", "TIME"),
        help=(
            "Output spectrogram size as frequency x time. "
            "Default: None"
        ),
    )

    return parser.parse_args()


# =============================================================================
# =============================================================================
# =================================== MAIN ====================================
# =============================================================================
# =============================================================================

def main():
    """
    Run EnhanceSNR processing on the input HDF5 database.

    The input database is opened in read-only mode. A completely new
    output database is created so that the original data remains
    unchanged.

    Each spectrogram is processed independently and written to the
    output table together with all of its original metadata columns.
    """

    # Parse command-line arguments
    args = parse_args()

    print("\nEnhanceSNR configuration")
    print("------------------------")

    print(f"Input       : {args.input_database}")
    print(f"Output      : {args.output_database}")
    print(f"k           : {args.k}")
    print(f"perc        : {args.perc}")
    print(f"thr_ft      : {args.threshold_scale_ft}")
    print(f"kernel_size : {tuple(args.kernel_size)}")
    print(f"n_iter      : {args.n_iter}")
    print(f"resize      : {args.resize}")
    print()

    # Create the EnhanceSNR transformation
    transform = EnhanceSNR(
        k=args.k,
        perc=args.perc,
        thr_ft=args.threshold_scale_ft,
        kernel_size=tuple(args.kernel_size),
        n_iter=args.n_iter,
        resized_axis=tuple(args.resize),
    )
    
    # Open the input HDF5 database & Locate the spectrogram table
    with tables.open_file(args.input_database, mode="r") as h5_in:

        # Locate the spectrogram table
        table_in = None

        for node in h5_in.walk_nodes("/", classname="Table"):
            if hasattr(node.attrs, "audio_repres"):
                table_in = node
                break

        # Stop if no suitable spectrogram table was found.
        if table_in is None:
            raise RuntimeError(
                "Could not find a table with "
                "'audio_repres' attribute."
            )
        
        print(f"Found input table: {table_in._v_pathname}")
        print(f"Number of rows: {table_in.nrows}")

        # Create the output HDF5 database
        output_path = Path(args.output_database)

        # Generate a timestamp using the current date and time.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Insert the timestamp before the file extension.
        output_database = (output_path.parent / f"{output_path.stem}_{timestamp}{output_path.suffix}")

        print(f"Output database: {output_database}")

        # Open the timestamped output database.
        with tables.open_file(output_database, mode="w") as h5_out:
            # Re-create the parent group structure
            parent_path = table_in._v_parent._v_pathname

            if parent_path == "/":
                # The table is directly under the root node.
                parent = h5_out.root
            else:
                # Create the corresponding parent group in the output database.
                parent = h5_out.create_group("/", parent_path.strip("/"))

            # Build new table description (processed spectrogram may have a different shape from the original data)
            old_description = table_in.description

            original_shape = old_description._v_colobjects["data"].shape
            
            if args.resize == (None, None):
                output_shape = original_shape
            else:
                output_shape = tuple(args.resize)

            new_columns = {}

            for name in old_description._v_names:
                # Retrieve the PyTables column definition.
                old_col = old_description._v_colobjects[name]

                if name == "data":
                    # Replace the original spectrogram column
                    new_columns[name] = tables.Float32Col(shape=output_shape, pos=old_col._v_pos)
                else:
                    new_columns[name] = copy.copy(old_col)

            # Create the output table
            table_out = h5_out.create_table(parent, table_in.name, description=new_columns, title=table_in.title)

            # Copy table-level attributes
            for attr_name in table_in.attrs._v_attrnames:

                try:
                    value = getattr(table_in.attrs, attr_name)
                    setattr(table_out.attrs, attr_name, value)
                except Exception as e:
                    print(f"Warning: could not copy attribute {attr_name}: {e}")

            # Extract spectrogram metadata required by EnhanceSNR
            meta = table_in.attrs.audio_repres

            freq_res = meta["freq_res"]
            time_res = meta["time_res"]
            freq_min = meta["freq_min"]

            print("\nProcessing...\n")

            # Prepare the output row object
            row_out = table_out.row

            # Process every spectrogram in the database
            for i in range(table_in.nrows):

                # Read one row from the input table
                row_in = table_in[i]

                # Extract the original spectrogram
                S = np.asarray(row_in["data"], dtype=np.float32)

                # Display processing progress.

                print(f"\rProcessing {i + 1}/{table_in.nrows}", end="")

                # ======================================================
                # Create a lightweight Ketos-like representation
                # ======================================================
                # EnhanceSNR expects an object containing:
                #
                #   data      -> spectrogram
                #   freq_res  -> frequency resolution
                #   time_res  -> time resolution
                #   freq_min  -> minimum frequency
                #
                # Instead of constructing a complete Ketos object, a
                # minimal object containing the required attributes is
                # sufficient for the transformation.
                # ======================================================

                class Rep:
                    """Minimal representation required by EnhanceSNR."""

                    pass

                rep = Rep()

                rep.data = S
                rep.freq_res = freq_res
                rep.time_res = time_res
                rep.freq_min = freq_min

                # Apply EnhanceSNR
                rep = transform(rep)

                # Validate the output shape
                expected_shape = output_shape

                if rep.data.shape != expected_shape:
                    raise RuntimeError(
                        f"Unexpected output shape at row {i}: "
                        f"{rep.data.shape}; "
                        f"expected {expected_shape}"
                    )

                # Copy the processed row into the output table
                for name in old_description._v_names:
                    if name == "data":
                        # Store the enhanced spectrogram.
                        row_out[name] = rep.data
                    else:
                        # Preserve the original value.
                        row_out[name] = row_in[name]

                # Append the completed row to the output table.
                row_out.append()

                # Periodically flush data to disk
                if (i + 1) % 100 == 0:
                    table_out.flush()

            # Final flush
            table_out.flush()
            print("\n")
            print(f"Output table shape: {table_out.cols.data.shape}")


    # Processing completed
    print("\nDONE")
    print(f"Enhanced database: {args.output_database}")

if __name__ == "__main__":
    main()

