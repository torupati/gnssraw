import argparse
import json
from pathlib import Path
from logging import getLogger, basicConfig, INFO


from app.gnss.epoch_series import smoothing_code_range_of_receiver
from app.gnss.satellite_signals import (
    EpochObservations,
    parse_rinex_observation_file,
    save_gnss_observations_to_json,
    save_ambiguity_statistics_to_csv,
    calculate_combined_observations,
)
from app.gnss.rtcm3 import read_rtcm3_file
from app.gnss.plot.observables import plot_satellite_observations

logger = getLogger(__name__)


def parse_plot_modes(mode_str):
    """Parse plot modes from comma-separated string.

    Args:
        mode_str: Comma-separated plot modes (e.g., '1' or '1,2,3')

    Returns:
        Single int if one mode, list of ints if multiple modes
    """
    modes = []
    for part in mode_str.split(","):
        try:
            mode = int(part.strip())
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid plot mode: {part.strip()}")
        if mode not in [1, 2, 3, 4, 5, 6]:
            raise argparse.ArgumentTypeError(
                f"Invalid plot mode: {mode}. Must be 1, 2, 3, 4, 5, or 6"
            )
        modes.append(mode)
    # Return single int if only one mode, otherwise return list
    return modes[0] if len(modes) == 1 else modes


def main():
    parser = argparse.ArgumentParser(
        description="Process RINEX observation files and generate plots"
    )
    parser.add_argument("input_file", type=str, help="Path to RINEX or RTCM3 file")
    parser.add_argument(
        "--input-type",
        type=str,
        choices=["RINEX3", "RTCM3"],
        default="RINEX3",
        help="Input file type: RINEX3 or RTCM3 (default: RINEX3)",
    )
    parser.add_argument(
        "--gpsweek",
        type=int,
        help="GPS week number (required when input-type=RTCM3)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./out/",
        help="Output directory for plots (default: ./out/)",
    )
    parser.add_argument(
        "--json",
        type=str,
        help="Save parsed data to JSON file (specify output file path)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Save ambiguity statistics to CSV file (specify output file path)",
    )
    parser.add_argument(
        "--plot-mode",
        type=parse_plot_modes,
        default=1,
        help=(
            "Plot data selection (can be comma-separated): 1=all, 2=SNR+widelane only, "
            "3=SNR+widelane+ionofree only, 4=PR+CP+Doppler+SNR only, "
            "5=Doppler+SNR+widelane+multipath only, "
            "6=Doppler+SNR+widelane only (no multipath). "
            "Example: --plot-mode 1 or --plot-mode 1,2,3"
        ),
    )
    parser.add_argument(
        "--show-statistics-figures",
        action="store_true",
        help="Show ambiguity statistics figures (widelane, ionofree, geofree, multipath)",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip generating plots",
    )
    parser.add_argument(
        "--signal-code-map",
        type=str,
        default=str(Path(__file__).parent / ".signal_code_map.json"),
        help=(
            "Path to JSON file that defines signal_code_map "
            "(required for RINEX3, default: .signal_code_map.json)"
        ),
    )
    parser.add_argument(
        "--plot-goodstyle",
        action="store_true",
        help=(
            "Apply a light-and-minimal style to plots "
            "(white/light-grey background, subtle grid, hidden top/right spines)"
        ),
    )
    parser.add_argument(
        "--carrier-smoothing",
        action="store_true",
        help="Enable carrier phase smoothing of pseudorange",
    )
    parser.add_argument(
        "--carrier-smoothing-slip-threshold",
        type=float,
        default=10.0,
        help=(
            "Cycle-slip detection threshold in meters for carrier smoothing "
            "(default: 10.0)"
        ),
    )
    parser.add_argument(
        "--carrier-smoothing-code-noise",
        type=float,
        default=3.0,
        help="Code noise sigma in meters for carrier smoothing (default: 3.0)",
    )
    parser.add_argument(
        "--carrier-smoothing-carrier-noise",
        type=float,
        default=0.02,
        help="Carrier noise sigma in meters for carrier smoothing (default: 0.02)",
    )

    args = parser.parse_args()

    # Setup logging
    basicConfig(level=INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Check if input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    # Create output directory if it doesn't exist
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Parse input file based on input type
    if args.input_type == "RINEX3":
        # Load signal code map for RINEX3
        signal_code_map_path = Path(args.signal_code_map)
        if not signal_code_map_path.exists():
            logger.error(f"Signal code map file not found: {signal_code_map_path}")
            return 1
        try:
            with signal_code_map_path.open("r", encoding="utf-8") as f:
                signal_code_map = json.load(f)
        except json.JSONDecodeError as exc:
            logger.error(
                f"Invalid JSON in signal code map file: {signal_code_map_path} ({exc})"
            )
            return 1

        # Parse RINEX file
        logger.info(f"Parsing RINEX3 file: {input_path}")
        epochs: list[EpochObservations] = parse_rinex_observation_file(
            str(input_path), signal_code_map
        )
        logger.info(
            f"... parsed {len(epochs)} epochs. {epochs[0].datetime if epochs else 'N/A'} to {epochs[-1].datetime if epochs else 'N/A'}"
        )
    elif args.input_type == "RTCM3":
        # Validate gpsweek argument
        if args.gpsweek is None:
            logger.error("--gpsweek is required when input-type=RTCM3")
            return 1

        # Parse RTCM3 file
        logger.info(f"Parsing RTCM3 file: {input_path} (GPS week: {args.gpsweek})")
        epochs: list[EpochObservations] = read_rtcm3_file(input_path, args.gpsweek)
        logger.info(
            f"... parsed {len(epochs)} epochs. {epochs[0].datetime if epochs else 'N/A'} to {epochs[-1].datetime if epochs else 'N/A'}"
        )
    else:
        logger.error(f"Unknown input type: {args.input_type}")
        return 1

    if args.carrier_smoothing:
        logger.info("Applying carrier phase smoothing to pseudorange...")
        epochs = smoothing_code_range_of_receiver(
            epochs,
            code_noise_m=args.carrier_smoothing_code_noise,
            carrier_noise_m=args.carrier_smoothing_carrier_noise,
            slip_threshold_m=args.carrier_smoothing_slip_threshold,
        )

    # Calculate combined observations
    logger.info("Calculating combined observations...")
    epochs = calculate_combined_observations(epochs)

    # Save to JSON if requested
    if args.json:
        json_output_path = Path(args.json)
        save_gnss_observations_to_json(epochs, json_output_path)
        logger.info(f"Saved parsed data to JSON: {json_output_path}")

    # Save ambiguity statistics to CSV if requested
    if args.csv:
        csv_output_path = Path(args.csv)
        save_ambiguity_statistics_to_csv(epochs, csv_output_path)
        logger.info(f"Saved ambiguity statistics to CSV: {csv_output_path}")

    # Generate plots
    if args.skip_plot:
        logger.info("Skipping plot generation (--skip-plot)")
    else:
        logger.info("Generating plots...")
        # Handle both single mode and multiple modes
        plot_modes = (
            args.plot_mode if isinstance(args.plot_mode, list) else [args.plot_mode]
        )
        for mode in plot_modes:
            plot_satellite_observations(
                epochs,
                output_dir,
                plot_mode=mode,
                show_ambiguity_statistics=args.show_statistics_figures,
                goodstyle=args.plot_goodstyle,
            )

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
