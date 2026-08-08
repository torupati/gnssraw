import logging
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

_GOODSTYLE_RC = {
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#fbfbfc",
    "axes.edgecolor": "#d1d5db",
    "axes.labelcolor": "#111827",
    "axes.grid": True,
    "grid.color": "#e5e7eb",
    "grid.linewidth": 0.6,
    "xtick.color": "#374151",
    "ytick.color": "#374151",
    "text.color": "#111827",
    "legend.frameon": False,
    "legend.fontsize": "small",
}


def _apply_goodstyle(fig, axes):
    """Apply a light-and-minimal style to *fig* and all *axes* in-place."""
    fig.patch.set_facecolor(_GOODSTYLE_RC["figure.facecolor"])
    colors = plt.cm.tab10.colors
    for ax in axes:
        ax.set_facecolor(_GOODSTYLE_RC["axes.facecolor"])
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_edgecolor(_GOODSTYLE_RC["axes.edgecolor"])
        ax.tick_params(colors=_GOODSTYLE_RC["xtick.color"])
        ax.yaxis.label.set_color(_GOODSTYLE_RC["axes.labelcolor"])
        ax.xaxis.label.set_color(_GOODSTYLE_RC["axes.labelcolor"])
        ax.grid(
            True,
            color=_GOODSTYLE_RC["grid.color"],
            linewidth=_GOODSTYLE_RC["grid.linewidth"],
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.set_frame_on(False)
        # Re-colour existing lines/scatter with tab10
        for idx, line in enumerate(ax.get_lines()):
            line.set_color(colors[idx % len(colors)])


def plot_satellite_observations(
    epochs,
    output_dir: Path,
    plot_mode: int = 1,
    show_ambiguity_statistics: bool = False,
    goodstyle: bool = False,
):
    """
    Plot observations for each satellite.

    Args:
        epochs: List of EpochObservations
        output_dir: Directory to save plots
        plot_mode: Mode of plotting
        show_ambiguity_statistics: Whether to show ambiguity statistics (widelane, ionofree, geofree, multipath)
        goodstyle: Apply a light-and-minimal style to the plot (white/light-grey bg, subtle grid, hidden top/right spines)
    """
    # Organize data by satellite
    satellite_data: dict[str, dict] = {}

    for epoch in epochs:
        for sat_list, system_code in [
            (epoch.satellites_gps, "G"),
            (epoch.satellites_qzss, "J"),
            (epoch.satellites_galileo, "E"),
            (epoch.satellites_glonass, "R"),
        ]:
            for sat_obs in sat_list:
                sat_id = f"{system_code}{sat_obs.prn:02d}"
                if sat_id not in satellite_data:
                    satellite_data[sat_id] = {
                        "pseudorange": {},
                        "carrier_phase": {},
                        "doppler": {},
                        "snr": {},
                        "ambiguities": {},  # key: combination like "L1_L2", value: {"widelane": {...}, "ionofree": {...}}
                    }

                for band_name, signal_obs in sat_obs.signals.items():
                    if band_name not in satellite_data[sat_id]["pseudorange"]:
                        satellite_data[sat_id]["pseudorange"][band_name] = {
                            "times": [],
                            "values": [],
                        }
                        satellite_data[sat_id]["carrier_phase"][band_name] = {
                            "times": [],
                            "values": [],
                        }
                        satellite_data[sat_id]["doppler"][band_name] = {
                            "times": [],
                            "values": [],
                        }
                        satellite_data[sat_id]["snr"][band_name] = {
                            "times": [],
                            "values": [],
                        }

                    satellite_data[sat_id]["pseudorange"][band_name]["times"].append(epoch.datetime)
                    satellite_data[sat_id]["pseudorange"][band_name]["values"].append(signal_obs.pseudorange)
                    satellite_data[sat_id]["carrier_phase"][band_name]["times"].append(epoch.datetime)
                    satellite_data[sat_id]["carrier_phase"][band_name]["values"].append(signal_obs.carrier_phase)
                    satellite_data[sat_id]["doppler"][band_name]["times"].append(epoch.datetime)
                    satellite_data[sat_id]["doppler"][band_name]["values"].append(signal_obs.doppler_)
                    satellite_data[sat_id]["snr"][band_name]["times"].append(epoch.datetime)
                    satellite_data[sat_id]["snr"][band_name]["values"].append(signal_obs.snr)

                # Add ambiguity data for all available combinations
                for comb_name, amb_obs in sat_obs.ambiguities.items():
                    if comb_name not in satellite_data[sat_id]["ambiguities"]:
                        satellite_data[sat_id]["ambiguities"][comb_name] = {
                            "widelane": {"times": [], "values": []},
                            "ionofree": {"times": [], "values": []},
                            "geofree": {"times": [], "values": []},
                            "multipath": {"times": [], "values": []},
                        }
                    satellite_data[sat_id]["ambiguities"][comb_name]["widelane"]["times"].append(epoch.datetime)
                    satellite_data[sat_id]["ambiguities"][comb_name]["widelane"]["values"].append(amb_obs.widelane)
                    satellite_data[sat_id]["ambiguities"][comb_name]["ionofree"]["times"].append(epoch.datetime)
                    satellite_data[sat_id]["ambiguities"][comb_name]["ionofree"]["values"].append(amb_obs.ionofree)
                    satellite_data[sat_id]["ambiguities"][comb_name]["geofree"]["times"].append(epoch.datetime)
                    satellite_data[sat_id]["ambiguities"][comb_name]["geofree"]["values"].append(amb_obs.geofree)
                    satellite_data[sat_id]["ambiguities"][comb_name]["multipath"]["times"].append(epoch.datetime)
                    satellite_data[sat_id]["ambiguities"][comb_name]["multipath"]["values"].append(amb_obs.multipath)

    # Create plots for each satellite
    for sat_id, data in satellite_data.items():
        logger.info(f"Plotting satellite {sat_id}")

        # Determine common time range across all plots for this satellite
        all_times = []
        for band_data in data["pseudorange"].values():
            all_times.extend(band_data["times"])
        for band_data in data["carrier_phase"].values():
            all_times.extend(band_data["times"])
        for band_data in data["doppler"].values():
            all_times.extend(band_data["times"])
        for band_data in data["snr"].values():
            all_times.extend(band_data["times"])
        for comb_data in data["ambiguities"].values():
            all_times.extend(comb_data["widelane"]["times"])
            all_times.extend(comb_data["ionofree"]["times"])
            all_times.extend(comb_data["geofree"]["times"])
            all_times.extend(comb_data["multipath"]["times"])
        if all_times:
            common_start = min(all_times)
            common_end = max(all_times)
        else:
            common_start = None
            common_end = None

        # Check if ambiguity data is available and count combinations
        num_ambiguity_combos = len(data["ambiguities"])
        has_ambiguity = num_ambiguity_combos > 0

        # Determine which plots to show based on plot_mode
        show_pseudorange = plot_mode in {1, 4}
        show_carrier_phase = plot_mode in {1, 4}
        show_doppler = plot_mode in {1, 4, 5, 6}
        show_snr = plot_mode in {1, 2, 3, 4, 5, 6}
        # For ambiguity-related plots, we will determine which ones to show based on the mode
        show_widelane = plot_mode in {1, 2, 3, 5, 6}
        show_ionofree = plot_mode in {1, 3}
        show_geofree = plot_mode in {1, 3}
        show_multipath = plot_mode in {1, 3, 5}

        # Create a figure with variable subplots
        num_rows = 0
        if show_pseudorange:
            num_rows += 1  # pseudorange
        if show_carrier_phase:
            num_rows += 1
        if show_doppler:
            num_rows += 1  # doppler only
        if show_snr:
            num_rows += 1
        if has_ambiguity:
            if show_widelane:
                num_rows += num_ambiguity_combos
            if show_ionofree:
                num_rows += num_ambiguity_combos
            if show_geofree:
                num_rows += num_ambiguity_combos
            if show_multipath:
                num_rows += num_ambiguity_combos  # multipath per combination

        fig, axes = plt.subplots(num_rows, 1, figsize=(10, 3 * num_rows), squeeze=False)
        fig.suptitle(f"Satellite {sat_id} Observations", fontsize=16)

        # Flatten to a 1D list of Axes for consistent indexing.
        axes = cast(Any, axes.flatten())

        plot_idx = 0

        # Plot pseudorange
        if show_pseudorange:
            for band_name, band_data in data["pseudorange"].items():
                if band_data["values"]:
                    axes[plot_idx].plot(
                        band_data["times"],
                        band_data["values"],
                        marker=".",
                        linestyle="None",
                        label=band_name,
                    )
            axes[plot_idx].set_ylabel("Pseudorange (m)")
            axes[plot_idx].legend()
            axes[plot_idx].grid(True)
            plot_idx += 1

        if show_carrier_phase:
            # Plot carrier phase
            for band_name, band_data in data["carrier_phase"].items():
                if band_data["values"]:
                    axes[plot_idx].plot(
                        band_data["times"],
                        band_data["values"],
                        marker=".",
                        linestyle="None",
                        label=band_name,
                    )
            axes[plot_idx].set_ylabel("Carrier Phase (cycles)")
            axes[plot_idx].legend()
            axes[plot_idx].grid(True)
            plot_idx += 1
        if show_doppler:
            # Plot doppler
            for band_name, band_data in data["doppler"].items():
                if band_data["values"]:
                    axes[plot_idx].plot(
                        band_data["times"],
                        band_data["values"],
                        marker=".",
                        linestyle="None",
                        label=band_name,
                    )
            axes[plot_idx].set_ylabel("Doppler (Hz)")
            axes[plot_idx].legend()
            axes[plot_idx].grid(True)
            plot_idx += 1

        # Plot SNR
        if show_snr:
            for band_name, band_data in data["snr"].items():
                if band_data["values"]:
                    axes[plot_idx].plot(
                        band_data["times"],
                        band_data["values"],
                        marker=".",
                        linestyle="None",
                        label=band_name,
                    )
            axes[plot_idx].set_ylabel("SNR (dB-Hz)")
            axes[plot_idx].legend()
            axes[plot_idx].grid(True)
            plot_idx += 1

        # Plot ambiguity data for each combination
        if has_ambiguity:
            colors = ["purple", "orange", "green", "red", "blue", "brown"]
            for comb_idx, (comb_name, comb_data) in enumerate(data["ambiguities"].items()):
                color_wl = colors[comb_idx * 2 % len(colors)]
                color_if = colors[(comb_idx * 2 + 1) % len(colors)]

                # Plot widelane ambiguity for this combination
                if show_widelane:
                    wl_times = comb_data["widelane"]["times"]
                    wl_values = comb_data["widelane"]["values"]
                    if wl_values:
                        axes[plot_idx].plot(
                            wl_times,
                            wl_values,
                            marker=".",
                            linestyle="None",
                            label=f"{comb_name} WL",
                            color=color_wl,
                        )
                        # Set ylim if range is less than 1
                        if max(wl_values) - min(wl_values) < 1:
                            mean_val = sum(wl_values) / len(wl_values)
                            axes[plot_idx].set_ylim(mean_val - 3, mean_val + 3)
                        if show_ambiguity_statistics:
                            mean_wl = sum(wl_values) / len(wl_values)
                            std_wl = (sum((x - mean_wl) ** 2 for x in wl_values) / len(wl_values)) ** 0.5
                            axes[plot_idx].set_title(
                                f"{comb_name} Widelane Ambiguity (mean={mean_wl:.2f}, std={std_wl:.2f}, max-min={max(wl_values) - min(wl_values):.2f} cycles)"
                            )
                    axes[plot_idx].set_ylabel(f"Widelane {comb_name} (cycles)")
                    axes[plot_idx].legend()
                    axes[plot_idx].grid(True)
                    plot_idx += 1

                # Plot ionofree ambiguity for this combination
                if show_ionofree:
                    if_times = comb_data["ionofree"]["times"]
                    if_values = comb_data["ionofree"]["values"]
                    if if_values:
                        axes[plot_idx].plot(
                            if_times,
                            if_values,
                            marker=".",
                            linestyle="None",
                            label=f"{comb_name} IF",
                            color=color_if,
                        )
                        # Set ylim if range is less than 1
                        if max(if_values) - min(if_values) < 1:
                            mean_val = sum(if_values) / len(if_values)
                            axes[plot_idx].set_ylim(mean_val - 3, mean_val + 3)
                        if show_ambiguity_statistics:
                            mean_if = sum(if_values) / len(if_values)
                            std_if = (sum((x - mean_if) ** 2 for x in if_values) / len(if_values)) ** 0.5
                            axes[plot_idx].set_title(
                                f"{comb_name} Ionofree Ambiguity (mean={mean_if:.2f}, std={std_if:.2f}, max-min={max(if_values) - min(if_values):.2f} cycles)"
                            )
                    axes[plot_idx].legend()
                    axes[plot_idx].grid(True)
                    plot_idx += 1
                # Plot geofree ambiguity for this combination
                if show_geofree:
                    gf_times = comb_data["geofree"]["times"]
                    gf_values = comb_data["geofree"]["values"]
                    if gf_values:
                        axes[plot_idx].plot(
                            gf_times,
                            gf_values,
                            marker=".",
                            linestyle="None",
                            label=f"{comb_name} GF",
                            color=color_if,
                        )
                        # Set ylim if range is less than 1
                        if max(gf_values) - min(gf_values) < 1:
                            mean_val = sum(gf_values) / len(gf_values)
                            axes[plot_idx].set_ylim(mean_val - 2, mean_val + 2)
                        if show_ambiguity_statistics:
                            mean_gf = sum(gf_values) / len(gf_values)
                            std_gf = (sum((x - mean_gf) ** 2 for x in gf_values) / len(gf_values)) ** 0.5
                            axes[plot_idx].set_title(
                                f"{comb_name} Geofree Ambiguity (mean={mean_gf:.2f}, std={std_gf:.2f}, max-min={max(gf_values) - min(gf_values):.2f} cycles)"
                            )
                    axes[plot_idx].set_ylabel(f"Geofree {comb_name} (cycles)")
                    axes[plot_idx].legend()
                    axes[plot_idx].grid(True)
                    plot_idx += 1
                # Plot multipath
                elif show_multipath:
                    mp_times = comb_data["multipath"]["times"]
                    mp_values = comb_data["multipath"]["values"]
                    if mp_values:
                        axes[plot_idx].plot(
                            mp_times,
                            mp_values,
                            marker=".",
                            linestyle="None",
                            label=f"{comb_name} Multipath",
                            color=color_if,
                        )
                        # Set ylim if range is less than 2
                        if max(mp_values) - min(mp_values) < 2:
                            mean_val = sum(mp_values) / len(mp_values)
                            axes[plot_idx].set_ylim(mean_val - 1, mean_val + 1)
                        if show_ambiguity_statistics:
                            mean_mp = sum(mp_values) / len(mp_values)
                            std_mp = (sum((x - mean_mp) ** 2 for x in mp_values) / len(mp_values)) ** 0.5
                            axes[plot_idx].set_title(
                                f"{comb_name} Multipath (mean={mean_mp:.2f}, std={std_mp:.2f}, max-min={max(mp_values) - min(mp_values):.2f})"
                            )
                    axes[plot_idx].legend()
                    axes[plot_idx].grid(True)
                    plot_idx += 1

            # Rotate x-axis labels for better readability
            plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")
        else:
            # Rotate x-axis labels for better readability
            plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")

        if goodstyle:
            _apply_goodstyle(fig, axes)

        plt.tight_layout(rect=(0, 0, 1, 0.97) if goodstyle else (0, 0, 1, 1))

        if common_start is not None and common_end is not None:
            for ax in axes:
                ax.set_xlim(common_start, common_end)

        if len(axes) > 0:
            axes[-1].set_xlabel("GPS Time")

        # Save the plot
        output_file = output_dir / f"{sat_id}_observations{plot_mode}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved plot to {output_file}")
