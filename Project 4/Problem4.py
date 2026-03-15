"""Frequentist-versus-Bayesian interval comparisons for Project 4 models.

This script compares linearized frequentist intervals with Bayesian credible and
posterior-predictive intervals for

1. the aluminum rod heat model from Problem 2, and
2. the SIR model from Problem 3.

It reports interval widths, empirical coverage of observed data, and generates a
summary comparison plot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Problem2 import frequentist_intervals_heat, run_problem2
from Problem3 import frequentist_intervals_sir, run_problem3

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "Figures & Data"


def empirical_coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Return the fraction of observations contained in an interval band."""

    return float(np.mean((y >= lower) & (y <= upper)))


def summarize_width(lower: np.ndarray, upper: np.ndarray) -> tuple[float, float]:
    """Return mean and maximum interval widths."""

    width = upper - lower
    return float(np.mean(width)), float(np.max(width))


def run_problem4() -> dict[str, dict[str, float]]:
    """Compare frequentist and Bayesian intervals for both Project 4 models."""

    heat_freq = frequentist_intervals_heat()
    sir_freq = frequentist_intervals_sir()

    heat_bayes = run_problem2(nsimu=4_000, burn_in=1_000, posterior_draws=1_000, seed=21, save_plots=False)
    sir_bayes = run_problem3(nsimu=2_500, burn_in=500, posterior_draws=750, seed=27, save_plots=False)

    summary = {
        "heat": {
            "frequentist_prediction_coverage": empirical_coverage(
                heat_freq.data_y, heat_freq.prediction_lower, heat_freq.prediction_upper
            ),
            "bayesian_prediction_coverage": empirical_coverage(
                heat_bayes.y_full, heat_bayes.predictive_lower, heat_bayes.predictive_upper
            ),
            "frequentist_confidence_mean_width": summarize_width(
                heat_freq.confidence_lower, heat_freq.confidence_upper
            )[0],
            "bayesian_credible_mean_width": summarize_width(
                heat_bayes.credible_lower, heat_bayes.credible_upper
            )[0],
        },
        "sir": {
            "frequentist_prediction_coverage": empirical_coverage(
                sir_freq.infected_data, sir_freq.prediction_lower, sir_freq.prediction_upper
            ),
            "bayesian_prediction_coverage": empirical_coverage(
                sir_bayes.infected_data, sir_bayes.predictive_lower, sir_bayes.predictive_upper
            ),
            "frequentist_confidence_mean_width": summarize_width(
                sir_freq.confidence_lower, sir_freq.confidence_upper
            )[0],
            "bayesian_credible_mean_width": summarize_width(
                sir_bayes.credible_lower, sir_bayes.credible_upper
            )[0],
        },
    }

    fig, axes = plt.subplots(2, 1, figsize=(11, 10))

    ax = axes[0]
    ax.fill_between(
        heat_freq.x,
        heat_freq.prediction_lower,
        heat_freq.prediction_upper,
        color="0.7",
        alpha=0.35,
        label="Frequentist prediction",
    )
    ax.fill_between(
        heat_freq.x,
        heat_freq.confidence_lower,
        heat_freq.confidence_upper,
        color="tab:orange",
        alpha=0.25,
        label="Frequentist confidence",
    )
    ax.plot(heat_freq.x, heat_freq.mean, color="black", lw=1.8, label="Mean response")
    ax.plot(heat_bayes.x_full, heat_bayes.credible_lower, "--", color="tab:blue", lw=1.6, label="Bayesian credible")
    ax.plot(heat_bayes.x_full, heat_bayes.credible_upper, "--", color="tab:blue", lw=1.6)
    ax.plot(heat_bayes.x_full, heat_bayes.predictive_lower, ":", color="tab:green", lw=2.0, label="Bayesian prediction")
    ax.plot(heat_bayes.x_full, heat_bayes.predictive_upper, ":", color="tab:green", lw=2.0)
    ax.scatter(heat_freq.data_x, heat_freq.data_y, color="crimson", marker="*", s=85, label="Observed data")
    ax.set_title("Heat model: frequentist vs Bayesian intervals")
    ax.set_xlabel("Position x (cm)")
    ax.set_ylabel("Temperature (C)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, ncol=2)

    ax = axes[1]
    ax.fill_between(
        sir_freq.t,
        sir_freq.prediction_lower,
        sir_freq.prediction_upper,
        color="0.7",
        alpha=0.35,
        label="Frequentist prediction",
    )
    ax.fill_between(
        sir_freq.t,
        sir_freq.confidence_lower,
        sir_freq.confidence_upper,
        color="tab:orange",
        alpha=0.25,
        label="Frequentist confidence",
    )
    ax.plot(sir_freq.t, sir_freq.mean, color="black", lw=1.8, label="Mean infected")
    ax.plot(sir_bayes.t_data, sir_bayes.credible_lower, "--", color="tab:blue", lw=1.6, label="Bayesian credible")
    ax.plot(sir_bayes.t_data, sir_bayes.credible_upper, "--", color="tab:blue", lw=1.6)
    ax.plot(sir_bayes.t_data, sir_bayes.predictive_lower, ":", color="tab:green", lw=2.0, label="Bayesian prediction")
    ax.plot(sir_bayes.t_data, sir_bayes.predictive_upper, ":", color="tab:green", lw=2.0)
    ax.scatter(sir_freq.t, sir_freq.infected_data, color="crimson", marker="*", s=65, label="Observed data")
    ax.set_title("SIR model: frequentist vs Bayesian intervals")
    ax.set_xlabel("Time")
    ax.set_ylabel("Infected population")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, ncol=2)

    fig.tight_layout()
    fig.savefig(DATA_DIR / "problem4_frequentist_vs_bayesian.png", dpi=180)
    plt.close(fig)

    return summary


def main() -> None:
    """Run the frequentist-versus-Bayesian comparison and print key findings."""

    summary = run_problem4()
    print("Problem 4 comparison complete.")
    print("Interpretation:")
    print("- Frequentist confidence bands quantify uncertainty in the mean response under repeated sampling.")
    print("- Bayesian credible bands quantify posterior uncertainty for the latent response given the data.")
    print("- Prediction bands include observation noise, so they are wider and target future observations.")
    print(
        f"Heat coverage (freq/bayes prediction): {summary['heat']['frequentist_prediction_coverage']:.3f} / "
        f"{summary['heat']['bayesian_prediction_coverage']:.3f}"
    )
    print(
        f"SIR coverage (freq/bayes prediction): {summary['sir']['frequentist_prediction_coverage']:.3f} / "
        f"{summary['sir']['bayesian_prediction_coverage']:.3f}"
    )
    print(
        f"Heat mean width (freq conf / bayes cred): {summary['heat']['frequentist_confidence_mean_width']:.3f} / "
        f"{summary['heat']['bayesian_credible_mean_width']:.3f}"
    )
    print(
        f"SIR mean width (freq conf / bayes cred): {summary['sir']['frequentist_confidence_mean_width']:.3f} / "
        f"{summary['sir']['bayesian_credible_mean_width']:.3f}"
    )
    print(f"Saved plot in: {DATA_DIR / 'problem4_frequentist_vs_bayesian.png'}")


if __name__ == "__main__":
    main()
