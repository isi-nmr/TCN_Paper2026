import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from utils.BrukerMRI import ReadParamFile
from utils.utils import getData, parse_args


def fitB0EddyCurrents(dataPath, scan, outRes, nExp=1):
    scanPath = dataPath / scan

    polynomials, testShape, _ = getData(scanPath, outRes=outRes)
    b0Phase = polynomials[0, 0, ...]
    b0Response = np.gradient(b0Phase, outRes, axis=-2)

    testShape = testShape / np.max(np.abs(testShape))
    diffShape = np.gradient(testShape, outRes)

    padSize = 9200
    tPad = np.linspace(0, padSize * outRes - outRes, padSize)

    def b0Model(x):
        spectralShape = np.fft.rfft(diffShape, padSize)
        for ind in range(x.size // 2):
            filt = x[2 * ind] * np.exp(-1 / x[2 * ind + 1] * tPad)
            spectralShape *= np.fft.rfft(filt)

        return np.fft.irfft(spectralShape)[: diffShape.size]

    def residualForFit(x, axis):
        sampleWeight = np.linspace(1, b0Response[:, axis].size, b0Response[:, axis].size)
        return (b0Model(x) - b0Response[:, axis]) / sampleWeight

    x0 = []
    bounds = ([], [])

    for exp in range(nExp):
        x0.extend((1, 60e-6 * (exp + 1)))
        bounds[0].extend((-np.inf, 10e-6))
        bounds[1].extend((np.inf, 1e-3))

    fits = []
    models = []
    residuals = []

    for axis in range(3):
        res = least_squares(
            residualForFit,
            x0,
            bounds=bounds,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            args=(axis,),
        )
        model = b0Model(res.x)
        fits.append(res.x)
        models.append(model)
        residuals.append(b0Response[:, axis] - model)

    return testShape, b0Response, np.asarray(models).T, np.asarray(residuals).T, fits


def makeFigure(testShape, b0Response, b0Fit, residuals, outRes, outPath, subplotAspect=0.6, subplotSpace=0.08):
    labels = ["Z", "X", "Y"]
    colors = ["blue", "green", "red"]
    timeMs = np.arange(testShape.size) * outRes * 1e3

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "lines.linewidth": 0.7,
            "figure.dpi": 300,
        }
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.85),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1, 1]},
    )
    fig.set_constrained_layout_pads(wspace=subplotSpace)

    axes[0].plot(timeMs, 1e-3 * testShape, color="black")
    axes[0].set_title("Test Gradient")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("G (kHz/mm)")

    for axis, (label, color) in enumerate(zip(labels, colors, strict=True)):
        axes[1].plot(timeMs, 1e-3 * b0Response[:, axis], color=color, linestyle=":", linewidth=0.9, label=f"{label} measured")
        axes[1].plot(timeMs, 1e-3 * b0Fit[:, axis], color=color, label=f"{label} fit")
        axes[2].plot(timeMs, 1e-3 * residuals[:, axis], color=color, label=label)

    axes[1].set_title("Fitted B0 Eddy Currents")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("B0 (kHz)")

    axes[1].set_ylim(-10, 10)

    axes[2].set_title("Residual")
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylabel("B0 (kHz)")
    axes[2].set_ylim(-10, 10)

    for ax in axes:
        ax.set_box_aspect(subplotAspect)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.tick_params(width=0.5)
        ax.yaxis.labelpad = 0.8
        ax.grid(True, color="0.9", linewidth=0.6)
        ax.set_xlim(timeMs[0], timeMs[-1])

    for ax in axes[1:]:
        legend = ax.legend(frameon=True, ncol=1, loc="lower right", framealpha=1)
        legend.get_frame().set_linewidth(0.5)

    os.makedirs(os.path.dirname(outPath), exist_ok=True)
    fig.savefig(outPath, bbox_inches="tight")
    fig.savefig(outPath.replace(".pdf", ".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def run(dataPath, scan, outRes):
    testShape, b0Response, b0Fit, residuals, fits = fitB0EddyCurrents(dataPath, scan, outRes)

    for axisLabel, fit in zip(["Z", "X", "Y"], fits, strict=True):
        for component in range(fit.size // 2):
            print(f"Axis {axisLabel} Component {component} A = {fit[component * 2] * 1e2:.2f} %, b = {fit[component * 2 + 1] * 1e6:.2f} us.")

    method = ReadParamFile(dataPath / scan / "method")

    makeFigure(
        testShape * method["PVM_GradCalConst"],
        b0Response,
        b0Fit,
        residuals,
        outRes,
        "paper2026/B0EddyCurrents.pdf",
    )


if __name__ == "__main__":
    args = parse_args("Create B0 eddy-current publication figure", "B0")
    run(args[0], args[1], args[2])
