import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from utils.BrukerMRI import ReadParamFile
from utils.utils import getData, parse_args


def fitPreemphasisEddyCurrents(dataPath, scan, outRes, nExp=1):
    scanPath = dataPath / scan

    polynomials, testShape, _ = getData(scanPath, outRes=outRes)
    measuredGrad = np.gradient(polynomials[-1, ...], outRes, axis=1)[0]

    method = ReadParamFile(scanPath / "method")
    gradCalConst = method["PVM_GradCalConst"]
    shapeAmplitude = method.get("ChirpAmplitude", method.get("TestShapeAmplitude", None))

    measuredGrad = measuredGrad / gradCalConst / 2 / np.pi / shapeAmplitude * 1e2
    testShape = testShape / gradCalConst / 2 / np.pi / shapeAmplitude * 1e2

    padSize = 64000
    tPad = np.linspace(0, padSize * outRes - outRes, padSize)
    diffShape = np.gradient(testShape)

    def eddyModel(x):
        spectralShape = np.fft.rfft(diffShape, padSize)
        for ind in range(x.size // 2):
            filt = x[2 * ind] * np.exp(-1 / x[2 * ind + 1] * tPad)
            spectralShape *= np.fft.rfft(filt)

        return np.fft.irfft(spectralShape)[: diffShape.size]

    tOrig = np.arange(measuredGrad.shape[0]) * outRes

    def fittedShape(x):
        return np.interp(tOrig, tOrig + x[-1], testShape - eddyModel(x))

    def residualForFit(x, axis):
        return fittedShape(x) - measuredGrad[:, axis]

    x0 = []
    bounds = ([], [])

    for exp in range(nExp):
        x0.extend((0.5, 50e-6 * (exp + 1)))
        bounds[0].extend((-np.inf, 10e-6))
        bounds[1].extend((np.inf, 1e-3))

    x0.append(0)
    bounds[0].append(-20e-6)
    bounds[1].append(20e-6)

    fits = []
    fittedGrads = []
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
        fittedGrad = fittedShape(res.x)
        fits.append(res.x)
        fittedGrads.append(fittedGrad)
        residuals.append(measuredGrad[:, axis] - fittedGrad)

    return testShape, measuredGrad, np.asarray(fittedGrads).T, np.asarray(residuals).T, fits


def findZeroCrossingWindow(testShape, timeMs, windowFraction=0.015):
    signs = np.signbit(testShape)
    crossingInds = np.flatnonzero(signs[:-1] != signs[1:])

    if crossingInds.size:
        midpoint = testShape.size // 2
        crossingInd = crossingInds[np.argmin(np.abs(crossingInds - midpoint))]
    else:
        crossingInd = int(np.argmin(np.abs(testShape)))

    halfWindow = max(4, int(testShape.size * windowFraction / 2))
    start = max(0, crossingInd - halfWindow)
    stop = min(testShape.size - 1, crossingInd + halfWindow)

    return timeMs[start], timeMs[stop], start, stop


def styleAxis(ax, subplotAspect=0.62):
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(width=0.5)
    ax.yaxis.labelpad = 0.8
    ax.grid(True, color="0.9", linewidth=0.6)


def addZeroCrossingInset(ax, timeMs, curves, colors, linestyles, xlim, indexWindow):
    inset = ax.inset_axes([0.25, 0.55, 0.38, 0.38])
    inset.set_in_layout(False)
    start, stop = indexWindow

    for curve, color, linestyle in zip(curves, colors, linestyles, strict=True):
        inset.plot(timeMs, curve, color=color, linestyle=linestyle, linewidth=0.6)

    inset.axhline(0, color="0.35", linewidth=0.3)
    inset.set_xlim(*xlim)

    yValues = np.concatenate([curve[start : stop + 1] for curve in curves])
    yMargin = 0.15 * max(np.ptp(yValues), 1e-9)
    inset.set_ylim(np.min(yValues) - yMargin, np.max(yValues) + yMargin)
    inset.tick_params(width=0.3, labelsize=5, pad=1)

    for spine in inset.spines.values():
        spine.set_linewidth(0.3)


def makeFigure(testShape, measuredGrad, fittedGrad, residuals, outRes, outPath, figSize=(7, 3)):
    axisOrder = [(1, "X", "green"), (2, "Y", "red"), (0, "Z", "blue")]
    timeMs = np.arange(testShape.size) * outRes * 1e3

    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 6,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "lines.linewidth": 0.7,
            "figure.dpi": 300,
        }
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=figSize,
        constrained_layout=True,
        gridspec_kw={"hspace": 0.02, "wspace": 0.08},
    )
    fig.set_constrained_layout_pads(wspace=0.08, hspace=0.015)

    zoomX0, zoomX1, zoomStart, zoomStop = findZeroCrossingWindow(testShape, timeMs)
    insetSpecs = []

    for col, (axis, label, color) in enumerate(axisOrder):
        topAx = axes[0, col]
        residualAx = axes[1, col]

        topAx.plot(timeMs, testShape, color="black", label="Test")
        topAx.plot(timeMs, measuredGrad[:, axis], color=color, linestyle=":", linewidth=0.9, label="Measured")
        topAx.plot(timeMs, fittedGrad[:, axis], color=color, label="Fit")
        topAx.set_title(f"{label} axis")
        topAx.tick_params(axis="x", labelbottom=False)
        topAx.set_ylabel("Gradient (a.u.)")
        topAx.set_xlim(0.2, 0.5)

        insetSpecs.append(
            (
                topAx,
                [testShape, measuredGrad[:, axis], fittedGrad[:, axis]],
                ["black", color, color],
                ["-", ":", "-"],
            )
        )

        residualAx.plot(timeMs, residuals[:, axis], color=color)
        residualAx.axhline(0, color="0.35", linewidth=0.45)
        residualAx.set_xlabel("Time (ms)")
        residualAx.set_ylabel("Residual (a.u.)")
        residualAx.set_xlim(0.2, 0.5)

        styleAxis(topAx)
        styleAxis(residualAx)

    legend = axes[0, 0].legend(frameon=True, ncol=1, loc="lower right", framealpha=1, fontsize=6)
    legend = axes[0, 1].legend(frameon=True, ncol=1, loc="lower right", framealpha=1)
    legend = axes[0, 2].legend(frameon=True, ncol=1, loc="lower right", framealpha=1)
    legend.get_frame().set_linewidth(0.5)

    os.makedirs(os.path.dirname(outPath), exist_ok=True)
    for topAx, curves, insetColors, linestyles in insetSpecs:
        addZeroCrossingInset(
            topAx,
            timeMs,
            curves,
            insetColors,
            linestyles,
            (zoomX0, zoomX1),
            (zoomStart, zoomStop),
        )
    fig.savefig(outPath, bbox_inches="tight")
    fig.savefig(outPath.replace(".pdf", ".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def run(dataPath, scan, outRes):
    testShape, measuredGrad, fittedGrad, residuals, fits = fitPreemphasisEddyCurrents(dataPath, scan, outRes)

    for axisLabel, fit in zip(["Z", "X", "Y"], fits, strict=True):
        for component in range((fit.size - 1) // 2):
            print(f"Axis {axisLabel} Component {component} A = {fit[component * 2] * 1e2:.2f} %, b = {fit[component * 2 + 1] * 1e6:.2f} us.")
        print(f"Axis {axisLabel} Delay {fit[-1] * 1e6:.2f} us.")

    makeFigure(
        testShape,
        measuredGrad,
        fittedGrad,
        residuals,
        outRes,
        "paper2026/PreemphasisEddyCurrents.pdf",
    )


if __name__ == "__main__":
    args = parse_args("Create preemphasis eddy-current publication figure", "preemp")
    run(args[0], args[1], args[2])
