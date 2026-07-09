from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy

from utils.BrukerMRI import ReadParamFile
from utils.utils import PaperDataPath, getData

DATASETS = [
    (
        PaperDataPath("before_adjustments_response_b0_preemphasis", 2),
        "Before Adjustments",
    ),
    (
        PaperDataPath("after_adjustments_response", 99),
        "After adjustments",
    ),
]


def computeResponse(scanPath, outRes=2e-6):
    method = ReadParamFile(scanPath / "method")
    polynomials, testShapes, _ = getData(
        scanPath,
        outRes=outRes,
        sliceIndices=None,
        gradientFirst=False,
        frequencyFilter=150e3,
    )

    polynomials = polynomials[-1, ...]
    grads = np.gradient(polynomials, outRes, axis=1)[0]

    fLen = int(testShapes.size / 2) + 1 if testShapes.size % 2 == 0 else int((testShapes.size + 1) / 2)
    f = np.linspace(0, 1 / outRes - 1 / outRes / testShapes.size, testShapes.size)[:fLen]

    x = np.expand_dims(np.fft.rfft(testShapes, axis=0), -1)
    y = np.fft.rfft(grads, axis=0)

    eps = 1e-6 * np.max(np.abs(x) ** 2)
    transfer = y * np.conj(x) / (np.abs(x) ** 2 + eps)

    transfer[:2, :] = np.exp(-np.angle(transfer[:2, :]) * 1j)
    transfer[0, :] = 1

    transferMask = np.abs(np.fft.rfft(np.expand_dims(testShapes, -1), axis=0))
    transferMask = transferMask / np.max(transferMask)
    transferMask = ((transferMask > 0.01) | (np.expand_dims(f, -1) < 1e2)) & (np.expand_dims(f, -1) < method["ChirpFmax"] * 1e3)
    transfer = transfer * transferMask

    impResp = np.fft.fftshift(np.fft.irfft(transfer, axis=0), axes=0)
    sizeResp = impResp.shape[0]
    impResp[: int(sizeResp * 0.4), :] = 0
    impResp[-int(sizeResp * 0.4) :, :] = 0

    transClean = np.fft.rfft(np.fft.fftshift(impResp, axes=0), axis=0)

    fstop = 250
    fstopInd = np.sum(f < fstop)
    axisLabels = ["Z", "X", "Y"]

    fRange = f[(f < 4000) & (f > 200)]
    for axis in range(3):
        fitRange = np.angle(transClean[:, axis])[(f < 4000) & (f > 200)]
        p = np.polyfit(fRange, fitRange, 1)
        transClean[:fstopInd, axis] = np.exp(1j * np.polyval(p, f[:fstopInd]))

    smoothPar = 1
    angleClean = np.unwrap(np.angle(transfer), axis=0) * np.expand_dims((f < 15e3), -1)
    for axis in range(3):
        smoothPars = scipy.interpolate.splrep(f, angleClean[:, axis], s=smoothPar)
        angleClean[:, axis] = scipy.interpolate.splev(f, smoothPars)

    fstop = 1000
    fstopInd = np.sum(f < fstop)
    transCleanMag = np.mean(np.abs(transClean[1:fstopInd, :]), keepdims=True, axis=0)
    transClean = transClean / transCleanMag
    transClean[0, :] = 1

    sortIdx = np.argsort(axisLabels)
    axisLabels = np.array(axisLabels)[sortIdx].tolist()
    transClean = transClean[:, sortIdx]

    return f, transClean, axisLabels


def styleAxis(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(width=0.5)
    ax.yaxis.labelpad = 0.8
    ax.grid(True, color="0.9", linewidth=0.6)


def makeFigure(responses, outPath="paper2026/ResponseComparison.pdf"):
    axisColors = {"X": "green", "Y": "red", "Z": "blue"}

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 6,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "lines.linewidth": 0.8,
            "figure.dpi": 300,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.35), constrained_layout=True, sharex="col")
    fig.set_constrained_layout_pads(wspace=0.08, hspace=0.05)

    for row, (label, f, transfer, axisLabels) in enumerate(responses):
        for axisInd, axisLabel in enumerate(axisLabels):
            color = axisColors[axisLabel]
            axes[row, 0].plot(
                f * 1e-3,
                np.abs(transfer[:, axisInd]),
                color=color,
                label=axisLabel,
            )
            axes[row, 1].plot(
                f * 1e-3,
                np.rad2deg(np.angle(transfer[:, axisInd])),
                color=color,
                label=axisLabel,
            )

        axes[row, 0].set_ylabel(f"{label}\nAmplitude (-)")
        axes[row, 1].set_ylabel("Phase (deg)")

        legend = axes[row, 1].legend(frameon=True, ncol=1, loc="lower right", framealpha=1)
        legend.get_frame().set_linewidth(0.5)

    axes[0, 0].set_title("Amplitude Response")
    axes[0, 1].set_title("Phase Response")

    for row in range(2):
        axes[row, 0].set_xlim(0, 30)
        axes[row, 0].set_ylim(0, 1.08)
        axes[row, 1].set_xlim(0, 30)
        axes[row, 1].set_ylim(-110, 50)

    axes[1, 0].set_xlabel("Frequency (kHz)")
    axes[1, 1].set_xlabel("Frequency (kHz)")

    for ax in axes.flat:
        styleAxis(ax)

    outPath = Path(outPath)
    outPath.parent.mkdir(exist_ok=True)
    fig.savefig(outPath, bbox_inches="tight")
    fig.savefig(outPath.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def run():
    responses = []
    for scanPath, label in DATASETS:
        f, transfer, axisLabels = computeResponse(scanPath)
        responses.append((label, f, transfer, axisLabels))

    makeFigure(responses)


if __name__ == "__main__":
    run()
