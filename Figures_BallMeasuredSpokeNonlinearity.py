from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.BrukerMRI import ReadParamFile


STUDY = Path("/mnt/md1/nmr-bruker/PV-360.3.7/vitous/20260817_091028_Test_ballTraj_1_2")
SCAN = 13
MINIMUM_X_SCALING = 0.1
DISPLAY_SCALINGS = np.array(
    [-1.0, -0.8, -0.6, -0.4, -0.2, -0.1, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
)
SMOOTHING_SAMPLES = 7


def smooth(signal, samples=SMOOTHING_SAMPLES):
    kernel = np.ones(samples) / samples
    padding = samples // 2
    return np.convolve(np.pad(signal, (padding, padding), mode="edge"), kernel, mode="valid")


def zeroCrossingSample(signal, expectedSample, searchHalfWidth=40):
    searchStart = expectedSample - searchHalfWidth
    searchStop = expectedSample + searchHalfWidth
    crossings = np.flatnonzero(
        signal[searchStart:searchStop] * signal[searchStart + 1 : searchStop + 1] <= 0
    ) + searchStart
    if crossings.size == 0:
        return np.nan
    crossing = crossings[np.argmin(np.abs(crossings - expectedSample))]
    delta = signal[crossing + 1] - signal[crossing]
    return crossing - signal[crossing] / delta if delta != 0 else np.nan


scanPath = STUDY / str(SCAN)
method = ReadParamFile(scanPath / "method")
xScaling = np.asarray(method["RadRead_GradAmpR"], dtype=float)
nSpokes = xScaling.size
nSamples = int(method["PVM_TrajSamples"])
measuredX = np.asarray(method["PVM_TrajKx"], dtype=float).reshape(nSpokes, nSamples).T
binarySamples = (scanPath / "traj").stat().st_size // (8 * 2 * nSpokes)
expectedCenterSample = nSamples - binarySamples + 258

displayIndices = [int(np.argmin(np.abs(xScaling - target))) for target in DISPLAY_SCALINGS]
colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(displayIndices)))

normalizedGradients = []
normalizedTrajectories = []
for spokeIndex in displayIndices:
    scale = xScaling[spokeIndex]
    trajectory = (measuredX[:, spokeIndex] - measuredX[0, spokeIndex]) / scale
    gradient = smooth(np.gradient(measuredX[:, spokeIndex])) / scale
    normalizedTrajectories.append(trajectory)
    normalizedGradients.append(gradient)

# As in the existing Sup_5 analysis, use one common display scale from the
# maximum positive-amplitude response rather than peak-normalizing each curve.
referenceIndex = int(np.argmax([xScaling[index] for index in displayIndices]))
gradientScale = np.max(np.abs(normalizedGradients[referenceIndex]))
trajectoryScale = np.max(np.abs(normalizedTrajectories[referenceIndex]))
normalizedGradients = [gradient / gradientScale for gradient in normalizedGradients]
normalizedTrajectories = [trajectory / trajectoryScale for trajectory in normalizedTrajectories]
trajectoryReference = normalizedTrajectories[referenceIndex]
trajectoryDeviations = [trajectory - trajectoryReference for trajectory in normalizedTrajectories]

gradientDwellTimeMs = float(method["GradRes"])
acquisitionDwellTimeMs = float(method["PVM_TrajDwAcq"])
nominalGradientRaster = np.asarray(method["RadRead_ReadGradShapeTraj"], dtype=float)
nominalGradientTimeMs = np.arange(nominalGradientRaster.size) * gradientDwellTimeMs
timeMs = np.arange(nSamples) * acquisitionDwellTimeMs
nominalGradient = np.interp(timeMs, nominalGradientTimeMs, nominalGradientRaster)
nominalGradient /= np.max(np.abs(nominalGradient))
nominalTrajectory = np.zeros_like(nominalGradient)
nominalTrajectory[1:] = np.cumsum(
    0.5 * (nominalGradient[:-1] + nominalGradient[1:]) * acquisitionDwellTimeMs
)
nominalCenterSample = zeroCrossingSample(nominalTrajectory, expectedCenterSample)
if not np.isfinite(nominalCenterSample):
    raise RuntimeError("Could not find the nominal post-dephasing k-space-center crossing")

validCenters = np.abs(xScaling) >= MINIMUM_X_SCALING
kSpaceCenters = np.array(
    [
        zeroCrossingSample(measuredX[:, spokeIndex], expectedCenterSample)
        for spokeIndex in range(nSpokes)
    ]
)
validCenters &= np.isfinite(kSpaceCenters)
xGradientAmplitude = xScaling * float(method["RadRead_ReadGrad"])

fig, axes = plt.subplots(3, 1, figsize=(6.2, 10.2))

for color, spokeIndex, gradient in zip(colors, displayIndices, normalizedGradients):
    axes[0].plot(
        timeMs,
        gradient,
        color=color,
        linewidth=1.6,
        label=f"{100 * xScaling[spokeIndex]:.1f}%",
    )
axes[0].plot(
    timeMs,
    nominalGradient,
    color="black",
    linestyle="--",
    linewidth=1.8,
    label="Nominal",
)
axes[0].set_ylabel("Normalized gradient (-)")
axes[0].set_title("Normalized nominal and measured gradient shapes")
axes[0].legend(
    title="X scaling",
    fontsize=7.5,
    title_fontsize=8,
    ncol=2,
    frameon=False,
    loc="lower right",
)

gradientZeroCrossings = np.flatnonzero(
    (nominalGradient[:-1] < 0) & (nominalGradient[1:] >= 0)
)
if gradientZeroCrossings.size == 0:
    raise RuntimeError("Could not find the nominal gradient zero crossing")
gradientZeroIndex = gradientZeroCrossings[0]
nominalGradientZeroSample = gradientZeroIndex - nominalGradient[gradientZeroIndex] / (
    nominalGradient[gradientZeroIndex + 1] - nominalGradient[gradientZeroIndex]
)
nominalGradientZeroTimeMs = nominalGradientZeroSample * acquisitionDwellTimeMs

gradientInset = axes[0].inset_axes([0.32, 0.14, 0.27, 0.29])
for color, gradient in zip(colors, normalizedGradients):
    gradientInset.plot(timeMs, gradient, color=color, linewidth=1.0)
gradientInset.plot(
    timeMs,
    nominalGradient,
    color="black",
    linestyle="--",
    linewidth=1.3,
)
gradientInset.axhline(0, color="0.4", linewidth=0.7)
gradientInset.axvline(
    nominalGradientZeroTimeMs, color="tab:red", linestyle="--", linewidth=0.9
)
gradientInset.set_xlim(nominalGradientZeroTimeMs - 0.04, nominalGradientZeroTimeMs + 0.04)
gradientInset.set_ylim(-0.35, 0.35)
gradientInset.set_title("Gradient zero-crossing zoom", fontsize=7.5)
gradientInset.set_xlabel("Time (ms)", fontsize=6.5, labelpad=1)
gradientInset.set_ylabel("Normalized gradient (-)", fontsize=6.5, labelpad=1)
gradientInset.tick_params(labelsize=7)
gradientInset.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

for color, spokeIndex, deviation in zip(colors, displayIndices, trajectoryDeviations):
    if spokeIndex == displayIndices[referenceIndex]:
        continue
    axes[1].plot(
        timeMs,
        deviation,
        color=color,
        linewidth=1.6,
        label=f"{100 * xScaling[spokeIndex]:.1f}%",
    )
axes[1].axhline(0, color="black", linewidth=0.8, alpha=0.55)
axes[1].set_ylabel("Normalized trajectory deviation\nfrom +100% reference (-)")
axes[1].set_title("Normalized trajectory deviations")

axes[2].scatter(
    xGradientAmplitude[validCenters],
    kSpaceCenters[validCenters],
    s=9,
    color="tab:blue",
    alpha=0.28,
    edgecolors="none",
    label="Measured spokes",
)
axes[2].axhline(
    nominalCenterSample,
    color="tab:red",
    linestyle="--",
    linewidth=1.2,
    label="Expected k-space center based on nominal-waveform ",
)
axes[2].set_xlabel("Commanded gradient amplitude (%)")
axes[2].set_ylabel("k-space center\n(readout sample)")
axes[2].set_title("K-space center versus gradient amplitude")
axes[2].legend(fontsize=7.5, frameon=False)

for panelLabel, axis in zip(("a", "b", "c"), axes):
    axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    axis.tick_params(labelsize=9)
    axis.text(0.0, 1.03, panelLabel, transform=axis.transAxes, fontsize=12, fontweight="bold")

axes[0].set_xlabel("Time (ms)")
axes[1].set_xlabel("Time (ms)")
fig.subplots_adjust(left=0.17, right=0.98, bottom=0.07, top=0.96, hspace=0.42)

outputDirectory = Path("paper2026")
outputDirectory.mkdir(exist_ok=True)
fig.savefig(outputDirectory / "Sup_5.png", dpi=600, bbox_inches="tight")
fig.savefig(outputDirectory / "Sup_5.pdf", bbox_inches="tight")
