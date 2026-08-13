import csv
import glob
import hashlib
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


AXES = ("X", "Y", "Z")
AXIS_DISPLAY_POLARITY = {"X": -1.0, "Y": 1.0, "Z": -1.0}
REQUESTED_AMPLITUDES = np.array([0.10, 0.15, 0.20, 0.35, 0.50, 0.80])
OUTPUT_RESOLUTION_SECONDS = 2e-6


def loadAxisDataset(axisName):
    matches = [path for path in glob.glob(f"datasets/trainingSet_{axisName}_*.pt") if "_B0_" not in path]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one cached {axisName}-axis training dataset, found {matches}")
    return torch.load(matches[0], map_location="cpu", weights_only=False)


def activeInputRange(inputWaveform):
    active = np.flatnonzero(np.abs(inputWaveform) > 1e-8)
    if active.size == 0:
        raise ValueError("Empty input waveform")
    return int(active[0]), int(active[-1] + 1)


def waveformFamilyHash(inputWaveform):
    start, stop = activeInputRange(inputWaveform)
    waveform = np.round(inputWaveform[start:stop], 6)
    return hashlib.sha1(waveform.tobytes()).hexdigest()


def normalize(signal):
    scale = np.max(np.abs(signal))
    return signal / scale if scale > 0 else signal


def estimateRepetitionSamples(inputWaveform):
    """Find the strongest non-zero-lag autocorrelation peak."""
    centered = inputWaveform - np.mean(inputWaveform)
    correlation = np.correlate(centered, centered, mode="full")[centered.size - 1 :]
    minimumLag = max(20, centered.size // 10)
    maximumLag = centered.size // 2
    search = correlation[minimumLag:maximumLag]
    localPeaks = np.flatnonzero((search[1:-1] > search[:-2]) & (search[1:-1] >= search[2:])) + 1
    if localPeaks.size == 0:
        raise RuntimeError("Could not detect a READOUT repetition")
    return minimumLag + int(localPeaks[np.argmax(search[localPeaks])])


datasets = {axisName: loadAxisDataset(axisName) for axisName in AXES}
referenceDataset = datasets["X"]

# Group repeated READOUT measurements by commanded shape and select a family
# that spans the requested low-to-high positive amplitudes.
families = defaultdict(list)
for sampleIndex, label in enumerate(referenceDataset["labels"]):
    if label != "READOUT":
        continue
    inputWaveform = referenceDataset["xData"][sampleIndex, 0].numpy()
    family = waveformFamilyHash(inputWaveform)
    amplitude = float(referenceDataset["xData"][sampleIndex, 1, 0])
    families[family].append((sampleIndex, amplitude))

eligibleFamilies = [
    (family, samples)
    for family, samples in families.items()
    if max(amplitude for _, amplitude in samples) >= REQUESTED_AMPLITUDES[-1] - 1e-3
]
if not eligibleFamilies:
    raise RuntimeError("No READOUT training family spans the requested amplitudes")

# Prefer the longest waveform among eligible families.
selectedFamily, familySamples = max(
    eligibleFamilies,
    key=lambda item: activeInputRange(referenceDataset["xData"][item[1][0][0], 0].numpy())[1]
    - activeInputRange(referenceDataset["xData"][item[1][0][0], 0].numpy())[0],
)
positiveSamples = [(index, amplitude) for index, amplitude in familySamples if amplitude > 0]
selectedSamples = [min(positiveSamples, key=lambda item: abs(item[1] - requested)) for requested in REQUESTED_AMPLITUDES]
familyExample = referenceDataset["xData"][selectedSamples[0][0], 0].numpy()
familyStart, familyStop = activeInputRange(familyExample)
# The autocorrelation period spans the alternating positive/negative pair used
# by this training sequence. Focus on the first third of the first readout.
repetitionSamples = estimateRepetitionSamples(familyExample[familyStart:familyStop]) // 6

fig, axes = plt.subplots(3, 3, figsize=(14, 8), sharex=True)
colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(selectedSamples)))
metrics = []
axisTrajectoryDifferences = []
plotData = {}

for row, axisName in enumerate(AXES):
    dataset = datasets[axisName]
    normalizedWaveforms = []
    normalizedTrajectories = []

    for color, (sampleIndex, requestedAmplitude) in zip(colors, selectedSamples):
        inputWaveform = dataset["xData"][sampleIndex, 0].numpy()
        start, stop = activeInputRange(inputWaveform)

        # The cached target is delayed by the model alignment offset. Detect its
        # valid region from the trajectory change and use the common waveform
        # duration, excluding padded samples at both ends.
        measuredWaveform = dataset["yData"][sampleIndex, 0].numpy()
        measuredTrajectory = dataset["yTrajData"][sampleIndex, 0].numpy()
        targetStart = start + int(dataset["cachePayload"]["cacheParams"]["shiftByNSamples"])
        targetStop = min(targetStart + repetitionSamples, stop)
        measuredWaveform = measuredWaveform[targetStart:targetStop]
        measuredTrajectory = measuredTrajectory[targetStart:targetStop]
        measuredTrajectory = measuredTrajectory - measuredTrajectory[0]
        measuredWaveform = measuredWaveform * AXIS_DISPLAY_POLARITY[axisName]
        measuredTrajectory = measuredTrajectory * AXIS_DISPLAY_POLARITY[axisName]

        waveformNormalized = normalize(measuredWaveform)
        trajectoryNormalized = normalize(measuredTrajectory)
        normalizedWaveforms.append(waveformNormalized)
        normalizedTrajectories.append(trajectoryNormalized)

        timeMs = np.arange(waveformNormalized.size) * OUTPUT_RESOLUTION_SECONDS * 1e3
        label = f"{requestedAmplitude * 100:.1f}% requested"
        axes[row, 0].plot(timeMs, waveformNormalized, color=color, linewidth=1.8, label=label)
        axes[row, 1].plot(timeMs, trajectoryNormalized, color=color, linewidth=1.8, label=label)

    baselineWaveform = normalizedWaveforms[0]
    baselineTrajectory = normalizedTrajectories[0]
    highestAmplitudeTrajectory = normalizedTrajectories[-1]
    trajectoryDifferences = [trajectory - highestAmplitudeTrajectory for trajectory in normalizedTrajectories]
    axisTrajectoryDifferences.extend(trajectoryDifferences)
    plotData[axisName] = {
        "timeMs": timeMs,
        "waveforms": normalizedWaveforms,
        "trajectories": normalizedTrajectories,
        "differences": trajectoryDifferences,
    }
    for color, difference, (_, requestedAmplitude) in zip(colors, trajectoryDifferences, selectedSamples):
        label = f"{requestedAmplitude * 100:.1f}% requested"
        axes[row, 2].plot(timeMs, difference, color=color, linewidth=1.8, label=label)

    for selectedIndex, (_, requestedAmplitude) in enumerate(selectedSamples):
        waveformDifference = normalizedWaveforms[selectedIndex] - baselineWaveform
        trajectoryDifference = normalizedTrajectories[selectedIndex] - baselineTrajectory
        metrics.append(
            {
                "axis": axisName,
                "requested_amplitude_percent": requestedAmplitude * 100,
                "waveform_rms_difference_from_low_amplitude": np.sqrt(np.mean(waveformDifference**2)),
                "trajectory_rms_difference_from_low_amplitude": np.sqrt(np.mean(trajectoryDifference**2)),
            }
        )

    axes[row, 0].set_ylabel(f"{axisName} axis\nNormalized gradient", fontsize=12)
    axes[row, 0].grid(True, linestyle="--", alpha=0.35)
    axes[row, 1].grid(True, linestyle="--", alpha=0.35)
    axes[row, 2].grid(True, linestyle="--", alpha=0.35)
    axes[row, 2].axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axes[row, 0].set_ylim(-1.05, 1.05)
    axes[row, 1].set_ylim(-1.05, 1.05)

axes[0, 0].set_title("Measured READOUT waveform", fontsize=14)
axes[0, 1].set_title("Measured trajectory", fontsize=14)
axes[0, 2].set_title("Trajectory difference from 80%", fontsize=14)
axes[-1, 0].set_xlabel("Time (ms)", fontsize=12)
axes[-1, 1].set_xlabel("Time (ms)", fontsize=12)
axes[-1, 2].set_xlabel("Time (ms)", fontsize=12)
axes[0, 1].legend(fontsize=10, loc="best")

differenceLimit = max(np.max(np.abs(difference)) for difference in axisTrajectoryDifferences)
if differenceLimit == 0:
    differenceLimit = 1.0
for row in range(len(AXES)):
    axes[row, 2].set_ylim(-1.05 * differenceLimit, 1.05 * differenceLimit)

fig.subplots_adjust(left=0.09, right=0.99, bottom=0.08, top=0.94, wspace=0.20, hspace=0.08)

outputDirectory = Path("paper2026")
outputDirectory.mkdir(exist_ok=True)
fig.savefig(outputDirectory / "TrainingReadoutAmplitudeNonlinearity.png", dpi=600, bbox_inches="tight")
fig.savefig(outputDirectory / "TrainingReadoutAmplitudeNonlinearity.pdf", bbox_inches="tight")

with open(outputDirectory / "TrainingReadoutAmplitudeNonlinearity.csv", "w", newline="", encoding="utf-8") as outputFile:
    writer = csv.DictWriter(outputFile, fieldnames=metrics[0].keys())
    writer.writeheader()
    writer.writerows(metrics)


def responseCrop(data):
    """Remove the leading near-zero interval while retaining onset context."""
    envelope = np.max(np.abs(np.stack(data["waveforms"])), axis=0)
    active = np.flatnonzero(envelope >= 0.08 * np.max(envelope))
    start = max(0, int(active[0]) - 3) if active.size else 0
    return slice(start, None)


def stylePublicationAxis(axis, *, ylabel, ylim=None):
    axis.set_xlabel("Time (ms)", fontsize=11)
    axis.set_ylabel(ylabel, fontsize=11)
    axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    axis.tick_params(labelsize=10)
    if ylim is not None:
        axis.set_ylim(*ylim)


imagesDirectory = Path("images")
imagesDirectory.mkdir(exist_ok=True)
representativeAxis = "X"
representativeData = plotData[representativeAxis]
crop = responseCrop(representativeData)
croppedTime = representativeData["timeMs"][crop]
croppedTime = croppedTime - croppedTime[0]

panelDefinitions = (
    ("waveforms", "Normalized measured gradient", "Waveform", (-1.05, 1.05)),
    ("trajectories", "Normalized measured trajectory", "Trajectory", (-1.05, 1.05)),
    ("differences", "Trajectory difference from 80%", "TrajectoryDifference", None),
)

# Standalone vector/raster panels for flexible manuscript assembly.
for dataKey, ylabel, fileStem, ylim in panelDefinitions:
    panelFig, panelAxis = plt.subplots(figsize=(4.6, 3.3))
    for color, curve, (_, requestedAmplitude) in zip(colors, representativeData[dataKey], selectedSamples):
        panelAxis.plot(
            croppedTime,
            curve[crop],
            color=color,
            linewidth=2.0,
            label=f"{requestedAmplitude * 100:.1f}%",
        )
    if dataKey == "differences":
        panelAxis.axhline(0, color="black", linewidth=0.8, alpha=0.55)
    stylePublicationAxis(panelAxis, ylabel=ylabel, ylim=ylim)
    panelAxis.legend(title="Requested amplitude", fontsize=8.5, title_fontsize=9, frameon=False, ncol=2)
    panelFig.tight_layout()
    panelFig.savefig(imagesDirectory / f"TrainingReadout{fileStem}.pdf", bbox_inches="tight")
    panelFig.savefig(imagesDirectory / f"TrainingReadout{fileStem}.png", dpi=600, bbox_inches="tight")
    plt.close(panelFig)

# Compact publication figure: one representative example of each panel type.
publicationFig, publicationAxes = plt.subplots(1, 3, figsize=(12.2, 3.45), sharex=True)
for panelAxis, (dataKey, ylabel, _, ylim), panelLabel in zip(publicationAxes, panelDefinitions, ("a", "b", "c")):
    for color, curve, (_, requestedAmplitude) in zip(colors, representativeData[dataKey], selectedSamples):
        panelAxis.plot(
            croppedTime,
            curve[crop],
            color=color,
            linewidth=1.9,
            label=f"{requestedAmplitude * 100:.1f}%",
        )
    if dataKey == "differences":
        panelAxis.axhline(0, color="black", linewidth=0.8, alpha=0.55)
    stylePublicationAxis(panelAxis, ylabel=ylabel, ylim=ylim)
    panelAxis.text(-0.14, 1.03, panelLabel, transform=panelAxis.transAxes, fontsize=13, fontweight="bold")

handles, labels = publicationAxes[0].get_legend_handles_labels()
publicationFig.legend(
    handles,
    labels,
    title="Requested amplitude",
    loc="upper center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=len(selectedSamples),
    frameon=False,
    fontsize=9,
    title_fontsize=9,
)
publicationFig.subplots_adjust(left=0.07, right=0.99, bottom=0.18, top=0.80, wspace=0.31)
publicationFig.savefig(imagesDirectory / "TrainingReadoutAmplitudeNonlinearity.pdf", bbox_inches="tight")
publicationFig.savefig(imagesDirectory / "TrainingReadoutAmplitudeNonlinearity.png", dpi=600, bbox_inches="tight")
