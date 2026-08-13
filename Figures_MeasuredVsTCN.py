import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
import torch
from torch.utils.data import random_split

from nn_models.dataset import createTrainingSet
from nn_models.TCN import TCN, TCNFull, TCNFullSkip
from utils.BrukerMRI import *
from utils.utils import LoadTestingShapeTypes, LoadTrainingDataConfig, load_config

os.makedirs("./paper2026", exist_ok=True)


def cumulativeIntegrate(y, dt, axis=-1):
    return np.cumsum(y, axis=axis) * dt


def getValidationIndices(datasetLen, trainFraction=0.85, seed=42):
    trainSize = int(trainFraction * datasetLen)
    valSize = datasetLen - trainSize
    _, valDataset = random_split(
        range(datasetLen),
        [trainSize, valSize],
        generator=torch.Generator().manual_seed(seed),
    )
    return np.asarray(valDataset.indices, dtype=np.int64)


def maskedRmse(pred, ref, weights):
    denom = np.sum(weights)
    if denom <= 0:
        return np.nan
    return np.sqrt(np.sum(((pred - ref) ** 2) * weights) / denom)


def maskedRange(ref, weights):
    valid = weights > 0
    if not np.any(valid):
        return np.nan
    refValid = ref[valid]
    return np.max(refValid) - np.min(refValid)


def maskedNrmse(pred, ref, weights):
    scale = maskedRange(ref, weights)
    if not np.isfinite(scale) or scale <= 0:
        return np.nan
    return maskedRmse(pred, ref, weights) / scale


def perCurveMaskedNrmse(pred, ref, weights):
    return np.asarray([maskedNrmse(pred[[ind]], ref[[ind]], weights[[ind]]) for ind in range(pred.shape[0])])


def delaySignal(signal, nSamples, axis=-1):
    delayed = np.zeros_like(signal)
    if nSamples == 0:
        return signal.copy()

    src = [slice(None)] * signal.ndim
    dst = [slice(None)] * signal.ndim
    if nSamples > 0:
        src[axis] = slice(0, -nSamples)
        dst[axis] = slice(nSamples, None)
    else:
        src[axis] = slice(-nSamples, None)
        dst[axis] = slice(0, nSamples)

    delayed[tuple(dst)] = signal[tuple(src)]
    return delayed


def filterTestingExamples(
    indices,
    xData,
    labels,
    *,
    shapeTypes=None,
    amplitudeMinMagnitude=None,
    amplitudeFraction=None,
    amplitudePerShape=True,
):
    labels = np.asarray(labels)
    selected = np.ones(indices.size, dtype=bool)

    if shapeTypes is not None:
        shapeTypes = set(shapeTypes)
        selected &= np.asarray([label in shapeTypes for label in labels[indices]])

    amplitudes = torch.abs(xData[indices, 1, 0]).detach().cpu().numpy()
    if amplitudeMinMagnitude is not None:
        selected &= amplitudes >= amplitudeMinMagnitude

    if amplitudeFraction is not None:
        if not np.any(selected):
            shapeMsg = "all shapes" if shapeTypes is None else ", ".join(sorted(shapeTypes))
            raise ValueError(f"No held-out examples matched shape filter [{shapeMsg}]")

        ampSelected = np.zeros_like(selected)
        if amplitudePerShape:
            for shapeName in np.unique(labels[indices][selected]):
                shapeMask = (labels[indices] == shapeName) & selected
                maxAmp = np.max(amplitudes[shapeMask])
                ampSelected |= shapeMask & (amplitudes >= amplitudeFraction * maxAmp)
        else:
            maxAmp = np.max(amplitudes[selected])
            ampSelected = amplitudes >= amplitudeFraction * maxAmp

        selected &= ampSelected

    if not np.any(selected):
        shapeMsg = "all shapes" if shapeTypes is None else ", ".join(sorted(shapeTypes))
        raise ValueError(
            f"No held-out examples matched shape filter [{shapeMsg}], "
            f"amplitude_min_magnitude={amplitudeMinMagnitude}, and amplitude_fraction={amplitudeFraction}"
        )

    return indices[selected]


def setViewportYlim(axis, timeMs, series, *, xMin=None, xMax=None, padFraction=0.08, symmetric=False):
    view = np.ones_like(timeMs, dtype=bool)
    if xMin is not None:
        view &= timeMs >= xMin
    if xMax is not None:
        view &= timeMs <= xMax

    values = []
    for item in series:
        arr = np.asarray(item)
        if arr.shape == timeMs.shape:
            arr = arr[view]
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            values.append(arr)

    if not values:
        return

    values = np.concatenate(values)
    if symmetric:
        limit = np.max(np.abs(values))
        limit = limit if limit > 0 else 1.0
        axis.set_ylim(-limit * (1.0 + padFraction), limit * (1.0 + padFraction))
        return

    yMin = np.min(values)
    yMax = np.max(values)
    pad = (np.abs(yMin) * padFraction if yMin != 0 else 1.0) if np.isclose(yMin, yMax) else (yMax - yMin) * padFraction
    axis.set_ylim(yMin - pad, yMax + pad)


def onlyValidSamples(y, valid):
    out = np.asarray(y, dtype=float).copy()
    out[~valid] = np.nan
    return out


def formatTrajectoryAxis(axis, label):
    """Show small trajectory values with an explicit publication-scale unit."""
    scale = 1e-4
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / scale:g}"))
    axis.yaxis.offsetText.set_visible(False)
    axis.set_ylabel(f"{label} (a.u.)", fontsize=11)
    axis.text(
        0.01,
        0.96,
        r"$\times 10^{-4}$",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fullConfig = load_config()
dataPaths, scansS = LoadTrainingDataConfig(fullConfig)
testingShapeTypes = LoadTestingShapeTypes(fullConfig)

config = fullConfig["model"]
modelType = config["model"]
nChannels = config["nChannels"]
nLayers = config["nLayers"]
kernel = config["kernelSize"]
shiftByNSamples = config["shiftByNSamples"]
filterFreq = config["filterFreq"]
nEpoch = config["nEpoch"]
lr = config["lr"]
weightDecay = config["weightDecay"]
torchRes = config["outRes"]


testingAmplitudeMinMagnitude = 0.5
testingAmplitudeFraction = 0.95
testingAmplitudePerShape = True

lineColors = {
    "Measured": "black",
    "Nominal": "steelblue",
    "GIRF corrected": "darkorange",
    "TCN corrected": "seagreen",
}
correctionMethods = ("Nominal", "GIRF corrected", "TCN corrected")
supplementPath = "./paper2026/Sup_4.pdf"
supplementPdf = PdfPages(supplementPath)

from utils.GradientCorrector import GradientCorector

grCorr = GradientCorector("AV-NEO,BGA-12,AfterBrukerTuneup")

os.makedirs("images", exist_ok=True)
os.makedirs("./paper2026", exist_ok=True)


def buildModel():
    if modelType == "TCNSkip":
        return TCNFullSkip(3, [nChannels] * nLayers, kernel, 0.15, shiftByNSamples)
    return TCNFull(3, [nChannels] * nLayers, kernel, 0.15, shiftByNSamples)


def loadAxisModel(axisSymbol):
    model = buildModel()
    skipStr = "_skip" if model.model_name == "TCNFullSkip" else "_"
    outPath = "utils/gradModels/grad" + axisSymbol + skipStr + f"_{model.num_channels}_{model.nLayers}_{model.kernel_size}"
    checkpoint = torch.load(outPath, map_location=device)["model_state_dict"]

    stateDict = {}
    for key, value in checkpoint.items():
        cleanKey = key[len("_orig_mod.") :] if key.startswith("_orig_mod.") else key
        stateDict[cleanKey] = value

    model.load_state_dict(stateDict)
    model.eval()
    model.to(device)
    return model


def addNrmseBoxplot(axis, distributionRows, waveform):
    axesOrder = ("X", "Y", "Z")
    methods = correctionMethods
    positions = []
    data = []
    colors = []
    groupCenters = []

    for axisInd, axisSymbol in enumerate(axesOrder):
        base = axisInd * (len(methods) + 1)
        groupCenters.append(base + 2)
        for methodInd, method in enumerate(methods):
            values = [
                row["nrmse"]
                for row in distributionRows
                if row["waveform"] == waveform and row["axis"] == axisSymbol and row["method"] == method and np.isfinite(row["nrmse"])
            ]
            if not values:
                values = [np.nan]
            data.append(values)
            positions.append(base + methodInd + 1)
            colors.append(lineColors[method])

    box = axis.boxplot(data, positions=positions, widths=0.65, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(color)
        patch.set_linewidth(0.8)
    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(0.9)
    for whisker in box["whiskers"]:
        whisker.set_color("0.35")
        whisker.set_linewidth(0.7)
    for cap in box["caps"]:
        cap.set_color("0.35")
        cap.set_linewidth(0.7)

    axis.set_xticks(groupCenters)
    axis.set_xticklabels(axesOrder)
    axis.set_xlabel("Axis")
    axis.set_ylabel("NRMSE (-)")
    axis.set_title(f"{waveform.capitalize()} waveform")
    axis.set_yscale("log")
    axis.grid(axis="y", linewidth=0.4, alpha=0.35)


def plotCombinedNrmseBoxplots(distributionRows, outPrefix):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
    addNrmseBoxplot(axes[0], distributionRows, "gradient")
    addNrmseBoxplot(axes[1], distributionRows, "trajectory")
    axes[0].set_xlabel("")
    methods = correctionMethods
    legendHandles = [plt.Line2D([0], [0], color=lineColors[method], linewidth=6, alpha=0.55, label=method) for method in methods]
    axes[0].legend(handles=legendHandles, loc="best")
    fig.tight_layout()
    fig.savefig(f"./paper2026/{outPrefix}.pdf", bbox_inches="tight", pad_inches=0)
    fig.savefig(f"./paper2026/{outPrefix}.png", bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)


def plotTrajectoryResidualHistograms(residualRows, outPrefix):
    axesOrder = ("X", "Y", "Z")
    methods = correctionMethods
    residualLookup = {(row["axis"], row["method"]): row["residuals"] for row in residualRows}
    allResiduals = [row["residuals"][np.isfinite(row["residuals"])] for row in residualRows if row["residuals"].size > 0]
    if not allResiduals:
        return

    allResiduals = np.concatenate(allResiduals)
    if allResiduals.size == 0:
        return

    maxAbs = np.nanpercentile(np.abs(allResiduals), 99.0)
    if not np.isfinite(maxAbs) or maxAbs <= 0:
        maxAbs = np.nanmax(np.abs(allResiduals))
    if not np.isfinite(maxAbs) or maxAbs <= 0:
        maxAbs = 1.0
    residualXlim = (-maxAbs, maxAbs)
    bins = np.linspace(residualXlim[0], residualXlim[1], 81)

    fig, axes = plt.subplots(len(axesOrder), len(methods), figsize=(9.2, 6.2), sharex=True, sharey="row")
    for axisInd, axisSymbol in enumerate(axesOrder):
        for methodInd, method in enumerate(methods):
            ax = axes[axisInd, methodInd]
            residuals = residualLookup.get((axisSymbol, method), np.asarray([]))
            residuals = residuals[np.isfinite(residuals)]
            residuals = residuals[np.abs(residuals) <= maxAbs]
            if residuals.size > 0:
                ax.hist(
                    residuals,
                    bins=bins,
                    density=True,
                    color=lineColors[method],
                    alpha=0.72,
                    linewidth=0,
                )
            ax.axvline(0, color="black", linewidth=0.7)
            ax.set_xlim(*residualXlim)
            ax.grid(axis="y", linewidth=0.4, alpha=0.3)
            if axisInd == 0:
                ax.set_title(method)
            if methodInd == 0:
                ax.set_ylabel(f"{axisSymbol}\nDensity")
            if axisInd == len(axesOrder) - 1:
                ax.set_xlabel("Trajectory residual (-)")

    fig.tight_layout()
    fig.savefig(f"./paper2026/{outPrefix}.pdf", bbox_inches="tight", pad_inches=0)
    fig.savefig(f"./paper2026/{outPrefix}.png", bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)


modelForParamCount = buildModel()
modelOriginal = TCN(2, 1, [48] * 5, 16, 0.2)
print(f"Number of trainable parameters: {sum(p.numel() for p in modelForParamCount.parameters() if p.requires_grad)}")
print(f"Number of trainable parameters in original model: {sum(p.numel() for p in modelOriginal.parameters() if p.requires_grad)}")


def evaluateAxis(axisSymbol, gradAxisInd):
    print(f"Evaluating axis {axisSymbol}")
    model = loadAxisModel(axisSymbol)

    shapeSplitEnabled = len(testingShapeTypes) > 0
    shapeKwargs = {"includeShapeTypes": testingShapeTypes} if shapeSplitEnabled else {"excludeShapeTypes": testingShapeTypes}

    xData, yData, yTraj, maskData, labels = createTrainingSet(
        dataPaths,
        scansS,
        gradAxisInd,
        doB0=False,
        doTraj=False,
        absoluteMapping=False,
        shiftByNSamples=shiftByNSamples,
        filterFreq=filterFreq,
        outRes=torchRes,
        estimateNoise=True,
        filterLowAmp=0,
        **shapeKwargs,
    )

    if shapeSplitEnabled:
        testIndices = np.arange(xData.shape[0], dtype=np.int64)
        shapeFilter = testingShapeTypes
        print(f"Axis {axisSymbol}: found {testIndices.size} shape-held-out curves: {', '.join(sorted(testingShapeTypes))}")
    else:
        testIndices = getValidationIndices(xData.shape[0], trainFraction=0.85, seed=42)
        shapeFilter = None
        print(f"Axis {axisSymbol}: found {testIndices.size} curves from the seeded 85/15 validation split")

    statsIndices = filterTestingExamples(
        testIndices,
        xData,
        labels,
        shapeTypes=shapeFilter,
        amplitudeMinMagnitude=testingAmplitudeMinMagnitude,
    )
    shapeSummary = "all shapes" if shapeFilter is None else ", ".join(sorted(shapeFilter))
    print(
        f"Axis {axisSymbol}: using {statsIndices.size} held-out curves for statistics after filters: "
        f"shapes={shapeSummary}, amplitude_min_magnitude={testingAmplitudeMinMagnitude}"
    )
    # Figure selection uses the full validation pool rather than the statistics
    # amplitude threshold, so waveform families with lower designed amplitudes
    # (notably ARCH_SPIRAL) remain eligible for the supplement.
    plotSourceIndices = filterTestingExamples(
        testIndices,
        xData,
        labels,
        amplitudeFraction=testingAmplitudeFraction,
        amplitudePerShape=testingAmplitudePerShape,
    )
    statsIndices = np.unique(np.concatenate((statsIndices, plotSourceIndices)))
    print(
        f"Axis {axisSymbol}: plotting {plotSourceIndices.size} held-out curves after near/max-amplitude filter: "
        f"amplitude_fraction={testingAmplitudeFraction}, per_shape={testingAmplitudePerShape}"
    )
    plotIndices = np.nonzero(np.isin(statsIndices, plotSourceIndices))[0]

    xData = xData[statsIndices]
    yData = yData[statsIndices]
    yTraj = yTraj[statsIndices]
    maskData = maskData[statsIndices]
    labels = [labels[ind] for ind in statsIndices]

    with torch.no_grad():
        yPred = model.forward(xData.to(device)).detach().cpu().numpy()

    xData = xData.detach().cpu().numpy()
    yData = yData.detach().cpu().numpy()
    yTraj = yTraj.detach().cpu().numpy()
    maskData = maskData.detach().cpu().numpy()

    t = np.arange(xData.shape[-1]) * torchRes * 1e3

    girfPred = np.zeros_like(yPred)
    theoryTraj = np.zeros_like(yTraj)
    girfTraj = np.zeros_like(yTraj)
    tcnTraj = np.zeros_like(yTraj)
    theoryGradAligned = delaySignal(xData[:, [0], :], shiftByNSamples, axis=-1)

    for curveInd in range(maskData.shape[0]):
        girfShape, _ = grCorr.systemTransform(
            xData[curveInd, 0, :],
            f"{axisSymbol}Grad",
            torchRes,
            gradientPreDelay=shiftByNSamples * torchRes,
        )
        girfPred[curveInd, 0, :] = girfShape
        theoryTraj[curveInd, 0, :] = cumulativeIntegrate(theoryGradAligned[curveInd, 0, :], torchRes, axis=-1)
        girfTraj[curveInd, 0, :] = cumulativeIntegrate(girfShape, torchRes, axis=-1)
        tcnTraj[curveInd, 0, :] = cumulativeIntegrate(yPred[curveInd, 0, :], torchRes, axis=-1)

    validMask = (maskData > 0).astype(np.float32)
    trajectoryPredictions = {
        "Nominal": theoryTraj,
        "GIRF corrected": girfTraj,
        "TCN corrected": tcnTraj,
    }
    gradientPredictions = {
        "Nominal": theoryGradAligned,
        "GIRF corrected": girfPred,
        "TCN corrected": yPred,
    }

    metricRows = []
    distributionRows = []
    residualRows = []
    for method, traj in trajectoryPredictions.items():
        rmse = maskedRmse(traj, yTraj, validMask)
        nmrse = maskedNrmse(traj, yTraj, validMask)
        metricRows.append((axisSymbol, method, rmse, nmrse, statsIndices.size))
        residualRows.append(
            {
                "axis": axisSymbol,
                "method": method,
                "residuals": (traj - yTraj)[validMask > 0].astype(np.float32, copy=False),
            }
        )
        print(f"Axis {axisSymbol} testing trajectory RMSE {method}: {rmse:.3E}, NMRSE: {nmrse:.3E}")

    for waveform, predictions, reference in (
        ("trajectory", trajectoryPredictions, yTraj),
        ("gradient", gradientPredictions, yData),
    ):
        for method, pred in predictions.items():
            curveNrmse = perCurveMaskedNrmse(pred, reference, validMask)
            for curveInd, value in enumerate(curveNrmse):
                distributionRows.append(
                    {
                        "axis": axisSymbol,
                        "waveform": waveform,
                        "method": method,
                        "curve_index": int(statsIndices[curveInd]),
                        "shape": labels[curveInd],
                        "nrmse": value,
                    }
                )

    perCurveTcnRmse = np.asarray([maskedRmse(tcnTraj[[ind]], yTraj[[ind]], validMask[[ind]]) for ind in range(maskData.shape[0])])
    if axisSymbol == "X" and plotIndices.size > 0:
        # Select one representative (median-error) example for every available
        # waveform family. This yields a compact, reproducible supplement rather
        # than a large set of arbitrarily sampled curves.
        exampleIndices = []
        for shapeName in sorted(set(labels[ind] for ind in plotIndices)):
            candidates = plotIndices[np.asarray([labels[ind] == shapeName for ind in plotIndices])]
            ordered = candidates[np.argsort(np.nan_to_num(perCurveTcnRmse[candidates], nan=np.inf))]
            exampleIndices.append(int(ordered[len(ordered) // 2]))

        for plotInd, curveInd in enumerate(exampleIndices):
            fig, ax = plt.subplots(4, 1, figsize=(7.2, 8.4), sharex=True)
            curveValid = maskData[curveInd, 0, :] > 0
            measured = onlyValidSamples(1e2 * yData[curveInd, 0, :] * xData[curveInd, 1, 0], curveValid)
            theoryGrad = onlyValidSamples(1e2 * theoryGradAligned[curveInd, 0, :] * xData[curveInd, 1, 0], curveValid)
            girfGrad = onlyValidSamples(1e2 * girfPred[curveInd, 0, :] * xData[curveInd, 1, 0], curveValid)
            tcnGrad = onlyValidSamples(1e2 * yPred[curveInd, 0, :] * xData[curveInd, 1, 0], curveValid)
            measuredTraj = onlyValidSamples(yTraj[curveInd, 0, :], curveValid)
            theoryCurveTraj = onlyValidSamples(theoryTraj[curveInd, 0, :], curveValid)
            girfCurveTraj = onlyValidSamples(girfTraj[curveInd, 0, :], curveValid)
            tcnCurveTraj = onlyValidSamples(tcnTraj[curveInd, 0, :], curveValid)
            theoryGradErr = theoryGrad - measured
            girfGradErr = girfGrad - measured
            tcnGradErr = tcnGrad - measured
            theoryTrajErr = theoryCurveTraj - measuredTraj
            girfTrajErr = girfCurveTraj - measuredTraj
            tcnTrajErr = tcnCurveTraj - measuredTraj

            ax[0].plot(t, theoryGrad, color=lineColors["Nominal"], label="Nominal")
            ax[0].plot(t, measured, color=lineColors["Measured"], label="Measured")
            ax[0].plot(t, girfGrad, color=lineColors["GIRF corrected"], label="GIRF corrected")
            ax[0].plot(t, tcnGrad, color=lineColors["TCN corrected"], label="TCN corrected")
            ax[0].set_ylabel(f"G{axisSymbol} (%)", fontsize=11)
            ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.34), ncol=4, frameon=False, fontsize=9)

            ax[1].plot(t, theoryGradErr, color=lineColors["Nominal"], label="Nominal")
            ax[1].plot(t, girfGradErr, color=lineColors["GIRF corrected"], label="GIRF corrected")
            ax[1].plot(t, tcnGradErr, color=lineColors["TCN corrected"], label="TCN corrected")
            ax[1].set_ylabel("Gradient error (%)", fontsize=11)
            ax[1].axhline(0, linewidth=1)

            ax[2].plot(t, measuredTraj, color=lineColors["Measured"], label="Measured")
            ax[2].plot(t, theoryCurveTraj, color=lineColors["Nominal"], label="Nominal")
            ax[2].plot(t, girfCurveTraj, color=lineColors["GIRF corrected"], label="GIRF corrected")
            ax[2].plot(t, tcnCurveTraj, color=lineColors["TCN corrected"], label="TCN corrected")
            formatTrajectoryAxis(ax[2], "Trajectory")

            ax[3].plot(t, theoryTrajErr, color=lineColors["Nominal"], label="Nominal")
            ax[3].plot(t, girfTrajErr, color=lineColors["GIRF corrected"], label="GIRF corrected")
            ax[3].plot(t, tcnTrajErr, color=lineColors["TCN corrected"], label="TCN corrected")
            ax[3].set_xlabel("Time (ms)", fontsize=11)
            formatTrajectoryAxis(ax[3], "Trajectory error")
            ax[3].axhline(0, linewidth=1)

            nonZeroIndices = np.flatnonzero(((xData[curveInd, 0, :] * xData[curveInd, 1, 0]) != 0) & curveValid)
            if nonZeroIndices.size > 0:
                # Anchor the viewport to the actual waveform onset and retain a
                # small amount of baseline, avoiding the former fixed t=0 crop.
                firstInd = max(0, int(nonZeroIndices[0]) - 20)
                lastInd = min(firstInd + 650, int(nonZeroIndices[-1]) + 1, xData.shape[-1] - 1)
                xMin = firstInd * torchRes * 1e3
                xMax = lastInd * torchRes * 1e3
            else:
                xMin = t[0]
                xMax = t[-1]

            ax[0].set_xlim(xMin, xMax)
            setViewportYlim(ax[0], t, [theoryGrad, measured, girfGrad, tcnGrad], xMin=xMin, xMax=xMax)
            setViewportYlim(
                ax[1],
                t,
                [theoryGradErr, girfGradErr, tcnGradErr],
                xMin=xMin,
                xMax=xMax,
                symmetric=True,
            )
            setViewportYlim(
                ax[2],
                t,
                [measuredTraj, theoryCurveTraj, girfCurveTraj, tcnCurveTraj],
                xMin=xMin,
                xMax=xMax,
            )
            setViewportYlim(
                ax[3],
                t,
                [theoryTrajErr, girfTrajErr, tcnTrajErr],
                xMin=xMin,
                xMax=xMax,
                symmetric=True,
            )

            for subplot in ax:
                subplot.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
                subplot.tick_params(labelsize=10)

            fig.suptitle(f"{labels[curveInd].replace('_', ' ')} waveform — {axisSymbol} axis", fontsize=14, y=0.985)
            fig.subplots_adjust(left=0.15, right=0.98, bottom=0.08, top=0.90, hspace=0.18)
            outputStem = f"images/testingTrajectory_{axisSymbol}_{labels[curveInd]}"
            fig.savefig(f"{outputStem}.pdf", bbox_inches="tight")
            fig.savefig(f"{outputStem}.png", bbox_inches="tight", dpi=600)
            supplementPdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        summaryCurve = exampleIndices[len(exampleIndices) // 2]
        summaryValid = maskData[summaryCurve, 0, :] > 0
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(t, onlyValidSamples(yTraj[summaryCurve, 0, :], summaryValid), color=lineColors["Measured"], label="Measured")
        ax.plot(t, onlyValidSamples(theoryTraj[summaryCurve, 0, :], summaryValid), color=lineColors["Nominal"], label="Nominal")
        ax.plot(t, onlyValidSamples(girfTraj[summaryCurve, 0, :], summaryValid), color=lineColors["GIRF corrected"], label="GIRF corrected")
        ax.plot(t, onlyValidSamples(tcnTraj[summaryCurve, 0, :], summaryValid), color=lineColors["TCN corrected"], label="TCN corrected")
        ax.set_xlabel("t (ms)")
        ax.set_ylabel("Trajectory (-)")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(f"./paper2026/resultingPrediction_{axisSymbol}.pdf", bbox_inches="tight", pad_inches=0)
        fig.savefig(f"./paper2026/resultingPrediction_{axisSymbol}.png", bbox_inches="tight", pad_inches=0)
        if axisSymbol == "X":
            fig.savefig("./paper2026/resultingPrediction.pdf", bbox_inches="tight", pad_inches=0)
            fig.savefig("./paper2026/resultingPrediction.png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    return metricRows, distributionRows, residualRows


gradAxes = ["Z", "X", "Y"]
metricRows = []
distributionRows = []
residualRows = []
for axisSymbol in ["X", "Y", "Z"]:
    axisMetricRows, axisDistributionRows, axisResidualRows = evaluateAxis(axisSymbol, gradAxes.index(axisSymbol))
    metricRows.extend(axisMetricRows)
    distributionRows.extend(axisDistributionRows)
    residualRows.extend(axisResidualRows)

supplementPdf.close()
print(f"Supplementary waveform PDF written to {supplementPath}")

with open("./paper2026/trajectoryRMSE_testing.csv", "w") as rmseFile:
    rmseFile.write("axis,method,trajectory_rmse,trajectory_nmrse,n_testing_curves\n")
    for axisSymbol, method, rmse, nmrse, nCurves in metricRows:
        rmseFile.write(f"{axisSymbol},{method},{rmse:.8e},{nmrse:.8e},{nCurves}\n")

with open("./paper2026/waveformNRMSE_testing_curves.csv", "w") as nrmseFile:
    nrmseFile.write("axis,waveform,method,curve_index,shape,nrmse\n")
    for row in distributionRows:
        nrmseFile.write(f"{row['axis']},{row['waveform']},{row['method']},{row['curve_index']},{row['shape']},{row['nrmse']:.8e}\n")

plotCombinedNrmseBoxplots(distributionRows, "waveformNRMSE_boxplots")
plotTrajectoryResidualHistograms(residualRows, "trajectoryResidualHistograms")
