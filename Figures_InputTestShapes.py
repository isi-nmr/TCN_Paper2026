import os

import matplotlib.pyplot as plt
import numpy as np

from utils.BrukerMRI import ReadParamFile
from utils.utils import LoadTestingShapeTypes, LoadTrainingDataConfig, load_config

skipShapeNames = {"PRBS"}
excludedFigureShapeNames = {}
preferredShapeScans = {
    "EPI": 13,
}
shapeColors = {
    "ARCH_SPIRAL": "tab:purple",
    "CHIRP": "tab:orange",
    "EPI": "tab:green",
    "MGE": "tab:red",
    "PRGW": "tab:brown",
    "READOUT": "black",
    "ROSE": "tab:pink",
    "TRAPZ_SERIES": "tab:cyan",
    "TRIANGLE": "tab:olive",
}
plotDurationMs = 4.0
outPath = "paper2026/InputTestShapes.pdf"
outRes = 2e-6


def getArray(method, *names):
    for name in names:
        value = method.get(name, None)
        if value is not None:
            return np.asarray(value, dtype=float)
    raise KeyError(f"Missing any of {names}")


def getShapeExampleRecord(dataPath, dataSetInd, scan, outRes):
    methodPath = os.path.join(dataPath, str(scan), "method")
    if not os.path.exists(methodPath):
        print(f"Skipping missing scan {scan}: {methodPath}")
        return None

    method = ReadParamFile(methodPath)
    if method.get("TestShape", "") in skipShapeNames:
        return None

    testShapeVec = getArray(method, "TestShapeVec")
    shapeScale = np.max(np.abs(testShapeVec))
    if testShapeVec.size == 0 or shapeScale == 0:
        print(f"Skipping empty shape in scan {scan}: {methodPath}")
        return None

    return {
        "dataSet": dataSetInd,
        "scan": scan,
        "shapeName": method.get("TestShape", ""),
        "nSamples": testShapeVec.size,
        "nFiniteSamples": np.count_nonzero(np.isfinite(testShapeVec)),
        "nNonZeroSamples": np.count_nonzero(np.isfinite(testShapeVec) & (testShapeVec != 0)),
        "durationMs": testShapeVec.size * outRes * 1e3,
        "timeMs": np.arange(testShapeVec.size) * outRes * 1e3,
        "normalizedShape": testShapeVec / shapeScale,
    }


def loadShapeRecords(dataPaths, scansS, outRes):
    records = []
    for dataSetInd, (dataPath, scans) in enumerate(zip(dataPaths, scansS, strict=True), start=1):
        for scan in scans:
            record = getShapeExampleRecord(dataPath, dataSetInd, scan, outRes)
            if record is not None:
                records.append(record)

    return records


def loadShapeExamples(records):
    examples = {}
    preferredExamples = {}
    for record in records:
        preferredScan = preferredShapeScans.get(record["shapeName"])
        if preferredScan == record["scan"]:
            preferredExamples[record["shapeName"]] = record
        examples.setdefault(record["shapeName"], record)

    examples.update(preferredExamples)

    return [examples[shapeName] for shapeName in sorted(examples)]


def plotRecord(axis, record):
    shapeName = record["shapeName"]
    plotMask = record["timeMs"] <= plotDurationMs
    axis.plot(record["timeMs"][plotMask], record["normalizedShape"][plotMask], color=shapeColors.get(shapeName, "black"), linewidth=1.4)
    axis.axhline(0, color="0.6", linewidth=0.5)
    axis.set_title(shapeName, fontsize=8)
    axis.set_xlim(0, plotDurationMs)
    axis.set_ylim(-1.08, 1.08)
    axis.tick_params(labelsize=7, length=2)
    axis.grid(True, color="0.9", linewidth=0.5)


def plotShapeExamples(records, outPath):
    if len(records) == 0:
        print(f"Skipping empty figure {outPath}")
        return

    nCols = 2
    nRows = int(np.ceil(len(records) / nCols))
    fig, axes = plt.subplots(nRows, nCols, figsize=(7.2, 1.45 * nRows), sharex=False, sharey=True)
    axesFlat = np.ravel(axes)

    for axis, record in zip(axesFlat, records, strict=False):
        plotRecord(axis, record)

    for axis in axesFlat[len(records) :]:
        axis.axis("off")

    for rowInd in range(nRows):
        axes[rowInd, 0].set_ylabel(r"$G_n$ (-)", fontsize=8)

    for axis in axes[-1, :]:
        if axis.has_data():
            axis.set_xlabel("t (ms)", fontsize=8)

    fig.tight_layout()
    fig.savefig(outPath, bbox_inches="tight")
    fig.savefig(outPath.replace(".pdf", ".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(os.path.dirname(outPath), exist_ok=True)

    config = load_config()
    dataPaths, scansS = LoadTrainingDataConfig(config)
    testingShapeTypes = LoadTestingShapeTypes(config)
    excludedShapeTypes = testingShapeTypes or excludedFigureShapeNames

    records = loadShapeRecords(dataPaths, scansS, outRes)
    plotRecords = [record for record in records if record["shapeName"] not in excludedShapeTypes]

    exampleRecords = loadShapeExamples(plotRecords)
    plotShapeExamples(exampleRecords, outPath)

    print(f"Wrote {outPath}")
    print(f"Wrote {outPath.replace('.pdf', '.png')}")


if __name__ == "__main__":
    main()
