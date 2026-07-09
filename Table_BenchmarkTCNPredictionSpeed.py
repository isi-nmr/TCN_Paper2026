import json
import time
from pathlib import Path

import torch

from nn_models.TCN import OriginalTCNSequencePredictor, TCN, TCNFull, TCNFullSkip


batchSize = 256
nSamplesList = [4096, 512]
warmupRepeats = 10
benchmarkRepeats = 50
deviceName = "cuda:0" if torch.cuda.is_available() else "cpu"
compileModel = False
seed = 42
latexOut = "paper2026/TCNPredictionSpeedTable.tex"


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timeModel(model, x, *, warmup, repeats, device):
    model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        synchronize(device)

        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            y = model(x)
            synchronize(device)
            timings.append(time.perf_counter() - start)

    return torch.tensor(timings), y


def buildCurrentModel(config, doSkip=True):
    nChannels = config["nChannels"]
    nLayers = config["nLayers"]
    kernel = config["kernelSize"]
    dropout = config["dropout"]
    shiftBySamples = config["shiftByNSamples"]

    if config.get("model") == "TCNSkip" and doSkip:
        return TCNFullSkip(3, [nChannels] * nLayers, kernel, dropout, shiftBySamples)

    return TCNFull(3, [nChannels] * nLayers, kernel, dropout, shiftBySamples)


def buildOriginalModel(config, *, windowScale=1):
    channels = [config.get("nChannels", 48)] * config.get("nLayers", 5)
    kernel = config.get("kernelSize", 16)
    dropout = config.get("dropout", 0.002)
    windowSize = config.get("windowSize", 75) * windowScale
    predictPoint = config.get("predictPoint", 65) * windowScale
    chunkSize = config.get("chunkSize", 32768)

    core = TCN(2, 1, channels, kernel, dropout)
    return OriginalTCNSequencePredictor(core, window_size=windowSize, predict_point=predictPoint, chunk_size=chunkSize)


def formatTiming(timings, batchSize, nSamples):
    timingsMs = timings * 1e3
    meanSeconds = timings.mean().item()

    return {
        "meanMs": timingsMs.mean().item(),
        "stdMs": timingsMs.std(unbiased=False).item(),
        "minMs": timingsMs.min().item(),
        "waveformsPerSecond": batchSize / meanSeconds,
    }


def printResult(nSamples, modelName, stats, outputShape):
    print(
        f"{nSamples:>5} | {modelName:<15} | "
        f"mean {stats['meanMs']:8.3f} ms | "
        f"std {stats['stdMs']:7.3f} ms | "
        f"min {stats['minMs']:8.3f} ms | "
        f"{stats['waveformsPerSecond']:10.1f} waveforms/s | "
        f"out {tuple(outputShape)}"
    )


def latexEscape(text):
    return text.replace("_", r"\_")


def latexBold(text):
    return rf"\textbf{{{text}}}"


def writeLatexTable(results, outPath):
    lines = [
        r"\begin{tabular}{rlrrrr}",
        r"\hline",
        r"Samples & Model & Mean (ms) & Std (ms) & Min (ms) & Waveforms/s \\",
        r"\hline",
    ]

    previousNSamples = None
    for result in results:
        if previousNSamples is not None and result["nSamples"] != previousNSamples:
            lines.append(r"\hline")

        sampleCell = str(result["nSamples"]) if result["nSamples"] != previousNSamples else ""
        modelCell = latexEscape(result["model"])
        meanCell = f"{result['meanMs']:.3f}"
        stdCell = f"{result['stdMs']:.3f}"
        minCell = f"{result['minMs']:.3f}"
        waveformsCell = f"{result['waveformsPerSecond']:.1f}"

        if result["model"] == "current TCN":
            modelCell = latexBold(modelCell)
            meanCell = latexBold(meanCell)
            stdCell = latexBold(stdCell)
            minCell = latexBold(minCell)
            waveformsCell = latexBold(waveformsCell)

        lines.append(
            f"{sampleCell} & "
            f"{modelCell} & "
            f"{meanCell} & "
            f"{stdCell} & "
            f"{minCell} & "
            f"{waveformsCell} \\\\"
        )
        previousNSamples = result["nSamples"]

    lines.extend([r"\hline", r"\end{tabular}"])

    outPath = Path(outPath)
    outPath.parent.mkdir(parents=True, exist_ok=True)
    outPath.write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    torch.manual_seed(seed)

    with open("config.json") as f:
        config = json.load(f)

    device = torch.device(deviceName)
    currentConfig = config["model"]
    originalConfig = config.get("originalModel", {})
    maxOriginalWindow = originalConfig.get("windowSize", 75) * 4
    for nSamples in nSamplesList:
        if nSamples < maxOriginalWindow:
            raise ValueError(f"nSamplesList values must be at least {maxOriginalWindow} for the original TCN 4x benchmark")

    print(f"Device: {device}")
    print(f"Input: batch={batchSize}, samples={nSamplesList}, warmup={warmupRepeats}, repeats={benchmarkRepeats}")
    print("Samples | Model           | Timing")
    print("-" * 104)
    results = []

    for nSamples in nSamplesList:
        xCurrent = torch.randn(batchSize, 3, nSamples, device=device)
        xOriginal = xCurrent[:, :2, :]

        models = [
            ("current TCN", buildCurrentModel(currentConfig).to(device), xCurrent),
            ("original TCN", buildOriginalModel(originalConfig).to(device), xOriginal),
            ("original TCN 4x", buildOriginalModel(originalConfig, windowScale=4).to(device), xOriginal),
        ]

        for modelName, model, x in models:
            if compileModel:
                model = torch.compile(model)

            timings, out = timeModel(model, x, warmup=warmupRepeats, repeats=benchmarkRepeats, device=device)
            stats = formatTiming(timings, batchSize, nSamples)

            printResult(nSamples, modelName, stats, out.shape)
            results.append({"nSamples": nSamples, "model": modelName, **stats})

        print("-" * 104)

    latexTable = writeLatexTable(results, latexOut)
    print(f"\nLaTeX table written to {latexOut}\n")
    print(latexTable)


if __name__ == "__main__":
    main()
