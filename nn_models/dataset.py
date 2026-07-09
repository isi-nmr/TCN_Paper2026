import hashlib
import json
import os

import numpy as np
import torch

from utils.BrukerMRI import ReadParamFile
from utils.utils import getData

DATASET_CACHE_VERSION = 1
DATASET_CACHE_DIR = "datasets"
GRAD_AXIS_NAMES = {0: "Z", 1: "X", 2: "Y"}


def _NormalizeForCache(value):
    if isinstance(value, dict):
        return {key: _NormalizeForCache(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple | list):
        return [_NormalizeForCache(item) for item in value]
    if isinstance(value, np.ndarray):
        return _NormalizeForCache(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _GetDatasetCacheInfo(dataPaths, scansS, gradAxisInd, cacheParams):
    payload = {
        "version": DATASET_CACHE_VERSION,
        "dataPaths": _NormalizeForCache(dataPaths),
        "scansS": _NormalizeForCache(scansS),
        "gradAxisInd": gradAxisInd,
        "cacheParams": _NormalizeForCache(cacheParams),
    }
    payloadJson = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    cacheHash = hashlib.sha256(payloadJson.encode("utf-8")).hexdigest()[:16]
    axisName = GRAD_AXIS_NAMES.get(gradAxisInd, f"axis{gradAxisInd}")
    b0Suffix = "_B0" if cacheParams["doB0"] else ""
    trajSuffix = "_traj" if cacheParams["doTraj"] else ""
    cachePath = os.path.join(DATASET_CACHE_DIR, f"trainingSet_{axisName}{b0Suffix}{trajSuffix}_{cacheHash}.pt")
    return cachePath, payload


def _LoadCachedTrainingSet(cachePath, cachePayload):
    if not os.path.exists(cachePath):
        return None

    try:
        cached = torch.load(cachePath, map_location="cpu", weights_only=False)
    except (EOFError, OSError, RuntimeError) as exc:
        print(f"Ignoring unreadable cached dataset {cachePath}: {exc}")
        return None

    if cached.get("cachePayload") != cachePayload:
        return None

    print(f"Loaded cached dataset: {cachePath}")
    return cached["xData"], cached["yData"], cached["yTrajData"], cached["mask"], cached["labels"]


def _SaveCachedTrainingSet(cachePath, cachePayload, xData, yData, yTrajData, mask, labels):
    os.makedirs(os.path.dirname(cachePath), exist_ok=True)
    tmpPath = f"{cachePath}.{os.getpid()}.tmp"
    torch.save(
        {
            "cachePayload": cachePayload,
            "xData": xData.detach().cpu(),
            "yData": yData.detach().cpu(),
            "yTrajData": yTrajData.detach().cpu(),
            "mask": mask.detach().cpu(),
            "labels": labels,
        },
        tmpPath,
    )
    os.replace(tmpPath, cachePath)
    print(f"Saved cached dataset: {cachePath}")


def _DatasetProgress(progressLabel, message):
    if progressLabel is not None:
        print(f"[dataset:{progressLabel}] {message}", flush=True)


def _build_input_channels(testShapeRelativeAmp):

    shape_channel = np.asarray(testShapeRelativeAmp, dtype=np.float32)
    # Match runtime preprocessing: derive the auxiliary feature from the physical
    # test shape after bringing it into the same normalized units as channel 0.
    gradient_channel = np.gradient(testShapeRelativeAmp).astype(np.float32, copy=False) * 15

    if not np.all(np.isfinite(gradient_channel)):
        raise ValueError("Gradient channel contains non-finite values")

    return shape_channel, gradient_channel


def UseShape(shapeName, includeShapeTypes, excludeShapeTypes):
    if includeShapeTypes is not None and shapeName not in includeShapeTypes:
        return False

    return excludeShapeTypes is None or shapeName not in excludeShapeTypes


def createTrainingSet(
    dataPaths,
    scansS,
    gradAxisInd,
    *,
    doB0=False,
    doTraj=False,
    prepend=0,
    onlyFirstLinStep=False,
    repeatShorter=False,
    absoluteMapping=False,
    estimateNoise=True,
    shiftByNSamples=18,
    filterFreq=None,
    outRes=2e-6,
    gradientFirst=False,
    filterLowAmp=3e-2,
    includeShapeTypes=None,
    excludeShapeTypes=None,
    useCache=True,
    progressLabel=None,
):
    includeShapeTypes = set(includeShapeTypes) if includeShapeTypes is not None else None
    excludeShapeTypes = set(excludeShapeTypes) if excludeShapeTypes is not None else None

    cacheParams = {
        "doB0": doB0,
        "doTraj": doTraj,
        "prepend": prepend,
        "onlyFirstLinStep": onlyFirstLinStep,
        "repeatShorter": repeatShorter,
        "absoluteMapping": absoluteMapping,
        "estimateNoise": estimateNoise,
        "shiftByNSamples": shiftByNSamples,
        "filterFreq": filterFreq,
        "outRes": outRes,
        "gradientFirst": gradientFirst,
        "filterLowAmp": filterLowAmp,
        "includeShapeTypes": includeShapeTypes,
        "excludeShapeTypes": excludeShapeTypes,
    }
    cachePath, cachePayload = _GetDatasetCacheInfo(dataPaths, scansS, gradAxisInd, cacheParams)
    if useCache:
        _DatasetProgress(progressLabel, f"Checking cache {cachePath}")
        cachedTrainingSet = _LoadCachedTrainingSet(cachePath, cachePayload)
        if cachedTrainingSet is not None:
            _DatasetProgress(progressLabel, "Cache hit")
            return cachedTrainingSet
        _DatasetProgress(progressLabel, "Cache miss; scanning raw training data")

    nSamples = []
    batchDim = 0

    dsetCrop = 9500
    minRandShift = 200
    maxRandShift = 400

    gapSize = 50

    gradShift = 0

    for scanind, dataPath in enumerate(dataPaths):
        scans = scansS[scanind]
        for scan in scans:
            _DatasetProgress(progressLabel, f"Counting scan {scan} in {dataPath}")
            acqp = ReadParamFile(os.path.join(dataPath, str(scan), "acqp"))
            method = ReadParamFile(os.path.join(dataPath, str(scan), "method"))
            if not UseShape(method["TestShape"], includeShapeTypes, excludeShapeTypes):
                continue

            samples = int(acqp["ACQ_jobs"][0][0] // 2 * ((1 / acqp["ACQ_jobs"][0][5]) / outRes))
            nSamples.append(samples)

            if np.array(method["LinearityGradAmps"]).any() == 0.0:
                batchDim += method["LinearitySteps"] - 1
                continue

            batchDim += method["LinearitySteps"]

    if len(nSamples) == 0:
        raise ValueError("No scans matched the requested shape split")

    maxSize = np.minimum(dsetCrop, np.max(nSamples)) + prepend + shiftByNSamples + 1 + maxRandShift

    batchDim *= 8

    xData = torch.zeros((batchDim, 3, maxSize))
    yData = torch.zeros((batchDim, 1, maxSize))
    yTrajData = torch.zeros((batchDim, 1, maxSize))

    mask = torch.ones_like(yData)

    indBatch = 0
    acqp = ReadParamFile(os.path.join(dataPaths[0], str(scansS[0][0]), "acqp"))

    nSlices = acqp["ACQ_GradientMatrix"].shape[0] // 3

    gradMatrix = acqp["ACQ_GradientMatrix"][::nSlices]

    sliceVec = np.expand_dims(np.array([0, 0, 1]), (0, 1))

    gradAmp = sliceVec @ gradMatrix

    labels = []
    randGen = np.random.default_rng(41)
    for scanind, dataPath in enumerate(dataPaths):
        scans = scansS[scanind]
        for scan in scans:
            _DatasetProgress(progressLabel, f"Loading scan {scan} in {dataPath}")
            method = ReadParamFile(os.path.join(dataPath, str(scan), "method"))
            acqp = ReadParamFile(os.path.join(dataPath, str(scan), "acqp"))
            testShapeName = method["TestShape"]
            if not UseShape(testShapeName, includeShapeTypes, excludeShapeTypes):
                continue

            if not np.allclose(gradMatrix[0], acqp["ACQ_GradientMatrix"][0]):
                raise Exception("Dif orient")

            polynomials, testShape, envelope = getData(
                dataPath + "/" + str(scan) + "/", outRes=outRes, frequencyFilter=filterFreq, gradientFirst=gradientFirst
            )
            _DatasetProgress(progressLabel, f"Loaded scan {scan}; shape={testShapeName}, current samples={indBatch}")

            polynomials = polynomials[0, ...] if doB0 else polynomials[-1, ...]

            if not doTraj:
                # pad = 32
                # origShape = polynomials.shape
                # pad_width = [(0, 0)] * polynomials.ndim
                # pad_width[axis] = (0, pad)
                # polynomialsPad = np.pad(polynomials, pad_width, mode="edge")
                # polynomialsUp = oversample(polynomialsPad, oversamplingFactor, axis=axis)
                # gradArr = np.gradient(polynomialsUp, outRes / oversamplingFactor, axis=1)

                # gradArr = reduce_oversampling(gradArr, oversamplingFactor, axis=axis)

                # gradArr = gradArr[:, : origShape[axis], :]
                if not gradientFirst:  # noqa: SIM108
                    # Use a centered derivative so targets stay aligned to the same
                    # sample-time convention as the trajectory grid.
                    gradArr = np.diff(polynomials, prepend=0, axis=1) / outRes
                    # gradArr = np.gradient(polynomials, outRes, axis=1)
                else:
                    gradArr = polynomials
            else:
                gradArr = polynomials if not gradientFirst else np.cumsum(gradArr, axis=1) * outRes

            gradCalConst = method["PVM_GradCalConst"]

            testShapeRelativeAmp = method["TestShapeVec"]
            linearityScaling = method["LinearityGradAmps"]

            if scanind == 2 and scan == 4:
                pass

            scale = np.max(np.abs(testShapeRelativeAmp))
            testShapeRelativeAmp /= scale
            inputShape, inputGradient = _build_input_channels(testShapeRelativeAmp)

            if np.abs(np.max(np.abs(testShapeRelativeAmp)) - 1) > 1e-8:
                raise Exception("scale err")

            if np.sum(np.abs(testShapeRelativeAmp)) == 0:
                raise Exception("Empty")

            if np.any(np.abs(np.diff(testShape)) > 14000):
                raise Exception("Shape Err")

            for step in range(linearityScaling.size):
                if linearityScaling[step] == 0.0:
                    continue

                amplitude = linearityScaling[step] * method.get("ChirpAmplitude", method.get("TestShapeAmplitude", None)) * 1e-2 * scale
                if np.abs(amplitude) < filterLowAmp:
                    continue

                if np.abs(amplitude) > 1:
                    raise Exception("Amplitude error")

                inpSize = np.minimum(testShapeRelativeAmp.size, dsetCrop)

                nRepeats = 1

                if testShapeName in {"READOUT", "MGE"}:
                    nRepeats = 1

                if absoluteMapping:
                    xData[indBatch, 1, :] = torch.abs(torch.tensor(amplitude))
                else:
                    xData[indBatch, 1, :] = amplitude

                axisPolarity = np.sum(gradAmp[gradAxisInd])
                tmpY = (
                    gradArr[step, :inpSize, gradAxisInd]
                    / gradCalConst
                    / 2
                    / np.pi
                    / method.get("ChirpAmplitude", method.get("TestShapeAmplitude", None))
                    * 1e2
                    / (linearityScaling[step])
                ) / scale  # to integral...

                tmpYF = torch.from_numpy(tmpY * axisPolarity)

                tmpTraj = (
                    polynomials[step, :inpSize, gradAxisInd]
                    / gradCalConst
                    / 2
                    / np.pi
                    / method.get("ChirpAmplitude", method.get("TestShapeAmplitude", None))
                    * 1e2
                    / (linearityScaling[step])
                ) / scale  # to integral...

                tmpTraj = torch.from_numpy(tmpTraj * axisPolarity)

                if estimateNoise:
                    noiseStart = np.var(tmpY[:80])
                    wStart = 1 / noiseStart

                for _ in range(nRepeats):
                    randShift = randGen.integers(minRandShift, maxRandShift)

                    rep = 0
                    if repeatShorter:
                        for rep in range(int((xData.shape[-1]) / (inpSize + gapSize))):
                            indStart = randShift + rep * (inpSize + gapSize)

                            indLastSampl = randShift + (rep - 1) * (inpSize + gapSize) + inpSize

                            indEnd = indStart + inpSize
                            indYEnd = indEnd + shiftByNSamples

                            if (indEnd + shiftByNSamples) > maxSize:
                                croppedSize = np.minimum(maxSize - indStart, testShapeRelativeAmp.size) - shiftByNSamples
                                xData[indBatch, 0, indStart : (indStart + croppedSize)] = torch.from_numpy(inputShape[:croppedSize] * axisPolarity)
                                xData[indBatch, 2, indStart : (indStart + croppedSize)] = torch.from_numpy(inputGradient[:croppedSize] * axisPolarity)
                                yData[
                                    indBatch,
                                    0,
                                    (indStart + shiftByNSamples + gradShift) : (indStart + croppedSize + shiftByNSamples + gradShift),
                                ] = tmpYF[:croppedSize]

                                lastTrajVal = yTrajData[indBatch, 0, indLastSampl]
                                yTrajData[indBatch, 0, indLastSampl:indStart] = lastTrajVal

                                yTrajData[
                                    indBatch,
                                    0,
                                    (indStart + shiftByNSamples) : (indStart + croppedSize + shiftByNSamples),
                                ] = tmpTraj[:croppedSize] + lastTrajVal

                                if estimateNoise:
                                    mask[indBatch, 0, (indStart + shiftByNSamples) : (indStart + croppedSize + shiftByNSamples)] = torch.from_numpy(
                                        wStart * envelope[step, gradAxisInd, :croppedSize] ** 2
                                    )
                                continue

                            xData[indBatch, 0, indStart:indEnd] = torch.from_numpy(inputShape[:inpSize] * axisPolarity)
                            xData[indBatch, 2, indStart:indEnd] = torch.from_numpy(inputGradient[:inpSize] * axisPolarity)
                            yData[
                                indBatch,
                                0,
                                (indStart + shiftByNSamples + gradShift) : (indEnd + shiftByNSamples + gradShift),
                            ] = tmpYF

                            lastTrajVal = yTrajData[indBatch, 0, indLastSampl]
                            yTrajData[indBatch, 0, indLastSampl:indStart] = lastTrajVal

                            yTrajData[
                                indBatch,
                                0,
                                (indStart + shiftByNSamples) : (indEnd + shiftByNSamples),
                            ] = tmpTraj + lastTrajVal

                            if estimateNoise:
                                mask[indBatch, 0, (indStart + shiftByNSamples) : (indEnd + shiftByNSamples)] = torch.from_numpy(
                                    wStart * envelope[step, gradAxisInd, :] ** 2
                                )

                    if rep == 0:
                        indYEnd = inpSize + randShift
                        if estimateNoise:
                            mask[indBatch, 0, : randShift + shiftByNSamples] = wStart * envelope[step, gradAxisInd, 0] ** 2

                        xData[indBatch, 0, randShift : inpSize + randShift] = torch.from_numpy(inputShape[:inpSize] * axisPolarity)
                        xData[indBatch, 2, randShift : inpSize + randShift] = torch.from_numpy(inputGradient[:inpSize] * axisPolarity)
                        yData[indBatch, 0, randShift + shiftByNSamples + gradShift : indYEnd + gradShift] = tmpYF[: (inpSize - shiftByNSamples)]
                        yTrajData[indBatch, 0, randShift + shiftByNSamples : indYEnd] = tmpTraj[: (inpSize - shiftByNSamples)]
                        if estimateNoise:
                            mask[indBatch, 0, randShift + shiftByNSamples : indYEnd] = torch.from_numpy(
                                wStart * envelope[step, gradAxisInd, : (inpSize - shiftByNSamples)] ** 2
                            )

                    if testShapeName == "READOUT" and step == 0:
                        pass

                    if torch.sum(torch.abs(xData[indBatch, 0, :])) == 0:
                        raise Exception("Empty")

                    mask[indBatch, 0, indYEnd - 5 :] = 0
                    # mask[indBatch,0,:randShift]=0

                    # xData[indBatch, 1,:]*=(mask[indBatch,0,:]>0)

                    yTrajData[indBatch, 0, indYEnd:] = yTrajData[indBatch, 0, indYEnd]

                    indBatch += 1
                    labels.append(testShapeName)

                    if onlyFirstLinStep:
                        break
                if onlyFirstLinStep:
                    break

    if doTraj:
        xData[:, 0, :] = torch.cumsum(xData[:, 0, :], axis=-1) * outRes
        xData *= 3600
        yData *= 3600
        yTrajData *= 3600

    mask[:indBatch] = mask[:indBatch] / torch.max(mask[:indBatch])

    # nZeros = 500
    # indBatch += nZeros
    # mask[indBatch - nZeros : indBatch] = 1
    # mask[:indBatch, 0, :minRandShift] = 0
    trainingSet = xData[:indBatch], yData[:indBatch], yTrajData[:indBatch], mask[:indBatch], labels
    if useCache:
        _SaveCachedTrainingSet(cachePath, cachePayload, *trainingSet)
        _DatasetProgress(progressLabel, f"Cache saved with {indBatch} samples")

    return trainingSet
