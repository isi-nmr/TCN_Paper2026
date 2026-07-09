import torch

from nn_models.TCN import TCN, OriginalTCNSequencePredictor, TCNFull, TCNFullSkip


def makeCurrentModelMetadata(
    axisSymbol,
    model,
    modelConfig,
    *,
    gradAxis,
    doB0,
    trainTraj,
    absoluteMapping,
):
    return {
        "formatVersion": 1,
        "modelFamily": "current",
        "axisSymbol": axisSymbol,
        "gradAxis": gradAxis,
        "doB0": doB0,
        "architecture": {
            "className": model.model_name,
            "inputSize": 3,
            "numChannels": list(model.num_channels),
            "nLayers": model.nLayers,
            "kernelSize": model.kernel_size,
            "dropout": modelConfig.get("dropout", 0.15),
            "skipSamples": model.skipSamples,
        },
        "runtime": {
            "modelType": modelConfig.get("model"),
            "shiftByNSamples": modelConfig.get("shiftByNSamples", model.skipSamples),
            "filterFreq": modelConfig.get("filterFreq"),
            "outRes": modelConfig.get("outRes", 2e-6),
            "integrationMethod": modelConfig.get("integrationMethod", modelConfig.get("lossIntegrationMethod", "cumsum")),
            "lossIntegrationMethod": modelConfig.get("lossIntegrationMethod", "trapz"),
            "trainTraj": trainTraj,
            "absoluteMapping": absoluteMapping,
        },
        "training": {
            "batchSize": modelConfig.get("batchSize"),
            "nEpoch": modelConfig.get("nEpoch"),
            "lr": modelConfig.get("lr"),
            "weightDecay": modelConfig.get("weightDecay"),
            "weightTraj": modelConfig.get("weightTraj", False),
            "gradTrajWeight_alpha": modelConfig.get("gradTrajWeight_alpha", 0.1),
            "earlyStopping": modelConfig.get("earlyStopping"),
        },
    }


def makeOriginalModelMetadata(axisSymbol, originalConfig, modelConfig):
    nChannels = originalConfig.get("nChannels", 48)
    nLayers = originalConfig.get("nLayers", 5)
    kernelSize = originalConfig.get("kernelSize", 16)
    return {
        "formatVersion": 1,
        "modelFamily": "original",
        "axisSymbol": axisSymbol,
        "gradAxis": axisSymbol,
        "doB0": False,
        "architecture": {
            "className": "OriginalTCNSequencePredictor",
            "inputSize": 2,
            "outputSize": 1,
            "numChannels": [nChannels] * nLayers,
            "nLayers": nLayers,
            "kernelSize": kernelSize,
            "dropout": originalConfig.get("dropout", 0.002),
            "windowSize": originalConfig.get("windowSize", 75),
            "predictPoint": originalConfig.get("predictPoint", 65),
        },
        "runtime": {
            "filterFreq": modelConfig.get("filterFreq"),
            "outRes": modelConfig.get("outRes", 2e-6),
            "shiftByNSamples": 0,
        },
        "training": {
            "batchSize": originalConfig.get("batchSize"),
            "nEpoch": originalConfig.get("nEpoch"),
            "lr": originalConfig.get("lr"),
            "weightDecay": originalConfig.get("weightDecay"),
            "eps": originalConfig.get("eps"),
            "beta1": originalConfig.get("beta1"),
            "beta2": originalConfig.get("beta2"),
            "earlyStopping": originalConfig.get("earlyStopping"),
        },
    }


def loadCheckpoint(path):
    return torch.load(path, map_location="cpu")


def cleanStateDict(stateDict):
    cleanState = {}
    for key, value in stateDict.items():
        cleanKey = key[len("_orig_mod.") :] if key.startswith("_orig_mod.") else key
        cleanState[cleanKey] = value
    return cleanState


def getMetadata(checkpoint):
    metadata = checkpoint.get("metadata")
    return metadata if isinstance(metadata, dict) else None


def buildCurrentModelFromMetadata(metadata):
    architecture = metadata["architecture"]
    className = architecture["className"]
    inputSize = architecture.get("inputSize", 3)
    numChannels = architecture["numChannels"]
    kernelSize = architecture["kernelSize"]
    dropout = architecture.get("dropout", 0.15)
    skipSamples = architecture.get("skipSamples", metadata.get("runtime", {}).get("shiftByNSamples", 0))

    if className == "TCNFullSkip":
        return TCNFullSkip(inputSize, numChannels, kernelSize, dropout, skipSamples)
    if className == "TCNFull":
        return TCNFull(inputSize, numChannels, kernelSize, dropout, skipSamples)

    raise ValueError(f"Unsupported current model class: {className}")


def buildOriginalModelFromMetadata(metadata):
    architecture = metadata["architecture"]
    channels = architecture["numChannels"]
    coreModel = TCN(
        architecture.get("inputSize", 2),
        architecture.get("outputSize", 1),
        channels,
        architecture["kernelSize"],
        architecture.get("dropout", 0.002),
    )
    return OriginalTCNSequencePredictor(
        coreModel,
        window_size=architecture.get("windowSize", 75),
        predict_point=architecture.get("predictPoint", 65),
    )
