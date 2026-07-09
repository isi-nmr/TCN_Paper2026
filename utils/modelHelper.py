import json
import os

from nn_models.TCN import TCNFull, TCNFullSkip
from utils.modelMetadata import buildCurrentModelFromMetadata, cleanStateDict, getMetadata, loadCheckpoint


def loadModel(config, axisSymbol):
    outPth = None
    if config is not None:
        nChannels = config["nChannels"]
        nLayers = config["nLayers"]
        kernel = config["kernelSize"]
        shiftByNSamples = config["shiftByNSamples"]

        if "B0" not in axisSymbol:
            model = TCNFullSkip(3, [nChannels] * nLayers, kernel, 0.15, shiftByNSamples)
        else:
            model = TCNFull(3, [nChannels] * nLayers, kernel, 0.15, shiftByNSamples)

        skipStr = "_skip" if model.model_name == "TCNFullSkip" else "_"
        outPth = "utils/gradModels/grad" + axisSymbol + skipStr + f"_{model.num_channels}_{model.nLayers}_{model.kernel_size}"

    if outPth is None or not os.path.exists(outPth):
        return None

    checkpoint = loadCheckpoint(outPth)
    metadata = getMetadata(checkpoint)
    if metadata is not None and metadata.get("modelFamily") == "current":
        model = buildCurrentModelFromMetadata(metadata)
    elif config is None:
        return None

    new_state_dict = cleanStateDict(checkpoint["model_state_dict"])
    model.load_state_dict(new_state_dict)
    model.eval()

    return model


def getModels():
    config = None
    if os.path.exists("config.json"):
        with open("config.json") as f:
            config = json.load(f)["model"]

    modelx = loadModel(config, "X")
    modely = loadModel(config, "Y")
    modelxb0 = loadModel(config, "XB0")
    modelyb0 = loadModel(config, "YB0")

    return modelx, modely, modelxb0, modelyb0
