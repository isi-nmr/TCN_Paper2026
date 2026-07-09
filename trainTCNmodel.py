import os

import torch
from torch.utils.data import TensorDataset

from nn_models.dataset import createTrainingSet
from nn_models.TCN import TCNFull, TCNFullSkip
from nn_models.training import train
from utils.modelMetadata import makeCurrentModelMetadata
from utils.utils import LoadTestingShapeTypes, LoadTrainingDataConfig, load_config

if __name__ == "__main__":
    fullConfig = load_config()
    dataPaths, scansS = LoadTrainingDataConfig(fullConfig)
    testingShapeTypes = LoadTestingShapeTypes(fullConfig)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    config = fullConfig["model"]
    modelType = config["model"]
    nChannels = config["nChannels"]
    nLayers = config["nLayers"]
    kernel = config["kernelSize"]
    shiftByNSamples = config["shiftByNSamples"]
    filterFreq = config["filterFreq"]
    nEpoch = config["nEpoch"]
    lr = config["lr"]
    weight_decay = config["weightDecay"]
    torchRes = config["outRes"]
    batchSize = config["batchSize"]
    dropOut = config["dropout"]
    lossAlpha = config.get("gradTrajWeight_alpha", 0.1)
    weightTraj = config.get("weightTraj", False)
    lossIntegrationMethod = config.get("lossIntegrationMethod", "trapz")
    earlyStopping = config.get("earlyStopping", {})

    trainTraj = False
    absoluteMapping = False

    continueTraining = False

    os.makedirs("utils/gradModels/", exist_ok=True)

    gradAxes = ["Z", "X", "Y"]
    traingradAxes = ["ZB0"]
    # traingradAxes = ["XB0", "YB0"]
    traingradAxes = ["X", "Y", "Z", "XB0", "YB0", "ZB0"]
    # traingradAxes = ["XB0", "YB0", "ZB0"]
    traingradAxes = ["X", "Y", "Z", "XB0", "YB0", "ZB0"]
    # traingradAxes = ["X", "Y"]
    for axisSymbol in traingradAxes:
        gradAxis = axisSymbol.rstrip("B0")

        doB0 = gradAxis != axisSymbol

        gradAxisInd = gradAxes.index(gradAxis)

        print("Training Axis " + axisSymbol)
        xData, yData, yTraj, mask, _ = createTrainingSet(
            dataPaths,
            scansS,
            gradAxisInd,
            doB0=doB0,
            doTraj=trainTraj,
            absoluteMapping=absoluteMapping,
            shiftByNSamples=shiftByNSamples,
            filterFreq=filterFreq,
            outRes=torchRes,
            estimateNoise=True,
            filterLowAmp=0,
            excludeShapeTypes=testingShapeTypes,
        )
        xData = xData.to(device)
        yData = yData.to(device)
        yTraj = yTraj.to(device)
        mask = mask.to(device)
        dataset = TensorDataset(xData, yData, yTraj, mask)
        if modelType == "TCNSkip" and not doB0:
            model = TCNFullSkip(3, [nChannels] * nLayers, kernel, dropOut, shiftByNSamples)
        else:
            model = TCNFull(3, [nChannels] * nLayers, kernel, dropOut, shiftByNSamples)

        skipStr = "_skip" if model.model_name == "TCNFullSkip" else "_"
        outPth = "utils/gradModels/grad" + axisSymbol + skipStr + f"_{model.num_channels}_{model.nLayers}_{model.kernel_size}"

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        if continueTraining and os.path.exists(outPth):
            print(f"Continuning training of {outPth}")
            checkpoint = torch.load(outPth)

            state_dict = checkpoint["model_state_dict"]

            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("_orig_mod."):
                    k = k[len("_orig_mod.") :]  # noqa: PLW2901
                new_state_dict[k] = v

            model.load_state_dict(new_state_dict)

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            state = checkpoint["optimizer_state_dict"]
            state["param_groups"][0]["weight_decay"] = weight_decay
            state["param_groups"][0]["lr"] = lr
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)

        model = torch.compile(model)
        model.to(device)
        train(
            dataset,
            model,
            optimizer,
            outPth=outPth,
            plotting=True,
            gradAxis=gradAxis,
            doB0=doB0,
            trainTraj=trainTraj,
            nEpoch=nEpoch,
            batchSize=batchSize,
            lossAlpha=lossAlpha,
            weightLossByAmplitude=weightTraj,
            outRes=torchRes,
            lossIntegrationMethod=lossIntegrationMethod,
            earlyStopping=earlyStopping,
            checkpointMetadata=makeCurrentModelMetadata(
                axisSymbol,
                model._orig_mod if hasattr(model, "_orig_mod") else model,
                config,
                gradAxis=gradAxis,
                doB0=doB0,
                trainTraj=trainTraj,
                absoluteMapping=absoluteMapping,
            ),
        )
