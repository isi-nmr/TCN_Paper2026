import copy
import json
import os

import numpy as np
import torch
from scipy.interpolate import CubicSpline, PchipInterpolator

from nn_models.TCN import TCN, OriginalTCNSequencePredictor, TCNFull, TCNFullSkip
from utils.modelMetadata import (
    buildCurrentModelFromMetadata,
    buildOriginalModelFromMetadata,
    cleanStateDict,
    getMetadata,
    loadCheckpoint,
)
from utils.utils import get_acquisition_dwell


class GradientCorectorML:
    XGradModel = None
    YGradModel = None
    ZGradModel = None

    XB0Model = None
    YB0Model = None
    ZB0Model = None
    modelRes = 2e-6
    integrationMethod = "cumsum"

    def __init__(self, system="BrukerUPT", *, load=True, model_family="current"):
        self.model_family = model_family
        self.loadSystem()

    def loadModel(self, config, axisSymbol):
        if self.model_family == "original":
            return self.loadOriginalModel(config, axisSymbol)

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
            outPth = os.path.join(
                "utils",
                "gradModels",
                f"grad{axisSymbol}{skipStr}_{model.num_channels}_{model.nLayers}_{model.kernel_size}",
            )

        if outPth is None or not os.path.exists(outPth):
            print(f"Model for axis {axisSymbol} is not present, will not be able to use it")
            return None
        try:
            checkpoint = loadCheckpoint(outPth)
        except Exception:
            print(f"Model for axis {axisSymbol} is not present, will not be able to use it")
            return None

        metadata = getMetadata(checkpoint)
        if metadata is not None and metadata.get("modelFamily") == "current":
            model = buildCurrentModelFromMetadata(metadata)
            self.applyModelMetadata(metadata)
        elif config is None:
            print(f"Model for axis {axisSymbol} has no metadata and config.json is not available")
            return None

        new_state_dict = cleanStateDict(checkpoint["model_state_dict"])
        model.load_state_dict(new_state_dict)
        model.eval()

        return model

    def loadOriginalModel(self, config, axisSymbol):
        if "B0" in axisSymbol:
            return None

        outPth = None
        if config is not None:
            original_config = self.full_config.get("originalModel", {})
            nChannels = original_config.get("nChannels", 48)
            nLayers = original_config.get("nLayers", 5)
            kernel = original_config.get("kernelSize", 16)
            dropout = original_config.get("dropout", 0.002)
            window_size = original_config.get("windowSize", 75)
            predict_point = original_config.get("predictPoint", 65)
            channels = [nChannels] * nLayers

            core_model = TCN(2, 1, channels, kernel, dropout)
            model = OriginalTCNSequencePredictor(core_model, window_size=window_size, predict_point=predict_point)

            outPth = os.path.join(
                "utils",
                "gradModels",
                f"originalTCN_grad{axisSymbol}_{channels}_{nLayers}_{kernel}_{window_size}_{predict_point}",
            )

        if outPth is None or not os.path.exists(outPth):
            print(f"Original TCN model for axis {axisSymbol} is not present, will not be able to use it")
            return None
        try:
            checkpoint = loadCheckpoint(outPth)
        except Exception:
            print(f"Original TCN model for axis {axisSymbol} is not present, will not be able to use it")
            return None

        metadata = getMetadata(checkpoint)
        if metadata is not None and metadata.get("modelFamily") == "original":
            model = buildOriginalModelFromMetadata(metadata)
            self.applyModelMetadata(metadata)
        elif config is None:
            print(f"Original TCN model for axis {axisSymbol} has no metadata and config.json is not available")
            return None

        new_state_dict = cleanStateDict(checkpoint["model_state_dict"])
        if metadata is not None and metadata.get("modelFamily") == "original":
            model.model.load_state_dict(new_state_dict)
        else:
            core_model.load_state_dict(new_state_dict)
        model.eval()

        return model

    def applyModelMetadata(self, metadata):
        runtime = metadata.get("runtime", {})
        self.modelRes = runtime.get("outRes", self.modelRes)
        self.integrationMethod = runtime.get("integrationMethod", self.integrationMethod)

    def loadSystem(self):
        if os.path.exists("config.json"):
            with open("config.json") as f:
                self.full_config = json.load(f)
            config = self.full_config["model"]
            self.modelRes = config.get("outRes", 2e-6)
            self.integrationMethod = config.get("integrationMethod", config.get("lossIntegrationMethod", "cumsum"))
        else:
            self.full_config = {}
            config = None

        self.XGradModel = self.loadModel(config, "X")
        self.YGradModel = self.loadModel(config, "Y")
        self.ZGradModel = self.loadModel(config, "Z")

        self.XB0Model = self.loadModel(config, "XB0")
        self.YB0Model = self.loadModel(config, "YB0")
        self.ZB0Model = self.loadModel(config, "ZB0")

    def hasGradientModels(self):
        return self.XGradModel is not None and self.YGradModel is not None and self.ZGradModel is not None

    def convertSpatialCoord(self, Ax1, Ax2, Ax3, transMatrix):
        """Convertes between two spatial systems by rotating and swapping axes

        Args:
            Ax1 (_type_): Input factors of first axis single value or 1D vector e.g. X or R
            Ax2 (_type_): Input factors of first axis single value or 1D vector e.g. Y or P
            Ax3 (_type_): Input factors of first axis single value or 1D vector e.g. Z or S
            transMatrix (_type_): Transformation matrix to transform between two coord systems

        Returns:
            _type_: Returns tranformed coordinates
        """
        Ax1 = np.expand_dims(Ax1, 0)
        Ax2 = np.expand_dims(Ax2, 0)
        Ax3 = np.expand_dims(Ax3, 0)

        inputS = np.concatenate((Ax1, Ax2, Ax3), axis=0)

        if len(inputS.shape) == 1:
            inputS = np.expand_dims(inputS, 0)
            tmp2 = np.matmul(inputS, transMatrix)
            return tmp2[:, 0], tmp2[:, 1], tmp2[:, 2]
        inputS = np.transpose(inputS)

        tmp2 = np.zeros_like(inputS)

        # for i in range(inputS.shape[0]):

        tmp2 = np.matmul(inputS, transMatrix)

        return tmp2[:, 0], tmp2[:, 1], tmp2[:, 2]

    def getgradsInTorchFormat(self, gradShape, gradAmps, gradRes, *, prepend=20, torchRes=None):
        if torchRes is None:
            torchRes = self.modelRes
        nSamplesTorch = int(np.ceil(gradShape.size * gradRes / torchRes))

        tAxisGrad = np.arange(gradShape.size) * gradRes
        tAxisTorch = np.arange(nSamplesTorch) * torchRes
        gradResample = np.interp(tAxisTorch, tAxisGrad, gradShape) if gradRes != torchRes else gradShape

        gradScaler = np.max(np.abs(gradShape))
        if np.abs(gradScaler) < 1e-12:
            raise ValueError("Input gradient shape has zero scale")

        gradResample = (gradResample / gradScaler).astype(np.float32, copy=False)
        gradDiff = np.gradient(gradResample).astype(np.float32, copy=False) * 15

        gradOut = np.zeros((len(gradAmps), 3, len(gradResample) + prepend), dtype=np.float32)

        for i in range(len(gradAmps)):
            Gr = gradScaler * gradAmps[i]
            gradOut[i, 0, prepend:] = gradResample  # * np.sign(Gr)
            gradOut[i, 2, prepend:] = gradDiff

            gradOut[i, 1, :] = Gr  # np.abs(Gr)

        return gradOut

    def cumulativeIntegrate(self, y, dt, method=None, axis=1):
        if method is None:
            method = self.integrationMethod

        if method == "cumsum":
            return np.cumsum(y, axis=axis) * dt

        if method == "trapz":
            out = np.zeros_like(y)
            slicer_prev = [slice(None)] * y.ndim
            slicer_next = [slice(None)] * y.ndim
            slicer_out = [slice(None)] * y.ndim
            slicer_prev[axis] = slice(0, -1)
            slicer_next[axis] = slice(1, None)
            slicer_out[axis] = slice(1, None)
            avg = 0.5 * (y[tuple(slicer_prev)] + y[tuple(slicer_next)])
            out[tuple(slicer_out)] = np.cumsum(avg, axis=axis) * dt
            return out

        if method == "simpson":
            y_move = np.moveaxis(y, axis, 1)
            out = np.zeros_like(y_move)
            n = y_move.shape[1]
            if n >= 2:
                out[:, 1, :] = 0.5 * (y_move[:, 0, :] + y_move[:, 1, :]) * dt
            for i in range(2, n):
                if i % 2 == 0:
                    out[:, i, :] = out[:, i - 2, :] + (dt / 3.0) * (y_move[:, i - 2, :] + 4.0 * y_move[:, i - 1, :] + y_move[:, i, :])
                else:
                    out[:, i, :] = out[:, i - 1, :] + 0.5 * (y_move[:, i - 1, :] + y_move[:, i, :]) * dt
            return np.moveaxis(out, 1, axis)

        if method in {"pchip", "cubic"}:
            axis = axis % y.ndim
            t = np.arange(y.shape[axis]) * dt
            y_move = np.moveaxis(y, axis, 0)
            y_flat = y_move.reshape(y_move.shape[0], -1)
            out = np.zeros_like(y_flat)

            for col in range(y_flat.shape[1]):
                if method == "pchip":
                    antider = PchipInterpolator(t, y_flat[:, col]).antiderivative()
                else:
                    antider = CubicSpline(t, y_flat[:, col]).antiderivative()
                out[:, col] = antider(t) - antider(0.0)

            out = out.reshape(y_move.shape)
            return np.moveaxis(out, 0, axis)

        raise ValueError(f"Unsupported integration method: {method}")

    def generateTrajectory(
        self,
        RShape,
        PShape,
        SShape,
        RFactor,
        PFactor,
        SFactor,
        gradRes,
        trajRes,
        transformMatrix,
        acqStartTime,
        acqStopTime,
        gradCalConst,
        FOV,
        SamplingMatrix,
    ):
        """Function generating trajectories based on input shapes and their scaling

        Args:
            RShape (_type_): Read shape in % of true gradient <-100:100> %
            PShape (_type_): Phase shape in % of true gradient <-100:100> %
            SShape (_type_): Slice shape in % of true gradient <-100:100> %
            RFactor (_type_): Read encoding mult factor in range of <-1:1> 1D array or a list
            PFactor (_type_): Phase encoding mult factor in range of <-1:1> 1D array or a list
            SFactor (_type_): Slice encoding  mult factor in range of <-1:1> 1D array or a list
            gradRes (_type_): Gradient shape resolution in seconds
            trajRes (_type_): Resulting trajectory resolution in seconds (equals 1/RBW)
            transformMatrix (_type_): Transformation matrix obtained by multiplying gradient transform matrix with results of getPoisitionMatrix
            (order matters) GradM @ PosM
            acqStartTime (_type_): Start of acquisition - generally 0 in seconds
            acqStopTime (_type_): Acq stop time - generally 1/RBW*NSamples in seconds
            gradCalConst (_type_): gradient Calibration Constant in Hz/mm
            FOV (_type_): Field of view in mm 1D array with 3 entries for FOV in Read, Phase and Slice coord -- e.g. PVM_Fov
            SamplingMatrix (_type_): Sampling matrix 1D array of resulting image resolution (resolution in pixels in Read, Phase Slice) -- e.g.
            PVM_Matrix
            initDelay (int, optional): Initial delay required in some cases where gradients are laging behind acquisition commands, in seconds.
            Defaults to 0.
            theoretical (bool, optional): Use of theoretical gradient shapes in producing trajectory. Defaults to False.

        Returns:
            _type_: Resulting trajectory in rad/px <-pi:pi> and B0 correction in rad
            To apply B0 correction multiply measured data with exp(-1j*B0corr)
            When theoretical option is selected B0corr is zero (ideal response)

        """
        prepend = 20
        # get into -1:1 range
        RFactor = np.asarray(RFactor) * 1e-2
        PFactor = np.asarray(PFactor) * 1e-2
        SFactor = np.asarray(SFactor) * 1e-2

        if not (len(RFactor) == len(PFactor) == len(SFactor)):
            raise ValueError("Encoding factors must have same length")

        XR, YR, ZR = self.convertSpatialCoord(1, 0, 0, transformMatrix)
        XP, YP, ZP = self.convertSpatialCoord(0, 1, 0, transformMatrix)
        XS, YS, ZS = self.convertSpatialCoord(0, 0, 1, transformMatrix)

        factors = [RFactor, PFactor, SFactor]

        axesScale = [[XR, YR, ZR], [XP, YP, ZP], [XS, YS, ZS]]

        Gx = np.zeros((RFactor.size, RShape.size + prepend))
        Gy = np.zeros((RFactor.size, RShape.size + prepend))
        Gz = np.zeros((RFactor.size, RShape.size + prepend))

        Bx = np.zeros((RFactor.size, RShape.size + prepend))
        By = np.zeros((RFactor.size, RShape.size + prepend))
        Bz = np.zeros((RFactor.size, RShape.size + prepend))

        for ind, shape in enumerate([copy.deepcopy(RShape), copy.deepcopy(PShape), copy.deepcopy(SShape)]):
            axesInd = axesScale[ind]
            gradOut = self.getgradsInTorchFormat(shape, factors[ind], gradRes, prepend=prepend)

            with torch.no_grad():
                gradX = copy.deepcopy(gradOut)
                gradX[:, 1, :] *= np.abs(axesInd[0])
                gradX[:, 0, :] *= np.sign(axesInd[0])
                gradX[:, 2, :] *= np.sign(axesInd[0])

                gradY = copy.deepcopy(gradOut)
                gradY[:, 1, :] *= np.abs(axesInd[1])
                gradY[:, 0, :] *= np.sign(axesInd[1])
                gradY[:, 2, :] *= np.sign(axesInd[1])

                gradZ = copy.deepcopy(gradOut)
                gradZ[:, 1, :] *= np.abs(axesInd[2])
                gradZ[:, 0, :] *= np.sign(axesInd[2])
                gradZ[:, 2, :] *= np.sign(axesInd[2])

                if np.any(gradX[:, 1, :]):
                    gradXOut = self.XGradModel.forward(torch.from_numpy(gradX))[:, 0, :].cpu() * gradX[:, 1, :]
                    if self.XB0Model is not None:
                        bxOut = self.XB0Model.forward(torch.from_numpy(gradX))[:, 0, :].cpu() * gradX[:, 1, :]
                    else:
                        bxOut = torch.zeros_like(torch.from_numpy(gradX[:, 0, :]))
                else:
                    gradXOut = torch.zeros_like(torch.from_numpy(gradX[:, 0, :]))
                    bxOut = torch.zeros_like(torch.from_numpy(gradX[:, 0, :]))

                if np.any(gradY[:, 1, :]):
                    gradYOut = self.YGradModel.forward(torch.from_numpy(gradY))[:, 0, :].cpu() * gradY[:, 1, :]
                    if self.YB0Model is not None:
                        byOut = self.YB0Model.forward(torch.from_numpy(gradY))[:, 0, :].cpu() * gradY[:, 1, :]
                    else:
                        byOut = torch.zeros_like(torch.from_numpy(gradY[:, 0, :]))
                else:
                    gradYOut = torch.zeros_like(torch.from_numpy(gradY[:, 0, :]))
                    byOut = torch.zeros_like(torch.from_numpy(gradY[:, 0, :]))

                if np.any(gradZ[:, 1, :]):
                    gradZOut = self.ZGradModel.forward(torch.from_numpy(gradZ))[:, 0, :].cpu() * gradZ[:, 1, :]
                    if self.YB0Model is not None:
                        bzOut = self.ZB0Model.forward(torch.from_numpy(gradZ))[:, 0, :].cpu() * gradZ[:, 1, :]
                    else:
                        bzOut = torch.zeros_like(torch.from_numpy(gradZ[:, 0, :]))
                else:
                    gradZOut = torch.zeros_like(torch.from_numpy(gradZ[:, 0, :]))
                    bzOut = torch.zeros_like(torch.from_numpy(gradZ[:, 0, :]))

                Bx += bxOut.numpy()
                By += byOut.numpy()
                Bz += bzOut.numpy()

                Gx += gradXOut.numpy()
                Gy += gradYOut.numpy()
                Gz += gradZOut.numpy()

        Gxyz = np.stack((Gx, Gy, Gz), axis=-1)
        Bxyz = np.stack((Bx, By, Bz), axis=-1)

        Gxyz = Gxyz[:, prepend:, ...]
        Bxyz = Bxyz[:, prepend:, ...]

        modelRes = self.modelRes

        Bout = self.cumulativeIntegrate(np.sum(Bxyz, axis=-1), modelRes, axis=1) * gradCalConst * 2 * np.pi

        Grps = Gxyz @ np.linalg.inv(transformMatrix)

        Trps = self.cumulativeIntegrate(Grps, modelRes, axis=1) * gradCalConst

        skip_samples = self.XGradModel.skipSamples if self.XGradModel is not None else 0
        gradTAxis = np.arange(Grps.shape[1]) * modelRes - skip_samples * modelRes

        samplTrajTAxis = []

        for i in range(len(acqStartTime)):
            samplTrajTAxis.append(  # noqa: PERF401
                np.linspace(
                    acqStartTime[i],
                    acqStopTime[i] - trajRes,
                    int(np.round((acqStopTime[i] - acqStartTime[i]) / trajRes)),
                )
            )

        length = RFactor.size

        traj = np.zeros((len(samplTrajTAxis[0]), length, 3, len(samplTrajTAxis)))

        Bcorr = np.zeros((len(samplTrajTAxis[0]), length, len(samplTrajTAxis)))

        for acq in range(traj.shape[-1]):
            for interleave in range(traj.shape[1]):
                for ax in range(traj.shape[2]):
                    # tChip = PchipInterpolator(gradTAxis, Trps[interleave, :, ax])
                    # traj[:, interleave, ax, acq] = tChip(samplTrajTAxis[acq]) * FOV[ax] / SamplingMatrix[ax] / 0.5 * np.pi

                    traj[:, interleave, ax, acq] = (
                        np.interp(samplTrajTAxis[acq], gradTAxis, Trps[interleave, :, ax]) * FOV[ax] / SamplingMatrix[ax] / 0.5 * np.pi
                    )

                bChip = PchipInterpolator(gradTAxis, Bout[interleave, :])
                Bcorr[:, interleave, acq] = bChip(samplTrajTAxis[acq])

                # Bcorr[:, interleave, acq] = np.interp(samplTrajTAxis[acq], gradTAxis, Bout[interleave, :])

        return (traj, Bcorr, [])

    def generateCorrectionsRadialCS(self, Method_file, acqp):
        """_summary_

        Args:
            Method_file (dict): Bruker Method file of Mac_CS pulse sequence
            PatPos (string): Patient position (from acqp) string e.g. Head_Prone
            SubType (string): SUBJECT type (from subject file)
        Returns:
            _type_: Trajectories in rad/px <-pi:pi> and B0 field contaminations in rad
            To apply B0 correction multiply measured echoes with exp(-1j*B0corr)
        """

        gradRes = Method_file["GradRes"] / 1e3

        trajRes = get_acquisition_dwell(Method_file, acqp)

        acqLen = int(acqp["ACQ_jobs"][0][0] / 2)

        readGradShape = Method_file["RadRead_ReadGradShapeTraj"].copy()
        phaseGradShape = Method_file["RadRead_Ph3GradShapeTraj"].copy()
        MGEGradShape = Method_file["RadRead_MGEGradShape"].copy()

        ReadGrad = Method_file["RadRead_ReadGrad"]

        PhaseGrad = ReadGrad

        Phase3DGrad = Method_file["RadRead_Phase3DGrad"]

        RFactor = Method_file["RadRead_GradAmpR"] * ReadGrad
        PFactor = Method_file["RadRead_GradAmpP"] * PhaseGrad
        SFactor = -Method_file["RadRead_GradAmpS"] * Phase3DGrad

        gradMatrix = np.squeeze(acqp["ACQ_GradientMatrix"])

        if Method_file["PVM_Fov"].shape[0] == 2:
            FOV = Method_file["PVM_Fov"]
            FOV = np.append(FOV, Method_file["PVM_SliceThick"])
            SamplingMatrix = Method_file["PVM_Matrix"]
            SamplingMatrix = np.append(SamplingMatrix, 1)
        else:
            FOV = Method_file["PVM_Fov"]
            SamplingMatrix = Method_file["PVM_Matrix"]

        transformMatrix = gradMatrix

        nEchoes = Method_file["PVM_NEchoImages"]
        afterAcqTime = Method_file["RadRead_AfterAcqWaitTime"] / 1e3
        interEchoTime = Method_file["RadRead_EchoFillDelay"] / 1e3
        interEchoTimeApprox = np.round((afterAcqTime + interEchoTime) / gradRes) * gradRes
        filler = np.zeros(int(interEchoTimeApprox / gradRes))
        mgeFiller = np.zeros_like(MGEGradShape)

        acqStartTimes = []
        acqStopTimes = []
        acqStartTimes.append(0)
        acqStopTimes.append(acqLen * trajRes)
        if nEchoes > 1:
            for i in range(nEchoes - 1):
                readGradShape = np.concatenate((readGradShape, filler))
                readGradShape = np.concatenate((readGradShape, MGEGradShape))
                phaseGradShape = np.concatenate((phaseGradShape, filler))
                phaseGradShape = np.concatenate((phaseGradShape, mgeFiller))
                if i == 0:
                    acqStartTimes.append(acqStartTimes[-1] + gradRes * (len(Method_file["RadRead_ReadGradShape"]) + len(filler)))
                    acqStopTimes.append(acqStartTimes[-1] + acqLen * trajRes)
                else:
                    acqStartTimes.append(acqStartTimes[-1] + gradRes * (len(MGEGradShape) + len(filler)) - gradRes)
                    acqStopTimes.append(acqStartTimes[-1] + acqLen * trajRes)

        trajCorrected, BCorrection, _shapes = self.generateTrajectory(
            readGradShape,
            readGradShape,
            phaseGradShape,
            RFactor,
            PFactor,
            SFactor,
            gradRes,
            trajRes,
            transformMatrix,
            acqStartTimes,
            acqStopTimes,
            Method_file["PVM_GradCalConst"],
            FOV,
            SamplingMatrix,
        )

        return trajCorrected, BCorrection

    def convertTrajToBartScale(self, method, traj, oversampling=1):
        matrix = method["PVM_Matrix"]

        for i in range(len(matrix)):
            traj[:, :, i, :] = traj[:, :, i, :] / np.pi * matrix[i] / oversampling
        return traj
