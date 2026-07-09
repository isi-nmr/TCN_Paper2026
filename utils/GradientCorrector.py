import json
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal


class GradientCorector:
    systemInfo = {}
    system = ""

    def __init__(self, system="BrukerUPT", *, load=True):
        self.system = system
        if load:
            self.loadSystem()

        else:
            self.systemInfo["XGrad"] = {}
            self.systemInfo["YGrad"] = {}
            self.systemInfo["ZGrad"] = {}

    def loadSystem(self):
        dirPath = os.path.dirname(__file__)
        with open(dirPath + "/gradSystemInfo.json") as jsonFile:
            contents = json.load(jsonFile)
        self.systemInfo = contents[self.system]

    def checkOutput(self, Method, Shape, PatPos=None, SubType=None, acqp=None, gradPre=0):
        """Checks the output of the virtual system for experiment with measured Trajectory

        Args:
            Method (dict): Method file
            Shape (_type_): Test Shape
            PatPos (String): Patient Position (from acqp file)
            SubType (String): Subject Type (from subject file)
        """
        trajX = Method["PVM_TrajKx"]
        trajX = np.expand_dims(trajX, 0)
        trajY = Method["PVM_TrajKy"]
        trajY = np.expand_dims(trajY, 0)
        trajZ = Method["PVM_TrajKz"]
        trajZ = np.expand_dims(trajZ, 0)

        BX = Method["PVM_TrajBx"]
        BY = Method["PVM_TrajBy"]
        BZ = Method["PVM_TrajBz"]
        if PatPos is not None:
            gradMatrix = np.squeeze(Method["PVM_SPackArrGradOrient"])
            transformMatrix = gradMatrix @ self.getPoisitionMatrix(PatPos, SubType)
        else:
            transformMatrix = np.squeeze(acqp["ACQ_GradientMatrix"])
        if trajZ.size == 1:
            trajZ = np.zeros_like(trajX)

        trajShapes = np.concatenate((trajX, trajY, trajZ), axis=0)

        gradDwell = Method["PVM_TrajDwGrad"] / 1e3
        trajDwell = Method["PVM_TrajDwAcq"] / 1e3
        gradCalConst = Method["PVM_GradCalConst"]

        PVM_EffSWh = Method["PVM_EffSWh"]
        readFov = Method["PVM_Fov"][0]

        ReadGrad = self.calcReadGrad(PVM_EffSWh, readFov, gradCalConst)

        dirVec = np.array([ReadGrad, ReadGrad, 0])

        xyzVec = self.convertSpatialCoord(dirVec[0], dirVec[1], dirVec[2], transformMatrix)

        GxMeasured = np.diff(trajShapes[0, :]) / trajDwell / gradCalConst * 100
        GyMeasured = np.diff(trajShapes[1, :]) / trajDwell / gradCalConst * 100
        GzMeasured = np.diff(trajShapes[2, :]) / trajDwell / gradCalConst * 100

        GxMeasured = np.insert(GxMeasured, (0), 0)
        GyMeasured = np.insert(GyMeasured, (0), 0)
        GzMeasured = np.insert(GzMeasured, (0), 0)

        gradTAxis = np.linspace(0, gradDwell * len(Shape) - gradDwell, len(Shape))
        trajTAxis = np.linspace(0, trajDwell * len(GzMeasured) - trajDwell, len(GzMeasured))

        RShapeX, RBX = self.systemTransform(Shape * xyzVec[0], "XGrad", shapeRes=gradDwell, gradientPreDelay=gradPre)
        RShapeY, RBY = self.systemTransform(Shape * xyzVec[1], "YGrad", shapeRes=gradDwell, gradientPreDelay=gradPre)
        RShapeZ, RBZ = self.systemTransform(Shape * xyzVec[2], "ZGrad", shapeRes=gradDwell, gradientPreDelay=gradPre)

        RShapeX = np.interp(trajTAxis, gradTAxis, RShapeX)
        RShapeY = np.interp(trajTAxis, gradTAxis, RShapeY)
        RShapeZ = np.interp(trajTAxis, gradTAxis, RShapeZ)

        RBX = np.interp(trajTAxis, gradTAxis, RBX)
        RBY = np.interp(trajTAxis, gradTAxis, RBY)
        RBZ = np.interp(trajTAxis, gradTAxis, RBZ)

        plotEnd = 1500

        mX = np.mean(RShapeX[75:150])
        mY = np.mean(RShapeY[75:150])
        mZ = np.mean(RShapeZ[75:150])

        mmX = np.mean(GxMeasured[75:150])
        mmY = np.mean(GyMeasured[75:150])
        mmZ = np.mean(GzMeasured[75:150])

        fX = mX / mmX
        fY = mY / mmY
        fZ = mZ / mmZ

        trajTAxis = trajTAxis * 1e6
        gradTAxis = gradTAxis * 1e6
        fig, axs = plt.subplots(3, 1)

        axs[0].plot(trajTAxis, (GxMeasured * fX), label="Measured", color="g")
        axs[0].plot(trajTAxis, RShapeX, label="Estimated", color="r", linestyle="--")
        axs[0].plot(gradTAxis, (Shape) * xyzVec[0], label="Theoretical", color="b")
        axs[0].legend()
        axs[0].set_xlim(0, plotEnd)
        # axs[0].set_xlabel('t[us]')
        axs[0].set_ylabel("Gx[%]")

        axs[1].plot(trajTAxis, (GyMeasured * fY), label="Measured", color="g")
        axs[1].plot(trajTAxis, RShapeY, label="Estimated", color="r", linestyle="--")

        axs[1].plot(gradTAxis, (Shape) * xyzVec[1], label="Theoretical", color="b")
        axs[1].legend()
        axs[1].set_xlim(0, plotEnd)
        # axs[1].set_xlabel('t[us]')
        axs[1].set_ylabel("Gy[%]")

        axs[2].plot(trajTAxis, (GzMeasured * fZ), label="Measured", color="g")
        axs[2].plot(trajTAxis, RShapeZ, label="Estimated", color="r", linestyle="--")
        axs[2].plot(gradTAxis, (Shape) * xyzVec[2], label="Theoretical", color="b")
        axs[2].legend()
        axs[2].set_xlim(0, plotEnd)
        axs[2].set_xlabel("t[us]")
        axs[2].set_ylabel("Gz[%]")

        plt.show()

        fig, axs = plt.subplots(3, 1)

        axs[0].plot(trajTAxis, (np.cumsum(GxMeasured * fX)), label="Measured", color="g")
        axs[0].plot(trajTAxis, np.cumsum(RShapeX), label="Estimated", color="r", linestyle="--")
        # axs[0].plot(gradTAxis, np.cumsum((Shape) * xyzVec[0]), label="Theoretical", color="b")
        axs[0].legend()
        axs[0].set_xlim(0, plotEnd)
        # axs[0].set_xlabel('t[us]')
        axs[0].set_ylabel("Tx[%]")

        axs[1].plot(trajTAxis, np.cumsum(GyMeasured * fY), label="Measured", color="g")
        axs[1].plot(trajTAxis, np.cumsum(RShapeY), label="Estimated", color="r", linestyle="--")

        # axs[1].plot(gradTAxis, np.cumsum((Shape) * xyzVec[1]), label="Theoretical", color="b")
        axs[1].legend()
        axs[1].set_xlim(0, plotEnd)
        # axs[1].set_xlabel('t[us]')
        axs[1].set_ylabel("Ty[%]")

        axs[2].plot(trajTAxis, np.cumsum(GzMeasured * fZ), label="Measured", color="g")
        axs[2].plot(trajTAxis, np.cumsum(RShapeZ), label="Estimated", color="r", linestyle="--")
        # axs[2].plot(gradTAxis, np.cumsum((Shape) * xyzVec[2]), label="Theoretical", color="b")
        axs[2].legend()
        axs[2].set_xlim(0, plotEnd)
        axs[2].set_xlabel("t[us]")
        axs[2].set_ylabel("Tz[%]")

        fig, axs = plt.subplots(3, 1)

        axs[0].plot(trajTAxis, (np.cumsum(RShapeX) - (np.cumsum(GxMeasured * fX))) / np.cumsum(GxMeasured * fX) * 10, label="Measured", color="g")
        axs[0].legend()
        axs[0].set_xlim(0, plotEnd)
        # axs[0].set_xlabel('t[us]')
        axs[0].set_ylim(-1, 1)
        axs[0].set_ylabel("Tx[%]")

        axs[1].plot(trajTAxis, (np.cumsum(RShapeY) - np.cumsum(GyMeasured * fY)) / np.cumsum(GyMeasured * fY) * 10, label="Measured", color="g")
        axs[1].legend()
        axs[1].set_xlim(0, plotEnd)
        axs[1].set_ylim(-1, 1)
        # axs[1].set_xlabel('t[us]')
        axs[1].set_ylabel("Ty[%]")

        axs[2].plot(trajTAxis, (np.cumsum(RShapeZ) - np.cumsum(GzMeasured * fZ)) / np.cumsum(GzMeasured * fZ) * 10, label="Measured", color="g")

        # axs[2].plot(gradTAxis, np.cumsum((Shape) * xyzVec[2]), label="Theoretical", color="b")
        axs[2].legend()
        axs[2].set_xlim(0, plotEnd)
        axs[1].set_ylim(-1, 1)
        axs[2].set_xlabel("t[us]")
        axs[2].set_ylabel("Tz[%]")

        mX = np.mean(np.cumsum(RBX)[75:150] * trajDwell)
        mY = np.mean(np.cumsum(RBY)[75:150] * trajDwell)
        mZ = np.mean(np.cumsum(RBZ)[75:150] * trajDwell)

        mmX = np.mean(BX[75:150])
        mmY = np.mean(BY[75:150])
        mmZ = np.mean(BZ[75:150])

        fX = mX / mmX
        fY = mY / mmY
        fZ = mZ / mmZ

        # fig, axs = plt.subplots(3, 1)
        # axs[0].plot(trajTAxis, (RBX)*trajDwell, label='EstimatedB0')
        # axs[0].plot(trajTAxis, (BX)*fX, label='MeasuredB0')
        # axs[0].legend()

        # axs[1].plot(trajTAxis,  (RBY)*trajDwell, label='EstimatedB0')
        # axs[1].plot(trajTAxis, (BY)*fY, label='MeasuredB0')
        # axs[1].legend()

        # axs[2].plot(trajTAxis, (RBZ)*trajDwell, label='EstimatedB0')
        # axs[2].plot(trajTAxis, (BZ)*fZ, label='MeasuredB0')
        # axs[2].legend()

        # print("Error in trajX is " +
        #       str(np.linalg.norm(np.cumsum(RShapeX)-np.cumsum(GxMeasured*fX))/len(GxMeasured)))
        # print("Error in trajY is " +
        #       str(np.linalg.norm(np.cumsum(RShapeY)-np.cumsum(GyMeasured*fZ))/len(GxMeasured)))
        # print("Error in trajZ is " +
        #       str(np.linalg.norm(np.cumsum(RShapeZ)-np.cumsum(GyMeasured*fZ))/len(GxMeasured)))

        # plt.savefig('testoutputB.png')
        # plt.show()

        fig, axs = plt.subplots(2, 1)
        transferX = np.array(self.systemInfo["XGrad"]["transAmplR"]).astype(np.complex64) + 1j * np.array(
            self.systemInfo["XGrad"]["transAmplI"]
        ).astype(np.complex64)
        transferY = np.array(self.systemInfo["YGrad"]["transAmplR"]).astype(np.complex64) + 1j * np.array(
            self.systemInfo["YGrad"]["transAmplI"]
        ).astype(np.complex64)
        transferZ = np.array(self.systemInfo["ZGrad"]["transAmplR"]).astype(np.complex64) + 1j * np.array(
            self.systemInfo["ZGrad"]["transAmplI"]
        ).astype(np.complex64)

        fAxis = np.array(self.systemInfo["XGrad"]["transFreq"]) / 1e3

        axs[0].plot(fAxis, abs(transferX), label="Gx", color="r")
        axs[0].plot(fAxis, abs(transferY), label="Gy", color="g")
        axs[0].plot(fAxis, abs(transferZ), label="Gz", color="b")

        # axs[0].set_xlabel('f[kHz]')
        axs[0].set_ylabel("|H|[-]")
        axs[0].set_xlim(0, 12.5)
        axs[0].legend()

        axs[1].plot(fAxis, np.angle(transferX), label="Gx", color="r")
        axs[1].plot(fAxis, np.angle(transferY), label="Gy", color="g")
        axs[1].plot(fAxis, np.angle(transferZ), label="Gz", color="b")

        axs[1].set_xlabel("f[kHz]")
        axs[1].set_ylabel("arg(H)[rad]")
        axs[1].set_xlim(0, 12.5)

        axs[1].set_ylim(-1, 0.1)
        axs[1].legend()

        plt.show()

    def calibrateB0Trapezoid(self, Method_file, trustStart=0, trustStop=6.5e3):
        BX = Method_file["PVM_TrajBx"]
        BY = Method_file["PVM_TrajBy"]
        BZ = Method_file["PVM_TrajBz"]

        gradRes = Method_file["PVM_TrajDwGrad"] / 1e3

        trajRes = Method_file["PVM_TrajDwAcq"] / 1e3

        readGradShape = Method_file["GradShape2"]

        gradCalConst = Method_file["PVM_GradCalConst"]

        readFov = Method_file["PVM_Fov"][0]

        PVM_EffSWh = Method_file["PVM_EffSWh"]

        # BX = np.diff(BX)/trajRes
        # BY = np.diff(BY)/trajRes
        # BZ = np.diff(BZ)/trajRes

        # BX = np.insert(BX, (0), 0)
        # BY = np.insert(BY, (0), 0)
        # BZ = np.insert(BZ, (0), 0)

        ReadGrad = self.calcReadGrad(PVM_EffSWh, readFov, gradCalConst)
        gradShape = readGradShape * ReadGrad

        gradTAxis = np.linspace(0, gradRes * len(gradShape) - gradRes, len(gradShape))
        trajTAxis = np.linspace(0, trajRes * len(BZ) - trajRes, len(BZ))

        gradShape = np.interp(trajTAxis, gradTAxis, gradShape)

        gradShape = self.delayWaveform(gradShape, 6e-6, trajRes)

        freqXB, TFBX = self.estimateTFChirp(
            gradShape,
            BX,
            trajRes,
            trustStop,
            0,
            BField=True,
            discartEnd=0,
            extrapolate=False,
            plot=True,
        )
        freqYB, TFBY = self.estimateTFChirp(
            gradShape,
            BY,
            trajRes,
            trustStop,
            0,
            BField=True,
            discartEnd=0,
            extrapolate=False,
            plot=True,
        )
        freqZB, TFBZ = self.estimateTFChirp(
            gradShape,
            BZ,
            trajRes,
            trustStop,
            0,
            BField=True,
            discartEnd=0,
            extrapolate=False,
            plot=True,
        )

        fillfreq = np.array(self.systemInfo["XGrad"]["BFreq"])
        fillfreqInd = (fillfreq >= trustStart) & (fillfreq <= trustStop)
        filler = np.interp(fillfreq[fillfreqInd], freqXB, TFBX)
        transfArray = np.array(self.systemInfo["XGrad"]["BAmplR"], dtype=np.complex128) + 1j * np.array(
            self.systemInfo["XGrad"]["BAmplI"], dtype=np.complex128
        )
        transfArray[fillfreqInd] = filler
        self.systemInfo["XGrad"]["BAmplR"] = np.real(transfArray).tolist()
        self.systemInfo["XGrad"]["BAmplI"] = np.imag(transfArray).tolist()

        fillfreq = np.array(self.systemInfo["YGrad"]["BFreq"])
        fillfreqInd = (fillfreq >= trustStart) & (fillfreq <= trustStop)
        filler = np.interp(fillfreq[fillfreqInd], freqYB, TFBY)
        transfArray = np.array(self.systemInfo["YGrad"]["BAmplR"], dtype=np.complex128) + 1j * np.array(
            self.systemInfo["YGrad"]["BAmplI"], dtype=np.complex128
        )
        transfArray[fillfreqInd] = filler
        self.systemInfo["YGrad"]["BAmplR"] = np.real(transfArray).tolist()
        self.systemInfo["YGrad"]["BAmplI"] = np.imag(transfArray).tolist()

        fillfreq = np.array(self.systemInfo["ZGrad"]["BFreq"])
        fillfreqInd = (fillfreq >= trustStart) & (fillfreq <= trustStop)
        filler = np.interp(fillfreq[fillfreqInd], freqZB, TFBZ)
        transfArray = np.array(self.systemInfo["ZGrad"]["BAmplR"], dtype=np.complex128) + 1j * np.array(
            self.systemInfo["ZGrad"]["BAmplI"], dtype=np.complex128
        )
        transfArray[fillfreqInd] = filler
        self.systemInfo["ZGrad"]["BAmplR"] = np.real(transfArray).tolist()
        self.systemInfo["ZGrad"]["BAmplI"] = np.imag(transfArray).tolist()

        self.testSystem(gradShape, trajRes, None, None, None, BX, BY, BZ)

        self.updateSystem()

        pass

    def testSystem(self, testShape, testShapeRes, Gx, Gy, Gz, Bx, By, Bz, dt=10e-6):
        """Tests the system response to test shape and compares to results obtained by exact measurement

        Args:
            testShape (_type_): Test Shape in units of percent of real gradient
            testShapeRes (_type_): Test shape temporal resolution in seconds
            Gx (_type_): Measured gradient shape when played on X Axis
            Gy (_type_): Measured gradient shape when played on Y Axis
            Gz (_type_): Measured gradient shape when played on Z Axis

            Bx (_type_): Measured B0 error when played on X Axis in rad
            By (_type_): Measured B0 error when played on X Axis in rad
            Bz (_type_): Measured B0 error when played on X Axis in rad
        """
        GxT, BxT = self.systemTransform(testShape, "XGrad", testShapeRes)
        GyT, ByT = self.systemTransform(testShape, "YGrad", testShapeRes)
        GzT, BzT = self.systemTransform(testShape, "ZGrad", testShapeRes)

        ommitEnd = 10
        if Gx is not None:
            fig, axs = plt.subplots(3, 1)
            axs[0].plot(np.cumsum(Gx[:-ommitEnd]), label="Measured")
            axs[0].plot(np.cumsum(GxT), label="Estimated")
            axs[0].legend()
            axs[1].plot(np.cumsum(Gy[:-ommitEnd]), label="Measured")
            axs[1].plot(np.cumsum(GyT), label="Estimated")
            axs[1].legend()
            axs[2].plot(np.cumsum(Gz[:-ommitEnd]), label="Measured")
            axs[2].plot(np.cumsum(GzT), label="Estimated")
            axs[2].legend()
            plt.savefig("testGrad.png")

            print("Error in trajX is " + str(np.linalg.norm(np.cumsum(Gx) - np.cumsum(GxT))))
            print("Error in trajY is " + str(np.linalg.norm(np.cumsum(Gy) - np.cumsum(GyT))))
            print("Error in trajZ is " + str(np.linalg.norm(np.cumsum(Gz) - np.cumsum(GzT))))
        plt.show()

        if Bx is not None:
            fig, axs = plt.subplots(3, 1)
            axs[0].plot((Bx[:-ommitEnd]), label="Measured")
            axs[0].plot((BxT), label="Estimated")
            axs[0].legend()

            axs[1].plot((By[:-ommitEnd]), label="Measured")
            axs[1].plot((ByT), label="Estimated")
            axs[1].legend()

            axs[2].plot((Bz[:-ommitEnd]), label="Measured")
            axs[2].plot((BzT), label="Estimated")
            axs[2].legend()
            plt.savefig("CalibB0.png")

            fig, axs = plt.subplots(3, 1)
            axs[0].plot(np.cumsum(Bx[:-ommitEnd]) * dt, label="Measured")
            axs[0].plot(np.cumsum(BxT) * dt, label="Estimated")
            axs[0].legend()

            axs[1].plot(np.cumsum(By[:-ommitEnd]) * dt, label="Measured")
            axs[1].plot(np.cumsum(ByT) * dt, label="Estimated")
            axs[1].legend()

            axs[2].plot(np.cumsum(Bz[:-ommitEnd]) * dt, label="Measured")
            axs[2].plot(np.cumsum(BzT) * dt, label="Estimated")
            axs[2].legend()

        plt.clf()

    def calibrateSystemChirp(self, Method):
        """Calibrates the System responses based on measured data by Chirp excitation

        Args:
            Method (_type_): Method file from measuring method
        """
        trajX = Method["PVM_TrajKx"]
        trajX = np.expand_dims(trajX, 0)
        trajY = Method["PVM_TrajKy"]
        trajY = np.expand_dims(trajY, 0)
        trajZ = Method["PVM_TrajKz"]
        trajZ = np.expand_dims(trajZ, 0)
        trajShapes = np.concatenate((trajX, trajY, trajZ), axis=0)

        BX = Method["PVM_TrajBx"]
        BY = Method["PVM_TrajBy"]
        BZ = Method["PVM_TrajBz"]

        ChirpAmpl = Method["ChirpAmplitude"]

        gradDwell = Method["PVM_TrajDwGrad"] / 1e3
        trajDwell = Method["PVM_TrajDwAcq"] / 1e3

        gradShape = Method["Chirp"] * ChirpAmpl

        gradCalConst = Method["PVM_GradCalConst"]

        GxMeasured = np.diff(trajShapes[0, :]) / trajDwell / gradCalConst * 100
        GyMeasured = np.diff(trajShapes[1, :]) / trajDwell / gradCalConst * 100
        GzMeasured = np.diff(trajShapes[2, :]) / trajDwell / gradCalConst * 100

        GxMeasured = np.insert(GxMeasured, (0), 0)
        GyMeasured = np.insert(GyMeasured, (0), 0)
        GzMeasured = np.insert(GzMeasured, (0), 0)

        # BX = np.diff(BX)/trajDwell
        # BY = np.diff(BY)/trajDwell
        # BZ = np.diff(BZ)/trajDwell

        # BX = np.insert(BX, (0), 0)
        # BY = np.insert(BY, (0), 0)
        # BZ = np.insert(BZ, (0), 0)

        gradTAxis = np.linspace(0, gradDwell * len(gradShape) - gradDwell, len(gradShape))
        trajTAxis = np.linspace(0, trajDwell * len(GzMeasured) - trajDwell, len(GzMeasured))

        RShape = np.interp(trajTAxis, gradTAxis, gradShape)

        # normGx = self.normalizeTraj(RShape, GxMeasured,0.1)
        # normGy = self.normalizeTraj(RShape, GyMeasured,0.1)
        # normGz = self.normalizeTraj(RShape, GzMeasured,0.1)

        normGx = GxMeasured
        normGy = GyMeasured
        normGz = GzMeasured

        freqX, TFX = self.estimateTFChirp(RShape, normGx, trajDwell, 12e3, 5e-2)
        freqY, TFY = self.estimateTFChirp(RShape, normGy, trajDwell, 12e3)
        freqZ, TFZ = self.estimateTFChirp(RShape, normGz, trajDwell, 12e3)

        self.systemInfo["XGrad"]["transFreq"] = freqX.tolist()
        self.systemInfo["XGrad"]["transAmplR"] = np.real(TFX).tolist()
        self.systemInfo["XGrad"]["transAmplI"] = np.imag(TFX).tolist()

        self.systemInfo["YGrad"]["transFreq"] = freqY.tolist()
        self.systemInfo["YGrad"]["transAmplR"] = np.real(TFY).tolist()
        self.systemInfo["YGrad"]["transAmplI"] = np.imag(TFY).tolist()

        self.systemInfo["ZGrad"]["transFreq"] = freqZ.tolist()
        self.systemInfo["ZGrad"]["transAmplR"] = np.real(TFZ).tolist()
        self.systemInfo["ZGrad"]["transAmplI"] = np.imag(TFZ).tolist()

        self.systemInfo["XGrad"]["delay"] = 0
        self.systemInfo["YGrad"]["delay"] = 0
        self.systemInfo["ZGrad"]["delay"] = 0

        # mBX=np.mean(BX)
        # BX=BX-mBX

        # mBY=np.mean(BY)
        # BY=BY-mBY

        # mBZ=np.mean(BZ)
        # BZ=BZ-mBZ

        freqXB, TFBX = self.estimateTFChirp(RShape, BX, trajDwell, 30e3, 0, extrapolate=False, BField=True, plot=False)
        freqYB, TFBY = self.estimateTFChirp(RShape, BY, trajDwell, 30e3, 0, extrapolate=False, BField=True, plot=False)
        freqZB, TFBZ = self.estimateTFChirp(RShape, BZ, trajDwell, 30e3, 0, extrapolate=False, BField=True, plot=False)

        self.systemInfo["XGrad"]["BFreq"] = freqXB.tolist()
        self.systemInfo["XGrad"]["BAmplR"] = np.real(TFBX).tolist()
        self.systemInfo["XGrad"]["BAmplI"] = np.imag(TFBX).tolist()

        self.systemInfo["XGrad"]["BBias"] = 0

        self.systemInfo["YGrad"]["BFreq"] = freqYB.tolist()
        self.systemInfo["YGrad"]["BAmplR"] = np.real(TFBY).tolist()
        self.systemInfo["YGrad"]["BAmplI"] = np.imag(TFBY).tolist()

        self.systemInfo["YGrad"]["BBias"] = 0

        self.systemInfo["ZGrad"]["BFreq"] = freqZB.tolist()
        self.systemInfo["ZGrad"]["BAmplR"] = np.real(TFBZ).tolist()
        self.systemInfo["ZGrad"]["BAmplI"] = np.imag(TFBZ).tolist()

        self.systemInfo["ZGrad"]["BBias"] = 0

        self.testSystem(
            RShape,
            trajDwell,
            GxMeasured,
            GyMeasured,
            GzMeasured,
            BX,
            BY,
            BZ,
            dt=trajDwell,
        )

        self.updateSystem()

        pass

    def estimateTFChirp(
        self,
        inputShape,
        measuredShape,
        newShapeRes,
        maxF=6e3,
        smoothPar=3e-2,
        *,
        BField=False,
        discartEnd=100,
        extrapolate=True,
        plot=False,
    ):
        """Estimates Gradient tranfer function from Chirp input

        Args:
            inputShape (_type_): Input Shape (theoretical gradient waveform)
            measuredShape (_type_): Output Shape (Measured gradient waveform) must be of same length as input
            newShapeRes (_type_): Temporal resolution in seconds of input and output
            maxF (_type_, optional): Maximum trusted frequency to explicitly extract from transfer function. Defaults to 6e3.
            smoothPar (_type_, optional): Smoothin parameter to smoothen the amplitude transfer characteristics. Defaults to 3e-2.
            BField (bool, optional): If calulating B0 field fork . Defaults to False.
            discartEnd (int, optional): discart some values from beginning and end. Defaults to 500.
            extrapolate (bool, optional): Extrapolate the characteristis to zero (used only in gradient trans function estimate). Defaults to True.
            plot (bool, optional): Whether to plot the resulting characteristics. Defaults to False.

        Returns:
            _type_: frequency axis of transfer fucntion, transfer Function
        """

        if discartEnd > 0:
            inputShape = np.copy(inputShape[:-discartEnd])
            measuredShape = np.copy(measuredShape[:-discartEnd])

        fAxis = np.linspace(0, 1 / newShapeRes - 1 / (newShapeRes * len(inputShape)), len(inputShape))

        H = np.fft.fft(measuredShape) / np.fft.fft(inputShape)

        # H=self.estimateTransferLeastSquares(inputShape,measuredShape,newShapeRes)

        if BField:
            # H[fAxis<50]=np.mean(H[((fAxis>200) & (fAxis<300))]) #The bias is not estimatable
            pass
        else:
            # The bias is not estimatable
            H[0] = np.mean(H[((fAxis > 200) & (fAxis < 300))])
        # H=H/np.max(abs(H[fAxis<maxF]))

        # Extrapolate to 0 transfer
        fs_2 = 0.5 * 1 / newShapeRes

        if extrapolate:
            pAmpl = np.polyfit(
                fAxis[(fAxis > 0.8 * maxF) & (fAxis < maxF)],
                abs(H[(fAxis > 0.8 * maxF) & (fAxis < maxF)]),
                1,
            )

            pPhase = np.polyfit(
                fAxis[(fAxis > 0.8 * maxF) & (fAxis < maxF)],
                np.angle(H[(fAxis > 0.8 * maxF) & (fAxis < maxF)]),
                1,
            )

            newH = np.copy(H)

            fillerAmpl = np.polyval(pAmpl, fAxis[fAxis > maxF])

            tmpAx = fAxis[fAxis > maxF]

            fillerAmpl[fillerAmpl < 0] = 0
            fillerAmpl[tmpAx > 40e3] = 0

            fillerPhase = np.polyval(pPhase, fAxis[fAxis > maxF])

            newH[fAxis > maxF] = fillerAmpl * np.exp(fillerPhase * 1j)

            newH = newH[fAxis < fs_2]

        else:
            newH = H
            newH[fAxis > maxF] = 0
            newH = newH[fAxis < fs_2]

        angleVal = np.angle(newH)

        amplVal = abs(newH)

        if smoothPar > 0:
            smothpars = scipy.interpolate.splrep(fAxis[fAxis < fs_2], amplVal, s=smoothPar)
            amplValSmooth = scipy.interpolate.splev(fAxis[fAxis < fs_2], smothpars)
        else:
            amplValSmooth = amplVal

        if plot:
            plt.clf()
            plt.plot(fAxis[fAxis < fs_2], abs(newH))
            plt.plot(fAxis[fAxis < fs_2], abs(amplValSmooth))
            plt.savefig("transferF.png")
            plt.show()
        newH = amplValSmooth * np.exp(1j * angleVal)

        return fAxis[fAxis < fs_2], newH

    def getPoisitionMatrix(self, ACQ_patient_pos, SUBJECT_type):
        # Subject types
        # ---------------------
        # Rodents: SUBJECT_type=Quadruped
        # Primates: SUBJECT_type=Biped
        # Other animals: SUBJECT_type=OtherAnimal
        # Material: SUBJECT_type=Other

        # ACQ_patient_pos may have the values:
        # -------------------------------------
        # Head_Supine - inverted Gx and Gz.
        # Head_Prone -  inverted Gy and Gz.
        # Head_Left -   inverted Gz. Gx and Gy are exchanged.
        # Head_Right -  inverted Gx, Gy and Gz. Gx and Gy are exchanged.
        # Foot_Supine - all gradients remain unchanged - used permanently in Other subj. type
        # Foot_Prone -  inverted Gx and Gy.
        # Foot_Left -   inverted Gy. Gx and Gy are exchanged.
        # Foot_Right -  inverted Gx. Gx and Gy are exchanged

        if SUBJECT_type == "Other":
            return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

        if ACQ_patient_pos == "Head_Supine":
            return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

        if ACQ_patient_pos == "Head_Prone":
            return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        if ACQ_patient_pos == "Head_Left":
            return np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

        if ACQ_patient_pos == "Head_Right":
            return np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])

        if ACQ_patient_pos == "Foot_Supine":
            return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        if ACQ_patient_pos == "Foot_Prone":
            return np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

        if ACQ_patient_pos == "Foot_Left":
            return np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

        if ACQ_patient_pos == "Foot_Right":
            return np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        raise Exception("Unknown patient Position: " + ACQ_patient_pos)

    def updateSystem(self):
        """Update Gradient System Info"""
        dirPath = os.path.dirname(__file__)

        with open(dirPath + "/gradSystemInfo.json") as jsonFile:
            contents = json.load(jsonFile)

        contents[self.system] = self.systemInfo

        with open(dirPath + "/gradSystemInfo.json", "w") as jsonFile:
            json.dump(contents, jsonFile, indent=2)

    def normalizeTraj(self, gradShape, trajectory, crop=0.1):
        normf = np.linalg.norm(trajectory[: int(crop * len(trajectory))]) / np.linalg.norm(gradShape[: int(crop * len(trajectory))])

        return trajectory / normf

    def forceSymmetry(self, X):
        """Symmetrize the input with respect to center. Used for generating symmetric transfer function from the first half

        Args:
            X (_type_): Input to be symmetrized

        Returns:
            _type_: Conjugate Symmetric output
        """
        if len(X) % 2 == 0:
            pivot = int(len(X) / 2)
            firstH = X[1:pivot]
            X[(pivot + 1) :] = np.conj(np.flip(firstH))
        else:
            pivot = int(np.floor(len(X) / 2))
            firstH = X[1:pivot]
            X[(pivot + 2) :] = np.conj(np.flip(firstH))

        return X

    def LTITransferDirect(self, waveform, bands, transferAmplR, transferAmplI, resolution):
        """Tranform of the input via a LTI (Linear Time Invariant) model

        Args:
            waveform (_type_): input Waveform
            bands (_type_): Frequency axis of LTI Transfer Function
            transferAmplR (_type_): Real part of LTI Transfer Function
            transferAmplI (_type_): Imaginary part of LTI Transfer Function
            resolution (_type_): Time resolution of the waveform in seconds

        Returns:
            _type_: Transformed input waveform
        """
        transferAmpl = np.array(transferAmplR).astype(np.complex64) + 1j * np.array(transferAmplI).astype(np.complex64)

        fLen = int(len(waveform) / 2) + 1 if len(waveform) % 2 == 0 else int((len(waveform) + 1) / 2)

        f = np.linspace(0, 1 / resolution - 1 / resolution / len(waveform), len(waveform))
        f = f[:fLen]

        H = np.zeros(fLen, dtype=np.complex64)

        fill = np.interp(f[f < max(bands)], bands, transferAmpl)

        H[f < max(bands)] = fill

        tmp = np.fft.irfft(np.fft.rfft(waveform) * H)

        return tmp  # noqa

    def delayWaveform(self, waveform, delay, shapeRes, interpFactor=30):
        """Delays input waveform by desired amount

        Args:
            waveform (_type_): inputWaveform
            delay (_type_): delay in seconds
            shapeRes (_type_): Shape resolution in seconds
            interpFactor (int, optional): default interpolation for sub sample shifts. Defaults to 30.

        Returns:
            _type_: Time shifted Waveform
        """
        if delay == 0:
            return waveform

        origTAxis = np.linspace(0, shapeRes * len(waveform) - shapeRes, len(waveform))
        newTAxis = np.linspace(0, shapeRes * len(waveform) - shapeRes, len(waveform)) - delay

        return np.interp(newTAxis, origTAxis, waveform, left=0, right=0)

        # origTAxis = np.linspace(0, shapeRes * len(waveform) - shapeRes, len(waveform))
        # interpTAxis = np.linspace(
        #     0,
        #     shapeRes * len(waveform) - shapeRes / interpFactor,
        #     len(waveform) * interpFactor,
        # )

        # waveform = np.interp(interpTAxis, origTAxis, waveform)

        # ShiftPts = int(delay / shapeRes * interpFactor)

        # waveform = np.insert(waveform, 0, np.zeros(ShiftPts))
        # waveform = waveform[:-(ShiftPts)]

        # return np.interp(origTAxis, interpTAxis, waveform, left=0, right=0)

    def systemTransform(self, inputWaveform, axis, shapeRes=8e-6, gradientPreDelay=0, *, removePad=True):
        """
        Plays virtually gradient Shape on desired gradient Axis
        Note: When transforming, the signal is padded with starting and ending values respectively. Ensure this wont distort the resulting shape
        too much.
        Ideally whole gradient Shape (e.g. incl. spoilers) should be inputed begining with zero and ending with zero values and cropped in the end
        after the transformation
        Args:
            inputWaveform (_type_): Gradient input (expected gradient Shape)
            axis (string): Gradient Axis To play it on (XGrad,YGrad,ZGrad)
            shapeRes (_type_, optional): Gradient Shape resolution in seconds. Defaults to 8e-6.
            gradientPreDelay (double, optional): Predelay between acquisition start and gradient start in seconds. Defaults to 0.

        Returns:
            _type_: Transformed gradient waveform and induced B0 field

        """
        oversamp = 4
        padWidth = 20
        inputWaveform = np.pad(inputWaveform, (0, padWidth), constant_values=inputWaveform[-1])
        inputWaveform = np.pad(inputWaveform, (padWidth, 0), constant_values=inputWaveform[0])

        tAxOrig = np.linspace(0, len(inputWaveform) * shapeRes - shapeRes, len(inputWaveform))
        tAxNew = np.linspace(
            0,
            len(inputWaveform) * shapeRes - shapeRes / oversamp,
            len(inputWaveform) * oversamp,
        )

        inputWaveform = np.interp(tAxNew, tAxOrig, inputWaveform)

        G = self.delayWaveform(inputWaveform, gradientPreDelay, shapeRes / oversamp, interpFactor=6)

        G = self.LTITransferDirect(
            G,
            self.systemInfo[axis]["transFreq"],
            self.systemInfo[axis]["transAmplR"],
            self.systemInfo[axis]["transAmplI"],
            shapeRes / oversamp,
        )

        B = self.LTITransferDirect(
            np.copy(G),
            self.systemInfo[axis]["BFreq"],
            self.systemInfo[axis]["BAmplR"],
            self.systemInfo[axis]["BAmplI"],
            shapeRes / oversamp,
        )

        G = np.interp(tAxOrig, tAxNew, G)
        B = np.interp(tAxOrig, tAxNew, B)
        if removePad:
            G = G[padWidth:-padWidth]
            B = B[padWidth:-padWidth]

        return G, B

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

    def calcReadGrad(self, EffSWh, FOV, gradCalConst):
        """_summary_

        Args:
            EffSWh (_type_): Effective Read BandWidth in Hz
            FOV (_type_): Field of view in mm
            gradCalConst (_type_): Gradient calibration constant in Hz/mm

        Returns:
            _type_: Required gradient strenght in % of gradCalConst
        """
        dt = 1 / EffSWh
        dk = 1 / FOV
        G = dk / dt
        return G / gradCalConst * 100

    def checkPhasingNew(self, Method_file, PatPos, SubType, *, theoretical=False):
        slselShapeX = Method_file["SlSel_GradShapeX"]
        slselShapeY = Method_file["SlSel_GradShapeY"]
        slselShapeZ = Method_file["SlSel_GradShapeZ"]
        slselShape = Method_file["SlSel_GradShapeT"]

        slselShapeX = slselShapeX / max(slselShapeX)
        slselShapeY = slselShapeY / max(slselShapeY)
        slselShapeZ = slselShapeZ / max(slselShapeZ)

        gradRes = Method_file["PVM_TrajDwGrad"] / 1e3

        gradPre = 0.0

        slselX, RBX = self.systemTransform(
            slselShapeX,
            "XGrad",
            shapeRes=gradRes,
            gradientPreDelay=gradPre,
            removePad=False,
        )
        slselY, RBY = self.systemTransform(
            slselShapeY,
            "YGrad",
            shapeRes=gradRes,
            gradientPreDelay=gradPre,
            removePad=False,
        )
        slselZ, RBZ = self.systemTransform(
            slselShapeZ,
            "ZGrad",
            shapeRes=gradRes,
            gradientPreDelay=gradPre,
            removePad=False,
        )

        padWidth = 20
        slselShape = np.pad(slselShapeX, (0, padWidth), constant_values=slselShapeX[-1])
        slselShape = np.pad(slselShape, (padWidth, 0), constant_values=slselShape[0])
        amplEnable = 25e-6
        tRefocus = (
            Method_file["PVM_RiseTime"] / 1e3
            + amplEnable
            + (1 - Method_file["SlSel_RF_pv_pulse"][9] / 1e2) * Method_file["SlSel_RF_pv_pulse"][0] / 1e3
        )
        tRefocus = tRefocus + padWidth * gradRes

        tPulseEnd = tRefocus + (Method_file["SlSel_RF_pv_pulse"][9] / 1e2) * Method_file["SlSel_RF_pv_pulse"][0] / 1e3

        tAxis = np.linspace(0, len(slselShape) * gradRes - gradRes, len(slselShape))

        plt.scatter(tRefocus * 1e6, slselShape[int(tRefocus / gradRes)])
        plt.scatter(tPulseEnd * 1e6, slselShape[int(tRefocus / gradRes)])

        tAxisOut = np.linspace(tRefocus, len(slselShape) * gradRes - gradRes, 6 * len(slselShape))

        slselXReph = np.interp(tAxisOut, tAxis, slselX)
        slselYReph = np.interp(tAxisOut, tAxis, slselY)
        slselZReph = np.interp(tAxisOut, tAxis, slselZ)

        slselTReph = np.interp(tAxisOut, tAxis, slselShape)

        plt.plot(tAxisOut * 1e6, slselTReph, label="T")
        plt.plot(tAxisOut * 1e6, slselXReph, label="X")
        plt.plot(tAxisOut * 1e6, slselYReph, label="Y")
        plt.plot(tAxisOut * 1e6, slselZReph, label="Z")

        plt.legend()

    def checkPhasing(self, Method_file, *, theoretical=False):
        slselShape = Method_file["GradShape1"]
        gradRes = Method_file["PVM_TrajDwGrad"] / 1e3

        gradPre = 0.0

        slselX, RBX = self.systemTransform(
            slselShape,
            "XGrad",
            shapeRes=gradRes,
            gradientPreDelay=gradPre,
            removePad=False,
        )
        slselY, RBY = self.systemTransform(
            slselShape,
            "YGrad",
            shapeRes=gradRes,
            gradientPreDelay=gradPre,
            removePad=False,
        )
        slselZ, RBZ = self.systemTransform(
            slselShape,
            "ZGrad",
            shapeRes=gradRes,
            gradientPreDelay=gradPre,
            removePad=False,
        )

        padWidth = 20
        slselShape = np.pad(slselShape, (0, padWidth), constant_values=slselShape[-1])
        slselShape = np.pad(slselShape, (padWidth, 0), constant_values=slselShape[0])
        amplEnable = 25e-6
        tRefocus = Method_file["PVM_RiseTime"] / 1e3 + amplEnable + (1 - Method_file["ExcPulse1"][9] / 1e2) * Method_file["ExcPulse1"][0] / 1e3
        tRefocus = tRefocus + padWidth * gradRes

        tPulseEnd = tRefocus + (Method_file["ExcPulse1"][9] / 1e2) * Method_file["ExcPulse1"][0] / 1e3

        tAxis = np.linspace(0, len(slselShape) * gradRes - gradRes, len(slselShape))

        plt.scatter(tRefocus * 1e6, slselShape[int(tRefocus / gradRes)])
        plt.scatter(tPulseEnd * 1e6, slselShape[int(tRefocus / gradRes)])

        tAxisOut = np.linspace(tRefocus, len(slselShape) * gradRes - gradRes, len(slselShape))

        slselXReph = np.interp(tAxisOut, tAxis, slselX)
        slselYReph = np.interp(tAxisOut, tAxis, slselY)
        slselZReph = np.interp(tAxisOut, tAxis, slselZ)

        slselTReph = np.interp(tAxisOut, tAxis, slselShape)

        plt.plot(tAxisOut * 1e6, slselTReph)
        plt.plot(tAxisOut * 1e6, slselXReph)
        plt.plot(tAxisOut * 1e6, slselYReph)
        plt.plot(tAxisOut * 1e6, slselZReph)
        pass

    def generateCorrectionsRadialCS(self, Method_file, acqp, *, theoretical=False):
        """_summary_

        Args:
            Method_file (dict): Bruker Method file of Mac_CS pulse sequence
            PatPos (string): Patient position (from acqp) string e.g. Head_Prone
            SubType (string): SUBJECT type (from subject file)
            theoretical (bool, optional):Weather to use theoretical trajectory. Defaults to False.

        Returns:
            _type_: Trajectories in rad/px <-pi:pi> and B0 field contaminations in rad
            To apply B0 correction multiply measured echoes with exp(-1j*B0corr)
        """

        gradRes = Method_file["PVM_TrajDwGrad"] / 1e3

        trajRes = 1 / (Method_file["PVM_EffSWh"])

        acqLen = int(acqp["ACQ_jobs"][0][0] / 2)

        readGradShape = Method_file["RadRead_ReadGradShapeTraj"]
        phaseGradShape = Method_file["RadRead_Ph3GradShapeTraj"]
        MGEGradShape = Method_file["RadRead_MGEGradShape"]

        gradCalConst = Method_file["PVM_GradCalConst"]

        readFov = Method_file["PVM_Fov"][0]
        phaseFov = Method_file["PVM_Fov"][1]

        PVM_EffSWh = Method_file["PVM_EffSWh"]

        ReadGrad = self.calcReadGrad(PVM_EffSWh, readFov, gradCalConst)

        PhaseGrad = self.calcReadGrad(PVM_EffSWh, phaseFov, gradCalConst)

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

        initDelay = -0e-6  # seconds

        transformMatrix = gradMatrix

        nEchoes = Method_file["PVM_NEchoImages"]
        afterAcqTime = Method_file["RadRead_AfterAcqWaitTime"] / 1e3
        interEchoTime = Method_file["RadRead_EchoFillDelay"] / 1e3
        interEchoTimeApprox = round((afterAcqTime + interEchoTime + gradRes) / gradRes) * gradRes
        filler = np.zeros(int(interEchoTimeApprox / gradRes))
        mgeFiller = np.zeros_like(MGEGradShape)

        acqStartTimes = []
        acqStopTimes = []
        acqStartTimes.append(0)
        acqStopTimes.append(acqLen * trajRes)  # Probably due to different RBW in chirp calibrating experiment and Radial sequence (should find out)
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

        trajCorrected, BCorrection, shapes = self.generateTrajectory(
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
            initDelay=initDelay,
            theoretical=theoretical,
        )

        return trajCorrected, BCorrection

    def generateCorrectionsMacCSnew(self, Method_file, acqp, SubType, *, theoretical=False):
        """_summary_

        Args:
            Method_file (dict): Bruker Method file of Mac_CS pulse sequence
            PatPos (string): Patient position (from acqp) string e.g. Head_Prone
            SubType (string): SUBJECT type (from subject file)
            theoretical (bool, optional):Weather to use theoretical trajectory. Defaults to False.

        Returns:
            _type_: Trajectories in rad/px <-pi:pi> and B0 field contaminations in rad
            To apply B0 correction multiply measured echoes with exp(-1j*B0corr)
        """
        PatPos = acqp["ACQ_patient_pos"]

        gradRes = Method_file["PVM_TrajDwGrad"] / 1e3

        trajRes = Method_file["PVM_TrajDwAcq"] / 1e3

        acqLen = int(acqp["ACQ_size"][0] / 2)

        readGradShape = Method_file["RadRead_ReadGradShapeTraj"]
        phaseGradShape = Method_file["RadRead_Ph3GradShapeTraj"]
        MGEGradShape = Method_file["RadRead_MGEGradShape"]

        gradCalConst = Method_file["PVM_GradCalConst"]

        readFov = Method_file["PVM_Fov"][0]
        phaseFov = Method_file["PVM_Fov"][1]

        PVM_EffSWh = Method_file["PVM_EffSWh"]

        ReadGrad = self.calcReadGrad(PVM_EffSWh, readFov, gradCalConst)

        PhaseGrad = self.calcReadGrad(PVM_EffSWh, phaseFov, gradCalConst)

        Phase3DGrad = Method_file["RadRead_Phase3DGrad"]

        RFactor = Method_file["RadRead_GradAmpR"] * ReadGrad
        PFactor = Method_file["RadRead_GradAmpP"] * PhaseGrad
        SFactor = -Method_file["RadRead_GradAmpS"] * Phase3DGrad

        gradMatrix = np.squeeze(Method_file["PVM_SPackArrGradOrient"])

        if Method_file["PVM_Fov"].shape[0] == 2:
            FOV = Method_file["PVM_Fov"]
            FOV = np.append(FOV, Method_file["PVM_SliceThick"])
            SamplingMatrix = Method_file["PVM_Matrix"]
            SamplingMatrix = np.append(SamplingMatrix, 1)
        else:
            FOV = Method_file["PVM_Fov"]
            SamplingMatrix = Method_file["PVM_Matrix"]

        initDelay = 10e-6

        transformMatrix = gradMatrix @ self.getPoisitionMatrix(PatPos, SubType)

        nEchoes = Method_file["PVM_NEchoImages"]
        afterAcqTime = Method_file["RadRead_AfterAcqWaitTime"] / 1e3
        interEchoTime = Method_file["RadRead_EchoFillDelay"] / 1e3
        interEchoTimeApprox = round((afterAcqTime + interEchoTime + gradRes) / gradRes) * gradRes
        filler = np.zeros(int(interEchoTimeApprox / gradRes))
        mgeFiller = np.zeros_like(MGEGradShape)

        acqStartTimes = []
        acqStopTimes = []
        acqStartTimes.append(0)
        acqStopTimes.append(acqLen * trajRes)  # Probably due to different RBW in chirp calibrating experiment and Radial sequence (should find out)
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

        trajCorrected, BCorrection, shapes = self.generateTrajectory(
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
            initDelay=initDelay,
            theoretical=theoretical,
        )

        return trajCorrected, BCorrection

    def generateCorrectionsMacCS(self, Method_file, acqp, SubType, *, theoretical=False):
        """_summary_

        Args:
            Method_file (dict): Bruker Method file of Mac_CS pulse sequence
            acqp (string): acquitision paramters
            SubType (string): SUBJECT type (from subject file)
            theoretical (bool, optional):Weather to use theoretical trajectory. Defaults to False.

        Returns:
            _type_: Trajectories in rad/px <-pi:pi> and B0 field contaminations in rad
            To apply B0 correction multiply measured echoes with exp(-1j*B0corr)
        """

        PatPos = acqp["ACQ_patient_pos"]

        gradRes = Method_file["PVM_TrajDwGrad"] / 1e3

        trajRes = Method_file["PVM_TrajDwAcq"] / 1e3

        trajX = Method_file["PVM_TrajKx"]
        trajX = np.expand_dims(trajX, 0)

        readGradShape = Method_file["GradShape2"]
        phaseGradShape = Method_file.get("GradShape3", np.zeros_like(readGradShape))

        gradCalConst = Method_file["PVM_GradCalConst"]

        readFov = Method_file["PVM_Fov"][0]
        phaseFov = Method_file["PVM_Fov"][1]

        PVM_EffSWh = Method_file["PVM_EffSWh"]

        ReadGrad = self.calcReadGrad(PVM_EffSWh, readFov, gradCalConst)

        PhaseGrad = self.calcReadGrad(PVM_EffSWh, phaseFov, gradCalConst)

        Phase3DGrad = Method_file["Phase3DGrad"]

        RFactor = Method_file["GradAmpR"] * ReadGrad
        PFactor = Method_file["GradAmpP"] * PhaseGrad
        SFactor = -Method_file["GradAmpS"] * Phase3DGrad * 2  # Ondra has -2* in PPG File

        gradMatrix = np.squeeze(Method_file["PVM_SPackArrGradOrient"])

        if Method_file["PVM_Fov"].shape[0] == 2:
            FOV = Method_file["PVM_Fov"]
            FOV = np.append(FOV, Method_file["PVM_SliceThick"])
            SamplingMatrix = Method_file["PVM_Matrix"]
            SamplingMatrix = np.append(SamplingMatrix, 1)
        else:
            FOV = Method_file["PVM_Fov"]
            SamplingMatrix = Method_file["PVM_Matrix"]

        initDelay = 6e-6  # Probably due to different RBW in chirp calibrating experiment and Radial sequence (should find out)

        transformMatrix = gradMatrix @ self.getPoisitionMatrix(PatPos, SubType)

        trajCorrected, BCorrection, _ = self.generateTrajectory(
            readGradShape,
            readGradShape,
            phaseGradShape,
            RFactor,
            PFactor,
            SFactor,
            gradRes,
            trajRes,
            transformMatrix,
            [0],
            [0 + trajX.shape[1] * trajRes],
            Method_file["PVM_GradCalConst"],
            FOV,
            SamplingMatrix,
            initDelay=initDelay,
            theoretical=theoretical,
        )

        return trajCorrected, BCorrection

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
        initDelay=0,
        *,
        theoretical=False,
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

        gradTAxis = np.linspace(0, gradRes * len(RShape) - gradRes, len(RShape))

        samplTrajTAxis = []

        for i in range(len(acqStartTime)):
            samplTrajTAxis.append(  # noqa: PERF401
                np.linspace(
                    acqStartTime[i],
                    acqStopTime[i] - trajRes,
                    int(np.round((acqStopTime[i] - acqStartTime[i]) / trajRes)),
                )
            )

        try:
            length = len(PFactor)
        except Exception as _:
            length = 1

        traj = np.zeros((len(samplTrajTAxis[0]), length, 3, len(samplTrajTAxis)))

        Bcorr = np.zeros((len(samplTrajTAxis[0]), length, len(samplTrajTAxis)))

        """
        Dirty Bruker secret -> More usual and probably correct would be opposite order (x,y,z) = GradMatrix*PosMatrix*(r,p,s)
        From PVMan:
        The physical (x,y,z) gradient values applied can be calculated by the equation:
        (x,y,z) = (r,p,s) * ACQ_grad_matrix * Position_Matrix
        For all subjects apart from “Material”, the position feet_supine is associated with the identity
        matrix. For material, no position can be selected, but a constant matrix (-1 0 0 0 1 0 0 0 -1) is
        used as Position_Matrix, i.e. x and z directions are inverted.
        """

        XR, YR, ZR = self.convertSpatialCoord(1, 0, 0, transformMatrix)
        XP, YP, ZP = self.convertSpatialCoord(0, 1, 0, transformMatrix)
        XS, YS, ZS = self.convertSpatialCoord(0, 0, 1, transformMatrix)

        if theoretical:
            RShapeX = RShape
            RShapeY = RShape
            RShapeZ = RShape

            PShapeX = PShape
            PShapeY = PShape
            PShapeZ = PShape

            SShapeX = SShape
            SShapeY = SShape
            SShapeZ = SShape

            RBX = np.zeros_like(SShapeX)
            RBY = np.zeros_like(SShapeX)
            RBZ = np.zeros_like(SShapeX)

            PBX = np.zeros_like(SShapeX)
            PBY = np.zeros_like(SShapeX)
            PBZ = np.zeros_like(SShapeX)

            SBX = np.zeros_like(SShapeX)
            SBY = np.zeros_like(SShapeX)
            SBZ = np.zeros_like(SShapeX)

        else:
            RShapeX, RBX = self.systemTransform(RShape, "XGrad", shapeRes=gradRes, gradientPreDelay=initDelay)
            RShapeY, RBY = self.systemTransform(RShape, "YGrad", shapeRes=gradRes, gradientPreDelay=initDelay)
            RShapeZ, RBZ = self.systemTransform(RShape, "ZGrad", shapeRes=gradRes, gradientPreDelay=initDelay)

            PShapeX, PBX = self.systemTransform(PShape, "XGrad", shapeRes=gradRes, gradientPreDelay=initDelay)
            PShapeY, PBY = self.systemTransform(PShape, "YGrad", shapeRes=gradRes, gradientPreDelay=initDelay)
            PShapeZ, PBZ = self.systemTransform(PShape, "ZGrad", shapeRes=gradRes, gradientPreDelay=initDelay)

            SShapeX, SBX = self.systemTransform(SShape, "XGrad", shapeRes=gradRes, gradientPreDelay=initDelay)
            SShapeY, SBY = self.systemTransform(SShape, "YGrad", shapeRes=gradRes, gradientPreDelay=initDelay)
            SShapeZ, SBZ = self.systemTransform(SShape, "ZGrad", shapeRes=gradRes, gradientPreDelay=initDelay)

        for i in range(length):
            X = RFactor[i] * (RShapeX * XR) + PFactor[i] * (PShapeX * XP) + SFactor[i] * (SShapeX * XS)

            Y = RFactor[i] * (RShapeY * YR) + PFactor[i] * (PShapeY * YP) + SFactor[i] * (SShapeY * YS)

            Z = RFactor[i] * (RShapeZ * ZR) + PFactor[i] * (PShapeZ * ZP) + SFactor[i] * (SShapeZ * ZS)

            R, P, S = self.convertSpatialCoord(X, Y, Z, np.linalg.inv(transformMatrix))

            BX = RFactor[i] * (RBX * XR) + PFactor[i] * (PBX * XP) + SFactor[i] * (SBX * XS)

            BY = RFactor[i] * (RBY * YR) + PFactor[i] * (PBY * YP) + SFactor[i] * (SBY * YS)

            BZ = RFactor[i] * (RBZ * ZR) + PFactor[i] * (PBZ * ZP) + SFactor[i] * (SBZ * ZS)

            B = (BX) + (BY) + (BZ)
            # B = (np.cumsum(BX)+np.cumsum(BY)+np.cumsum(BZ)) * \
            #     trajRes  # Convert to Radians

            trajR = np.cumsum(R) * gradRes * gradCalConst * FOV[0] / SamplingMatrix[0] / 100 / 0.5 * np.pi
            trajP = np.cumsum(P) * gradRes * gradCalConst * FOV[1] / SamplingMatrix[1] / 100 / 0.5 * np.pi
            trajS = np.cumsum(S) * gradRes * gradCalConst * FOV[2] / SamplingMatrix[2] / 100 / 0.5 * np.pi

            for acq in range(len(samplTrajTAxis)):
                Bcorr[:, i, acq] = np.interp(samplTrajTAxis[acq], gradTAxis, B)
                traj[:, i, 0, acq] = np.interp(samplTrajTAxis[acq], gradTAxis, trajR)
                traj[:, i, 1, acq] = np.interp(samplTrajTAxis[acq], gradTAxis, trajP)
                traj[:, i, 2, acq] = np.interp(samplTrajTAxis[acq], gradTAxis, trajS)

        return (
            traj,
            Bcorr,
            [
                RShapeX,
                RShapeY,
                RShapeZ,
                PShapeX,
                PShapeY,
                PShapeZ,
                SShapeX,
                SShapeY,
                SShapeZ,
            ],
        )

    def convertTrajToBartScale(self, method, traj, oversampling=1):
        matrix = method["PVM_Matrix"]

        for i in range(len(matrix)):
            traj[:, :, i, :] = traj[:, :, i, :] / np.pi * matrix[i] / oversampling
        return traj
