import json

import matplotlib.pyplot as plt
import numpy as np
import scipy

from utils.BrukerMRI import *
from utils.utils import getData, parse_args


def run(dataPath, scan, outRes):
    gradientFirst = False
    scanPath = dataPath / scan
    method = ReadParamFile(scanPath / "method")
    trajRes = outRes
    polynomials, testShapes, env = getData(scanPath, outRes=outRes, sliceIndices=None, gradientFirst=gradientFirst, frequencyFilter=150e3)

    polynomials = polynomials[-1, ...]  # get through slice dir
    grads = polynomials[0] if gradientFirst else np.gradient(polynomials, outRes, axis=1)[0]

    fLen = int(testShapes.size / 2) + 1 if testShapes.size % 2 == 0 else int((testShapes.size + 1) / 2)

    f = np.linspace(0, (1 / (trajRes) - 1 / trajRes / testShapes.size), testShapes.size)
    f = f[:fLen]

    # transfer = np.fft.rfft(grads, axis=0) / np.fft.rfft(np.expand_dims(testShapes, -1), axis=0)
    # Wiener
    X = np.expand_dims(np.fft.rfft(testShapes, axis=0), -1)
    Y = np.fft.rfft(grads, axis=0)

    eps = 1e-6 * np.max(np.abs(X) ** 2)
    transfer = Y * np.conj(X) / (np.abs(X) ** 2 + eps)

    transfer[:2, :] = 1 * np.exp(-np.angle(transfer[:2, :]) * 1j)
    transfer[0, :] = 1
    # impulseResponse = np.fft.fftshift(np.fft.irfft(transfer, axis=0), axes=0)

    transferMask = np.fft.rfft(np.expand_dims(testShapes, -1), axis=0)
    transferMask = np.abs(transferMask)
    transferMask = transferMask / np.max(transferMask)
    transferMask = ((transferMask > 0.01) | (np.expand_dims(f, -1) < 1e2)) & (np.expand_dims(f, -1) < method["ChirpFmax"] * 1e3)
    transfer = transfer * transferMask

    # smoothPar = 1e-2
    # for i in range(3):
    #     smothpars = scipy.interpolate.splrep(f, np.abs(transfer[:, i]), s=smoothPar)
    #     transfer[:, i] = scipy.interpolate.splev(f, smothpars) * np.exp(1j * np.angle(transfer[:, i]))

    impResp = np.fft.fftshift(np.fft.irfft(transfer, axis=0), axes=0)

    sizeResp = impResp.shape[0]
    impResp[: int(sizeResp * 0.4), :] = 0
    impResp[-int(sizeResp * 0.4) :, :] = 0

    transclean = np.fft.rfft(np.fft.fftshift(impResp, axes=0), axis=0)

    # fix low frequencies

    fstop = 250

    fstopInd = np.sum(f < fstop)

    axLabels = ["Z", "X", "Y"]

    print(method["ACQ_RxFilterSettings"][0])
    fRange = f[(f < 4000) & (f > 200)]
    for i in range(3):
        fitRange = np.angle(transclean[:, i])[(f < 4000) & (f > 200)]

        p = np.polyfit(fRange, fitRange, 1)
        print(f"Mean Delay {-p[0] * 1e6 / 2 / np.pi:.2f} us {axLabels[i]}")
        transclean[:fstopInd, i] = 1 * np.exp(1j * np.polyval(p, f[:fstopInd]))

    smoothPar = 1

    angleClean = np.unwrap(np.angle(transfer)) * np.expand_dims((f < 15e3), -1)
    for i in range(3):
        smothpars = scipy.interpolate.splrep(f, angleClean[:, i], s=smoothPar)
        angleClean[:, i] = scipy.interpolate.splev(f, smothpars)

    fig, ax = plt.subplots(2, 1)

    df = np.diff(f)[0]

    delay = -np.diff(angleClean, axis=0, append=0) / (df * 2 * np.pi) * 1e6  # us

    fstop = 1000
    fstopInd = np.sum(f < fstop)
    transcleanMag = np.mean(np.abs(transclean[1:fstopInd, :]), keepdims=True, axis=0)

    transclean = transclean / transcleanMag

    transclean[0, :] = 1

    indSort = np.argsort(axLabels)

    axLabels = np.array(axLabels)[indSort].tolist()

    transclean = transclean[:, indSort]

    ax[0].plot(f * 1e-3, np.abs(transclean))
    ax[1].plot(f * 1e-3, np.rad2deg(np.angle(transclean)))
    # ax[2].plot(f * 1e-3, delay)

    ax[0].set_ylim(0, 1.05)
    ax[1].set_ylim(-120, 20)
    # ax[2].set_ylim(-0.05, 20)

    ax[0].set_xlim(0, 50)
    ax[1].set_xlim(0, 50)
    # ax[2].set_xlim(0, 42)

    ax[0].legend(axLabels)
    ax[1].legend(axLabels)

    ax[1].set_xlabel("f[kHz]")

    ax[0].set_ylabel("A[-]")
    ax[1].set_ylabel("phi[deg]")

    ax[1].set_xlabel("f[kHz]")

    fig.suptitle("Transfer function")

    system = {}

    xgrad = {}

    xgrad["transFreq"] = f.tolist()
    xgrad["transAmplR"] = transclean[:, axLabels.index("X")].real.tolist()
    xgrad["transAmplI"] = transclean[:, axLabels.index("X")].imag.tolist()
    xgrad["delay"] = 0

    xgrad["BFreq"] = f.tolist()
    xgrad["BAmplR"] = transclean[:, axLabels.index("X")].real.tolist()
    xgrad["BAmplI"] = transclean[:, axLabels.index("X")].imag.tolist()
    xgrad["BBias"] = 0

    ygrad = {}

    ygrad["transFreq"] = f.tolist()
    ygrad["transAmplR"] = transclean[:, axLabels.index("Y")].real.tolist()
    ygrad["transAmplI"] = transclean[:, axLabels.index("Y")].imag.tolist()
    ygrad["delay"] = 0

    ygrad["BFreq"] = f.tolist()
    ygrad["BAmplR"] = transclean[:, axLabels.index("Y")].real.tolist()
    ygrad["BAmplI"] = transclean[:, axLabels.index("Y")].imag.tolist()
    ygrad["BBias"] = 0

    zgrad = {}
    zgrad["transFreq"] = f.tolist()
    zgrad["transAmplR"] = transclean[:, axLabels.index("Z")].real.tolist()
    zgrad["transAmplI"] = transclean[:, axLabels.index("Z")].imag.tolist()
    zgrad["delay"] = 0
    zgrad["BFreq"] = f.tolist()
    zgrad["BAmplR"] = transclean[:, axLabels.index("Z")].real.tolist()
    zgrad["BAmplI"] = transclean[:, axLabels.index("Z")].imag.tolist()
    zgrad["BBias"] = 0

    system["XGrad"] = xgrad
    system["YGrad"] = ygrad
    system["ZGrad"] = zgrad

    os.makedirs("caloutputs", exist_ok=True)
    with open("caloutputs/gradTransfer.json", "w") as f:
        json.dump(system, f, indent=1)

    plt.savefig("caloutputs/gradTransfer.png")
    plt.show()

    pass


if __name__ == "__main__":
    args = parse_args("Compute gradient transfer function", "response")
    run(args[0], args[1], args[2])
