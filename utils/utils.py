import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from utils.BrukerMRI import *


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ResolveProjectPath(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def PaperDataRoot(config=None):
    if config is None:
        config = load_config()

    return ResolveProjectPath(config.get("paperDataRoot", "paperData"))


def PaperDataPath(dataset, scan=None, config=None):
    path = PaperDataRoot(config) / dataset
    return path / str(scan) if scan is not None else path


def oversample(x, factor, axis=-1, pad=0):
    """
    Oversample signal with edge padding to suppress ringing.

    Parameters
    ----------
    x : ndarray
        Input signal
    factor : int
        Oversampling factor (>1)
    axis : int
        Axis along which to resample
    pad : int
        Number of samples to pad on each side
        (before oversampling)

    Returns
    -------
    y : ndarray
        Oversampled signal
    """
    if factor == 1:
        return x

    if pad > 0:
        pad_width = [(0, 0)] * x.ndim
        pad_width[axis] = (0, pad)

        x_pad = np.pad(x, pad_width, mode="edge")
    else:
        x_pad = x

    y_pad = resample_poly(x_pad, up=factor, down=1, axis=axis)

    if pad > 0:
        slicer = [slice(None)] * y_pad.ndim
        slicer[axis] = slice(0, -pad * factor)
        y = y_pad[tuple(slicer)]
    else:
        y = y_pad

    return y


def reduce_oversampling(x, factor, axis=-1, pad=0):
    if factor == 1:
        return x

    if pad > 0:
        pad_width = [(0, 0)] * x.ndim
        pad_width[axis] = (0, pad)

        x_pad = np.pad(x, pad_width, mode="edge")
    else:
        x_pad = x

    y_pad = resample_poly(x_pad, up=1, down=factor, axis=axis)

    if pad > 0:
        slicer = [slice(None)] * y_pad.ndim
        slicer[axis] = slice(0, -int(pad // factor))
        y = y_pad[tuple(slicer)]
    else:
        y = y_pad

    return y


def circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance to edge
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    return dist_from_center < radius


def spectFilt(x, fs, fstop, axis=-1, width=None, pad_frac=0.25):
    """
    FFT-based low-pass filter with edge padding to reduce ringing.

    Parameters
    ----------
    x : ndarray
        Input signal
    fs : float
        Sampling frequency
    fstop : float
        Cutoff frequency
    axis : int
        Axis along which filtering is performed
    width : float or None
        Transition width of sigmoid
    pad_frac : float
        Fraction of signal length used for padding
    """

    axis = axis % x.ndim

    n = x.shape[axis]

    if width is None:
        width = 0.05 * fstop

    # --- padding length along selected axis ---
    pad = int(n * pad_frac)

    # build pad specification for all axes
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (pad, pad)

    # edge padding
    xpad = np.pad(x, pad_width=pad_width, mode="edge")

    # --- FFT ---
    Xs = np.fft.rfft(xpad, axis=axis)

    n_pad = xpad.shape[axis]
    f = np.fft.rfftfreq(n_pad, d=1 / fs)

    # smooth low-pass mask
    mask = 1.0 / (1.0 + np.exp((f - fstop) / width))

    # reshape mask for broadcasting along axis
    shape = [1] * xpad.ndim
    shape[axis] = mask.size
    mask = mask.reshape(shape)

    Xs *= mask

    # --- inverse FFT ---
    ypad = np.fft.irfft(Xs, n=n_pad, axis=axis)

    # --- remove padding ---
    slicer = [slice(None)] * x.ndim
    slicer[axis] = slice(pad, -pad)

    return ypad[tuple(slicer)]


def resampleArr(arrIn: np.ndarray, inRes, outRes, axis=-1, tOut=None):
    """
    Resample array along a selected axis using linear interpolation.

    Parameters
    ----------
    arrIn : ndarray
        Input array
    inRes : float
        Input sampling interval
    outRes : float
        Output sampling interval
    axis : int
        Axis along which resampling is performed
    """
    if inRes == outRes and tOut is None:
        return arrIn

    arrIn = np.asarray(arrIn)
    axis = axis % arrIn.ndim

    ratio = outRes / inRes
    factor = round(ratio)

    n_in = arrIn.shape[axis]

    # --- integer downsampling (fast path) ---
    if np.isclose(ratio, factor) and tOut is None:
        slicer = [slice(None)] * arrIn.ndim
        slicer[axis] = slice(None, None, factor)
        return arrIn[tuple(slicer)]

    # --- output size ---
    nSamplesOut = int(inRes * n_in / outRes)

    # time coordinates
    tIn = np.arange(n_in) * inRes
    if tOut is None:
        tOut = np.arange(nSamplesOut) * outRes
    else:
        nSamplesOut = tOut.size

    # move resampling axis to front
    arr = np.moveaxis(arrIn, axis, 0)

    # flatten remaining dims
    orig_shape = arr.shape
    arr = arr.reshape(n_in, -1)

    # output container
    arrOut = np.empty((nSamplesOut, arr.shape[1]), dtype=arrIn.dtype)

    # interpolate each column
    for i in range(arr.shape[1]):
        arrOut[:, i] = np.interp(tOut, tIn, arr[:, i])

    # restore original shape
    new_shape = (nSamplesOut, *orig_shape[1:])
    arrOut = arrOut.reshape(new_shape)

    # move axis back
    return np.moveaxis(arrOut, 0, axis)


def get_acquisition_dwell(method, acqp=None):

    spectral_width = float(np.asarray(acqp["ACQ_jobs"])[0][5])
    return 1 / spectral_width



    return np.sum(data * weights, axis=coil_axis)


def weighted_polyfit(x, y, w, degree):
    """
    Proper weighted polynomial fit along last axis.
    Returns coeffs ordered [B0, B1, ..., BN].
    """

    # Vandermonde
    A = np.stack([x**k for k in range(degree + 1)], axis=-1)  # (..., samples, deg+1)

    sw = np.sqrt(w)[..., None]  # sqrt weights

    Aw = A * sw
    yw = y[..., None] * sw

    ATA = np.matmul(np.swapaxes(Aw, -1, -2), Aw)
    ATy = np.matmul(np.swapaxes(Aw, -1, -2), yw)

    return np.linalg.solve(ATA, ATy)[..., 0]


def convertToTraj(chirpArr, method, sliceOffsets, *, sphereDia=55, crossterm=None, sliceIndices=None, gradientFirst=False, trajRes=None):
    if crossterm is None:
        crossterm = False



    mag = np.abs(chirpArr)[:, :, :, :, 0, :, :, :]

    angles = np.angle(chirpArr[:, :, :, :, 1, :, :, :] * np.conj(chirpArr[:, :, :, :, 0, :, :, :]))

    means = np.mean(angles[..., 15:35], -1, keepdims=True)
    angles = angles - means

    angles[..., :15] = 0

    trajValues = np.unwrap(angles)  # rads

    strongInd = np.argmax(np.sum(mag, (0, 1, 2, 3, 4, -1)))

    mag = mag[..., strongInd, :]
    trajValues = trajValues[..., strongInd, :]

    if not crossterm:
        if sliceIndices is not None:
            slicePositions = sliceOffsets[sliceIndices]
            trajValues = trajValues[:, 0, 0, sliceIndices, ...]
            mag = mag[:, 0, 0, sliceIndices, ...]
        else:
            slicePositions = sliceOffsets
            trajValues = trajValues[:, 0, 0, ...]
            mag = mag[:, 0, 0, ...]

        trajValues = np.transpose(trajValues, (0, 3, 2, 1))
        mag = np.transpose(mag, (0, 3, 2, 1))
        w = mag**2 * 1e-16
        y = (
            (trajValues / np.expand_dims(np.array(method["AmpScaler"]), (0, 1, 2)))
            if len(method["AmpScaler"]) > 1 and np.all(method["AmpScaler"] != 1)
            else trajValues
        )

        if gradientFirst:
            if trajRes is None:
                trajRes = 1 / float(method["PVM_EffSWh"])
            y = np.gradient(y, trajRes, axis=1)

        x = np.expand_dims(slicePositions, (0, 1, 2))

        # Move sample axis last if needed (you already did)
        coeffs = weighted_polyfit(x, y, w, 1)

        B0W = coeffs[..., 0]
        trajW = coeffs[..., 1]

        return np.stack((B0W, trajW))

        sumw = np.sum(w, -1)

        sumy = np.sum(w * y, -1)
        sumxx = np.sum(w * x**2, -1)
        sumx = np.sum(w * x, -1)
        sumxy = np.sum(w * x * y, -1)

        den = sumw * sumxx - sumx**2

        B0 = (sumy * sumxx - sumx * sumxy) / den
        traj = (sumw * sumxy - sumx * sumy) / den

        return np.stack((B0, traj))

    signalArray = chirpArr[:, :, :, :, 0, :, :]

    if signalArray.shape[0] > 1:
        raise Exception("Linearity mode in CSI regime not supported")

    mask = np.abs(signalArray[0, ..., 10]) > 0.14 * np.max(np.abs(signalArray[0, ..., 10]))

    circMask = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]))

    PVM_Matrix = method["PVM_Matrix"]

    for sliceInd in range(len(sliceOffsets)):
        sliceOffset = sliceOffsets[sliceInd]
        projectionRadius = np.sqrt((sphereDia / 2) ** 2 - np.abs(sliceOffset) ** 2)

        projectionRadiusRel = projectionRadius / method["PVM_Fov"][0] * PVM_Matrix[0]
        # Get linear index of max
        idx = np.argmax(np.abs(signalArray))

        # Convert to (row, col)
        pos = np.unravel_index(idx, signalArray[0, :, :, sliceInd, 2, 10].shape)

        circMask[:, :, sliceInd] = circular_mask(mask.shape[0], mask.shape[1], center=[pos[0], pos[1]], radius=projectionRadiusRel - 1)

    nx, ny = PVM_Matrix[0], PVM_Matrix[1]
    dx, dy = method["PVM_SpatResol"][0], method["PVM_SpatResol"][1]

    # X coordinates
    X = (np.arange(nx) - nx / 2 + 0.5) * dx - 0.5 * dx

    # Y coordinates
    Y = (np.arange(ny) - ny / 2 + 0.5) * dy - 0.5 * dy

    # Z coordinates
    Z = sliceOffsets  # already in mm
    solution = np.zeros((1, 3, 4, trajValues.shape[-1]))
    for ori in range(trajValues.shape[4]):
        bMatrix = np.zeros((int(np.sum(mask[..., ori] * circMask)), trajValues.shape[5]))
        AMatrix = np.zeros((int(np.sum(mask[..., ori] * circMask)), 4))

        ind = 0

        signal = np.abs(signalArray[0, :, :, :, ori, :])

        for x in range(trajValues.shape[1]):
            for y in range(trajValues.shape[2]):
                for z in range(trajValues.shape[3]):
                    if not circMask[x, y, z]:
                        continue

                    if not mask[x, y, z, ori]:
                        continue

                    if np.any(signal[x, y, z, 10:] < 0.1 * np.max(signal[x, y, z, :])):
                        continue

                    bMatrix[ind, :] = spectFilt(trajValues[0, x, y, z, ori, :], method["PVM_EffSWh"], 200000)
                    AMatrix[ind, 0] = 1
                    AMatrix[ind, 1] = X[x]
                    AMatrix[ind, 2] = Y[y]
                    AMatrix[ind, 3] = Z[z]
                    ind += 1

        bMatrix = bMatrix[:ind, :]
        AMatrix = AMatrix[:ind, :]
        solution_all, _, _, _ = np.linalg.lstsq(AMatrix, bMatrix, rcond=None)
        solution[0, ori, :, :] = solution_all
        pass

    return np.transpose(solution, (2, 0, 3, 1))


def circshift_subpixel(img, shift_y, shift_x):
    """
    Circular sub-pixel shift using Fourier-domain phase ramps.
    shift_x, shift_y can be fractional (e.g., ±0.5)
    """
    ny, nx = img.shape[:2]

    # frequency coordinates
    ky = np.fft.fftfreq(ny).reshape(-1, 1)  # column vector
    kx = np.fft.fftfreq(nx).reshape(1, -1)  # row vector

    # 2D FFT
    F = np.fft.fftn(img, axes=(0, 1))

    # phase ramp for translation
    phase = np.exp(-2j * np.pi * (ky * shift_y + kx * shift_x))

    # apply shift
    F_shifted = F * np.expand_dims(phase, (-1, -2, -3, -4, -5))

    # inverse FFT
    return np.fft.ifftn(F_shifted, axes=(0, 1)).real


def getData(scanPath, *, outRes=2e-6, sliceIndices=None, frequencyFilter=None, gradientFirst=False):
    if type(scanPath) is str:
        scanPath = Path(scanPath)

    data, nav = ReadJob(scanPath)
    method = ReadParamFile(scanPath / "method")
    acqp = ReadParamFile(scanPath / "acqp")
    nSlices = method["PVM_SPackArrNSlices"][0]
    nCoil = data.shape[0]

    crossterm = method.get("CSI_Regime", False) == "Yes"

    if crossterm:
        ph1Steps = method["PVM_EncMatrix"][0]
        ph2Steps = method["PVM_EncMatrix"][0]
        phSteps = len(method["PVM_EncGenSteps0"])

    else:
        ph1Steps = 1
        ph2Steps = 1
        phSteps = 1

    nLinSteps = method["LinearitySteps"]

    chirpArr = np.zeros((nLinSteps, ph1Steps, ph2Steps, nSlices, 2, 3, nCoil, data.shape[1]), dtype=np.complex128)

    sliceOffsets = method["PVM_SliceOffset"][:nSlices]  # mm

    ind = 0

    minPh0 = np.min(method["PVM_EncSteps0"])
    minPh1 = np.min(method["PVM_EncSteps1"])

    phCorr = np.angle(np.mean(nav[:, 10:-10, :], 1)) if nav is not None else np.ones_like(data)[0, [0], ...]

    indx = np.array(method["PVM_EncGenSteps0"], dtype=np.int64) - minPh0 if crossterm else np.array(0)
    indy = np.array(method["PVM_EncGenSteps1"], dtype=np.int64) - minPh1 if crossterm else np.array(0)

    phCorrInd = 0

    for lin in range(nLinSteps):
        for ori in range(3):
            for indPh in range(phSteps):
                for exp in range(2):
                    phCorrInst = phCorr[:, [phCorrInd]]
                    phCorrInd += 1
                    for i in range(nSlices):
                        if crossterm:
                            chirpArr[
                                lin,
                                indx[indPh],
                                indy[indPh],
                                i,
                                exp,
                                ori,
                                :,
                                :,
                            ] = data[:, :, ind] * np.exp(-1j * phCorrInst)
                        else:
                            chirpArr[
                                lin,
                                0,
                                0,
                                i,
                                exp,
                                ori,
                                :,
                                :,
                            ] = data[:, :, ind] * np.exp(-1j * phCorrInst)
                        ind += 1

    if crossterm:  # do simple CSI reco
        chirpArr = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(chirpArr, axes=(1, 2)), axes=(1, 2)), axes=(1, 2))

    gradCalConst = method["PVM_GradCalConst"]
    gradRes = 2e-6  # should be read from method but is kept at 2us anyways;

    testShape = (
        np.array(method.get("Chirp", method.get("TestShapeVec", None)))
        * method.get("ChirpAmplitude", method.get("TestShapeAmplitude", None))
        * gradCalConst
        * 2
        * np.pi
        * 1e-2
    )  # rad/smm

    trajRes = get_acquisition_dwell(method, acqp)

    envelope = np.sqrt(np.sum(np.abs(chirpArr) ** 2, -2))
    envelope = np.mean(envelope, axis=3)

    envelope = np.clip(np.nan_to_num(envelope[..., 1, :, :] / envelope[..., 0, :, :], nan=1e-12, posinf=1e-12, neginf=1e-12), 1e-12, 1)
    envelope = spectFilt(envelope, 1 / trajRes, 50e3, axis=-1)
    testShape = resampleArr(testShape, gradRes, outRes)

    envelope = resampleArr(envelope, trajRes, outRes, axis=-1)
    if not crossterm:
        envelope = envelope[:, 0, 0, :, :]  # remove penc dims

    tAxGrad = np.arange(testShape.size) * outRes

    polynomials = convertToTraj(
        chirpArr,
        method,
        sliceOffsets,
        crossterm=crossterm,
        sliceIndices=sliceIndices,
        gradientFirst=gradientFirst,
        trajRes=trajRes,
    )
    polynomialsFilt = spectFilt(polynomials, 1 / trajRes, axis=2, fstop=frequencyFilter) if frequencyFilter else polynomials

    polynomials = resampleArr(polynomialsFilt, trajRes, outRes, axis=2, tOut=tAxGrad)

    return polynomials, testShape, envelope


def load_config(path="config.json"):
    cfg_path = ResolveProjectPath(path)

    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)

    return {}


def LoadTrainingDataConfig(config, key="trainingData"):
    dataConfig = config[key]
    dataPaths = dataConfig["dataPaths"]
    scansS = dataConfig["scans"]

    if len(dataPaths) == 0:
        raise ValueError(f"Config section '{key}' must define at least one data path")

    if len(dataPaths) != len(scansS):
        raise ValueError(f"Config section '{key}' must contain the same number of dataPaths and scans entries")

    return [str(ResolveProjectPath(path)) for path in dataPaths], [[int(scan) for scan in scans] for scans in scansS]


def LoadTestingShapeTypes(config, key="trainingData"):
    return set(config[key]["testingShapes"])


def parse_args(description, configKey):
    parser = argparse.ArgumentParser(description=description)

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("data_path", nargs="?", default=None)
    parser.add_argument("scan", nargs="?", default=None)

    parser.add_argument("--outRes", type=float, default=None)

    args = parser.parse_args()

    config = load_config().get(configKey, {})

    dataPath = ResolveProjectPath(args.data_path or config.get("dataPath", "."))

    scan = args.scan or config.get("scan", "1")

    outRes = args.outRes if args.outRes is not None else config.get("outRes", 2e-6)

    return dataPath, scan, outRes
