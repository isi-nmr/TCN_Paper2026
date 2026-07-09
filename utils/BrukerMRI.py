import os
import re
from pathlib import Path

import numpy as np

"""

Jiri Vitous 2025

"""


regexNew = r"##\$(\w+)\s*= ?(?:\( ?([\d,\. ]+) ?\))?\s*(<[\w_\/=\. \(\)]+>|[\w\s,\d\.\(\)\-<>\+\$@*;#:_\/ ]+)(?=\s##\$|\s\$\$|##END=)"
regexPar = r"\#\#\$([\w _\d]+)="

pat = re.compile(regexNew)

patPar = re.compile(regexPar, flags=0)

tupleRegex = re.compile(r"([^,()]+)", flags=0)

patRLE = r"@(\d+)\*\(([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\)"
regexRLE = re.compile(patRLE)


def ReadRawData(filepath):
    with open(filepath) as f:
        return np.fromfile(f, dtype=np.int32)


def ReadProcessedData(filepath, Visu):
    DataType = Visu["VisuCoreWordType"]

    if DataType == "_32BIT_SGN_INT":
        format = np.int32
    elif DataType == "_16BIT_SGN_INT":
        format = np.int16
    elif DataType == "_32BIT_FLOAT":
        format = np.float32
    elif DataType == "_64BIT_FLOAT":
        format = np.float64
    elif DataType == "_8BIT_UNSGN_INT":
        format = np.uint8
    else:
        raise Exception("Wrong Data type")

    with open(filepath) as f:
        data = np.fromfile(f, dtype=format)

    if len(Visu["VisuCoreSize"]) > 2:
        # Visu Core is 3D
        data = data.reshape(
            Visu["VisuCoreSize"][0],
            Visu["VisuCoreSize"][1],
            Visu["VisuCoreSize"][2],
            -1,
            order="F",
        )
        # if data.ndim == 4:
        #     data_length = data.shape[-1]
        # else:
        #     data_length = 1

    else:
        # Visu Core is 2D
        data = data.reshape(Visu["VisuCoreSize"][0], Visu["VisuCoreSize"][1], -1, order="F")
        # if data.ndim == 3:
        #     data_length = data.shape[-1]
        # else:
        #     data_length = 1

        # data_reshaped = np.zeros([data.shape[1], data.shape[0], data_length])
        # for i in range(0, data_length):
        #     data_reshaped[:, :, i] = np.rot90(data[:, :, i])

    data = np.swapaxes(data, 0, 1)

    return data.astype(np.float64)


def getParameter(filepath, parName):
    parNameRegex = f"##\\${parName}\\s*=(?:\\( ([\\d, ]+) \\))?\\s*(<[\\w_\\/=\\. \\(\\))]+>|[\\w\\s,\\d\\.\\(\\)\\-<>\\+\\$;@*#:_\\/ ]+)(?=\\s##\\$|\\s\\$\\$|##END=)"

    with open(filepath) as f:
        dataIn = f.read()

    parameter = re.search(parNameRegex, dataIn)
    if parameter is None:
        return None

    return parseValue(parameter[2]) if parameter[1] == "" or parameter[1] is None else parseArray(parameter[2], parameter[1])


def canBeInt(value):
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False


def canBeFloat(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def getShape(strShape):
    if ".." in strShape:
        splitStr = strShape.split("..")
        return int(splitStr[1]) + 1

    if "," in strShape:
        return np.array(strShape.split(", "), dtype=np.int64)

    return int(strShape)


def parseTupleArray(structString):
    structs = re.findall(r"\(([^()]+)\)", structString)
    struct_arrlist = []
    if not structs:
        return 0
    for string in structs:
        struct_arrlist.append(parseTuple(string))  # noqa: PERF401

    return struct_arrlist


def parseArrayWRLE(string):
    matches = regexRLE.findall(string)
    splitStr = string.split(" ")

    arrLen = len(splitStr) - len(matches)

    for m in matches:
        arrLen += int(m[0])

    outArray = np.zeros(arrLen, dtype=float)
    ind = 0
    matchInd = 0
    for _, el in enumerate(splitStr):
        if "@" in el:
            outArray[ind : ind + int(matches[matchInd][0])] = float(matches[matchInd][1])
            ind += int(matches[matchInd][0])
            matchInd += 1
        else:
            outArray[ind] = float(el)
            ind += 1

    return outArray


def parseArray(str, num):
    str = str.replace("\n", "")

    arrSize = getShape(num)
    if str[0] == "(":
        return parseTupleArray(str)

    if regexRLE.search(str):
        return parseArrayWRLE(str)

    if str[0] == "<":
        return str.replace("<", "").replace(">", "")

    splitString = str.split(" ")

    if canBeInt(splitString[0]) and "." not in str:
        arr = np.array(splitString, dtype=int)
        if not np.isscalar(arrSize):
            return np.reshape(arr, arrSize)
        return arr

    if canBeFloat(splitString[0]):
        if splitString[-1] == "":
            del splitString[-1]

        arr = np.array(splitString, dtype=float)
        return np.reshape(arr, arrSize)

    if str[0] == "(" and arrSize == 1:
        return [parseTuple(str)]

    return splitString


def parseValue(data):
    if canBeInt(data):
        return int(data)

    if canBeFloat(data):
        return float(data)

    if data[-1] == "\n":
        data = data[:-1]

    if data[0] == " ":
        data = data[1:]

    if data[0] == "(":
        return parseTuple(data)

    if data[0] == "<":
        data = data.replace("<", "").replace(">", "")
    return data


def parseTuple(data):
    data = data.replace("\n", "")

    tupleMatch = re.findall(tupleRegex, data)

    for ind, _ in enumerate(tupleMatch):
        tupleMatch[ind] = parseValue(tupleMatch[ind])

    return tupleMatch


def ReadParamFile(path):
    with open(path) as f:
        dataIn = f.read()
    data = re.findall(pat, dataIn)

    dictOut = {}

    for match in data:
        if "PreempFilters" in match[0]:
            pass

        if match[1] == "":
            dictOut[match[0]] = parseValue(match[2])
        else:
            dictOut[match[0]] = parseArray(match[2], match[1])

    return dictOut


# ***********************************************************
# -----------------------------------------------------------
# ***********************************************************


if __name__ == "__main__":
    pass


"""

Libs functionality added by Jiri Vitous and ISI Brno team 2023+

"""


def ReadFid(fidpath):
    acqp = ReadParamFile(fidpath + "acqp")
    fidOrig = ReadRawData(fidpath + "fid")

    fidOrig = fidOrig[0::2] + 1j * fidOrig[1::2]  # combine real and imaginary

    bits = 32

    blockSize = int(np.ceil(acqp["ACQ_size"][0] * acqp["ACQ_ReceiverSelect"].count("Yes") * (bits / 8) / 1024) * 1024 / (bits / 8) / 2)
    fid = np.reshape(fidOrig, [blockSize, -1], order="F")
    fid = fid[: int(0.5 * acqp["ACQ_size"][0] * acqp["ACQ_ReceiverSelect"].count("Yes")), :]

    numSelrec = acqp["ACQ_ReceiverSelect"].count("Yes")
    fidOut = np.zeros([numSelrec, int(fid.shape[0] / numSelrec), fid.shape[1]], dtype=np.complex64)
    for i in range(fid.shape[1]):
        for j in range(numSelrec):
            fidOut[j, :, i] = fid[fidOut.shape[1] * j : fidOut.shape[1] * (j + 1), i]
    return fidOut


def ReadJob(fidpath):
    if type(fidpath) is str:
        fidpath = Path(fidpath)

    acqp = ReadParamFile(fidpath / "acqp")
    numSelrec = acqp["ACQ_ReceiverSelect"].count("Yes")

    def map_and_reshape(filepath, job_idx):
        """Map Bruker rawdata file directly into shape (numSelrec, pointsPerReceiver, nAcq)"""
        if not os.path.isfile(filepath):
            return None

        # Determine block size
        blockLen = int(acqp["ACQ_jobs"][job_idx][0] * numSelrec / 2)

        # Map the file as int32
        fidOrig = np.memmap(filepath, dtype=np.int32, mode="r")

        # Combine real + imaginary: shape -> (numComplex,)
        fidOrig = fidOrig.reshape(-1, 2)
        fidComplex = fidOrig[:, 0] + 1j * fidOrig[:, 1]

        # Number of acquisitions
        nAcq = fidComplex.size // blockLen

        # Reshape into (pointsPerReceiver, numSelrec, nAcq) in Fortran order
        fidReshaped = fidComplex.reshape(blockLen, nAcq, order="F")
        pointsPerReceiver = blockLen // numSelrec

        # Slice into receivers: (numSelrec, pointsPerReceiver, nAcq)
        job = fidReshaped.reshape(numSelrec, pointsPerReceiver, nAcq)
        return job.astype(np.complex64)

    # Map job0
    job0_path = fidpath / "rawdata.job0"
    job0 = map_and_reshape(job0_path, 0)

    # Map job1 or navigator
    if os.path.isfile(fidpath / "rawdata.job1") or os.path.isfile(fidpath / "rawdata.Navigator"):
        fname = fidpath / "rawdata.job1" if os.path.isfile(fidpath / "rawdata.job1") else fidpath / "rawdata.Navigator"
        job1 = map_and_reshape(fname, 1)
    else:
        job1 = None

    return job0, job1


def ReadTraj(filepath, ndim, nSamples):
    with open(filepath) as f:
        traj = np.fromfile(f, dtype=np.float64)

    kx = np.zeros((nSamples, int(len(traj) / ndim / nSamples)))
    ky = np.zeros((nSamples, int(len(traj) / ndim / nSamples)))
    kz = np.zeros((nSamples, int(len(traj) / ndim / nSamples)))
    temp = 0
    for i in range(0, len(traj) - ndim, nSamples * ndim):
        kx[:, temp] = traj[i : i + ndim * nSamples - 1 : ndim]
        ky[:, temp] = traj[i + 1 : i + ndim * nSamples : ndim]
        if ndim > 2:
            kz[:, temp] = traj[i + 2 : i + ndim * nSamples + 1 : ndim]
        temp = temp + 1

    return kx, ky, kz


"""
Main Function for importing non cartesian data
"""


def ImportBrkrNonCart(fidpath):
    fid = ReadFid(fidpath)

    methodFile = ReadParamFile(fidpath + "method")

    digNp = methodFile["PVM_DigNp"]

    projections = fid[:, -digNp:, :]

    kx, ky, kz = ReadTraj(fidpath + "traj", len(methodFile["PVM_Matrix"]), digNp)

    """
    Crop dummy
    """
    if "NProjPerInv" in methodFile:
        D = methodFile["NProjPerInv"] - methodFile["PVM_DummyScans"]
        projections = projections[:, :, D:]
        kx = kx[:, D:]
        ky = ky[:, D:]
        kz = kz[:, D:]
        navigator = fid[:, : fid.shape[1] - digNp, D:]
    else:
        navigator = fid[:, : fid.shape[1] - digNp, :]
    return projections, kx, ky, kz, navigator


def makeScanList(dataPath):
    dir_list = os.listdir(dataPath)
    scan_names = {}
    for dir in dir_list:
        if os.path.exists(dataPath + "/" + dir + "/acqp"):
            acqp = ReadParamFile(dataPath + "/" + dir + "/acqp")
            if "ACQ_scan_name" in acqp:
                name = acqp["ACQ_scan_name"]
                if type(name) is list:
                    scan_names[acqp["ACQ_scan_name"][0][0]] = dataPath + "/" + dir + "/"
                else:
                    scan_names[acqp["ACQ_scan_name"]] = dataPath + "/" + dir + "/"

    return scan_names


def read2dSeq(scanFolder, tdSeqSubfolder="1"):
    acqp = ReadParamFile(scanFolder + "/acqp")
    method = ReadParamFile(scanFolder + "/method")

    VisuPars = ReadParamFile(scanFolder + "/pdata/" + tdSeqSubfolder + "/visu_pars")

    # Data temp is in Core X number of frames
    DataTemp = ReadProcessedData(scanFolder + "/pdata/" + tdSeqSubfolder + "/2dseq", VisuPars)
    # If there are no frames add frame dimensionality
    if len(DataTemp.shape) == VisuPars["VisuCoreDim"]:
        DataTemp = np.expand_dims(DataTemp, axis=-1)

    # Rescale Data
    if VisuPars["VisuSeriesTypeId"] != "DERIVED_ISA":
        if "RG" in acqp:
            try:
                RG = acqp["RG"][0]
            except Exception:
                RG = acqp["RG"]
        else:  # For PV 360 import.
            try:
                RG = method["PVM_RgValue"][0]
            except Exception:
                RG = method["PVM_RgValue"]
    else:
        RG = 1

    for frame in range(DataTemp.shape[-1]):
        if len(VisuPars["VisuCoreDataSlope"]) > 1:
            DataTemp[..., frame] = DataTemp[..., frame] * VisuPars["VisuCoreDataSlope"][frame] * RG + VisuPars["VisuCoreDataOffs"][frame]
        else:
            DataTemp[..., frame] = DataTemp[..., frame] * VisuPars["VisuCoreDataSlope"] * RG + VisuPars["VisuCoreDataOffs"]

    if "VisuFGOrderDesc" not in VisuPars:
        print("No VisuFGOrderDesc in VisuPars, returning core")
        return DataTemp

    # Resolve dimensionality
    dims = []
    dim_desc = []
    for order in VisuPars["VisuFGOrderDesc"]:
        dims.append(order[0])
        dim_desc.append(order[1])

    # Data after FG framing
    TotalDims = list(DataTemp.shape[:-1]) + dims

    DataBrkr = np.copy(DataTemp)
    DataBrkr = np.reshape(DataBrkr, TotalDims, order="F")

    # Formation of Bruker data into Perflab shape

    CoreDimDesc = ["x", "y"] if VisuPars["VisuCoreDim"] == 2 else ["x", "y", "FG_SLICE"]

    # If we have FG Slice object in the file and core is 3D we need to distingush the SLICE of 3rd dimension
    # and possible SlicePacks
    if VisuPars["VisuCoreDim"] > 2 and "FG_SLICE" in dim_desc:
        dim_desc[dim_desc.index("FG_SLICE")] = "FG_SLICE_PACK"

    TotalDimDesc = CoreDimDesc + dim_desc

    # This Describes data order in the resulting Data1/2 variable

    # Make ASL Compatible

    if "FG_IRMODE" in TotalDimDesc and "FG_MOVIE" in TotalDimDesc:
        ExpectedDimDesc = ["x", "y", "FG_SLICE", "FG_MOVIE", "FG_IRMODE"]
    ExpectedDimDesc = ["x", "y", "FG_SLICE", "FG_MOVIE", "FG_ISA"] if "FG_ISA" in dim_desc else ["x", "y", "FG_SLICE", "FG_CYCLE", "FG_ECHO"]

    transform = []
    ExistingDims = []
    for item in ExpectedDimDesc:
        try:
            transform.append(TotalDimDesc.index(item))
            ExistingDims.append(item)
        except Exception:
            pass

    DimList = list(np.linspace(0, len(TotalDimDesc) - 1, len(TotalDimDesc)).astype(int))

    UnwantedDims = [i for i in DimList if i not in transform]

    TotalDimIndexesReord = transform + UnwantedDims

    DataBrkr = np.transpose(DataBrkr, TotalDimIndexesReord)
    # Update the dimension description by reordering the list
    TotalDimDesc = [TotalDimDesc[i] for i in TotalDimIndexesReord]
    TotalDims = [TotalDims[i] for i in TotalDimIndexesReord]

    # Here we have only dimensions (in the same order as wanted) that we need,
    # but some of them might be missing, we have to fill "ones" in there
    # e.g. for 2D MGE: Input:(x,y,ECHO) -> Output:(x,y,1,1,ECHO) -> one slice one time frame

    DimSizeList = []
    for item in ExpectedDimDesc:
        try:
            IndexOfItem = TotalDimDesc.index(item)
            DimSizeList.append(TotalDims[IndexOfItem])
        except Exception:
            DimSizeList.append(1)

    return np.reshape(DataBrkr, DimSizeList, order="F")
