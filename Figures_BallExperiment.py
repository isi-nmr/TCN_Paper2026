import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.BrukerMRI import *
from utils.GradientCorrector import GradientCorector
from utils.GradientCorrectorML import GradientCorectorML
from utils.utils import PaperDataPath

LOCAL_BART_FALLBACK = Path("/home/vitous/git_work/uipp/bart/bart10")


def findBartToolbox():
    bartCommand = shutil.which("bart")
    if bartCommand is None:
        return LOCAL_BART_FALLBACK

    bartCommand = Path(bartCommand).resolve()
    candidates = [
        Path(os.environ["BART_TOOLBOX_PATH"]).expanduser() if "BART_TOOLBOX_PATH" in os.environ else None,
        Path(os.environ["TOOLBOX_PATH"]).expanduser() if "TOOLBOX_PATH" in os.environ else None,
        bartCommand.parent,
        bartCommand.parent.parent,
    ]

    for candidate in candidates:
        if candidate is not None and (candidate / "python").exists():
            return candidate

    return bartCommand.parent


def setupBart():
    bartPath = findBartToolbox()
    os.environ["TOOLBOX_PATH"] = str(bartPath)
    os.environ["BART_TOOLBOX_PATH"] = str(bartPath)
    os.environ["PATH"] = str(bartPath) + os.pathsep + os.environ.get("PATH", "")
    os.environ["OMP_NUM_THREADS"] = "8"

    bartPythonPath = bartPath / "python"
    if bartPythonPath.exists():
        sys.path.insert(0, str(bartPythonPath))
    sys.path.insert(0, str(bartPath))


setupBart()

import cfl
from bart import bart

"""
+------+----------+-----------------------------+
|   ID |   Folder | ScanName                    |
+======+==========+=============================+
|   10 |       10 | radial_CS 256 400 (E10)     |
+------+----------+-----------------------------+
|   14 |       14 | radial_CS 256 500 (E14)     |
+------+----------+-----------------------------+
|   15 |       15 | radial_CS 256 600 (E15)     |
+------+----------+-----------------------------+
|   16 |       16 | radial_CS 256 700 (E16)     |
+------+----------+-----------------------------+
|   12 |       12 | radial_CS 256 800 (E12)     |
+------+----------+-----------------------------+
|   13 |       13 | radial_CS 256 900 (E13)     |
+------+----------+-----------------------------+
|   11 |       11 | radial_CS 256 1000 (E11)    |
+------+----------+-----------------------------+
|   17 |       17 | radial_CS 256 400 GA (E17)  |
+------+----------+-----------------------------+
|   18 |       18 | radial_CS 256 500 GA (E18)  |
+------+----------+-----------------------------+
|   19 |       19 | radial_CS 256 600 GA (E19)  |
+------+----------+-----------------------------+
|   20 |       20 | radial_CS 256 700 GA (E20)  |
+------+----------+-----------------------------+
|   21 |       21 | radial_CS 256 800 GA (E21)  |
+------+----------+-----------------------------+
|   22 |       22 | radial_CS 256 900 GA (E22)  |
+------+----------+-----------------------------+
|   23 |       23 | radial_CS 256 1000 GA (E23) |
+------+----------+-----------------------------+
"""

gCorr = GradientCorectorML()
gCorrGIRF = GradientCorector("AV-NEO,BGA-12,AfterBrukerTuneup")

studyFolder = str(PaperDataPath("radial_ball_phantom"))


scans = [17, 18, 19, 20, 21, 22, 23]

imageList = []
imageTeoList = []
imageGirfList = []
imageB0List = []
imageEstList = []
imageEstB0List = []
imageEstB0ModelList = []
bwList = []

for scan in scans:
    scanPath = studyFolder + "/" + str(scan) + "/"

    methodFile = ReadParamFile(scanPath + "/method")

    bwList.append(methodFile["PVM_EffSWh"])

    acqp = ReadParamFile(scanPath + "/acqp")
    fid, _ = ReadJob(scanPath)

    trajXM = np.expand_dims(methodFile["PVM_TrajKx"], -1)
    trajYM = np.expand_dims(methodFile["PVM_TrajKy"], -1)

    gx = np.gradient(trajXM, axis=0).T
    gy = np.gradient(trajYM, axis=0).T

    gx = gx[:, :-1]
    gy = gy[:, :-1]

    digNp = int(fid.shape[1] - methodFile["RadRead_ReadDephPoints"] if methodFile["RadRead_AcqMode"] == "ECHO" else fid.shape[1])

    traj, bCorr = gCorr.generateCorrectionsRadialCS(methodFile, acqp)
    traj = gCorr.convertTrajToBartScale(methodFile, traj, oversampling=2)
    traj[:, :, 2, :] = 0
    traj = np.transpose(traj[..., 0], (2, 0, 1))
    traj = traj[:, -digNp:, :]

    trajGirf, bCorrGirf = gCorrGIRF.generateCorrectionsRadialCS(methodFile, acqp)
    trajGirf = gCorrGIRF.convertTrajToBartScale(methodFile, trajGirf, oversampling=4)
    trajGirf[:, :, 2, :] = 0
    trajGirf = np.transpose(trajGirf[..., 0], (2, 0, 1))
    trajGirf = trajGirf[:, -digNp:, :]

    trajTeo, _ = gCorrGIRF.generateCorrectionsRadialCS(methodFile, acqp, theoretical=True)
    trajTeo = gCorrGIRF.convertTrajToBartScale(methodFile, trajTeo, oversampling=4)
    trajTeo[:, :, 2, :] = 0
    trajTeo = np.transpose(trajTeo[..., 0], (2, 0, 1))
    trajTeo = trajTeo[:, -digNp:, :]

    phaseTot = np.transpose(bCorr, (2, 0, 1))

    rawData = fid.reshape((fid.shape[0], fid.shape[1], 1, -1), order="F")

    rawData = rawData[:, :, :, :]

    recoData = rawData[:, -digNp:, ...]
    navig = rawData[:, 3:14, ...]

    phaseNavig = np.angle(np.mean(navig, 1)[0])

    trajM = np.expand_dims(np.concatenate((trajXM, trajYM), -1).T[:, :-10], -1)

    gradAmps = np.expand_dims(
        np.concatenate((np.expand_dims(methodFile["RadRead_GradAmpR"], 0), np.expand_dims(methodFile["RadRead_GradAmpP"], 0)), 0), 1
    )

    trajM = trajM * gradAmps

    trajM = trajM[:, -digNp:, :]
    # trajM = trajM / np.max(np.abs(trajM)) * methodFile["PVM_Matrix"][0] * methodFile["RadRead_ReadOversample"] / 2

    trajM = trajM / np.max(np.abs(trajM)) * np.max(np.abs(traj))

    trajZM = np.zeros((1, trajM.shape[1], trajM.shape[2]))

    trajM = np.concatenate((trajM, trajZM), 0)

    phaseCorModel = np.expand_dims(phaseTot[:, -digNp:, :], (-1, -2, -3))

    recoData = np.expand_dims(np.transpose(np.expand_dims(recoData, 0), [0, 2, 4, 1, 3]), -2)

    maxInds = np.argmax(np.sqrt(np.sum(np.squeeze(recoData**2), -1)), 0)
    phases = np.zeros_like(maxInds, dtype=np.double)
    for i in range(len(maxInds)):
        phases[i] = np.angle(recoData[0, int(maxInds[i]), i, 0, 0, 0])

    projAngles = np.atan2(methodFile["RadRead_GradAmpR"], methodFile["RadRead_GradAmpP"])

    phCorr = np.expand_dims(phases, (0, 1, -1, -2, -3))

    recoDataPhase = recoData * np.exp(-1j * phCorr)
    recoDataPhaseModel = recoData * np.exp(-1j * phaseCorModel)  # phaseCorModel

    maxInds = np.argmax(np.sqrt(np.sum(np.squeeze(recoDataPhaseModel**2), -1)), 0)
    phases = np.zeros_like(maxInds, dtype=np.double)
    for i in range(len(maxInds)):
        phases[i] = np.angle(recoDataPhaseModel[0, int(maxInds[i]), i, 0, 0, 0])

    projAngles = np.atan2(methodFile["RadRead_GradAmpR"], methodFile["RadRead_GradAmpP"])

    phCorr = np.expand_dims(phases, (0, 1, -1, -2, -3))

    maxInds = np.argmax(np.sqrt(np.sum(np.squeeze(recoData**2), -1)), 0)
    phases = np.zeros_like(maxInds, dtype=np.double)
    for i in range(len(maxInds)):
        phases[i] = np.angle(recoData[0, int(maxInds[i]), i, 0, 0, 0])

    projAngles = np.atan2(methodFile["RadRead_GradAmpR"], methodFile["RadRead_GradAmpP"])

    b = 14
    a = 110

    # timeStamp = time.time()

    timestamp = "11"
    os.makedirs("./tmp", exist_ok=True)
    trajPath = "./tmp/traj" + timestamp
    patPath = "./tmp/pat" + timestamp
    kspacePath = "./tmp/kSpace" + timestamp

    bartCommand = ""

    # bartCommand += f"nlinv {reg} --reg-iter 350 -d4 -a {float(a)} -b {float(b)} -g -S -x {newTotalSize}:{newTotalSize}:{1}"
    bartCommand += f"nlinv -d4 -a {float(a)} -b {float(b)} -g -S -x {methodFile['PVM_Matrix'][0]}:{methodFile['PVM_Matrix'][1]}:{1}"

    bartCommand += " -i 20"

    bartCommand += f" -t {trajPath}"
    bartCommand += f" {kspacePath}"

    cfl.writecfl(kspacePath, recoData[:, :, :, ...])
    cfl.writecfl(trajPath, trajTeo[:, :, :, ...])
    imageTeoTraj, maps = bart(2, bartCommand)

    cfl.writecfl(kspacePath, recoData[:, :, :, ...])
    cfl.writecfl(trajPath, trajM[:, :, :, ...])
    imageMeasTraj, maps = bart(2, bartCommand)

    cfl.writecfl(kspacePath, recoData[:, :, :, ...])
    cfl.writecfl(trajPath, trajGirf[:, :, :, ...])
    imageGirfTraj, maps = bart(2, bartCommand)

    cfl.writecfl(kspacePath, recoDataPhase[:, :, :, ...])
    cfl.writecfl(trajPath, trajM[:, :, :, ...])
    imageMeasTrajB0, maps = bart(2, bartCommand)

    cfl.writecfl(trajPath, traj[:, :, :, ...])
    imageEstTraj, maps = bart(2, bartCommand)

    cfl.writecfl(trajPath, traj[:, :, :, ...])
    cfl.writecfl(kspacePath, recoDataPhase[:, :, :, ...])
    imageEstTrajB0, maps = bart(2, bartCommand)

    cfl.writecfl(trajPath, traj[:, :, :, ...])
    cfl.writecfl(kspacePath, recoDataPhaseModel[:, :, :, ...])
    imageEstTrajB0Model, maps = bart(2, bartCommand)

    imageList.append(imageMeasTraj)
    imageTeoList.append(imageTeoTraj)
    imageGirfList.append(imageGirfTraj)
    imageB0List.append(imageMeasTrajB0)
    imageEstList.append(imageEstTraj)
    imageEstB0List.append(imageEstTrajB0)
    imageEstB0ModelList.append(imageEstTrajB0Model)

    pass


pass


from matplotlib.gridspec import GridSpec


def formatBw(bwHz):
    if bwHz >= 1e6:
        return f"{bwHz / 1e6:g} MHz"
    return f"{bwHz / 1e3:g} kHz"


nCols = len(imageList)
rowLabels = ["Theoretical", "Measured", "GIRF Corrected", "Estimated"]
rowLabels.extend(["Estimated B0 Model"])
nRows = len(rowLabels)

imageAspect = np.abs(imageTeoList[0]).shape[0] / np.abs(imageTeoList[0]).shape[1]
cellWidth = 2.2
fig = plt.figure(figsize=(cellWidth * nCols, cellWidth * imageAspect * nRows + 0.35), facecolor="black")
gs = GridSpec(nRows, nCols, figure=fig, wspace=0, hspace=0)

for c in range(nCols):
    images = [
        np.abs(imageTeoList[c]),
        np.abs(imageList[c]),
        np.abs(imageGirfList[c]),
        np.abs(imageEstList[c]),
    ]
    images.extend([np.abs(imageEstB0ModelList[c])])

    for r in range(nRows):
        ax = fig.add_subplot(gs[r, c])
        ax.set_facecolor("black")

        ax.imshow(
            images[r],
            cmap="gray",
            interpolation="nearest",
            aspect="equal",
            vmin=0,
            vmax=0.3 * np.max(images[r]),
        )
        ax.set_box_aspect(images[r].shape[0] / images[r].shape[1])

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

        if r == 0:
            ax.set_title(formatBw(bwList[c]), fontsize=12, pad=4, color="white")

        if c == 0:
            ax.set_ylabel(rowLabels[r], fontsize=12, labelpad=25, color="white")

plt.subplots_adjust(
    left=0.05,
    right=1,
    bottom=0,
    top=0.93,
)

plt.show()
os.makedirs("paper2026", exist_ok=True)
fig.savefig("paper2026/BallGrid.png", dpi=600, facecolor="black")
fig.savefig("paper2026/BallGrid.pdf", facecolor="black")

plt.clf()
"""
SSIM
"""

from skimage.metrics import structural_similarity as ssim

ref = np.abs(imageB0List[0])

ssimResults = {"Theoretical": [], "Measured": [], "GIRF": [], "Estimated": [], "Estimated B0 Model": []}

borderPixels = 10

# create frame mask
mask = np.zeros_like(ref, dtype=bool)
mask[borderPixels:-borderPixels, borderPixels:-borderPixels] = True

for i in range(len(imageList)):
    _, ssimMap = ssim(ref, np.abs(imageTeoList[i]), full=True, data_range=ref.max() - ref.min())
    ssimResults["Theoretical"].append(np.median(ssimMap[mask]))

    _, ssimMap = ssim(ref, np.abs(imageList[i]), full=True, data_range=ref.max() - ref.min())
    ssimResults["Measured"].append(np.median(ssimMap[mask]))

    _, ssimMap = ssim(ref, np.abs(imageGirfList[i]), full=True, data_range=ref.max() - ref.min())
    ssimResults["GIRF"].append(np.median(ssimMap[mask]))

    _, ssimMap = ssim(ref, np.abs(imageEstList[i]), full=True, data_range=ref.max() - ref.min())
    ssimResults["Estimated"].append(np.median(ssimMap[mask]))

    # _, ssim_map = ssim(ref, np.abs(imageEstB0List[i]), full=True, data_range=ref.max() - ref.min())
    # ssim_results["Estimated B0"].append(np.median(ssim_map[mask]))

    _, ssimMap = ssim(ref, np.abs(imageEstB0ModelList[i]), full=True, data_range=ref.max() - ref.min())
    ssimResults["Estimated B0 Model"].append(np.median(ssimMap[mask]))


bwKhz = np.array(bwList) * 1e-3


def latexEscape(text):
    return text.replace("_", r"\_")


def writeSsimLatexTable(bwValuesKhz, results, outPath):
    methods = ["Theoretical", "Measured", "GIRF", "Estimated"]
    methods.append("Estimated B0 Model")

    columnSpec = "r" + "r" * len(methods)
    lines = [
        rf"\begin{{tabular}}{{{columnSpec}}}",
        r"\hline",
        "Bandwidth (kHz) & " + " & ".join(latexEscape(method) for method in methods) + r" \\",
        r"\hline",
    ]

    for rowIdx, bwValue in enumerate(bwValuesKhz):
        row = [f"{bwValue:g}"]
        row.extend(f"{float(results[method][rowIdx]):.4f}" for method in methods)
        lines.append(" & ".join(row) + r" \\")

    lines.extend([r"\hline", r"\end{tabular}"])

    os.makedirs(os.path.dirname(outPath), exist_ok=True)
    with open(outPath, "w", encoding="utf-8") as tableFile:
        tableFile.write("\n".join(lines) + "\n")

    return "\n".join(lines)


ssimTable = writeSsimLatexTable(bwKhz, ssimResults, "paper2026/BallSSIMTable.tex")
print(f"LaTeX SSIM table written to paper2026/BallSSIMTable.tex\n{ssimTable}")

fig = plt.figure(figsize=(6, 4))

plt.plot(bwKhz, ssimResults["Theoretical"], marker="o", linestyle="--", linewidth=1.5, label="Theoretical")
plt.plot(bwKhz, ssimResults["Measured"], marker="s", linestyle="--", linewidth=1.5, label="Measured")
plt.plot(bwKhz, ssimResults["GIRF"], marker="^", linestyle="-.", linewidth=1.5, label="GIRF")
plt.plot(bwKhz, ssimResults["Estimated"], marker="d", linestyle=":", linewidth=1.5, label="Estimated")
plt.plot(bwKhz, ssimResults["Estimated B0 Model"], marker="v", linestyle="-", linewidth=1.5, label="Estimated B0 Model")

plt.xlabel("Bandwidth (kHz)", fontsize=10)
plt.ylabel("SSIM", fontsize=10)

plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(fontsize=9)

plt.tight_layout()

plt.savefig("paper2026/BallSSIM.png", dpi=600, bbox_inches="tight")
plt.savefig("paper2026/BallSSIM.pdf", bbox_inches="tight")

plt.show()
pass
