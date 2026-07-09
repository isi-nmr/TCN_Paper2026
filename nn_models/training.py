import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, random_split

from nn_models.components import compute_loss_terms, cumulative_trapezoid, customMSELoss
from utils.BrukerMRI import *


def train(
    dataset,
    model,
    optimizer,
    outPth,
    nEpoch=500,
    *,
    trainTraj=False,
    plotting=True,
    gradAxis="X",
    batchSize=256,
    doB0=False,
    lossAlpha=1.0,
    weightLossByAmplitude=False,
    outRes=2e-6,
    lossIntegrationMethod="trapz",
    checkpointMetadata=None,
    earlyStopping=None,
    trial: optuna.trial.Trial = None,
):
    loss_fn = customMSELoss

    training_loss = []
    val_loss = []

    # Define split sizes
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)

    best_train_cost = torch.inf
    earlyStopping = earlyStopping or {}
    earlyStoppingEnabled = earlyStopping.get("enabled", False)
    earlyStoppingPatience = int(earlyStopping.get("patience", 50))
    earlyStoppingMinDelta = float(earlyStopping.get("minDelta", 0.0))
    earlyStoppingMinEpoch = int(earlyStopping.get("minEpoch", 0))
    earlyStoppingBestCost = torch.inf
    epochsWithoutImprovement = 0

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=10)

    cosine = CosineAnnealingLR(optimizer, T_max=nEpoch - 10)

    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[10])

    for epoch in range(nEpoch):
        # perform network training
        model.train()
        train_cost = 0
        train_mse_cost = 0
        train_traj_cost = 0
        n_batches = 0
        grad_normMax = 0
        for X_batch, y_batch, y_Traj, mask_batch in train_loader:
            # with autocast(device_type='cuda', dtype=torch.float32):
            optimizer.zero_grad()
            if model.model_name in {"TCNFull", "TCNFullSkip"}:
                y_pred = model(X_batch)
            else:
                # Recurrent nets return hidden as well
                y_pred, _ = model(X_batch)

            loss = loss_fn(
                X_batch,
                y_pred,
                y_batch,
                y_Traj,
                mask_batch,
                alpha=lossAlpha,
                weight=weightLossByAmplitude,
                dt=outRes,
                integral_method=lossIntegrationMethod,
            )
            mse_term, integral_term = compute_loss_terms(
                X_batch,
                y_pred,
                y_batch,
                y_Traj,
                mask_batch,
                weight=weightLossByAmplitude,
                dt=outRes,
                integral_method=lossIntegrationMethod,
            )

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,  # typical values: 0.5, 1.0, 5.0
            )
            if grad_norm > grad_normMax:
                grad_normMax = grad_norm.detach().item()
            optimizer.step()
            train_cost += loss.detach().item()
            train_mse_cost += mse_term.detach().item()
            train_traj_cost += integral_term.detach().item()
            n_batches += 1
        print(np.max(grad_normMax))
        model.eval()
        val_cost = 0
        val_mse_cost = 0
        val_traj_cost = 0
        n_val_batches = 0
        bestCurve = []
        worstCurve = []
        bestCurveIn = []
        worstCurveIn = []
        with torch.no_grad():  # no gradients needed for validation
            bestLoss = np.inf
            worstLoss = 0

            plotEpoch = epoch % 10 == 0

            for X_batch, y_batch, y_Traj, mask_batch in val_loader:
                if model.model_name in {"TCNFull", "TCNFullSkip"}:
                    y_pred = model(X_batch)
                else:
                    y_pred, _ = model(X_batch)

                loss = loss_fn(
                    X_batch,
                    y_pred,
                    y_batch,
                    y_Traj,
                    mask_batch,
                    alpha=lossAlpha,
                    weight=weightLossByAmplitude,
                    dt=outRes,
                    integral_method=lossIntegrationMethod,
                )
                mse_term, integral_term = compute_loss_terms(
                    X_batch,
                    y_pred,
                    y_batch,
                    y_Traj,
                    mask_batch,
                    weight=weightLossByAmplitude,
                    dt=outRes,
                    integral_method=lossIntegrationMethod,
                )

                if plotEpoch and plotting:
                    pred = cumulative_trapezoid(y_pred, 2e-6) if trainTraj else y_pred

                    err = (pred - y_batch) ** 2
                    weights = mask_batch * torch.abs(X_batch[:, [1], :])

                    lossItems = (err * weights).sum(dim=(-1, -2)) / weights.sum(dim=(-1, -2)).clamp_min(1e-12)

                    min_val, min_idx = torch.min(lossItems, dim=0)
                    max_val, max_idx = torch.max(lossItems, dim=0)

                    if min_val < bestLoss:
                        bestCurve = y_pred[min_idx, 0, :].detach().cpu().numpy()
                        bestCurveIn = y_batch[min_idx, 0, :].detach().cpu().numpy()
                        bestLoss = min_val

                    if max_val > worstLoss:
                        worstCurve = y_pred[max_idx, 0, :].detach().cpu().numpy()
                        worstCurveIn = y_batch[max_idx, 0, :].detach().cpu().numpy()
                        worstLoss = max_val

                val_cost += loss.item()  # accumulate scalar value
                val_mse_cost += mse_term.item()
                val_traj_cost += integral_term.item()
                n_val_batches += 1

        val_cost /= n_val_batches  # average validation loss
        val_mse_cost /= n_val_batches
        val_traj_cost /= n_val_batches

        train_cost /= n_batches
        train_mse_cost /= n_batches
        train_traj_cost /= n_batches
        training_loss.append(train_cost)

        val_loss.append(val_cost)

        if val_cost < earlyStoppingBestCost - earlyStoppingMinDelta:
            earlyStoppingBestCost = val_cost
            epochsWithoutImprovement = 0
        else:
            epochsWithoutImprovement += 1

        if val_cost < best_train_cost:
            best_train_cost = val_cost

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metadata": checkpointMetadata,
                    "loss": loss.item(),
                    "val_cost": val_cost,
                    "train_cost": train_cost,
                },
                outPth,
            )
            if hasattr(model, "log_scale"):
                print(
                    f"✅ Saved new best model with val_cost {val_cost:.2E}, train_cost {train_cost:.2E}, scale {torch.exp(model.log_scale).detach().item():.2E}"
                )
            else:
                print(f"✅ Saved new best model with val_cost {val_cost:.2E}, train_cost {train_cost:.2E}")

        # step the scheduler once per epoch
        scheduler.step()
        if plotEpoch and plotting:
            outName = gradAxis if not doB0 else gradAxis + "B0"
            plotProgress(
                epoch, train_cost, val_cost, training_loss, val_loss, optimizer, bestCurveIn, bestCurve, worstCurveIn, worstCurve, outName, trainTraj
            )

        if lossAlpha < 1:
            print(
                "Loss terms "
                f"epoch {epoch}: "
                f"train[mse={train_mse_cost:.2E}, traj={train_traj_cost:.2E}] "
                f"val[mse={val_mse_cost:.2E}, traj={val_traj_cost:.2E}]"
            )

        # Optuna pruning
        if trial is not None:
            trial.report(val_cost, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if earlyStoppingEnabled and epoch + 1 >= earlyStoppingMinEpoch and epochsWithoutImprovement >= earlyStoppingPatience:
            print(
                "Early stopping: "
                f"validation loss did not improve by {earlyStoppingMinDelta:.2E} "
                f"for {earlyStoppingPatience} epochs. Best val_cost {best_train_cost:.2E}."
            )
            break

    return val_loss


def plotProgress(
    epoch, train_cost, val_cost, training_loss, val_loss, optimizer, bestCurveIn, bestCurve, worstCurveIn, worstCurve, outName, trainTraj
):
    print(f"Epoch {epoch}: train cost {train_cost:.2E}, validation cost{val_cost:.2E}")
    print(f"Epoch {epoch + 1}, lr={optimizer.param_groups[0]['lr']:.2E}")

    fig, ax = plt.subplots(5, 1, figsize=(8, 6))  # make sure it's 3 rows, 1 column

    # Plot 1 — Training progress
    ax[0].set_title("Training Progress")
    ax[0].set_yscale("log")
    ax[0].plot(training_loss, label="train")
    ax[0].plot(val_loss, label="val")
    ax[0].set_ylabel("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()
    if len(val_loss) > 15:
        ax[0].set_ylim(np.minimum(1e-5, np.min(val_loss)), np.maximum(1e-5, np.max(val_loss[10:])))
    else:
        ax[0].set_ylim(np.minimum(1e-5, np.min(val_loss)), np.maximum(1e-5, np.max(val_loss)))

    # Plot 2 — Best curve
    ax[1].set_title("Best")
    ax[1].plot(bestCurveIn, label="Input")

    if trainTraj:
        ax[1].plot((torch.from_numpy(bestCurve)), label="Output")
    else:
        ax[1].plot(bestCurve, label="Output")

    if trainTraj:
        ax[2].plot((torch.from_numpy(bestCurve)) - bestCurveIn, label="Output")
    else:
        ax[2].plot(bestCurve - bestCurveIn, label="Output")

    ax[1].legend()

    # Plot 3 — Worst curve
    ax[3].set_title("Worst")
    ax[3].plot(worstCurveIn, label="Input")
    if trainTraj:
        ax[3].plot((torch.from_numpy(worstCurve)), label="Output")
    else:
        ax[3].plot(worstCurve, label="Output")

    ax[3].legend()

    if trainTraj:
        ax[4].plot((torch.from_numpy(worstCurve)) - worstCurveIn, label="Output")
    else:
        ax[4].plot(worstCurve - worstCurveIn, label="Output")

    non_zero_indices = np.flatnonzero((bestCurveIn) != 0)
    lastInd = non_zero_indices[-1] - 1 if len(non_zero_indices) > 0 else bestCurveIn.size

    firstInd = np.maximum(0, non_zero_indices[0] - 10 if len(non_zero_indices) > 0 else 0)
    ax[1].set_xlim(firstInd, lastInd)
    ax[2].set_xlim(firstInd, lastInd)

    non_zero_indices = np.flatnonzero((worstCurveIn) != 0)
    lastInd = non_zero_indices[-1] - 1 if len(non_zero_indices) > 0 else worstCurveIn.size
    firstInd = np.maximum(0, non_zero_indices[0] - 10 if len(non_zero_indices) > 0 else 0)

    ax[3].set_xlim(firstInd, lastInd)
    ax[4].set_xlim(firstInd, lastInd)

    # Layout and save
    fig.tight_layout()
    plt.savefig(f"train{outName}.png")
    plt.close(fig)
