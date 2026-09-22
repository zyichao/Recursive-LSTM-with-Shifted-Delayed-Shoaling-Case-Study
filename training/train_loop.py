# training/train_loop.py

import torch

from training.losses import PlainMSELoss


def train_baseline_lstm(
    model,
    train_loader,
    val_loader,
    device,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    n_epochs: int = 300,
    early_stopping_patience: int = 300,
    loss_fn=None,
    verbose: bool = True,
):
    """Standard training loop with early stopping on validation loss;
    best-weights are restored at the end, so training past the optimum
    doesn't leave the model worse off.

    `train_loader`/`val_loader` batches are (X, y, zero_crossing_flags)
    triples (see preprocessing/rolling_windows.py). `loss_fn` is called as
    `loss_fn(pred, y, zero_crossing_flags)`; defaults to plain unweighted
    MSE (see training/losses.py for the paper's weighted alternative).

    Returns
    -------
    model : the trained model (best weights restored)
    history : dict with "train_loss" and "val_loss" lists
    best_val_loss : float
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = loss_fn if loss_fn is not None else PlainMSELoss()

    train_loss_history, val_loss_history = [], []
    best_val_loss, best_state, epochs_without_improvement = float("inf"), None, 0

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for xb, yb, zcb in train_loader:
            xb, yb, zcb = xb.to(device), yb.to(device), zcb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb, zcb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for xb, yb, zcb in val_loader:
                xb, yb, zcb = xb.to(device), yb.to(device), zcb.to(device)
                running_val_loss += loss_fn(model(xb), yb, zcb).item() * len(xb)
        val_loss = running_val_loss / len(val_loader.dataset)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        if verbose:
            print(f"epoch {epoch+1:3d}/{n_epochs}: train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1} (no val improvement for {early_stopping_patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"Restored best weights (val_loss={best_val_loss:.6f}).")

    history = {"train_loss": train_loss_history, "val_loss": val_loss_history}
    return model, history, best_val_loss
