import numpy as np
import torch
import torch.optim as optim
from loguru import logger


def mse_loss(predictions, ratings, model, l2_reg=1e-5, device="cpu"):
    """
    Compute the MSE loss with L2 regularization (weight decay).

    Args:
        predictions (tensor): Predicted ratings from the model.
        ratings (tensor): True ratings given by users to items.
        model (torch.nn.Module): The model to retrieve the parameters for L2 regularization.
        l2_reg (float): The L2 regularization coefficient (weight decay).
        device (torch.device): The device to perform computations on.

    Returns:
        torch.Tensor: MSE loss with L2 regularization.
    """
    predictions = predictions.to(device)
    ratings = ratings.to(device)

    # Compute standard MSE loss
    mse_loss = torch.mean((predictions - ratings) ** 2)

    # L2 regularization: sum of the squared norms of all parameters
    if l2_reg:
        l2_loss = 0
        for param in model.parameters():
            l2_loss += torch.norm(param) ** 2

        # Combine MSE loss with L2 regularization
        total_loss = mse_loss + l2_reg * l2_loss
    else:
        total_loss = mse_loss
    return total_loss


def log_gradients(model, step):
    """
    Log the gradient norms for each parameter of the model to check for vanishing or exploding gradients.

    Args:
        model: The model being trained.
        step (int): The current training step.
    """
    total_norm = 0
    param_count = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()  # L2 norm of the gradient
            total_norm += param_norm**2
            param_count += 1
            logger.info(f"Step {step}, Gradient Norm for {name}: {param_norm:.6f}")

    total_norm = total_norm**0.5
    logger.info(f"Step {step}, Total Gradient Norm: {total_norm:.6f}")


def train(
    model,
    dataloader,
    val_dataloader=None,
    early_stopping=True,
    patience=5,
    epochs=10,
    lr=0.001,
    delta_perc=0.01,  # Use percentage decrease instead of absolute delta
    print_steps=100,  # Print global loss at every print_steps
    device="cpu",  # Added device argument
    gradient_clipping=True,
    progress_bar_type="tqdm",
):
    """
    Train the model using MSE loss with early stopping based on percentage loss decrease.
    Args:
        model: The model to be trained.
        dataloader: Dataloader for training data.
        val_dataloader: Dataloader for validation data (optional).
        early_stopping (bool): Whether to use early stopping.
        patience (int): Patience for early stopping.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        delta_perc (float): Minimum percentage decrease in validation loss to reset patience.
        device (str or torch.device): The device to perform computations on ('cpu' or 'cuda').
    """
    device = torch.device(device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1000, gamma=0.9, verbose=True
    )  # Example: decreases LR every 1000 steps by a factor of 0.1
    best_val_loss = np.inf
    no_improvement_count = 0
    step = 0

    if progress_bar_type == "tqdm_notebook":
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    for epoch in tqdm(range(epochs), desc="Epochs", position=0):
        model.train()
        total_loss = 0

        # Total number of batches for progress bar
        num_batches = len(dataloader)

        # Inner loop for batches with tqdm
        batch_iterator = tqdm(
            enumerate(dataloader),
            desc=f"Training Epoch {epoch+1}",
            position=1,
            leave=False,
            total=num_batches,  # Total number of batches for progress tracking
        )

        for batch_idx, (users, items, ratings) in batch_iterator:
            step += 1

            # Move inputs and targets to the specified device
            users = users.to(device)
            items = items.to(device)
            ratings = ratings.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            predictions = model.predict(users, items)

            # Forward pass: Compute MSE loss
            loss = mse_loss(
                predictions, ratings, l2_reg=None, model=model, device=device
            )

            # Backward pass
            loss.backward()

            if gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()

            scheduler.step()

            total_loss += loss.item()

            # Print global loss every `print_steps`
            if step % print_steps == 0:
                log_gradients(model, step)

                # Retrieve and log the learning rate from the optimizer
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(f"Step {step}, Learning Rate: {current_lr:.6f}")

                logger.info(
                    f"Step {step}, Global Loss: {total_loss / (batch_idx + 1):.4f}"
                )

            # Update tqdm progress bar with global loss
            batch_iterator.set_postfix(Global_Loss=total_loss / (batch_idx + 1))

        avg_train_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch + 1}, Loss: {avg_train_loss:.4f}")

        # Validation phase
        if val_dataloader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for users, items, ratings in val_dataloader:
                    # Move validation data to the specified device
                    users = users.to(device)
                    items = items.to(device)
                    ratings = ratings.to(device)

                    predictions = model.predict(users, items)
                    val_loss += mse_loss(
                        predictions, ratings, model=model, device=device
                    ).item()

            val_loss /= len(val_dataloader)
            logger.info(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")

            # Early stopping check based on percentage decrease
            if early_stopping:
                if val_loss < best_val_loss * (
                    1 - delta_perc
                ):  # Check if the loss decreased by at least delta_perc
                    best_val_loss = val_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
