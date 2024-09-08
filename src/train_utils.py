import numpy as np
import torch
import torch.optim as optim
from loguru import logger


class MetricLogCallback:
    def __init__(self):
        self.payloads = []

    def process_payload(self, payload: dict):
        self.payloads.append(payload)


def mse_loss(predictions, ratings, model, l2_reg=1e-5, device="cpu"):
    predictions = predictions.to(device)
    ratings = ratings.to(device)

    mse_loss = torch.mean((predictions - ratings) ** 2)

    if l2_reg:
        l2_loss = 0
        for param in model.parameters():
            l2_loss += torch.norm(param) ** 2
        total_loss = mse_loss + l2_reg * l2_loss
    else:
        total_loss = mse_loss
    return total_loss


def log_gradients(model):
    total_norm = 0
    param_count = 0
    gradient_metrics = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm**2
            param_count += 1
            gradient_metrics[f"grad_norm_{name}"] = param_norm

    total_norm = total_norm**0.5
    gradient_metrics["total_grad_norm"] = total_norm
    return gradient_metrics


def train(
    model,
    dataloader,
    val_dataloader=None,
    early_stopping=True,
    patience=5,
    epochs=10,
    lr=0.001,
    delta_perc=0.01,
    print_steps=100,
    device="cpu",
    gradient_clipping=True,
    progress_bar_type="tqdm",
    callbacks=[],
):

    device = torch.device(device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1000, gamma=0.9, verbose=True
    )
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
        num_batches = len(dataloader)
        batch_iterator = tqdm(
            enumerate(dataloader),
            desc=f"Training Epoch {epoch+1}",
            position=1,
            leave=False,
            total=num_batches,
        )

        for batch_idx, (users, items, ratings) in batch_iterator:
            step += 1

            users = users.to(device)
            items = items.to(device)
            ratings = ratings.to(device)

            optimizer.zero_grad()

            predictions = model.predict(users, items)
            loss = mse_loss(
                predictions, ratings, l2_reg=None, model=model, device=device
            )

            loss.backward()

            if gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if step % print_steps == 0:
                # Collect metric values
                metric_log_payload = {"step": step, "dataset": "train"}
                metric_log_payload["global_loss"] = total_loss / (batch_idx + 1)
                metric_log_payload["learning_rate"] = optimizer.param_groups[0]["lr"]
                gradient_metrics = log_gradients(model)
                metric_log_payload.update(gradient_metrics)

                # Log metrics
                logger.info(
                    f"Step {step}, Global Loss: {metric_log_payload['global_loss']:.4f}"
                )
                logger.info(
                    f"Step {step}, Learning Rate: {metric_log_payload['learning_rate']:.6f}"
                )
                logger.info(f"Step {step}, Gradient Norms: {gradient_metrics}")

                for callback in callbacks:
                    callback(metric_log_payload)

            batch_iterator.set_postfix(Global_Loss=total_loss / (batch_idx + 1))

        avg_train_loss = total_loss / num_batches
        epoch_metric_log_payload = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
        }
        logger.info(f"Epoch {epoch + 1}, Loss: {avg_train_loss:.4f}")

        if val_dataloader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for users, items, ratings in val_dataloader:
                    users = users.to(device)
                    items = items.to(device)
                    ratings = ratings.to(device)

                    predictions = model.predict(users, items)
                    val_loss += mse_loss(
                        predictions, ratings, model=model, device=device
                    ).item()

            val_loss /= len(val_dataloader)
            epoch_metric_log_payload["val_loss"] = val_loss
            logger.info(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")

            if early_stopping:
                if val_loss < best_val_loss * (1 - delta_perc):
                    best_val_loss = val_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        for callback in callbacks:
            callback(epoch_metric_log_payload)
