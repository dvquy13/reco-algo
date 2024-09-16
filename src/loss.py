import torch


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
