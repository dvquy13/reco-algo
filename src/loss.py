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


def bpr_loss(pos_scores, neg_scores, target):
    """
    Computes the Bayesian Personalized Ranking (BPR) loss.

    Args:
        pos_scores: Predicted scores for positive items.
        neg_scores: Predicted scores for negative items.
        target: Tensor containing target labels (+1 or -1). For BPR, this can be ignored
                or set to +1, as BPR inherently assumes positive items should have higher scores.
        l2_reg: L2 regularization coefficient.

    Returns:
        loss: The computed BPR loss (scalar tensor).
    """
    # Compute the difference between positive and negative scores
    diff = pos_scores - neg_scores  # Should be positive if the model is performing well

    # Apply the sigmoid function and compute the logarithm
    loss = -torch.mean(
        torch.log(torch.sigmoid(diff) + 1e-10)
    )  # Adding epsilon for numerical stability

    return loss
