import numpy as np
import torch


def graph_laplacian_regularization(adjacency_matrix, embedding_matrix):
    """
    Calculates the Graph Laplacian regularization term.

    Parameters:
    adjacency_matrix (numpy.ndarray): The adjacency matrix of the graph (n x n matrix).
    signal_vector (numpy.ndarray): The signal vector (n-dimensional vector).

    Returns:
    float: The Graph Laplacian regularization term.
    """
    # Calculate the degree matrix
    degree_matrix = torch.diag(torch.sum(adjacency_matrix, axis=1))

    # Calculate the Laplacian matrix
    laplacian_matrix = degree_matrix - adjacency_matrix

    # Calculate the Graph Laplacian regularization term
    regularization_term = (
        torch.trace(
            torch.mm(embedding_matrix.T, torch.mm(laplacian_matrix, embedding_matrix))
        )
        / 2
    )

    return regularization_term


def graph_sparsity_regularization(
    adjacency_matrix, alpha=0.1, beta=0.1, device="cuda:0"
):
    n = adjacency_matrix.shape[0]
    one = torch.ones(n).reshape(1, n).to(device)
    log_term = -1 * alpha * torch.mm(one, torch.log(torch.mm(adjacency_matrix, one.T)))
    norm_term = beta * torch.linalg.norm(adjacency_matrix) ** 2 / 2

    return log_term + norm_term


def graph_reconstruction_loss(adjacency_matrix_new, adjacency_matrix_old):
    n = adjacency_matrix_new.shape[0]
    temp = torch.multiply(
        adjacency_matrix_old, torch.log(adjacency_matrix_new)
    ) + torch.multiply(
        (1 - adjacency_matrix_old), (1 - torch.log(1 - adjacency_matrix_new))
    )
    loss = torch.linalg.norm(temp) ** 2 / (1 / 2 * (n**2))
    return loss
