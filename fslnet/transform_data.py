import torch


def minimax_norm(dataset):
    """
    Apply Min-Max normalization to the dataset.
    """
    # Identify constant features
    constant_features = dataset.std(dim=0) == 0

    # Clone the dataset to avoid modifying the original data
    normalized_dataset = dataset.clone()

    # Normalize non-constant features
    non_constant_mask = ~constant_features
    non_constant_data = dataset[:, non_constant_mask]
    if non_constant_data.shape[1] > 0:
        feature_min = non_constant_data.min(dim=0).values
        feature_max = non_constant_data.max(dim=0).values
        normalized_dataset[:, non_constant_mask] = (non_constant_data - feature_min) / (feature_max - feature_min)
        
    # Set constant features to 0
    normalized_dataset[:, constant_features] = 0

    return normalized_dataset


def standard_norm(dataset):
    """
    Apply Standard normalization to the dataset.
    """
    # Compute mean and standard deviation
    mean = dataset.mean(dim=0)
    std = dataset.std(dim=0)

    # Identify constant features
    constant_features = std == 0

    # Clone the dataset to avoid modifying the original data
    normalized_dataset = dataset.clone()

    # Standardize non-constant features
    non_constant_mask = ~constant_features
    non_constant_data = dataset[:, non_constant_mask]

    if non_constant_data.shape[1] > 0:
        normalized_dataset[:, non_constant_mask] = (non_constant_data - mean[non_constant_mask]) / std[non_constant_mask]
        
    # Set constant features to 0
    normalized_dataset[:, constant_features] = 0

    return normalized_dataset


def normalize(reference, query, normalization):
    """
    Normalize ('minmax' or 'standard') the combined reference and query datasets.
    """
    dataset = torch.cat([reference, query], dim=0)
    if normalization == 'minmax':
        dataset = minimax_norm(dataset)
    elif normalization == 'standard':
        dataset = standard_norm(dataset)
    else:
        raise ValueError("Normalization must be 'minmax' or 'standard'.")
    
    reference, query = torch.split(dataset, [reference.shape[0], query.shape[0]], dim=0)
    
    return reference, query
