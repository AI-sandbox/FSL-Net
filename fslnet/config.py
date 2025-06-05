# NeuralEmbeddingNet hyperparameters
neural_cfg = {
    'process_jointly': True,
    'hidden_dim': 64,
    'kernel_size': 5,
    'dilation': 1,
    'num_res_blocks': 5,
    'use_attention': False,  # or 'EVA' or 'MHA'
    'num_heads': 4,
    'adaptive_proj': 'default',
    'num_landmarks': 8,
    'eva_window_factor': 4,
    'window_size': 7,
    'overlap_window': False,
    'fp32': False,
    'qkv_bias': True,
    'attn_drop': 0.0,
    'proj_drop': 0.0,
    'activation': 'Tanh',
}

# StatisticalMeasures hyperparameters
stat_cfg = {
    'mean': True,
    'mean_moments': True,
    'median': True,
    'std': True,
    'mad': True,
    'kurtosis': False,
    'skewness': False,
    'index_dispersion_1': False,
    'index_dispersion_2': False,
    'trimmed_mean_deviation': False,
    'histogram': True,
    'empirical_cdf': True,
    'kernel_density_estimation': False,
    'eps': 1e-5,
    'mean_orders': [2, 3],
    'trim_percentage': 0.1,
    'n_bins_df': [100],
}

# MomentExtractionNet hyperparameters
moment_cfg = {
    'hidden_dim': 64,
    'kernel_size': 75,
    'dilation': 1,
    'activation': 'ReLU',
}

# PredictionNet hyperparameters
pred_cfg = {
    'hidden_dim': 64,
    'kernel_size': 5,
    'dilation': 1,
    'num_res_blocks': 7,
    'use_attention': False,
    'num_heads': 4,
    'adaptive_proj': 'default',
    'num_landmarks': 8,
    'eva_window_factor': 4,
    'window_size': 7,
    'overlap_window': False,
    'fp32': False,
    'qkv_bias': True,
    'attn_drop': 0.0,
    'proj_drop': 0.0,
    'activation': 'Tanh',
}

# Global settings
fslnet_cfg = {
    'z_embedding': True,
    'eps': 1e-5,
    'combination': 'normalize_square',
    'normalization': 'standard',
    'n_features': 784,  # used for dummy tensor sizing
}
