from __future__ import annotations
from types import SimpleNamespace
from importlib.resources import files
from typing import Union, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import copy
import math

from fslnet.transform_data import normalize
from fslnet.config import neural_cfg, stat_cfg, moment_cfg, pred_cfg, fslnet_cfg


################################################################################
# Residual 1D Convolutional Block with Optional Attention
################################################################################
class ResidualConv1D(nn.Module):
    """
    Residual 1D convolutional block for feature-wise processing, with optional attention.
    
    Implements a residual block with two convolutional layers and skip connections. Optionally, applies EVA or 
    multi-head self-attention between convolutions in the neural embedding and prediction subnets of FSL-Net.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: str,
        dilation: int = 1,
        padding: int = 0,
        use_attention: Union[str, bool] = False,
        num_heads: int = 4,
        adaptive_proj: str = 'default',
        num_landmarks: int = 8,
        eva_window_factor: int = 4,
        window_size: int = 7,
        overlap_window: bool = False,
        fp32: bool = False,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Convolution kernel size.
            activation (str): Activation function ('GELU', 'ReLU', 'Tanh', 'LeakyReLU', 'SiLU').
            dilation (int, default=1): Spacing between kernel elements. Default is 1.
            padding (int or str, default=0): Padding added to both sides. Default is 0.
            use_attention (str, default=False): Attention type: 'EVA', 'MHA', or False for plain conv. Default is False.
            num_heads (int, default=4): Number of attention heads for EVA/MHA. Default is 4.
            adaptive_proj (str, default='default'): EVA adaptive projection. Default is 'default'.
            num_landmarks (int, default=8): Number of EVA landmarks. Default is 8.
            eva_window_factor (int, default=4): EVA window factor. Default is 4.
            window_size (int, default=7): EVA window size. Default is 7.
            overlap_window (bool, default=False): EVA windows overlap. Default is False.
            fp32 (bool, default=False): Use fp32 in EVA. Default is False.
            qkv_bias (bool, default=True): EVA QKV bias. Default is True.
            attn_drop (float, default=0.0): EVA/MHA attention dropout. Default is 0.0.
            proj_drop (float, default=0.0): EVA projection dropout. Default is 0.0.
        """
        super(ResidualConv1D, self).__init__()
        self.use_attention = use_attention

        # First convolution + batch norm
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels, track_running_stats=False)

        # Attention or second convolution
        if use_attention == 'EVA':
            try:
                from attention_modules import EVA
            except ImportError:
                raise ImportError("EVA attention module not found. Please define or install EVA.")
            self.attn = EVA(
                dim=out_channels,
                num_heads=num_heads,
                adaptive_proj=adaptive_proj,
                num_landmarks=num_landmarks,
                use_t5_rpe=False,
                use_rpe=False,
                window_size=window_size,
                attn_2d=False,
                overlap_window=overlap_window,
                fp32=fp32,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop
            )
        elif use_attention == 'MHA':
            self.attn = nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, dropout=attn_drop)
        else:
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
            self.bn2 = nn.BatchNorm1d(out_channels, track_running_stats=False)
        
        # Activation function
        activations = {
            'GELU': nn.GELU(),
            'ReLU': nn.ReLU(),
            'Tanh': nn.Tanh(),
            'LeakyReLU': nn.LeakyReLU(),
            'SiLU': nn.SiLU()
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}. Choose from {list(activations.keys())}")
        self.activation = activations[activation]
        
        # 1D convolution if skip connection dims do not match
        self.match_dimensions = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels, track_running_stats=False)
            ) if in_channels != out_channels else nn.Identity()
        )
        self.eva_window_factor = eva_window_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input (batch, in_channels, seq_len).

        Returns:
            torch.Tensor: Output (batch, out_channels, seq_len).
        """
        # Project input to match output channels if needed (for skip connection)
        # If in_channels == out_channels, this is just the identity mapping
        residual = self.match_dimensions(x)

        # First convolution, batch norm, and activation
        out = self.activation(self.bn1(self.conv1(x)))

        # Apply attention or a second convolution and batch norm
        if self.use_attention == 'EVA':
            n_features = out.shape[2]
            # Compute window size
            window_size = min(max(8, n_features // self.eva_window_factor), 32)
            seq_length = (math.ceil(n_features/window_size))*window_size
            # Compute padding size and add padding
            padding_size = seq_length - n_features
            padding_tensor = torch.zeros(out.shape[0], out.shape[1], padding_size).to(x.device)
            out = torch.cat((out, padding_tensor), dim=2)
            key_padding_mask = torch.zeros(out.shape[0], seq_length).to(x.device)
            # Mask the padded region
            key_padding_mask[:, n_features:] = 1
            out = self.attn(out, key_padding_mask=key_padding_mask, window_size=window_size)
            out = out[:, :, :n_features]
        elif self.use_attention == 'MHA':
            out = out.permute(0, 2, 1)
            out, _ = self.attn(out, out, out, need_weights=False)
            out = out.permute(0, 2, 1)
        else:
            # Second conv + batch norm
            out = self.bn2(self.conv2(out))

        # Residual skip connection
        out += residual
        
        return out


################################################################################
# Neural Embedding Network
################################################################################
class NeuralEmbeddingNet(nn.Module):
    """
    A class for the Neural Embedding Network.

    The Neural Embedding Network maps each dataset (reference or query) into a high-level 
    feature representation via a stack of feature-wise 1D convolutional residual blocks, enabling the 
    extraction of complex, distributional properties across features, and feature-wise pooliing.
    """
    def __init__(
        self,
        process_jointly: bool = True,
        hidden_dim: int = 64,
        kernel_size: int = 5,
        dilation: int = 1,
        num_res_blocks: int = 5,
        use_attention: Union[str, bool] = False,
        num_heads: int = 4,
        adaptive_proj: str = 'default',
        num_landmarks: int = 8,
        eva_window_factor: int = 4,
        window_size: int = 7,
        overlap_window: bool = False,
        fp32: bool = False,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        activation: str = 'Tanh'
    ):
        """
        Args:
            process_jointly (bool, default=True): Jointly process reference and query inputs. Default is True.
            hidden_dim (int, default=64): Number of output channels in hidden layers. Default is 64.
            kernel_size (int, default=5): Size of the convolving kernel. Default is 5.
            dilation (int, default=1): Spacing between kernel elements. Default is 1.
            num_res_blocks (int, default=5): Number of residual blocks. Default is 5.
            use_attention (str, default=False): Attention type: 'EVA', 'MHA', or False for plain conv. Default is False.
            num_heads (int, default=4): Number of attention heads for EVA/MHA. Default is 4.
            adaptive_proj (str, default='default'): EVA adaptive projection. Default is 'default'.
            num_landmarks (int, default=8): Number of EVA landmarks. Default is 8.
            eva_window_factor (int, default=4): EVA window factor. Default is 4.
            window_size (int, default=7): EVA window size. Default is 7.
            overlap_window (bool, default=False): EVA windows overlap. Default is False.
            fp32 (bool, default=False): Use fp32 in EVA. Default is False.
            qkv_bias (bool, default=True): EVA QKV bias. Default is True.
            attn_drop (float, default=0.0): EVA/MHA attention dropout. Default is 0.0.
            proj_drop (float, default=0.0): EVA projection dropout. Default is 0.0.
            activation (str, default='Tanh'): Activation function ('GELU', 'ReLU', 'Tanh', 'LeakyReLU', 'SiLU'). Default is 'Tanh'.
        """
        super(NeuralEmbeddingNet, self).__init__()
        self.process_jointly = process_jointly
        
        # First convolution + batch norm
        self.initial_conv = nn.Conv1d(1, hidden_dim, kernel_size=kernel_size, dilation=dilation, padding='same')
        self.initial_bn = nn.BatchNorm1d(hidden_dim, track_running_stats=False)
        
        # Stack of residual 1D convolution/attention blocks
        self.res_blocks = nn.ModuleList([
            ResidualConv1D(
                hidden_dim, 
                hidden_dim, 
                activation=activation, 
                kernel_size=kernel_size, 
                dilation=dilation, 
                padding='same', 
                use_attention=use_attention, 
                num_heads=num_heads, 
                adaptive_proj=adaptive_proj, 
                num_landmarks=num_landmarks, 
                eva_window_factor=eva_window_factor, 
                window_size=window_size, 
                overlap_window=overlap_window, 
                fp32=fp32, 
                qkv_bias=qkv_bias, 
                attn_drop=attn_drop, 
                proj_drop=proj_drop
        )] + [ResidualConv1D(
            hidden_dim,
            hidden_dim, 
            activation=activation, 
            kernel_size=kernel_size, 
            dilation=dilation, 
            padding='same',
            use_attention=use_attention, 
            num_heads=num_heads, 
            adaptive_proj=adaptive_proj, 
            num_landmarks=num_landmarks, 
            eva_window_factor=eva_window_factor, 
            window_size=window_size, 
            overlap_window=overlap_window, 
            fp32=fp32, 
            qkv_bias=qkv_bias, 
            attn_drop=attn_drop, 
            proj_drop=proj_drop
        ) for _ in range(num_res_blocks - 1)])
    
    def forward(self, x: List[torch.Tensor]) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass: produces dataset-level feature embeddings for reference and/or query data.

        Args:
            x (list of torch.Tensor): 
                - If process_jointly=True: [reference, query], each (n_samples, 1, n_features), dtype=torch.float32.
                - If process_jointly=False: [data], shape (n_samples, 1, n_features).

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
                - If process_jointly is True: 
                    A tuple `(ref_neural_embedding, que_neural_embedding)`. Each tensor has shape `(1, hidden_dim, n_features)`.

                - If process_jointly is False:
                    Single embedding. Shape: `(1, hidden_dim, n_features)`.
        """
        if self.process_jointly:
            # For joint processing, stack reference and query datasets
            assert len(x) == 2, 'If `process_jointly` is True, the input must be a list of two tensors.'
            reference, query = x
            n_batches_ref = reference.shape[0]
            x = torch.cat((reference, query), dim=0)
        else:
            assert len(x) == 1, 'If `process_jointly` is False, the input must be a list of one tensor.'
            x = x[0]

        # First convolution + batch norm
        x = self.initial_bn(self.initial_conv(x))

        # Pass through all residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)

        if self.process_jointly:
            # For joint processing, separate the reference and query neural embeddings and apply mean sample-wise pooling
            ref_neural_embedding = torch.mean(x[:n_batches_ref], dim=0).unsqueeze(0)
            que_neural_embedding = torch.mean(x[n_batches_ref:], dim=0).unsqueeze(0)
            return ref_neural_embedding, que_neural_embedding
        else:
            # For separate processing, apply mean sample-wise pooling
            x = torch.mean(x, dim=0).unsqueeze(0)

            return x


################################################################################
# Statistical Measures
################################################################################
class StatisticalMeasures(nn.Module):
    """
    A class for computing Statistical Measures from input reference and query.
    """
    def __init__(
        self,
        mean: bool = True,
        mean_moments: bool = True,
        median: bool = True,
        std: bool = True,
        mad: bool = True,
        kurtosis: bool = False,
        skewness: bool = False,
        index_dispersion_1: bool = False,
        index_dispersion_2: bool = False,
        trimmed_mean_deviation: bool = False,
        histogram: bool = True,
        empirical_cdf: bool = True,
        kernel_density_estimation: bool = False,
        eps: float = 1e-5,
        mean_orders: List[int] = [2, 3],
        trim_percentage: float = 0.1,
        n_bins_df: List[int] = [100]
    ):
        """
        Args:
            mean (bool, default=True): If True, compute the mean. Default is True.
            mean_moments (bool, default=True): If True, compute higher-order mean moments. Default is True.
            median (bool, default=True): If True, compute the median. Default is True.
            std (bool, default=True): Compute the standard deviation. Default is True.
            mad (bool, default=True): Compute the mean absolute deviation. Default is True.
            kurtosis (bool, default=False): If True, compute the kurtosis. Default is False.
            skewness (bool, default=False): If True, compute the skewness. Default is False.
            index_dispersion_1 (bool, default=False): If True, compute the index of dispersion as var/mean. Default is False.
            index_dispersion_2 (bool, default=False): If True, compute the index of dispersion as var/(mean + eps). Default is False.
            trimmed_mean_deviation (bool, default=False): If True, compute the deviation from trimmed mean. Default is False.
            histogram (bool, default=True): If True, compute the histogram. Default is True.
            empirical_cdf (bool, default=True): If True, compute the empirical CDF. Default is True.
            kernel_density_estimation (bool, default=False): If True, compute the KDE (Gaussian). Default is False.
            eps (float, default=1e-5): Small epsilon to avoid division by zero. Default is 1e-5.
            mean_orders (list of int, default=[2, 3]): Orders of moments to compute. Default is [2, 3].
            trim_percentage (float, default=0.1): Fraction of data to trim for trimmed mean. Default is 0.1.
            n_bins_df (list of int, default=[100]): Binning options for histogram and CDF. Default is [100].
        """
        super(StatisticalMeasures, self).__init__()
        self.mean = mean
        self.mean_moments = mean_moments
        self.median = median
        self.std = std
        self.mad = mad
        self.kurtosis = kurtosis
        self.skewness = skewness
        self.index_dispersion_1 = index_dispersion_1
        self.index_dispersion_2 = index_dispersion_2
        self.trimmed_mean_deviation = trimmed_mean_deviation
        self.histogram = histogram
        self.empirical_cdf = empirical_cdf
        self.kernel_density_estimation = kernel_density_estimation
        self.eps = eps
        self.trim_percentage = trim_percentage
        self.mean_orders = mean_orders
        self.n_bins_df = n_bins_df
        
        self.mean_ = None
        self.mean_orders_ = None
        self.median_ = None
        self.std_ = None
        self.centered_ = None
        self.mad_ = None
        self.kurtosis_ = None
        self.skewness_ = None
        self.index_dispersion_1_ = None
        self.index_dispersion_2_ = None
        self.trimmed_mean_deviation_ = None
        self.empirical_cdf_ = None
        self.histogram_ = None
        self.kernel_density_estimation_ = None
        
    def copy(self) -> "StatisticalMeasures":
        """
        Returns a deep copy of the object, to allow independent use for reference/query.
        """
        return copy.deepcopy(self)    
    
    def compute_mean_orders(self, x: torch.Tensor) -> None:
        """
        Compute higher-order feature moments along sample axis.

        Args:
            x (torch.Tensor): Shape: (n_samples, 1, n_features).
        
        Sets:
            self.mean_orders_: Tensor of shape (1, len(mean_orders)*n_features)
        """
        mean_embeddings = []
        for order in self.mean_orders:
            mean_x = torch.mean(x**order, dim=0).unsqueeze(0)
            mean_embeddings.append(mean_x)

        # Concatenate moments along channel axis
        self.mean_orders_ = torch.cat(mean_embeddings, dim=1)
        
    def compute_mean(self, x: torch.Tensor) -> None:
        """
        Compute mean along sample axis.

        Args:
            x (torch.Tensor): Shape: (n_samples, 1, n_features)
        
        Sets:
            self.mean_: Tensor (1, 1, n_features)
        """
        self.mean_ = torch.mean(x, dim=0).unsqueeze(0)
        
    def compute_median(self, x: torch.Tensor) -> None:
        """
        Compute median along sample axis.

        Args:
            x (torch.Tensor): Shape: (n_samples, 1, n_features).
        
        Sets:
            self.median_: Tensor (1, 1, features)
        """
        self.median_ = x.median(dim=0).values.unsqueeze(0)
        
    def compute_std(self, x: torch.Tensor) -> None:
        """
        Compute standard deviation along sample axis.

        Args:
            x (torch.Tensor): Shape: (n_samples, 1, n_features).

        Sets:
            self.std_: Tensor (1, 1, n_features)
        """
        self.std_ = x.std(dim=0).unsqueeze(0)

    def compute_centered(self, x: torch.Tensor) -> None:
        """
        Center data by featurewise mean.

        Args:
            x (torch.Tensor): Shape: (n_samples, 1, n_features).

        Sets:
            self.centered_: Tensor (n_samples, 1, n_features)
        """
        if self.mean_ is None:
            self.compute_mean(x)
            
        self.centered_ = x - self.mean_

    def compute_mad(self, x: torch.Tensor) -> None:
        """
        Compute mean absolute deviation (MAD) along sample axis.

        Args:
            x (torch.Tensor): Shape: (n_samples, 1, n_features).

        Sets:
            self.mad_: Tensor (1, 1, n_features)
        """
        if self.centered_ is None:
            self.compute_centered(x)
            
        self.mad_ = torch.mean(torch.abs(self.centered_), dim=0).unsqueeze(0)

    def compute_kurtosis(self, x: torch.Tensor) -> None:
        """
        Compute kurtosis along sample axis.

        Args:
            x (torch.Tensor): Shape: (n_samples, 1, n_features).

        Sets:
            self.kurtosis_: Tensor (1, 1, n_features)
        """
        if self.centered_ is None:
            self.compute_centered(x)
        if self.std_ is None:
            self.compute_std(x)

        # Compute kurtosis using the formula
        standardized = self.centered_ / (self.std_ + self.eps)
        fourth_power = torch.pow(standardized, 4)
        fourth_moment = torch.mean(fourth_power, dim=0).unsqueeze(0)

        self.kurtosis_ = fourth_moment

    def compute_skewness(self, x: torch.Tensor) -> None:
        """
        Compute skewness along sample axis.

        Args:
            x (torch.Tensor): Shape: (n_samples, 1, n_features).

        Sets:
            self.skewness_: Tensor (1, 1, n_features)
        """
        if self.centered_ is None:
            self.compute_centered(x)
        if self.std_ is None:
            self.compute_std(x)
        
        # Compute skewness using the formula
        standardized = self.centered_ / (self.std_ + self.eps)
        third_power = torch.pow(standardized, 3)
        third_moment = torch.mean(third_power, dim=0).unsqueeze(0)

        self.skewness_ = third_moment

    def compute_index_dispersion_1(self, x: torch.Tensor) -> None:
        """
        Compute variance/mean along sample axis (index of dispersion).

        Args:
            x (torch.Tensor): Shape: (n_samples, 1, n_features).

        Sets:
            self.index_dispersion_1_: Tensor (1, 1, n_features)
        """
        if self.mean_ is None:
            self.compute_mean(x)
        if self.std_ is None:
            self.compute_std(x)
        
        mean = self.mean_
        var = self.std_**2
        
        # Identify features with a mean of zero
        zero_mean_features = mean == 0

        # Define index of dispersion as the variance initially
        index_dispersion = var.clone()

        # Obtain mask of features with mean different than zero
        non_constant_mask = ~zero_mean_features

        # Compute index of dispersion by dividing variance/mean
        index_dispersion[non_constant_mask] = var[non_constant_mask]/mean[non_constant_mask]

        self.index_dispersion_1_ = index_dispersion
        
    def compute_index_dispersion_2(self, x: torch.Tensor) -> None:
        """
        Compute variance/(mean+eps) along sample axis (index of dispersion).

        Args:
            x (torch.Tensor): Shape: (n_samples, 1, n_features).

        Sets:
            self.index_dispersion_2_: Tensor (1, 1, n_features)
        """
        if self.mean_ is None:
            self.compute_mean(x)
        if self.std_ is None:
            self.compute_std(x)
        
        self.index_dispersion_2_ = (self.std_**2)/(self.mean_+self.eps)
        
    def compute_trimmed_mean_deviation(self, x: torch.Tensor) -> None:
        """
        Compute mean absolute deviation from the trimmed mean, along sample axis.

        Args:
            x (torch.Tensor): Shape: (n_samples, 1, n_features).

        Sets:
            self.trimmed_mean_deviation_: Tensor (1, 1, n_features)
        """
        # Calculate the number of elements to trim from both ends
        trim_size = int(self.trim_percentage * x.size(0) / 2)

        # Sort the data along each feature dimension
        sorted_data, _ = torch.sort(x, dim=0)

        # Trim the data along each feature dimension
        trimmed_data = sorted_data[trim_size:-trim_size, :]

        # Calculate the mean of the trimmed data along each feature dimension
        trimmed_mean = torch.mean(trimmed_data, dim=0)

        # Calculate the mean absolute deviation from the trimmed mean along each feature dimension
        self.trimmed_mean_deviation_ = torch.mean(torch.abs(trimmed_data - trimmed_mean), dim=0).unsqueeze(0)
      
    def compute_empirical_cdf(self, x: torch.Tensor, n_bins_df: int) -> None:
        """
        Compute the empirical cumulative distribution function (CDF) for each feature along sample axis.

        Args:
            x (torch.Tensor): Shape: (n_samples, 1, n_features).
            n_bins_df (int): Number of bins at which to estimate the CDF per feature.

        Sets:
            self.empirical_cdf_: Tensor (1, n_bins_df, n_features)
        """
        hist_thresholds_ = torch.linspace(0, 1, n_bins_df, dtype=torch.float32).to(x.device)
        x = torch.squeeze(x)
        
        empirical_cdf = torch.mean((x.unsqueeze(-1) <= hist_thresholds_).to(torch.float), dim=0)
        self.empirical_cdf_ = empirical_cdf.permute(1, 0).unsqueeze(0)
      
    def compute_histogram(self, x: torch.Tensor, n_bins_df) -> None:
        """
        Compute histogram for each feature along sample axis.

        Args:
            x (torch.Tensor): Shape: (n_samples, 1, n_features).
            n_bins_df (int): Number of bins at which to estimate the histogram per feature.

        Sets:
            self.histogram_: Tensor (1, n_bins_df, n_features)
        """
        if self.empirical_cdf_ is None:
            self.compute_empirical_cdf(x, n_bins_df)
        histogram = torch.diff(self.empirical_cdf_, dim=1)

        # Add first bin from empirical CDF
        first_bin_value = self.empirical_cdf_[0, 0, 0].view(1, 1, 1).expand(histogram.shape[0], 1, histogram.shape[2])
        self.histogram_ = torch.cat((first_bin_value, histogram), dim=1)

    def compute_feature_histogram(self, feature_data: pd.Series) -> np.ndarray:
        """
        Compute the kernel density estimate (KDE) for a single feature.

        Args:
            feature_data (pd.Series): Data for one feature, shape (n_samples,).

        Returns:
            np.ndarray: Estimated probability density function values at evenly spaced sample points. Shape: (n_bins,).
        """
        from sklearn.neighbors import KernelDensity
        kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
        kde.fit(feature_data.values.reshape(-1, 1))
        sample_points = np.linspace(feature_data.min(), feature_data.max(), self.n_bins).reshape(-1, 1)
        log_density = kde.score_samples(sample_points)
        return np.exp(log_density)

    def compute_kernel_density(self, x: torch.Tensor, n_bins_df) -> None:
        """
        Compute featurewise kernel density estimation (KDE) for input data along sample axis.

        Args:
            x (torch.Tensor): Shape: (n_samples, 1, n_features).

        Sets:
            self.kernel_density_estimation_: Tensor (1, n_bins_df, n_features)
        """
        from sklearn.neighbors import KernelDensity
        device = x.device
        x = x.squeeze(1).to('cpu')
        kde = KernelDensity().fit(x)
        self.kernel_density_estimation_ = kde.score_samples(x)
        
        # Create a DataFrame from the input data
        df = pd.DataFrame(x)
        
        # Apply the compute_feature_histogram function to each feature
        self.n_bins = n_bins_df
        histograms = np.array(df.apply(self.compute_feature_histogram))
        
        # Convert the resulting DataFrame to a torch array
        histograms = torch.tensor(histograms, dtype=torch.float32)
        self.kernel_density_estimation_ = histograms.unsqueeze(0).to(device)

    def forward(self, x: torch.Tensor, x_minmax: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: produces dataset-level feature embeddings for reference or query data.
        
        Args:
            x (torch.Tensor): Shape: (n_samples, 1, n_features).
            x_minmax (torch.Tensor): (n_samples, 1, n_features), min-max normalized for histogram/CDF.

        Returns:
            torch.Tensor: (1, stat_dim, n_features) concatenated across all selected measures
        """
        embeddings = []
        
        if self.mean:
            self.compute_mean(x)
            embeddings.append(self.mean_)
        
        if self.mean_moments:
            self.compute_mean_orders(x)
            embeddings.append(self.mean_orders_)
        
        if self.median:
            self.compute_median(x)
            embeddings.append(self.median_)
        
        if self.std:
            self.compute_std(x)
            embeddings.append(self.std_)
        
        if self.mad:
            self.compute_mad(x)
            embeddings.append(self.mad_)
        
        if self.kurtosis:
            self.compute_kurtosis(x)
            embeddings.append(self.kurtosis_)
        
        if self.skewness:
            self.compute_skewness(x)
            embeddings.append(self.skewness_)
        
        if self.index_dispersion_1:
            self.compute_index_dispersion_1(x)
            embeddings.append(self.index_dispersion_1_)
        
        if self.index_dispersion_2:
            self.compute_index_dispersion_2(x)
            embeddings.append(self.index_dispersion_2_)
        
        if self.trimmed_mean_deviation:
            self.compute_trimmed_mean_deviation(x)
            embeddings.append(self.trimmed_mean_deviation_)
        
        # For histogram and CDFs, use min-max normalized input
        for n_bins_df in self.n_bins_df:
            if self.empirical_cdf:
                self.compute_empirical_cdf(x_minmax, n_bins_df)
                embeddings.append(self.empirical_cdf_)

            if self.histogram:
                self.compute_histogram(x_minmax, n_bins_df)
                embeddings.append(self.histogram_)
            
            self.empirical_cdf_ = None

            if self.kernel_density_estimation:
                self.compute_kernel_density(x, n_bins_df)
                embeddings.append(self.kernel_density_estimation_)
        
        # Concatenate all statistica measures functional maps
        concatenated_embeddings = torch.cat(embeddings, dim=1)
        
        return concatenated_embeddings


################################################################################
# Moment Extraction Network
################################################################################
class MomentExtractionNet(nn.Module):
    """
    A class for the Moment Extraction Network. 

    The Moment Extraction Network extracts correlations and higher-order moments across features from 
    each dataset (reference or query) via a feature-wise 1D convolutional with large kernel size 
    and sample-wise pooling.
    """
    def __init__(
        self,
        hidden_dim: int = 64,
        kernel_size: int = 75,
        dilation: int = 1,
        activation: str = 'ReLU'
    ):
        """
        Args:
            hidden_dim (int, default=64): Number of output channels in hidden layers. Default is 64.
            kernel_size (int, default=75): Size of the convolving kernel. Default is 75.
            dilation (int, default=1): Spacing between kernel elements. Default is 1.
            activation (str, default='ReLU'): Activation function ('Square', 'ReLU', 'Sigmoid', 'Tanh'). Default is 'ReLU'.
        """
        super(MomentExtractionNet, self).__init__()
        
        # Convolution that changes channels from 1 to `hidden_dim` + batch norm
        self.initial_conv = nn.Conv1d(1, hidden_dim, kernel_size=kernel_size, dilation=dilation, padding='same')
        self.bn = nn.BatchNorm1d(hidden_dim, track_running_stats=False)
        
        # Activation function
        self.activation = activation

        activations = ['Square', 'ReLU', 'Sigmoid', 'Tanh']
        if activation not in ['Square', 'ReLU', 'Sigmoid', 'Tanh']:
            raise ValueError(f"Unsupported activation: {activation}. Choose from {activations}")

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Forward pass through the Moment Extraction Network.
        """
        # Convolution + batch norm
        x = self.bn(self.initial_conv(x))

        # Apply activation function
        if self.activation == 'Square':
            x = torch.square(x)
        elif self.activation == 'ReLU':
            x = self.relu(x)
        elif self.activation == 'Sigmoid':
            x = self.sigmoid(x)
        elif self.activation == 'Tanh':
            x = self.tanh(x)
        else:
            raise ValueError(f"Invalid activation: {self.activation}")
        
        # Mean sample-wise pooling
        x = torch.mean(x, dim=0).unsqueeze(0)

        return x


################################################################################
# Prediction Network
################################################################################
class PredictionNet(nn.Module):
    """
    A class for the Prediction Network. 

    The Prediction Network takes the combined per-feature statistical functionals between the reference and the query 
    as input and predicts the per-feature probability of being shifted (corrupted), as well as logits.
    It consists of an initial batch normalization, a stack of 1D residual convolutional blocks, and a 
    final convolution to reduce the channel dimension.
    """
    def __init__(
        self, 
        in_channels: int = 335, 
        hidden_dim: int = 64, 
        kernel_size: int = 5, 
        dilation: int = 1, 
        num_res_blocks: int = 7,
        use_attention: Union[str, bool] = False, 
        num_heads: int = 4, 
        adaptive_proj: str = 'default', 
        num_landmarks: int = 8,
        eva_window_factor: int = 4, 
        window_size: int = 7, 
        overlap_window: bool = False, 
        fp32: bool = False, 
        qkv_bias: bool = True, 
        attn_drop: float = 0., 
        proj_drop: float = 0.,
        activation: str = 'Tanh'
    ):
        """
        Args:
            in_channels (int, default=335): Number of input channels. Default is 335.
            hidden_dim (int, default=64): Number of output channels in hidden layers. Default is 64.
            kernel_size (int, default=5): Size of the convolving kernel. Default is 5.
            dilation (int, default=1): Spacing between kernel elements. Default is 1.
            num_res_blocks (int, default=7): Number of residual blocks. Default is 7.
            use_attention (str, default=False): Attention type: 'EVA', 'MHA', or False for plain conv. Default is False.
            num_heads (int, default=4): Number of attention heads for EVA/MHA. Default is 4.
            adaptive_proj (str, default='default'): EVA adaptive projection. Default is 'default'.
            num_landmarks (int, default=8): Number of EVA landmarks. Default is 8.
            eva_window_factor (int, default=4): EVA window factor. Default is 4.
            window_size (int, default=7): EVA window size. Default is 7.
            overlap_window (bool, default=False): EVA windows overlap. Default is False.
            fp32 (bool, default=False): Use fp32 in EVA. Default is False.
            qkv_bias (bool, default=True): EVA QKV bias. Default is True.
            attn_drop (float, default=0.0): EVA/MHA attention dropout. Default is 0.0.
            proj_drop (float, default=0.0): EVA projection dropout. Default is 0.0.
            activation (str, default='GELU'): Activation function ('GELU', 'ReLU', 'Tanh', 'LeakyReLU', 'SiLU'). Default is 'Tanh'.
        """
        super(PredictionNet, self).__init__()
        # Initial batch norm
        self.initial_bn = nn.BatchNorm1d(in_channels, track_running_stats=False)
        # Stack of residual 1D convolution/attention blocks
        self.res_blocks = nn.ModuleList([
            ResidualConv1D(in_channels, hidden_dim, activation=activation, kernel_size=kernel_size, dilation=dilation, padding='same',
                           use_attention=use_attention, num_heads=num_heads, adaptive_proj=adaptive_proj, 
                            num_landmarks=num_landmarks, eva_window_factor=eva_window_factor, window_size=window_size, overlap_window=overlap_window, fp32=fp32, 
                            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop
                           )
        ] + [
            ResidualConv1D(hidden_dim, hidden_dim, activation=activation, kernel_size=kernel_size, dilation=dilation, padding='same',
                           use_attention=use_attention, num_heads=num_heads, adaptive_proj=adaptive_proj, 
                            num_landmarks=num_landmarks, eva_window_factor=eva_window_factor, window_size=window_size, overlap_window=overlap_window, fp32=fp32, 
                            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop
                           ) 
            for _ in range(num_res_blocks - 1)
        ])
        # Final convolutional layer to reduce the channels to 1 for output prediction
        self.final_conv = nn.Conv1d(hidden_dim, 1, kernel_size=kernel_size, dilation=dilation, padding='same')

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: predicts per-feature shift probabilities and logits.

         Args:
            x (torch.Tensor): The combined per-feature statistical functionals between the reference and the query. Shape: `(1, in_channels, n_features).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - probs (torch.Tensor): Predicted probabilities for each feature being shifted, shape (1, n_features).s
                - logits (torch.Tensor): Raw logits before sigmoid activation, shape (1, n_features).
        """
        # Initial batch norm
        x = self.initial_bn(x)
        
        # Pass through all residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Final convolution
        x_logits = self.final_conv(x)
        # Apply sigmoid to get probabilities
        x = torch.sigmoid(x_logits)
        # Reshape to (1, n_features)
        x = x.permute(0, 2, 1).squeeze(2)
        x_logits = x_logits.permute(0, 2, 1).squeeze(2)
        
        return x, x_logits


################################################################################
# FSLNet: Feature Shift Localization Network
################################################################################
class FSLNet(nn.Module):
    """
    Combines neural, statistical, and moment embeddings from the reference and the query datasets for
    prediction of feature shifts.
    """
    _model_cache: dict[tuple[str, str], FSLNet] = {}
    
    def __init__(
        self, 
        neural_embedding_net: NeuralEmbeddingNet,
        statistical_measures: StatisticalMeasures, 
        moment_extraction_net: MomentExtractionNet, 
        prediction_net: PredictionNet,
        z_embedding: bool = True,
        eps: float = 1e-5,
        combination: str = 'normalize_square',
        normalization: str = 'standard'
    ):
        """
        Args:
            neural_embedding_net (nn.Module): 
                Neural embedding subnetwork.
            statistical_measures (nn.Module): 
                Statistical measures module.
            moment_extraction_net (nn.Module): 
                Moment extraction subnetwork.
            prediction_net (nn.Module): 
                Prediction subnetwork module.
            z_embedding (bool, default=True): 
                If True, compute z-normalized mean/std. Default is True.
            eps (float, default=1e-5): 
                Small epsilon to avoid division by zero. Default is 1e-5.
            combination (str, default='normalize_square'): 
                Merging operation for combining the statistical functional maps between the reference and query ('diff', 
                'square', 'normalize', 'normalize_square', 'normalize_square_concat_log', 'concat'). 
                Default is 'normalize_square'.
            normalization (str, default='standard'): Feature normalization method. Default is 'standard'.
        """
        super(FSLNet, self).__init__()
        
        self.neural_embedding_net = neural_embedding_net
        self.statistical_measures = statistical_measures
        self.moment_extraction_net = moment_extraction_net
        self.prediction_net = prediction_net
        self.z_embedding = z_embedding
        self.eps = eps
        self.combination = combination
        self.normalization = normalization
        
        self.ref_neural_embedding_ = None
        self.que_neural_embedding_ = None
        
    def normalize_data(self, reference: torch.Tensor, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize reference and query datasets according to the selected normalization strategy.

        Args:
            reference (torch.Tensor): Reference data, shape (n_samples, n_features), dtype=torch.float32.
            query (torch.Tensor): Query data, shape (n_samples, n_features).

        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
                - reference (torch.Tensor): Normalized reference, shape (n_samples_ref, 1, n_features)
                - query (torch.Tensor): Normalized query, shape (n_samples_que, 1, n_features)
                - reference_minmax (torch.Tensor or None): Min-max normalized reference or None.
                - query_minmax (torch.Tensor or None): Min-max normalized query or None.
        """
        # If using histogram or empirical CDF in statistics, always perform min-max normalization for those
        if self.statistical_measures is not None and \
            (self.statistical_measures.histogram or self.statistical_measures.empirical_cdf):
            reference_minmax, query_minmax = normalize(reference, query, 'minmax')
            if self.normalization != 'minmax':
                reference, query = normalize(reference, query, self.normalization)
            else:
                reference, query = reference_minmax, query_minmax
            reference_minmax, query_minmax = reference_minmax.unsqueeze(1), query_minmax.unsqueeze(1)
        else:
            reference_minmax, query_minmax = None, None
            reference, query = normalize(reference, query, self.normalization)
        
        return reference.unsqueeze(1), query.unsqueeze(1), reference_minmax, query_minmax
    
    def _compute_z_embedding(
        self, 
        reference: torch.Tensor, 
        query: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute feature‐wise z‐embedding, i.e. (mean_ref – mean_query) / (std_ref + eps).

        Returns:
            torch.Tensor: Shape (1, n_features).
        """
        if self.statistical_measures is not None:
            if self.statistical_measures_ref.mean_ is None:
                self.statistical_measures_ref.compute_mean(reference)
            if self.statistical_measures_que.mean_ is None:
                self.statistical_measures_que.compute_mean(query)
            if self.statistical_measures_ref.std_ is None:
                self.statistical_measures_ref.compute_std(reference)
            
            ref_mean = self.statistical_measures_ref.mean_
            que_mean = self.statistical_measures_que.mean_
            ref_std  = self.statistical_measures_ref.std_
        else:
            ref_vals = reference.squeeze(1)  # (n_samples_ref, n_features)
            que_vals = query.squeeze(1)      # (n_samples_que, n_features)
            ref_mean = torch.mean(ref_vals, dim=0, keepdim=True)
            que_mean = torch.mean(que_vals, dim=0, keepdim=True)
            ref_std  = torch.std(ref_vals, dim=0, keepdim=True)
        
        z = (ref_mean - que_mean) / (ref_std + self.eps)
        return z

    def _merge_operation(
        self, 
        ref_embedding: torch.Tensor, 
        que_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Merge/ref‐que combination logic based on self.combination.
        
        Args:
            ref_embedding (torch.Tensor): shape (1, C, n_features)
            que_embedding (torch.Tensor): shape (1, C, n_features)
        
        Returns:
            combined_embedding (torch.Tensor), shape depends on combination:
              - 'diff': (1, C, n_features)
              - 'square': (1, C, n_features)
              - 'normalize': (1, C, n_features)
              - 'normalize_square': (1, C, n_features)
              - 'normalize_square_concat_log': (1, 2*C, n_features)
              - 'concat': (1, 2*C, n_features)
        """
        if self.combination == 'diff':
            return ref_embedding - que_embedding

        elif self.combination == 'square':
            return (ref_embedding - que_embedding) ** 2

        elif self.combination == 'normalize':
            norm_ref = torch.norm(ref_embedding, p=2, dim=2, keepdim=True)
            return (ref_embedding - que_embedding) / (norm_ref + self.eps)

        elif self.combination == 'normalize_square':
            norm_ref = torch.norm(ref_embedding, p=2, dim=2, keepdim=True)
            return (ref_embedding - que_embedding) ** 2 / (norm_ref + self.eps)

        elif self.combination == 'normalize_square_concat_log':
            norm_ref = torch.norm(ref_embedding, p=2, dim=2, keepdim=True)
            norm_square = (ref_embedding - que_embedding) ** 2 / (norm_ref + self.eps)
            log_combined = torch.log(norm_square + 1e-8)
            return torch.cat((norm_square, log_combined), dim=1)

        elif self.combination == 'concat':
            return torch.cat((ref_embedding, que_embedding), dim=1)

        else:
            raise ValueError(
                "Invalid combination '{}'. Must be one of: "
                "'diff', 'square', 'normalize', 'normalize_square', "
                "'normalize_square_concat_log', 'concat'.".format(self.combination)
            )

    def forward(self, reference: torch.Tensor, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through FSLNet for shift localization.

        Args:
            reference (torch.Tensor): Reference batch, shape (n_samples, n_features), dtype=torch.float32.
            query (torch.Tensor): Query batch, shape (n_samples, n_features), dtype=torch.float32.

        Returns:
            Tuple[torch.Tensor, list]:
                - y_pred (torch.Tensor): Tensor of shape (1, n_features), dtype=torch.float32.
                    Predicted probability of each feature being shifted. Values are in [0, 1].

                - embeddings (list): Intermediate outputs used in prediction:
                    - ref_neural_embedding (torch.Tensor or None): Neural embedding of the reference batch,
                    or None if the neural embedding module is not used.
                    - que_neural_embedding (torch.Tensor or None): Neural embedding of the query batch,
                    or None if the neural embedding module is not used.
                    - y_logits (torch.Tensor): Raw logits from the prediction network before sigmoid activation.
        """
        ref_embedding = []
        que_embedding = []
        
        with torch.no_grad():
            # Normalize input data as required by statistics and neural modules
            reference, query, reference_minmax, query_minmax = self.normalize_data(reference, query)
            
            # Compute per-feature statistical measures
            if self.statistical_measures is not None:
                self.statistical_measures_ref = self.statistical_measures.copy()
                self.statistical_measures_que = self.statistical_measures.copy()
                
                ref_embedding.append(self.statistical_measures_ref(reference, reference_minmax))
                que_embedding.append(self.statistical_measures_que(query, query_minmax))
        
        # Forward pass through the Neural Embedding Network
        if self.neural_embedding_net is not None:
            if self.neural_embedding_net.process_jointly:
                ref_neural_embedding, que_neural_embedding = self.neural_embedding_net([reference, query])
            else:
                ref_neural_embedding = self.neural_embedding_net([reference])
                que_neural_embedding = self.neural_embedding_net([query])
            ref_embedding.append(ref_neural_embedding)
            que_embedding.append(que_neural_embedding)
        else:
            ref_neural_embedding = None
            que_neural_embedding = None

        # Forward pass through the Moment Extraction Network
        if self.moment_extraction_net is not None:
            ref_embedding.append(self.moment_extraction_net(reference))
            que_embedding.append(self.moment_extraction_net(query))
        
        # Concatenate reference and query statistical functionals
        ref_embedding = torch.cat(ref_embedding, dim=1)
        que_embedding = torch.cat(que_embedding, dim=1)

        # Merge operation
        combined_embedding = self._merge_operation(ref_embedding, que_embedding)
        
        if self.z_embedding:
            # Compute and append z‐embedding if requested
            z_embed = self._compute_z_embedding(reference, query)
            combined_embedding = torch.cat((combined_embedding, z_embed), dim=1)
        
        # Forward pass through the Prediction Network
        y_pred, y_logits = self.prediction_net(combined_embedding)

        return y_pred, [ref_neural_embedding, que_neural_embedding, y_logits]

    @classmethod
    def _load_config(cls, model_path: Union[str, None], device: str) -> SimpleNamespace:
        """
        Construct a configuration namespace containing all required settings.
        """
        if model_path is None:
            try:
                model_path = files("fslnet.checkpoints").joinpath("fslnet.pth")
            except ModuleNotFoundError as e:
                raise FileNotFoundError("Default checkpoint not found. Please specify `model_path`.") from e

        cfg = SimpleNamespace(
            model_path=model_path or files("fslnet.checkpoints").joinpath("fslnet.pth"),
            neural_cfg=neural_cfg,
            stat_cfg=stat_cfg,
            moment_cfg=moment_cfg,
            pred_cfg=pred_cfg,
            fslnet_cfg=fslnet_cfg,
            device=device,
        )
        return cfg

    @classmethod
    def from_pretrained(cls, model_path: str | None = None, device: str = "cpu") -> FSLNet:
        """
        Instantiate FSLNet from a pretrained checkpoint.

        Args:
            model_path (str or None): Local path to checkpoint. If None, uses files("fslnet.checkpoints").joinpath("fslnet.pth").
            device (str): Device on which to load the model (e.g., 'cpu' or 'cuda:0').

        Returns:
            FSLNet: Loaded model set to eval() mode.
        """
        cfg = cls._load_config(model_path, device)
        cache_key = (cfg.model_path, cfg.device)

        # Return cached model if available
        if cache_key in cls._model_cache:
            return cls._model_cache[cache_key]

        # Build each submodule
        neural_embedding_net = NeuralEmbeddingNet(**cfg.neural_cfg).to(cfg.device)
        statistical_measures = StatisticalMeasures(**cfg.stat_cfg).to(cfg.device)
        moment_extraction_net = MomentExtractionNet(**cfg.moment_cfg).to(cfg.device)

        # Determine input dimensions for PredictionNet by dummy forward
        dummy = torch.randn(2, 1, 2).to(cfg.device)
        with torch.no_grad():
            stat_out = statistical_measures.copy()(dummy, dummy)
            stat_dim = stat_out.shape[1]
            ref_neu_dummy, _ = neural_embedding_net([dummy, dummy])
            neu_dim = ref_neu_dummy.shape[1]
            mom_out = moment_extraction_net(dummy)
            mom_dim = mom_out.shape[1]
            base_channels = stat_dim + neu_dim + mom_dim
            pred_in = base_channels + (1 if cfg.fslnet_cfg["z_embedding"] else 0)

        # Update prediction network's in_channels
        cfg.pred_cfg["in_channels"] = pred_in
        pred_net = PredictionNet(**cfg.pred_cfg).to(cfg.device)

        # Instantiate FSLNet
        model = cls(
            neural_embedding_net=neural_embedding_net,
            statistical_measures=statistical_measures,
            moment_extraction_net=moment_extraction_net,
            prediction_net=pred_net,
            z_embedding=cfg.fslnet_cfg["z_embedding"],
            eps=cfg.fslnet_cfg["eps"],
            combination=cfg.fslnet_cfg["combination"],
            normalization=cfg.fslnet_cfg["normalization"],
        ).to(cfg.device)

        # Load weights
        checkpoint_path = str(cfg.model_path)   # <- use cfg.model_path
        print(f"--> Loading FSLNet weights from '{checkpoint_path}' onto {cfg.device} ...", flush=True)
        state_dict = torch.load(checkpoint_path, map_location=torch.device(cfg.device))
        model.load_state_dict(state_dict)
        model.eval()
        print("--> FSLNet loaded and set to eval().", flush=True)

        # Cache and return
        cls._model_cache[cache_key] = model
        return model
