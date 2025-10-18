"""
Utility functions for FASTLight

This module provides helper functions for benchmarking, visualization,
and analysis of FASTLight tokenizers.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from .core import FASTLight, FASTLightRLE


def benchmark_tokenizer(
    tokenizer: Any,
    test_data: np.ndarray,
    n_runs: int = 10
) -> Dict[str, float]:
    """
    Benchmark tokenizer performance
    
    Args:
        tokenizer: Tokenizer to benchmark
        test_data: Test data of shape (N, H, D)
        n_runs: Number of benchmark runs
        
    Returns:
        Dictionary with timing statistics
    """
    encode_times = []
    decode_times = []
    token_counts = []
    
    for _ in range(n_runs):
        # Encode timing
        start_time = time.perf_counter()
        tokens_list = []
        for chunk in test_data:
            tokens = tokenizer.encode(chunk)
            tokens_list.append(tokens)
        encode_time = time.perf_counter() - start_time
        
        # Decode timing
        start_time = time.perf_counter()
        for i, tokens in enumerate(tokens_list):
            _ = tokenizer.decode(tokens)
        decode_time = time.perf_counter() - start_time
        
        encode_times.append(encode_time / len(test_data))
        decode_times.append(decode_time / len(test_data))
        token_counts.append([len(tokens) for tokens in tokens_list])
    
    return {
        "encode_time_ms": np.mean(encode_times) * 1000,
        "decode_time_ms": np.mean(decode_times) * 1000,
        "total_time_ms": (np.mean(encode_times) + np.mean(decode_times)) * 1000,
        "avg_tokens_per_chunk": np.mean([np.mean(counts) for counts in token_counts]),
        "std_tokens_per_chunk": np.mean([np.std(counts) for counts in token_counts]),
    }


def compare_tokenizers(
    tokenizers: Dict[str, Any],
    test_data: np.ndarray,
    n_runs: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple tokenizers
    
    Args:
        tokenizers: Dictionary mapping names to tokenizer instances
        test_data: Test data of shape (N, H, D)
        n_runs: Number of benchmark runs
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    for name, tokenizer in tokenizers.items():
        print(f"Benchmarking {name}...")
        results[name] = benchmark_tokenizer(tokenizer, test_data, n_runs)
    
    return results


def visualize_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    title: str = "Reconstruction Comparison",
    save_path: Optional[str] = None
) -> None:
    """
    Visualize original vs reconstructed action sequences
    
    Args:
        original: Original action sequence (H, D)
        reconstructed: Reconstructed action sequence (H, D)
        title: Plot title
        save_path: Optional path to save the plot
    """
    H, D = original.shape
    dim_names = [f'Joint {i}' for i in range(D-1)] + ['Gripper']
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(title, fontsize=16)
    
    timesteps = np.arange(H)
    
    for i in range(D):
        ax = axes[i // 4, i % 4]
        
        ax.plot(timesteps, original[:, i], 'b-', label='Original', linewidth=2)
        ax.plot(timesteps, reconstructed[:, i], 'r--', label='Reconstructed', linewidth=2)
        
        ax.set_title(dim_names[i])
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplot
    if D < 8:
        axes[1, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def analyze_compression_quality(
    tokenizer: Any,
    test_data: np.ndarray
) -> Dict[str, Any]:
    """
    Analyze compression quality metrics
    
    Args:
        tokenizer: Tokenizer to analyze
        test_data: Test data of shape (N, H, D)
        
    Returns:
        Dictionary with quality metrics
    """
    mse_values = []
    token_counts = []
    
    for chunk in test_data:
        # Encode and decode
        tokens = tokenizer.encode(chunk)
        reconstructed = tokenizer.decode(tokens)
        
        # Calculate MSE
        mse = np.mean((chunk - reconstructed) ** 2)
        mse_values.append(mse)
        token_counts.append(len(tokens))
    
    # Per-dimension analysis
    dim_mse = []
    for chunk in test_data:
        tokens = tokenizer.encode(chunk)
        reconstructed = tokenizer.decode(tokens)
        
        chunk_dim_mse = []
        for dim in range(chunk.shape[1]):
            dim_mse_val = np.mean((chunk[:, dim] - reconstructed[:, dim]) ** 2)
            chunk_dim_mse.append(dim_mse_val)
        dim_mse.append(chunk_dim_mse)
    
    dim_mse = np.array(dim_mse)
    
    return {
        "overall_mse": {
            "mean": np.mean(mse_values),
            "median": np.median(mse_values),
            "std": np.std(mse_values),
            "min": np.min(mse_values),
            "max": np.max(mse_values),
        },
        "per_dimension_mse": {
            "mean": np.mean(dim_mse, axis=0).tolist(),
            "std": np.std(dim_mse, axis=0).tolist(),
        },
        "token_counts": {
            "mean": np.mean(token_counts),
            "std": np.std(token_counts),
            "min": np.min(token_counts),
            "max": np.max(token_counts),
        },
        "compression_ratio": (test_data.shape[1] * test_data.shape[2]) / np.mean(token_counts),
    }


def plot_compression_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot compression comparison between tokenizers
    
    Args:
        results: Results from compare_tokenizers
        save_path: Optional path to save the plot
    """
    names = list(results.keys())
    
    # Extract metrics
    encode_times = [results[name]["encode_time_ms"] for name in names]
    decode_times = [results[name]["decode_time_ms"] for name in names]
    avg_tokens = [results[name]["avg_tokens_per_chunk"] for name in names]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Encode time
    axes[0, 0].bar(names, encode_times, alpha=0.7, edgecolor='black')
    axes[0, 0].set_ylabel('Encode Time (ms)')
    axes[0, 0].set_title('Encoding Speed')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Decode time
    axes[0, 1].bar(names, decode_times, alpha=0.7, edgecolor='black')
    axes[0, 1].set_ylabel('Decode Time (ms)')
    axes[0, 1].set_title('Decoding Speed')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Token count
    axes[1, 0].bar(names, avg_tokens, alpha=0.7, edgecolor='black')
    axes[1, 0].set_ylabel('Avg Tokens per Chunk')
    axes[1, 0].set_title('Compression')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Total time
    total_times = [encode_times[i] + decode_times[i] for i in range(len(names))]
    axes[1, 1].bar(names, total_times, alpha=0.7, edgecolor='black')
    axes[1, 1].set_ylabel('Total Time (ms)')
    axes[1, 1].set_title('Total Processing Time')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def create_sample_data(
    n_episodes: int = 10,
    time_horizon: int = 15,
    action_dim: int = 7,
    seed: int = 42
) -> np.ndarray:
    """
    Create sample robotic action data for testing
    
    Args:
        n_episodes: Number of episodes
        time_horizon: Time steps per episode
        action_dim: Number of action dimensions
        seed: Random seed
        
    Returns:
        Sample data of shape (n_episodes, time_horizon, action_dim)
    """
    np.random.seed(seed)
    
    # Generate realistic robotic action data
    data = []
    for _ in range(n_episodes):
        # Create smooth trajectories with some noise
        episode = np.zeros((time_horizon, action_dim))
        
        for dim in range(action_dim):
            # Generate smooth trajectory
            t = np.linspace(0, 2 * np.pi, time_horizon)
            base_signal = np.sin(t) + 0.5 * np.sin(2 * t)
            
            # Add some random variation
            noise = np.random.normal(0, 0.1, time_horizon)
            episode[:, dim] = base_signal + noise
        
        data.append(episode)
    
    return np.array(data)


def print_compression_report(
    tokenizer: Any,
    test_data: np.ndarray,
    name: str = "FASTLight"
) -> None:
    """
    Print a comprehensive compression report
    
    Args:
        tokenizer: Tokenizer to analyze
        test_data: Test data
        name: Tokenizer name for the report
    """
    print(f"\n{'='*60}")
    print(f"COMPRESSION REPORT: {name}")
    print(f"{'='*60}")
    
    # Get quality analysis
    quality = analyze_compression_quality(tokenizer, test_data)
    
    # Get benchmark results
    benchmark = benchmark_tokenizer(tokenizer, test_data)
    
    print(f"\n📊 Compression Metrics:")
    print(f"   Avg tokens per chunk: {quality['token_counts']['mean']:.1f} ± {quality['token_counts']['std']:.1f}")
    print(f"   Compression ratio: {quality['compression_ratio']:.2f}x")
    print(f"   Token range: {quality['token_counts']['min']}-{quality['token_counts']['max']}")
    
    print(f"\n🎯 Reconstruction Quality:")
    print(f"   Mean MSE: {quality['overall_mse']['mean']:.6f}")
    print(f"   Median MSE: {quality['overall_mse']['median']:.6f}")
    print(f"   MSE range: {quality['overall_mse']['min']:.6f} - {quality['overall_mse']['max']:.6f}")
    
    print(f"\n⚡ Performance:")
    print(f"   Encode time: {benchmark['encode_time_ms']:.3f} ms/chunk")
    print(f"   Decode time: {benchmark['decode_time_ms']:.3f} ms/chunk")
    print(f"   Total time: {benchmark['total_time_ms']:.3f} ms/chunk")
    
    if hasattr(tokenizer, 'get_compression_stats'):
        stats = tokenizer.get_compression_stats()
        print(f"\n🔧 Configuration:")
        print(f"   DCT coefficients: {stats.get('n_coeffs', 'N/A')}")
        print(f"   Quantization scale: {stats.get('scale', 'N/A')}")
        print(f"   Vocabulary size: {stats.get('vocab_size', 'N/A')}")
    
    print(f"\n{'='*60}")
