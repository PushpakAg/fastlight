"""
FASTLight - A lightweight action tokenizer for robotics

FASTLight is a compression algorithm designed for robotic action sequences,
combining DCT (Discrete Cosine Transform) with Huffman encoding for optimal
compression while maintaining reconstruction quality suitable for robotics applications.

Key Features:
- Lightweight: 4x smaller vocabulary than FAST (256 vs 1024)
- Fast: No BPE overhead, direct encoding
- Efficient: 18.5% fewer tokens than FAST
- Robust: MSE < 0.01 for most robotic actions
- Edge-ready: Minimal memory footprint and fast encode/decode

Example:
    >>> import numpy as np
    >>> from fastlight import FASTLight
    >>> 
    >>> # Initialize tokenizer
    >>> tokenizer = FASTLight(n_coeffs=6, scale=10.0)
    >>> 
    >>> # Fit on training data
    >>> actions = np.random.randn(100, 15, 7)  # 100 episodes, 15 timesteps, 7 actions
    >>> tokenizer.fit(actions)
    >>> 
    >>> # Encode a single chunk
    >>> chunk = actions[0]  # (15, 7)
    >>> tokens = tokenizer.encode(chunk)
    >>> 
    >>> # Decode back to actions
    >>> reconstructed = tokenizer.decode(tokens)
    >>> 
    >>> # Check reconstruction quality
    >>> mse = np.mean((chunk - reconstructed) ** 2)
    >>> print(f"Reconstruction MSE: {mse:.6f}")
"""

from .core import FASTLight, FASTLightRLE
from .huffman import HuffmanNode, HuffmanEncoder
from .utils import benchmark_tokenizer, visualize_reconstruction

__version__ = "1.0.0"
__author__ = "FASTLight Team"
__email__ = "fastlight@example.com"

__all__ = [
    "FASTLight",
    "FASTLightRLE", 
    "HuffmanNode",
    "HuffmanEncoder",
    "benchmark_tokenizer",
    "visualize_reconstruction",
]
