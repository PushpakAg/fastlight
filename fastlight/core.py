"""
Core FASTLight implementation with Huffman encoding

This module contains the main FASTLight class that combines DCT compression
with Huffman encoding for optimal action sequence compression.
"""

import numpy as np
from scipy.fftpack import dct, idct
from typing import List, Optional, Tuple, Union
from collections import Counter

from .huffman import HuffmanEncoder


class FASTLight:
    """
    Lightweight action tokenizer for edge deployment with Huffman encoding
    
    FASTLight combines DCT (Discrete Cosine Transform) compression with Huffman
    encoding to achieve superior compression ratios while maintaining reconstruction
    quality suitable for robotics applications.
    
    Key Features:
    - DCT-based compression with configurable coefficients
    - Huffman encoding for optimal bit-level compression
    - Minimal memory footprint (no BPE overhead)
    - Fast encode/decode suitable for real-time applications
    - Edge device optimized
    
    Design:
    - k=6 DCT coefficients (captures 99.5% energy)
    - Huffman encoding for quantized coefficients
    - 8-bit direct encoding with escape codes for rare values
    - No BPE (simpler, faster)
    
    Target Performance:
    - 27-30 tokens/chunk (vs FAST's 40)
    - MSE < 0.001 (vs FAST's 0.0007)
    - 10x faster encode/decode than FAST
    - 4x smaller vocabulary (256 vs 1024)
    """
    
    def __init__(self, n_coeffs: int = 6, scale: float = 10.0):
        """
        Initialize FASTLight tokenizer
        
        Args:
            n_coeffs: Number of DCT coefficients to keep (default: 6)
            scale: Quantization scale factor (default: 10.0)
        """
        self.n_coeffs = n_coeffs
        self.scale = scale
        
        # Normalization parameters (learned from training data)
        self.q01: Optional[np.ndarray] = None
        self.q99: Optional[np.ndarray] = None
        
        # Huffman encoder
        self.huffman_encoder = HuffmanEncoder()
        
        # Track fitting status
        self.is_fitted = False
    
    def fit(self, actions: np.ndarray) -> 'FASTLight':
        """
        Learn normalization parameters and build Huffman tree from training data
        
        Args:
            actions: Training data of shape (N, H, D) where:
                N = number of episodes
                H = time horizon (e.g., 15)
                D = action dimensions (e.g., 7)
        
        Returns:
            Self for method chaining
        """
        if actions.ndim != 3:
            raise ValueError(f"Expected 3D array (N, H, D), got {actions.ndim}D")
        
        # Learn global normalization parameters
        all_data = actions.reshape(-1, actions.shape[-1])
        self.q01 = np.percentile(all_data, 1, axis=0)
        self.q99 = np.percentile(all_data, 99, axis=0)
        
        # Collect all quantized DCT coefficients to build frequency table
        print(f"Building Huffman tree from {actions.shape[0]} episodes...")
        all_coeffs = []
        
        for episode in actions:
            # Normalize
            normalized = self._normalize_chunk(episode)
            
            # DCT per dimension
            dct_coeffs = self._apply_dct(normalized)
            
            # Keep top-k coefficients
            dct_truncated = dct_coeffs[:self.n_coeffs, :]
            
            # Quantize
            quantized = np.round(dct_truncated * self.scale).astype(np.int16)
            
            # Flatten (column-first: low-freq components first)
            flattened = quantized.T.flatten()
            all_coeffs.extend(flattened)
        
        # Build Huffman tree from coefficient frequencies
        value_freq = Counter(all_coeffs)
        self.huffman_encoder.build_tree(value_freq)
        
        self.is_fitted = True
        
        print(f"  Huffman tree built with {len(self.huffman_encoder.encode_table)} symbols")
        print(f"  Most common values: {value_freq.most_common(10)}")
        print(f"  Average code length: {self.huffman_encoder.get_average_code_length(value_freq):.2f} bits")
        
        return self
    
    def encode(self, action_chunk: np.ndarray) -> List[int]:
        """
        Encode action chunk to tokens using DCT + Huffman encoding
        
        Args:
            action_chunk: Action chunk of shape (H, D)
            
        Returns:
            List of bytes representing the encoded chunk
        """
        if not self.is_fitted:
            raise ValueError("Must fit tokenizer first")
        
        if action_chunk.ndim != 2:
            raise ValueError(f"Expected 2D array (H, D), got {action_chunk.ndim}D")
        
        H, D = action_chunk.shape
        
        # 1. Normalize
        normalized = self._normalize_chunk(action_chunk)
        
        # 2. DCT per dimension
        dct_coeffs = self._apply_dct(normalized)
        
        # 3. Keep only top-k coefficients
        dct_truncated = dct_coeffs[:self.n_coeffs, :]
        
        # 4. Quantize to integers
        quantized = np.round(dct_truncated * self.scale).astype(np.int16)
        
        # 5. Flatten (column-first: low-freq components first)
        flattened = quantized.T.flatten()
        
        # 6. Huffman encode to bitstring
        bitstring = self.huffman_encoder.encode_symbols(flattened.tolist())
        
        # 7. Pack bits into bytes
        tokens = self._pack_bits_to_bytes(bitstring)
        
        return tokens
    
    def decode(self, tokens: List[int], time_horizon: int = 15, action_dim: int = 7) -> np.ndarray:
        """
        Decode tokens back to actions
        
        Args:
            tokens: List of bytes representing encoded chunk
            time_horizon: Expected time horizon H
            action_dim: Expected action dimension D
            
        Returns:
            Reconstructed action chunk of shape (H, D)
        """
        if not self.is_fitted:
            raise ValueError("Must fit tokenizer first")
        
        # 1. Unpack bytes to bitstring
        bitstring = self._unpack_bytes_to_bits(tokens)
        
        # 2. Huffman decode bitstring to values
        values = self.huffman_encoder.decode_bitstring(bitstring)
        
        # 3. Reshape to (D, k)
        expected_size = action_dim * self.n_coeffs
        if len(values) < expected_size:
            # Pad with zeros if we don't have enough values
            values.extend([0] * (expected_size - len(values)))
        elif len(values) > expected_size:
            # Truncate if we have too many values (shouldn't happen with proper encoding)
            values = values[:expected_size]
        
        # Ensure we have exactly the expected number of values
        values = values[:expected_size]
        
        quantized = np.array(values).reshape(action_dim, self.n_coeffs).T
        
        # 4. Dequantize
        dct_coeffs = quantized.astype(float) / self.scale
        
        # 5. Pad with zeros to full time horizon
        full_dct = np.zeros((time_horizon, action_dim))
        full_dct[:self.n_coeffs, :] = dct_coeffs
        
        # 6. Inverse DCT per dimension
        normalized = self._apply_idct(full_dct)
        
        # 7. Denormalize
        actions = self._denormalize_chunk(normalized)
        
        return actions
    
    def _normalize_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Normalize chunk to [-1, 1] range"""
        if self.q01 is None or self.q99 is None:
            raise ValueError("Normalization parameters not fitted")
        
        normalized = 2 * (chunk - self.q01) / (self.q99 - self.q01 + 1e-8) - 1
        return np.clip(normalized, -1, 1)
    
    def _denormalize_chunk(self, normalized: np.ndarray) -> np.ndarray:
        """Denormalize chunk from [-1, 1] range back to original scale"""
        if self.q01 is None or self.q99 is None:
            raise ValueError("Normalization parameters not fitted")
        
        return (normalized + 1) * (self.q99 - self.q01) / 2 + self.q01
    
    def _apply_dct(self, normalized: np.ndarray) -> np.ndarray:
        """Apply DCT to each dimension"""
        dct_coeffs = np.zeros_like(normalized)
        for dim in range(normalized.shape[1]):
            dct_coeffs[:, dim] = dct(normalized[:, dim], norm='ortho')
        return dct_coeffs
    
    def _apply_idct(self, dct_coeffs: np.ndarray) -> np.ndarray:
        """Apply inverse DCT to each dimension"""
        normalized = np.zeros_like(dct_coeffs)
        for dim in range(dct_coeffs.shape[1]):
            normalized[:, dim] = idct(dct_coeffs[:, dim], norm='ortho')
        return normalized
    
    def _pack_bits_to_bytes(self, bitstring: str) -> List[int]:
        """Pack bitstring into list of bytes"""
        # Pad to multiple of 8
        padding = (8 - len(bitstring) % 8) % 8
        bitstring += "0" * padding
        
        # Convert to bytes
        tokens = []
        for i in range(0, len(bitstring), 8):
            byte = int(bitstring[i:i+8], 2)
            tokens.append(byte)
        
        # Store padding length in first byte
        tokens.insert(0, padding)
        
        return tokens
    
    def _unpack_bytes_to_bits(self, tokens: List[int]) -> str:
        """Unpack bytes to bitstring"""
        padding = tokens[0]
        bitstring = ""
        for byte in tokens[1:]:
            bitstring += format(byte, '08b')
        
        # Remove padding
        if padding > 0:
            bitstring = bitstring[:-padding]
        
        return bitstring
    
    def __call__(self, action_chunks: np.ndarray) -> List[List[int]]:
        """
        Batch encode (compatible with FAST interface)
        
        Args:
            action_chunks: (B, H, D) numpy array or (H, D) for single chunk
            
        Returns:
            List of token lists
        """
        if action_chunks.ndim == 2:
            # Single chunk
            return [self.encode(action_chunks)]
        else:
            # Batch
            return [self.encode(chunk) for chunk in action_chunks]
    
    def get_compression_stats(self) -> dict:
        """
        Get compression statistics
        
        Returns:
            Dictionary with compression metrics
        """
        if not self.is_fitted:
            return {"error": "Not fitted"}
        
        return {
            "n_coeffs": self.n_coeffs,
            "scale": self.scale,
            "vocab_size": len(self.huffman_encoder.encode_table),
            "is_fitted": self.is_fitted,
        }


class FASTLightRLE:
    """
    FASTLight variant using Run-Length Encoding instead of Huffman
    
    This is a simpler alternative that uses RLE for zero compression,
    which may be faster for some applications but typically achieves
    lower compression ratios than Huffman encoding.
    """
    
    def __init__(self, n_coeffs: int = 6, scale: float = 10.0):
        """
        Initialize FASTLight RLE tokenizer
        
        Args:
            n_coeffs: Number of DCT coefficients to keep
            scale: Quantization scale factor
        """
        self.n_coeffs = n_coeffs
        self.scale = scale
        
        # Special tokens for RLE
        self.ZERO_RUN = 0
        self.MIN_VALUE = 1
        self.MAX_VALUE = 254
        self.OFFSET = 127
        
        # Normalization parameters
        self.q01: Optional[np.ndarray] = None
        self.q99: Optional[np.ndarray] = None
        self.is_fitted = False
    
    def fit(self, actions: np.ndarray) -> 'FASTLightRLE':
        """Fit normalization parameters"""
        all_data = actions.reshape(-1, actions.shape[-1])
        self.q01 = np.percentile(all_data, 1, axis=0)
        self.q99 = np.percentile(all_data, 99, axis=0)
        self.is_fitted = True
        return self
    
    def encode(self, action_chunk: np.ndarray) -> List[int]:
        """Encode using DCT + RLE"""
        if not self.is_fitted:
            raise ValueError("Must fit tokenizer first")
        
        # Normalize and apply DCT (same as Huffman version)
        normalized = 2 * (action_chunk - self.q01) / (self.q99 - self.q01 + 1e-8) - 1
        normalized = np.clip(normalized, -1, 1)
        
        dct_coeffs = np.zeros_like(normalized)
        for dim in range(normalized.shape[1]):
            dct_coeffs[:, dim] = dct(normalized[:, dim], norm='ortho')
        
        dct_truncated = dct_coeffs[:self.n_coeffs, :]
        quantized = np.round(dct_truncated * self.scale).astype(np.int16)
        flattened = quantized.T.flatten()
        
        # RLE encoding
        return self._encode_with_rle(flattened)
    
    def decode(self, tokens: List[int], time_horizon: int = 15, action_dim: int = 7) -> np.ndarray:
        """Decode using RLE + inverse DCT"""
        if not self.is_fitted:
            raise ValueError("Must fit tokenizer first")
        
        values = self._decode_rle(tokens)
        expected_size = action_dim * self.n_coeffs
        if len(values) < expected_size:
            values.extend([0] * (expected_size - len(values)))
        values = values[:expected_size]
        
        quantized = np.array(values).reshape(action_dim, self.n_coeffs).T
        dct_coeffs = quantized.astype(float) / self.scale
        
        full_dct = np.zeros((time_horizon, action_dim))
        full_dct[:self.n_coeffs, :] = dct_coeffs
        
        normalized = np.zeros((time_horizon, action_dim))
        for dim in range(action_dim):
            normalized[:, dim] = idct(full_dct[:, dim], norm='ortho')
        
        actions = (normalized + 1) * (self.q99 - self.q01) / 2 + self.q01
        return actions
    
    def _encode_with_rle(self, values: np.ndarray) -> List[int]:
        """Encode with run-length encoding for zeros"""
        tokens = []
        zero_count = 0
        
        for val in values:
            if val == 0:
                zero_count += 1
            else:
                # Flush accumulated zeros
                if zero_count > 0:
                    while zero_count > 0:
                        chunk_size = min(zero_count, 253)
                        tokens.extend([self.ZERO_RUN, chunk_size])
                        zero_count -= chunk_size
                
                # Encode non-zero value
                token = int(np.clip(val + self.OFFSET, self.MIN_VALUE, self.MAX_VALUE))
                tokens.append(token)
        
        # Flush remaining zeros
        if zero_count > 0:
            while zero_count > 0:
                chunk_size = min(zero_count, 253)
                tokens.extend([self.ZERO_RUN, chunk_size])
                zero_count -= chunk_size
        
        return tokens
    
    def _decode_rle(self, tokens: List[int]) -> List[int]:
        """Decode run-length encoding"""
        values = []
        i = 0
        
        while i < len(tokens):
            if tokens[i] == self.ZERO_RUN:
                if i + 1 < len(tokens):
                    zero_count = tokens[i + 1]
                    values.extend([0] * zero_count)
                    i += 2
                else:
                    i += 1
            else:
                val = tokens[i] - self.OFFSET
                values.append(val)
                i += 1
        
        return values
    
    def __call__(self, action_chunks: np.ndarray) -> List[List[int]]:
        """Batch encode"""
        if action_chunks.ndim == 2:
            return [self.encode(action_chunks)]
        else:
            return [self.encode(chunk) for chunk in action_chunks]
