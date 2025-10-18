"""
Tests for FASTLight core functionality
"""

import pytest
import numpy as np
from unittest.mock import patch

from fastlight.core import FASTLight, FASTLightRLE
from fastlight.utils import create_sample_data


class TestFASTLight:
    """Test FASTLight class with Huffman encoding"""
    
    def test_initialization(self):
        """Test FASTLight initialization"""
        tokenizer = FASTLight(n_coeffs=6, scale=10.0)
        
        assert tokenizer.n_coeffs == 6
        assert tokenizer.scale == 10.0
        assert tokenizer.q01 is None
        assert tokenizer.q99 is None
        assert not tokenizer.is_fitted
    
    def test_fit_basic(self):
        """Test basic fitting functionality"""
        tokenizer = FASTLight(n_coeffs=4, scale=5.0)
        data = create_sample_data(n_episodes=5, time_horizon=15, action_dim=7)
        
        tokenizer.fit(data)
        
        assert tokenizer.is_fitted
        assert tokenizer.q01 is not None
        assert tokenizer.q99 is not None
        assert tokenizer.q01.shape == (7,)
        assert tokenizer.q99.shape == (7,)
        assert tokenizer.huffman_encoder.is_built
    
    def test_fit_invalid_input(self):
        """Test fitting with invalid input"""
        tokenizer = FASTLight()
        
        # Wrong dimensions
        with pytest.raises(ValueError, match="Expected 3D array"):
            tokenizer.fit(np.random.randn(10, 15))  # 2D instead of 3D
    
    def test_encode_decode_roundtrip(self):
        """Test encoding and decoding roundtrip"""
        tokenizer = FASTLight(n_coeffs=4, scale=5.0)
        data = create_sample_data(n_episodes=3, time_horizon=15, action_dim=7)
        
        tokenizer.fit(data)
        
        # Test single chunk
        chunk = data[0]  # (15, 7)
        tokens = tokenizer.encode(chunk)
        reconstructed = tokenizer.decode(tokens)
        
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        assert reconstructed.shape == chunk.shape
        assert reconstructed.dtype == np.float64
    
    def test_encode_decode_quality(self):
        """Test reconstruction quality"""
        tokenizer = FASTLight(n_coeffs=6, scale=10.0)
        data = create_sample_data(n_episodes=5, time_horizon=15, action_dim=7, seed=42)
        
        tokenizer.fit(data)
        
        chunk = data[0]
        tokens = tokenizer.encode(chunk)
        reconstructed = tokenizer.decode(tokens)
        
        mse = np.mean((chunk - reconstructed) ** 2)
        assert mse < 0.1  # Should have reasonable reconstruction quality
    
    def test_batch_encoding(self):
        """Test batch encoding interface"""
        tokenizer = FASTLight(n_coeffs=4, scale=5.0)
        data = create_sample_data(n_episodes=3, time_horizon=15, action_dim=7)
        
        tokenizer.fit(data)
        
        # Test batch encoding
        batch_tokens = tokenizer(data)
        assert len(batch_tokens) == 3
        assert all(isinstance(tokens, list) for tokens in batch_tokens)
        
        # Test single chunk encoding
        single_tokens = tokenizer(data[0])
        assert len(single_tokens) == 1
        assert isinstance(single_tokens[0], list)
    
    def test_encode_before_fit(self):
        """Test that encoding fails before fitting"""
        tokenizer = FASTLight()
        chunk = np.random.randn(15, 7)
        
        with pytest.raises(ValueError, match="Must fit tokenizer first"):
            tokenizer.encode(chunk)
    
    def test_decode_before_fit(self):
        """Test that decoding fails before fitting"""
        tokenizer = FASTLight()
        tokens = [1, 2, 3, 4, 5]
        
        with pytest.raises(ValueError, match="Must fit tokenizer first"):
            tokenizer.decode(tokens)
    
    def test_compression_stats(self):
        """Test compression statistics"""
        tokenizer = FASTLight(n_coeffs=4, scale=5.0)
        data = create_sample_data(n_episodes=3, time_horizon=15, action_dim=7)
        
        tokenizer.fit(data)
        stats = tokenizer.get_compression_stats()
        
        assert stats["n_coeffs"] == 4
        assert stats["scale"] == 5.0
        assert stats["is_fitted"] is True
        assert "vocab_size" in stats
    
    def test_different_parameters(self):
        """Test with different parameter combinations"""
        data = create_sample_data(n_episodes=3, time_horizon=15, action_dim=7)
        
        # Test different n_coeffs
        for n_coeffs in [3, 6, 9]:
            tokenizer = FASTLight(n_coeffs=n_coeffs, scale=10.0)
            tokenizer.fit(data)
            
            chunk = data[0]
            tokens = tokenizer.encode(chunk)
            reconstructed = tokenizer.decode(tokens)
            
            assert reconstructed.shape == chunk.shape
            assert len(tokens) > 0
    
    def test_scale_parameter(self):
        """Test different scale parameters"""
        data = create_sample_data(n_episodes=3, time_horizon=15, action_dim=7)
        
        for scale in [1.0, 5.0, 20.0]:
            tokenizer = FASTLight(n_coeffs=4, scale=scale)
            tokenizer.fit(data)
            
            chunk = data[0]
            tokens = tokenizer.encode(chunk)
            reconstructed = tokenizer.decode(tokens)
            
            assert reconstructed.shape == chunk.shape
            assert len(tokens) > 0


class TestFASTLightRLE:
    """Test FASTLightRLE class with Run-Length Encoding"""
    
    def test_initialization(self):
        """Test FASTLightRLE initialization"""
        tokenizer = FASTLightRLE(n_coeffs=6, scale=10.0)
        
        assert tokenizer.n_coeffs == 6
        assert tokenizer.scale == 10.0
        assert tokenizer.ZERO_RUN == 0
        assert tokenizer.OFFSET == 127
        assert not tokenizer.is_fitted
    
    def test_fit_basic(self):
        """Test basic fitting functionality"""
        tokenizer = FASTLightRLE(n_coeffs=4, scale=5.0)
        data = create_sample_data(n_episodes=5, time_horizon=15, action_dim=7)
        
        tokenizer.fit(data)
        
        assert tokenizer.is_fitted
        assert tokenizer.q01 is not None
        assert tokenizer.q99 is not None
        assert tokenizer.q01.shape == (7,)
        assert tokenizer.q99.shape == (7,)
    
    def test_encode_decode_roundtrip(self):
        """Test encoding and decoding roundtrip"""
        tokenizer = FASTLightRLE(n_coeffs=4, scale=5.0)
        data = create_sample_data(n_episodes=3, time_horizon=15, action_dim=7)
        
        tokenizer.fit(data)
        
        chunk = data[0]
        tokens = tokenizer.encode(chunk)
        reconstructed = tokenizer.decode(tokens)
        
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        assert reconstructed.shape == chunk.shape
    
    def test_rle_encoding(self):
        """Test RLE encoding of zeros"""
        tokenizer = FASTLightRLE(n_coeffs=4, scale=5.0)
        data = create_sample_data(n_episodes=3, time_horizon=15, action_dim=7)
        
        tokenizer.fit(data)
        
        # Create chunk with many zeros
        chunk = np.zeros((15, 7))
        chunk[0, 0] = 1.0
        chunk[14, 6] = -1.0
        
        tokens = tokenizer.encode(chunk)
        reconstructed = tokenizer.decode(tokens)
        
        # Should compress well due to many zeros
        assert len(tokens) < 50  # Should be much smaller than naive encoding
        assert reconstructed.shape == chunk.shape
    
    def test_batch_encoding(self):
        """Test batch encoding interface"""
        tokenizer = FASTLightRLE(n_coeffs=4, scale=5.0)
        data = create_sample_data(n_episodes=3, time_horizon=15, action_dim=7)
        
        tokenizer.fit(data)
        
        batch_tokens = tokenizer(data)
        assert len(batch_tokens) == 3
        assert all(isinstance(tokens, list) for tokens in batch_tokens)


class TestFASTLightComparison:
    """Test comparison between FASTLight variants"""
    
    def test_huffman_vs_rle_compression(self):
        """Test compression comparison between Huffman and RLE"""
        data = create_sample_data(n_episodes=5, time_horizon=15, action_dim=7, seed=42)
        
        # Initialize both tokenizers
        huffman_tokenizer = FASTLight(n_coeffs=4, scale=5.0)
        rle_tokenizer = FASTLightRLE(n_coeffs=4, scale=5.0)
        
        # Fit both
        huffman_tokenizer.fit(data)
        rle_tokenizer.fit(data)
        
        # Test on same chunk
        chunk = data[0]
        
        huffman_tokens = huffman_tokenizer.encode(chunk)
        rle_tokens = rle_tokenizer.encode(chunk)
        
        huffman_reconstructed = huffman_tokenizer.decode(huffman_tokens)
        rle_reconstructed = rle_tokenizer.decode(rle_tokens)
        
        # Both should work
        assert huffman_reconstructed.shape == chunk.shape
        assert rle_reconstructed.shape == chunk.shape
        
        # Calculate MSE for both
        huffman_mse = np.mean((chunk - huffman_reconstructed) ** 2)
        rle_mse = np.mean((chunk - rle_reconstructed) ** 2)
        
        # Both should have reasonable quality
        assert huffman_mse < 0.1
        assert rle_mse < 0.1
    
    def test_parameter_consistency(self):
        """Test that both variants handle parameters consistently"""
        data = create_sample_data(n_episodes=3, time_horizon=15, action_dim=7)
        
        for n_coeffs in [3, 6]:
            for scale in [5.0, 10.0]:
                huffman_tokenizer = FASTLight(n_coeffs=n_coeffs, scale=scale)
                rle_tokenizer = FASTLightRLE(n_coeffs=n_coeffs, scale=scale)
                
                huffman_tokenizer.fit(data)
                rle_tokenizer.fit(data)
                
                chunk = data[0]
                
                huffman_tokens = huffman_tokenizer.encode(chunk)
                rle_tokens = rle_tokenizer.encode(chunk)
                
                # Both should produce valid tokens
                assert len(huffman_tokens) > 0
                assert len(rle_tokens) > 0
                
                # Both should reconstruct properly
                huffman_reconstructed = huffman_tokenizer.decode(huffman_tokens)
                rle_reconstructed = rle_tokenizer.decode(rle_tokens)
                
                assert huffman_reconstructed.shape == chunk.shape
                assert rle_reconstructed.shape == chunk.shape


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_chunk(self):
        """Test handling of edge case data"""
        tokenizer = FASTLight(n_coeffs=4, scale=5.0)
        
        # Create data with very small values
        data = np.random.randn(3, 15, 7) * 0.001
        tokenizer.fit(data)
        
        chunk = data[0]
        tokens = tokenizer.encode(chunk)
        reconstructed = tokenizer.decode(tokens)
        
        assert reconstructed.shape == chunk.shape
    
    def test_large_values(self):
        """Test handling of large values"""
        tokenizer = FASTLight(n_coeffs=4, scale=5.0)
        
        # Create data with large values
        data = np.random.randn(3, 15, 7) * 100
        tokenizer.fit(data)
        
        chunk = data[0]
        tokens = tokenizer.encode(chunk)
        reconstructed = tokenizer.decode(tokens)
        
        assert reconstructed.shape == chunk.shape
    
    def test_constant_values(self):
        """Test handling of constant values"""
        tokenizer = FASTLight(n_coeffs=4, scale=5.0)
        
        # Create data with constant values
        data = np.ones((3, 15, 7)) * 5.0
        tokenizer.fit(data)
        
        chunk = data[0]
        tokens = tokenizer.encode(chunk)
        reconstructed = tokenizer.decode(tokens)
        
        assert reconstructed.shape == chunk.shape
        # Should still work even with constant data
