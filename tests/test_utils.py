"""
Tests for utility functions
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from fastlight.core import FASTLight, FASTLightRLE
from fastlight.utils import (
    benchmark_tokenizer,
    compare_tokenizers,
    visualize_reconstruction,
    analyze_compression_quality,
    plot_compression_comparison,
    create_sample_data,
    print_compression_report,
)


class TestSampleData:
    """Test sample data generation"""
    
    def test_create_sample_data_basic(self):
        """Test basic sample data creation"""
        data = create_sample_data(n_episodes=5, time_horizon=15, action_dim=7, seed=42)
        
        assert data.shape == (5, 15, 7)
        assert data.dtype == np.float64
        assert not np.any(np.isnan(data))
        assert not np.any(np.isinf(data))
    
    def test_create_sample_data_deterministic(self):
        """Test that sample data is deterministic with same seed"""
        data1 = create_sample_data(n_episodes=3, time_horizon=10, action_dim=5, seed=123)
        data2 = create_sample_data(n_episodes=3, time_horizon=10, action_dim=5, seed=123)
        
        np.testing.assert_array_equal(data1, data2)
    
    def test_create_sample_data_different_seeds(self):
        """Test that different seeds produce different data"""
        data1 = create_sample_data(n_episodes=3, time_horizon=10, action_dim=5, seed=123)
        data2 = create_sample_data(n_episodes=3, time_horizon=10, action_dim=5, seed=456)
        
        assert not np.array_equal(data1, data2)
    
    def test_create_sample_data_parameters(self):
        """Test different parameter combinations"""
        for n_episodes in [1, 5, 10]:
            for time_horizon in [5, 15, 30]:
                for action_dim in [3, 7, 12]:
                    data = create_sample_data(
                        n_episodes=n_episodes,
                        time_horizon=time_horizon,
                        action_dim=action_dim,
                        seed=42
                    )
                    
                    assert data.shape == (n_episodes, time_horizon, action_dim)


class TestBenchmarking:
    """Test benchmarking functionality"""
    
    def test_benchmark_tokenizer_basic(self):
        """Test basic benchmarking"""
        tokenizer = FASTLight(n_coeffs=4, scale=5.0)
        data = create_sample_data(n_episodes=3, time_horizon=15, action_dim=7, seed=42)
        
        tokenizer.fit(data)
        
        results = benchmark_tokenizer(tokenizer, data, n_runs=3)
        
        assert "encode_time_ms" in results
        assert "decode_time_ms" in results
        assert "total_time_ms" in results
        assert "avg_tokens_per_chunk" in results
        assert "std_tokens_per_chunk" in results
        
        assert results["encode_time_ms"] > 0
        assert results["decode_time_ms"] > 0
        assert results["total_time_ms"] > 0
        assert results["avg_tokens_per_chunk"] > 0
    
    def test_compare_tokenizers(self):
        """Test tokenizer comparison"""
        data = create_sample_data(n_episodes=3, time_horizon=15, action_dim=7, seed=42)
        
        huffman_tokenizer = FASTLight(n_coeffs=4, scale=5.0)
        rle_tokenizer = FASTLightRLE(n_coeffs=4, scale=5.0)
        
        huffman_tokenizer.fit(data)
        rle_tokenizer.fit(data)
        
        tokenizers = {
            "Huffman": huffman_tokenizer,
            "RLE": rle_tokenizer,
        }
        
        results = compare_tokenizers(tokenizers, data, n_runs=2)
        
        assert "Huffman" in results
        assert "RLE" in results
        
        for name in ["Huffman", "RLE"]:
            assert "encode_time_ms" in results[name]
            assert "decode_time_ms" in results[name]
            assert "avg_tokens_per_chunk" in results[name]


class TestCompressionAnalysis:
    """Test compression quality analysis"""
    
    def test_analyze_compression_quality(self):
        """Test compression quality analysis"""
        tokenizer = FASTLight(n_coeffs=4, scale=5.0)
        data = create_sample_data(n_episodes=3, time_horizon=15, action_dim=7, seed=42)
        
        tokenizer.fit(data)
        
        quality = analyze_compression_quality(tokenizer, data)
        
        assert "overall_mse" in quality
        assert "per_dimension_mse" in quality
        assert "token_counts" in quality
        assert "compression_ratio" in quality
        
        # Check overall MSE structure
        overall_mse = quality["overall_mse"]
        assert "mean" in overall_mse
        assert "median" in overall_mse
        assert "std" in overall_mse
        assert "min" in overall_mse
        assert "max" in overall_mse
        
        # Check per-dimension MSE structure
        per_dim_mse = quality["per_dimension_mse"]
        assert "mean" in per_dim_mse
        assert "std" in per_dim_mse
        assert len(per_dim_mse["mean"]) == 7  # 7 action dimensions
        
        # Check token counts structure
        token_counts = quality["token_counts"]
        assert "mean" in token_counts
        assert "std" in token_counts
        assert "min" in token_counts
        assert "max" in token_counts
        
        # Check compression ratio
        assert quality["compression_ratio"] > 0
    
    def test_compression_quality_values(self):
        """Test that compression quality values are reasonable"""
        tokenizer = FASTLight(n_coeffs=4, scale=5.0)
        data = create_sample_data(n_episodes=5, time_horizon=15, action_dim=7, seed=42)
        
        tokenizer.fit(data)
        
        quality = analyze_compression_quality(tokenizer, data)
        
        # MSE should be reasonable
        assert 0 <= quality["overall_mse"]["mean"] < 1.0
        assert 0 <= quality["overall_mse"]["min"] <= quality["overall_mse"]["max"]
        
        # Token counts should be reasonable
        assert quality["token_counts"]["min"] > 0
        assert quality["token_counts"]["mean"] > 0
        assert quality["token_counts"]["min"] <= quality["token_counts"]["max"]
        
        # Compression ratio should be positive
        assert quality["compression_ratio"] > 0


class TestVisualization:
    """Test visualization functions"""
    
    def test_visualize_reconstruction_basic(self):
        """Test basic reconstruction visualization"""
        # Create test data
        original = np.random.randn(15, 7)
        reconstructed = original + np.random.normal(0, 0.1, (15, 7))
        
        # Test that function runs without error
        # Note: We can't easily test the actual plot, but we can test that it doesn't crash
        try:
            visualize_reconstruction(original, reconstructed, "Test Plot")
            # If we get here, the function ran successfully
            assert True
        except Exception as e:
            pytest.fail(f"visualize_reconstruction raised an exception: {e}")
    
    def test_plot_compression_comparison(self):
        """Test compression comparison plotting"""
        # Create mock results
        results = {
            "Huffman": {
                "encode_time_ms": 1.5,
                "decode_time_ms": 2.0,
                "avg_tokens_per_chunk": 25.0,
            },
            "RLE": {
                "encode_time_ms": 1.0,
                "decode_time_ms": 1.5,
                "avg_tokens_per_chunk": 30.0,
            },
        }
        
        # Test that function runs without error
        try:
            plot_compression_comparison(results)
            # If we get here, the function ran successfully
            assert True
        except Exception as e:
            pytest.fail(f"plot_compression_comparison raised an exception: {e}")


class TestCompressionReport:
    """Test compression report functionality"""
    
    def test_print_compression_report(self):
        """Test compression report printing"""
        tokenizer = FASTLight(n_coeffs=4, scale=5.0)
        data = create_sample_data(n_episodes=3, time_horizon=15, action_dim=7, seed=42)
        
        tokenizer.fit(data)
        
        # Test that function runs without error
        # We can't easily test the printed output, but we can test that it doesn't crash
        try:
            print_compression_report(tokenizer, data, "Test Tokenizer")
            # If we get here, the function ran successfully
            assert True
        except Exception as e:
            pytest.fail(f"print_compression_report raised an exception: {e}")


class TestIntegration:
    """Integration tests for utility functions"""
    
    def test_full_workflow(self):
        """Test complete workflow with utilities"""
        # Create sample data
        data = create_sample_data(n_episodes=5, time_horizon=15, action_dim=7, seed=42)
        
        # Initialize tokenizers
        huffman_tokenizer = FASTLight(n_coeffs=4, scale=5.0)
        rle_tokenizer = FASTLightRLE(n_coeffs=4, scale=5.0)
        
        # Fit tokenizers
        huffman_tokenizer.fit(data)
        rle_tokenizer.fit(data)
        
        # Benchmark both
        tokenizers = {
            "Huffman": huffman_tokenizer,
            "RLE": rle_tokenizer,
        }
        
        benchmark_results = compare_tokenizers(tokenizers, data, n_runs=2)
        
        # Analyze quality
        huffman_quality = analyze_compression_quality(huffman_tokenizer, data)
        rle_quality = analyze_compression_quality(rle_tokenizer, data)
        
        # Test visualization
        chunk = data[0]
        huffman_tokens = huffman_tokenizer.encode(chunk)
        huffman_reconstructed = huffman_tokenizer.decode(huffman_tokens)
        
        # All should work without errors
        assert len(benchmark_results) == 2
        assert "overall_mse" in huffman_quality
        assert "overall_mse" in rle_quality
        assert huffman_reconstructed.shape == chunk.shape
    
    def test_error_handling(self):
        """Test error handling in utility functions"""
        # Test with unfitted tokenizer
        tokenizer = FASTLight()
        data = create_sample_data(n_episodes=3, time_horizon=15, action_dim=7)
        
        # These should raise errors
        with pytest.raises(ValueError):
            benchmark_tokenizer(tokenizer, data)
        
        with pytest.raises(ValueError):
            analyze_compression_quality(tokenizer, data)
        
        with pytest.raises(ValueError):
            print_compression_report(tokenizer, data)
