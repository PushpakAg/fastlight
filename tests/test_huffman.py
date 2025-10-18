"""
Tests for Huffman encoding functionality
"""

import pytest
import numpy as np
from collections import Counter

from fastlight.huffman import HuffmanNode, HuffmanEncoder


class TestHuffmanNode:
    """Test HuffmanNode class"""
    
    def test_leaf_node_creation(self):
        """Test creating a leaf node"""
        node = HuffmanNode(42, 10)
        assert node.value == 42
        assert node.freq == 10
        assert node.left is None
        assert node.right is None
        assert node.is_leaf()
    
    def test_internal_node_creation(self):
        """Test creating an internal node"""
        left = HuffmanNode(1, 5)
        right = HuffmanNode(2, 3)
        parent = HuffmanNode(None, 8)
        parent.left = left
        parent.right = right
        
        assert parent.value is None
        assert parent.freq == 8
        assert parent.left == left
        assert parent.right == right
        assert not parent.is_leaf()
    
    def test_comparison(self):
        """Test node comparison for heapq"""
        node1 = HuffmanNode(1, 5)
        node2 = HuffmanNode(2, 3)
        node3 = HuffmanNode(3, 7)
        
        assert node2 < node1
        assert node1 < node3
        assert node2 < node3


class TestHuffmanEncoder:
    """Test HuffmanEncoder class"""
    
    def test_build_tree_simple(self):
        """Test building Huffman tree with simple frequencies"""
        encoder = HuffmanEncoder()
        frequencies = {1: 5, 2: 3, 3: 2}
        
        encoder.build_tree(frequencies)
        
        assert encoder.is_built
        assert encoder.tree is not None
        assert len(encoder.encode_table) == 3
        assert 1 in encoder.encode_table
        assert 2 in encoder.encode_table
        assert 3 in encoder.encode_table
    
    def test_build_tree_single_symbol(self):
        """Test building tree with single symbol"""
        encoder = HuffmanEncoder()
        frequencies = {42: 10}
        
        encoder.build_tree(frequencies)
        
        assert encoder.is_built
        assert encoder.encode_table[42] == "0"
    
    def test_build_tree_empty(self):
        """Test building tree with empty frequencies"""
        encoder = HuffmanEncoder()
        
        with pytest.raises(ValueError, match="Cannot build Huffman tree with empty frequencies"):
            encoder.build_tree({})
    
    def test_encode_decode_roundtrip(self):
        """Test encoding and decoding roundtrip"""
        encoder = HuffmanEncoder()
        frequencies = {0: 10, 1: 5, 2: 3, 3: 1}
        encoder.build_tree(frequencies)
        
        symbols = [0, 1, 2, 3, 0, 1, 0]
        bitstring = encoder.encode_symbols(symbols)
        decoded = encoder.decode_bitstring(bitstring)
        
        assert decoded == symbols
    
    def test_encode_decode_with_unknown_symbol(self):
        """Test encoding/decoding with unknown symbols (escape codes)"""
        encoder = HuffmanEncoder()
        frequencies = {0: 10, 1: 5}
        encoder.build_tree(frequencies)
        
        symbols = [0, 1, 999]  # 999 is unknown
        bitstring = encoder.encode_symbols(symbols)
        decoded = encoder.decode_bitstring(bitstring)
        
        assert decoded == symbols
    
    def test_average_code_length(self):
        """Test average code length calculation"""
        encoder = HuffmanEncoder()
        frequencies = {0: 10, 1: 5, 2: 3, 3: 1}
        encoder.build_tree(frequencies)
        
        avg_length = encoder.get_average_code_length(frequencies)
        assert avg_length > 0
        assert avg_length < 2.0  # Should be less than 2 bits on average
    
    def test_compression_ratio(self):
        """Test compression ratio calculation"""
        encoder = HuffmanEncoder()
        frequencies = {0: 10, 1: 5, 2: 3, 3: 1}
        encoder.build_tree(frequencies)
        
        ratio = encoder.get_compression_ratio(frequencies, original_bits=8)
        assert ratio > 1.0  # Should achieve compression
        assert ratio < 8.0  # But not more than 8x
    
    def test_optimal_encoding(self):
        """Test that more frequent symbols get shorter codes"""
        encoder = HuffmanEncoder()
        frequencies = {0: 100, 1: 50, 2: 25, 3: 10}
        encoder.build_tree(frequencies)
        
        # Most frequent symbol should have shortest code
        code_0 = encoder.encode_table[0]
        code_3 = encoder.encode_table[3]
        
        assert len(code_0) <= len(code_3)
    
    def test_bitstring_packing(self):
        """Test that bitstrings are properly padded"""
        encoder = HuffmanEncoder()
        frequencies = {0: 1, 1: 1}
        encoder.build_tree(frequencies)
        
        # Create a bitstring that's not multiple of 8
        symbols = [0, 1]  # Should create short bitstring
        bitstring = encoder.encode_symbols(symbols)
        
        # Should be able to decode without issues
        decoded = encoder.decode_bitstring(bitstring)
        assert decoded == symbols


class TestHuffmanIntegration:
    """Integration tests for Huffman encoding"""
    
    def test_realistic_frequency_distribution(self):
        """Test with realistic frequency distribution"""
        encoder = HuffmanEncoder()
        
        # Simulate DCT coefficient frequencies (many zeros, few large values)
        frequencies = {0: 1000}  # Many zeros
        for i in range(1, 21):
            frequencies[i] = max(1, 50 - i * 2)  # Decreasing frequency
        for i in range(-20, 0):
            frequencies[i] = max(1, 50 + i * 2)  # Negative values
        
        encoder.build_tree(frequencies)
        
        # Test encoding/decoding
        symbols = [0] * 100 + [1, 2, 3, -1, -2] + [0] * 50
        bitstring = encoder.encode_symbols(symbols)
        decoded = encoder.decode_bitstring(bitstring)
        
        assert decoded == symbols
        
        # Check compression
        ratio = encoder.get_compression_ratio(frequencies, original_bits=16)
        assert ratio > 2.0  # Should achieve good compression
    
    def test_large_dataset(self):
        """Test with larger dataset"""
        encoder = HuffmanEncoder()
        
        # Generate large frequency distribution
        np.random.seed(42)
        symbols = np.random.choice(range(-50, 51), size=10000, p=None)
        frequencies = Counter(symbols)
        
        encoder.build_tree(frequencies)
        
        # Test encoding/decoding
        test_symbols = symbols[:1000].tolist()
        bitstring = encoder.encode_symbols(test_symbols)
        decoded = encoder.decode_bitstring(bitstring)
        
        assert decoded == test_symbols
