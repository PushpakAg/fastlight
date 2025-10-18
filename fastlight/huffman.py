"""
Huffman encoding implementation for FASTLight

This module provides the core Huffman encoding/decoding functionality
used by FASTLight for optimal compression of DCT coefficients.
"""

import heapq
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union


class HuffmanNode:
    """
    Node in a Huffman tree for encoding/decoding
    
    Attributes:
        value: The symbol value (None for internal nodes)
        freq: Frequency of the symbol
        left: Left child node
        right: Right child node
    """
    
    def __init__(self, value: Optional[int], freq: int):
        self.value = value
        self.freq = freq
        self.left: Optional['HuffmanNode'] = None
        self.right: Optional['HuffmanNode'] = None
    
    def __lt__(self, other: 'HuffmanNode') -> bool:
        """Enable comparison for heapq"""
        return self.freq < other.freq
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return self.value is not None


class HuffmanEncoder:
    """
    Huffman encoder/decoder for variable-length coding
    
    This class builds a Huffman tree from symbol frequencies and provides
    methods to encode symbols to bitstrings and decode bitstrings back to symbols.
    """
    
    def __init__(self):
        self.tree: Optional[HuffmanNode] = None
        self.encode_table: Dict[int, str] = {}
        self.decode_tree: Optional[HuffmanNode] = None
        self.is_built = False
    
    def build_tree(self, frequencies: Dict[int, int]) -> None:
        """
        Build Huffman tree from symbol frequencies
        
        Args:
            frequencies: Dictionary mapping symbols to their frequencies
        """
        if not frequencies:
            raise ValueError("Cannot build Huffman tree with empty frequencies")
        
        # Create leaf nodes
        heap = [HuffmanNode(symbol, freq) for symbol, freq in frequencies.items()]
        heapq.heapify(heap)
        
        # Build tree bottom-up
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            # Create internal node
            parent = HuffmanNode(None, left.freq + right.freq)
            parent.left = left
            parent.right = right
            
            heapq.heappush(heap, parent)
        
        # Store root
        self.tree = heap[0]
        self.decode_tree = self.tree
        
        # Build encoding table
        self.encode_table = {}
        self._build_encode_table(self.tree, "")
        
        self.is_built = True
    
    def _build_encode_table(self, node: HuffmanNode, code: str) -> None:
        """
        Recursively build encoding table from Huffman tree
        
        Args:
            node: Current node in the tree
            code: Current bitstring code
        """
        if node is None:
            return
        
        if node.is_leaf():
            # Store encoding for leaf node
            self.encode_table[node.value] = code if code else "0"
            return
        
        # Recurse to children
        self._build_encode_table(node.left, code + "0")
        self._build_encode_table(node.right, code + "1")
    
    def encode_symbols(self, symbols: List[int]) -> str:
        """
        Encode a list of symbols to a bitstring
        
        Args:
            symbols: List of symbols to encode
            
        Returns:
            Bitstring representation
        """
        if not self.is_built:
            raise ValueError("Must build Huffman tree before encoding")
        
        bitstring = ""
        for symbol in symbols:
            if symbol in self.encode_table:
                bitstring += self.encode_table[symbol]
            else:
                # Handle unknown symbols with escape code
                bitstring += "11111111"  # Escape code
                # Encode as 16-bit signed integer
                val_bits = format(symbol & 0xFFFF, '016b')
                bitstring += val_bits
        
        return bitstring
    
    def decode_bitstring(self, bitstring: str) -> List[int]:
        """
        Decode a bitstring back to symbols
        
        Args:
            bitstring: Bitstring to decode
            
        Returns:
            List of decoded symbols
        """
        if not self.is_built:
            raise ValueError("Must build Huffman tree before decoding")
        
        symbols = []
        i = 0
        
        while i < len(bitstring):
            # Check for escape code
            if bitstring[i:i+8] == "11111111":
                # Next 16 bits are raw value
                i += 8
                if i + 16 <= len(bitstring):
                    val_bits = bitstring[i:i+16]
                    val = int(val_bits, 2)
                    # Convert from unsigned to signed
                    if val >= 32768:
                        val -= 65536
                    symbols.append(val)
                    i += 16
                else:
                    break
            else:
                # Traverse Huffman tree
                node = self.decode_tree
                while not node.is_leaf() and i < len(bitstring):
                    if bitstring[i] == '0':
                        node = node.left
                    else:
                        node = node.right
                    i += 1
                
                if node.is_leaf():
                    symbols.append(node.value)
        
        return symbols
    
    def get_average_code_length(self, frequencies: Dict[int, int]) -> float:
        """
        Calculate average code length in bits
        
        Args:
            frequencies: Symbol frequencies
            
        Returns:
            Average code length in bits
        """
        if not self.is_built:
            raise ValueError("Must build Huffman tree first")
        
        total_freq = sum(frequencies.values())
        avg_len = sum(
            len(self.encode_table[symbol]) * freq 
            for symbol, freq in frequencies.items()
            if symbol in self.encode_table
        ) / total_freq
        
        return avg_len
    
    def get_compression_ratio(self, frequencies: Dict[int, int], original_bits: int = 16) -> float:
        """
        Calculate compression ratio compared to fixed-width encoding
        
        Args:
            frequencies: Symbol frequencies
            original_bits: Bits per symbol in original encoding
            
        Returns:
            Compression ratio (original_bits / average_code_length)
        """
        avg_len = self.get_average_code_length(frequencies)
        return original_bits / avg_len if avg_len > 0 else 1.0
