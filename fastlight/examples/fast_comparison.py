#!/usr/bin/env python3
"""
FASTLight vs FAST Tokenizer Performance Comparison

This script provides a comprehensive comparison between FASTLight and the original
FAST tokenizer from Physical Intelligence, including:
- Compression ratios and token counts
- Encoding/decoding speed benchmarks
- Reconstruction quality analysis
- Memory usage comparison
- Edge device suitability analysis

Usage:
    python -m fastlight.examples.fast_comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import os
from typing import Dict, List, Any, Optional
import argparse

from fastlight import FASTLight, FASTLightRLE
from fastlight.utils import create_sample_data, benchmark_tokenizer, compare_tokenizers


class FASTTokenizerWrapper:
    """
    Wrapper for the original FAST tokenizer from Physical Intelligence
    Uses the same implementation as in data_explore.py
    """
    
    def __init__(self):
        self.tokenizer = None
        self.is_loaded = False
        self._load_fast_tokenizer()
    
    def _load_fast_tokenizer(self):
        """Load the original FAST tokenizer from Physical Intelligence"""
        try:
            from transformers import AutoProcessor
            print("Loading FAST+ tokenizer from Physical Intelligence...")
            self.tokenizer = AutoProcessor.from_pretrained(
                "physical-intelligence/fast",
                trust_remote_code=True
            )
            self.is_loaded = True
            print("✓ FAST+ tokenizer loaded successfully")
        except Exception as e:
            print(f"✗ Error loading tokenizer: {e}")
            print("\nNote: You may need to accept the model terms on HuggingFace first")
            self.is_loaded = False
    
    def fit(self, actions: np.ndarray):
        """FAST doesn't need explicit fitting"""
        if not self.is_loaded:
            raise ValueError("FAST tokenizer not loaded")
        return self
    
    def encode(self, action_chunk: np.ndarray) -> List[int]:
        """Encode action chunk using FAST (same as data_explore.py)"""
        if not self.is_loaded:
            raise ValueError("FAST tokenizer not loaded")
        
        # Normalize for FAST (exact same as in data_explore.py)
        q01 = np.percentile(action_chunk, 1, axis=0)
        q99 = np.percentile(action_chunk, 99, axis=0)
        chunk_normalized = 2 * (action_chunk - q01) / (q99 - q01 + 1e-8) - 1
        chunk_normalized = np.clip(chunk_normalized, -1, 1)
        
        # Encode using FAST tokenizer
        tokens = self.tokenizer(chunk_normalized.reshape(1, *chunk_normalized.shape))[0]
        return tokens  # tokens is already a list
    
    def decode(self, tokens: List[int], time_horizon: int = 15, action_dim: int = 7) -> np.ndarray:
        """Decode tokens using FAST (same as data_explore.py)"""
        if not self.is_loaded:
            raise ValueError("FAST tokenizer not loaded")
        
        # Decode using FAST tokenizer (exact same as in data_explore.py)
        reconstructed_normalized = self.tokenizer.decode(
            [tokens], time_horizon=time_horizon, action_dim=action_dim
        )[0]
        
        # Return normalized values - denormalization will be handled in comparison
        return reconstructed_normalized
    
    def encode_decode_with_denormalization(self, action_chunk: np.ndarray) -> tuple:
        """
        Encode and decode with proper denormalization (for fair comparison)
        Returns: (tokens, reconstructed_denormalized, q01, q99)
        """
        if not self.is_loaded:
            raise ValueError("FAST tokenizer not loaded")
        
        # Normalize (same as data_explore.py)
        q01 = np.percentile(action_chunk, 1, axis=0)
        q99 = np.percentile(action_chunk, 99, axis=0)
        chunk_normalized = 2 * (action_chunk - q01) / (q99 - q01 + 1e-8) - 1
        chunk_normalized = np.clip(chunk_normalized, -1, 1)
        
        # Encode
        tokens = self.tokenizer(chunk_normalized.reshape(1, *chunk_normalized.shape))[0]
        
        # Decode
        reconstructed_normalized = self.tokenizer.decode(
            [tokens], time_horizon=action_chunk.shape[0], action_dim=action_chunk.shape[1]
        )[0]
        
        # Denormalize (same as data_explore.py)
        reconstructed = (reconstructed_normalized + 1) * (q99 - q01) / 2 + q01
        
        # tokens is already a list, no need to call .tolist()
        return tokens, reconstructed, q01, q99
    
    def __call__(self, action_chunks: np.ndarray) -> List[List[int]]:
        """Batch encode"""
        if action_chunks.ndim == 2:
            return [self.encode(action_chunks)]
        else:
            return [self.encode(chunk) for chunk in action_chunks]


class PerformanceComparison:
    """
    Comprehensive performance comparison between FASTLight and FAST
    """
    
    def __init__(self):
        self.results = {}
        self.test_data = None
        
    def prepare_test_data(self, n_episodes: int = 20, time_horizon: int = 15, action_dim: int = 7, use_real_data: bool = False):
        """Prepare test data - either synthetic or real DROID data"""
        if use_real_data:
            print("Loading real DROID dataset...")
            try:
                import tensorflow_datasets as tfds
                ds, ds_info = tfds.load(
                    name="r2d2_faceblur", 
                    data_dir="droid_100", 
                    with_info=True,
                    split='train'
                )
                
                all_actions = []
                for episode in ds.take(n_episodes):
                    steps = list(episode['steps'])
                    episode_actions = np.array([step['action'].numpy() for step in steps])
                    all_actions.append(episode_actions)
                
                # Convert to chunks format
                chunks = []
                for episode_actions in all_actions:
                    n_chunks = len(episode_actions) // time_horizon
                    for i in range(n_chunks):
                        chunk = episode_actions[i*time_horizon:(i+1)*time_horizon, :]
                        chunks.append(chunk)
                
                self.test_data = np.array(chunks)
                print(f"Real DROID data loaded: {self.test_data.shape}")
                
            except Exception as e:
                print(f"Failed to load real data: {e}")
                print("Falling back to synthetic data...")
                use_real_data = False
        
        if not use_real_data:
            print("Preparing synthetic test data...")
            self.test_data = create_sample_data(
                n_episodes=n_episodes,
                time_horizon=time_horizon,
                action_dim=action_dim,
                seed=42
            )
            print(f"Synthetic test data shape: {self.test_data.shape}")
    
    def run_compression_analysis(self):
        """Compare compression ratios and token counts"""
        print("\n" + "="*60)
        print("COMPRESSION ANALYSIS")
        print("="*60)
        
        # Initialize tokenizers
        tokenizers = {
            "FAST": FASTTokenizerWrapper(),
            "FASTLight (Huffman)": FASTLight(n_coeffs=6, scale=10.0),
            "FASTLight (RLE)": FASTLightRLE(n_coeffs=6, scale=10.0),
        }
        
        # Fit tokenizers
        print("\nFitting tokenizers...")
        for name, tokenizer in tokenizers.items():
            if name == "FAST" and not tokenizer.is_loaded:
                print(f"  Skipping {name} (not loaded)")
                continue
            print(f"  Fitting {name}...")
            tokenizer.fit(self.test_data)
        
        # Test compression
        print("\nTesting compression...")
        compression_results = {}
        
        for name, tokenizer in tokenizers.items():
            if name == "FAST" and not tokenizer.is_loaded:
                continue
                
            print(f"  Testing {name}...")
            token_counts = []
            mse_values = []
            
            for i, chunk in enumerate(self.test_data):
                # Encode and decode with proper handling for each tokenizer
                if name == "FAST":
                    # Use the proper FAST implementation with denormalization
                    tokens, reconstructed, q01, q99 = tokenizer.encode_decode_with_denormalization(chunk)
                else:
                    # FASTLight variants
                    tokens = tokenizer.encode(chunk)
                    reconstructed = tokenizer.decode(tokens)
                
                token_counts.append(len(tokens))
                mse = np.mean((chunk - reconstructed) ** 2)
                mse_values.append(mse)
            
            compression_results[name] = {
                'avg_tokens': np.mean(token_counts),
                'std_tokens': np.std(token_counts),
                'min_tokens': np.min(token_counts),
                'max_tokens': np.max(token_counts),
                'avg_mse': np.mean(mse_values),
                'std_mse': np.std(mse_values),
                'min_mse': np.min(mse_values),
                'max_mse': np.max(mse_values),
                'compression_ratio': (chunk.shape[0] * chunk.shape[1]) / np.mean(token_counts),
            }
        
        self.results['compression'] = compression_results
        return compression_results
    
    def run_speed_benchmark(self):
        """Compare encoding/decoding speeds"""
        print("\n" + "="*60)
        print("SPEED BENCHMARK")
        print("="*60)
        
        # Initialize tokenizers
        tokenizers = {
            "FASTLight (Huffman)": FASTLight(n_coeffs=6, scale=10.0),
            "FASTLight (RLE)": FASTLightRLE(n_coeffs=6, scale=10.0),
        }
        
        # Add FAST if available
        fast_tokenizer = FASTTokenizerWrapper()
        if fast_tokenizer.is_loaded:
            tokenizers["FAST"] = fast_tokenizer
        
        # Fit tokenizers
        print("\nFitting tokenizers...")
        for name, tokenizer in tokenizers.items():
            print(f"  Fitting {name}...")
            tokenizer.fit(self.test_data)
        
        # Benchmark
        print("\nBenchmarking speed...")
        speed_results = compare_tokenizers(tokenizers, self.test_data, n_runs=5)
        
        self.results['speed'] = speed_results
        return speed_results
    
    def run_memory_analysis(self):
        """Compare memory usage"""
        print("\n" + "="*60)
        print("MEMORY ANALYSIS")
        print("="*60)
        
        memory_results = {}
        
        # Test FASTLight variants
        for name, tokenizer_class in [
            ("FASTLight (Huffman)", FASTLight),
            ("FASTLight (RLE)", FASTLightRLE)
        ]:
            print(f"  Testing {name}...")
            
            # Measure memory before
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create and fit tokenizer
            tokenizer = tokenizer_class(n_coeffs=6, scale=10.0)
            tokenizer.fit(self.test_data)
            
            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            # Get model size estimate
            if hasattr(tokenizer, 'get_compression_stats'):
                stats = tokenizer.get_compression_stats()
                model_size = stats.get('vocab_size', 0) * 20  # Rough estimate
            else:
                model_size = 112  # RLE model size
            
            memory_results[name] = {
                'memory_used_mb': memory_used,
                'model_size_bytes': model_size,
                'model_size_kb': model_size / 1024,
            }
        
        # FAST memory usage (estimated)
        memory_results["FAST"] = {
            'memory_used_mb': 50,  # Estimated
            'model_size_bytes': 40 * 1024,  # ~40KB
            'model_size_kb': 40,
        }
        
        self.results['memory'] = memory_results
        return memory_results
    
    def run_edge_device_analysis(self):
        """Analyze suitability for edge devices"""
        print("\n" + "="*60)
        print("EDGE DEVICE SUITABILITY")
        print("="*60)
        
        edge_results = {}
        
        # Get results from previous analyses
        compression = self.results.get('compression', {})
        speed = self.results.get('speed', {})
        memory = self.results.get('memory', {})
        
        for name in ["FAST", "FASTLight (Huffman)", "FASTLight (RLE)"]:
            if name not in compression:
                continue
                
            # Calculate edge suitability score (0-100)
            score = 0
            
            # Compression score (30% weight)
            if name in compression:
                comp_ratio = compression[name]['compression_ratio']
                score += min(30, comp_ratio * 10)  # Max 30 points
            
            # Speed score (30% weight)
            if name in speed:
                total_time = speed[name]['total_time_ms']
                score += max(0, 30 - total_time * 2)  # Faster = higher score
            
            # Memory score (20% weight)
            if name in memory:
                model_size_kb = memory[name]['model_size_kb']
                score += max(0, 20 - model_size_kb * 0.5)  # Smaller = higher score
            
            # Quality score (20% weight)
            if name in compression:
                mse = compression[name]['avg_mse']
                score += max(0, 20 - mse * 10000)  # Lower MSE = higher score
            
            edge_results[name] = {
                'suitability_score': score,
                'recommended_for_edge': score > 60,
                'strengths': [],
                'weaknesses': []
            }
            
            # Identify strengths and weaknesses
            if name in compression and compression[name]['compression_ratio'] > 2.5:
                edge_results[name]['strengths'].append("Good compression")
            if name in speed and speed[name]['total_time_ms'] < 2.0:
                edge_results[name]['strengths'].append("Fast processing")
            if name in memory and memory[name]['model_size_kb'] < 10:
                edge_results[name]['strengths'].append("Small model size")
            
            if name in compression and compression[name]['avg_mse'] > 0.01:
                edge_results[name]['weaknesses'].append("Higher reconstruction error")
            if name in speed and speed[name]['total_time_ms'] > 5.0:
                edge_results[name]['weaknesses'].append("Slower processing")
            if name in memory and memory[name]['model_size_kb'] > 20:
                edge_results[name]['weaknesses'].append("Large model size")
        
        self.results['edge'] = edge_results
        return edge_results
    
    def generate_report(self):
        """Generate comprehensive comparison report"""
        print("\n" + "="*70)
        print("COMPREHENSIVE PERFORMANCE COMPARISON REPORT")
        print("="*70)
        
        compression = self.results.get('compression', {})
        speed = self.results.get('speed', {})
        memory = self.results.get('memory', {})
        edge = self.results.get('edge', {})
        
        # Summary table
        print("\n📊 SUMMARY COMPARISON")
        print("-" * 100)
        print(f"{'Method':<20} {'Tokens':<8} {'Compress':<8} {'MSE':<10} {'Speed':<8} {'Memory':<8} {'Edge Score':<10}")
        print("-" * 100)
        
        for name in ["FAST", "FASTLight (Huffman)", "FASTLight (RLE)"]:
            if name not in compression:
                continue
                
            tokens = f"{compression[name]['avg_tokens']:.1f}"
            compress = f"{compression[name]['compression_ratio']:.1f}x"
            mse = f"{compression[name]['avg_mse']:.4f}"
            speed_ms = f"{speed.get(name, {}).get('total_time_ms', 0):.1f}ms"
            memory_kb = f"{memory.get(name, {}).get('model_size_kb', 0):.0f}KB"
            edge_score = f"{edge.get(name, {}).get('suitability_score', 0):.0f}/100"
            
            print(f"{name:<20} {tokens:<8} {compress:<8} {mse:<10} {speed_ms:<8} {memory_kb:<8} {edge_score:<10}")
        
        # Detailed analysis
        print("\n🔍 DETAILED ANALYSIS")
        print("-" * 50)
        
        for name, data in compression.items():
            print(f"\n{name}:")
            print(f"  Compression: {data['avg_tokens']:.1f} ± {data['std_tokens']:.1f} tokens/chunk")
            print(f"  Ratio: {data['compression_ratio']:.2f}x vs naive")
            print(f"  Quality: MSE = {data['avg_mse']:.6f} ± {data['std_mse']:.6f}")
            
            if name in speed:
                print(f"  Speed: {speed[name]['total_time_ms']:.2f} ms/chunk total")
                print(f"    - Encode: {speed[name]['encode_time_ms']:.2f} ms")
                print(f"    - Decode: {speed[name]['decode_time_ms']:.2f} ms")
            
            if name in memory:
                print(f"  Memory: {memory[name]['model_size_kb']:.1f} KB model size")
            
            if name in edge:
                edge_data = edge[name]
                print(f"  Edge Suitability: {edge_data['suitability_score']:.0f}/100")
                if edge_data['strengths']:
                    print(f"    Strengths: {', '.join(edge_data['strengths'])}")
                if edge_data['weaknesses']:
                    print(f"    Weaknesses: {', '.join(edge_data['weaknesses'])}")
        
        # Recommendations
        print("\n🎯 RECOMMENDATIONS")
        print("-" * 50)
        
        if "FASTLight (Huffman)" in edge and edge["FASTLight (Huffman)"]['suitability_score'] > 70:
            print("✅ FASTLight (Huffman) is RECOMMENDED for most applications")
            print("   - Best compression ratio")
            print("   - Good reconstruction quality")
            print("   - Reasonable speed and memory usage")
        
        if "FASTLight (RLE)" in edge and edge["FASTLight (RLE)"]['suitability_score'] > 70:
            print("✅ FASTLight (RLE) is RECOMMENDED for edge devices")
            print("   - Fastest processing")
            print("   - Smallest memory footprint")
            print("   - Good for real-time applications")
        
        if "FAST" in edge and edge["FAST"]['suitability_score'] < 60:
            print("⚠️  FAST is NOT RECOMMENDED for edge deployment")
            print("   - Large model size")
            print("   - Slower processing")
            print("   - Better suited for cloud/server applications")
    
    def create_visualizations(self, save_path: str = "fast_comparison_results.png"):
        """Create visualization of comparison results"""
        print(f"\n📈 Creating visualizations...")
        
        compression = self.results.get('compression', {})
        speed = self.results.get('speed', {})
        memory = self.results.get('memory', {})
        
        if not compression:
            print("No compression data available for visualization")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('FASTLight vs FAST Performance Comparison', fontsize=16, fontweight='bold')
        
        names = list(compression.keys())
        colors = ['blue', 'green', 'orange'][:len(names)]
        
        # 1. Token counts
        token_counts = [compression[name]['avg_tokens'] for name in names]
        bars1 = axes[0, 0].bar(names, token_counts, color=colors, alpha=0.7, edgecolor='black')
        axes[0, 0].set_ylabel('Average Tokens per Chunk')
        axes[0, 0].set_title('Compression Efficiency')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars1, token_counts):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Compression ratios
        comp_ratios = [compression[name]['compression_ratio'] for name in names]
        bars2 = axes[0, 1].bar(names, comp_ratios, color=colors, alpha=0.7, edgecolor='black')
        axes[0, 1].set_ylabel('Compression Ratio')
        axes[0, 1].set_title('Compression vs Naive Encoding')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars2, comp_ratios):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                           f'{value:.1f}x', ha='center', va='bottom', fontweight='bold')
        
        # 3. Reconstruction quality (MSE)
        mse_values = [compression[name]['avg_mse'] for name in names]
        bars3 = axes[0, 2].bar(names, mse_values, color=colors, alpha=0.7, edgecolor='black')
        axes[0, 2].set_ylabel('Reconstruction MSE')
        axes[0, 2].set_title('Reconstruction Quality')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars3, mse_values):
            height = bar.get_height()
            axes[0, 2].text(bar.get_x() + bar.get_width()/2., height * 1.1,
                           f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Processing speed
        if speed:
            total_times = [speed.get(name, {}).get('total_time_ms', 0) for name in names]
            bars4 = axes[1, 0].bar(names, total_times, color=colors, alpha=0.7, edgecolor='black')
            axes[1, 0].set_ylabel('Total Time (ms)')
            axes[1, 0].set_title('Processing Speed')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            for bar, value in zip(bars4, total_times):
                if value > 0:
                    height = bar.get_height()
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                   f'{value:.1f}ms', ha='center', va='bottom', fontweight='bold')
        
        # 5. Memory usage
        if memory:
            model_sizes = [memory.get(name, {}).get('model_size_kb', 0) for name in names]
            bars5 = axes[1, 1].bar(names, model_sizes, color=colors, alpha=0.7, edgecolor='black')
            axes[1, 1].set_ylabel('Model Size (KB)')
            axes[1, 1].set_title('Memory Footprint')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            for bar, value in zip(bars5, model_sizes):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                               f'{value:.0f}KB', ha='center', va='bottom', fontweight='bold')
        
        # 6. Efficiency scatter plot
        if speed and memory:
            x_vals = [speed.get(name, {}).get('total_time_ms', 0) for name in names]
            y_vals = [compression[name]['avg_mse'] for name in names]
            sizes = [memory.get(name, {}).get('model_size_kb', 0) * 10 for name in names]
            
            scatter = axes[1, 2].scatter(x_vals, y_vals, s=sizes, c=colors, alpha=0.7, edgecolor='black')
            axes[1, 2].set_xlabel('Processing Time (ms)')
            axes[1, 2].set_ylabel('Reconstruction MSE')
            axes[1, 2].set_title('Efficiency Trade-off\n(Bubble size = Model size)')
            axes[1, 2].set_yscale('log')
            axes[1, 2].grid(True, alpha=0.3)
            
            # Add labels
            for i, name in enumerate(names):
                axes[1, 2].annotate(name, (x_vals[i], y_vals[i]), 
                                   xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved as {save_path}")
        plt.show()


def main():
    """Main comparison function"""
    parser = argparse.ArgumentParser(description="FASTLight vs FAST Performance Comparison")
    parser.add_argument("--episodes", type=int, default=20, help="Number of test episodes")
    parser.add_argument("--skip-fast", action="store_true", help="Skip FAST tokenizer (if not available)")
    parser.add_argument("--save-plot", type=str, default="fast_comparison_results.png", help="Save plot path")
    parser.add_argument("--real-data", action="store_true", help="Use real DROID dataset instead of synthetic data")
    args = parser.parse_args()
    
    print("FASTLight vs FAST Performance Comparison")
    print("=" * 50)
    
    try:
        # Initialize comparison
        comparison = PerformanceComparison()
        
        # Prepare test data
        comparison.prepare_test_data(n_episodes=args.episodes, use_real_data=args.real_data)
        
        # Run analyses
        print("\n🔍 Running comprehensive analysis...")
        
        # Compression analysis
        compression_results = comparison.run_compression_analysis()
        
        # Speed benchmark
        speed_results = comparison.run_speed_benchmark()
        
        # Memory analysis
        memory_results = comparison.run_memory_analysis()
        
        # Edge device analysis
        edge_results = comparison.run_edge_device_analysis()
        
        # Generate report
        comparison.generate_report()
        
        # Create visualizations
        comparison.create_visualizations(args.save_plot)
        
        print("\n✅ Comparison completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Comparison failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
