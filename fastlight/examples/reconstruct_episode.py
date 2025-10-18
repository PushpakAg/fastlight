
#!/usr/bin/env python3
"""
Create action reconstruction comparison for episode 1
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from fastlight import FASTLight, FASTLightRLE

class FASTTokenizerWrapper:
    """Wrapper for the original FAST tokenizer from Physical Intelligence"""
    
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
            self.is_loaded = False
    
    def fit(self, actions: np.ndarray):
        """FAST doesn't need explicit fitting"""
        if not self.is_loaded:
            raise ValueError("FAST tokenizer not loaded")
        return self
    
    def encode_decode_with_denormalization(self, action_chunk: np.ndarray) -> tuple:
        """Encode and decode with proper denormalization"""
        if not self.is_loaded:
            raise ValueError("FAST tokenizer not loaded")
        
        # Normalize for FAST
        q01 = np.percentile(action_chunk, 1, axis=0)
        q99 = np.percentile(action_chunk, 99, axis=0)
        chunk_normalized = 2 * (action_chunk - q01) / (q99 - q01 + 1e-8) - 1
        chunk_normalized = np.clip(chunk_normalized, -1, 1)
        
        # Encode using FAST tokenizer
        tokens = self.tokenizer(chunk_normalized.reshape(1, *chunk_normalized.shape))[0]
        
        # Decode using FAST tokenizer
        reconstructed_normalized = self.tokenizer.decode(
            [tokens], time_horizon=action_chunk.shape[0], action_dim=action_chunk.shape[1]
        )[0]
        
        # Denormalize
        reconstructed = (reconstructed_normalized + 1) * (q99 - q01) / 2 + q01
        
        return tokens, reconstructed, q01, q99

def create_episode1_reconstruction():
    """Create action reconstruction comparison for episode 1"""
    
    # Load real data
    print("Loading DROID dataset...")
    ds, ds_info = tfds.load(
        name="r2d2_faceblur", 
        data_dir="droid_100", 
        with_info=True,
        split='train'
    )
    
    # Get episode 1 (second episode)
    episodes = list(ds.take(2))
    episode = episodes[1]  # Episode 1
    steps = list(episode['steps'])
    episode_actions = np.array([step['action'].numpy() for step in steps])
    
    print(f"Episode 1 has {len(episode_actions)} steps")
    
    # Take first chunk of episode 1
    chunk = episode_actions[:15, :]
    
    # Initialize tokenizers
    fast_tokenizer = FASTTokenizerWrapper()
    huffman_tokenizer = FASTLight(n_coeffs=6, scale=10.0)
    rle_tokenizer = FASTLightRLE(n_coeffs=6, scale=10.0)
    
    # Fit tokenizers on the chunk
    if fast_tokenizer.is_loaded:
        fast_tokenizer.fit(chunk.reshape(1, 15, 7))
    huffman_tokenizer.fit(chunk.reshape(1, 15, 7))
    rle_tokenizer.fit(chunk.reshape(1, 15, 7))
    
    # Get reconstructions
    results = {}
    
    # FAST reconstruction
    if fast_tokenizer.is_loaded:
        try:
            fast_tokens, fast_recon, _, _ = fast_tokenizer.encode_decode_with_denormalization(chunk)
            results['FAST'] = {
                'tokens': fast_tokens,
                'reconstruction': fast_recon,
                'mse': np.mean((chunk - fast_recon) ** 2),
                'color': 'purple',
                'style': '-.',
                'label': 'FAST (Original)'
            }
            print(f"FAST: {len(fast_tokens)} tokens, MSE = {results['FAST']['mse']:.6f}")
        except Exception as e:
            print(f"FAST error: {e}")
            results['FAST'] = None
    else:
        results['FAST'] = None
    
    # Huffman reconstruction
    try:
        huffman_tokens = huffman_tokenizer.encode(chunk)
        huffman_recon = huffman_tokenizer.decode(huffman_tokens)
        results['Huffman'] = {
            'tokens': huffman_tokens,
            'reconstruction': huffman_recon,
            'mse': np.mean((chunk - huffman_recon) ** 2),
            'color': 'red',
            'style': '--',
            'label': 'FASTLight (Huffman)'
        }
        print(f"Huffman: {len(huffman_tokens)} tokens, MSE = {results['Huffman']['mse']:.6f}")
    except Exception as e:
        print(f"Huffman error: {e}")
        results['Huffman'] = None
    
    # RLE reconstruction
    try:
        rle_tokens = rle_tokenizer.encode(chunk)
        rle_recon = rle_tokenizer.decode(rle_tokens)
        results['RLE'] = {
            'tokens': rle_tokens,
            'reconstruction': rle_recon,
            'mse': np.mean((chunk - rle_recon) ** 2),
            'color': 'green',
            'style': ':',
            'label': 'FASTLight (RLE)'
        }
        print(f"RLE: {len(rle_tokens)} tokens, MSE = {results['RLE']['mse']:.6f}")
    except Exception as e:
        print(f"RLE error: {e}")
        results['RLE'] = None
    
    # Create plot
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    fig.suptitle('Action Reconstruction Comparison - Episode 1', fontsize=16, fontweight='bold')
    
    action_names = ['joint_vel_0', 'joint_vel_1', 'joint_vel_2', 
                    'joint_vel_3', 'joint_vel_4', 'joint_vel_5', 'gripper_pos']
    
    for i in range(7):
        ax = axes[i // 4, i % 4]
        
        # Plot original
        ax.plot(chunk[:, i], 'b-', label='Original', linewidth=3, alpha=0.8)
        
        # Plot reconstructions
        for method, data in results.items():
            if data is not None:
                ax.plot(data['reconstruction'][:, i], 
                       color=data['color'], 
                       linestyle=data['style'], 
                       label=data['label'], 
                       linewidth=2, alpha=0.8)
        
        ax.set_title(f'{action_names[i]}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Action Value')
    
    # Add performance metrics
    metrics_text = "Performance Metrics:\n\n"
    for method, data in results.items():
        if data is not None:
            metrics_text += f"{data['label']}:\n"
            metrics_text += f"  Tokens: {len(data['tokens'])}\n"
            metrics_text += f"  MSE: {data['mse']:.6f}\n\n"
    
    axes[1, 3].text(0.05, 0.95, metrics_text, transform=axes[1, 3].transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('episode1_action_reconstruction.png', dpi=150, bbox_inches='tight')
    print("Episode 1 action reconstruction comparison saved as episode1_action_reconstruction.png")
    plt.show()
    
    return results

if __name__ == "__main__":
    create_episode1_reconstruction()
# %%
