import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import PIL.Image as Image

# Load dataset
ds, ds_info = tfds.load(
    name="r2d2_faceblur", 
    data_dir="droid_100", 
    with_info=True,
    split='train'
)

def visualize_episode(episode, episode_idx=0):
    """Visualize a single episode"""
    
    # Extract all steps
    steps = list(episode['steps'])
    n_steps = len(steps)
    
    print(f"Episode {episode_idx}:")
    print(f"  Total steps: {n_steps}")
    print(f"  Language: {steps[0]['language_instruction'].numpy().decode()}")
    print(f"  Action shape: {steps[0]['action'].shape}")
    print(f"  Action dtype: {steps[0]['action'].dtype}")
    
    # Collect actions for visualization
    actions = np.array([step['action'].numpy() for step in steps])
    
    # Plot action trajectories
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Episode {episode_idx}: {steps[0]["language_instruction"].numpy().decode()}')
    
    action_names = ['joint_vel_0', 'joint_vel_1', 'joint_vel_2', 
                    'joint_vel_3', 'joint_vel_4', 'joint_vel_5', 'gripper_pos']
    
    for i in range(7):
        ax = axes[i // 4, i % 4]
        ax.plot(actions[:, i])
        ax.set_title(action_names[i])
        ax.set_xlabel('Timestep')
        ax.grid(True, alpha=0.3)
    
    axes[1, 3].axis('off')  # Hide the 8th subplot
    
    plt.tight_layout()
    plt.savefig(f'episode_{episode_idx}_actions.png', dpi=150)
    plt.show()
    
    # Visualize images from first, middle, and last frames
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('Episode Frames: [First, Middle, Last]')
    
    for idx, step_idx in enumerate([0, n_steps//2, n_steps-1]):
        step = steps[step_idx]
        
        # Exterior camera 1
        axes[idx, 0].imshow(step['observation']['exterior_image_1_left'].numpy())
        axes[idx, 0].set_title(f'Exterior 1 (t={step_idx})')
        axes[idx, 0].axis('off')
        
        # Exterior camera 2
        axes[idx, 1].imshow(step['observation']['exterior_image_2_left'].numpy())
        axes[idx, 1].set_title(f'Exterior 2 (t={step_idx})')
        axes[idx, 1].axis('off')
        
        # Wrist camera
        axes[idx, 2].imshow(step['observation']['wrist_image_left'].numpy())
        axes[idx, 2].set_title(f'Wrist (t={step_idx})')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'episode_{episode_idx}_images.png', dpi=150)
    plt.show()
    
    return actions

# Visualize first few episodes
for i, episode in enumerate(ds.take(3)):
    actions = visualize_episode(episode, episode_idx=i)
    print("-" * 80)