#!/usr/bin/env python3
"""
FASTLight Robotics Integration Examples

This script demonstrates how to integrate FASTLight with common robotics frameworks
including ROS2, PyBullet, and custom robotic control systems.

Usage:
    python -m fastlight.examples.robotics_integration
"""

import numpy as np
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from fastlight import FASTLight, FASTLightRLE
from fastlight.utils import create_sample_data


@dataclass
class JointState:
    """Simple joint state representation"""
    position: float
    velocity: float
    effort: float
    name: str


@dataclass
class ActionCommand:
    """Action command for robot control"""
    joint_velocities: List[float]
    gripper_position: float
    timestamp: float


class RobotActionCompressor:
    """
    Example integration of FASTLight with a robotic system
    """
    
    def __init__(self, n_joints: int = 6, use_huffman: bool = True):
        self.n_joints = n_joints
        self.action_dim = n_joints + 1  # joints + gripper
        
        # Initialize tokenizer
        if use_huffman:
            self.tokenizer = FASTLight(n_coeffs=6, scale=10.0)
        else:
            self.tokenizer = FASTLightRLE(n_coeffs=6, scale=10.0)
        
        self.is_trained = False
        self.compression_stats = {}
    
    def train_on_demonstrations(self, demonstrations: List[List[ActionCommand]]):
        """
        Train the tokenizer on demonstration data
        
        Args:
            demonstrations: List of demonstrations, each containing action commands
        """
        print(f"Training on {len(demonstrations)} demonstrations...")
        
        # Convert demonstrations to numpy arrays
        training_data = []
        for demo in demonstrations:
            # Convert to numpy array (timesteps, action_dim)
            demo_array = np.zeros((len(demo), self.action_dim))
            for i, action in enumerate(demo):
                demo_array[i, :self.n_joints] = action.joint_velocities
                demo_array[i, self.n_joints] = action.gripper_position
            
            training_data.append(demo_array)
        
        # Stack into training format (n_episodes, timesteps, action_dim)
        training_array = np.stack(training_data, axis=0)
        
        # Fit tokenizer
        self.tokenizer.fit(training_array)
        self.is_trained = True
        
        # Calculate compression stats
        self._calculate_compression_stats(training_array)
        
        print("Training completed!")
        print(f"  Compression ratio: {self.compression_stats['compression_ratio']:.2f}x")
        print(f"  Avg tokens per chunk: {self.compression_stats['avg_tokens']:.1f}")
        print(f"  Reconstruction MSE: {self.compression_stats['avg_mse']:.6f}")
    
    def compress_action_sequence(self, actions: List[ActionCommand]) -> Dict[str, Any]:
        """
        Compress a sequence of actions
        
        Args:
            actions: List of action commands
            
        Returns:
            Dictionary containing compressed data and metadata
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained first")
        
        # Convert to numpy array
        action_array = np.zeros((len(actions), self.action_dim))
        for i, action in enumerate(actions):
            action_array[i, :self.n_joints] = action.joint_velocities
            action_array[i, self.n_joints] = action.gripper_position
        
        # Compress
        start_time = time.perf_counter()
        tokens = self.tokenizer.encode(action_array)
        encode_time = time.perf_counter() - start_time
        
        # Create compressed data structure
        compressed_data = {
            'tokens': tokens,
            'metadata': {
                'n_joints': self.n_joints,
                'action_dim': self.action_dim,
                'sequence_length': len(actions),
                'encode_time_ms': encode_time * 1000,
                'compression_ratio': action_array.size / len(tokens),
                'timestamp': time.time(),
            }
        }
        
        return compressed_data
    
    def decompress_action_sequence(self, compressed_data: Dict[str, Any]) -> List[ActionCommand]:
        """
        Decompress a sequence of actions
        
        Args:
            compressed_data: Dictionary containing compressed data
            
        Returns:
            List of action commands
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained first")
        
        tokens = compressed_data['tokens']
        metadata = compressed_data['metadata']
        
        # Decompress
        start_time = time.perf_counter()
        action_array = self.tokenizer.decode(tokens, time_horizon=metadata['sequence_length'])
        decode_time = time.perf_counter() - start_time
        
        # Convert back to action commands
        actions = []
        for i in range(metadata['sequence_length']):
            action = ActionCommand(
                joint_velocities=action_array[i, :self.n_joints].tolist(),
                gripper_position=float(action_array[i, self.n_joints]),
                timestamp=time.time()
            )
            actions.append(action)
        
        return actions
    
    def _calculate_compression_stats(self, training_data: np.ndarray):
        """Calculate compression statistics"""
        mse_values = []
        token_counts = []
        
        for episode in training_data:
            tokens = self.tokenizer.encode(episode)
            reconstructed = self.tokenizer.decode(tokens)
            
            mse = np.mean((episode - reconstructed) ** 2)
            mse_values.append(mse)
            token_counts.append(len(tokens))
        
        self.compression_stats = {
            'avg_mse': np.mean(mse_values),
            'avg_tokens': np.mean(token_counts),
            'compression_ratio': training_data.shape[1] * training_data.shape[2] / np.mean(token_counts),
        }


class ROS2ActionPublisher:
    """
    Example ROS2 integration for publishing compressed actions
    """
    
    def __init__(self, compressor: RobotActionCompressor):
        self.compressor = compressor
        self.published_sequences = 0
        self.total_compression_ratio = 0.0
    
    def publish_compressed_actions(self, actions: List[ActionCommand], topic_name: str = "/compressed_actions"):
        """
        Publish compressed actions to ROS2 topic
        
        Args:
            actions: List of action commands
            topic_name: ROS2 topic name
        """
        # Compress actions
        compressed_data = self.compressor.compress_action_sequence(actions)
        
        # In a real ROS2 implementation, you would publish the compressed data
        # For this example, we'll just simulate the publishing
        print(f"Publishing to {topic_name}:")
        print(f"  Original size: {len(actions) * self.compressor.action_dim * 4} bytes")
        print(f"  Compressed size: {len(compressed_data['tokens'])} bytes")
        print(f"  Compression ratio: {compressed_data['metadata']['compression_ratio']:.2f}x")
        print(f"  Encode time: {compressed_data['metadata']['encode_time_ms']:.2f} ms")
        
        # Update statistics
        self.published_sequences += 1
        self.total_compression_ratio += compressed_data['metadata']['compression_ratio']
        
        return compressed_data
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get publishing statistics"""
        return {
            'published_sequences': self.published_sequences,
            'avg_compression_ratio': self.total_compression_ratio / max(1, self.published_sequences),
        }


class PyBulletSimulation:
    """
    Example integration with PyBullet simulation
    """
    
    def __init__(self, compressor: RobotActionCompressor):
        self.compressor = compressor
        self.simulation_data = []
    
    def record_demonstration(self, joint_states: List[List[JointState]], 
                           action_commands: List[List[ActionCommand]]):
        """
        Record a demonstration from PyBullet simulation
        
        Args:
            joint_states: Joint states at each timestep
            action_commands: Action commands sent to robot
        """
        print(f"Recording demonstration with {len(action_commands)} timesteps...")
        
        # Convert to training format
        demo_actions = []
        for timestep_actions in action_commands:
            for action in timestep_actions:
                demo_actions.append(action)
        
        self.simulation_data.append(demo_actions)
    
    def train_from_simulations(self):
        """Train tokenizer from recorded simulations"""
        if not self.simulation_data:
            raise ValueError("No simulation data recorded")
        
        self.compressor.train_on_demonstrations(self.simulation_data)
    
    def replay_compressed_actions(self, compressed_data: Dict[str, Any]):
        """
        Replay compressed actions in simulation
        
        Args:
            compressed_data: Compressed action data
        """
        # Decompress actions
        actions = self.compressor.decompress_action_sequence(compressed_data)
        
        print(f"Replaying {len(actions)} actions in simulation...")
        
        # In a real PyBullet implementation, you would apply these actions to the robot
        for i, action in enumerate(actions):
            print(f"  Timestep {i}: Joint velocities = {action.joint_velocities}, Gripper = {action.gripper_position:.3f}")
            # Apply action to robot in simulation
            time.sleep(0.01)  # Simulate control loop timing


def create_sample_demonstrations(n_demos: int = 5, demo_length: int = 20) -> List[List[ActionCommand]]:
    """Create sample demonstration data"""
    demonstrations = []
    
    for demo_idx in range(n_demos):
        demo_actions = []
        
        for timestep in range(demo_length):
            # Create realistic action commands
            joint_velocities = np.random.normal(0, 0.5, 6).tolist()  # 6 joints
            gripper_position = 0.5 * (1 + np.sin(timestep * 0.1 + demo_idx))
            
            action = ActionCommand(
                joint_velocities=joint_velocities,
                gripper_position=gripper_position,
                timestamp=time.time()
            )
            demo_actions.append(action)
        
        demonstrations.append(demo_actions)
    
    return demonstrations


def demo_robotics_integration():
    """Demonstrate robotics integration examples"""
    print("=" * 70)
    print("FASTLight Robotics Integration Demo")
    print("=" * 70)
    
    # Create sample demonstrations
    print("Creating sample demonstration data...")
    demonstrations = create_sample_demonstrations(n_demos=10, demo_length=25)
    print(f"Created {len(demonstrations)} demonstrations")
    
    # Initialize compressor
    print("\nInitializing action compressor...")
    compressor = RobotActionCompressor(n_joints=6, use_huffman=True)
    
    # Train on demonstrations
    compressor.train_on_demonstrations(demonstrations)
    
    # Test compression/decompression
    print("\nTesting compression/decompression...")
    test_actions = demonstrations[0]  # Use first demonstration
    
    # Compress
    compressed_data = compressor.compress_action_sequence(test_actions)
    print(f"Compressed {len(test_actions)} actions to {len(compressed_data['tokens'])} tokens")
    
    # Decompress
    decompressed_actions = compressor.decompress_action_sequence(compressed_data)
    print(f"Decompressed to {len(decompressed_actions)} actions")
    
    # Verify reconstruction quality
    original_array = np.zeros((len(test_actions), 7))
    reconstructed_array = np.zeros((len(decompressed_actions), 7))
    
    for i, (orig, recon) in enumerate(zip(test_actions, decompressed_actions)):
        original_array[i, :6] = orig.joint_velocities
        original_array[i, 6] = orig.gripper_position
        reconstructed_array[i, :6] = recon.joint_velocities
        reconstructed_array[i, 6] = recon.gripper_position
    
    mse = np.mean((original_array - reconstructed_array) ** 2)
    print(f"Reconstruction MSE: {mse:.6f}")
    
    # ROS2 integration demo
    print("\n" + "-" * 50)
    print("ROS2 Integration Demo")
    print("-" * 50)
    
    ros2_publisher = ROS2ActionPublisher(compressor)
    
    # Publish several action sequences
    for i in range(3):
        actions = demonstrations[i]
        compressed_data = ros2_publisher.publish_compressed_actions(actions, f"/robot_actions_{i}")
    
    stats = ros2_publisher.get_statistics()
    print(f"\nROS2 Publishing Statistics:")
    print(f"  Published sequences: {stats['published_sequences']}")
    print(f"  Average compression ratio: {stats['avg_compression_ratio']:.2f}x")
    
    # PyBullet integration demo
    print("\n" + "-" * 50)
    print("PyBullet Integration Demo")
    print("-" * 50)
    
    pybullet_sim = PyBulletSimulation(compressor)
    
    # Record some demonstrations
    for i in range(3):
        # Simulate joint states and action commands
        joint_states = [[JointState(0, 0, 0, f"joint_{j}") for j in range(6)] for _ in range(25)]
        action_commands = [demonstrations[i]]
        
        pybullet_sim.record_demonstration(joint_states, action_commands)
    
    # Train from simulations
    pybullet_sim.train_from_simulations()
    
    # Replay compressed actions
    compressed_data = compressor.compress_action_sequence(demonstrations[0])
    pybullet_sim.replay_compressed_actions(compressed_data)
    
    print("\n" + "=" * 70)
    print("Robotics Integration Demo Completed!")
    print("=" * 70)
    
    print("\nKey Integration Points:")
    print("• Train tokenizer on demonstration data")
    print("• Compress action sequences for efficient storage/transmission")
    print("• Decompress actions for robot control")
    print("• Integrate with ROS2 for distributed robotics")
    print("• Use with PyBullet for simulation and training")
    print("• Achieve 3x+ compression with minimal quality loss")


def main():
    """Main function"""
    try:
        demo_robotics_integration()
        return 0
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
