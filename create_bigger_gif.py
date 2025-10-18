#!/usr/bin/env python3
"""
Create a bigger GIF from DROID dataset
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os

def create_bigger_gif():
    """Create a bigger GIF from actual DROID dataset"""
    
    # Load the tfrecord files
    tfrecord_files = [
        f"/home/pushpak/droid_100/droid_100/r2d2_faceblur/1.0.0/r2d2_faceblur-train.tfrecord-{i:05d}-of-00031"
        for i in range(3)  # Use first 3 files
    ]
    
    # Filter existing files
    existing_files = [f for f in tfrecord_files if os.path.exists(f)]
    if not existing_files:
        print("No tfrecord files found")
        return None
    
    print(f"Found {len(existing_files)} tfrecord files")
    
    # Create dataset from tfrecord files
    raw_dataset = tf.data.TFRecordDataset(existing_files)
    
    # Parse the tfrecord data
    def parse_tfrecord(example):
        feature_description = {
            'steps/observation/exterior_image_1_left': tf.io.RaggedFeature(tf.string),
            'steps/observation/exterior_image_2_left': tf.io.RaggedFeature(tf.string),
            'steps/observation/wrist_image_left': tf.io.RaggedFeature(tf.string),
        }
        return tf.io.parse_single_example(example, feature_description)
    
    dataset = raw_dataset.map(parse_tfrecord)
    
    images = []
    episode_count = 0
    
    for episode in dataset.take(1):  # Take first episode
        episode_count += 1
        print(f"Processing episode {episode_count}...")
        
        # Get the image data
        img1_data = episode['steps/observation/exterior_image_1_left'].numpy()
        img2_data = episode['steps/observation/exterior_image_2_left'].numpy()
        img3_data = episode['steps/observation/wrist_image_left'].numpy()
        
        # Find the minimum length to avoid index errors
        min_length = min(len(img1_data), len(img2_data), len(img3_data))
        
        print(f"Episode has {min_length} steps")
        
        for i in range(min(min_length, 30)):  # Limit to 40 steps
            try:
                # Get the images for this step
                img1_bytes = img1_data[i]
                img2_bytes = img2_data[i]
                img3_bytes = img3_data[i]
                
                # Decode the images
                img1 = tf.image.decode_jpeg(img1_bytes).numpy()
                img2 = tf.image.decode_jpeg(img2_bytes).numpy()
                img3 = tf.image.decode_jpeg(img3_bytes).numpy()
                
                # Resize to bigger size
                target_size = (256, 256)  # Increased from 64x64 to 128x128
                img1 = tf.image.resize(img1, target_size).numpy().astype(np.uint8)
                img2 = tf.image.resize(img2, target_size).numpy().astype(np.uint8)
                img3 = tf.image.resize(img3, target_size).numpy().astype(np.uint8)
                
                # Concatenate horizontally
                combined = np.concatenate((img1, img2, img3), axis=1)
                
                # Convert to PIL Image
                images.append(Image.fromarray(combined))
                
            except Exception as e:
                print(f"Error processing step {i}: {e}")
                continue
        
        # Only process first episode
        break
    
    print(f"Created {len(images)} images")
    
    if images:
        # Save as GIF
        output_path = "episode_visualization.gif"
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=int(1000/15),  # 15Hz = ~67ms per frame
            loop=0  # Infinite loop
        )
        print(f"Bigger GIF saved to {output_path}")
        return output_path
    else:
        print("No images created")
        return None

if __name__ == "__main__":
    gif_path = create_bigger_gif()
    if gif_path:
        print(f"Successfully created bigger GIF: {gif_path}")
    else:
        print("Failed to create GIF")