import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

def get_sample_image():
    # Get a single image from MNIST
    dataset = datasets.MNIST('./data', train=True, download=True, 
                           transform=transforms.ToTensor())
    return dataset[0][0]  # Get first image

def apply_augmentations(image):
    # Define our augmentation transform
    aug_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(
            degrees=20,
            translate=(0.02, 0.1),
            scale=(0.98, 1.02),
        ),
        transforms.ToTensor()
    ])
    
    # Generate multiple augmented versions
    augmented_images = [aug_transform(image) for _ in range(8)]
    return augmented_images

def plot_augmentations(original, augmented):
    plt.figure(figsize=(10, 10))
    
    # Plot original image in center
    plt.subplot(3, 3, 5)
    plt.imshow(original.squeeze(), cmap='gray')
    plt.title('Original', pad=10)
    plt.axis('off')
    
    # Define augmentation descriptions
    aug_names = [
        'Rot+Trans',      # Rotation + Translation
        'Scale+Rot',      # Scaling + Rotation
        'Trans+Scale',    # Translation + Scaling
        'Rot',            # Rotation only
        'Trans',          # Translation only
        'Scale',          # Scaling only
        'Rot+Scale',      # Rotation + Scaling
        'Trans+Rot'       # Translation + Rotation
    ]
    
    # Plot augmented images around it
    positions = [1, 2, 3, 4, 6, 7, 8, 9]  # Skip position 5 (center)
    for idx, pos in enumerate(positions):
        plt.subplot(3, 3, pos)
        plt.imshow(augmented[idx].squeeze(), cmap='gray')
        plt.title(aug_names[idx], pad=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_examples.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Get original image
    original_image = get_sample_image()
    
    # Generate augmented versions
    augmented_images = apply_augmentations(original_image)
    
    # Plot and save
    plot_augmentations(original_image, augmented_images)

if __name__ == "__main__":
    main() 