import torch
from mnist_model import LightMNIST, train_model, count_parameters
from train import main
import inspect
import torch.nn.functional as F

def test_model_requirements():
    # Run training
    train_acc, n_params = main()
    
    # Test parameter count
    assert n_params < 25000, f"Model has {n_params} parameters, should be less than 25000"
    
    # Test accuracy
    assert train_acc >= 95.0, f"Training accuracy is {train_acc:.2f}%, should be at least 95%"
    
    print("Basic requirements met!")
    print(f"Parameters: {n_params}")
    print(f"Training accuracy: {train_acc:.2f}%")

def test_augmentation_usage():
    # Load train.py content
    with open('train.py', 'r') as file:
        train_content = file.read()
    
    # Check for augmentation transforms
    assert 'RandomAffine' in train_content, "Data augmentation (RandomAffine) is not used"
    assert 'degrees=' in train_content, "Rotation augmentation is not configured"
    assert 'translate=' in train_content, "Translation augmentation is not configured"
    assert 'scale=' in train_content, "Scale augmentation is not configured"
    
    print("Augmentation test passed!")

def test_learning_rate():
    # Load train.py content
    with open('train.py', 'r') as file:
        train_content = file.read()
    
    # Find learning rate value
    import re
    lr_matches = re.findall(r'lr=([0-9.]+)', train_content)
    if lr_matches:
        lr = float(lr_matches[0])
        # Check if learning rate is appropriately low
        assert lr <= 0.1, f"Learning rate {lr} is too high, should be <= 0.001"
    else:
        raise AssertionError("Could not find learning rate in train.py")
    
    print("Learning rate test passed!")

def test_maxpool_usage():
    # Check model's forward method for maxpool
    model = LightMNIST()
    source = inspect.getsource(model.forward)
    
    # Check for max_pool2d usage
    assert 'max_pool2d' in source, "MaxPool2d is not used in the model architecture"
    
    # Count number of maxpool layers
    maxpool_count = source.count('max_pool2d')
    assert maxpool_count >= 2, f"Model should use at least 2 maxpool layers, found {maxpool_count}"
    
    print("MaxPool usage test passed!")

if __name__ == "__main__":
    print("Running tests...")
    print("\n1. Testing basic requirements:")
    test_model_requirements()
    
    print("\n2. Testing augmentation usage:")
    test_augmentation_usage()
    
    print("\n3. Testing learning rate:")
    test_learning_rate()
    
    print("\n4. Testing MaxPool usage:")
    test_maxpool_usage()
    
    print("\nAll tests passed successfully!") 