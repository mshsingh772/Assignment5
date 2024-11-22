import torch
from mnist_model import LightMNIST, train_model, count_parameters
from train import main

def test_model_requirements():
    # Run training
    train_acc, n_params = main()
    
    # Test parameter count
    assert n_params < 25000, f"Model has {n_params} parameters, should be less than 25000"
    
    # Test accuracy
    assert train_acc >= 95.0, f"Training accuracy is {train_acc:.2f}%, should be at least 95%"
    
    print("All requirements met!")
    print(f"Parameters: {n_params}")
    print(f"Training accuracy: {train_acc:.2f}%")

if __name__ == "__main__":
    test_model_requirements() 