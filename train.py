import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mnist_model import LightMNIST, train_model, test_model, count_parameters

def main():
    # Force CPU usage
    device = torch.device("cpu")
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Reduce batch size for CPU training
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=500)
    
    # Initialize model
    model = LightMNIST().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Print model parameters
    n_params = count_parameters(model)
    print(f"Number of parameters: {n_params}")
    
    # Train for one epoch
    train_acc = train_model(model, device, train_loader, optimizer, epoch=1)
    test_acc = test_model(model, device, test_loader)
    
    print(f"Training accuracy: {train_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    
    # Save model
    torch.save(model.state_dict(), "mnist_model.pth")
    
    return train_acc, n_params

if __name__ == "__main__":
    main() 