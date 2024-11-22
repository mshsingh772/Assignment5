import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mnist_model import LightMNIST, train_model, test_model, count_parameters

def main():
    device = torch.device("cpu")
    
    # Simple transform without augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=500)
    
    model = LightMNIST().to(device)
    
    # Simple SGD with momentum
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        nesterov=True
    )
    
    n_params = count_parameters(model)
    print(f"Number of parameters: {n_params}")
    
    train_acc = train_model(model, device, train_loader, optimizer, None, epoch=1)
    test_acc = test_model(model, device, test_loader)
    
    print(f"Training accuracy: {train_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    
    torch.save(model.state_dict(), "mnist_model.pth")
    
    return train_acc, n_params

if __name__ == "__main__":
    main() 