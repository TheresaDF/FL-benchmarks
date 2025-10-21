from data.utils.datasets import * 
from src.utils.models import * 
import torch.optim as optim 
from tqdm import tqdm 
import argparse
import platform 
import os 

# set device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FLBENCH_ROOT = os.getcwd()


def get_dataset(dataset_name, train_transform=None, test_transform=None):
    """
    Returns train_loader and test_loader for the full dataset.
    """
    root = FLBENCH_ROOT + "/data/" + dataset_name
    with open(root + "/args.json", "r") as f:
        args = json.load(f)

    # Load full dataset
    dataset = DATASETS[dataset_name](root, args)

    # Switch dataset transforms to centralized train/test transforms if provided
    dataset.train_data_transform = train_transform
    dataset.test_data_transform = test_transform
    dataset.train_target_transform = None
    dataset.test_target_transform = None

    # Split into train/test using the standard CIFAR100/MNIST split if not already
    # Here we assume the dataset has all data concatenated
    N = len(dataset)
    split_ratio = 0.8
    train_size = int(N * split_ratio)
    test_size = N - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

def evaluate(model, loader):
    """
    Evaluate model on a dataset.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    return correct / total

def train(args):
    # init model 
    model = MODELS[args.model_name](dataset = args.dataset, pretrained = False)
    model.to(device)
    model.train()

    # get data 
    train_loader, test_loader = get_dataset(args.dataset) 

    # get opt and cri 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    # init results 
    train_accuracies = [None] * args.epochs 
    test_accuracies = [None] * args.epochs  

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            _, predicted = torch.max(outputs, 1)  # argmax over classes
            correct += (predicted == y).sum().item()
            total += y.size(0)

        train_acc = correct / total

        # Evaluate on test set
        test_acc = evaluate(model, test_loader)
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Test Acc {test_acc:.4f}")

        # store results 
        train_accuracies[epoch] = train_acc 
        test_accuracies[epoch] = test_acc 

    # save results
    if not os.path.exists("out/central"): 
        os.makedirs("out/central")
    np.save(f"out/central/{args.save_name}.npy", {"train": train_accuracies, "test": test_accuracies})



if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 375)
    parser.add_argument("--dataset", type = str, default = "cifar100")
    parser.add_argument("--eval_frequency", type = int, default = 5)
    parser.add_argument("--model_name", type = str, default = "avgcnn")
    parser.add_argument("--learning_rate", type = float, default = 0.0001)
    parser.add_argument("--momentum", type = float, default = 0.9)
    parser.add_argument("--save_name", type = str, default = "results")

    args = parser.parse_args()
    train(args)




