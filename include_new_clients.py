from numpy import isin
from numpy.testing import rundocs
from torch.utils.data import Subset, DataLoader
from data.utils.datasets import * 
from types import SimpleNamespace
from src.utils.models import * 
from copy import deepcopy
from pathlib import Path 
from tqdm import tqdm 
from utils import * 
import platform 
import argparse
import pickle 
import torch 
import json 
import yaml
import glob 
import os 


FLBENCH_ROOT = os.getcwd()


# set device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(run_dir, config_file, dataset): 
    # read model and config file 
    model_path = glob.glob(run_dir + "/*.pt")[0]
    network = config_file['model']['name']

    model = MODELS[network](dataset = dataset, pretrained = False)
    model.load_state_dict(torch.load(model_path))

    return model, network 

def load_config(run_dir): 
    config_path = run_dir + "/.hydra/config.yaml"

    with open(config_path, "r") as f: 
        cfg_dict = yaml.safe_load(f)
    f.close()
    return cfg_dict 

def load_data_partition(dataset_name : str) -> list[dict]: 
    try:
        partition_path = (
            FLBENCH_ROOT / "data" / dataset_name / "partition.pkl"
        )
        with open(partition_path, "rb") as f:
            data_partition = pickle.load(f)
    except:
        raise FileNotFoundError(f"Please partition {dataset_name} first.")
    
    return data_partition['data_indices'], len(data_partition['data_indices'])

def get_dataset(dataset_name) -> BaseDataset:
    root = FLBENCH_ROOT / "data" / dataset_name
    with open(root / "args.json", "r") as f: 
        args = json.load(f)
    
    dataset = DATASETS[dataset_name](root, args)
    data_indices, num_clients = load_data_partition(dataset_name)
    return dataset, data_indices, num_clients

def get_client_datasets(dataset, data_indices, client_id):
    indices = data_indices[client_id]
    train_set = Subset(dataset, indices["train"])
    val_set = Subset(dataset, indices["val"])
    test_set = Subset(dataset, indices["test"])
    return train_set, val_set, test_set

def get_client_loaders(dataset, partition, client_id, batch_size=64):
    train_set, val_set, test_set = get_client_datasets(dataset, partition, client_id)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def init_client_models(model_name, model, client_num, dataset): 
    """
    Initialize client_num models with the base layers 
    from model and a random classifier 
    """
    base_layers = deepcopy(model.base.state_dict())
    client_models = [None] * client_num 

    for i in range(client_num): 
        client_model = MODELS[model_name](dataset = dataset, pretrained = False)
        client_model.base.load_state_dict(base_layers)
        
        # keep the classifier random 
        client_model.train()
        client_models[i] = client_model 
    
    return client_models

def test(test_loader, client_model):
    # set model to non-training mode 
    client_model.eval() 

    # init parameters 
    correct = 0 
    total = 0 
    with torch.no_grad(): 
        for x, y in test_loader: 
            x, y = x.to(device), y.to(device)
            outputs = client_model(x)
            predicted = torch.argmax(outputs, dim=1) 
            total += y.size(0)
            correct += (predicted == y).sum().item()
        
    
    acc = correct / total 
    return acc     


def finetune(client_model, train_loader, test_loader, config_args, epochs): 
    # freeze the base layers 
    for param in client_model.base.parameters(): 
        param.requires_grad = False 
    
    optimizer_cls = get_client_optimizer_cls(config_args)
    # Determine classifier parameters dynamically
    classifier_params = (
        [client_model.classifier] if isinstance(client_model.classifier, torch.nn.Parameter)
        else list(client_model.classifier.parameters())
    )
    optimizer = optimizer_cls(classifier_params)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    client_model.to(device)
    accuracies = np.zeros(epochs)
    for e in tqdm(range(epochs)): 
        client_model.train()
        losses = 0 

        # finetune one round
        for x, y in train_loader: 
            x, y, = x.to(device), y.to(device)
            
            optimizer.zero_grad()

            output = client_model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            losses += loss 
        print(losses)

        # test
        acc = test(test_loader, client_model) 
        accuracies[e] = acc 
    
    return accuracies * 100 
    

def save_results(results, run_dir): 
    with open(run_dir + "/include_clients_results.pkl", "wb") as out: 
        pickle.dump(results, out)
    out.close()


def include_new_clients(args): 
    # load configurations 
    config_args = load_config(args.run_dir)

    # fix seed 
    torch.manual_seed(config_args['common']['seed'])

    # get model and dataset 
    model, model_name = load_model(args.run_dir, config_args, config_args['dataset']['name'])
    dataset, data_indices, num_clients = get_dataset(config_args['dataset']['name'])

    if len(dataset) == 30000: 
        print(f"Length of dataset {len(dataset)}: Running on similar clients")
    else: 
        print(f"Length of dataset {len(dataset)}: Running on different clients")
    
    # get client models
    client_models = init_client_models(model_name, model, num_clients, config_args['dataset']['name'])

    # finetune each model 
    results = {}
    for client in range(num_clients): 
        train_loader, _, test_loader = get_client_loaders(dataset, data_indices, client)
        accuracies = finetune(client_models[client], train_loader, test_loader, config_args, args.finetune_epoch)
        results[client] = accuracies
    
    # save results 
    save_results(results, args.run_dir)




if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", help="directory to run with model and config files", type = str)
    parser.add_argument("--finetune_epoch", type = int, default = 20)

    args = parser.parse_args()

    include_new_clients(args)