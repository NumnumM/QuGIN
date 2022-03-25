import json
import os

import torch
import torch.nn as nn
from ogb.utils.mol import smiles2graph
from sklearn.metrics import mean_absolute_error
from tdc.benchmark_group import admet_group
from torch import optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from model import DrugNet


class Task():
    def __init__(self, model, train_df, valid_df, test_df):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(),lr=0.001)
        self.criterion=nn.MSELoss()
        
        train_dataset = list(self._get_dataset(train_df))
        valid_dataset = list(self._get_dataset(valid_df))
        test_dataset = list(self._get_dataset(test_df))

        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
    def _get_dataset(self, df):
        for index, mol in tqdm(df.iterrows(), total=df.shape[0]):
            graph = smiles2graph(mol['Drug'])
            label = torch.tensor(mol["Y"], dtype=torch.float32)
        
            data = Data(x = torch.from_numpy(graph['node_feat']), 
                        edge_index = torch.from_numpy(graph['edge_index']),
                        edge_attr = torch.from_numpy(graph['edge_feat']),
                        num_node = graph['num_nodes'],
                        y = label)
            yield data
    
    def train(self):
        self.model.train()
        loss_per_epoch_train = 0
        label_lst = []
        train_pred = []
        for data in self.train_loader:
            node_feature = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(torch.int32).to(device)
            batch = data.batch.to(device)
            label = data.y.squeeze().to(device)
            
            self.optimizer.zero_grad(set_to_none=True)
            predict = self.model(node_feature, edge_index, edge_attr, batch)
            label_lst.append(label)
            train_pred.append(predict)
            loss = self.criterion(predict, label)
            loss.backward()
            self.optimizer.step()         #每个batch更新一次参数
            loss_per_epoch_train += loss.item()
            
        loss_per_epoch_train = loss_per_epoch_train / len(self.train_loader)
        return loss_per_epoch_train, torch.cat(train_pred, dim=0).tolist(), torch.cat(label_lst, dim=0).tolist()
    
    @torch.no_grad()
    def valid(self):
        loss_per_epoch_test = 0
        self.model.eval()
        label_lst = []
        test_pred = []
        for data in self.valid_loader:
            node_feature = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            batch = data.batch.to(device)
            label = data.y.squeeze().to(device)
            predict = self.model(node_feature, edge_index, edge_attr, batch)
            label_lst.append(label)
            test_pred.append(predict)
            loss = self.criterion(predict, label)
            loss_per_epoch_test += loss.item()
        # 计算经过一个epoch的训练后再测试集上的损失和精度
        loss_per_epoch_test = loss_per_epoch_test / len(self.valid_loader)
        return loss_per_epoch_test, torch.cat(test_pred, dim=0).tolist(), torch.cat(label_lst, dim=0).tolist()
    
    @torch.no_grad()
    def test(self):
        loss_per_epoch_test = 0
        self.model.eval()
        test_pred = []
        for data in self.test_loader:
            node_feature = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            batch = data.batch.to(device)
            predict = self.model(node_feature, edge_index, edge_attr, batch)
            test_pred.append(predict)
        # 计算经过一个epoch的训练后再测试集上的损失和精度
        loss_per_epoch_test = loss_per_epoch_test / len(self.valid_loader)
        return torch.cat(test_pred, dim=0).tolist()
  

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 100
    
    group = admet_group(path = 'data/')
    predictions_list = []

    for seed in [1, 2, 3, 4, 5]:
        benchmark = group.get('LD50_Zhu') 
        # all benchmark names in a benchmark group are stored in group.dataset_names
        predictions = {}
        name = benchmark['name']
        train_val, test = benchmark['train_val'], benchmark['test']
        train, valid = group.get_train_valid_split(benchmark = name, split_type = 'default', seed = seed)
        model = DrugNet(128, 1, 0.1).to(device)
        task = Task(model, train, valid, test)

        train_loss_lst = []
        valid_loss_lst = []
        train_mae_lst = []
        valid_mae_lst = []
        
        min_loss = 100
        for epoch in tqdm(range(epochs)):
            # ——————————————————————train—————————————————————
            loss_per_epoch_train, train_predict, train_label = task.train()
            train_loss_lst.append(loss_per_epoch_train)
            
            # ——————————————————————valid—————————————————————
            loss_per_epoch_valid, valid_predict, valid_label = task.valid()
            valid_loss_lst.append(loss_per_epoch_valid)
            
            # ——————————————————————score_MAE—————————————————
            train_mae = mean_absolute_error(train_label, train_predict)
            train_mae_lst.append(train_mae)
            
            valid_mae = mean_absolute_error(valid_label, valid_predict)
            valid_mae_lst.append(valid_mae)
            
            # ——————————————————————save_model————————————————
            if (loss_per_epoch_valid < min_loss) and (epoch > 50):
                test_predict = task.test()
                min_loss = loss_per_epoch_valid
                torch.save(model, f'./data/cache/model_{seed}.pkl')

            # ——————————————————————print—————————————————————
            print(f'train_loss: {loss_per_epoch_train:.3f} || train_mae: {train_mae:.3f}')
            print(f'valid_loss: {loss_per_epoch_valid:.3f} || valid_mae: {valid_mae:.3f}')

        save_path = "./data/cache/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        dict = {"train_loss": train_loss_lst, "test_loss": valid_loss_lst, "train_mae": train_mae_lst, "valid_mae": valid_mae_lst}
        with open(save_path + f"train_data{1}.json", "w") as f:
            json.dump(dict, f)
        
        predictions[name] = test_predict
        predictions_list.append(predictions)

    results = group.evaluate_many(predictions_list)
    print(results)
    print('Finished training ')
    save_path = "./data/cache/"
    with open(save_path + "result.json", "w") as f:
        json.dump(results, f)
