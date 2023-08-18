from typing import Dict, Any, List, Union, Tuple
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn import STConv
from torch_geometric.nn import SAGEConv

class OutputLayer(torch.nn.Module):
    def __init__(self, c, T, n):
        super(OutputLayer, self).__init__()
        self.tconv1 = torch.nn.Conv2d(c, c, (T, 1), 1, dilation=1, padding=(0,0))
        self.ln = torch.nn.LayerNorm([n,c])
        self.tconv2 = torch.nn.Conv2d(c, c, (1, 1), 1, dilation=1, padding=(0,0))
        self.fc = torch.nn.Conv2d(c, 1, 1)

    def forward(self, x):
        x = self.tconv1(x)
        x = self.ln(x.permute(0,2,3,1))
        x = self.tconv2(x.permute(0,3,1,2))
        out = self.fc(x)
        return out

class STGNNModel(torch.nn.Module):
    def __init__(
            self,
            n_nodes: int,
            channel_sizes: List[Any],
            n_layers: int,
            kernel_size: int,
            K: int,
            window_size: int,
            normalization='sym',
            bias=True,
            loss_func=torch.nn.MSELoss(),
            optimizer_params: Dict[str, Any] = {},
            optimizer=None,
    ):
        super(STGNNModel, self).__init__()
        self.layers = torch.nn.ModuleList([])
        for l in range(n_layers):
            in_size, hidden_size, out_size = channel_sizes[l]
            self.layers.append(STConv(num_nodes=n_nodes,
                                      in_channels=in_size,
                                      hidden_channels=hidden_size,
                                      out_channels=out_size,
                                      kernel_size=kernel_size,
                                      K=K,
                                      normalization=normalization,
                                      bias=bias,
                                      ))
        self.layers.append(OutputLayer(channel_sizes[-1][-1],
                           window_size-2*n_layers*(kernel_size-1),
                           n_nodes,
                           ))
        self.loss_func = loss_func
        if optimizer is not None:
            self.optimizer = optimizer(self.parameters(), **optimizer_params) 
        else:
            self.optimizer = torch.optim.Adam(self.parameters(),**optimizer_params)
        
    def forward(self, x, edge_index, edge_weight):
        for layer in self.layers:
            if isinstance(layer, OutputLayer):
                x = layer(x.permute(0,3,1,2))
            else:
                x = layer(x, edge_index, edge_weight)
        return x
    
    def fit(self, dataloader_train, dataloader_val, edge_index, edge_weight, epochs=100, run_id=None):

        self.train()
        total_loss, n = 0.0, 0
        min_val_loss = np.inf
        epoch_pbar = tqdm(range(epochs), desc="Training Progress, Epoch", position=0)
        for i in epoch_pbar:
            # for data in dataloader:
            for X,y in dataloader_train:
                pred_y = self.forward(X, edge_index, edge_weight).view(len(X),-1)

                loss = self.loss_func(pred_y, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n += y.shape[0]
            
            # compute validation loss
            val_loss = self.compute_loss(dataloader_val, edge_index, edge_weight)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(self.state_dict(), './data/06_models/stgnn_best_model.pt')
            
            train_loss = total_loss/n
            print(f"Epoch: {i}, Training Loss: {train_loss}, Validation Loss: {val_loss}")
            
            self.train()


    def predict(self, X, edge_index, edge_weight):
        self.eval()
        with torch.no_grad():
            y_pred = self(X, edge_index, edge_weight).view(len(X),-1)
            y_pred = y_pred.cpu().numpy()
        return np.maximum(1e-10, y_pred)

    def compute_loss(self, dataloader, edge_index, edge_weight):
        self.eval()
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for X,y in dataloader:
                y_pred = self(X, edge_index, edge_weight).view(len(X),-1)
                l = self.loss_func(y_pred, y)
                total_loss += l.item() * y.shape[0]
                n += y.shape[0]
            return total_loss/n

    def compute_metrics(self, dataloader, scaler, edge_index, edge_weight):
        self.eval()
        with torch.no_grad():
            y_all, y_pred_all = [], []
            mae, mape, mse = [], [], []
            for X,y in dataloader:
                y_pred = self(X, edge_index, edge_weight).view(len(X),-1)

                if scaler is not None:
                    y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
                    y_pred = scaler.inverse_transform(y_pred.cpu().numpy()).reshape(-1)
                else:
                    y = y.cpu().numpy().reshape(-1)
                    y_pred = y_pred.cpu().numpy().reshape(-1)
                
                y = np.maximum(1e-10, y)
                y_pred = np.maximum(1e-10, y_pred)

                y_all.append(y)
                y_pred_all.append(y_pred)

                d = np.abs(y-y_pred)
                mae += d.tolist()
                mape += np.abs(d/y).tolist()
                mse += (d**2).tolist()
            
            MAE = np.array(mae).mean()
            MAPE = np.array(mape).mean()
            RMSE = np.sqrt(np.array(mse).mean())

            return MAE, MAPE, RMSE, y_all, y_pred_all


class GraphSAGE(torch.nn.Module):
    def __init__(
            self,
            n_layers: int,
            in_channels: Union[int, Tuple[int, int]],
            hidden_channels: int,
            out_channels: int,
            aggr_func: str='mean',
            dropout: float=0.1,
            loss_func=torch.nn.MSELoss(),
            optimizer_params: Dict[str, Any] = {},
            optimizer=None,
    ):
        super(GraphSAGE, self).__init__()
        
        self.dropout = dropout
        
        self.layers = torch.nn.ModuleList([])
        self.layers.append(SAGEConv(in_channels, hidden_channels, aggr_func))
        for _ in range(n_layers-2):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels, aggr_func))
        self.layers.append(SAGEConv(hidden_channels, out_channels, aggr_func))

        self.loss_func = loss_func
        if optimizer is not None:
            self.optimizer = optimizer(self.parameters(), **optimizer_params) 
        else:
            self.optimizer = torch.optim.Adam(self.parameters(),**optimizer_params)
        
    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout)
        x = self.layers[-1](x, edge_index)
        return x
    
    def fit(self, dataloader_train, dataloader_val, epochs=100, run_id=None):

        self.train()
        total_loss, n = 0.0, 0
        min_val_loss = np.inf
        epoch_pbar = tqdm(range(epochs), desc="Training Progress, Epoch", position=0)
        for i in epoch_pbar:
            for data in dataloader_train:
                X, y, edge_index = data.x, data.y, data.edge_index
                pred_y = self.forward(X, edge_index)

                loss = self.loss_func(pred_y, y.unsqueeze(-1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n += y.shape[0]
            
            # compute validation loss
            val_loss = self.compute_loss(dataloader_val)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(self.state_dict(), './data/06_models/simple-gnn_best_model.pt')
            
            train_loss = total_loss/n
            print(f"Epoch: {i}, Training Loss: {train_loss}, Validation Loss: {val_loss}")
            
            self.train()

    def predict(self, data):
        self.eval()
        X, edge_index = data.x, data.edge_index
        with torch.no_grad():
            y_pred = self(X, edge_index)
            y_pred = y_pred.cpu().numpy()
        return np.maximum(1e-10, y_pred)
    

    def compute_loss(self, dataloader):
        self.eval()
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for data in dataloader:
                X, y, edge_index = data.x, data.y, data.edge_index
                y_pred = self(X, edge_index)
                l = self.loss_func(y_pred, y.unsqueeze(-1))
                total_loss += l.item() * y.shape[0]
                n += y.shape[0]
            return total_loss/n

    def compute_metrics(self, dataloader, scaler):
        self.eval()
        with torch.no_grad():
            y_all, y_pred_all = [], []
            mae, mape, mse = [], [], []
            for data in dataloader:
                X, y, edge_index = data.x, data.y, data.edge_index
                y_pred = self(X, edge_index)

                if scaler is not None:
                    y = scaler.inverse_transform(y.cpu().numpy().reshape(data.num_graphs,-1)).reshape(-1)
                    y_pred = scaler.inverse_transform(y_pred.cpu().numpy().reshape(data.num_graphs,-1)).reshape(-1)
                else:
                    y = y.cpu().numpy().reshape(-1)
                    y_pred = y_pred.cpu().numpy().reshape(-1)
                
                y = np.maximum(1e-10, y)
                y_pred = np.maximum(1e-10, y_pred)

                d = np.abs(y-y_pred)
                mae += d.tolist()
                mape += np.abs(d/y).tolist()
                mse += (d**2).tolist()

                y_all.append(y)
                y_pred_all.append(y_pred)
            
            MAE = np.array(mae).mean()
            MAPE = np.array(mape).mean()
            RMSE = np.sqrt(np.array(mse).mean())

            return MAE, MAPE, RMSE, y_all, y_pred_all