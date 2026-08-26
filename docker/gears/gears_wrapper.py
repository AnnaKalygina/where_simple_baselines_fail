"""
GEARS model wrapper for CellSimBench integration.
Handles training and prediction with internal checkpointing.
"""

import logging
import pickle
import json
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from gears import PertData
import os
import random

# VENDORED (D3): drop the cellsimbench dependency. The two symbols the atheus
# wrapper imported from cellsimbench — a slim DataManager (no DEG gate) and a
# Path-aware JSON encoder — are inlined here directly, so there is no separate
# in-container harness module to bind-mount.


class PathEncoder(json.JSONEncoder):
    """JSON encoder that serializes ``pathlib.Path`` as ``str``."""

    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


class DataManager:
    """Minimal data loader: read one combined h5ad from ``config['data_path']``.

    The wrapper uses only ``DataManager(config)``, ``.load_dataset()`` and reads
    back ``.adata``. ``load_dataset()`` reads the combined h5ad (per-cell split
    column + all cell types inline) and returns the AnnData, clearing the gene
    index name to match the upstream behaviour some GEARS code paths assume.
    """

    def __init__(self, dataset_config: Dict):
        self.config = dataset_config
        self.adata = None

    def load_dataset(self):
        path = Path(self.config["data_path"])
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        print(f"[harness] Loading dataset from {path} ...", flush=True)
        self.adata = sc.read_h5ad(path)
        self.adata.var.index.name = None
        print(f"[harness] Loaded AnnData with shape: {self.adata.shape}", flush=True)
        return self.adata

log = logging.getLogger(__name__)


# ==========================================================================================================================================================
# =================== BASIC GEARS IMPLEMENTATION ===================
# ==========================================================================================================================================================

from copy import deepcopy
import os
import pickle
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch_geometric.seed import seed_everything

from gears.model import GEARS_Model
from gears.inference import evaluate, compute_metrics, deeper_analysis, \
                  non_dropout_analysis
from gears.utils import uncertainty_loss_fct, parse_any_pert, \
                  get_similarity_network, print_sys, GeneSimNetwork, \
                  create_cell_graph_dataset_for_prediction, get_mean_control, \
                  get_GI_genes_idx, get_GI_params
from torch_geometric.data import DataLoader
from tqdm import tqdm

torch.manual_seed(0)
seed_everything(0)

import warnings
warnings.filterwarnings("ignore")

class GEARS:
    """
    GEARS base model class
    """

    def __init__(self, pert_data, 
                 device = 'cuda',
                 weight_bias_track = False, 
                 proj_name = 'GEARS', 
                 exp_name = 'GEARS',
                 loss_weights_dict = None,
                 use_mse_loss = False):
        """
        Initialize GEARS model

        Parameters
        ----------
        pert_data: PertData object
            dataloader for perturbation data
        device: str
            Device to run the model on. Default: 'cuda'
        weight_bias_track: bool
            Whether to track performance on wandb. Default: False
        proj_name: str
            Project name for wandb. Default: 'GEARS'
        exp_name: str
            Experiment name for wandb. Default: 'GEARS'
        loss_weights: dict
            Dictionary of loss weights for each loss function. Default: None
        use_mse_loss: bool
            Whether to use MSE loss. Default: False

        Returns
        -------
        None

        """

        self.weight_bias_track = weight_bias_track
        
        if self.weight_bias_track:
            import wandb
            wandb.init(project=proj_name, name=exp_name)  
            self.wandb = wandb
        else:
            self.wandb = None
        
        self.device = device
        self.config = None
        
        self.dataloader = pert_data.dataloader
        self.adata = pert_data.adata
        self.node_map = pert_data.node_map
        self.node_map_pert = pert_data.node_map_pert
        self.data_path = pert_data.data_path
        self.dataset_name = pert_data.dataset_name
        self.split = pert_data.split
        self.seed = pert_data.seed
        self.train_gene_set_size = pert_data.train_gene_set_size
        self.set2conditions = pert_data.set2conditions
        self.subgroup = pert_data.subgroup
        self.gene_list = pert_data.gene_names.values.tolist()
        self.pert_list = pert_data.pert_names.tolist()
        self.num_genes = len(self.gene_list)
        self.num_perts = len(self.pert_list)
        self.default_pert_graph = pert_data.default_pert_graph
        self.gene2go = pert_data.gene2go
        self.saved_pred = {}
        self.saved_logvar_sum = {}
        self.loss_weights_dict = loss_weights_dict
        self.use_mse_loss = use_mse_loss

        self.ctrl_expression = torch.tensor(
            np.mean(self.adata[self.adata.obs.condition == 'ctrl'].X,
                    axis=0)).reshape(-1, ).to(self.device)
        pert_full_id2pert = dict(self.adata.obs[['condition_name', 'condition']].values)
        self.dict_filter = {pert_full_id2pert[i]: j for i, j in
                            self.adata.uns['non_zeros_gene_idx'].items() if
                            i in pert_full_id2pert}
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
        
        gene_dict = {g:i for i,g in enumerate(self.gene_list)}
        self.pert2gene = {p: gene_dict[pert] for p, pert in
                          enumerate(self.pert_list) if pert in self.gene_list}

    def tunable_parameters(self):
        """
        Return the tunable parameters of the model

        Returns
        -------
        dict
            Tunable parameters of the model

        """

        return {'hidden_size': 'hidden dimension, default 64',
                'num_go_gnn_layers': 'number of GNN layers for GO graph, default 1',
                'num_gene_gnn_layers': 'number of GNN layers for co-expression gene graph, default 1',
                'decoder_hidden_size': 'hidden dimension for gene-specific decoder, default 16',
                'num_similar_genes_go_graph': 'number of maximum similar K genes in the GO graph, default 20',
                'num_similar_genes_co_express_graph': 'number of maximum similar K genes in the co expression graph, default 20',
                'coexpress_threshold': 'pearson correlation threshold when constructing coexpression graph, default 0.4',
                'uncertainty': 'whether or not to turn on uncertainty mode, default False',
                'uncertainty_reg': 'regularization term to balance uncertainty loss and prediction loss, default 1',
                'direction_lambda': 'regularization term to balance direction loss and prediction loss, default 1'
               }
    
    def model_initialize(self, hidden_size = 64,
                         num_go_gnn_layers = 1, 
                         num_gene_gnn_layers = 1,
                         decoder_hidden_size = 16,
                         num_similar_genes_go_graph = 20,
                         num_similar_genes_co_express_graph = 20,                    
                         coexpress_threshold = 0.4,
                         uncertainty = False, 
                         uncertainty_reg = 1,
                         direction_lambda = 1e-1,
                         G_go = None,
                         G_go_weight = None,
                         G_coexpress = None,
                         G_coexpress_weight = None,
                         no_perturb = False,
                         **kwargs
                        ):
        """
        Initialize the model

        Parameters
        ----------
        hidden_size: int
            hidden dimension, default 64
        num_go_gnn_layers: int
            number of GNN layers for GO graph, default 1
        num_gene_gnn_layers: int
            number of GNN layers for co-expression gene graph, default 1
        decoder_hidden_size: int
            hidden dimension for gene-specific decoder, default 16
        num_similar_genes_go_graph: int
            number of maximum similar K genes in the GO graph, default 20
        num_similar_genes_co_express_graph: int
            number of maximum similar K genes in the co expression graph, default 20
        coexpress_threshold: float
            pearson correlation threshold when constructing coexpression graph, default 0.4
        uncertainty: bool
            whether or not to turn on uncertainty mode, default False
        uncertainty_reg: float
            regularization term to balance uncertainty loss and prediction loss, default 1
        direction_lambda: float
            regularization term to balance direction loss and prediction loss, default 1
        G_go: scipy.sparse.csr_matrix
            GO graph, default None
        G_go_weight: scipy.sparse.csr_matrix
            GO graph edge weights, default None
        G_coexpress: scipy.sparse.csr_matrix
            co-expression graph, default None
        G_coexpress_weight: scipy.sparse.csr_matrix
            co-expression graph edge weights, default None
        no_perturb: bool
            predict no perturbation condition, default False

        Returns
        -------
        None
        """
        
        self.config = {'hidden_size': hidden_size,
                       'num_go_gnn_layers' : num_go_gnn_layers, 
                       'num_gene_gnn_layers' : num_gene_gnn_layers,
                       'decoder_hidden_size' : decoder_hidden_size,
                       'num_similar_genes_go_graph' : num_similar_genes_go_graph,
                       'num_similar_genes_co_express_graph' : num_similar_genes_co_express_graph,
                       'coexpress_threshold': coexpress_threshold,
                       'uncertainty' : uncertainty, 
                       'uncertainty_reg' : uncertainty_reg,
                       'direction_lambda' : direction_lambda,
                       'G_go': G_go,
                       'G_go_weight': G_go_weight,
                       'G_coexpress': G_coexpress,
                       'G_coexpress_weight': G_coexpress_weight,
                       'device': self.device,
                       'num_genes': self.num_genes,
                       'num_perts': self.num_perts,
                       'no_perturb': no_perturb
                      }
        
        if self.wandb:
            self.wandb.config.update(self.config)
        
        if self.config['G_coexpress'] is None:
            ## calculating co expression similarity graph
            edge_list = get_similarity_network(network_type='co-express',
                                               adata=self.adata,
                                               threshold=coexpress_threshold,
                                               k=num_similar_genes_co_express_graph,
                                               data_path=self.data_path,
                                               data_name=self.dataset_name,
                                               split=self.split, seed=self.seed,
                                               gene2go=self.gene2go,
                                               train_gene_set_size=self.train_gene_set_size,
                                               set2conditions=self.set2conditions)

            sim_network = GeneSimNetwork(edge_list, self.gene_list, node_map = self.node_map)
            self.config['G_coexpress'] = sim_network.edge_index
            self.config['G_coexpress_weight'] = sim_network.edge_weight
        
        if self.config['G_go'] is None:
            ## calculating gene ontology similarity graph
            edge_list = get_similarity_network(network_type='go',
                                               adata=self.adata,
                                               threshold=coexpress_threshold,
                                               k=num_similar_genes_go_graph,
                                               pert_list=self.pert_list,
                                               data_path=self.data_path,
                                               data_name=self.dataset_name,
                                               split=self.split, seed=self.seed,
                                               train_gene_set_size=self.train_gene_set_size,
                                               set2conditions=self.set2conditions,
                                               gene2go=self.gene2go,
                                               default_pert_graph=self.default_pert_graph)

            sim_network = GeneSimNetwork(edge_list, self.pert_list, node_map = self.node_map_pert)
            self.config['G_go'] = sim_network.edge_index
            self.config['G_go_weight'] = sim_network.edge_weight
            
        self.model = GEARS_Model(self.config).to(self.device)
        self.best_model = deepcopy(self.model)
        
    def load_pretrained(self, path):
        """
        Load pretrained model

        Parameters
        ----------
        path: str
            path to the pretrained model

        Returns
        -------
        None
        """

        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        
        del config['device'], config['num_genes'], config['num_perts']
        self.model_initialize(**config)
        self.config = config
        
        state_dict = torch.load(os.path.join(path, 'model.pt'), map_location = torch.device('cpu'))
        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.best_model = self.model
    
    def save_model(self, path):
        """
        Save the model

        Parameters
        ----------
        path: str
            path to save the model

        Returns
        -------
        None

        """
        if not os.path.exists(path):
            os.mkdir(path)
        
        if self.config is None:
            raise ValueError('No model is initialized...')
        
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
       
        torch.save(self.best_model.state_dict(), os.path.join(path, 'model.pt'))
    
    def predict(self, pert_list, ctrl_adata_override=None):
        """
        Predict the transcriptome given a list of genes/gene combinations being
        perturbed

        ctrl_adata_override: when supplied, use these cells as the unperturbed
        context fed to create_cell_graph_dataset_for_prediction instead of all
        controls in self.adata. Lets the wrapper run inference per (covariate,
        perturbation) by feeding cell-line-specific controls — the input
        expression carries cell-line baseline, so predictions become
        cell-line-specific without any model arch change.

        Parameters
        ----------
        pert_list: list
            list of genes/gene combiantions to be perturbed

        Returns
        -------
        results_pred: dict
            dictionary of predicted transcriptome
        results_logvar: dict
            dictionary of uncertainty score

        """
        ## given a list of single/combo genes, return the transcriptome
        ## if uncertainty mode is on, also return uncertainty score.

        if ctrl_adata_override is not None:
            self.ctrl_adata = ctrl_adata_override
        else:
            self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
        for pert in pert_list:
            for i in pert:
                if i not in self.pert_list:
                    raise ValueError(i+ " is not in the perturbation graph. "
                                        "Please select from GEARS.pert_list!")
        
        if self.config['uncertainty']:
            results_logvar = {}
            
        self.best_model = self.best_model.to(self.device)
        self.best_model.eval()
        results_pred = {}
        results_logvar_sum = {}
        
        for pert in pert_list:
            try:
                #If prediction is already saved, then skip inference
                results_pred['_'.join(pert)] = self.saved_pred['_'.join(pert)]
                if self.config['uncertainty']:
                    results_logvar_sum['_'.join(pert)] = self.saved_logvar_sum['_'.join(pert)]
                continue
            except:
                pass
            
            cg = create_cell_graph_dataset_for_prediction(pert, self.ctrl_adata,
                                                    self.pert_list, self.device)
            loader = DataLoader(cg, 300, shuffle = False)
            batch = next(iter(loader))
            batch.to(self.device)

            with torch.no_grad():
                if self.config['uncertainty']:
                    p, unc = self.best_model(batch)
                    results_logvar['_'.join(pert)] = np.mean(unc.detach().cpu().numpy(), axis = 0)
                    results_logvar_sum['_'.join(pert)] = np.exp(-np.mean(results_logvar['_'.join(pert)]))
                else:
                    p = self.best_model(batch)
                    
            results_pred['_'.join(pert)] = np.mean(p.detach().cpu().numpy(), axis = 0)
                
        self.saved_pred.update(results_pred)
        
        if self.config['uncertainty']:
            self.saved_logvar_sum.update(results_logvar_sum)
            return results_pred, results_logvar_sum
        else:
            return results_pred
        
    def GI_predict(self, combo, GI_genes_file='./genes_with_hi_mean.npy'):
        """
        Predict the GI scores following perturbation of a given gene combination

        Parameters
        ----------
        combo: list
            list of genes to be perturbed
        GI_genes_file: str
            path to the file containing genes with high mean expression

        Returns
        -------
        GI scores for the given combinatorial perturbation based on GEARS
        predictions

        """

        ## if uncertainty mode is on, also return uncertainty score.
        try:
            # If prediction is already saved, then skip inference
            pred = {}
            pred[combo[0]] = self.saved_pred[combo[0]]
            pred[combo[1]] = self.saved_pred[combo[1]]
            pred['_'.join(combo)] = self.saved_pred['_'.join(combo)]
        except:
            if self.config['uncertainty']:
                pred = self.predict([[combo[0]], [combo[1]], combo])[0]
            else:
                pred = self.predict([[combo[0]], [combo[1]], combo])

        mean_control = get_mean_control(self.adata).values  
        pred = {p:pred[p]-mean_control for p in pred} 

        if GI_genes_file is not None:
            # If focussing on a specific subset of genes for calculating metrics
            GI_genes_idx = get_GI_genes_idx(self.adata, GI_genes_file)       
        else:
            GI_genes_idx = np.arange(len(self.adata.var.gene_name.values))
            
        pred = {p:pred[p][GI_genes_idx] for p in pred}
        return get_GI_params(pred, combo)
    
    def plot_perturbation(self, query, save_file = None):
        """
        Plot the perturbation graph

        Parameters
        ----------
        query: str
            condition to be queried
        save_file: str
            path to save the plot

        Returns
        -------
        None

        """

        import seaborn as sns
        import matplotlib.pyplot as plt
        
        sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5)

        adata = self.adata
        gene2idx = self.node_map
        cond2name = dict(adata.obs[['condition', 'condition_name']].values)
        gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))

        de_idx = [gene2idx[gene_raw2id[i]] for i in
                  adata.uns['top_non_dropout_de_20'][cond2name[query]]]
        genes = [gene_raw2id[i] for i in
                 adata.uns['top_non_dropout_de_20'][cond2name[query]]]
        truth = adata[adata.obs.condition == query].X.toarray()[:, de_idx]
        
        query_ = [q for q in query.split('+') if q != 'ctrl']
        pred = self.predict([query_])['_'.join(query_)][de_idx]
        ctrl_means = adata[adata.obs['condition'] == 'ctrl'].to_df().mean()[
            de_idx].values

        pred = pred - ctrl_means
        truth = truth - ctrl_means
        
        plt.figure(figsize=[16.5,4.5])
        plt.title(query)
        plt.boxplot(truth, showfliers=False,
                    medianprops = dict(linewidth=0))    

        for i in range(pred.shape[0]):
            _ = plt.scatter(i+1, pred[i], color='red')

        plt.axhline(0, linestyle="dashed", color = 'green')

        ax = plt.gca()
        ax.xaxis.set_ticklabels(genes, rotation = 90)

        plt.ylabel("Change in Gene Expression over Control",labelpad=10)
        plt.tick_params(axis='x', which='major', pad=5)
        plt.tick_params(axis='y', which='major', pad=5)
        sns.despine()
        
        if save_file:
            plt.savefig(save_file, bbox_inches='tight')
        plt.show()
    
    
    def train(self, epochs = 20, 
              lr = 1e-3,
              weight_decay = 5e-4
             ):
        """
        Train the model

        Parameters
        ----------
        epochs: int
            number of epochs to train
        lr: float
            learning rate
        weight_decay: float
            weight decay

        Returns
        -------
        None

        """
        
        train_loader = self.dataloader['train_loader']
        val_loader = self.dataloader['val_loader']
            
        self.model = self.model.to(self.device)
        best_model = deepcopy(self.model)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay = weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

        min_val = np.inf
        print_sys('Start Training...')

        for epoch in range(epochs):
            self.model.train()

            for step, batch in enumerate(train_loader):
                batch.to(self.device)
                optimizer.zero_grad()
                y = batch.y
                if self.config['uncertainty']:
                    pred, logvar = self.model(batch)
                    print(pred.shape, logvar.shape, y.shape)
                    loss = uncertainty_loss_fct(pred, logvar, y, batch.pert,
                                      reg = self.config['uncertainty_reg'],
                                      ctrl = self.ctrl_expression, 
                                      dict_filter = self.dict_filter,
                                      direction_lambda = self.config['direction_lambda'])
                else:
                    pred = self.model(batch)
                    loss = loss_fct(pred, y, batch.pert,
                                  ctrl = self.ctrl_expression, 
                                  dict_filter = self.dict_filter,
                                  direction_lambda = self.config['direction_lambda'],
                                  loss_weights_dict = self.loss_weights_dict,
                                  use_mse_loss = self.use_mse_loss)
                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                optimizer.step()

                if self.wandb:
                    self.wandb.log({'training_loss': loss.item()})

                if step % 50 == 0:
                    log = "Epoch {} Step {} Train Loss: {:.4f}" 
                    print_sys(log.format(epoch + 1, step + 1, loss.item()))

            scheduler.step()
            # Evaluate model performance on train and val set
            train_res = evaluate(train_loader, self.model,
                                 self.config['uncertainty'], self.device)
            val_res = evaluate(val_loader, self.model,
                                 self.config['uncertainty'], self.device)
            train_metrics, _ = compute_metrics(train_res)
            val_metrics, _ = compute_metrics(val_res)

            # Print epoch performance
            log = "Epoch {}: Train Overall MSE: {:.4f} " \
                  "Validation Overall MSE: {:.4f}. "
            print_sys(log.format(epoch + 1, train_metrics['mse'], 
                             val_metrics['mse']))
                             
            # Print Pearson correlation metrics for overall
            log = "Train Overall Pearson: {:.4f} " \
                  "Validation Overall Pearson: {:.4f}. "
            print_sys(log.format(train_metrics['pearson'],
                             val_metrics['pearson']))
            
            # Print epoch performance for DE genes
            log = "Train Top 20 DE MSE: {:.4f} " \
                  "Validation Top 20 DE MSE: {:.4f}. "
            print_sys(log.format(train_metrics['mse_de'],
                             val_metrics['mse_de']))
                             
            # Print Pearson correlation metrics for DE genes
            log = "Train Top 20 DE Pearson: {:.4f} " \
                  "Validation Top 20 DE Pearson: {:.4f}. "
            print_sys(log.format(train_metrics['pearson_de'],
                             val_metrics['pearson_de']))
            
            if self.wandb:
                metrics = ['mse', 'pearson']
                for m in metrics:
                    self.wandb.log({'train_' + m: train_metrics[m],
                               'val_'+m: val_metrics[m],
                               'train_de_' + m: train_metrics[m + '_de'],
                               'val_de_'+m: val_metrics[m + '_de']})
               
            if val_metrics['mse_de'] < min_val:
                min_val = val_metrics['mse_de']
                best_model = deepcopy(self.model)
                
        print_sys("Done!")
        self.best_model = best_model

        if 'test_loader' not in self.dataloader:
            print_sys('Done! No test dataloader detected.')
            return
            
        
        print_sys('Done!')


def loss_fct(pred, y, perts, ctrl = None, direction_lambda = 1e-3, dict_filter = None, loss_weights_dict = None, use_mse_loss = False):
    """
    Main MSE Loss function, includes direction loss

    Args:
        pred (torch.tensor): predicted values
        y (torch.tensor): true values
        perts (list): list of perturbations
        ctrl (str): control perturbation
        direction_lambda (float): direction loss weight hyperparameter
        dict_filter (dict): dictionary of perturbations to conditions
        loss_weights_dict (dict): dictionary of loss weights for each perturbation
        use_mse_loss (bool): whether to use MSE loss

    """
    gamma = 2
    mse_p = torch.nn.MSELoss()
    perts = np.array(perts)
    losses = torch.tensor(0.0, requires_grad=True).to(pred.device)

    for p in set(perts):
        pert_idx = np.where(perts == p)[0]
        
        if loss_weights_dict is not None:
            pred_p = pred[pert_idx]
            y_p = y[pert_idx]
            weights = loss_weights_dict[p]
            weights = torch.tensor(weights).to(pred.device)
            weights = weights[:pred_p.shape[1]]
        else:
            # during training, we remove the all zero genes into calculation of loss.
            # this gives a cleaner direction loss. empirically, the performance stays the same.
            if p!= 'ctrl':
                retain_idx = dict_filter[p]
                pred_p = pred[pert_idx][:, retain_idx]
                y_p = y[pert_idx][:, retain_idx]
            else:
                pred_p = pred[pert_idx]
                y_p = y[pert_idx]
            weights = torch.ones(pred_p.shape[1]).to(pred.device)
        
        if not use_mse_loss:
            losses = losses + torch.sum(weights * (pred_p - y_p)**(2 + gamma))/pred_p.shape[0]/pred_p.shape[1]
        else:
            losses = losses + torch.sum(weights * (pred_p - y_p)**2)/pred_p.shape[0]/pred_p.shape[1]

        ## direction loss
        if not use_mse_loss:
            if (p!= 'ctrl'):
                if loss_weights_dict is not None:
                    losses = losses + torch.sum(weights * direction_lambda *
                                        (torch.sign(y_p - ctrl) -
                                         torch.sign(pred_p - ctrl))**2)/\
                                         pred_p.shape[0]/pred_p.shape[1]
                else:
                    losses = losses + torch.sum(weights * direction_lambda *
                                        (torch.sign(y_p - ctrl[retain_idx]) -
                                         torch.sign(pred_p - ctrl[retain_idx]))**2)/\
                                         pred_p.shape[0]/pred_p.shape[1]
            else:
                losses = losses + torch.sum(weights * direction_lambda * (torch.sign(y_p - ctrl) -
                                                torch.sign(pred_p - ctrl))**2)/\
                                                pred_p.shape[0]/pred_p.shape[1]
    return losses/(len(set(perts)))


class GEARSWrapper:
    """GEARS model wrapper handling its own checkpointing and data conversion."""
    
    def __init__(self, config: Dict):
        self.config = config
        self._set_seed()                 # F1: make config['seed'] actually drive training
        self.model = None
        self.pert_data = None
        self.gene2go = self._load_gene2go()
        self.data_manager = DataManager(self.config)

    def _set_seed(self) -> None:
        """Seed every RNG from ``config['seed']`` (default 42) so the run is
        seed-controlled. The module-level ``torch.manual_seed(0)`` /
        ``seed_everything(0)`` at import fire before any config exists and are
        superseded here (covers both train and predict).

        We deliberately do NOT call ``torch.use_deterministic_algorithms(True)``:
        it raises on the nondeterministic PyG scatter ops GEARS uses. So same-seed
        runs get much closer but not bitwise-identical; residual CUDA
        nondeterminism is expected. The point is that the seed genuinely controls
        training (previously it was ignored — only CUDA nondeterminism varied).
        """
        seed = int(self.config.get('seed', 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        seed_everything(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        log.info(f"[seed] all RNGs seeded from config seed = {seed}")

    def _load_gene2go(self) -> Dict:
        """Load gene2go dictionary from pickle file."""
        # Default to /app/gene2go_all.pkl (Docker), fall back to alongside this script (Native).
        if 'gene2go_path' in self.config:
            gene2go_path = self.config['gene2go_path']
        elif os.path.exists('/app/gene2go_all.pkl'):
            gene2go_path = '/app/gene2go_all.pkl'
        else:
            gene2go_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gene2go_all.pkl')
        try:
            with open(gene2go_path, 'rb') as f:
                gene2go = pickle.load(f)
            log.info(f"Loaded gene2go dictionary with {len(gene2go)} entries")
            return gene2go
        except Exception as e:
            log.error(f"Failed to load gene2go from {gene2go_path}: {e}")
            raise
        
    def train(self):
        """Train GEARS model with internal checkpointing."""
        
        log.info("Starting GEARS training process...")
        
        # 1. Load and preprocess data
        log.info("Loading CellSimBench data...")
        adata = self.data_manager.load_dataset()
        log.info(f"Loaded data with shape: {adata.shape}")

        # 1b. Optional residualization: subtract a baseline's per-condition prediction
        # from adata.X so GEARS trains on residuals. Predictions add it back at inference.
        residualize_against = self.config['hyperparameters'].get('residualize_against')
        if residualize_against:
            log.info(f"Residualization enabled: {residualize_against}")
            baseline_dict = self._compute_baseline_predictions(adata, residualize_against)
            adata = self._apply_residualization(adata, baseline_dict)
            output_dir = Path(self.config['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            self._save_residualizer(output_dir, baseline_dict, residualize_against)

        # 2. Convert to GEARS format
        log.info("Converting to GEARS format...")
        self.pert_data = self._convert_to_gears_format(adata)
        
        # 3. Prepare splits
        log.info("Preparing data splits...")
        self._prepare_gears_splits()
        
        # 4. Get dataloaders
        log.info("Creating data loaders...")
        self.pert_data.get_dataloader(
            batch_size=self.config['hyperparameters']['batch_size'],
            test_batch_size=self.config['hyperparameters']['test_batch_size']
        )
        
        # 5. Initialize GEARS model
        log.info("Initializing GEARS model...")
        
        # Handle weights - remove covariate prefixes if they exist
        loss_weights_dict = None

        # VENDORED CHANGE: wire W&B tracking from our config (was hardcoded off).
        # Enabled when config['wandb'] is true; the endpoint/API key come from the
        # WANDB_BASE_URL / WANDB_API_KEY env vars passed into the container.
        self.model = GEARS(
            self.pert_data,
            device='cuda',
            weight_bias_track=bool(self.config.get('wandb', False)),
            loss_weights_dict=loss_weights_dict,
            use_mse_loss=self.config['hyperparameters']['use_mse_loss'],
            proj_name=self.config.get('wandb_project', 'gears-ct'),
            exp_name=self.config.get('wandb_run', 'gears_training'),
        )
        
        # Initialize model with hyperparameters
        model_params = self._get_model_params()
        self.model.model_initialize(**model_params)
        
        # 6. Check for existing checkpoints
        checkpoint_dir = Path(self.config['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # NOTE: no resume branch. The host wipes the run dir before every train
        # (TrainedPredictor._prepare_run_dir), so a resume path here was dead code
        # that would silently reactivate — training N more epochs on top of
        # existing weights while the metadata claimed a fresh run — if that wipe
        # were ever relaxed. Resumption, if wanted, must be an explicit mode in
        # the contract, not an implicit consequence of a leftover directory.

        # 7. Train with GEARS' own checkpointing
        log.info("Starting model training...")
        epochs = self.config['hyperparameters']['epochs']
        lr = self.config['hyperparameters']['lr']
        weight_decay = self.config['hyperparameters']['weight_decay']
        
        self.model.train(epochs=epochs, lr=lr, weight_decay=weight_decay)
        
        # 8. Save final model
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log.info(f"Saving model to {output_dir}")
        self.model.save_model(str(output_dir))
        self._save_metadata(output_dir)
        
        log.info("Training completed successfully")
        
    def predict(self):
        """Generate predictions using trained GEARS model."""
        
        log.info("Starting GEARS prediction process...")
        
        # 1. Load trained model
        model_path = Path(self.config['model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        log.info(f"Loading model from {model_path}")
        
        # 2. Load and preprocess data for prediction
        adata = self.data_manager.load_dataset()
        # Stash for _generate_predictions: needed to enumerate the *test* cov(s)
        # from obs[split_name] == 'test' (held-out cells are absent from the
        # post-leakage-filter self.model.adata).
        self._orig_adata = adata
        self.pert_data = self._convert_to_gears_format_for_prediction(adata)
        
        # 3. Get dataloaders
        log.info("Creating data loaders...")
        self.pert_data.get_dataloader(
            batch_size=self.config['hyperparameters']['batch_size'],
            test_batch_size=self.config['hyperparameters']['test_batch_size']
        )
        
        # 4. Initialize GEARS model
        log.info("Initializing GEARS model...")
        
        # Handle weights - remove covariate prefixes if they exist
        loss_weights_dict = None

        
        self.model = GEARS(
            self.pert_data, 
            device='cuda',
            weight_bias_track=False,
            loss_weights_dict=loss_weights_dict,
            use_mse_loss=self.config['hyperparameters']['use_mse_loss'],
            proj_name='cellsimbench',
            exp_name='gears_prediction'
        )
            
        # Initialize model with hyperparameters
        model_params = self._get_model_params()
        self.model.model_initialize(**model_params)
        
        # 5. Load the pre-trained model
        log.info("Loading pre-trained weights...")
        self.model.load_pretrained(str(model_path))
        
        # 6. Generate predictions
        test_conditions = self.config['test_conditions']
        log.info(f"Generating predictions for {len(test_conditions)} conditions")
        
        predictions = self._generate_predictions(test_conditions)
        
        # 7. Convert to CellSimBench format
        log.info("Converting predictions to CellSimBench format...")
        predictions_adata = self._convert_to_cellsimbench_format(predictions, adata)
        
        # 8. Save predictions
        output_path = self.config['output_path']
        log.info(f"Saving predictions to {output_path}")
        predictions_adata.write_h5ad(output_path)
        
        log.info("Prediction completed successfully")
        
    def _convert_to_gears_format(self, adata: sc.AnnData) -> PertData:
        """Convert CellSimBench data to GEARS PertData format."""
        
        # Create a copy for GEARS processing
        adata_gears = adata.copy()

        # Leakage-safe per-cell split filter. For UnseenCell (s2) / UnseenBoth
        # (s3) / UnseenPair (s4) regimes, the same perturbation can be 'train'
        # for one covariate and 'test' for another, so the downstream
        # condition-list filter (train_conditions / test_conditions) is not
        # enough — the test (cov, pert) cells would still be pulled into the
        # PertData object via cells of the same condition in OTHER covs. Drop
        # test-labelled cells here, before PertData processing.
        split_name = self.config.get('split_name')
        # UnseenPert (s1): test conditions are disjoint from train by definition.
        # The framework's condition-list filter (train_conditions / test_conditions)
        # already prevents leakage. Applying the per-cell filter here would delete
        # all cells of test perts, leaving GEARS' dataset_processed without those
        # keys → KeyError at get_dataloader on the test split.
        is_unseen_pert = bool(split_name) and (
            '_s1_' in split_name or 'UnseenPert' in split_name
        )
        if is_unseen_pert:
            log.info(
                f"Per-cell split filter SKIPPED for UnseenPert regime "
                f"(split_name={split_name!r})"
            )
        elif split_name and split_name in adata_gears.obs.columns:
            split_col = adata_gears.obs[split_name].astype(str)
            n_before = adata_gears.n_obs
            adata_gears = adata_gears[split_col.isin(['train', 'val'])].copy()
            log.info(
                f"Per-cell split filter on '{split_name}': {n_before} → "
                f"{adata_gears.n_obs} cells (dropped held-out test cells)"
            )
        else:
            log.warning(
                "No 'split_name' in config or column missing — skipping per-cell "
                "split filter. This is unsafe for s2/s3/s4 when covariates are on."
            )

        # TODO: We should be passing the control value as a parameter
        CTRL_VALUE = 'ctrl_iegfp'

        # Remove all the rows where the condition contains 'ctrl' but is not "control" or CTRL_VALUE
        adata_gears = adata_gears[~adata_gears.obs['condition'].str.contains('ctrl') | adata_gears.obs['condition'].isin(['control', CTRL_VALUE])]
                
        # Process condition labels for GEARS format
        # GEARS expects: 'ctrl', 'GENE1+ctrl', 'GENE1+GENE2'
        def process_condition(cond):
            # TODO: We should be passing the control value as a parameter
            if cond == 'control' or cond == 'ctrl' or cond == CTRL_VALUE:
                return 'ctrl'
            elif '+' not in cond:
                # Single perturbation - add +ctrl
                return f"{cond}+ctrl"
            else:
                # Already in correct format for combo perturbations
                return cond

        adata_gears.obs['condition'] = adata_gears.obs['condition'].astype(str)
        adata_gears.obs['condition'] = adata_gears.obs['condition'].apply(process_condition)

        # Resolve covariate column. Multi-cell-line datasets keep their real
        # per-cell label so cell_type metadata flows through GEARS's PertData;
        # the prediction-time fix in _generate_predictions then uses
        # cell-line-specific control cells to elicit cell-line-specific predicted
        # expression. Single-cell-line datasets fall back to the original
        # "NOTHING" placeholder so behaviour is unchanged.
        cov_field = self.config.get('covariate_key', None)   # F4: single canonical covariate key
        if cov_field and cov_field in adata_gears.obs.columns and adata_gears.obs[cov_field].nunique() >= 2:
            log.info(
                f"GEARS preserving real cov column {cov_field!r} ({adata_gears.obs[cov_field].nunique()} categories)"
            )
            adata_gears.obs['cell_type'] = adata_gears.obs[cov_field].astype(str).values
            self._gears_cov_field = cov_field
        else:
            adata_gears.obs['cell_type'] = "NOTHING"
            self._gears_cov_field = None

        # millerh1/GEARS PertData.new_data_process requires var['gene_name'].
        # Phase 2 datasets (jiang24/mcfaline23/replogle22) don't have it.
        if 'gene_name' not in adata_gears.var.columns:
            adata_gears.var['gene_name'] = adata_gears.var_names
        
        # Use output directory for persistent storage
        output_dir = Path(self.config['output_dir'])
        processed_data_dir = output_dir / 'processed_data'
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        log.info(f"Saving processed GEARS data to: {processed_data_dir}")
        
        # Create standard PertData object
        pert_data = PertData(str(processed_data_dir), default_pert_graph=False, gene2go=self.gene2go)
        
        processed_dataset_path = processed_data_dir / 'cellsimbench_gears'
        cell_graphs_file = processed_dataset_path / 'data_pyg' / 'cell_graphs.pkl'
        if not cell_graphs_file.exists():
            log.info("Creating new processed GEARS dataset...")
            pert_data.new_data_process(dataset_name='cellsimbench_gears', adata=adata_gears)
        else:
            log.info("Loading existing processed GEARS dataset...")
            pert_data.load(data_path=str(processed_dataset_path))
        
        return pert_data
        
    def _convert_to_gears_format_for_prediction(self, adata: sc.AnnData) -> PertData:
        """Convert CellSimBench data to GEARS PertData format for prediction only."""
        
        # Use the same output directory as training for loading processed data
        if 'model_path' in self.config:
            # For prediction, model_path points to the trained model directory
            model_dir = Path(self.config['model_path'])
            processed_data_dir = model_dir / 'processed_data'
        else:
            raise FileNotFoundError(f"Processed GEARS data not found. Please run training first.")
        
        processed_dataset_path = processed_data_dir / 'cellsimbench_gears'
        cell_graphs_file = processed_dataset_path / 'data_pyg' / 'cell_graphs.pkl'
        split_dict_file = processed_data_dir / 'cellsimbench_split_dict.pkl'
        
        log.info(f"Loading processed GEARS data from: {processed_dataset_path}")
        
        # Create standard PertData object and load existing processed data
        pert_data = PertData(str(processed_data_dir), default_pert_graph=False, gene2go=self.gene2go)
        
        # Always load existing processed data
        if cell_graphs_file.exists():
            log.info("Loading existing processed GEARS data for prediction...")
            pert_data.load(data_path=str(processed_dataset_path))
        else:
            raise FileNotFoundError(
                f"Processed GEARS data not found at '{cell_graphs_file}'. "
                "Please run training first to create the required data structures."
            )
        
        # Load and prepare splits
        if split_dict_file.exists():
            log.info("Loading existing split dictionary...")
            pert_data.prepare_split(split='custom', seed=int(self.config.get('seed', 42)), split_dict_path=str(split_dict_file))
        else:
            raise FileNotFoundError(
                f"Split dictionary not found at '{split_dict_file}'. "
                "Please run training first to create the required data structures."
            )
        
        return pert_data
        
    def _prepare_gears_splits(self):
        """Prepare train/val/test splits for GEARS."""
        
        # Get conditions from config
        train_conditions = self.config['train_conditions']
        val_conditions = self.config['val_conditions']
        test_conditions = self.config['test_conditions']
        
        # Convert to GEARS format
        def convert_conditions(conditions):
            gears_conditions = []
            for cond in conditions:
                if cond == 'control':
                    gears_conditions.append('ctrl')
                elif '+' not in cond:
                    gears_conditions.append(f"{cond}+ctrl")
                else:
                    gears_conditions.append(cond)
            return gears_conditions
        
        split_dict = {
            'train': convert_conditions(train_conditions),
            'val': convert_conditions(val_conditions),
            'test': convert_conditions(test_conditions)
        }
        
        # Filter out genes not in gene2go
        for split in ['train', 'val', 'test']:
            original_count = len(split_dict[split])
            filtered_conditions = []
            
            for cond in split_dict[split]:
                if cond == 'ctrl':
                    filtered_conditions.append(cond)
                else:
                    # Parse genes from condition
                    genes = self._parse_perturbation(cond)
                    if all(gene in self.pert_data.gene2go.keys() for gene in genes):
                        filtered_conditions.append(cond)
            
            split_dict[split] = filtered_conditions
            filtered_count = len(split_dict[split])
            log.info(f"{split} split: {filtered_count}/{original_count} conditions kept")
        
        # Save split dictionary to persistent location
        output_dir = Path(self.config['output_dir'])
        processed_data_dir = output_dir / 'processed_data'
        split_path = processed_data_dir / 'cellsimbench_split_dict.pkl'
        
        log.info(f"Saving split dictionary to: {split_path}")
        with open(split_path, 'wb') as f:
            pickle.dump(split_dict, f)
        
        # Prepare split in PertData
        self.pert_data.prepare_split(split='custom', seed=int(self.config.get('seed', 42)), split_dict_path=str(split_path))
        
    def _parse_perturbation(self, pert: str) -> List[str]:
        """Parse perturbation string into list of genes."""
        if pert == 'ctrl':
            return []
        elif '+ctrl' in pert:
            return [pert.replace('+ctrl', '')]
        elif '+' in pert:
            return pert.split('+')
        else:
            return [pert]
            
    def _get_model_params(self) -> Dict:
        """Get model initialization parameters from config."""
        hyperparams = self.config['hyperparameters']
        
        params = {
            'hidden_size': hyperparams['hidden_size'],
            'num_go_gnn_layers': hyperparams['num_go_gnn_layers'],
            'num_gene_gnn_layers': hyperparams['num_gene_gnn_layers'],
            'decoder_hidden_size': hyperparams['decoder_hidden_size'],
            'num_similar_genes_go_graph': hyperparams['num_similar_genes_go_graph'],
            'num_similar_genes_co_express_graph': hyperparams['num_similar_genes_co_express_graph'],
            'coexpress_threshold': hyperparams['coexpress_threshold'],
            'uncertainty': hyperparams['uncertainty'],
            'uncertainty_reg': hyperparams['uncertainty_reg'],
            'direction_lambda': hyperparams['direction_lambda']
        }
        
        return params
        
    def _generate_predictions(self, test_conditions: List[str]) -> Dict:
        """Generate GEARS predictions for test conditions."""
        
        # Convert conditions to GEARS format and parse into gene lists
        gears_conditions = []
        for cond in test_conditions:
            if cond != 'control':  # Skip control condition
                # Convert to GEARS format first
                if '+' not in cond:
                    # Single perturbation - add +ctrl for parsing
                    gears_cond = f"{cond}+ctrl"
                else:
                    gears_cond = cond
                    
                # Parse into gene list
                genes = self._parse_perturbation(gears_cond)
                # Check if all genes are in gene2go dictionary
                if genes and all(gene in self.pert_data.gene2go.keys() for gene in genes):
                    gears_conditions.append(genes)
                    log.info(f"Adding condition for prediction: {cond} -> {genes}")
                else:
                    log.warning(f"Skipping condition {cond}: genes {genes} not found in gene2go")
        
        log.info(f"Generating predictions for {len(gears_conditions)} valid conditions")

        if not gears_conditions:
            log.warning("No valid conditions found for prediction")
            return {}

        # Decide cov_values from the TEST split of the unfiltered dataset.
        #
        # self.model.adata is the post-leakage-filter PertData adata — at training
        # time, the per-cell split filter dropped rows where obs[split_name]=='test'
        # to prevent UnseenCell/Both/Pair leakage. That filter is correct, but it
        # also strips the held-out cov entirely for s2/s3, so enumerating from
        # self.model.adata at predict time emits an in-distribution dump on the
        # *training* covs instead of OOD predictions for the held-out test cov(s).
        #
        # Fix: read the cov column from the ORIGINAL adata, restricted to
        # obs[split_name]=='test'. Ctrl cells for the held-out cov come from the
        # original adata too (they don't exist in self.model.adata).
        cov_field = self.config.get('covariate_key') or 'cell_type'   # F4
        ctrl_model = self.model.adata[self.model.adata.obs['condition'] == 'ctrl']
        orig = getattr(self, '_orig_adata', None)
        split_name = self.config.get('split_name')

        cov_values: List[str]
        ctrl_full = ctrl_model  # default for legacy paths
        used_test_split = False
        if (
            orig is not None
            and split_name
            and split_name in orig.obs.columns
            and cov_field in orig.obs.columns
        ):
            test_mask = orig.obs[split_name].astype(str) == 'test'
            test_covs = sorted(
                orig.obs.loc[test_mask, cov_field].astype(str).unique().tolist()
            )
            if test_covs:
                cov_values = test_covs
                used_test_split = True
                # Align ctrl cells from original adata to the model's gene set.
                # If the model's vars are a strict subset of orig.vars, reindex;
                # otherwise just use orig as-is and let downstream code handle it.
                try:
                    model_vars = self.pert_data.adata.var_names
                    if set(model_vars).issubset(set(orig.var_names)):
                        orig_ctrl_src = orig[:, model_vars]
                    else:
                        orig_ctrl_src = orig
                except Exception:
                    orig_ctrl_src = orig
                # Original h5ad uses 'control' (or 'ctrl_iegfp'); only the
                # wrapper-converted self.model.adata uses 'ctrl'. Match all
                # control labels the trainer recognises (see line ~924).
                CTRL_LABELS_ORIG = {'control', 'ctrl', 'ctrl_iegfp'}
                ctrl_full = orig_ctrl_src[
                    orig_ctrl_src.obs['condition'].astype(str).isin(CTRL_LABELS_ORIG)
                ]
                log.info(
                    f"GEARS cov_values from test split '{split_name}': {cov_values} "
                    f"(orig ctrl pool {len(ctrl_full)} cells)"
                )
            else:
                log.warning(
                    f"GEARS: no 'test' cells in '{split_name}'; falling back to "
                    f"model adata covs"
                )
        if not used_test_split:
            if cov_field in ctrl_model.obs.columns:
                cov_values = sorted(
                    ctrl_model.obs[cov_field].astype(str).unique().tolist()
                )
            else:
                cov_values = ['NOTHING']

        # cov-aware mode applies whenever we have a real cov column and at least
        # one non-placeholder cov. Even a single test cov (UnseenCell) goes through
        # the per-cov loop so predictions are tagged with the cov label.
        cov_aware = bool(cov_values) and 'NOTHING' not in cov_values

        if not cov_aware:
            return self.model.predict(gears_conditions)

        # Cov-aware path: returns dict keyed by (cov, "_".join(pert))
        results: Dict = {}
        for cov in cov_values:
            cov_ctrl = ctrl_full[ctrl_full.obs[cov_field].astype(str) == cov] \
                if cov_field in ctrl_full.obs.columns else ctrl_full
            if len(cov_ctrl) == 0:
                log.warning(f"  GEARS: no ctrl cells for cov={cov}, skipping")
                continue
            log.info(f"  GEARS predicting cov={cov} using {len(cov_ctrl)} ctrl cells")
            # Invalidate the per-pert cache so identical pert keys recompute fresh
            # for each covariate (saved_pred is keyed only by pert string).
            self.model.saved_pred = {}
            preds = self.model.predict(gears_conditions, ctrl_adata_override=cov_ctrl)
            for cond_key, vec in preds.items():
                results[(cov, cond_key)] = vec
        return results
        
    def _convert_to_cellsimbench_format(self, predictions: Dict, original_adata: sc.AnnData) -> sc.AnnData:
        """Convert GEARS predictions to CellSimBench format."""
        
        # Handle empty predictions
        if not predictions:
            log.warning("No predictions to convert - returning empty AnnData")
            obs_df = pd.DataFrame(columns=['condition', 'pair_key'])
            adata_pred = sc.AnnData(X=np.empty((0, original_adata.n_vars)), obs=obs_df)
            adata_pred.var_names = original_adata.var_names
            return adata_pred
        
        test_conditions = self.config['test_conditions']

        # If the model was trained on residuals, reload the baseline so we can
        # add it back to each per-condition prediction before saving.
        residualizer = self._load_residualizer(Path(self.config['model_path']))

        # Detect cov-aware predictions: keys are (cov, cond_key) tuples in cov-aware
        # mode and plain "cond_key" strings in legacy mode. Normalize both shapes.
        if predictions and isinstance(next(iter(predictions.keys())), tuple):
            # cov-aware: dict[(cov, cond_key)] -> vec
            cov_aware = True
            keys_by_cond: Dict[str, List[str]] = {}
            for (cov, cond_key) in predictions.keys():
                keys_by_cond.setdefault(cond_key, []).append(cov)
        else:
            cov_aware = False
            keys_by_cond = {k: [""] for k in predictions.keys()}

        prediction_list = []
        condition_list = []
        covariate_list = []
        pair_key_list = []

        for condition in test_conditions:
            if condition == 'control':
                continue
            # Find which GEARS key matches this condition
            cond_key_match = None
            for key in keys_by_cond:
                genes = key.split('_')
                if len(genes) == 1 and condition == genes[0]:
                    cond_key_match = key; break
                elif len(genes) > 1 and condition == '+'.join(genes):
                    cond_key_match = key; break
            if cond_key_match is None:
                log.warning(f"No prediction found for {condition}")
                continue

            for cov in keys_by_cond[cond_key_match]:
                lookup_key = (cov, cond_key_match) if cov_aware else cond_key_match
                pred_vec = predictions[lookup_key]
                if residualizer is not None:
                    residualizer_is_cov_aware = (
                        bool(residualizer) and isinstance(next(iter(residualizer.keys())), tuple)
                    )
                    if residualizer_is_cov_aware:
                        # Try (cov_lower, condition) first; fall back to other casings
                        baseline_vec = residualizer.get((cov.lower(), condition))
                        if baseline_vec is None:
                            baseline_vec = residualizer.get((cov, condition))
                    else:
                        baseline_vec = residualizer.get(condition)
                    if baseline_vec is None:
                        log.warning(
                            f"Residualizer missing baseline for "
                            f"{(cov, condition) if residualizer_is_cov_aware else condition!r}; "
                            f"prediction will be returned as raw residual."
                        )
                    else:
                        pred_vec = pred_vec + baseline_vec
                prediction_list.append(pred_vec)
                condition_list.append(condition)
                covariate_list.append(cov if cov else "none")
                pair_key_list.append(f"{cov}_{condition}" if cov else condition)

        if not prediction_list:
            raise ValueError("No valid predictions found for test conditions")

        prediction_matrix = np.vstack(prediction_list)

        if cov_aware:
            obs_idx = [f"{c}__{cond}" for c, cond in zip(covariate_list, condition_list)]
        else:
            obs_idx = list(condition_list)
        obs_df = pd.DataFrame(
            {
                'covariate': covariate_list,
                'condition': condition_list,
                'pair_key': pair_key_list,
            },
            index=pd.Index(obs_idx),
        )

        adata_pred = sc.AnnData(X=prediction_matrix, obs=obs_df)
        adata_pred.var_names = original_adata.var_names
        log.info(
            f"GEARS output adata shape {adata_pred.shape}; cov-aware={cov_aware}, "
            f"n_covariates={len(set(covariate_list))}"
        )
        return adata_pred
        
    def _compute_baseline_predictions(self, adata: sc.AnnData, baseline_name: str) -> Dict[str, np.ndarray]:
        """Train a baseline model on adata and return per-condition predictions.

        Used for residualization. Predictions are computed for every unique
        condition in ``adata.obs['condition']`` so each cell can have its own
        condition's baseline subtracted.

        Supported baselines:
          - 'linear_regression': per-condition prediction from a one-hot linear
            regression baseline. Keys are condition strings.
          - 'dmts_npz': per-(pert, cov) absolute baseline derived from a
            DM+TS delta-from-ctrl NPZ (path via config['hyperparameters']['dmts_npz_path']).
            Keys are (cov_lower, condition) tuples. Cov-aware residualization
            requires that adata.obs has the covariate column declared in
            config['covariate_key'] (default 'cell_type').
        """
        if baseline_name == 'dmts_npz':
            return self._compute_baseline_predictions_dmts_npz(adata)
        if baseline_name == 'linear_regression':
            from cellsimbench.models.onehot_linear_regression import OneHotLinearRegressionModel
            builtin_cls = OneHotLinearRegressionModel
        else:
            raise ValueError(f"Unsupported residualize_against baseline: {baseline_name}")

        # OneHotLinearRegressionModel.predict needs data_manager.adata loaded.
        # data_manager loads dataset internally; ensure it's populated.
        if self.data_manager.adata is None:
            self.data_manager.load_dataset()

        model_cfg = {'name': baseline_name, 'hyperparameters': {}}
        builtin = builtin_cls(model_cfg)
        split_name = self.config['split_name']
        all_conditions = list(adata.obs['condition'].astype(str).unique())
        log.info(f"Fitting {baseline_name} baseline and predicting for {len(all_conditions)} conditions")
        pred_adata = builtin.predict(
            data_manager=self.data_manager,
            test_conditions=all_conditions,
            split_name=split_name,
        )

        baseline_dict: Dict[str, np.ndarray] = {}
        x_pred = np.asarray(pred_adata.X)
        for i, cond in enumerate(pred_adata.obs['condition'].astype(str).tolist()):
            baseline_dict[cond] = x_pred[i].astype(np.float32)

        log.info(
            f"Baseline produced predictions for {len(baseline_dict)}/{len(all_conditions)} conditions"
        )
        return baseline_dict

    def _compute_baseline_predictions_dmts_npz(self, adata: sc.AnnData) -> Dict[Tuple[str, str], np.ndarray]:
        """Load DM+TS NPZ + ctrl-mean per cov → absolute baseline keyed by (cov_lower, pert).

        NPZ format (from build_replogle22_dmts_per_fold.py): keys 'pert_keys',
        'cov_keys' (lowercased), 'gene_names', 'baseline' (n_pairs, n_genes_npz)
        in DELTA-FROM-CTRL space. Absolute baseline = ctrl_mean[cov] + delta.
        """
        path = self.config['hyperparameters'].get('dmts_npz_path')
        if not path:
            raise ValueError("residualize_against='dmts_npz' requires hyperparameters.dmts_npz_path")
        log.info(f"Loading DM+TS NPZ from {path}")
        npz = np.load(path, allow_pickle=True)
        pert_keys = np.asarray(npz['pert_keys']).astype(str)
        cov_keys = np.asarray(npz['cov_keys']).astype(str)
        gene_names_npz = np.asarray(npz['gene_names']).astype(str).tolist()
        baseline_delta = np.asarray(npz['baseline']).astype(np.float32)
        log.info(f"  NPZ: {baseline_delta.shape}, covs={sorted(set(cov_keys))}")

        cov_col = self.config.get('covariate_key', 'cell_type')
        if cov_col not in adata.obs.columns:
            raise ValueError(f"Covariate column '{cov_col}' missing from adata.obs for DMTS residualization")
        cond_series = adata.obs['condition'].astype(str)
        is_ctrl = cond_series.str.lower().isin({'control', 'ctrl', 'ctrl_iegfp', 'non-targeting', 'non_targeting'}).to_numpy()
        cov_arr = adata.obs[cov_col].astype(str).str.lower().to_numpy()

        ctrl_means: Dict[str, np.ndarray] = {}
        import scipy.sparse as sp
        for c in sorted(set(cov_arr)):
            mask = is_ctrl & (cov_arr == c)
            if not mask.any():
                log.warning(f"  cov={c}: no control cells found"); continue
            sub = adata.X[mask].toarray() if sp.issparse(adata.X) else np.asarray(adata.X[mask])
            ctrl_means[c] = np.asarray(sub.mean(axis=0)).reshape(-1).astype(np.float32)
            log.info(f"  ctrl_mean[{c}]: from {mask.sum()} cells")

        gene_names_adata = list(adata.var_names)
        if gene_names_adata == gene_names_npz:
            gene_map = np.arange(len(gene_names_adata), dtype=int)
        else:
            g_npz_idx = {g: i for i, g in enumerate(gene_names_npz)}
            gene_map = np.asarray([g_npz_idx.get(g, -1) for g in gene_names_adata], dtype=int)

        baseline_dict: Dict[Tuple[str, str], np.ndarray] = {}
        valid = gene_map >= 0
        n_unmapped = int((~valid).sum())
        if n_unmapped:
            log.warning(f"  {n_unmapped}/{len(gene_names_adata)} adata genes not in NPZ — those positions stay at ctrl_mean")
        for i, (p, c) in enumerate(zip(pert_keys, cov_keys)):
            if c not in ctrl_means:
                continue
            cm = ctrl_means[c]
            delta_aligned = np.zeros(len(gene_names_adata), dtype=np.float32)
            delta_aligned[valid] = baseline_delta[i, gene_map[valid]]
            baseline_dict[(c, p)] = cm + delta_aligned

        log.info(f"  built absolute baselines for {len(baseline_dict)} (cov, pert) pairs")
        return baseline_dict

    def _apply_residualization(self, adata: sc.AnnData, baseline_dict: Dict) -> sc.AnnData:
        """Subtract per-condition (or per-(cov, condition)) baseline from each cell's expression.

        ``baseline_dict`` keyed by ``condition`` (legacy linear_regression) OR by
        ``(cov_lower, condition)`` tuple (dmts_npz). Inferred from the first key.
        """
        import scipy.sparse as sp

        conditions = adata.obs['condition'].astype(str).to_numpy()
        cov_aware = bool(baseline_dict) and isinstance(next(iter(baseline_dict.keys())), tuple)
        if cov_aware:
            cov_col = self.config.get('covariate_key', 'cell_type')
            covs = adata.obs[cov_col].astype(str).str.lower().to_numpy()
            keys = list(zip(covs.tolist(), conditions.tolist()))
        else:
            keys = list(conditions)
        n_cells, n_genes = adata.shape
        baseline_matrix = np.zeros((n_cells, n_genes), dtype=np.float32)
        n_subtracted = 0
        n_missing = 0
        for i, key in enumerate(keys):
            vec = baseline_dict.get(key)
            if vec is None:
                n_missing += 1
                continue
            baseline_matrix[i] = vec
            n_subtracted += 1

        log.info(
            f"Residualizing: subtracting baseline from {n_subtracted}/{n_cells} cells "
            f"({n_missing} unchanged due to missing baseline)"
        )

        X = adata.X
        was_sparse = sp.issparse(X)
        if was_sparse:
            X = X.toarray()
        X_residual = np.asarray(X, dtype=np.float32) - baseline_matrix
        # Downstream GEARS code (gears.data_utils.get_dropout_non_zero_genes)
        # calls adata.X.toarray(), so keep the matrix sparse if it started sparse.
        adata.X = sp.csr_matrix(X_residual) if was_sparse else X_residual
        return adata

    def _save_residualizer(
        self,
        output_dir: Path,
        baseline_dict: Dict,
        baseline_name: str,
    ) -> None:
        """Persist baseline predictions for inference-time re-addition.

        Tuple keys (cov, pert) are stored as ``"cov::pert"`` strings; plain
        condition-string keys are stored as-is. ``cov_aware`` is recorded so
        load can round-trip the right key shape.
        """
        keys_iter = list(baseline_dict.keys())
        cov_aware = bool(keys_iter) and isinstance(keys_iter[0], tuple)
        if cov_aware:
            condition_strs = [f"{c}::{p}" for (c, p) in keys_iter]
        else:
            condition_strs = list(keys_iter)
        matrix = np.vstack([baseline_dict[k] for k in keys_iter]).astype(np.float32)
        np.savez(
            output_dir / 'residualizer.npz',
            conditions=np.array(condition_strs, dtype=object),
            baseline=matrix,
            baseline_name=np.array(baseline_name, dtype=object),
            cov_aware=np.array(cov_aware),
        )
        log.info(
            f"Saved residualizer to {output_dir/'residualizer.npz'} "
            f"({len(condition_strs)} entries, {matrix.shape[1]} genes, cov_aware={cov_aware})"
        )

    def _load_residualizer(self, model_dir: Path) -> Optional[Dict]:
        """Load residualizer if present; returns None if file is missing.

        Returns a dict keyed by either condition string (legacy) or
        ``(cov_lower, pert)`` tuple (cov-aware), depending on how it was saved.
        """
        path = model_dir / 'residualizer.npz'
        if not path.exists():
            return None
        data = np.load(path, allow_pickle=True)
        conditions = list(data['conditions'])
        matrix = data['baseline']
        cov_aware = bool(data['cov_aware'].item()) if 'cov_aware' in data.files else False
        log.info(f"Loaded residualizer from {path}: {len(conditions)} entries, cov_aware={cov_aware}")
        if cov_aware:
            out: Dict[Tuple[str, str], np.ndarray] = {}
            for i, s in enumerate(conditions):
                s = str(s)
                if "::" not in s:
                    log.warning(f"residualizer entry '{s}' lacks '::' separator; skipping")
                    continue
                c, p = s.split("::", 1)
                out[(c, p)] = matrix[i]
            return out
        return {str(c): matrix[i] for i, c in enumerate(conditions)}

    def _save_metadata(self, output_dir: Path):
        """Save training metadata."""
        metadata = {
            'model_type': 'GEARS',
            'config': self.config,
            'data_shape': self.pert_data.adata.shape if self.pert_data else None,
            'n_genes': len(self.pert_data.gene_names) if self.pert_data else None,
            'n_perturbations': len(self.pert_data.pert_names) if self.pert_data else None
        }

        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, cls=PathEncoder)