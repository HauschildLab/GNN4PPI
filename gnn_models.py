# Created by julia at 25.07.2024

import pickle
import torch
import torch.nn as nn
import torchmetrics
import lightning as pl
import torch_geometric as tg
from torch_geometric.nn.conv import GCNConv, GATConv, ChebConv
from torch_geometric.data import Data


################################LIGHTNING CLASSES###########################################
class LightningClass(pl.LightningModule):
    """
    Class which contains foundation of Lightning, incl. training, testing, validation, prediction wrt step and
    configuration of the optimizer
    """

    def __init__(self, store_predict=False):
        super(LightningClass, self).__init__()

        # For prediction storage
        self.store_predict = store_predict
        self.val_targets = torch.Tensor(device='cpu')
        self.val_outputs = torch.Tensor(device='cpu')
        self.val_predictions = torch.Tensor(device='cpu')
        self.test_targets = torch.Tensor(device='cpu')
        self.test_outputs = torch.Tensor(device='cpu')
        self.test_predictions = torch.Tensor(device='cpu')

    def training_step(self, batch, batch_idx):
        loss, scores, y = self.step(batch, batch_idx)
        accuracy = self.accuracy(torch.sigmoid(scores), y)

        self.log_dict(
            {"train_loss": loss, "train_accuracy": accuracy},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=y.shape[0]
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self.step(batch, batch_idx)
        accuracy = self.accuracy(torch.sigmoid(scores), y)

        self.log_dict(
            {"val_loss": loss, "val_accuracy": accuracy},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=y.shape[0]
        )

        if self.store_predict:
            # For prediction storage, only stored in validation if mode on
            y = y.to('cpu')
            outputs = scores.to('cpu')
            predictions = torch.sigmoid(scores).to('cpu')
            self.val_targets = torch.cat((self.val_targets, y))
            self.val_outputs = torch.cat((self.val_outputs, outputs))
            self.val_predictions = torch.cat((self.val_predictions, predictions))

        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self.step(batch, batch_idx)
        accuracy = self.accuracy(torch.sigmoid(scores), y)
        self.log_dict({"test_loss": loss, "test_accuracy": accuracy},
                      prog_bar=True, sync_dist=True, batch_size=batch.batch[-1] + 1)

        # For prediction storage
        y = y.to('cpu')
        outputs = scores.to('cpu')
        predictions = torch.sigmoid(scores).to('cpu')
        self.test_targets = torch.cat((self.test_targets, y))
        self.test_outputs = torch.cat((self.test_outputs, outputs))
        self.test_predictions = torch.cat((self.test_predictions, predictions))

        return loss

    def predict_step(self, batch, batch_idx):
        __, scores, __ = self.step(self, batch, batch_idx)
        preds = torch.round(torch.sigmoid(scores))
        return preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


##############################################################

class LightningClassPlain(LightningClass):
    """
    Extension of Lightning class, specification of step function for plain data (for MLP)
    """

    def __init__(self, store_predict=False):
        super(LightningClassPlain, self).__init__(store_predict)

    def step(self, batch, batch_idx):
        x, y = batch.x, batch.y
        x = x.reshape(y.shape[0], -1, x.shape[-1])
        scores = self.forward(x).flatten()
        loss = self.loss_fn(scores, y)
        return loss, scores, y


##############################################################

class LightningClassGraph(LightningClass):
    """
    Extension of Lightning class, specification of step function for graph data (for GCN, GAT)
    """

    def __init__(self, store_predict=False):
        super(LightningClassGraph, self).__init__(store_predict)

    def step(self, batch, batch_idx):
        batch.y = batch.y.float()
        x, y, edges, batch_assign = batch.x, batch.y, batch.edge_index, batch.batch

        if hasattr(batch, 'coarsening_idx'):
            coarsening_idx = batch.coarsening_idx
            scores = self.forward(x, edges, batch_assign, coarsening_idx).flatten()
        else:
            scores = self.forward(x, edges, batch_assign).flatten()

        loss = self.loss_fn(scores, y)
        return loss, scores, y


###############################################MODELS#######################################################
class MLP(LightningClassPlain):
    """
    Multilayer Perceptron for binary classification with same width in all layers, at least 1 hidden layer,
     baseline model for benchmarking
    """

    def __init__(
            self,
            number_input_features: int,
            number_hidden_layers: int,
            width=64,
            learning_rate=0.001,
            weight_decay=0,
            dropout_rate=0,
            residual=False,
            dense=False,
            width_hidden_reduced_dense=True,
            store_predict=False
    ):
        """
        Args:
            number_input_features: number of input features, i.e. number of nodes * number of features per node
            number_hidden_layers: number of hidden layers (=depth)
            width: width of the network (=number of neurons per layer), same for all layers
            learning_rate: learning rate of optimizer
            weight_decay: weights decay for optimizer
            dropout_rate: dropout rate during training for better generalization
            residual: activates residual skip connections (over one layer)
            dense: activates dense skip connections between all consecutive layers
            width_hidden_reduced_dense: if activated, reduces the width in hidden layers in dense net
        """

        super().__init__(store_predict)
        # Hyperparameter
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = nn.Dropout(p=dropout_rate)
        self.residual = residual
        self.dense = dense

        if width_hidden_reduced_dense:
            width_hidden = int(width / 2)
        else:
            width_hidden = width

        # Model & train functions
        self.activation = nn.ReLU()
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Metrics
        self.accuracy = torchmetrics.classification.BinaryAccuracy()

        # Check if number of hidden layers is at least 1
        if number_hidden_layers < 1:
            raise ValueError(
                "Invalid number of hidden layers. The MLP has at least one hidden layer."
            )

        # Check that only res or dense mode is activated
        if residual and dense:
            raise ValueError("Residual and dense mode are activated, which does not make sense."
                             " Please overthink tis choice. Maybe you meant: residual=False, dense=True ?")

        # Model architecture
        ############### First Layer ###############
        self.linear_first = nn.Linear(number_input_features, width)

        ############### Following hidden layers ###############
        self.hidden_layers = nn.ModuleList()

        # for dense layers: networks width grows with depth!
        if dense:
            for idx in range(number_hidden_layers - 1):
                self.hidden_layers.append(nn.Linear(width + idx * width_hidden, width_hidden))

        else:
            for _ in range(number_hidden_layers - 1):
                self.hidden_layers.append(nn.Linear(width, width))

        ############### Final layer ###############
        if dense:
            self.linear_last = nn.Linear(width + (number_hidden_layers - 1) * width_hidden, 1)

        else:
            self.linear_last = nn.Linear(width, 1)

        # Initialization (weights Xavier, bias zero)
        self.linear_first.apply(self.init_weights_and_biases)
        self.hidden_layers.apply(self.init_weights_and_biases)
        self.linear_last.apply(self.init_weights_and_biases)

        self.save_hyperparameters()

    def forward(self, x):
        # Flatten x over data dimension (batch dimension 0)
        x = x.flatten(start_dim=-2, end_dim=-1)

        x = self.linear_first(x)
        x = self.activation(x)
        x = self.dropout(x)

        for layer in self.hidden_layers:
            if self.residual:
                res = x
                x = layer(x)
                x = self.activation(x)
                x = self.dropout(x)
                x = x + res

            elif self.dense:
                res = x
                x = layer(x)
                x = self.activation(x)
                x = self.dropout(x)
                x = torch.cat((x, res), dim=-1)

            else:
                x = layer(x)
                x = self.activation(x)
                x = self.dropout(x)

        x = self.linear_last(x)
        return x

    def init_weights_and_biases(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)


#############################################################

class GCN(LightningClassGraph):
    """
    Graph Convolutional Network for binary classification with same width (number of channels) in all layers,
     at least 1 hidden (=GCN) layer
    """

    def __init__(
            self,
            number_input_channels: int,
            number_hidden_layers: int,
            width=64,
            output_dim=64,
            learning_rate=0.001,
            weight_decay=0,
            dropout_rate=0,
            residual=False,
            dense=False,
            coarsening=False,
            width_hidden_reduced_dense=True,
            store_predict=False
    ):
        """
        Args:
            number_input_channels: number of input channels, i.e. number of features per node
            number_hidden_layers: number of hidden GCN layers (=depth)
            width: width of layers, i.e. number of features learned per node
            output_dim: output dimension of last GCN layer, relevant for Light models
            learning_rate: learning rate of optimizer
            weight_decay: weights decay for optimizer
            dropout_rate: dropout rate during training for better generalization
            residual: activates residual skip connections (over one layer)
            dense: activates dense skip connections between all consecutive layers
            coarsening: activates that a coarsened version of the graph is used
            width_hidden_reduced_dense: if activated, reduces the width in hidden layers in dense net
        """

        super().__init__(store_predict)
        # Hyperparameter
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = nn.Dropout(p=dropout_rate)
        self.residual = residual
        self.dense = dense
        self.coarsening = coarsening

        if width_hidden_reduced_dense:
            width_hidden = int(width / 2)
            if dense:
                output_dim = width_hidden

        else:
            width_hidden = width

        # Model & train functions
        self.activation = nn.ReLU()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.pooling = tg.nn.pool.global_mean_pool  #for readout
        self.max_pooling = tg.nn.pool.max_pool      #for coarsening

        # Metrics
        self.accuracy = torchmetrics.classification.BinaryAccuracy()

        # Check if number of hidden layers is at least 1
        if number_hidden_layers < 1:
            raise ValueError(
                "Invalid number of hidden layers. The MLP has at least one hidden layer."
            )

        # Check that only res or dense mode is activated
        if residual and dense:
            raise ValueError("Residual and dense mode are activated, which does not make sense."
                             " Please overthink this choice. Maybe you meant: residual=False, dense=True ?")

        if residual and coarsening:
            raise ValueError("Residual and coarsening are activated, which does not make sense/ is not implemented yet."
                             "Please overthink this choice. Maybe you meant: residual=False, coarsening=True ?")

        if dense and coarsening:
            raise ValueError("Dense and coarsening are activated, which does not make sense/ is not implemented yet."
                             "Please overthink this choice. Maybe you meant: dense=False, coarsening=True ?")

        if  coarsening and number_hidden_layers!=2:
            raise ValueError("Coarsening is currently only implemented with exactly 2 hidden layers (as precomputed, fixed coarsening is used)."
                             "Please overthink this choice. Maybe you meant: number_hidden_layer=2, coarsening=True ?")


        # Model architecture
        ############### GCN layers ###############
        self.first_layer = GCNConv(in_channels=number_input_channels, out_channels=width, cached=False, normalize=True,
                                   add_self_loops=True, bias=True)

        self.hidden_layers = nn.ModuleList()
        if dense and number_hidden_layers >= 2:
            for idx in range(number_hidden_layers - 2):
                self.hidden_layers.append(
                    GCNConv(in_channels=width + idx * width_hidden, out_channels=width_hidden, cached=False,
                            normalize=True, add_self_loops=True, bias=True))
            self.hidden_layers.append(
                GCNConv(in_channels=width + (idx+1) * width_hidden, out_channels=output_dim, cached=False,
                        normalize=True, add_self_loops=True, bias=True))

        elif number_hidden_layers >= 2:
            for _ in range(number_hidden_layers - 2):
                self.hidden_layers.append(
                    GCNConv(in_channels=width, out_channels=width, cached=False, normalize=True, add_self_loops=True,
                            bias=True))
            self.hidden_layers.append(
                GCNConv(in_channels=width, out_channels=output_dim, cached=False, normalize=True, add_self_loops=True,
                        bias=True))

        ############### Final layer ###############
        if dense:
            self.linear_last = nn.Linear(width + (number_hidden_layers - 2) * width_hidden + output_dim, 1)

        else:
            self.linear_last = nn.Linear(output_dim, 1)

        # Initialization (weights Xavier, bias zero), not needed for GCNConv (Xavier and zeros init already in GCNConv)
        self.linear_last.apply(self.init_weights_and_biases)

        self.save_hyperparameters()

    def forward(self, x, edges, batch_assign, coarsening_idx=None):
        x = self.first_layer(x, edges)
        x = self.activation(x)
        x = self.dropout(x)

        # direct coarsening if needed
        if self.coarsening:
            coarse_counter = 0
            cluster = coarsening_idx[coarse_counter]

            #Offset for batch matching
            if batch_assign is not None:
                cluster_offset = cluster + (batch_assign*(cluster.max() + 1))
            else:
                cluster_offset = cluster

            coarsed_graph = self.max_pooling(cluster_offset, Data(x=x, edge_index=edges, batch=batch_assign))
            x, edges, batch_assign = coarsed_graph.x, coarsed_graph.edge_index, coarsed_graph.batch

            coarse_counter +=1

        for layer in self.hidden_layers:
            if self.residual:
                res = x
                x = layer(x, edges)
                x = self.activation(x)
                x = self.dropout(x)
                x = x + res

            elif self.dense:
                res = x
                x = layer(x, edges)
                x = self.activation(x)
                x = self.dropout(x)
                x = torch.cat((x, res), dim=1)

            else:
                x = layer(x, edges)
                x = self.activation(x)
                x = self.dropout(x)

                if self.coarsening:
                    cluster = coarsening_idx[coarse_counter]
                    # Offset for batch matching
                    if batch_assign is not None:
                        cluster_offset = cluster + (batch_assign * (cluster.max() + 1))
                    else:
                        cluster_offset = cluster

                    coarsed_graph = self.max_pooling(cluster_offset, Data(x=x, edge_index=edges, batch=batch_assign))
                    x, edges, batch_assign = coarsed_graph.x, coarsed_graph.edge_index, coarsed_graph.batch

                    coarse_counter += 1


        x = self.pooling(x, batch_assign)
        x = self.dropout(x)
        x = self.linear_last(x)
        return x

    def init_weights_and_biases(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

########################################################
class GAT(LightningClassGraph):
    """
    Graph Convolutional Network for binary classification with same width (number of channels) in all layers,
     at least 1 hidden (=GCN with attention) layer
    """

    def __init__(
            self,
            number_input_channels: int,
            number_hidden_layers: int,
            heads=8,
            width=8,
            output_dim=8,
            learning_rate=0.001,
            weight_decay=0,
            dropout_rate_normal=0,
            dropout_rate_attention=0,
            residual=False,
            dense=False,
            coarsening=False,
            width_hidden_reduced_dense=True,
            store_predict=False
    ):
        """
        Args:
            number_input_channels: number of input channels, i.e. number of nodes * number of features per node
            number_hidden_layers: number of hidden GAT layers (=depth)
            heads: number of attention heads within attention mechanism
            width: width of layers, i.e. number of features learned per node
            output_dim: output dimension of last GAT layer, relevant for Light models
            learning_rate: learning rate of optimizer
            weight_decay: weights decay for optimizer
            dropout_rate_normal: dropout rate during training for better generalization
            dropout_rate_attention: dropout rate for attention heads
            residual: activates residual skip connections (over one layer)
            dense: activates dense skip connections between all consecutive layers
            coarsening: activates that a coarsened version of the graph is used
            width_hidden_reduced_dense: if activated, reduces the width in hidden layers in dense net
        """

        super().__init__(store_predict)
        # Hyperparameter
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = nn.Dropout(p=dropout_rate_normal)
        self.residual = residual
        self.dense = dense
        self.coarsening = coarsening

        if width_hidden_reduced_dense:
            width_hidden = int(width / 2)
            if dense:
                output_dim = width_hidden

        else:
            width_hidden = width

        # Model & train functions
        self.activation = nn.ELU()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.pooling = tg.nn.pool.global_mean_pool  #for readout
        self.max_pooling = tg.nn.pool.max_pool      #for coarsening

        # Metrics
        self.accuracy = torchmetrics.classification.BinaryAccuracy()

        # Check if number of hidden layers is at least 1
        if number_hidden_layers < 1:
            raise ValueError(
                "Invalid number of hidden layers. The MLP has at least one hidden layer."
            )

        # Check that only res or dense mode is activated
        if residual and dense:
            raise ValueError("Residual and dense mode are activated, which does not make sense."
                             " Please overthink tis choice. Maybe you meant: residual=False, dense=True ?")

        if residual and coarsening:
            raise ValueError("Residual and coarsening are activated, which does not make sense/ is not implemented yet."
                             "Please overthink this choice. Maybe you meant: residual=False, coarsening=True ?")

        if dense and coarsening:
            raise ValueError("Dense and coarsening are activated, which does not make sense/ is not implemented yet."
                             "Please overthink this choice. Maybe you meant: dense=False, coarsening=True ?")

        if  coarsening and number_hidden_layers!=2:
            raise ValueError("Coarsening is currently only implemented with exactly 2 hidden layers (as precomputed, fixed coarsening is used)."
                             "Please overthink this choice. Maybe you meant: number_hidden_layer=2, coarsening=True ?")


        # Model architecture
        ############### GAT layers ###############
        self.first_layer = GATConv(in_channels=number_input_channels, out_channels=width, heads=heads,
                                   dropout=dropout_rate_attention, add_self_loops=True, bias=True)

        self.hidden_layers = nn.ModuleList()

        if dense:
            for idx in range(number_hidden_layers - 2):
                self.hidden_layers.append(
                    GATConv(in_channels=(width + idx * width_hidden) * heads, out_channels=width_hidden, heads=heads,
                            dropout=dropout_rate_attention, add_self_loops=True, bias=True))

            self.hidden_layers.append(
                GATConv(in_channels=(width + (idx+1) * width_hidden) * heads, out_channels=output_dim, heads=heads,
                        dropout=dropout_rate_attention, add_self_loops=True, bias=True))

            # ALTERNATIVE:
            # add GAT with averaging over heads instead of concatenating
            # self.hidden_layers.append(
            #   GATConv(in_channels=(width + idx * width_hidden) * heads, out_channels=width_hidden, heads=heads,
            #            dropout=dropout_rate_attention, add_self_loops=True, bias=True, concat=False))

        else:
            for _ in range(number_hidden_layers - 2):
                self.hidden_layers.append(
                    GATConv(in_channels=width * heads, out_channels=width, heads=heads, dropout=dropout_rate_attention,
                            add_self_loops=True, bias=True))

            self.hidden_layers.append(
                GATConv(in_channels=width * heads, out_channels=output_dim, heads=heads, dropout=dropout_rate_attention,
                        add_self_loops=True, bias=True))

            # ALTERNATIVE_
            # add GAT with averaging over heads instead of concatenating
            # self.hidden_layers.append(
            #    GATConv(in_channels=width * heads, out_channels=width, heads=heads, dropout=dropout_rate_attention,
            #            add_self_loops=True, bias=True, concat=False))


        ############### Final layer ###############
        if dense:
            self.linear_last = nn.Linear((width + (number_hidden_layers - 2) * width_hidden + output_dim) * heads, 1)

        else:
            self.linear_last = nn.Linear(output_dim * heads, 1)

        # Initialization (weights Xavier, bias zero), not needed for GATConv (Xavier and zeros init already in GCNConv)
        self.linear_last.apply(self.init_weights_and_biases)

        self.save_hyperparameters()

    def forward(self, x, edges, batch_assign, coarsening_idx=None):

        x = self.first_layer(x, edges)
        x = self.activation(x)
        x = self.dropout(x)

        # direct coarsening if needed
        if self.coarsening:
            coarse_counter = 0
            cluster = coarsening_idx[coarse_counter]

            # Offset for batch matching
            if batch_assign is not None:
                cluster_offset = cluster + (batch_assign * (cluster.max() + 1))
            else:
                cluster_offset = cluster

            coarsed_graph = self.max_pooling(cluster_offset, Data(x=x, edge_index=edges, batch=batch_assign))
            x, edges, batch_assign = coarsed_graph.x, coarsed_graph.edge_index, coarsed_graph.batch

            coarse_counter += 1

        for layer in self.hidden_layers:
            if self.residual:
                res = x
                x = layer(x, edges)
                x = self.activation(x)
                x = self.dropout(x)
                x = x + res

            elif self.dense:
                res = x
                x = layer(x, edges)
                x = self.activation(x)
                x = self.dropout(x)
                x = torch.cat((x, res), dim=1)

            else:
                x = layer(x, edges)
                x = self.activation(x)
                x = self.dropout(x)

                if self.coarsening:
                    cluster = coarsening_idx[coarse_counter]
                    # Offset for batch matching
                    if batch_assign is not None:
                        cluster_offset = cluster + (batch_assign * (cluster.max() + 1))
                    else:
                        cluster_offset = cluster

                    coarsed_graph = self.max_pooling(cluster_offset, Data(x=x, edge_index=edges, batch=batch_assign))
                    x, edges, batch_assign = coarsed_graph.x, coarsed_graph.edge_index, coarsed_graph.batch

                    coarse_counter += 1

        x = self.pooling(x, batch_assign)
        x = self.dropout(x)
        x = self.linear_last(x)
        return x

    def init_weights_and_biases(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)


#############################################################################
class GCN2MLP(LightningClassGraph):
    """
    Graph Convolutional Network for binary classification with same width (number of channels) in all layers,
     at least 1 hidden (=GCN) layer; no global average pooling, but final MLP layer at the end
    """

    def __init__(
            self,
            number_input_channels: int,
            number_output_nodes: int,
            number_hidden_layers: int,
            width=64,
            output_dim=64,
            learning_rate=0.001,
            weight_decay=0,
            dropout_rate=0,
            residual=False,
            dense=False,
            coarsening=False,
            width_hidden_reduced_dense=True,
            store_predict=False
    ):
        """
        Args:
            number_input_channels: number of input channels, i.e. number of features per node
            number_output_nodes: number of nodes for output (normally same as input, for coarsening different)
            number_hidden_layers: number of hidden GCN layers (=depth)
            width: width of layers, i.e. number of features learned per node
            output_dim: output dimension of last GCN layer, relevant for Light models
            learning_rate: learning rate of optimizer
            weight_decay: weights decay for optimizer
            dropout_rate: dropout rate during training for better generalization
            residual: activates residual skip connections (over one layer)
            dense: activates dense skip connections between all consecutive layers
            coarsening: activates that a coarsened version of the graph is used
            width_hidden_reduced_dense: if activated, reduces the width in hidden layers in dense net
        """

        super().__init__(store_predict)
        # Hyperparameter
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = nn.Dropout(p=dropout_rate)
        self.residual = residual
        self.dense = dense
        self.coarsening = coarsening

        if width_hidden_reduced_dense:
            width_hidden = int(width / 2)
            if dense:
                output_dim = width_hidden

        else:
            width_hidden = width

        # Model & train functions
        self.activation = nn.ReLU()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.max_pooling = tg.nn.pool.max_pool      #for coarsening

        # Metrics
        self.accuracy = torchmetrics.classification.BinaryAccuracy()

        # Check if number of hidden layers is at least 1
        if number_hidden_layers < 1:
            raise ValueError(
                "Invalid number of hidden layers. The MLP has at least one hidden layer."
            )

        # Check that only res or dense mode is activated
        if residual and dense:
            raise ValueError("Residual and dense mode are activated, which does not make sense."
                             " Please overthink tis choice. Maybe you meant: residual=False, dense=True ?")

        if residual and coarsening:
            raise ValueError("Residual and coarsening are activated, which does not make sense/ is not implemented yet."
                             "Please overthink this choice. Maybe you meant: residual=False, coarsening=True ?")

        if dense and coarsening:
            raise ValueError("Dense and coarsening are activated, which does not make sense/ is not implemented yet."
                             "Please overthink this choice. Maybe you meant: dense=False, coarsening=True ?")

        if  coarsening and number_hidden_layers!=2:
            raise ValueError("Coarsening is currently only implemented with exactly 2 hidden layers (as precomputed, fixed coarsening is used)."
                             "Please overthink this choice. Maybe you meant: number_hidden_layer=2, coarsening=True ?")

        # Model architecture
        ############### GCN layers ###############
        self.first_layer = GCNConv(in_channels=number_input_channels, out_channels=width, cached=False, normalize=True,
                                   add_self_loops=True, bias=True)

        self.hidden_layers = nn.ModuleList()
        if dense:
            for idx in range(number_hidden_layers - 2):
                self.hidden_layers.append(
                    GCNConv(in_channels=width + idx * width_hidden, out_channels=width_hidden, cached=False,
                            normalize=True, add_self_loops=True, bias=True))
            self.hidden_layers.append(
                GCNConv(in_channels=width + (idx+1) * width_hidden, out_channels=output_dim, cached=False,
                        normalize=True, add_self_loops=True, bias=True))

        else:
            for _ in range(number_hidden_layers - 2):
                self.hidden_layers.append(
                    GCNConv(in_channels=width, out_channels=width, cached=False, normalize=True, add_self_loops=True,
                            bias=True))
            self.hidden_layers.append(
                GCNConv(in_channels=width, out_channels=output_dim, cached=False, normalize=True, add_self_loops=True,
                        bias=True))


        ############### Final layer ###############
        if dense:
            self.linear_last = nn.Linear((width + (number_hidden_layers - 2) * width_hidden + output_dim)*number_output_nodes, 1)

        else:
            self.linear_last = nn.Linear(output_dim*number_output_nodes, 1)

        # Initialization (weights Xavier, bias zero), not needed for GCNConv (Xavier and zeros init already in GCNConv)
        self.linear_last.apply(self.init_weights_and_biases)

        self.save_hyperparameters()

    def forward(self, x, edges, batch_assign,  coarsening_idx=None):
        x = self.first_layer(x, edges)
        x = self.activation(x)
        x = self.dropout(x)

        # direct coarsening if needed
        if self.coarsening:
            coarse_counter = 0
            cluster = coarsening_idx[coarse_counter]

            # Offset for batch matching
            if batch_assign is not None:
                cluster_offset = cluster + (batch_assign * (cluster.max() + 1))
            else:
                cluster_offset = cluster

            coarsed_graph = self.max_pooling(cluster_offset, Data(x=x, edge_index=edges, batch=batch_assign))
            x, edges, batch_assign = coarsed_graph.x, coarsed_graph.edge_index, coarsed_graph.batch

            coarse_counter += 1

        for layer in self.hidden_layers:
            if self.residual:
                res = x
                x = layer(x, edges)
                x = self.activation(x)
                x = self.dropout(x)
                x = x + res

            elif self.dense:
                res = x
                x = layer(x, edges)
                x = self.activation(x)
                x = self.dropout(x)
                x = torch.cat((x, res), dim=1)

            else:
                x = layer(x, edges)
                x = self.activation(x)
                x = self.dropout(x)

                if self.coarsening:
                    cluster = coarsening_idx[coarse_counter]
                    # Offset for batch matching
                    if batch_assign is not None:
                        cluster_offset = cluster + (batch_assign * (cluster.max() + 1))
                    else:
                        cluster_offset = cluster

                    coarsed_graph = self.max_pooling(cluster_offset, Data(x=x, edge_index=edges, batch=batch_assign))
                    x, edges, batch_assign = coarsed_graph.x, coarsed_graph.edge_index, coarsed_graph.batch

                    coarse_counter += 1

        x = x.reshape((-1, self.linear_last.in_features))
        x = self.linear_last(x)
        return x

    def init_weights_and_biases(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)


###################################################################################
class GAT2MLP(LightningClassGraph):
    """
    Graph Convolutional Network for binary classification with same width (number of channels) in all layers,
     at least 1 hidden (=GCN with attention) layer
    """

    def __init__(
            self,
            number_input_channels: int,
            number_output_nodes: int,
            number_hidden_layers: int,
            heads=8,
            width=8,
            output_dim=8,
            learning_rate=0.001,
            weight_decay=0,
            dropout_rate_normal=0,
            dropout_rate_attention=0,
            residual=False,
            dense=False,
            coarsening=False,
            width_hidden_reduced_dense=True,
            store_predict=False
    ):
        """
        Args:
            number_input_channels: number of input channels, i.e. number of nodes * number of features per node
            number_output_nodes: number of nodes for output (normally same as input, for coarsening different)
            number_hidden_layers: number of hidden GAT layers (=depth)
            heads: number of attention heads within attention mechanism
            width: width of layers, i.e. number of features learned per node
            output_dim: output dimension of last GCN layer, relevant for Light models
            learning_rate: learning rate of optimizer
            weight_decay: weights decay for optimizer
            dropout_rate_normal: dropout rate during training for better generalization
            dropout_rate_attention: dropout rate for attention heads
            residual: activates residual skip connections (over one layer)
            dense: activates dense skip connections between all consecutive layers
            coarsening: activates that a coarsened version of the graph is used
            width_hidden_reduced_dense: if activated, reduces the width in hidden layers in dense net
        """

        super().__init__(store_predict)
        # Hyperparameter
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = nn.Dropout(p=dropout_rate_normal)
        self.residual = residual
        self.dense = dense
        self.coarsening = coarsening

        if width_hidden_reduced_dense:
            width_hidden = int(width / 2)
            if dense:
                output_dim = width_hidden

        else:
            width_hidden = width

        # Model & train functions
        self.activation = nn.ELU()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.max_pooling = tg.nn.pool.max_pool      #for coarsening

        # Metrics
        self.accuracy = torchmetrics.classification.BinaryAccuracy()

        # Check if number of hidden layers is at least 1
        if number_hidden_layers < 1:
            raise ValueError(
                "Invalid number of hidden layers. The MLP has at least one hidden layer."
            )

        # Check that only res or dense mode is activated
        if residual and dense:
            raise ValueError("Residual and dense mode are activated, which does not make sense."
                             " Please overthink tis choice. Maybe you meant: residual=False, dense=True ?")

        if residual and coarsening:
            raise ValueError("Residual and coarsening are activated, which does not make sense/ is not implemented yet."
                             "Please overthink this choice. Maybe you meant: residual=False, coarsening=True ?")

        if dense and coarsening:
            raise ValueError("Dense and coarsening are activated, which does not make sense/ is not implemented yet."
                             "Please overthink this choice. Maybe you meant: dense=False, coarsening=True ?")

        if  coarsening and number_hidden_layers!=2:
            raise ValueError("Coarsening is currently only implemented with exactly 2 hidden layers (as precomputed, fixed coarsening is used)."
                             "Please overthink this choice. Maybe you meant: number_hidden_layer=2, coarsening=True ?")


        # Model architecture
        ############### GAT layers ###############
        self.first_layer = GATConv(in_channels=number_input_channels, out_channels=width, heads=heads,
                                   dropout=dropout_rate_attention, add_self_loops=True, bias=True)

        self.hidden_layers = nn.ModuleList()

        if dense:
            for idx in range(number_hidden_layers - 2): #before: error - up to number_hidden_layers - 1
                self.hidden_layers.append(
                    GATConv(in_channels=(width + idx * width_hidden) * heads, out_channels=width_hidden, heads=heads,
                            dropout=dropout_rate_attention, add_self_loops=True, bias=True))

            self.hidden_layers.append(
                GATConv(in_channels=(width + (idx+1) * width_hidden) * heads, out_channels=output_dim, heads=heads,
                        dropout=dropout_rate_attention, add_self_loops=True, bias=True))

            # ALTERNATIVE:
            # add GAT with averaging over heads instead of concatenating
            # self.hidden_layers.append(
            #   GATConv(in_channels=(width + idx * width_hidden) * heads, out_channels=width_hidden, heads=heads,
            #            dropout=dropout_rate_attention, add_self_loops=True, bias=True, concat=False))

        else:
            for _ in range(number_hidden_layers - 2):
                self.hidden_layers.append(
                    GATConv(in_channels=width * heads, out_channels=width, heads=heads, dropout=dropout_rate_attention,
                            add_self_loops=True, bias=True))

            self.hidden_layers.append(
                GATConv(in_channels=width * heads, out_channels=output_dim, heads=heads, dropout=dropout_rate_attention,
                        add_self_loops=True, bias=True))

            # ALTERNATIVE_
            # add GAT with averaging over heads instead of concatenating
            # self.hidden_layers.append(
            #    GATConv(in_channels=width * heads, out_channels=width, heads=heads, dropout=dropout_rate_attention,
            #            add_self_loops=True, bias=True, concat=False))


        ############### Final layer ###############
        if dense:
            self.linear_last = nn.Linear((width + (number_hidden_layers - 2) * width_hidden + output_dim) * heads * number_output_nodes, 1)

        else:
            self.linear_last = nn.Linear(output_dim * heads * number_output_nodes, 1)

        # Initialization (weights Xavier, bias zero), not needed for GATConv (Xavier and zeros init already in GCNConv)
        self.linear_last.apply(self.init_weights_and_biases)

        self.save_hyperparameters()

    def forward(self, x, edges, batch_assign, coarsening_idx=None):

        x = self.first_layer(x, edges)
        x = self.activation(x)
        x = self.dropout(x)

        # direct coarsening if needed
        if self.coarsening:
            coarse_counter = 0
            cluster = coarsening_idx[coarse_counter]

            # Offset for batch matching
            if batch_assign is not None:
                cluster_offset = cluster + (batch_assign * (cluster.max() + 1))
            else:
                cluster_offset = cluster

            coarsed_graph = self.max_pooling(cluster_offset, Data(x=x, edge_index=edges, batch=batch_assign))
            x, edges, batch_assign = coarsed_graph.x, coarsed_graph.edge_index, coarsed_graph.batch

            coarse_counter += 1

        for layer in self.hidden_layers:
            if self.residual:
                res = x
                x = layer(x, edges)
                x = self.activation(x)
                x = self.dropout(x)
                x = x + res

            elif self.dense:
                res = x
                x = layer(x, edges)
                x = self.activation(x)
                x = self.dropout(x)
                x = torch.cat((x, res), dim=1)

            else:
                x = layer(x, edges)
                x = self.activation(x)
                x = self.dropout(x)

                if self.coarsening:
                    cluster = coarsening_idx[coarse_counter]
                    # Offset for batch matching
                    if batch_assign is not None:
                        cluster_offset = cluster + (batch_assign * (cluster.max() + 1))
                    else:
                        cluster_offset = cluster

                    coarsed_graph = self.max_pooling(cluster_offset, Data(x=x, edge_index=edges, batch=batch_assign))
                    x, edges, batch_assign = coarsed_graph.x, coarsed_graph.edge_index, coarsed_graph.batch

                    coarse_counter += 1


        x = x.reshape((-1, self.linear_last.in_features))
        x = self.linear_last(x)
        return x

    def init_weights_and_biases(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)


###################################################################################
class ChebNet(LightningClassGraph):
    """
    Graph Convolutional Network for binary classification with same width (number of channels) in all layers,
     at least 1 hidden (=GCN) layer
    """

    def __init__(
            self,
            number_input_channels: int,
            number_hidden_layers: int,
            width=64,
            output_dim=64,
            K=8,
            learning_rate=0.001,
            weight_decay=0,
            dropout_rate=0,
            residual=False,
            dense=False,
            coarsening=False,
            width_hidden_reduced_dense=True,
            store_predict=False
    ):
        """
        Args:
            number_input_channels: number of input channels, i.e. number of features per node
            number_hidden_layers: number of hidden GCN layers (=depth)
            width: width of layers, i.e. number of features learned per node
            output_dim: output dimension of last GCN layer, relevant for Light models
            K: Chebyshev filter size, i.e. size of hop-neighbourhood integrated
            learning_rate: learning rate of optimizer
            weight_decay: weights decay for optimizer
            dropout_rate: dropout rate during training for better generalization
            residual: activates residual skip connections (over one layer)
            dense: activates dense skip connections between all consecutive layers
            width_hidden_reduced_dense: if activated, reduces the width in hidden layers in dense net
        """

        super().__init__(store_predict)
        # Hyperparameter
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = nn.Dropout(p=dropout_rate)
        self.residual = residual
        self.dense = dense
        self.coarsening = coarsening

        if width_hidden_reduced_dense:
            width_hidden = int(width / 2)
            if dense:
                output_dim = width_hidden

        else:
            width_hidden = width

        # Model & train functions
        self.activation = nn.ReLU()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.pooling = tg.nn.pool.global_mean_pool  #for readout
        self.max_pooling = tg.nn.pool.max_pool      #for coarsening

        # Metrics
        self.accuracy = torchmetrics.classification.BinaryAccuracy()

        # Check if number of hidden layers is at least 1
        if number_hidden_layers < 1:
            raise ValueError(
                "Invalid number of hidden layers. The MLP has at least one hidden layer."
            )

        # Check that only res or dense mode is activated
        if residual and dense:
            raise ValueError("Residual and dense mode are activated, which does not make sense."
                             " Please overthink tis choice. Maybe you meant: residual=False, dense=True ?")

        if residual and coarsening:
            raise ValueError("Residual and coarsening are activated, which does not make sense/ is not implemented yet."
                             "Please overthink this choice. Maybe you meant: residual=False, coarsening=True ?")

        if dense and coarsening:
            raise ValueError("Dense and coarsening are activated, which does not make sense/ is not implemented yet."
                             "Please overthink this choice. Maybe you meant: dense=False, coarsening=True ?")

        if  coarsening and number_hidden_layers!=2:
            raise ValueError("Coarsening is currently only implemented with exactly 2 hidden layers (as precomputed, fixed coarsening is used)."
                             "Please overthink this choice. Maybe you meant: number_hidden_layer=2, coarsening=True ?")

        # Model architecture
        ############### GCN layers ###############
        self.first_layer = ChebConv(in_channels=number_input_channels, out_channels=width, K=K, normalization="sym",
                                    bias=True)

        self.hidden_layers = nn.ModuleList()
        if dense:
            for idx in range(number_hidden_layers - 2):
                self.hidden_layers.append(
                    ChebConv(in_channels=width + idx * width_hidden, out_channels=width_hidden, K=K,
                             normalization="sym", bias=True))
            self.hidden_layers.append(
                ChebConv(in_channels=width + (idx+1) * width_hidden, out_channels=output_dim, K=K, normalization="sym",
                         bias=True))

        else:
            for _ in range(number_hidden_layers - 2):
                self.hidden_layers.append(
                    ChebConv(in_channels=width, out_channels=width, K=K, normalization="sym", bias=True))
            self.hidden_layers.append(
                ChebConv(in_channels=width, out_channels=output_dim, K=K, normalization="sym", bias=True))

        ############### Final layer ###############
        if dense:
            self.linear_last = nn.Linear(width + (number_hidden_layers - 2) * width_hidden + output_dim, 1)

        else:
            self.linear_last = nn.Linear(output_dim, 1)

        # Initialization (weights Xavier, bias zero), not needed for GCNConv (Xavier and zeros init already in GCNConv)
        self.linear_last.apply(self.init_weights_and_biases)

        self.save_hyperparameters()

    def forward(self, x, edges, batch_assign, coarsening_idx=None):

        x = self.first_layer(x, edges)
        x = self.activation(x)
        x = self.dropout(x)

        # direct coarsening if needed
        if self.coarsening:
            coarse_counter = 0
            cluster = coarsening_idx[coarse_counter]

            # Offset for batch matching
            if batch_assign is not None:
                cluster_offset = cluster + (batch_assign * (cluster.max() + 1))
            else:
                cluster_offset = cluster

            coarsed_graph = self.max_pooling(cluster_offset, Data(x=x, edge_index=edges, batch=batch_assign))
            x, edges, batch_assign = coarsed_graph.x, coarsed_graph.edge_index, coarsed_graph.batch

            coarse_counter += 1

        for layer in self.hidden_layers:
            if self.residual:
                res = x
                x = layer(x, edges)
                x = self.activation(x)
                x = self.dropout(x)
                x = x + res

            elif self.dense:
                res = x
                x = layer(x, edges)
                x = self.activation(x)
                x = self.dropout(x)
                x = torch.cat((x, res), dim=1)

            else:
                x = layer(x, edges)
                x = self.activation(x)
                x = self.dropout(x)

                if self.coarsening:
                    cluster = coarsening_idx[coarse_counter]
                    # Offset for batch matching
                    if batch_assign is not None:
                        cluster_offset = cluster + (batch_assign * (cluster.max() + 1))
                    else:
                        cluster_offset = cluster

                    coarsed_graph = self.max_pooling(cluster_offset, Data(x=x, edge_index=edges, batch=batch_assign))
                    x, edges, batch_assign = coarsed_graph.x, coarsed_graph.edge_index, coarsed_graph.batch

                    coarse_counter += 1

        x = self.pooling(x, batch_assign)
        x = self.dropout(x)
        x = self.linear_last(x)
        return x

    def init_weights_and_biases(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)


###################################################################################
class ChebNet2MLP(LightningClassGraph):
    """
    Graph Convolutional Network for binary classification with same width (number of channels) in all layers,
     at least 1 hidden (=GCN) layer
    """

    def __init__(
            self,
            number_input_channels: int,
            number_output_nodes: int,
            number_hidden_layers: int,
            width=64,
            output_dim=64,
            K=8,
            learning_rate=0.001,
            weight_decay=0,
            dropout_rate=0,
            residual=False,
            dense=False,
            coarsening=False,
            width_hidden_reduced_dense=True,
            store_predict=False
    ):
        """
        Args:
            number_input_channels: number of input channels, i.e. number of features per node
            number_output_nodes: number of nodes for output (normally same as input, for coarsening different)
            number_hidden_layers: number of hidden GCN layers (=depth)
            width: width of layers, i.e. number of features learned per node
            output_dim: output dimension of last GCN layer, relevant for Light models
            K: Chebyshev filter size, i.e. size of hop-neighbourhood integrated
            learning_rate: learning rate of optimizer
            weight_decay: weights decay for optimizer
            dropout_rate: dropout rate during training for better generalization
            residual: activates residual skip connections (over one layer)
            dense: activates dense skip connections between all consecutive layers
            coarsening: activates that a coarsened version of the graph is used
            width_hidden_reduced_dense: if activated, reduces the width in hidden layers in dense net
        """

        super().__init__(store_predict)
        # Hyperparameter
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = nn.Dropout(p=dropout_rate)
        self.residual = residual
        self.dense = dense
        self.coarsening = coarsening

        if width_hidden_reduced_dense:
            width_hidden = int(width / 2)
            if dense:
                output_dim = width_hidden

        else:
            width_hidden = width

        # Model & train functions
        self.activation = nn.ReLU()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.max_pooling = tg.nn.pool.max_pool      #for coarsening


        # Metrics
        self.accuracy = torchmetrics.classification.BinaryAccuracy()

        # Check if number of hidden layers is at least 1
        if number_hidden_layers < 1:
            raise ValueError(
                "Invalid number of hidden layers. The MLP has at least one hidden layer."
            )

        # Check that only res or dense mode is activated
        if residual and dense:
            raise ValueError("Residual and dense mode are activated, which does not make sense."
                             " Please overthink tis choice. Maybe you meant: residual=False, dense=True ?")

        if residual and coarsening:
            raise ValueError("Residual and coarsening are activated, which does not make sense/ is not implemented yet."
                             "Please overthink this choice. Maybe you meant: residual=False, coarsening=True ?")

        if dense and coarsening:
            raise ValueError("Dense and coarsening are activated, which does not make sense/ is not implemented yet."
                             "Please overthink this choice. Maybe you meant: dense=False, coarsening=True ?")

        if  coarsening and number_hidden_layers!=2:
            raise ValueError("Coarsening is currently only implemented with exactly 2 hidden layers (as precomputed, fixed coarsening is used)."
                             "Please overthink this choice. Maybe you meant: number_hidden_layer=2, coarsening=True ?")


        # Model architecture
        ############### GCN layers ###############
        self.first_layer = ChebConv(in_channels=number_input_channels, out_channels=width, K=K, normalization="sym",
                                    bias=True)

        self.hidden_layers = nn.ModuleList()
        if dense:
            for idx in range(number_hidden_layers - 2):
                self.hidden_layers.append(
                    ChebConv(in_channels=width + idx * width_hidden, out_channels=width_hidden, K=K,
                             normalization="sym", bias=True))
            self.hidden_layers.append(
                ChebConv(in_channels=width + (idx+1) * width_hidden, out_channels=output_dim, K=K, normalization="sym",
                         bias=True))

        else:
            for _ in range(number_hidden_layers - 2):
                self.hidden_layers.append(
                    ChebConv(in_channels=width, out_channels=width, K=K, normalization="sym", bias=True))
            self.hidden_layers.append(
                ChebConv(in_channels=width, out_channels=output_dim, K=K, normalization="sym", bias=True))

        ############### Final layer ###############
        if dense:
            self.linear_last = nn.Linear((width + (number_hidden_layers - 2) * width_hidden + output_dim)*number_output_nodes, 1)

        else:
            self.linear_last = nn.Linear(output_dim*number_output_nodes, 1)

        # Initialization (weights Xavier, bias zero), not needed for GCNConv (Xavier and zeros init already in GCNConv)
        self.linear_last.apply(self.init_weights_and_biases)

        self.save_hyperparameters()


    def forward(self, x, edges, batch_assign,  coarsening_idx=None):

        x = self.first_layer(x, edges)
        x = self.activation(x)
        x = self.dropout(x)

        # direct coarsening if needed
        if self.coarsening:
            coarse_counter = 0
            cluster = coarsening_idx[coarse_counter]

            # Offset for batch matching
            if batch_assign is not None:
                cluster_offset = cluster + (batch_assign * (cluster.max() + 1))
            else:
                cluster_offset = cluster

            coarsed_graph = self.max_pooling(cluster_offset, Data(x=x, edge_index=edges, batch=batch_assign))
            x, edges, batch_assign = coarsed_graph.x, coarsed_graph.edge_index, coarsed_graph.batch

            coarse_counter += 1

        for layer in self.hidden_layers:
            if self.residual:
                res = x
                x = layer(x, edges)
                x = self.activation(x)
                x = self.dropout(x)
                x = x + res

            elif self.dense:
                res = x
                x = layer(x, edges)
                x = self.activation(x)
                x = self.dropout(x)
                x = torch.cat((x, res), dim=1)

            else:
                x = layer(x, edges)
                x = self.activation(x)
                x = self.dropout(x)

                if self.coarsening:
                    cluster = coarsening_idx[coarse_counter]
                    # Offset for batch matching
                    if batch_assign is not None:
                        cluster_offset = cluster + (batch_assign * (cluster.max() + 1))
                    else:
                        cluster_offset = cluster

                    coarsed_graph = self.max_pooling(cluster_offset, Data(x=x, edge_index=edges, batch=batch_assign))
                    x, edges, batch_assign = coarsed_graph.x, coarsed_graph.edge_index, coarsed_graph.batch

                    coarse_counter += 1

        x = x.reshape((-1, self.linear_last.in_features))
        x = self.linear_last(x)
        return x

    def init_weights_and_biases(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)


###################################################################################

def test_model_step(graphs_list_test, coarsening_list_test):

    number_nodes = graphs_list_test[0].x.shape[0]
    input_channels = graphs_list_test[0].x.shape[1]

    print('Number nodes: ', number_nodes)
    print('Number channels: ', input_channels)

    input_dimension = number_nodes*input_channels
    number_hidden = 2
    number_hidden_res = 3

    mlp_model = MLP(input_dimension, number_hidden)
    gcn_model = GCN(input_channels, number_hidden)
    gat_model = GAT(input_channels, number_hidden, heads=2)
    chebnet_model = ChebNet(input_channels, number_hidden)
    res_mlp_model = MLP(input_dimension, number_hidden_res, residual=True)
    res_gcn_model = GCN(input_channels, number_hidden_res, residual=True)
    res_gat_model = GAT(input_channels, number_hidden_res, residual=True)
    res_chebnet_model = ChebNet(input_channels, number_hidden_res, residual=True)
    dense_mlp_model = MLP(input_dimension, number_hidden_res, residual=False, dense=True)
    dense_gcn_model = GCN(input_channels, number_hidden_res, residual=False, dense=True)
    dense_gat_model = GAT(input_channels, number_hidden_res, residual=False, dense=True)
    dense_chebnet_model = ChebNet(input_channels, number_hidden_res, residual=False, dense=True)

    gcn_preprocess_model = GCN2MLP(input_channels, number_nodes, number_hidden)
    res_gcn_preprocess_model = GCN2MLP(input_channels, number_nodes, number_hidden_res, residual=True)
    dense_gcn_preprocess_model = GCN2MLP(input_channels, number_nodes, number_hidden_res, residual=False,
                                               dense=True)
    gat_preprocess_model = GAT2MLP(input_channels, number_nodes, number_hidden, heads=2)
    res_gat_preprocess_model = GAT2MLP(input_channels, number_nodes, number_hidden_res, residual=True, heads=2)
    dense_gat_preprocess_model = GAT2MLP(input_channels, number_nodes, number_hidden_res, residual=False,
                                               dense=True, heads=2)

    chebnet_preprocess_model = ChebNet2MLP(input_channels, number_nodes, number_hidden)
    res_chebnet_preprocess_model = ChebNet2MLP(input_channels, number_nodes, number_hidden_res, residual=True)
    dense_chebnet_preprocess_model = ChebNet2MLP(input_channels, number_nodes, number_hidden_res, residual=False, dense=True)

    input_first_plain = graphs_list_test[0].x
    input_first_graph = [graphs_list_test[0].x, graphs_list_test[0].edge_index]

    #COARSENING
    coarse_index = coarsening_list_test
    number_coarsening_nodes = len(coarsening_list_test[1].unique())

    coarsening_gcn_model = GCN(input_channels, number_hidden, coarsening=True)
    coarsening_gcn_preprocess_model = GCN2MLP(input_channels, number_coarsening_nodes, number_hidden, coarsening=True)
    coarsening_gat_model = GAT(input_channels, number_hidden, heads=2, coarsening=True)
    coarsening_gat_preprocess_model = GAT2MLP(input_channels, number_coarsening_nodes, number_hidden, heads=2, coarsening=True)
    coarsening_chebnet_model = ChebNet(input_channels, number_hidden, coarsening=True)
    coarsening_chebnet_preprocess_model = ChebNet2MLP(input_channels, number_coarsening_nodes, number_hidden, coarsening=True)

    ############ Run MLP once to check if it runs principally ############
    print('MLP RESULTS')
    output = mlp_model(input_first_plain)
    print("Output: ", output)
    print("Prediction: ", torch.sigmoid(output))

    ############ Run GCN once to check if it runs principally ############
    print('GCN RESULTS')
    output = gcn_model(input_first_graph[0], input_first_graph[1], None)
    print("Output: ", output)
    print("Prediction: ", torch.sigmoid(output))

    ############ Run GAT once to check if it runs principally ############
    print('GAT RESULTS')
    output = gat_model(input_first_graph[0], input_first_graph[1], None)
    print("Output: ", output)
    print("Prediction: ", torch.sigmoid(output))

    ############ Run ChebNet once to check if it runs principally ############
    print('ChebNet RESULTS')
    output = chebnet_model(input_first_graph[0], input_first_graph[1], None)
    print("Output: ", output)
    print("Prediction: ", torch.sigmoid(output))

    ############ Run GCNPreprocess once to check if it runs principally ############
    print('GCN2MLP RESULTS')
    output = gcn_preprocess_model(input_first_graph[0], input_first_graph[1], None)
    print("Output: ", output)
    print("Prediction: ", torch.sigmoid(output))

    ############ Run GATPreprocess once to check if it runs principally ############
    print('GAT2MLP RESULTS')
    output = gat_preprocess_model(input_first_graph[0], input_first_graph[1], None)
    print("Output: ", output)
    print("Prediction: ", torch.sigmoid(output))

    ############ Run ChebNetPreprocess once to check if it runs principally ############
    print('ChebNet2MLP RESULTS')
    output = chebnet_preprocess_model(input_first_graph[0], input_first_graph[1], None)
    print("Output: ", output)
    print("Prediction: ", torch.sigmoid(output))

    ############ Run CoarseningGCN once to check if it runs principally ############
    print('CoarseningGCN RESULTS')
    output = coarsening_gcn_model(input_first_graph[0], input_first_graph[1], None, coarse_index )
    print("Output: ", output)
    print("Prediction: ", torch.sigmoid(output))

    ############ Run CoarseningGCN2MLP once to check if it runs principally ############
    print('CoarseningGCN2MLP RESULTS')
    output = coarsening_gcn_preprocess_model(input_first_graph[0], input_first_graph[1], None, coarse_index)
    print("Output: ", output)
    print("Prediction: ", torch.sigmoid(output))

    ############ Run CoarseningGAT once to check if it runs principally ############
    print('CoarseningGAT RESULTS')
    output = coarsening_gat_model(input_first_graph[0], input_first_graph[1], None, coarse_index)
    print("Output: ", output)
    print("Prediction: ", torch.sigmoid(output))

    ############ Run CoarseningGAT2MLP once to check if it runs principally ############
    print('CoarseningGAT2MLP RESULTS')
    output = coarsening_gat_preprocess_model(input_first_graph[0], input_first_graph[1], None, coarse_index)
    print("Output: ", output)
    print("Prediction: ", torch.sigmoid(output))

    ############ Run CoarseningChebNet once to check if it runs principally ############
    print('CoarseningChebNet RESULTS')
    output = coarsening_chebnet_model(input_first_graph[0], input_first_graph[1], None, coarse_index)
    print("Output: ", output)
    print("Prediction: ", torch.sigmoid(output))

    ############ Run CoarseningGCN once to check if it runs principally ############
    print('CoarseningChebNet2MLP RESULTS')
    output = coarsening_chebnet_preprocess_model(input_first_graph[0], input_first_graph[1], None, coarse_index)
    print("Output: ", output)
    print("Prediction: ", torch.sigmoid(output))

    ############ Run ResMLP once to check if it runs principally ############
    print('ResMLP RESULTS')
    output = res_mlp_model(input_first_plain)
    print("Output: ", output)
    print("Prediction: ", torch.sigmoid(output))

    ############ Run ResGCN once to check if it runs principally ############
    print('ResGCN RESULTS')
    output = res_gcn_model(input_first_graph[0], input_first_graph[1], None)
    print("Output: ", output)
    print("Prediction: ", torch.sigmoid(output))

    ############ Run ResGAT once to check if it runs principally ############
    print('ResGAT RESULTS')
    output = res_gat_model(input_first_graph[0], input_first_graph[1], None)
    print("Output: ", output)
    print("Prediction: ", torch.sigmoid(output))

    ############ Run ResChebNet once to check if it runs principally ############
    print('ResChebNet RESULTS')
    output = res_chebnet_model(input_first_graph[0], input_first_graph[1], None)
    print("Output: ", output)
    print("Prediction: ", torch.sigmoid(output))

    ############ Run ResGCNPreprocess once to check if it runs principally ############
    print('ResGCN2MLP RESULTS')
    output = res_gcn_preprocess_model(input_first_graph[0], input_first_graph[1], None)
    print("Output: ", output)
    print("Prediction: ", torch.sigmoid(output))

    ############ Run ResGATPreprocess once to check if it runs principally ############
    print('ResGAT2MLP RESULTS')
    output = res_gat_preprocess_model(input_first_graph[0], input_first_graph[1], None)
    print("Output: ", output)
    print("Prediction: ", torch.sigmoid(output))

    ############ Run ResChebNetPreprocess once to check if it runs principally ############
    print('ResChebNet2MLP RESULTS')
    output = res_chebnet_preprocess_model(input_first_graph[0], input_first_graph[1], None)
    print("Output: ", output)
    print("Prediction: ", torch.sigmoid(output))

    ############ Run DenseMLP once to check if it runs principally ############
    print('DenseMLP RESULTS')
    output = dense_mlp_model(input_first_plain)
    print("Output: ", output)
    print("Prediction: ", torch.sigmoid(output))

    ############ Run DenseGCN once to check if it runs principally ############
    print('DenseGCN RESULTS')
    output = dense_gcn_model(input_first_graph[0], input_first_graph[1], None)
    print("Output: ", output)
    print("Prediction: ", torch.sigmoid(output))

    ############ Run DenseGAT once to check if it runs principally ############
    print('DenseGAT RESULTS')
    output = dense_gat_model(input_first_graph[0], input_first_graph[1], None)
    print("Output: ", output)
    print("Prediction: ", torch.sigmoid(output))

    ############ Run DenseChebNet once to check if it runs principally ############
    print('DenseChebNet RESULTS')
    output = dense_chebnet_model(input_first_graph[0], input_first_graph[1], None)
    print("Output: ", output)
    print("Prediction: ", torch.sigmoid(output))

    ############ Run DenseGCNPreprocess once to check if it runs principally ############
    print('DenseGCN2MLP RESULTS')
    output = dense_gcn_preprocess_model(input_first_graph[0], input_first_graph[1], None)
    print("Output: ", output)
    print("Prediction: ", torch.sigmoid(output))

    ############ Run DenseGATPreprocess once to check if it runs principally ############
    print('DenseGAT2MLP RESULTS')
    output = dense_gat_preprocess_model(input_first_graph[0], input_first_graph[1], None)
    print("Output: ", output)
    print("Prediction: ", torch.sigmoid(output))

    ############ Run DenseChebNet2MLP once to check if it runs principally ############
    print('DenseChebNet2MLP RESULTS')
    output = dense_chebnet_preprocess_model(input_first_graph[0], input_first_graph[1], None)
    print("Output: ", output)
    print("Prediction: ", torch.sigmoid(output))

####################################################################################

def main():
    print('######################')
    print('TEST MODELS ON KIRC DATA')
    print('######################')

    with open("./data/kirc_random_nodes_preprocessed_all.pkl", "rb") as f:
        graphs_list_test = pickle.load(f)

    coarsening_list_test = torch.load("./coarsening/coarsening_0.pt")

    test_model_step(graphs_list_test, coarsening_list_test)

    '''
    ##############################################################################
    print('######################')
    print('TEST MODELS ON BC SUBTYPE DATA')
    print('######################')

    with open("./data/bc_subtypes_normalized.pkl", "rb") as f:
        graphs_list_test = pickle.load(f)

    test_model_step(graphs_list_test)

    ##############################################################################
    
    print('######################')
    print('TEST MODELS ON BC METASTATIC DATA')
    print('######################')

    with open("./data/bc_metastatic_normalized.pkl", "rb") as f:
        graphs_list_test = pickle.load(f)

    test_model_step(graphs_list_test)
    '''

    return


if __name__ == "__main__":
    main()
