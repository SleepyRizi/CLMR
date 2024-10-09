import torch
import torch.nn as nn
import torchmetrics
from copy import deepcopy
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Tuple
from tqdm import tqdm


class LinearEvaluation(LightningModule):
    def __init__(self, args, encoder: nn.Module, hidden_dim: int, output_dim: int):
        super().__init__()
        self.save_hyperparameters(args)

        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        if self.hparams.finetuner_mlp:
            self.model = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        else:
            self.model = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim))
        self.criterion = self.configure_criterion()

        self.accuracy = torchmetrics.Accuracy()
        self.average_precision = torchmetrics.AveragePrecision(pos_label=1)

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        preds = self._forward_representations(x, y)
        loss = self.criterion(preds, y)
        return loss, preds

    def _forward_representations(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Perform a forward pass using either the representations, or the input data (that we still)
        need to extract the represenations from using our encoder.
        """
        if x.shape[-1] == self.hidden_dim:
            h0 = x
        else:
            with torch.no_grad():
                h0 = self.encoder(x)
        return self.model(h0)

    def training_step(self, batch, _) -> Tensor:
        x, y = batch
        loss, preds = self.forward(x, y)

        self.log("Train/accuracy", self.accuracy(preds, y))
        # self.log("Train/pr_auc", self.average_precision(preds, y))
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, batch, _) -> Tensor:
        x, y = batch
        loss, preds = self.forward(x, y)

        self.log("Valid/accuracy", self.accuracy(preds, y))
        # self.log("Valid/pr_auc", self.average_precision(preds, y))
        self.log("Valid/loss", loss)
        return loss

    def configure_criterion(self) -> nn.Module:
        if self.hparams.dataset in ["magnatagatune", "msd"]:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        return criterion

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.finetuner_learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=False,
        )
        if scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "Valid/loss",
            }
        else:
            return {"optimizer": optimizer}

    def extract_representations(self, dataloader: DataLoader) -> Dataset:

        representations = []
        ys = []
        for x, y in tqdm(dataloader):
            with torch.no_grad():
                h0 = self.encoder(x)
                representations.append(h0)
                ys.append(y)

        if len(representations) > 1:
            representations = torch.cat(representations, dim=0)
            ys = torch.cat(ys, dim=0)
        else:
            representations = representations[0]
            ys = ys[0]

        tensor_dataset = TensorDataset(representations, ys)
        return tensor_dataset
#===========================================================================================================================================================

# import torch
# import torch.nn as nn
# import torchmetrics
# from pytorch_lightning import LightningModule
# from torch import Tensor
# from torch.utils.data import DataLoader, Dataset, TensorDataset
# from typing import Tuple
# from tqdm import tqdm


# class LinearEvaluation(LightningModule):
#     def __init__(self, args, encoder: nn.Module, hidden_dim: int, output_dim: int):
#         super().__init__()
#         self.save_hyperparameters(args)

#         self.encoder = encoder
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim

#         #architecture with additional layers
#         self.model = nn.Sequential(
#             nn.Linear(self.hidden_dim, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(256, self.output_dim),
#         )

#         self.criterion = self.configure_criterion()

#         # Updated metrics with dynamic number of classes
#         self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.output_dim)
#         self.average_precision = torchmetrics.AveragePrecision(task='multiclass', num_classes=self.output_dim)

#     def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
#         preds = self._forward_representations(x)
#         loss = self.criterion(preds, y)
#         return loss, preds

#     def _forward_representations(self, x: Tensor) -> Tensor:
#         """
#         Perform a forward pass using either the representations or the input data
#         (which requires extracting representations using the encoder).
#         """
#         if x.shape[-1] == self.hidden_dim:
#             h0 = x
#         else:
#             with torch.no_grad():
#                 h0 = self.encoder(x)
#                 # Flatten h0 if necessary
#                 h0 = h0.view(h0.size(0), -1)

#         # Ensure h0 has the correct shape
#         assert h0.shape[1] == self.hidden_dim, f"Expected h0 to have {self.hidden_dim} features, but got {h0.shape[1]}"

#         return self.model(h0)

#     def training_step(self, batch, _) -> Tensor:
#         x, y = batch
#         loss, preds = self.forward(x, y)

#         self.log("Train/accuracy", self.accuracy(preds, y), prog_bar=True)
#         self.log("Train/loss", loss)
#         return loss

#     def validation_step(self, batch, _) -> Tensor:
#         x, y = batch
#         loss, preds = self.forward(x, y)

#         self.log("Valid/accuracy", self.accuracy(preds, y), prog_bar=True)
#         self.log("Valid/loss", loss)
#         return loss

#     def configure_criterion(self) -> nn.Module:
#         if self.hparams.dataset in ["magnatagatune", "msd"]:
#             criterion = nn.BCEWithLogitsLoss()
#         else:
#             criterion = nn.CrossEntropyLoss()
#         return criterion

#     def configure_optimizers(self) -> dict:
#         optimizer = torch.optim.Adam(
#             self.model.parameters(),
#             lr=self.hparams.finetuner_learning_rate,
#             weight_decay=self.hparams.weight_decay,
#         )
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer,
#             mode="min",
#             factor=0.1,
#             patience=5,
#             threshold=0.0001,
#             threshold_mode="rel",
#             cooldown=0,
#             min_lr=0,
#             eps=1e-08,
#             verbose=False,
#         )
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": scheduler,
#             "monitor": "Valid/loss",
#         }

#     def extract_representations(self, dataloader: DataLoader) -> Dataset:
#         representations = []
#         ys = []
#         for x, y in tqdm(dataloader):
#             with torch.no_grad():
#                 h0 = self.encoder(x)
#                 # Flatten h0 if necessary
#                 h0 = h0.view(h0.size(0), -1)
#                 # Ensure h0 has the correct shape
#                 assert h0.shape[1] == self.hidden_dim, f"Expected h0 to have {self.hidden_dim} features, but got {h0.shape[1]}"

#                 representations.append(h0.cpu())
#                 ys.append(y.cpu())

#         representations = torch.cat(representations, dim=0)
#         ys = torch.cat(ys, dim=0)

#         tensor_dataset = TensorDataset(representations, ys)
#         return tensor_dataset


# =====================================================================================================================================
# import torch
# import torch.nn as nn
# import torchmetrics
# from pytorch_lightning import LightningModule
# from torch import Tensor
# from torch.utils.data import DataLoader, Dataset, TensorDataset
# from typing import Tuple
# from tqdm import tqdm
# import torch.nn.functional as F


# class ResidualBlock(nn.Module):
#     def __init__(self, in_features, out_features, dropout_rate=0.5):
#         super(ResidualBlock, self).__init__()
#         self.fc1 = nn.Linear(in_features, out_features)
#         self.bn1 = nn.BatchNorm1d(out_features)
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.fc2 = nn.Linear(out_features, out_features)
#         self.bn2 = nn.BatchNorm1d(out_features)

#         # Shortcut connection
#         if in_features != out_features:
#             self.shortcut = nn.Sequential(
#                 nn.Linear(in_features, out_features),
#                 nn.BatchNorm1d(out_features)
#             )
#         else:
#             self.shortcut = nn.Identity()

#     def forward(self, x):
#         residual = self.shortcut(x)
#         out = self.fc1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.dropout(out)
#         out = self.fc2(out)
#         out = self.bn2(out)
#         out += residual
#         out = self.relu(out)
#         return out


# class LinearEvaluation(LightningModule):
#     def __init__(self, args, encoder: nn.Module, hidden_dim: int, output_dim: int):
#         super().__init__()
#         self.save_hyperparameters(args)

#         self.encoder = encoder
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim

#         # architecture with mor residual blocks
#         self.model = nn.Sequential(
#             nn.Linear(self.hidden_dim, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             ResidualBlock(1024, 1024, dropout_rate=0.5),
#             ResidualBlock(1024, 512, dropout_rate=0.5),
#             ResidualBlock(512, 256, dropout_rate=0.5),
#             nn.Linear(256, self.output_dim)
#         )

#         self.criterion = self.configure_criterion()

#         # Metrics
#         self.train_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.output_dim)
#         self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.output_dim)

#         self.train_f1 = torchmetrics.F1Score(task='multiclass', num_classes=self.output_dim, average='weighted')
#         self.val_f1 = torchmetrics.F1Score(task='multiclass', num_classes=self.output_dim, average='weighted')

#     def forward(self, x: Tensor) -> Tensor:
#         return self.model(x)

#     def training_step(self, batch, batch_idx) -> Tensor:
#         x, y = batch
#         preds = self(x)
#         loss = self.criterion(preds, y)

#         # Log metrics
#         self.train_accuracy(preds.softmax(dim=-1), y)
#         self.train_f1(preds.softmax(dim=-1), y)
#         self.log("Train/loss", loss, on_step=False, on_epoch=True)
#         self.log("Train/accuracy", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
#         self.log("Train/f1", self.train_f1, on_step=False, on_epoch=True)
#         return loss

#     def validation_step(self, batch, batch_idx) -> Tensor:
#         x, y = batch
#         preds = self(x)
#         loss = self.criterion(preds, y)

#         # Log metrics
#         self.val_accuracy(preds.softmax(dim=-1), y)
#         self.val_f1(preds.softmax(dim=-1), y)
#         self.log("Valid/loss", loss, on_step=False, on_epoch=True)
#         self.log("Valid/accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
#         self.log("Valid/f1", self.val_f1, on_step=False, on_epoch=True)
#         return loss

#     def configure_criterion(self) -> nn.Module:
#         if self.hparams.dataset in ["magnatagatune", "msd"]:
#             criterion = nn.BCEWithLogitsLoss()
#         else:
#             criterion = nn.CrossEntropyLoss()
#         return criterion

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(
#             self.model.parameters(),
#             lr=self.hparams.finetuner_learning_rate,
#             weight_decay=self.hparams.weight_decay,
#         )
#         scheduler = torch.optim.lr_scheduler.OneCycleLR(
#             optimizer,
#             max_lr=self.hparams.finetuner_learning_rate,
#             total_steps=self.trainer.estimated_stepping_batches,
#             anneal_strategy='cos',
#             cycle_momentum=False,
#         )
#         return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

#     def extract_representations(self, dataloader: DataLoader) -> Dataset:
#         representations = []
#         ys = []
#         for x, y in tqdm(dataloader):
#             x = x.to(self.device)
#             y = y.to(self.device)
#             with torch.no_grad():
#                 h0 = self.encoder(x)
#                 # Flatten h0 if necessary
#                 h0 = h0.view(h0.size(0), -1)
#                 # Ensure h0 has the correct shape
#                 assert h0.shape[1] == self.hidden_dim, f"Expected h0 to have {self.hidden_dim} features, but got {h0.shape[1]}"

#                 representations.append(h0.cpu())
#                 ys.append(y.cpu())

#         representations = torch.cat(representations, dim=0)
#         ys = torch.cat(ys, dim=0)

#         tensor_dataset = TensorDataset(representations, ys)
#         return tensor_dataset


