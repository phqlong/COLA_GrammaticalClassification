import wandb
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoModelForSequenceClassification

import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class ColaModel(pl.LightningModule):
    def __init__(self,
        model_checkpoint : str = "google/bert_uncased_L-2_H-128_A-2",
        num_classes : int = 2,
        lr : float = 1e-2
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_classes)
        # When Using AutoModelForSequenceClassification instead of AutoModel, noneed to make classifier head
        # self.classfier = nn.Linear(self.model.config.hidden_size, self.hparams.num_classes)
        self.criterion = nn.CrossEntropyLoss()

        # Initialize metrics
        self.train_acc_metric = Accuracy('binary')
        self.val_acc_metric = Accuracy('binary')
        self.f1_metric = F1Score('binary', num_classes=num_classes)
        
        self.precision_macro_metric = Precision('binary', average="macro", num_classes=num_classes)
        self.recall_macro_metric = Recall('binary', average="macro", num_classes=num_classes)
        
        self.precision_micro_metric = Precision('binary', average="micro")
        self.recall_micro_metric = Recall('binary', average="micro")

    def forward(self, inputs):
        """Return logits from SequenceClassifierOutput object"""
        outputs = self.model(input_ids=inputs['input_ids'], 
                             attention_mask=inputs['attention_mask'])
        # h_cls = outputs.last_hidden_state[:, 0]
        # logits = self.classfier(h_cls)
        return outputs.logits

    def model_step(self, batch):
        labels = batch['label']
        logits = self(batch)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels, logits

    def training_step(self, batch, batch_idx):
        loss, preds, labels, logits = self.model_step(batch)
        acc = self.train_acc_metric(preds, labels)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels, logits = self.model_step(batch)

        # Metrics
        # val_acc = torch.tensor(accuracy_score(preds.cpu(), targets.cpu()))
        acc = self.val_acc_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # Logging metrics
        # Loss should on_step=True
        self.log("val_loss", loss, prog_bar=True, on_step=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_precision_macro", precision_macro, prog_bar=False)
        self.log("val_recall_macro", recall_macro, prog_bar=False)
        self.log("val_precision_micro", precision_micro, prog_bar=False)
        self.log("val_recall_micro", recall_micro, prog_bar=False)
        self.log("val_f1", f1, prog_bar=True)

        # Return value will be aggerated in validation_epoch_end 
        return {"labels": labels, "preds": preds, "logits": logits}

    def validation_epoch_end(self, outputs):
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        logits = torch.cat([x["logits"] for x in outputs]).detach().cpu().numpy()

        ## There are multiple ways to track the metrics
        # 1. Confusion matrix plotting using inbuilt W&B method
        # self.logger.experiment.log(
        #     {
        #         "conf": wandb.plot.confusion_matrix(
        #             probs=logits, y_true=labels
        #         )
        #     }
        # )


        # 2. Confusion Matrix plotting using scikit-learn method
        wandb.log({"cm": wandb.sklearn.plot_confusion_matrix(labels, preds)})

        # 3. Confusion Matric plotting using Seaborn
        # data = confusion_matrix(labels, preds)
        # df_cm = pd.DataFrame(data, columns=np.unique(labels), index=np.unique(labels))
        # df_cm.index.name = "Actual"
        # df_cm.columns.name = "Predicted"
        # plt.figure(figsize=(7, 4))
        # plot = sns.heatmap(
        #     df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}
        # )  # font size
        # self.logger.experiment.log({"Confusion Matrix": wandb.Image(plot)})

        # self.logger.experiment.log(
        #     {"roc": wandb.plot.roc_curve(labels, logits)}
        # )


    def test_step(self, batch, batch_idx):
        loss, preds, targets, logits = self.model_step(batch)
        preds = preds.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        test_acc = accuracy_score(preds, targets)
        test_acc = torch.tensor(test_acc)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", test_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
