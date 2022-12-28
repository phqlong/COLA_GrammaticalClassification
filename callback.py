import wandb
import pandas as pd
import pytorch_lightning as pl


class LogPredictionsCallback(pl.Callback):
    """
    Using WandbLogger to log Images, Text and More
    Pytorch Lightning is extensible through its callback system. We can create a custom callback to automatically log sample predictions during validation. `WandbLogger` provides convenient media logging functions:
    * `WandbLogger.log_text` for text data
    * `WandbLogger.log_image` for images
    * `WandbLogger.log_table` for [W&B Tables](https://docs.wandb.ai/guides/data-vis).

    An alternate to `self.log` in the Model class is directly using `wandb.log({dict})` or `trainer.logger.experiment.log({dict})`

    In this case we log the first 20 images in the first batch of the validation dataset along with the predicted and ground truth labels.
    """
    def on_validation_batch_end(
        self, 
        trainer, 
        pl_module, 
        outputs,
        batch, 
        batch_idx, 
        dataloader_idx
    ):
        """Called when the validation batch ends."""
 
        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case
        
        # Let's log 10 sample predictions from each batch
        sentences = batch['sentence']
        labels = outputs['labels'].detach().cpu().numpy()
        preds = outputs['preds'].detach().cpu().numpy()


        df = pd.DataFrame(
            {"Sentence": sentences, "Label": labels, "Predicted": preds}
        )

        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=df.sample(n=10), allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )