import os
import logging
import pytorch_lightning as pl

from typing import Any, Dict, Optional, Tuple
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)

class ColaDatamodule(pl.LightningDataModule):
    def __init__(self,
                 model_checkpoint: str = "google/bert_uncased_L-2_H-128_A-2",
                 data_dir: str = os.getcwd() + "\data", # On Windows must use full path
                 num_workers: int = 4,
                 batch_size: int = 64,
                 max_length : int = 128
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.train_data: Optional[Dataset] = None
        self.val_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """
        COLA is subset of GLUE benchmark
        The Corpus of Linguistic Acceptability consists of English acceptability 
        judgments drawn from books and journal articles on linguistic theory. 
        Each example is a sequence of words annotated with whether it is a 
        grammatical English sentence.
        """
        logger.info("Load & Prepare COLA Dataset from GLUE benchmark")
        cola_dataset = load_dataset("glue", "cola") #keep_in_memory=True, cache_dir=self.hparams.data_dir)
        logger.info("CoLA Dataset:")
        logger.info(cola_dataset)

        return super().prepare_data()

    def setup(self, stage=None):
        def tokenize(example):
            return self.tokenizer(
                example['sentence'],
                truncation=True,
                padding='max_length',
                max_length=self.hparams.max_length
            )
        if not self.train_data and not self.val_data and not self.test_data:
            cola_dataset = load_dataset("glue", "cola") #, cache_dir=self.hparams.data_dir)
            self.train_data = cola_dataset['train']
            self.val_data = cola_dataset['validation']
            self.test_data = cola_dataset['test']

        if stage in [None, 'fit']:
            self.train_data = self.train_data.map(tokenize, batched=True)
            self.train_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

            self.val_data = self.val_data.map(tokenize, batched=True)
            self.val_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label", "sentence"]
            )
        else:
            self.test_data = self.test_data.map(tokenize, batched=True)
            self.test_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label", "sentence"]
            )
        logger.info(f"Setup datset in stage {stage}")

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)


if __name__ == "__main__":
    data_model = ColaDatamodule()
    data_model.prepare_data()
    data_model.setup()
    print(next(iter(data_model.train_dataloader()))["input_ids"].shape)