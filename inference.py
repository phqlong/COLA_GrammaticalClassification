import torch

from datamodule import ColaDatamodule
from model import ColaModel


class ColaPredictor:
    def __init__(self, 
        model_dirs : str = "./models/",
        model_checkpoint : str = "epoch=3-step=536",
    ):
        # loading the trained model
        self.model_path = model_dirs + model_checkpoint
        self.model = ColaModel.load_from_checkpoint(self.model_path)

        # keep the model in eval mode
        self.model.eval()
        self.model.freeze()
        
        self.processor = ColaDatamodule()
        self.softmax = torch.nn.Softmax(dim=0)
        
        self.lables = ["unacceptable", "acceptable"]

    def predict(self, text):
        # text => run time input
        inference_sample = {"sentence": text}
        # tokenizing the input
        processed = self.processor.tokenize_data(inference_sample)
        # predictions
        logits = self.model(
            torch.tensor([processed["input_ids"]]),
            torch.tensor([processed["attention_mask"]]),
        )
        scores = self.softmax(logits[0]).tolist()
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": score})
        return predictions

if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    predictor = ColaPredictor("./models/epoch=0-step=267.ckpt")
    print(predictor.predict(sentence))
