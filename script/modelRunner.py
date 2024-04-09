import torch
from torch import nn
import pytorch_lightning as pl
from collections import OrderedDict
import fenbin
from pytorch_lightning.loggers import TensorBoardLogger

modelPath = "C:/Users/hacke/Documents/GitHub/chess-4200/script/bestModel.ckpt"
class EvaluationModel(pl.LightningModule):
  def __init__(self,learning_rate=1e-3,batch_size=1024,layer_count=10):
    super().__init__()
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    layers = []
    layers.append(("linear-0", nn.Linear(285, 808)))
    layers.append(("relu-0", nn.ReLU()))

    for i in range(1, layer_count - 1):
        layers.append((f"linear-{i}", nn.Linear(808, 808)))
        layers.append((f"relu-{i}", nn.ReLU()))

    layers.append((f"linear-{layer_count - 1}", nn.Linear(808, 1)))
    self.seq = nn.Sequential(OrderedDict(layers))

  def forward(self, x):
    return self.seq(x)

  def training_step(self, batch, batch_idx):
    x, y = batch['binary'], batch['eval']
    y_hat = self(x)
    loss = F.l1_loss(y_hat, y)
    self.log("train_loss", loss)
    return loss

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
  
configs = [
           {"layer_count": 4, "batch_size": 512},
           ]
for config in configs:
    version_name = f'{config["batch_size"]}-layer_count-{config["layer_count"]}'
    logger = TensorBoardLogger("lightning_logs", name="chessml", version=version_name)
    model = EvaluationModel(layer_count=config["layer_count"], batch_size=config["batch_size"])
    break

checkpoint = torch.load(modelPath)
model = EvaluationModel(layer_count=config["layer_count"], batch_size=config["batch_size"])
model.load_state_dict(checkpoint['state_dict'])
model.eval()

def evalPos(fen):
    return model(fenbin.fen_to_binary(fen)).item()