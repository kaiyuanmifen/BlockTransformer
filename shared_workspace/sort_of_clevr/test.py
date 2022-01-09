import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from transformers import TransformerEncoder
from einops import rearrange, repeat
from transformer_utilities.set_transformer import SetTransformer

from transformer.Models import Encoder as Vanilla_transformer_encoder

from transformer.Models_block import Encoder as Block_transformer_encoder

