from typing import Any
import torch.nn as nn

################################################
#           dataset
################################################
vocab_out_true: Any = {'O', 'B-Complication'}
example_tokens: Any = ['Boa', 'noite']
example_tags: Any = ['O', 'B-Complication']

################################################
#           model
################################################
hidden_dim: Any = 4
num_classes: Any = 2
vocab_size: Any = 10
out_w2id: Any = {'O': 0, 'B-Complication': 1}

################################################
#           trainer
################################################
model: Any = nn.Linear(64, 2)
batch: Any = 2

################################################
#           trainer
################################################
output_folder: Any = 'output/'
