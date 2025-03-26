import torch
from transformers import BertConfig
from torchviz import make_dot
from model import BertWithNumeric

# Create a dummy config and model
config = BertConfig(num_labels=2)
model = BertWithNumeric(config)

# Dummy inputs
batch_size = 1
seq_length = 10
dummy_input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
dummy_attention_mask = torch.ones((batch_size, seq_length))
dummy_amount = torch.randn(batch_size, 1)

# Forward pass
output = model(dummy_input_ids, dummy_attention_mask, dummy_amount)

# Visualize the graph using logits as the output
dot = make_dot(output["logits"], params=dict(model.named_parameters()))
dot.render("model_architecture", format="png")
