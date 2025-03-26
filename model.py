import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
import torch.nn.functional as F

class BertWithNumeric(BertPreTrainedModel):
    """
    A custom BERT model that:
      1) Uses a standard BertModel backbone.
      2) Appends one numeric feature (`amount`) to the pooled [CLS] embedding.
      3) Passes the combined vector into a 2-layer feed-forward classifier.

    Note: The 'original_category' is appended to the text in the data loader, so
    it's already part of the `input_ids` that BERT sees. No extra code is needed
    in this model to handle that.
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        # 1) BertModel backbone
        self.bert = BertModel(config)
        
        # 2) Classifier that accepts [CLS] embedding + 1 numeric feature
        #    config.hidden_size + 1 is needed because we have 'amount' as an extra float
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size + 1, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, self.num_labels),
        )
        
        # Post-initialization from Hugging Face (e.g., weights init if needed)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        amount=None,
        labels=None
    ):
        """
        Forward pass for training/inference.
        
        Arguments:
            input_ids (torch.LongTensor): Encoded token IDs of shape [batch_size, seq_len].
            attention_mask (torch.FloatTensor): Mask for ignoring padding; same shape as input_ids.
            amount (torch.FloatTensor): Numeric feature of shape [batch_size, 1].
            labels (torch.LongTensor, optional): Class labels of shape [batch_size]. 
                                                If provided, compute cross-entropy loss.
        
        Returns:
            dict with keys:
              - 'loss' (torch.FloatTensor, optional): Computed only if `labels` is provided.
              - 'logits' (torch.FloatTensor): Shape [batch_size, num_labels].
              - 'hidden_states' (optional): If you set output_hidden_states=True in BERT config.
              - 'attentions' (optional): If you set output_attentions=True in BERT config.
        """
        # Pass tokens through BERT
        # If you want intermediate states or attentions, set output_hidden_states/output_attentions
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Pooled output from the [CLS] token: shape [batch_size, hidden_size]
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Concatenate the numeric feature to the CLS embedding
        # shape becomes [batch_size, hidden_size + 1]
        combined = torch.cat((pooled_output, amount), dim=1)

        # Run through the classifier
        logits = self.classifier(combined)

        # Optionally compute loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        # Prepare output in dictionary form
        model_outputs = {
            "loss": loss,
            "logits": logits
        }
        # Include hidden_states/attentions if they exist
        if outputs.hidden_states is not None:
            model_outputs["hidden_states"] = outputs.hidden_states
        if outputs.attentions is not None:
            model_outputs["attentions"] = outputs.attentions

        return model_outputs
