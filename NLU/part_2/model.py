import torch.nn as nn

class BertFineTuning(nn.Module):
  def __init__(self, bert, out_slots, out_intents):
    super(BertFineTuning, self).__init__()

    self.bert = bert

    self.in_intents = self.in_slots = 768 # bert-base last hidden state's size

    self.intent_output = nn.Linear(self.in_intents, out_intents)
    self.slots_outputs = nn.Linear(self.in_slots, out_slots)

  def forward(self, data): 
    bert_output = self.bert(**data)
    hidden_states = bert_output.last_hidden_state

    cls_hidden_states = hidden_states[:, 0, :]
    hidden_states = hidden_states * (data['attention_mask']).unsqueeze(-1)

    intent_out = self.intent_output(cls_hidden_states) 
    slots_out = self.slots_outputs(hidden_states[:, 1:-1, :]) # do not consider [CLS] and [SEP]tokens

    return intent_out, slots_out.permute(0, 2, 1) # permute needed for loss computation with cel