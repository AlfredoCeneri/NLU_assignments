import torch
from functions import PAD_TOKEN, device
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from functions import *
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F
import torch.utils.data as data
import json
from collections import Counter
from sklearn.model_selection import train_test_split

def load_data(path) :
  '''
      input: path/to/data
      output: json 
  '''
  dataset = []
  with open(path)  as f:
      dataset = json.loads(f.read() ) 
  return dataset

class Lang() :
  def __init__(self, words, intents, slots, cutoff=0) :
    self.word2id = self.w2id(words, cutoff=cutoff, unk=True) 
    self.slot2id = self.lab2id(slots, pad=True) 
    self.intent2id = self.lab2id(intents, pad=False) 
    self.id2word = {v:k for k, v in self.word2id.items() }
    self.id2slot = {v:k for k, v in self.slot2id.items() }
    self.id2intent = {v:k for k, v in self.intent2id.items() }
        
  def w2id(self, elements, cutoff=None, unk=True) :
    vocab = {'pad': PAD_TOKEN}
    if unk:
        vocab['unk']= len(vocab) 
    count = Counter(elements) 
    for k, v in count.items() :
        if v > cutoff:
            vocab[k]= len(vocab) 
    return vocab
    
  def lab2id(self, elements, pad=True) :
    vocab = {}
    if pad:
        vocab['pad']= PAD_TOKEN
    for elem in elements:
        if elem not in vocab.keys():
            vocab[elem]= len(vocab)
    return vocab

# used to handle the subtokenization issue when pre-tokenizing a dataset
# in particular, labels corresponding to a word are repeated for each eventual subtoken
def align_labels_to_tokens(original_sent, labels, tokenizer):
    words = original_sent
    input_ids = [101]
    stretched_labels = []

    for j, w in enumerate(words) :
        tokenized = tokenizer(w, add_special_tokens=False) 
        tmp_input_ids = tokenized["input_ids"]

        if len(tmp_input_ids) == 1: # word corresponds to only one id
            input_ids.append(tmp_input_ids[0]) 
            stretched_labels.append(labels[j]) 
        else: # word corresponds to more than one id, stretch label
            for i in tmp_input_ids:
                input_ids.append(i) 
                stretched_labels.append(labels[j]) 

    input_ids.append(102)
    token_type_ids = [0]* len(input_ids) 
    attention_mask = [1]* len(input_ids) 

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.int32) ,
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.int32) ,
        "attention_mask": torch.tensor(attention_mask, dtype=torch.int32) 
    }, stretched_labels

class IntentsAndSlotsForBert(data.Dataset) :
  def __init__(self, data, tokenizer, lang) :
    # sents and slots are not split
    self.sents = [x["utterance"]for x in data]
    self.slots = [x["slots"]for x in data]
    self.intents = [x["intent"]for x in data]

    self.sentence_encoding = []
    self.slots_per_sentence = []
    self.intent_per_sentence = []

    # transforming sents into encoded sents (done to avoid calling tokenizer before forward pass)
    for i in range(len(self.sents) ) :
      sent = self.sents[i].split(' ') 
      sent_slots = self.slots[i].split(' ') 

      toks, stretched_slots = align_labels_to_tokens(sent, sent_slots, tokenizer) 
      self.sentence_encoding.append(toks) 
      self.slots_per_sentence.append(self.map_sequence(stretched_slots, lang.slot2id) ) 
      self.intent_per_sentence.append(lang.intent2id[self.intents[i]]) 

  def map_sequence(self, data, mapper) :
    res = []
    for d in data:
      if d in mapper.keys() :
        res.append(mapper[d]) 
      else:
        raise KeyError("data not recognized in mapper") 
    return res

  def __len__(self) :
    return len(self.intent_per_sentence) 

  def __getitem__(self, idx) :
    return {
      "encoded_sentence": self.sentence_encoding[idx], # contains input_ids, data_types and attention_mask
      "slots": self.slots_per_sentence[idx],
      "intent": self.intent_per_sentence[idx]
    }

def finetuning_collate_fn(batch): 
	batch.sort(key=lambda x: len(x['encoded_sentence']['input_ids']), reverse=True)
	maxlen = max(len(x['encoded_sentence']['input_ids']) for x in batch)
	slots_maxlen = max(len(x['slots']) for x in batch) 

	new_item = {
		'encoded_sentence': {
			'input_ids': [], 
			'token_type_ids': [], 
			'attention_mask': []
		},
		'slots': [], 
		'intents': [], 
		'lengths': []
	}

	tokens = [x['encoded_sentence']['input_ids'] for x in batch]
	token_type_ids = [x['encoded_sentence']['token_type_ids'] for x in batch]
	attention_masks = [x['encoded_sentence']['attention_mask'] for x in batch]
	slots = [x["slots"] for x in batch]
	
	for tok, ttid, mask in zip(tokens, token_type_ids, attention_masks):
		padded_tok = F.pad(tok, (0, maxlen - len(tok)), value=0)
		padded_ttid = F.pad(ttid, (0, maxlen - len(ttid)), value=0)
		padded_mask = F.pad(mask, (0, maxlen - len(mask)), value=0)
			
		new_item['encoded_sentence']['input_ids'].append(padded_tok)
		new_item['encoded_sentence']['token_type_ids'].append(padded_ttid)
		new_item['encoded_sentence']['attention_mask'].append(padded_mask)

	new_item['encoded_sentence']['input_ids'] = torch.stack(new_item['encoded_sentence']['input_ids']).to(device)
	new_item['encoded_sentence']['token_type_ids'] = torch.stack(new_item['encoded_sentence']['token_type_ids']).to(device)
	new_item['encoded_sentence']['attention_mask'] = torch.stack(new_item['encoded_sentence']['attention_mask']).to(device)

	for s in slots:
		new_item['lengths'].append(len(s))
		padded_s = s + [0] * ( slots_maxlen - len(s))
		new_item['slots'].append(padded_s)

	new_item['slots'] = torch.tensor(new_item['slots'], device=device)
	new_item['intents'] = torch.tensor([x['intent'] for x in batch], device=device)
	new_item['lengths'] = torch.tensor(new_item['lengths'], device=device)
 
	return new_item

# returns a list of lengths corresponding to the number of subtokens per each word in utterance
def get_subwords_count(utterance, tokenizer):
	lengths = []
	for w in utterance:
		toks = tokenizer.encode(w, add_special_tokens=False)
		lengths.append(len(toks))
	return lengths

# used to handle the subtokenization issue while evaluating the model
# in particular, labels corresponding to the first subtoken of a word determine the whole word's label
def update_references(hyp_slots, ref_slots, lengths):
	updated_hyp_slots = []
	updated_ref_slots = []

	i = 0; k = 0
	while i < sum(lengths):
		updated_hyp_slots.append(hyp_slots[i])
		updated_ref_slots.append(ref_slots[i])
		i += lengths[k]; k += 1

	return updated_hyp_slots, updated_ref_slots

def get_dataloaders_for_bert(batchsize, tokenizer):
	tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json')) 
	test_raw = load_data(os.path.join('dataset','ATIS','test.json')) 

	portion = 0.10

	intents = [x['intent']for x in tmp_train_raw]
	count_y = Counter(intents) 

	labels = []
	inputs = []
	mini_train = []

	for id_y, y in enumerate(intents) :
			if count_y[y]> 1:
					inputs.append(tmp_train_raw[id_y]) 
					labels.append(y) 
			else:
					mini_train.append(tmp_train_raw[id_y]) 


	X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, random_state=42, shuffle=True, stratify=labels) 
	X_train.extend(mini_train) 

	train_raw = X_train
	dev_raw = X_dev


	slots = [x["slots"]for x in train_raw]
	intents = [x["intent"]for x in train_raw]


	PAD_TOKEN = 0

	# tokenizing raw dataset
	corpus = train_raw + dev_raw + test_raw 
	words = sum([x['utterance'].split()  for x in corpus], [])   
	slots = set(sum([line['slots'].split()  for line in corpus],[])) 
	intents = set([line['intent']for line in corpus]) 

	lang = Lang(words, intents, slots, cutoff=0) 

	train_dataset = IntentsAndSlotsForBert(train_raw, tokenizer, lang) 
	dev_dataset = IntentsAndSlotsForBert(dev_raw, tokenizer, lang) 
	test_dataset = IntentsAndSlotsForBert(test_raw, tokenizer, lang) 

	train_loader = DataLoader(train_dataset, batch_size=batchsize, collate_fn=finetuning_collate_fn, shuffle=True) 
	dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=finetuning_collate_fn, shuffle=True) 
	test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=finetuning_collate_fn) 

	return train_loader, dev_loader, test_loader, lang