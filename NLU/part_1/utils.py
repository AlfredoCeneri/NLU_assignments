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

class IntentsAndSlots (data.Dataset) :
  def __init__(self, dataset, lang, unk='unk') :
    self.utterances = []
    self.intents = []
    self.slots = []
    self.unk = unk
    
    for x in dataset:
      self.utterances.append(x['utterance']) 
      self.slots.append(x['slots']) 
      self.intents.append(x['intent']) 
        
    self.utt_ids = self.mapping_seq(self.utterances, lang.word2id) 
    self.slot_ids = self.mapping_seq(self.slots, lang.slot2id) 
    self.intent_ids = self.mapping_lab(self.intents, lang.intent2id) 

  def __len__(self) :
      return len(self.utterances) 

  def __getitem__(self, idx) :
    utt = torch.Tensor(self.utt_ids[idx]) 
    slots = torch.Tensor(self.slot_ids[idx]) 
    intent = self.intent_ids[idx]
    sample = {'utterance': utt, 'slots': slots, 'intent': intent}
    return sample
     
  def mapping_lab(self, data, mapper) :
    return [mapper[x]if x in mapper else mapper[self.unk]for x in data]
    
  def mapping_seq(self, data, mapper) : # Map sequences to number
    res = []
    for seq in data:
        tmp_seq = []
        for x in seq.split() : # split on white spaces
            if x in mapper:
                tmp_seq.append(mapper[x]) 
            else:
                tmp_seq.append(mapper[self.unk]) 
        res.append(tmp_seq) 
    return res
  
def collate_fn(data):
	def merge(sequences):
		lengths = [len(seq) for seq in sequences]
		max_len = 1 if max(lengths) == 0 else max(lengths)
		padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
		for i, seq in enumerate(sequences):
			end = lengths[i]
			padded_seqs[i, :end] = seq # We copy each sequence into the matrix
		# print(padded_seqs)
		padded_seqs = padded_seqs.detach()# We remove these tensors from the computational graph
		return padded_seqs, lengths

	# Sort data by seq lengths
	data.sort(key=lambda x: len(x['utterance']), reverse=True)
	new_item = {}
	for key in data[0].keys():
		new_item[key] = [d[key] for d in data]

		# We just need one length for packed pad seq, since len(utt)== len(slots)
	src_utt, _ = merge(new_item['utterance'])
	y_slots, y_lengths = merge(new_item["slots"])
 
	src_utt = src_utt.to(device)
	y_slots = y_slots.to(device)
	intent = torch.tensor(new_item["intent"], dtype=torch.long, device = device)
	y_lengths = torch.LongTensor(y_lengths).to(device)
 
	new_item["utterances"] = src_utt
	new_item["intents"] = intent
	new_item["y_slots"] = y_slots
	new_item["slots_len"] = y_lengths
	return new_item

def get_dataloaders(batch):
	tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json') ) 
	test_raw = load_data(os.path.join('dataset','ATIS','test.json') ) 

	sents = [x["utterance"].split(' ')  for x in tmp_train_raw]
	slots = [x["slots"].split(' ')  for x in tmp_train_raw]
	intents = [x['intent']for x in tmp_train_raw]

	portion = 0.10
	count_y = Counter(intents) 

	labels = []
	inputs = []
	mini_train = []

	for id_y, y in enumerate(intents) :
		if count_y[y]> 1: # train on intents that exists only once 
			inputs.append(tmp_train_raw[id_y]) 
			labels.append(y) 
		else:
			mini_train.append(tmp_train_raw[id_y]) 

	X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, random_state=42, shuffle=True, stratify=labels) 

	X_train.extend(mini_train) 
	train_raw = X_train
	dev_raw = X_dev

	y_test = [x['intent']for x in test_raw]

	# creating a lang class
	corpus = train_raw + dev_raw + test_raw 
	words = sum([x['utterance'].split()  for x in corpus], [])   
	slots = set(sum([line['slots'].split()  for line in corpus],[]) ) 
	intents = set([line['intent']for line in corpus]) 

	lang = Lang(words, intents, slots, cutoff=0) 

	# load data with lang class (already ids) 
	train_dataset = IntentsAndSlots(train_raw, lang) 
	dev_dataset = IntentsAndSlots(dev_raw, lang) 
	test_dataset = IntentsAndSlots(test_raw, lang)  

	train_loader = DataLoader(train_dataset, batch_size=batch, collate_fn=collate_fn, shuffle=True)  
	dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn) 
	test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn) 
 
	return train_loader, dev_loader, test_loader, lang