import torch
import os
import torch.nn as nn
from sklearn.metrics import f1_score
import numpy as np

device = 'cuda:0'
os.environ['CUDA_LAUNCH_BLOCKING']= "1" 
PAD_TOKEN = 0


def init_weights(mat):
	for m in mat.modules():
		if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:

			for name, param in m.named_parameters():
				if 'weight_ih' in name:
					for idx in range(4):
						mul = param.shape[0]// 4
					torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
				elif 'weight_hh' in name:
					for idx in range(4):
						mul = param.shape[0]// 4
					torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
				elif 'bias' in name:
					param.data.fill_(0)
		else:
			if type(m) in [nn.Linear]:
				torch.nn.init.uniform_(m.weight, -0.01, 0.01)
				if m.bias != None:
					m.bias.data.fill_(0.01)

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
	model.train()
	loss_array = []
	for sample in data:
		optimizer.zero_grad()

		slots, intent = model(sample['utterances'], sample['slots_len'])

		loss_intent = criterion_intents(intent, sample['intents'])
		loss_slot = criterion_slots(slots, sample['y_slots'])
		loss = loss_intent + loss_slot	

		loss_array.append(loss.item())
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
		optimizer.step()
	return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
	model.eval()
	loss_array = []

	ref_intents = []
	hyp_intents = []

	ref_slots = []
	hyp_slots = []
 
	# used for confusion matrix
	true_intents = []	
	pred_intents = []
	with torch.no_grad(): 
		for sample in data:
			slots, intents = model(sample['utterances'], sample['slots_len'])
			loss_intent = criterion_intents(intents, sample['intents'])
			loss_slot = criterion_slots(slots, sample['y_slots'])
			loss = loss_intent + loss_slot

			# calculate sum(cel) for both tasks
			loss_array.append(loss.item())
			
			out_intents = [lang.id2intent[x]for x in torch.argmax(intents, dim=1).tolist()]

			gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]

			true_intents.append(gt_intents)
			pred_intents.append(out_intents)

			ref_intents.extend(gt_intents)
			hyp_intents.extend(out_intents)

			output_slots = torch.argmax(slots, dim=1)
			for id_seq, seq in enumerate(output_slots):
				length = sample['slots_len'].tolist()[id_seq]
				utt_ids = sample['utterance'][id_seq][:length].tolist()
				gt_ids = sample['y_slots'][id_seq].tolist()
		
				gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
				utterance = [lang.id2word[elem] for elem in utt_ids]
				to_decode = seq[:length].tolist()

				ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
				hyp_slots.append([(w, lang.id2slot[s]) for w, s in zip(utterance, to_decode)])

	slots_f1 = calculate_slot_f1(ref_slots, hyp_slots)
	intents_accuracy = calculate_intent_accuracy(ref_intents, hyp_intents)
  
	return slots_f1, intents_accuracy, loss_array

# custom functions to calculate intent accuracy and f1 score for slot filling
def calculate_intent_accuracy(reference_intents, hypothesis_intents):
	correct_predictions = sum(1 for ref, hyp in zip(reference_intents, hypothesis_intents) if ref[1] == hyp[1])
	total_predictions = len(reference_intents)

	if total_predictions == 0:
			return 0.0
	accuracy = correct_predictions / total_predictions
	return accuracy

def calculate_slot_f1(reference_slots, hypothesis_slots):
	ref_labels = []
	hyp_labels = []

	for ref_sentence, hyp_sentence in zip(reference_slots, hypothesis_slots):
		ref_labels.extend([label for _, label in ref_sentence])
		hyp_labels.extend([label for _, label in hyp_sentence])

	f1 = f1_score(ref_labels, hyp_labels, average='weighted')
	return f1
	 