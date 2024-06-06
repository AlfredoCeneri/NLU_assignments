import os
from sklearn.metrics import f1_score
import torch
import torch.nn as nn


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

def train_loop_finetuning(model, data, criterion_slots, criterion_intents, optimizer, clip=5):
	model.train()
	loss_array = []
	for sample in data:
		optimizer.zero_grad()
		intent, slots = model(sample['encoded_sentence'])
		loss_intent = criterion_intents(intent, sample['intents'])
		loss_slot = criterion_slots(slots, sample['slots'])
		loss = loss_slot + loss_intent

		loss_array.append(loss.item())
		torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
		loss.backward()
		optimizer.step()

	return loss_array

def eval_loop_finetuning(model, data, criterion_slots, criterion_intents, lang, tokenizer):
	model.eval()
	loss_array = []

	ref_intents = []
	hyp_intents = []

	reference_slots = []
	hypothesis_slots = []

	true_slots = []
	pred_slots = []
	with torch.no_grad(): 
		for sample in data:
			intents, slots = model(sample['encoded_sentence'])
			loss_intent = criterion_intents(intents, sample['intents'])
			ref_slots_per_token = sample['slots']

			loss_slot = criterion_slots(slots, ref_slots_per_token)
			loss = loss_intent + loss_slot

			loss_array.append(loss.item())

			out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()]

			gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
			ref_intents.extend(gt_intents)
			hyp_intents.extend(out_intents)

			output_slots = torch.argmax(slots, dim=1)

			for id_seq, seq in enumerate(output_slots):
				length = sample['lengths'][id_seq]
				utt_ids = sample['encoded_sentence']['input_ids'][id_seq][:length+2].tolist()
				utterance = tokenizer.decode(utt_ids).split(' ')
				utterance = utterance[1:-1]

				hyp_slot_ids = seq[:length].tolist()
				hyp_slots = [lang.id2slot[slot_id] for slot_id in hyp_slot_ids]
				ref_slot_ids = sample['slots'][id_seq].tolist()
				ref_slot_ids = ref_slot_ids[:length]
				ref_slots = [lang.id2slot[elem] for elem in ref_slot_ids]

				if len(utterance)!= len(ref_slots): 
					lengths = get_subwords_count(utterance, tokenizer)
					ref_slots, hyp_slots = update_references(ref_slots, hyp_slots, lengths)

				reference_slots.append([(w, ref) for w, ref in zip(utterance, ref_slots)])
				hypothesis_slots.append([(w, hyp) for w, hyp in zip(utterance, hyp_slots)])
		

	slots_f1 = calculate_slot_f1(reference_slots, hypothesis_slots)
	intents_accuracy = calculate_intent_accuracy(true_slots, pred_slots) 
	return slots_f1, intents_accuracy, loss_array

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
	 