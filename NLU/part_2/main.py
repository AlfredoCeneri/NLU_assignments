from transformers import BertTokenizer, BertModel
from pprint import pprint
from utils import *
from functions import *
from tqdm import tqdm
from model import BertFineTuning
import torch.optim as optim
import numpy as np 
from pprint import pprint
import copy
import argparse
import matplotlib.pyplot as plt
import numpy as np

#####################
# PARSING ARGUMENTS #
#####################

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--log', type=bool, default=False)
parser.add_argument('--model_name', type=str, default=None)

args = parser.parse_args()

################
# LOADING DATA #
################

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_loader, dev_loader, test_loader, lang = get_dataloaders_for_bert(args.batch, tokenizer) 

#####################
# INSTANTIATE MODEL #
#####################

model = BertFineTuning(BertModel.from_pretrained("bert-base-uncased") , len(lang.id2slot) , len(lang.id2intent)).to(device) 

##################
# TRAINING MODEL #
##################

optimizer = optim.Adam(model.parameters() , lr = args.lr) 

# weights for weighted cross entropy loss
slot_weights = [1 if i != 0 else 1e-2 for i in lang.id2slot]
weight_sum = sum(slot_weights)
slot_weights = torch.tensor(slot_weights, dtype = torch.float, device = device)

total_intents = len(lang.intent2id)
intent_weights = torch.tensor([weight_sum / total_intents for _ in range(total_intents)], dtype=torch.float, device = device)

criterion_slots = nn.CrossEntropyLoss(weight = slot_weights, ignore_index=PAD_TOKEN) 
criterion_intents = nn.CrossEntropyLoss(weight = intent_weights)

n_epochs = 200
patience = 20
losses_train = []
dev_slots_f1 = []
dev_intents_accuracy = []
sampled_epochs = []
best_f1 = 0
pbar = tqdm(range(1, n_epochs)) 
clip = 5
logging_interval = 1

best_model = None
try:
	for epoch in pbar:
		loss = train_loop_finetuning(model, train_loader, criterion_slots, criterion_intents, optimizer, clip=clip)
		if epoch % logging_interval == 0:
			losses_train.append(np.asarray(loss).mean())
			slots_f1, intent_accuracy, dev_loss = eval_loop_finetuning(model, dev_loader, criterion_slots, criterion_intents, lang, tokenizer) 
			
			pbar.set_description(f"slots f1: {slots_f1}")

			dev_slots_f1.append(slots_f1)
			dev_intents_accuracy.append(intent_accuracy)
			epoch_count =+ 1

			if slots_f1 > best_f1:
				best_f1 = slots_f1
				patience = 20
				best_model = copy.deepcopy(model).to(device)
			else:
				patience -= 1
			if patience <= 0:
				break 
except KeyboardInterrupt:
	print("quitting training early")

	if args.log:
		if len(dev_slots_f1) != 0:
			with open("f1_score s", "w") as slots_f1_file, open("intent_accuracy", "w") as intent_accuracy_file:
				for f1, acc in zip(dev_slots_f1, dev_intents_accuracy):
					slots_f1_file.write(f1 + '\n')
					intent_accuracy_file.write(acc + '\n')
   
	exit(0)

 
##################
# EVALUATE MODEL #
##################

slots_f1, intent_accuracy, dev_loss = eval_loop_finetuning(model, dev_loader, criterion_slots, criterion_intents, lang, tokenizer) 

print(f"{slots_f1, intent_accuracy}")

if args.model_name is not None:
  path = f"./bin/{args.model_name}.pt"
  torch.save(best_model, path)

# plot slots f1 and intent accuracy
if args.log:
	with open(f"f1_score_{args.lr}", "w") as slots_f1_file, open(f"intent_accuracy_{args.lr}", "w") as intent_accuracy_file:
		for f1, acc in zip(dev_slots_f1, dev_intents_accuracy):
			slots_f1_file.write(str(f1) + '\n')
			intent_accuracy_file.write(str(acc) + '\n')

if args.model_name:
  path = f"./bin/{args.model_name}.pt"
  torch.save(best_model, path)
