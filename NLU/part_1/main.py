import copy
from functions import *
from utils import *
from model import *

import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt

################
# PARSING ARGS #
################

import argparse
parser = argparse.ArgumentParser()

# hyperparameters to optimize
parser.add_argument('--emb', type=int, default=350)
parser.add_argument('--hidden', type=int, default=500)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--batch', type=int, default=64)

# parameters for 1.1
parser.add_argument('--bidirect', type=bool, default=False)
parser.add_argument('--do', type=float, default=0)

# parameters for logging
parser.add_argument('--log', type=bool, default=False)
parser.add_argument('--f1_filename', type=str, default="f1")
parser.add_argument('--acc_filename', type=str, default="acc")
parser.add_argument('--model_name', type=str, default=None)

args = parser.parse_args()

################
# LOADING DATA #
################

train_loader, dev_loader, test_loader, lang = get_dataloaders(args.batch)

out_slot = len(lang.id2slot)
out_int = len(lang.id2intent)
vocab_len = len(lang.id2word)

#######################
# INSTANTIATING MODEL #
#######################

model = ModelIAS(
  args.hidden, 
  out_slot, 
  out_int, 
  args.emb, 
  vocab_len, 
  pad_index=PAD_TOKEN, 
  n_layer=args.layers, 
  dropout = args.do, 
  bidirectional=args.bidirect
).to(device) 

model.apply(init_weights) 

##################
# TRAINING MODEL #
##################

# setting weights for cross entropy loss
slot_weights = [1 if i != 0 else 1e-2 for i in lang.id2slot]
weight_sum = sum(slot_weights)
slot_weights = torch.tensor(slot_weights, dtype = torch.float, device = device)

total_intents = len(lang.intent2id)
intent_weights = torch.tensor([weight_sum / total_intents for _ in range(total_intents)], dtype=torch.float, device = device)

criterion_slots = nn.CrossEntropyLoss(weight = slot_weights, ignore_index=PAD_TOKEN) 
criterion_intents = nn.CrossEntropyLoss(weight = intent_weights)

lr = args.lr 
clip = 5

optimizer = optim.Adam(model.parameters() , lr=lr)

criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN) 
criterion_intents = nn.CrossEntropyLoss()  

n_epochs = 200
patience = 5
losses_train = []
losses_dev = []
sampled_epochs = []
best_f1 = 0
pbar = tqdm(range(1, n_epochs)) 

dev_slots_f1 = []
dev_intents_accuracy = []

best_model = None
for x in pbar:
	loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=clip) 
	if x % 1 == 0:
		slots_f1, intent_acc, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang) 
			
		sampled_epochs.append(x) 
		losses_train.append(np.asarray(loss).mean()) 
		losses_dev.append(np.asarray(loss_dev).mean())
		
		dev_slots_f1.append(slots_f1)
		dev_intents_accuracy.append(intent_acc)
		
		f1 = slots_f1
		if f1 > best_f1:
			best_f1 = f1
			best_model = copy.deepcopy(model).to(device)
			patience = 5
		else:
			patience -= 1
		if patience <= 0:
			break 

####################
# EVALUATING MODEL #
####################

if best_model is not None:
	slots_f1, intent_acc, dev_losses = eval_loop(test_loader, criterion_slots, criterion_intents, best_model, lang)     
else:
	slots_f1, intent_acc, dev_losses = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)     

print(slots_f1)
print(intent_acc)

if args.log:
	with open(f"{args.f1_filename}_{args.lr}", "w") as slots_f1_file, open(f"{args.acc_filename}_{args.lr}", "w") as intent_accuracy_file:
		for f1, acc in zip(dev_slots_f1, dev_intents_accuracy):
			slots_f1_file.write(str(f1) + '\n')
			intent_accuracy_file.write(str(acc) + '\n')
 
if args.model_name is not None:
    path = f"./bin/{args.model_name}.pt"
    torch.save(best_model, path)