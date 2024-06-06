from utils import *
from functions import *
from model import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import copy
import torch.optim as optim
from utils import DEVICE
import argparse

###############
# ARG PARSING #
###############

parser = argparse.ArgumentParser()

# general hyperparameters
parser.add_argument('--lr', type=float, default=1.5)            # learning rate
parser.add_argument('--emb', type=int, default=300)             # embedding size
parser.add_argument('--hidden', type=int, default=300)          # hidden size
parser.add_argument('--batch', type=int, default=64)            # training batch size
parser.add_argument('--layers', type=int, default=1)            # number of LSTM layers

# assignment parameters
parser.add_argument('--do', type=float, default=0)              # naive dropout rate
parser.add_argument('--var_do', type=float, default=0)          # variational dropout rate
parser.add_argument('--Adam', type=bool, default=False)         # flag for using adam or SGD
parser.add_argument('--wt', type=bool, default=False)           # flag for weight tying
parser.add_argument('--nt', type=bool, default=False)           # flag for ntavgsgd 

# parameters for visualization / recording results
parser.add_argument('--plot', type=bool, default=False)         # flag for plotting ppl
parser.add_argument('--ppl_filename', type=str, default=None)   # flag for recording ppl in separate file
parser.add_argument('--model_name', type=str, default=None)     # flag for saving best model's dict state

args = parser.parse_args()


try:
    check_args(args)
except ValueError as ve:
    print(f"[*] Error: {ve}")
    print("[*] Exiting...")
    exit(0)


#############
# LOAD DATA #
#############

train_loader, dev_loader, test_loader, lang = load_data(args.batch)
print(len(lang.word2id))

#####################
# INSTANTIATE MODEL #
#####################

vocab_len = len(lang.word2id)
model = LM_LSTM(
    args.emb, 
    args.hidden, 
    vocab_len, 
    pad_index=lang.word2id["<pad>"], 
    naive_dropout = args.do,
    variational_dropout = args.var_do,
    weight_tying = args.wt,
    n_layers= args.layers
    ).to(DEVICE)

model.apply(init_weights)

###############
# TRAIN MODEL #
###############

n_epochs = 100
patience = 10
losses_train = []
losses_dev = []
best_ppl = math.inf
best_model = None
pbar = tqdm(range(1,n_epochs)) 

if args.Adam:
    optimizer = optim.AdamW(model.parameters(), lr = args.lr)
else:
    optimizer = optim.SGD(model.parameters(), lr = args.lr)

ppls = []
state_dicts = []
criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
clip = 5

try:
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
        if epoch % 1 == 0: # logging interval is the same as iterations in an epoch, as suggested ntavsgd paper
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)
            if  ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to(DEVICE)
                patience = 10
            else:
                patience -= 1
            if patience <= 0: 
                break 
            state_dicts.append(model.state_dict())
            ppls.append(ppl_dev)
except KeyboardInterrupt:
    print("[*] exiting training...")
    if best_model is not None:
        final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
        print(f"final test set ppl: {final_ppl}")
    if args.ppl_filename is not None and len(ppls) != 0:
        with open(args.ppl_filename, 'w') as f:
            for ppl in ppls:
                f.write(str(ppl) + '\n')
    exit(0)
        

############
# NtAvgSGD #
############

if args.nt and not args.Adam:
    n = 5 # as suggested by paper
    best_model.load_state_dict(nt_avg_sdg_weights(state_dicts[n:], ppls[n:]))

###########
# LOGGING #
###########

if args.plot:
    import matplotlib.pyplot as plt
    plt.plot([i for i in range(epoch-1)], ppls)
    plt.show()

if args.ppl_filename:
    with open(args.ppl_filename, 'w') as f:
        for ppl in ppls:
            f.write(str(ppl) + '\n')

final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
print(f"{final_ppl}") # used for scripts that optimize hyperparameters


if args.model_name is not None:
    if best_model is None:
        best_model = model
    
    path = f"./bin/{args.model_name}.pt"
    torch.save(best_model.state_dict(), path)