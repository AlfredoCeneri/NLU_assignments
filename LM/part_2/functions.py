import math
import torch
import torch.nn as nn

def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []
    
    for sample in data:
        optimizer.zero_grad() 
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step()
        
    return sum(loss_array)/sum(number_of_tokens)

# from a list of state dicts and corresponding ppls, return the averaged state dict
def nt_avg_sdg_weights(state_dicts, ppls):
    print("averaging weights after ppl worsens")
    trigger = -1

    bestppl = ppls[0]
    for i, p in enumerate(ppls[1:]):
        if p > bestppl:
            trigger = i
            break
    
    if trigger == -1:
        return state_dicts[-1]
    
    avg_state_dict = {}
    first_state_dict = state_dicts[0]
    for name, param in first_state_dict.items():
        avg_state_dict[name] = param.clone().zero_()

    for state_dict in state_dicts:
        for name, param in state_dict.items():
            avg_state_dict[name] += param
    
    length = len(state_dicts)
    for name, param in avg_state_dict.items():
        avg_state_dict[name] /= (length - trigger + 1)
    return avg_state_dict

def eval_loop(data, eval_criterion, model):
    model.eval()

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            total_loss += loss.item() * sample["number_tokens"]
            total_tokens += sample["number_tokens"]

    ppl = math.exp(total_loss / total_tokens)
    loss_to_return = total_loss / total_tokens

    return ppl, loss_to_return

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

def check_args(args):
    if args.nt and args.Adam:
        raise ValueError("Cannot use NtAvgSgd with Adam optimizer.")






