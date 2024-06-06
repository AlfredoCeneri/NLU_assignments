import copy
import torch
import torch.utils.data as data
from functools import partial
from torch.utils.data import DataLoader
from torch.nn import Parameter

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 

def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output

def get_vocab(corpus, special_tokens=[]):
    output = {}
    i = 0 
    for st in special_tokens:
        output[st] = i
        i += 1
    for sentence in corpus:
        for w in sentence.split():
            if w not in output:
                output[w] = i
                i += 1
    return output

class Lang():
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}
    def get_vocab(self, corpus, special_tokens=[]):
        output = {}
        i = 0 
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output

class PennTreeBank (data.Dataset):
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []
        
        for sentence in corpus:
            self.source.append(sentence.split()[0:-1]) # We get from the first token till the second-last token
            self.target.append(sentence.split()[1:]) # We get from the second token till the last token
            # See example in section 6.2
        
        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src= torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample
     
    def mapping_seq(self, data, lang):
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    break
            res.append(tmp_seq)
        return res
    
def collate_fn(data, pad_token):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
        padded_seqs = padded_seqs.detach()
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x["source"]), reverse=True) 
    new_item = {}
    for key in data[0].keys(): 
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])
    
    new_item["source"] = source.to(DEVICE)
    new_item["target"] = target.to(DEVICE)
    new_item["number_tokens"] = sum(lengths)
    return new_item

def load_data(training_batch_size):
    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_loader = DataLoader(
        PennTreeBank(train_raw, lang), 
        batch_size = training_batch_size,
        collate_fn = partial(collate_fn, pad_token = lang.word2id["<pad>"]),
        shuffle = True
    )

    dev_lodaer = DataLoader(
        PennTreeBank(dev_raw, lang),
        batch_size = 128,
        collate_fn = partial(collate_fn, pad_token = lang.word2id["<pad>"]),
        shuffle = False 
    )

    test_loader = DataLoader(
        PennTreeBank(test_raw, lang),
        batch_size = 128,
        collate_fn = partial(collate_fn, pad_token = lang.word2id["<pad>"]),
        shuffle = False 
    )

    return train_loader, dev_lodaer, test_loader, lang

########################################################
# WRAPPER FOR MODULES TO IMPLEMENT VARIATIONAL DROPOUT #
########################################################

# taken from https://github.com/salesforce/awd-lstm-lm/

class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        return

    def _setup(self):
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        input_to_hidden = []
        hidden_to_hidden = []
        for name_w in self.weights:
            if "ih" in name_w:
                input_to_hidden.append(name_w)
            elif "h" in name_w:
                hidden_to_hidden.append(name_w) 
        
        w = None # stores values to set parameters with
        if self.variational:
            raw_w = getattr(self.module, input_to_hidden[0] + '_raw')
            mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1)).to(DEVICE)
            mask = torch.nn.functional.dropout(mask, p=self.dropout, training=self.module.training).expand_as(raw_w)
            w = mask * raw_w
            setattr(self.module, input_to_hidden[0], w)
            
            for ih in input_to_hidden[1:]:
                raw_w = getattr(self.module, ih + '_raw')
                w = mask * raw_w
                setattr(self.module, ih, w)
            
            raw_w = getattr(self.module, hidden_to_hidden[0] + '_raw')
            mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1)).to(DEVICE)
            mask = torch.nn.functional.dropout(mask, p=self.dropout, training=self.module.training).expand_as(raw_w)
            w = mask * raw_w
            setattr(self.module, hidden_to_hidden[0], w)
            for hh in hidden_to_hidden[1:]:
                raw_w = getattr(self.module, hh + '_raw')
                w = mask * raw_w
                setattr(self.module, hh, w)
         
    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)