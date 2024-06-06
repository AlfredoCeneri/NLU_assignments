import torch
from functions import PAD_TOKEN, device
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from functions import *
from torch.utils.data import DataLoader
from transformers import BertTokenizer

def collate_fn( data ):
	def merge( sequences ):
		'''
		merge from batch * sent_len to batch * max_len
		'''
		lengths = [ len( seq ) for seq in sequences ]
		max_len = 1 if max( lengths ) == 0 else max( lengths )
		# Pad token is zero in our case
		# So we create a matrix full of PAD_TOKEN ( i.e. 0 )with the shape
		# batch_size X maximum length of a sequence
		padded_seqs = torch.LongTensor( len( sequences ),max_len ).fill_( PAD_TOKEN )
		for i, seq in enumerate( sequences ):
			end = lengths[ i ]
			padded_seqs[ i, :end ]  = seq # We copy each sequence into the matrix
		# print( padded_seqs )
		padded_seqs = padded_seqs.detach()# We remove these tensors from the computational graph
		return padded_seqs, lengths

	# Sort data by seq lengths
	data.sort( key=lambda x: len( x[ 'utterance' ] ), reverse=True )
	new_item = {}
	for key in data[ 0 ] .keys():
		new_item[ key ]  = [ d[ key ]  for d in data ]

		# We just need one length for packed pad seq, since len( utt )== len( slots )
	src_utt, _ = merge( new_item[ 'utterance' ] )
	y_slots, y_lengths = merge( new_item[ "slots" ] )
 
	src_utt = src_utt.to( device )
	y_slots = y_slots.to( device )
	intent = torch.tensor( new_item[ "intent" ], dtype=torch.long, device = device )
	y_lengths = torch.LongTensor( y_lengths ).to( device )
 
	new_item[ "utterances" ]  = src_utt
	new_item[ "intents" ]  = intent
	new_item[ "y_slots" ]  = y_slots
	new_item[ "slots_len" ]  = y_lengths
	return new_item

def init_weights( mat ):
	for m in mat.modules():
		if type( m ) in [ nn.GRU, nn.LSTM, nn.RNN ] :

			for name, param in m.named_parameters():
				if 'weight_ih' in name:
					for idx in range( 4 ):
						mul = param.shape[ 0 ] // 4
					torch.nn.init.xavier_uniform_( param[ idx*mul:( idx+1 )*mul ] )
				elif 'weight_hh' in name:
					for idx in range( 4 ):
						mul = param.shape[ 0 ] // 4
					torch.nn.init.orthogonal_( param[ idx*mul:( idx+1 )*mul ] )
				elif 'bias' in name:
					param.data.fill_( 0 )
		else:
			if type( m ) in [ nn.Linear ] :
				torch.nn.init.uniform_( m.weight, -0.01, 0.01 )
				if m.bias != None:
					m.bias.data.fill_( 0.01 )

def train_loop( data, optimizer, criterion_slots, criterion_intents, model, clip=5 ):
	model.train()
	loss_array = []
	for sample in data:
		optimizer.zero_grad()

		slots, intent = model( sample[ 'utterances' ] , sample[ 'slots_len' ] )

		loss_intent = criterion_intents( intent, sample[ 'intents' ] )
		loss_slot = criterion_slots( slots, sample[ 'y_slots' ] )
		loss = loss_intent + loss_slot

		loss_array.append( loss.item() )
		loss.backward()
		torch.nn.utils.clip_grad_norm_( model.parameters(), clip )
		optimizer.step()
	return loss_array

def eval_loop( data, criterion_slots, criterion_intents, model, lang ):
	model.eval()
	loss_array = []

	ref_intents = []
	hyp_intents = []

	ref_slots = []
	hyp_slots = []
	with torch.no_grad():
		for sample in data:
			slots, intents = model( sample[ 'utterances' ] , sample[ 'slots_len' ] )
			loss_intent = criterion_intents( intents, sample[ 'intents' ] )
			loss_slot = criterion_slots( slots, sample[ 'y_slots' ] )
			loss = loss_intent + loss_slot

			loss_array.append( loss.item() )

			out_intents = [ lang.id2intent[ x ] for x in torch.argmax( intents, dim=1 ).tolist() ]

			gt_intents = [ lang.id2intent[ x ]  for x in sample[ 'intents' ] .tolist() ]

			ref_intents.extend( gt_intents )
			hyp_intents.extend( out_intents )

			output_slots = torch.argmax( slots, dim=1 )

			for id_seq, seq in enumerate( output_slots ):
				length = sample[ 'slots_len' ].tolist()[ id_seq ]
				utt_ids = sample[ 'utterance' ] [ id_seq ] [ :length ].tolist()
				gt_ids = sample[ 'y_slots' ] [ id_seq ].tolist()
				gt_slots = [ lang.id2slot[ elem ]  for elem in gt_ids[ :length ] ]
				utterance = [ lang.id2word[ elem ]  for elem in utt_ids ]
				to_decode = seq[ :length ].tolist()
				ref_slots.append( [ ( utterance[ id_el ] , elem ) for id_el, elem in enumerate( gt_slots ) ] ) # change
				tmp_seq = []
				for id_el, elem in enumerate( to_decode ):
					tmp_seq.append( ( utterance[ id_el ] , lang.id2slot[ elem ]  ) )
				hyp_slots.append( tmp_seq )

	slots_f1 = calculate_slot_f1( ref_slots, hyp_slots )
	intents_accuracy = calculate_intent_accuracy( ref_intents, hyp_intents )
	return slots_f1, intents_accuracy, loss_array

def calculate_intent_accuracy( reference_intents, hypothesis_intents ):
	correct_predictions = sum( 1 for ref, hyp in zip( reference_intents, hypothesis_intents ) if ref[ 1 ]  == hyp[ 1 ] )
	total_predictions = len( reference_intents )

	if total_predictions == 0:
			return 0.0
	accuracy = correct_predictions / total_predictions
	return accuracy

def calculate_slot_f1( reference_slots, hypothesis_slots ):
	ref_labels = []
	hyp_labels = []

	for ref_sentence, hyp_sentence in zip( reference_slots, hypothesis_slots ):
		ref_labels.extend( [ label for _, label in ref_sentence ] )
		hyp_labels.extend( [ label for _, label in hyp_sentence ] )

	if not ref_labels:
			return 0.0
	f1 = f1_score( ref_labels, hyp_labels, average='weighted' )
	return f1
 
def create_dataloaders( batch ):
	tmp_train_raw = load_data( os.path.join( 'dataset','ATIS','train.json' )  ) 
	test_raw = load_data( os.path.join( 'dataset','ATIS','test.json' )  ) 

	sents = [x["utterance"].split( ' ' )  for x in tmp_train_raw]
	slots = [x["slots"].split( ' ' )  for x in tmp_train_raw]
	intents = [x['intent'] for x in tmp_train_raw]

	portion = 0.10
	count_y = Counter( intents ) 

	labels = []
	inputs = []
	mini_train = []

	for id_y, y in enumerate( intents ) :
		if count_y[y] > 1: # train on intents that exists only once 
			inputs.append( tmp_train_raw[id_y] ) 
			labels.append( y ) 
		else:
			mini_train.append( tmp_train_raw[id_y] ) 

	X_train, X_dev, y_train, y_dev = train_test_split( inputs, labels, test_size=portion, random_state=42, shuffle=True, stratify=labels ) 

	X_train.extend( mini_train ) 
	train_raw = X_train
	dev_raw = X_dev

	y_test = [x['intent'] for x in test_raw] 

	# creating a lang class
	corpus = train_raw + dev_raw + test_raw 
	words = sum( [x['utterance'].split()  for x in corpus], [] )   
	slots = set( sum( [line['slots'].split()  for line in corpus],[] )  ) 
	intents = set( [line['intent'] for line in corpus] ) 

	lang = Lang( words, intents, slots, cutoff=0 ) 

	train_dataset = IntentsAndSlots( train_raw, lang ) 
	dev_dataset = IntentsAndSlots( dev_raw, lang ) 
	test_dataset = IntentsAndSlots( test_raw, lang )  

	train_loader = DataLoader( train_dataset, batch_size=batch, collate_fn=collate_fn, shuffle=True )  
	dev_loader = DataLoader( dev_dataset, batch_size=64, collate_fn=collate_fn ) 
	test_loader = DataLoader( test_dataset, batch_size=64, collate_fn=collate_fn ) 
 
	return train_loader, dev_loader, test_loader, lang