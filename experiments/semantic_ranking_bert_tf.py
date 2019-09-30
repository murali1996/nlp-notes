from sklearn.metrics.pairwise import euclidean_distances as eu_dist
import matplotlib.pyplot as plt,pandas as pd #for eval stats
from sklearn.metrics import accuracy_score
import numpy as np, os, math, re
import tensorflow as tf
from tqdm import tqdm
import copy
import random
import sys

from helpers import load_data, split_data
from bert_wrapper_tf.wrapper import Model


# ======================================================================
# Some Utility methods
# ======================================================================
def progressBar(value, endvalue, names, values, bar_length=20):
	assert(len(names)==len(values));
	percent = float(value) / endvalue
	arrow = '-' * int(round(percent * bar_length)-1) + '>'
	spaces = ' ' * (bar_length - len(arrow));
	string = '';
	for name, value in zip(names,values):
		temp = '|| {0}: {1:.4f} '.format(name, value);
		string+=temp;
	sys.stdout.write("\rPercent: [{0}] {1}% {2}".format(arrow + spaces, int(round(percent * 100)), string))
	sys.stdout.flush()
	return
def process_txt(list_sentences):
	list_sentences = copy.deepcopy(list_sentences)
	for i, sent in enumerate(list_sentences):
		sent = sent.strip()
		sent_tokens = re.findall(r"\w+|[-+]?\d*[.-]\d+|\d+|[A-Za-z0-9_-]+|[?!\)\(]", sent)
		sent = " ".join(sent_tokens)
		list_sentences[i] = sent
	return list_sentences
def get_random_negatives(y_raw, y_unique_labels=[], list_size=1):
	y_unique_labels = copy.deepcopy(y_unique_labels)
	if len(y_unique_labels)==0:
		_, idx = np.unique(y_raw, return_index=True)
		y_unique_labels = np.asarray(y_raw[np.sort(idx)])
	else:
		set_a = set(y_unique_labels)
		set_b = set(np.unique(y_raw))
		if len(set_b-set_a)!=0:
			raise Exception("{} names in y_raw unavailable in y_unique_labels: {}".format(len(set_b-set_a),set_b-set_a))
	if not isinstance(y_unique_labels, list):
		y_unique_labels = y_unique_labels.tolist()
	assert list_size<len(y_unique_labels)
	#
	y_raw_negatives =  []
	for _, y_ in tqdm(enumerate(y_raw)):
		possible_negs = y_unique_labels.remove(y_)
		neg_samples = np.random.choice(y_unique_labels, size=list_size, replace=False).tolist()
		if list_size==1:
			neg_samples = neg_samples[0]
		y_raw_negatives.append(neg_samples)
		y_unique_labels+=[y_]
	return y_raw_negatives
def get_nonrandom_negatives(y_raw, dist_mat, y_unique_labels, list_size=1):
	"""
	NOTE:
	Since the dist_mat column indices correspond to preordered y_unique_labels, you must give 
	appropriate y_unique_labels
	"""
	dist_mat = copy.deepcopy(dist_mat)
	y_unique_labels = copy.deepcopy(y_unique_labels)
	assert len(y_raw)==dist_mat.shape[0]
	assert dist_mat.shape[1]==len(y_unique_labels)
	if not isinstance(y_unique_labels, list):
		y_unique_labels = y_unique_labels.tolist()
	assert list_size<len(y_unique_labels)
	#
	y_raw_negatives =  []
	for i, y_ in tqdm(enumerate(y_raw)):
		dist_row = dist_mat[i,:].reshape(-1)
		mask_this_idx = y_unique_labels.index(y_)  #np.where(y_unique_labels==y_)[0][0]
		dist_row[mask_this_idx] = 1
		dist_row = 1/dist_row
		dist_row[mask_this_idx] = 0
		dist_row = dist_row/np.sum(dist_row)
		neg_samples = np.random.choice(y_unique_labels, size=list_size, replace=False, p=dist_row).tolist()
		if list_size==1:
			neg_samples = neg_samples[0]
		y_raw_negatives.append(neg_samples)
	return y_raw_negatives
def get_hard_negatives(y_raw, dist_mat, y_unique_labels, list_size=1):
	#print("IMPORTANT:: indices along each row in dist_mat MUST correspond to the order of labels in y_unique_labels")
	assert len(y_raw)==dist_mat.shape[0]
	assert dist_mat.shape[1]==len(y_unique_labels)
	if type(y_unique_labels)!=list:
		y_unique_labels = y_unique_labels.tolist()
	assert list_size<len(y_unique_labels)
	#
	y_raw_negatives =  []
	for i, y_ in tqdm(enumerate(y_raw)):
		idx = y_unique_labels.index(y_)
		dist_row = dist_mat[i,:].reshape(-1)
		min_idxs = np.argsort(dist_row) # ascending order
		if list_size>1:
			neg_sample_idxs = min_idxs[:list_size].tolist()
			if idx in neg_sample_idxs:
				neg_sample_idxs = min_idxs[:list_size+1].tolist()
				neg_sample_idxs.remove(idx)
			neg_samples = []
			for ind in neg_sample_idxs:
				neg_samples.append(y_unique_labels[ind])
		else:
			best_k = min_idxs[:15]
			try: #idx may or maynot be there!
				best_k.remove(idx)
			except:
				pass
			neg_sample_idx = np.random.choice(best_k, size=1, replace=False).tolist()[0]
			neg_samples = y_unique_labels[neg_sample_idx]
		y_raw_negatives.append(neg_samples)
	return y_raw_negatives



# ======================================================================
# Some methods with respect to bert_wrapper_tf.wrapper
# ======================================================================
def get_representations_type1(model, sess, find_query_vectors, inputFeatures, BATCH_SIZE):
	#
	get_this_attribute = getattr(model, "qvec_prime_infer") if find_query_vectors else getattr(model, "pvec_prime_infer")
	print("Obtaining vectors from the attribute: {}".format(get_this_attribute))
	#
	vectors = []
	INFER_NUM = len(inputFeatures)
	print("No.of sentences for obtaining representation vectors: {}".format(INFER_NUM))
	n_infer_batches = int(math.ceil(INFER_NUM/BATCH_SIZE))
	all_indices = np.arange(INFER_NUM)
	for i in tqdm(range(n_infer_batches)):
		begin_index = i * BATCH_SIZE
		end_index = min((i + 1) * BATCH_SIZE, INFER_NUM)
		batch_index = all_indices[begin_index : end_index]
		#
		batch_input_ids = np.asarray([f.input_ids for f in inputFeatures[batch_index]], dtype=np.int32)
		batch_input_mask = np.asarray([f.input_mask for f in inputFeatures[batch_index]], dtype=np.int32)
		batch_token_type_ids = np.asarray([f.segment_ids for f in inputFeatures[batch_index]], dtype=np.int32)
		#
		batch_vectors = \
			sess.run(get_this_attribute,
						feed_dict={
									model.batch_input_ids__tensor:batch_input_ids,
									model.batch_input_mask__tensor:batch_input_mask,
									model.batch_token_type_ids__tensor:batch_token_type_ids,
									model.part_size:len(batch_index)
								  }
					)
		if i==0:
			vectors = batch_vectors;
		else:
			vectors = np.concatenate((vectors, batch_vectors), axis=0)
	print("Shape Obtained: {}".format(vectors.shape))
	return np.asarray(vectors)
def get_bert_prediction_scores_type1(x_raw, y_raw, model, sess, batch_size,
									 y_raw_merged=[], excel_title="", ckpt_dir=None, return_thresScores=False):
	assert len(x_raw)==len(y_raw)
	x_raw = copy.deepcopy(x_raw)
	y_raw = copy.deepcopy(y_raw)
	y_raw_merged = copy.deepcopy(y_raw_merged)
	#
	if not len(y_raw_merged)==0:
		try:
			assert len(set(y_raw)-set(y_raw_merged))==0
		except:
			raise Exception("y_raw has lables not available in non-empty y_raw_merged")
		_, idx = np.unique(y_raw_merged, return_index=True)
		y_raw_uniques = np.asarray(y_raw_merged[np.sort(idx)])
	else:
		_, idx = np.unique(y_raw, return_index=True)
		y_raw_uniques = np.asarray(y_raw[np.sort(idx)])
	y_tr = []
	for label in y_raw:
		y_tr.append(np.where(y_raw_uniques==label)[0][0])
	y_tr = np.asarray(y_tr)
	#
	inputFeatures_query = np.asarray( model.get_InputFeatures(text_a_list=process_txt(x_raw)) )
	vectors_query = get_representations_type1(model=model, sess=sess, find_query_vectors=True, inputFeatures=inputFeatures_query, BATCH_SIZE=batch_size)
	inputFeatures_label = np.asarray( model.get_InputFeatures(text_a_list=process_txt(y_raw_uniques)) )
	vectors_label = get_representations_type1(model=model, sess=sess, find_query_vectors=False, inputFeatures=inputFeatures_label, BATCH_SIZE=batch_size)
	dist = eu_dist(vectors_query, vectors_label)
	y_tr_pred = np.argmin(dist, axis=-1)
	acc = accuracy_score(y_tr, y_tr_pred)
	print("Accuarcy", acc)
	#
	idx_dist = np.argsort(dist)
	ranks_ = []
	for i in range(len(idx_dist)):
		ranks_.append(np.where(idx_dist[i,:]==y_tr[i])[0][0])
	ranks_ = np.asarray(ranks_)+1 # converting to 1 base
	print("Mean Rank: {}, Rank Standard Deviation: {}".format(np.mean(ranks_),np.std(ranks_)))
	#
	if not (excel_title=="" or ckpt_dir==None):
		# Print to excel: Query | Predicted Label(PL) 1 | PL 2 | PL 3 | True Label | Correct Prediction(True/False)
		excel = []
		for i in range(len(x_raw)):
			excel.append([x_raw[i],y_raw_uniques[idx_dist[i,0]],y_raw_uniques[idx_dist[i,1]],y_raw_uniques[idx_dist[i,2]],y_raw[i],y_tr[i]==y_tr_pred[i]])
		df = pd.DataFrame(excel, columns=["Query","PL1","PL2","PL3","True Label","Is Correct prediction?"])
		filepath = os.path.join(ckpt_dir, excel_title+".xlsx")
		df.to_excel(filepath, index=False)
	#
	if return_thresScores:
		label_scores = {}
		labels_thresMax = []
		labels_thresMin = []
		for i, label in enumerate(y_raw_uniques):
			label_scores[label] = []
			rows = np.where(y_raw==label)[0]
			rows = rows[np.where(y_tr[rows]==y_tr_pred[rows])[0]] 
			cols = np.where(y_raw_uniques==label)[0]
			if len(rows)==0 or len(cols)==0:
				#print(np.where(y_raw==label)[0], rows, cols, label)
				print("Probably, there are no true positives for this class-label!!! Label-{}: {}".format(i,label))
			else:
				label_scores[label] = dist[rows,cols].reshape(-1).tolist()
				labels_thresMax.append(max(label_scores[label]))
				labels_thresMin.append(min(label_scores[label]))	
		return acc, ranks_, y_raw_uniques, vectors_query, vectors_label, label_scores, labels_thresMax, labels_thresMin
	return acc, ranks_, y_raw_uniques, vectors_query, vectors_label



# ======================================================================
# Make data folder and load data
# ======================================================================
def data_func():
	DATASET_NAME = "$$$$$$$_all_data" #"$$$$$$$_all_data" #"$$$$$$$_vde_data"
	DATASET_TXT =  "all_training.txt" #"all_training.txt" #"vde_training.txt"
	SRC_DATA_DIR = '../../data/'
	DST_DATA_DIR = './data/{}'.format(DATASET_NAME)
	if not os.path.exists(DST_DATA_DIR):
		split_data.seperate_seen_unseen_classes(
			inp_as_list = True,
			src_dir = os.path.join(SRC_DATA_DIR, '_$$$$$$$_data'),
			file_paths = [DATASET_TXT],
			dest_dir = DST_DATA_DIR, #os.path.join(SRC_DATA_DIR, DATASET_NAME)
			lst=None,
			seen_size=0.7)
		split_data.seperate_train_validation(
			inp_as_list =True,
			src_dir = DST_DATA_DIR, #os.path.join(SRC_DATA_DIR, DATASET_NAME)
			file_path = "train_shuffle.txt",
			lst=None,
			train_size=.7)
	data = load_data.Data(use3_directory=None, elmo2_directory=None, glove_directory=None)
	data( data_dir = DST_DATA_DIR,  files = ["train_shuffle.txt","validation_shuffle.txt","test.txt"], codes= ["tr","va","te"])
	return data


if __name__=="__main__":

	data = data_func()


	# ======================================================================
	# BERT CLS embedding with Triplet/Pairwise Loss objective
	# ======================================================================
	# ======================================================================
	# Like with SWAG dataset modeling (in bERT paper), we concat a BATCH_SIZE 
	# of queries, a BATCH_SIZE of corresponding positive lables and a 
	# BATCH_SIZE of corresponding negative lables (3 items in total)
	# to BERT pre-trained architecture.
	# ======================================================================
	# ======================================================================
	# Training
	# ======================================================================
	tf_config = tf.ConfigProto()
	tf_config.allow_soft_placement = True
	tf_config.log_device_placement = False
	tf_config.gpu_options.allow_growth = True
	tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
	tf.logging.set_verbosity(tf.logging.INFO)
	#
	tr_measures = { "TRAIN_BATCH_SIZE":50,
	 				"START_EPOCH":0,
	 				"N_EPOCHS":500,
	 				"SAVE_EVERY_Nth_EPOCH":50,
	 				"epoch_losses":[], "epoch_L2dists":[], "lowest_loss":None, "lowest_loss_epoch":None, 
					"epoch_acc":[], "highest_acc":None, "highest_acc_epoch":None
					}
	va_measures = { "VAL_BATCH_SIZE":50,
					"VAL_EVERY_Nth_EPOCH":10,
					"epoch_acc":[], "highest_acc":None, "highest_acc_epoch":None
					}
	te_measures = { "epoch_acc":[], "highest_acc":None, "highest_acc_epoch":None }
	#
	restore_ckpt_dir = None #"./checkpoints/allData_TripletLoss_01/uncased_L-12_H-768_A-12"
	restore_model_name = None #"wsm_data_with_hard_negs_TripletLoss_lowest_loss_model_0.ckpt"
	save_ckpt_dir = "./checkpoints/allData_SquashAndTripletLoss_01" #"./checkpoints/allData_TripletLoss_01_vde"
	save_model_name = "bertModel.ckpt"
	#
	train_model = Model(gpu_devices=[1,2])
	train_model.restore_pretrained_bert_config(ckpt_dir=restore_ckpt_dir, max_seq_len=35, cased=None)
	train_model.set_base_ops(is_training=True)
	#train_model.set_custom_ops_TripletLoss(is_training=True)
	train_model.set_custom_ops_SquashAndTripletLoss(is_training=True)
	train_sess = tf.Session(graph = train_model.tf_graph, config=tf_config)
	train_sess.__enter__()
	train_model.restore_weights(sess=train_sess, model_name=restore_model_name, ckpt_dir=restore_ckpt_dir)
	#
	#
	infer_model = Model(gpu_devices=[2,1])
	infer_model.restore_pretrained_bert_config(ckpt_dir=restore_ckpt_dir, max_seq_len=35, cased=None)
	infer_model.set_base_ops(is_training=False)
	#infer_model.set_custom_ops_TripletLoss(is_training=False)
	infer_model.set_custom_ops_SquashAndTripletLoss(is_training=False)
	infer_sess = tf.Session(graph = infer_model.tf_graph, config=tf_config)
	infer_sess.__enter__()
	#
	try:
		x_tr_processed = process_txt(data.x_tr_raw)
		x_tr_inputFeatures = np.asarray( train_model.get_InputFeatures(text_a_list=x_tr_processed) )
		x_tr_inputFeatureTokenized = np.asarray([f.joined_tokens for f in x_tr_inputFeatures])
		y_tr_processed = process_txt(data.y_tr_raw)
		y_tr_inputFeatures = np.asarray( train_model.get_InputFeatures(text_a_list=y_tr_processed) )
		y_tr_inputFeatureTokenized = np.asarray([f.joined_tokens for f in y_tr_inputFeatures])
		_, idx = np.unique(data.y_tr_raw, return_index=True)
		unique_y_tr_raw = np.asarray(data.y_tr_raw[np.sort(idx)])
		unique_y_tr_processed = process_txt(unique_y_tr_raw)
		unique_y_tr_inputFeatures = np.asarray( train_model.get_InputFeatures(text_a_list=unique_y_tr_processed) )
		unique_y_tr_inputFeatureTokenized = np.asarray([f.joined_tokens for f in unique_y_tr_inputFeatures])
		train_model.dump_tokenized_sents(
			data.x_tr_raw, x_tr_processed, x_tr_inputFeatureTokenized, data.y_tr_raw, y_tr_processed, y_tr_inputFeatureTokenized,
			column_names=["raw_query","processed_query","tokenized_query","raw_label","processed_label","tokenized_label"],
			file_title="tr_data",
			ckpt_dir=save_ckpt_dir
			)
	except:
		raise Exception("tr data could not be made")
	try:
		x_va_processed = process_txt(data.x_va_raw)
		x_va_inputFeatures = np.asarray( train_model.get_InputFeatures(text_a_list=x_va_processed) )
		x_va_inputFeatureTokenized = np.asarray([f.joined_tokens for f in x_va_inputFeatures])
		y_va_processed = process_txt(data.y_va_raw)
		y_va_inputFeatures = np.asarray( train_model.get_InputFeatures(text_a_list=y_va_processed) )
		y_va_inputFeatureTokenized = np.asarray([f.joined_tokens for f in y_va_inputFeatures])
		_, idx = np.unique(data.y_va_raw, return_index=True)
		unique_y_va_raw = np.asarray(data.y_va_raw[np.sort(idx)])
		unique_y_va_processed = process_txt(unique_y_va_raw)
		unique_y_va_inputFeatures = np.asarray( train_model.get_InputFeatures(text_a_list=unique_y_va_processed) )
		unique_y_va_inputFeatureTokenized = np.asarray([f.joined_tokens for f in unique_y_va_inputFeatures])
		train_model.dump_tokenized_sents(
			data.x_va_raw, x_va_processed, x_va_inputFeatureTokenized, data.y_va_raw, y_va_processed, y_va_inputFeatureTokenized,
			column_names=["raw_query","processed_query","tokenized_query","raw_label","processed_label","tokenized_label"],
			file_title="va_data",
			ckpt_dir=save_ckpt_dir
			)
	except:
		raise Exception("va data could not be made")
	try:
		x_te_processed = process_txt(data.x_te_raw)
		x_te_inputFeatures = np.asarray( train_model.get_InputFeatures(text_a_list=x_te_processed) )
		x_te_inputFeatureTokenized = np.asarray([f.joined_tokens for f in x_te_inputFeatures])
		y_te_processed = process_txt(data.y_te_raw)
		y_te_inputFeatures = np.asarray( train_model.get_InputFeatures(text_a_list=y_te_processed) )
		y_te_inputFeatureTokenized = np.asarray([f.joined_tokens for f in y_te_inputFeatures])
		_, idx = np.unique(data.y_te_raw, return_index=True)
		unique_y_te_raw = np.asarray(data.y_te_raw[np.sort(idx)])
		unique_y_te_processed = process_txt(unique_y_te_raw)
		unique_y_te_inputFeatures = np.asarray( train_model.get_InputFeatures(text_a_list=unique_y_te_processed) )
		unique_y_te_inputFeatureTokenized = np.asarray([f.joined_tokens for f in unique_y_te_inputFeatures])
		train_model.dump_tokenized_sents(
			data.x_te_raw, x_te_processed, x_te_inputFeatureTokenized, data.y_te_raw, y_te_processed, y_te_inputFeatureTokenized,
			column_names=["raw_query","processed_query","tokenized_query","raw_label","processed_label","tokenized_label"],
			file_title="te_data",
			ckpt_dir=save_ckpt_dir
			)
	except:
		raise Exception("te data could not be made")
	#
	for epoch in np.arange(tr_measures["START_EPOCH"], tr_measures["N_EPOCHS"]):
		print("=======================================================================")
		print("=======================================================================")
		tf.logging.info("Epoch: {}".format(epoch))
		#
		# negative samples generation
		if epoch<=10:
			tf.logging.info("Generating Random Negative Samples")
			y_tr_negs_processed = get_random_negatives(y_tr_processed, y_unique_labels=unique_y_tr_processed, list_size=1)
		elif epoch>10 and epoch<=20:
			if random.uniform(0, 1)>=0.5:
				tf.logging.info("Generating Random Negative Samples")
				y_tr_negs_processed = get_random_negatives(y_tr_processed, y_unique_labels=unique_y_tr_processed, list_size=1)
			else:
				tf.logging.info("Generating Non-Random Negative Samples")
				vectors1 = get_representations_type1(model=train_model, sess=train_sess, find_query_vectors=True, inputFeatures=x_tr_inputFeatures, BATCH_SIZE=va_measures["VAL_BATCH_SIZE"])
				vectors2 = get_representations_type1(model=train_model, sess=train_sess, find_query_vectors=False, inputFeatures=unique_y_tr_inputFeatures, BATCH_SIZE=va_measures["VAL_BATCH_SIZE"])
				dist_mat = eu_dist(vectors1, vectors2)					
				y_tr_negs_processed = get_nonrandom_negatives(y_tr_processed, dist_mat, y_unique_labels=unique_y_tr_processed, list_size=1)
		elif epoch>20:
			if random.uniform(0, 1)<=0.15:
				tf.logging.info("Generating Random Negative Samples")
				y_tr_negs_processed = get_random_negatives(y_tr_processed, y_unique_labels=unique_y_tr_processed, list_size=1)
			else:
				if random.uniform(0, 1)<=0.90:
					tf.logging.info("Generating Non-Random Negative Samples")
					vectors1 = get_representations_type1(model=train_model, sess=train_sess, find_query_vectors=True, inputFeatures=x_tr_inputFeatures, BATCH_SIZE=va_measures["VAL_BATCH_SIZE"])
					vectors2 = get_representations_type1(model=train_model, sess=train_sess, find_query_vectors=False, inputFeatures=unique_y_tr_inputFeatures, BATCH_SIZE=va_measures["VAL_BATCH_SIZE"])
					dist_mat = eu_dist(vectors1, vectors2)					
					y_tr_negs_processed = get_nonrandom_negatives(y_tr_processed, dist_mat, y_unique_labels=unique_y_tr_processed, list_size=1)					
				else:
					tf.logging.info("Generating Hard Negative Samples")
					vectors1 = get_representations_type1(model=train_model, sess=train_sess, find_query_vectors=True, inputFeatures=x_tr_inputFeatures, BATCH_SIZE=va_measures["VAL_BATCH_SIZE"])
					vectors2 = get_representations_type1(model=train_model, sess=train_sess, find_query_vectors=False, inputFeatures=unique_y_tr_inputFeatures, BATCH_SIZE=va_measures["VAL_BATCH_SIZE"])
					dist_mat = eu_dist(vectors1, vectors2)
					y_tr_negs_processed = get_hard_negatives(y_tr_processed, dist_mat, y_unique_labels=unique_y_tr_processed, list_size=1)				
		y_tr_negs_inputFeatures = np.asarray( train_model.get_InputFeatures(text_a_list=y_tr_negs_processed) )
		assert len(x_tr_inputFeatures)==len(y_tr_inputFeatures)==len(y_tr_negs_inputFeatures)
		#
		TRAIN_NUM = len(x_tr_inputFeatures)
		n_train_batches = int(math.ceil(TRAIN_NUM/tr_measures["TRAIN_BATCH_SIZE"]))
		all_indices = np.arange(TRAIN_NUM)
		np.random.shuffle(all_indices)
		epoch_loss = 0
		epoch_pdist = 0
		epoch_ndist = 0
		for i in tqdm(range(n_train_batches)):
			begin_index = i * tr_measures["TRAIN_BATCH_SIZE"]
			end_index = min((i + 1) * tr_measures["TRAIN_BATCH_SIZE"], TRAIN_NUM)
			batch_index = all_indices[begin_index : end_index]
			batch_inputFeatures_tr = np.hstack([ x_tr_inputFeatures[batch_index],
				                                 y_tr_inputFeatures[batch_index],
				                                 y_tr_negs_inputFeatures[batch_index]
				                               ])
			batch_input_ids = np.asarray([f.input_ids for f in batch_inputFeatures_tr], dtype=np.int32)
			batch_input_mask = np.asarray([f.input_mask for f in batch_inputFeatures_tr], dtype=np.int32)
			batch_token_type_ids = np.asarray([f.segment_ids for f in batch_inputFeatures_tr], dtype=np.int32)
			#
			_, mean_loss, mean_pdist, mean_ndist = \
				train_sess.run([train_model.train_op, train_model.loss, train_model.pdist, train_model.ndist],
							feed_dict={ train_model.batch_input_ids__tensor:batch_input_ids,
										train_model.batch_input_mask__tensor:batch_input_mask,
										train_model.batch_token_type_ids__tensor:batch_token_type_ids,
										train_model.part_size:len(batch_index),
										train_model.bert_lr:0.00002, #2e-5
										train_model.custom_lr:0.000075, #7.5e-5
										train_model.margin_gamma:5
									  }
						)
			epoch_loss+=mean_loss
			epoch_pdist+=mean_pdist
			epoch_ndist+=mean_ndist
		epoch_loss/=n_train_batches
		epoch_pdist/=n_train_batches
		epoch_ndist/=n_train_batches
		tf.logging.info("Lowest Epoch Loss until now: {} at epoch: {}". \
			format(tr_measures["lowest_loss"],tr_measures["lowest_loss_epoch"]))
		tf.logging.info("Epoch Loss (avg. over all batches): {}".format(epoch_loss))
		tf.logging.info("Epoch pos distance (avg. over all batches): {}".format(epoch_pdist))
		tf.logging.info("Epoch neg distance (avg. over all batches): {}".format(epoch_ndist))
		tr_measures["epoch_losses"].append((epoch, epoch_loss))
		tr_measures["epoch_L2dists"].append((epoch, epoch_pdist, epoch_ndist))
		#
		if epoch%tr_measures["SAVE_EVERY_Nth_EPOCH"]==0:
			tf.logging.info("Saving weights because of SAVE_EVERY_Nth_EPOCH")
			model_name = "".join(save_model_name.split(".")[:-1])+"_epoch{}Model.ckpt".format(epoch)
			train_model.save_weights(train_sess, model_name=model_name, ckpt_dir=save_ckpt_dir)
		if epoch==0 or epoch_loss<=tr_measures["lowest_loss"]:
			tr_measures["lowest_loss"], tr_measures["lowest_loss_epoch"] = epoch_loss, epoch
			tf.logging.info("Saving weights because of lowest_loss while training")
			model_name = "".join(save_model_name.split(".")[:-1])+"_lowestTrainLossModel.ckpt"
			train_model.save_weights(train_sess, model_name=model_name, ckpt_dir=save_ckpt_dir)
		if epoch%va_measures["VAL_EVERY_Nth_EPOCH"]==0:
			#
			# save weights temporarily
			tf.logging.info("Saving weights temporarily so as to obtain inference results")
			model_name_temporary = "".join(save_model_name.split(".")[:-1])+"_temporary.ckpt"
			train_model.save_weights(train_sess, model_name=model_name_temporary, ckpt_dir=save_ckpt_dir)			
			#
			# load weights temporarily
			tf.logging.info("Loading weights temporarily so as to obtain inference results")
			infer_model.restore_weights(sess=infer_sess, model_name=model_name_temporary, ckpt_dir=save_ckpt_dir)
			#
			acc1, _, _, _, _ = get_bert_prediction_scores_type1(data.x_tr_raw, data.y_tr_raw, infer_model, 
																infer_sess, batch_size=va_measures["VAL_BATCH_SIZE"])
			tf.logging.info("Highest Train Accuracy until now: {} at epoch: {}".\
								format(tr_measures["highest_acc"],tr_measures["highest_acc_epoch"]))
			tf.logging.info("Epoch Train Accuracy: {}".format(acc1))
			tr_measures["epoch_acc"].append((epoch, acc1))
			if epoch==0 or tr_measures["highest_acc"]<=acc1:
				tr_measures["highest_acc"], tr_measures["highest_acc_epoch"] = acc1, epoch
			#
			acc2, _, _, _, _ = get_bert_prediction_scores_type1(data.x_va_raw, data.y_va_raw, infer_model, 
																infer_sess, batch_size=va_measures["VAL_BATCH_SIZE"])
			tf.logging.info("Highest Validation Accuracy until now: {} at epoch: {}".\
								format(va_measures["highest_acc"],va_measures["highest_acc_epoch"]))
			tf.logging.info("Epoch Validation Accuracy: {}".format(acc2))
			va_measures["epoch_acc"].append((epoch, acc2))
			if epoch==0 or va_measures["highest_acc"]<=acc2:
				va_measures["highest_acc"], va_measures["highest_acc_epoch"] = acc2, epoch
			#
			acc3, _, _, _, _ = get_bert_prediction_scores_type1(data.x_te_raw, data.y_te_raw, infer_model, 
																infer_sess, batch_size=va_measures["VAL_BATCH_SIZE"])
			tf.logging.info("Highest Test Accuracy until now: {} at epoch: {}".\
								format(te_measures["highest_acc"],te_measures["highest_acc_epoch"]))
			tf.logging.info("Epoch Test Accuracy: {}".format(acc3))
			te_measures["epoch_acc"].append((epoch, acc3))
			if epoch==0 or te_measures["highest_acc"]<=acc3:
				te_measures["highest_acc"], te_measures["highest_acc_epoch"] = acc3, epoch
		#
		tf.logging.info("Dumping tr_, va_ and te_ measures in temp files for record-keeping")
		train_model.dump_json(tr_measures, "tr_measures_temp", open_mode="w", ckpt_dir=save_ckpt_dir)
		train_model.dump_json(va_measures, "va_measures_temp", open_mode="w", ckpt_dir=save_ckpt_dir)
		train_model.dump_json(te_measures, "te_measures_temp", open_mode="w", ckpt_dir=save_ckpt_dir)
	#
	tf.logging.info("Training Complete...")
	train_model.dump_json(tr_measures, "tr_measures", open_mode="a", ckpt_dir=save_ckpt_dir)
	train_model.dump_json(va_measures, "va_measures", open_mode="a", ckpt_dir=save_ckpt_dir)
	train_model.dump_json(te_measures, "te_measures", open_mode="a", ckpt_dir=save_ckpt_dir)
	train_sess.close()
	# ======================================================================
	# Inferencel; consists of methods more than just validation
	# ======================================================================
	do_evaluation = False 
	if do_evaluation:
		tf_config = tf.ConfigProto()
		tf_config.allow_soft_placement = True
		tf_config.log_device_placement = False
		tf_config.gpu_options.allow_growth = True
		tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
		#
		batch_size = 32
		#
		# :examples:
		# "./checkpoints/allData_with_hardNegatives_tripletLoss_01/uncased_L-12_H-768_A-12"
		# "wsm_data_with_hard_negs_TripletLoss_lowest_loss_model_0.ckpt"
		restore_ckpt_dir = "./checkpoints/allData_with_hardNegatives_tripletLoss_01_vde" 
		restore_model_name = "bertModel_lowestTrainLossModel.ckpt" 
		model = Model(gpu_devices=[2])
		model.restore_pretrained_bert_config(ckpt_dir=restore_ckpt_dir, max_seq_len=35, cased=None)
		model.set_base_ops(is_training=False)
		model.set_custom_ops_TripletLoss(is_training=False)
		sess = tf.Session(graph = model.tf_graph, config=tf_config)
		sess.__enter__()
		model.restore_weights(sess=sess, model_name=restore_model_name, ckpt_dir=restore_ckpt_dir)
		#
		# Test Labels and Train Labels aren't combined for predictions (a less practical scenario)
		tr_acc, ranks_tr, tr_labels, _, vectors_tr_label, tr_labels_thresList, tr_labels_thresMax, tr_labels_thresMin = \
			get_bert_prediction_scores_type1(data.x_tr_raw, data.y_tr_raw, model, sess, batch_size=batch_size, 
											 excel_title="train_set_uncombinedLabels", 
											 ckpt_dir=restore_ckpt_dir, return_thresScores=True)
		va_acc, ranks_va, va_labels, _, _, va_labels_thresList, va_labels_thresMax, va_labels_thresMin= \
			get_bert_prediction_scores_type1(data.x_va_raw, data.y_va_raw, model, sess, batch_size=batch_size,
											 excel_title="validation_set_uncombinedLabels",
											 ckpt_dir=restore_ckpt_dir, return_thresScores=True)
		te_acc, ranks_te, _, vectors_te_query, _ = \
			get_bert_prediction_scores_type1(data.x_te_raw, data.y_te_raw, model, sess, batch_size=batch_size,
											 excel_title="test_set_uncombinedLabels",
											 ckpt_dir=restore_ckpt_dir)
		#
		# Test Labels and Train Labels are combined for predictions (a more practical scenario)
		all_labels = np.hstack([data.y_tr_raw,data.y_va_raw,data.y_te_raw])
		tr_acc, ranks_tr, _, _, _ = \
			get_bert_prediction_scores_type1(data.x_tr_raw, data.y_tr_raw, model, sess, batch_size=batch_size, y_raw_merged=all_labels, excel_title="train_set_combinedLabels", ckpt_dir=restore_ckpt_dir)
		va_acc, ranks_va, _, _, _ = \
			get_bert_prediction_scores_type1(data.x_va_raw, data.y_va_raw, model, sess, batch_size=batch_size, y_raw_merged=all_labels, excel_title="validation_set_combinedLabels", ckpt_dir=restore_ckpt_dir)
		te_acc, ranks_te, _, _, _ = \
			get_bert_prediction_scores_type1(data.x_te_raw, data.y_te_raw, model, sess, batch_size=batch_size, y_raw_merged=all_labels, excel_title="test_set_combinedLabels", ckpt_dir=restore_ckpt_dir)
		#
		# Compute Known Class Thresholds for New Intent Detection
		# For each class label, obtain the train and validation paraphrases' L2 distances
		assert np.all(tr_labels==va_labels)
		known_class_thres = np.ceil(np.maximum(tr_labels_thresMax, va_labels_thresMax))
		dist_te_nid = eu_dist(vectors_te_query, vectors_tr_label)
		out_faq_preds = []
		problematic_preds = {}
		for i, _ in enumerate(dist_te_nid):
			known_class_dist = dist_te_nid[i,:]
			if np.all(known_class_dist>known_class_thres):
				out_faq_preds.append(1)
				problematic_preds[i] = []
			else:
				problematic_preds[i] = known_class_dist[np.where(known_class_dist<=known_class_thres)[0]].tolist()
				out_faq_preds.append(0)
		out_faq_preds = np.asarray(out_faq_preds)
		print("OUT-FAQ accuarcy: {}".format(np.sum(out_faq_preds)/len(dist_te_nid)))
		#
		sess.close()
	

