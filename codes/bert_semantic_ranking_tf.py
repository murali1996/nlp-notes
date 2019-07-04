

import numpy as np, os, math, re
import pandas as pd #for eval stats
import matplotlib.pyplot as plt #for eval stats
import copy
import random
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances as eu_dist
from sklearn.metrics import accuracy_score
import tensorflow as tf

from helpers import load_data, split_data
import sys
from wrappers.bert_wrapper_tf.wrapper import Model



# Some Utility methods
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
# Some negative sampling methods
def get_random_negatives(x_tr_raw, y_tr_raw, y_unique_labels=[], list_size=1):
	#
	print("Generating Random Negative Samples Begin.....")
	assert len(x_tr_raw)==len(y_tr_raw)
	if len(y_unique_labels)==0:
		y_unique_labels = np.unique(y_tr_raw)
	else:
		set_a = set(y_unique_labels)
		set_b = set(np.unique(y_tr_raw))
		if len(set_b-set_a)!=0:
			raise Exception("{} names in y_tr_raw unavailable in y_unique_labels: {}".format(len(set_b-set_a),set_b-set_a))
		y_unique_labels = y_unique_labels
	assert list_size<len(y_unique_labels)
	if type(y_unique_labels)!=list:
		y_unique_labels = y_unique_labels.tolist()
	#
	y_tr_raw_negatives =  []
	for y_ in tqdm(y_tr_raw):
		possible_negs = y_unique_labels.remove(y_)
		neg_samples = np.random.choice(y_unique_labels, size=list_size, replace=False).tolist()
		if list_size==1:
			neg_samples = neg_samples[0]
		y_tr_raw_negatives.append(neg_samples)
		y_unique_labels+=[y_]
	print("Generating Random Negative Samples End.....")
	return y_tr_raw_negatives
def get_hard_negatives(x_tr_raw, y_tr_raw, dist_mat, y_unique_labels, list_size=1):
	#
	print("Generating Hard Negative Samples Begin.....")
	print("IMPORTANT:: indices along each row in dist_mat MUST correspond to the order of labels in y_unique_labels")
	assert len(x_tr_raw)==len(y_tr_raw)==dist_mat.shape[0]
	assert dist_mat.shape[1]==len(y_unique_labels)
	assert list_size<len(y_unique_labels)
	if type(y_unique_labels)!=list:
		y_unique_labels = y_unique_labels.tolist()
	#
	y_tr_raw_negatives =  []
	for i, y_ in tqdm(enumerate(y_tr_raw)):
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
			best_k = min_idxs[:3]
			try: #idx may or maynot be there!
				best_k.remove(idx)
			except:
				pass
			neg_sample_idx = np.random.choice(best_k, size=1, replace=False).tolist()[0]
			neg_samples = y_unique_labels[neg_sample_idx]
		y_tr_raw_negatives.append(neg_samples)
	print("Generating Hard Negative Samples End.....")
	return y_tr_raw_negatives
# Some methods with respect to bert_wrapper_tf.wrapper
def get_representations_type1(
	model: "object of bert_wrapper_tf.wrapper after executing init ops",
	sess: "tf session",
	find_query_vectors: "True for query vectors and False for label vectors",
	inputFeatures: "array of objects of inputFeatures",
	BATCH_SIZE):
	#
	get_this_attribute = getattr(model, "qvec_prime_inference") if find_query_vectors else getattr(model, "pvec_prime_inference")
	print("Obtaing vectors from the attribute: {}".format(get_this_attribute))
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
def get_bert_prediction_scores_type1(x_raw, y_raw, model, sess, batch_size, y_raw_merged=[], excel_title="", ckpt_dir=None, return_thresScores=False):
	assert len(x_raw)==len(y_raw)
	x_raw = copy.deepcopy(x_raw)
	y_raw = copy.deepcopy(y_raw)
	#
	if not len(y_raw_merged)==0:
		y_raw_merged = copy.deepcopy(y_raw_merged)
		# if isinstance(y_raw_merged, np.ndarray): y_raw_merged = y_raw_merged.tolist()
		try:
			assert len(set(y_raw)-set(y_raw_merged))==0
		except:
			raise Exception("y_raw has lables not available in non-empty y_raw_merged")
		_, idx = np.unique(y_raw_merged, return_index=True)
		idx_sort = np.sort(idx)
		y_raw_uniques = np.asarray(y_raw_merged[idx_sort])
	else:
		_, idx = np.unique(y_raw, return_index=True)
		idx_sort = np.sort(idx)
		y_raw_uniques = np.asarray(y_raw[idx_sort])
	y_tr = []
	for label in y_raw:
		y_tr.append(np.where(y_raw_uniques==label)[0][0])
	y_tr = np.asarray(y_tr)
	#
	inputFeatures_query = model.get_InputFeatures(text_a_list=process_txt(x_raw))
	inputFeatures_query = np.asarray(inputFeatures_query)
	vectors_query = get_representations_type1(model=model, sess=sess, find_query_vectors=True, inputFeatures=inputFeatures_query, BATCH_SIZE=batch_size)
	inputFeatures_label = model.get_InputFeatures(text_a_list=process_txt(y_raw_uniques))
	inputFeatures_label = np.asarray(inputFeatures_label)
	vectors_label = get_representations_type1(model=model, sess=sess, find_query_vectors=False, inputFeatures=inputFeatures_label, BATCH_SIZE=batch_size)
	dist = eu_dist(vectors_query, vectors_label)
	idx_dist = np.argsort(dist)
	pred_labels_idx = np.argmin(dist, axis=-1)
	#pred_labels = np.asarray(y_raw_uniques[pred_labels_idx])
	acc = accuracy_score(y_tr, pred_labels_idx)
	print("Accuarcy", acc)
	#
	ranks_ = []
	for i, _ in enumerate(idx_dist):
		ranks_.append(np.where(idx_dist[i,:]==y_tr[i])[0][0])
	ranks_ = np.asarray(ranks_)+1 # converting to 1 base
	print("Mean Rank: {}, Rank Standard Deviation: {}".format(np.mean(ranks_),np.std(ranks_)))
	#
	if not (excel_title=="" or ckpt_dir==None):
		# Print to excel: Query | Predicted Label(PL) 1 | PL 2 | PL 3 | True Label | Correct Prediction(True/False)
		excel = []
		idx3 = idx_dist[:,:3]
		pl3 = y_raw_uniques[idx3]
		for i, _ in enumerate(x_raw):
			excel.append([x_raw[i],pl3[i][0],pl3[i][1],pl3[i][2],y_raw[i],y_raw[i]==pl3[i][0]])
		df = pd.DataFrame(excel, columns=["Query","PL1","PL2","PL3","True Label","Is Correct prediction?"])
		filepath = os.path.join(ckpt_dir, excel_title+".xlsx")
		df.to_excel(filepath, index=False)
	#
	if return_thresScores:
		label_scores = {}
		labels_thresMax = []
		labels_thresMin = []
		for label in y_raw_uniques:
			label_scores[label] = []
			rows = np.where(y_raw==label)[0]
			# further collect correctly predicted rows only
			rows = rows[np.where(y_tr[rows]==pred_labels_idx[rows])[0]] 
			cols = np.where(y_raw_uniques==label)[0]
			label_scores[label] = dist[rows,cols].reshape(-1).tolist()
			labels_thresMax.append(max(label_scores[label]))
			labels_thresMin.append(min(label_scores[label]))	
		return acc, ranks_, y_raw_uniques, vectors_query, vectors_label, label_scores, labels_thresMax, labels_thresMin
	return acc, ranks_, y_raw_uniques, vectors_query, vectors_label


if __name__=="__main__":

	# ======================================================================
	# Make data folder and load data
	# ======================================================================

	DATASET_NAME = "$$$$_all_data"
	DATASET_TXT =  "all_training.txt"

	SRC_DATA_DIR = '../../data/'
	DST_DATA_DIR = './data/{}'.format(DATASET_NAME)

	'''
	split_data.seperate_seen_unseen_classes(
		inp_as_list = True,
		src_dir = os.path.join(SRC_DATA_DIR, '_$$$$_data'),
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
	'''

	data = load_data.Data(use3_directory=None, elmo2_directory=None, glove_directory=None)
	data(
		data_dir = DST_DATA_DIR, #os.path.join(SRC_DATA_DIR, DATASET_NAME),
		files = ["train_shuffle.txt","validation_shuffle.txt","test.txt"],
		codes= ["tr","va","te"]
		)

	# ======================================================================
	# BERT CLS embedding with Triplet Loss objective
	# ======================================================================
	# ======================================================================
	# Like with SWAG dataset modeling (in bERT paper), we concat a BATCH_SIZE 
	# of queries, a BATCH_SIZE of corresponding positive lables and a 
	# BATCH_SIZE of corresponding negative lables to BERT pre-trained architecture.
	# ======================================================================
	do_save_tokenized_sents = False
	do_training, do_validation = True, True
	do_evaluation = False # consists of methods more than just validation

	# ======================================================================
	# Training
	# ======================================================================
	if do_training:
		tf_config = tf.ConfigProto()
		tf_config.allow_soft_placement = True
		tf_config.log_device_placement = False
		tf_config.gpu_options.allow_growth = True
		tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
		tf.logging.set_verbosity(tf.logging.INFO)
		#
		# input batch size to bert will get tripled in this triplet loss code
		#
		tr_measures = {"TRAIN_BATCH_SIZE":32, "START_EPOCH":0, "N_EPOCHS":500, "SAVE_EVERY_Nth_EPOCH":50,
						"epoch_losses":[], "epoch_L2dists":[], "lowest_loss":None, "lowest_loss_epoch":None, 
						"epoch_acc":[], "highest_acc":None, "highest_acc_epoch":None}
		val_measures = {"VAL_BATCH_SIZE":128, "VAL_EVERY_Nth_EPOCH":5,
						"epoch_acc":[], "highest_acc":None, "highest_acc_epoch":None}
		if tr_measures["START_EPOCH"]!=0:
			restore_ckpt_dir = "./checkpoints/allData_with_hardNegatives_tripletLoss_01/uncased_L-12_H-768_A-12"
			restore_model_name = "wsm_data_TripletLoss_train_model_0.ckpt"
			#
			model = Model(gpu_devices=[2])
			model.restore_pretrained_bert_config(ckpt_dir=restore_ckpt_dir, max_seq_len=35)
			model.set_base_ops(is_training=True) # is_training will only impact dropout
			model.set_custom_ops_TripletLoss(is_training=True)
			sess = tf.Session(graph = model.tf_graph, config=tf_config)
			sess.__enter__()
			model.restore_weights(model_name=restore_model_name, ckpt_dir=restore_ckpt_dir, sess=sess)
		else:
			restore_ckpt_dir = None
			restore_model_name = None
			#
			model = Model(gpu_devices=[1])
			model.restore_pretrained_bert_config(ckpt_dir=restore_ckpt_dir, max_seq_len=35)
			model.set_base_ops(is_training=True)
			model.set_custom_ops_TripletLoss(is_training=True)
			model.restore_weights(model_name=restore_model_name, ckpt_dir=restore_ckpt_dir)
			sess = tf.Session(graph = model.tf_graph, config=tf_config)
			sess.__enter__()
			sess.run(model.init_op)
		save_ckpt_dir = "./checkpoints/allData_with_hardNegatives_tripletLoss_02"
		save_model_name = "bertModel.ckpt"
		# ======================================================================
		# Save and analyze the tokenized data
		# ======================================================================
		if do_save_tokenized_sents:
			print("Saving the tokenized data starts...")
			#
			inputFeatures_tr_query = model.get_InputFeatures(text_a_list=process_txt(data.x_tr_raw))
			inputFeatures_tr_query = np.asarray(inputFeatures_tr_query)
			inputTokens_tr_query = np.asarray([f.joined_tokens for f in inputFeatures_tr_query])
			inputFeatures_tr_label = model.get_InputFeatures(text_a_list=process_txt(data.y_tr_raw))
			inputFeatures_tr_label = np.asarray(inputFeatures_tr_label)
			inputTokens_tr_label = np.asarray([f.joined_tokens for f in inputFeatures_tr_label])
			model.save_tokenized_sents(
				data.x_tr_raw, inputTokens_tr_query, data.y_tr_raw, inputTokens_tr_label,
				column_names=["raw_query","tokenized_query","raw_label","tokenized_label"],
				file_title="tokenized_tr_uncased",
				ckpt_dir=save_ckpt_dir
				)
			#
			inputFeatures_va_query = model.get_InputFeatures(text_a_list=process_txt(data.x_va_raw))
			inputFeatures_va_query = np.asarray(inputFeatures_va_query)
			inputTokens_va_query = np.asarray([f.joined_tokens for f in inputFeatures_va_query])
			inputFeatures_va_label = model.get_InputFeatures(text_a_list=process_txt(data.y_va_raw))
			inputFeatures_va_label = np.asarray(inputFeatures_va_label)
			inputTokens_va_label = np.asarray([f.joined_tokens for f in inputFeatures_va_label])
			model.save_tokenized_sents(
				data.x_va_raw, inputTokens_va_query, data.y_va_raw, inputTokens_va_label,
				column_names=["raw_query","tokenized_query","raw_label","tokenized_label"],
				file_title="tokenized_va_uncased",
				ckpt_dir=save_ckpt_dir
				)
			#
			inputFeatures_te_query = model.get_InputFeatures(text_a_list=process_txt(data.x_te_raw))
			inputFeatures_te_query = np.asarray(inputFeatures_te_query)
			inputTokens_te_query = np.asarray([f.joined_tokens for f in inputFeatures_te_query])
			inputFeatures_te_label = model.get_InputFeatures(text_a_list=process_txt(data.y_te_raw))
			inputFeatures_te_label = np.asarray(inputFeatures_te_label)
			inputTokens_te_label = np.asarray([f.joined_tokens for f in inputFeatures_te_label])
			model.save_tokenized_sents(
				data.x_te_raw, inputTokens_te_query, data.y_te_raw, inputTokens_te_label,
				column_names=["raw_query","tokenized_query","raw_label","tokenized_label"],
				file_title="tokenized_te_uncased",
				ckpt_dir=save_ckpt_dir
				)
			print("Saving the tokenized data ends...")
		#
		for epoch in np.arange(tr_measures["START_EPOCH"], tr_measures["N_EPOCHS"]):
			print("=======================================================================")
			print("=======================================================================")
			tf.logging.info("Epoch: {}".format(epoch))
			#
			y_unique_labels = [*data.label2id_tr.keys()]
			if epoch<=8:
				y_tr_raw_negs = get_random_negatives(process_txt(data.x_tr_raw), process_txt(data.y_tr_raw), y_unique_labels=process_txt(y_unique_labels), list_size=1)
			else:
				prob = 0.5 if epoch<15 else 0.75
				if random.uniform(0, 1)>=prob:
					y_tr_raw_negs = get_random_negatives(process_txt(data.x_tr_raw), process_txt(data.y_tr_raw), y_unique_labels=process_txt(y_unique_labels), list_size=1)
				else:
					inputFeatures_tr_query = model.get_InputFeatures(text_a_list=process_txt(data.x_tr_raw))
					inputFeatures_tr_query = np.asarray(inputFeatures_tr_query)
					vectors_tr_query = get_representations_type1(model=model, sess=sess, find_query_vectors=True, inputFeatures=inputFeatures_tr_query, BATCH_SIZE=val_measures["VAL_BATCH_SIZE"])
					#
					inputFeatures_tr_label = model.get_InputFeatures(text_a_list=process_txt(y_unique_labels))
					inputFeatures_tr_label = np.asarray(inputFeatures_tr_label)
					vectors_tr_label = get_representations_type1(model=model, sess=sess, find_query_vectors=False, inputFeatures=inputFeatures_tr_label, BATCH_SIZE=val_measures["VAL_BATCH_SIZE"])
					#
					dist_tr = eu_dist(vectors_tr_query, vectors_tr_label)
					y_tr_raw_negs = get_hard_negatives(process_txt(data.x_tr_raw), process_txt(data.y_tr_raw), dist_tr, y_unique_labels=process_txt(y_unique_labels), list_size=1)
			#
			inputFeatures_tr_query = model.get_InputFeatures(text_a_list=process_txt(data.x_tr_raw))
			inputFeatures_tr_query = np.asarray(inputFeatures_tr_query)
			inputFeatures_tr_pos = model.get_InputFeatures(text_a_list=process_txt(data.y_tr_raw))
			inputFeatures_tr_pos = np.asarray(inputFeatures_tr_pos)
			inputFeatures_tr_neg = model.get_InputFeatures(text_a_list=process_txt(y_tr_raw_negs))
			inputFeatures_tr_neg = np.asarray(inputFeatures_tr_neg)
			#
			assert(len(inputFeatures_tr_query)==len(inputFeatures_tr_pos)==len(inputFeatures_tr_neg))
			TRAIN_NUM = len(inputFeatures_tr_query)
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
				#
				batch_inputFeatures_tr = np.hstack([inputFeatures_tr_query[batch_index],inputFeatures_tr_pos[batch_index],inputFeatures_tr_neg[batch_index]])
				batch_input_ids = np.asarray([f.input_ids for f in batch_inputFeatures_tr], dtype=np.int32)
				batch_input_mask = np.asarray([f.input_mask for f in batch_inputFeatures_tr], dtype=np.int32)
				batch_token_type_ids = np.asarray([f.segment_ids for f in batch_inputFeatures_tr], dtype=np.int32)
				#
				_, mean_loss, mean_pdist, mean_ndist = \
					sess.run([model.train_op, model.loss, model.pdist, model.ndist],
								feed_dict={
											model.batch_input_ids__tensor:batch_input_ids,
											model.batch_input_mask__tensor:batch_input_mask,
											model.batch_token_type_ids__tensor:batch_token_type_ids,
											model.part_size:len(batch_index),
											model.learning_rate:0.00002,
											model.margin_gamma:5
										  }
							)
				#
				epoch_loss+=mean_loss
				epoch_pdist+=mean_pdist
				epoch_ndist+=mean_ndist
			epoch_loss/=n_train_batches
			epoch_pdist/=n_train_batches
			epoch_ndist/=n_train_batches
			tf.logging.info("Lowest Epoch Loss until now: {} at epoch: {}".format(tr_measures["lowest_loss"],tr_measures["lowest_loss_epoch"]))
			tf.logging.info("Epoch Loss (avg. over all batches): {}".format(epoch_loss))
			tf.logging.info("Epoch pos distance (avg. over all batches): {}".format(epoch_pdist))
			tf.logging.info("Epoch neg distance (avg. over all batches): {}".format(epoch_ndist))
			tr_measures["epoch_losses"].append((epoch, epoch_loss))
			tr_measures["epoch_L2dists"].append((epoch, epoch_pdist, epoch_ndist))
			#
			save_name = "allData_with_hardNegatives_tripletLoss"
			if epoch%tr_measures["SAVE_EVERY_Nth_EPOCH"]==0:
				print("Saving weights at a pre-defined epoch intervals...")
				model_name = "".join(save_model_name.split(".")[:-1])+"_epoch{}Model.ckpt".format(epoch)
				model.save_weights(sess, model_name=model_name, ckpt_dir=save_ckpt_dir)
			if epoch==0 or epoch_loss<=tr_measures["lowest_loss"]:
				print("Saving weights at an epoch when loss is lowered...")
				tr_measures["lowest_loss"], tr_measures["lowest_loss_epoch"] = epoch_loss, epoch
				model_name = "".join(save_model_name.split(".")[:-1])+"_lowestTrainLossModel.ckpt"
				model.save_weights(sess, model_name=model_name, ckpt_dir=save_ckpt_dir)
			#
			if do_validation and epoch%val_measures["VAL_EVERY_Nth_EPOCH"]==0:
				print("Evaluating Accuracy and Mean Ranks...")
				acc1, _ = get_bert_prediction_scores_type1(data.x_tr_raw, data.y_tr_raw, model, sess, batch_size=100)
				tf.logging.info("Highest Train Accuracy until now: {} at epoch: {}".format(tr_measures["highest_acc"],tr_measures["highest_acc_epoch"]))
				tf.logging.info("Epoch Train Accuracy: {}".format(acc1))
				tr_measures["epoch_acc"].append((epoch, acc1))
				if epoch==0 or tr_measures["highest_acc"]<=acc1:
					tr_measures["highest_acc"], tr_measures["highest_acc_epoch"] = acc1, epoch
				acc2, _ = get_bert_prediction_scores_type1(data.x_va_raw, data.y_va_raw, model, sess, batch_size=val_measures["VAL_BATCH_SIZE"])
				tf.logging.info("Highest Validation Accuracy until now: {} at epoch: {}".format(val_measures["highest_acc"],val_measures["highest_acc_epoch"]))
				tf.logging.info("Epoch Validation Accuracy: {}".format(acc2))
				val_measures["epoch_acc"].append((epoch, acc2))
				if epoch==0 or val_measures["highest_acc"]<=acc2:
					val_measures["highest_acc"], val_measures["highest_acc_epoch"] = acc2, epoch
		#
		tf.logging.info("Training Complete...")
		model.dump_json(tr_measures, "tr_measures", ckpt_dir=save_ckpt_dir)
		model.dump_json(val_measures, "val_measures", ckpt_dir=save_ckpt_dir)
		sess.close()

	# ======================================================================
	# Inference
	# ======================================================================
	if do_evaluation:
		tf_config = tf.ConfigProto()
		tf_config.allow_soft_placement = True
		tf_config.log_device_placement = False
		tf_config.gpu_options.allow_growth = True
		tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
		#
		restore_ckpt_dir = "./checkpoints/allData_with_hardNegatives_tripletLoss_01/uncased_L-12_H-768_A-12"
		restore_model_name = "wsm_data_with_hard_negs_TripletLoss_lowest_loss_model_0.ckpt"
		model = Model(gpu_devices=[2])
		model.restore_pretrained_bert_config(ckpt_dir=restore_ckpt_dir, max_seq_len=35)
		model.set_base_ops(is_training=False)
		model.set_custom_ops_TripletLoss(is_training=False)
		sess = tf.Session(graph = model.tf_graph, config=tf_config)
		sess.__enter__()
		model.restore_weights(model_name="wsm_data_with_hard_negs_TripletLoss_lowest_loss_model_0.ckpt", ckpt_dir=restore_ckpt_dir, sess=sess)
		#
		# Test Labels and Train Labels aren't combined for predictions (a less practical scenario)
		tr_acc, ranks_tr, tr_labels, _, vectors_tr_label, tr_labels_thresList, tr_labels_thresMax, tr_labels_thresMin = \
			get_bert_prediction_scores_type1(data.x_tr_raw, data.y_tr_raw, model, sess, batch_size=100, excel_title="train_set_uncombinedLabels", ckpt_dir=restore_ckpt_dir, return_thresScores=True)
		va_acc, ranks_va, va_labels, _, _, va_labels_thresList, va_labels_thresMax, va_labels_thresMin= \
			get_bert_prediction_scores_type1(data.x_va_raw, data.y_va_raw, model, sess, batch_size=100, excel_title="validation_set_uncombinedLabels", ckpt_dir=restore_ckpt_dir, return_thresScores=True)
		te_acc, ranks_te, _, vectors_te_query, _ = \
			get_bert_prediction_scores_type1(data.x_te_raw, data.y_te_raw, model, sess, batch_size=100, excel_title="test_set_uncombinedLabels", ckpt_dir=restore_ckpt_dir)
		#
		# Test Labels and Train Labels are combined for predictions (a more practical scenario)
		all_labels = np.hstack([data.y_tr_raw,data.y_va_raw,data.y_te_raw])
		tr_acc, ranks_tr, _, _, _ = \
			get_bert_prediction_scores_type1(data.x_tr_raw, data.y_tr_raw, model, sess, batch_size=100, y_raw_merged=all_labels, excel_title="train_set_combinedLabels", ckpt_dir=restore_ckpt_dir)
		va_acc, ranks_va, _, _, _ = \
			get_bert_prediction_scores_type1(data.x_va_raw, data.y_va_raw, model, sess, batch_size=100, y_raw_merged=all_labels, excel_title="validation_set_combinedLabels", ckpt_dir=restore_ckpt_dir)
		te_acc, ranks_te, _, _, _ = \
			get_bert_prediction_scores_type1(data.x_te_raw, data.y_te_raw, model, sess, batch_size=100, y_raw_merged=all_labels, excel_title="test_set_combinedLabels", ckpt_dir=restore_ckpt_dir)
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
	




'''
	# ======================================================================
	# Approach abandoned because of inference time complexity
	# ======================================================================
	# ======================================================================
	# BERT CLS embedding to 0-1 classification with BCE
	# ======================================================================
	do_training = True
	
	# ======================================================================
	# Training
	# ======================================================================
	if do_training:
		BATCH_SIZE = 50
		N_EPOCHS = 100
		tf_config = tf.ConfigProto()
		tf_config.allow_soft_placement = True
		tf_config.log_device_placement = False
		tf_config.gpu_options.allow_growth = True
		tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9

		model = Model(gpu_devices=[0])
		model.restore_pretrained_bert_config(max_seq_len=64)
		model.set_base_ops(is_training=True)
		model.set_custom_ops_BCELoss(is_training=True)
		model.restore_weights()

		epoch_losses = []
		lowest_loss = -1
		lowest_loss_epoch = -1
		with tf.Session(graph = model.tf_graph, config=tf_config) as sess:
			sess.run(model.init_op)
			#
			for epoch in np.arange(N_EPOCHS):
				#
				print("=======================================================================")
				print("=======================================================================")
				print("Epoch: {}".format(epoch))
				#
				y_unique_labels = [*data.label2id_tr.keys()]
				y_tr_raw_negs = get_random_negatives(data.x_tr_raw, data.y_tr_raw, y_unique_labels=y_unique_labels, list_size=1)
				"""
				for i, triplet in enumerate(zip(data.x_tr_raw, data.y_tr_raw, y_tr_raw_negatives)):
					print(triplet)
					if i!=0 and i%20==0:
						break;
				"""
				all_unique_labels = ["neg", "pos"] # neg-0, pos-1
				label_list = ["pos"]*len(data.x_tr_raw)
				inputFeatures_tr_pos = model.get_InputFeatures(text_a_list=data.x_tr_raw, text_b_list=data.y_tr_raw, label_list=label_list, all_unique_labels=all_unique_labels)
				inputFeatures_tr_pos = np.asarray(inputFeatures_tr_pos)

				all_unique_labels = ["neg", "pos"] # neg-0, pos-1
				label_list = ["neg"]*len(data.x_tr_raw)
				inputFeatures_tr_neg = model.get_InputFeatures(text_a_list=data.x_tr_raw, text_b_list=y_tr_raw_negs, label_list=label_list, all_unique_labels=all_unique_labels)	
				inputFeatures_tr_neg = np.asarray(inputFeatures_tr_neg)

				inputFeatures_tr = np.hstack([inputFeatures_tr_pos,inputFeatures_tr_neg]) # each feature object has input_ids, input_mask, segment_ids, label_id, is_real_example attributes
				#
				TRAIN_NUM = len(inputFeatures_tr)
				n_train_batches = int(math.ceil(TRAIN_NUM/BATCH_SIZE))
				all_indices = np.arange(TRAIN_NUM)
				np.random.shuffle(all_indices)
				epoch_loss = 0
				for i in tqdm(range(n_train_batches)):
					begin_index = i * BATCH_SIZE
					end_index = min((i + 1) * BATCH_SIZE, TRAIN_NUM)
					batch_index = all_indices[begin_index : end_index]
					#
					batch_input_ids = np.asarray([f.input_ids for f in inputFeatures_tr[batch_index]], dtype=np.int32)
					batch_input_mask = np.asarray([f.input_mask for f in inputFeatures_tr[batch_index]], dtype=np.int32)
					batch_token_type_ids = np.asarray([f.segment_ids for f in inputFeatures_tr[batch_index]], dtype=np.int32)
					batch_label_ids = np.asarray([f.label_id for f in inputFeatures_tr[batch_index]], dtype=np.int32).reshape(-1,1)
					#
					_, mean_loss = \
						sess.run([model.train_op, model.loss],
									feed_dict={
												model.batch_input_ids__tensor:batch_input_ids,
												model.batch_input_mask__tensor:batch_input_mask,
												model.batch_token_type_ids__tensor:batch_token_type_ids,
												model.true_labels:batch_label_ids,
												model.learning_rate:0.001
											  }
								)
					#
					epoch_loss+=mean_loss
				epoch_loss/=n_train_batches
				print("Epoch Loss (avg. over all batches): {}".format(epoch_loss))
				epoch_losses.append(epoch_loss)
				if epoch==0 or epoch_loss<=lowest_loss:
					lowest_loss = epoch_loss
					lowest_loss_epoch = epoch
					model.save_weights(sess, model_name="wsm_train_model_0.ckpt", ckpt_dir='./checkpoints')
'''





'''
		inputFeatures_tr_query = model.get_InputFeatures(text_a_list=process_txt(data.x_tr_raw))
		inputFeatures_tr_query = np.asarray(inputFeatures_tr_query)
		vectors_tr_query = get_representations_type1(model=model, sess=sess, find_query_vectors=True, inputFeatures=inputFeatures_tr_query, BATCH_SIZE=BATCH_SIZE)
		tr_labels = [*data.label2id_tr.keys()]
		inputFeatures_tr_label = model.get_InputFeatures(text_a_list=process_txt(tr_labels))
		inputFeatures_tr_label = np.asarray(inputFeatures_tr_label)
		vectors_tr_label = get_representations_type1(model=model, sess=sess, find_query_vectors=False, inputFeatures=inputFeatures_tr_label, BATCH_SIZE=BATCH_SIZE)
		#
		inputFeatures_va_query = model.get_InputFeatures(text_a_list=process_txt(data.x_va_raw))
		inputFeatures_va_query = np.asarray(inputFeatures_va_query)
		vectors_va_query = get_representations_type1(model=model, sess=sess, find_query_vectors=True, inputFeatures=inputFeatures_va_query, BATCH_SIZE=BATCH_SIZE)
		va_labels = [*data.label2id_va.keys()]
		inputFeatures_va_label = model.get_InputFeatures(text_a_list=process_txt(va_labels))
		inputFeatures_va_label = np.asarray(inputFeatures_va_label)
		vectors_va_label = get_representations_type1(model=model, sess=sess, find_query_vectors=False, inputFeatures=inputFeatures_va_label, BATCH_SIZE=BATCH_SIZE)
		#
		inputFeatures_te_query = model.get_InputFeatures(text_a_list=process_txt(data.x_te_raw))
		inputFeatures_te_query = np.asarray(inputFeatures_te_query)
		vectors_te_query = get_representations_type1(model=model, sess=sess, find_query_vectors=True, inputFeatures=inputFeatures_te_query, BATCH_SIZE=BATCH_SIZE)
		te_labels = [*data.label2id_te.keys()]
		inputFeatures_te_label = model.get_InputFeatures(text_a_list=process_txt(te_labels))
		inputFeatures_te_label = np.asarray(inputFeatures_te_label)
		vectors_te_label = get_representations_type1(model=model, sess=sess, find_query_vectors=False, inputFeatures=inputFeatures_te_label, BATCH_SIZE=BATCH_SIZE)
		#
		dist_tr = eu_dist(vectors_tr_query, vectors_tr_label)
		dist_va = eu_dist(vectors_va_query, vectors_va_label)
		dist_te = eu_dist(vectors_te_query, vectors_te_label)
		#
		predicted_labels_tr = np.argmin(dist_tr, axis=-1)
		predicted_labels_va = np.argmin(dist_va, axis=-1)
		predicted_labels_te = np.argmin(dist_te, axis=-1)
		#
		# Print Accuracy
		tr_acc = accuracy_score(data.y_tr, predicted_labels_tr)
		print("Training Data Accuarcy", tr_acc)
		va_acc = accuracy_score(data.y_va, predicted_labels_va)
		print("Validation Data Accuarcy", va_acc)
		te_acc = accuracy_score(data.y_te, predicted_labels_te)
		print("Testing Data Accuarcy", te_acc)
		#
		# Print Mean Rank
		idx = np.argsort(dist_tr)
		ranks_tr = []
		for i, _ in enumerate(idx):
			ranks_tr.append(np.where(idx[i,:]==data.y_tr[i])[0][0])
		ranks_tr = np.asarray(ranks_tr)+1 # converting to 1 base
		print("Training Mean Rank", np.mean(ranks_tr))
		idx = np.argsort(dist_va)
		ranks_va = []
		for i, _ in enumerate(idx):
			ranks_va.append(np.where(idx[i,:]==data.y_va[i])[0][0])
		ranks_va = np.asarray(ranks_va)+1 # converting to 1 base
		print("Validation Mean Rank", np.mean(ranks_va))
		idx = np.argsort(dist_te)
		ranks_te = []
		for i, _ in enumerate(idx):
			ranks_te.append(np.where(idx[i,:]==data.y_te[i])[0][0])
		ranks_te = np.asarray(ranks_te)+1 # converting to 1 base
		print("Testing Mean Rank", np.mean(ranks_te))	
		#

'''
