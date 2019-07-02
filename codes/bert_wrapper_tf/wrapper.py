# ======================================================================
# coding: utf-8
# Author: Sai Muralidhar 
# ======================================================================
# ======================================================================
# :notes:
# Changes made wrt original code of BERT at https://github.com/google-research/bert
# 1) In modeling.py 
#    	In BertModel() class, made the following addition:
#			def get_cls_vector(self):
#				return self.first_token_tensor
# 2) In run_classifier.py()
#    	In convert_single_example() method, made the following replacement
#			As-is:
#				label_id = label_map[example.label]
#			To-be:
#				try:
#					label_id = label_map[example.label]
#				except:
#					label_id = None
# 				# And then comment the print lines below it
# 3) In run_classifier.py()
#		In the imports line, made the following replacement
#			As-is:
#				import modeling, optimization, tokenization
#			To-be:
#				from . import modeling, optimization, tokenization
# 4) In run_classifier.py()
#		In InputFeatures() class, made the following addition
#			self.token_len = token_len
#		In convert_single_example() method, 
#			In isinstance(example, PaddingInputExample) check,
#				In the return object, made the following addition
#					token_len = max_seq_length
#			In feature = InputFeatures(...) object instantiation, made the following addition
#				token_len = len(tokens)
# ======================================================================


print("/".join(__file__.split("/")[:-1]))

import os, sys, math, numpy as np
from tqdm import tqdm # to maintain progress bar
import shutil # to copy files
import tensorflow as tf

# ======================================================================
# Download & Load BERT codes and make changes to files as required in :notes:
# ======================================================================

# 1. Download codes
# !git clone https://murali1996:<password>@github.com/murali1996/nlp.git
# !git clone https://github.com/google-research/bert

# 2. Download pretrained weights
# BERT_PRETRAINED_MODELS_DIR = 'BERT_PRETRAINED_MODELS'
# BERT_MODEL_NAME = 'cased_L-12_H-768_A-12'
# BERT_PRETRAINED_DIR = os.path.join(BERT_PRETRAINED_MODELS_DIR,BERT_MODEL_NAME)
# if not os.path.exists(BERT_PRETRAINED_MODELS_DIR):
# 	os.mkdir(BERT_PRETRAINED_MODELS_DIR)
# os.system('wget https://storage.googleapis.com/bert_models/2018_10_18/{}.zip -O {}/{}.zip'.format(BERT_MODEL_NAME,BERT_PRETRAINED_MODELS_DIR,BERT_MODEL_NAME))
# os.system('unzip {}/{}.zip -d {}/'.format(BERT_PRETRAINED_MODELS_DIR,BERT_MODEL_NAME,BERT_PRETRAINED_MODELS_DIR))


# ======================================================================
# Bert-Wrapper Model Class
# ======================================================================

from .bert_master import run_classifier, optimization, tokenization, modeling

class Model(object):
	"""
	- Custom Model is the one built on top of BERT and include both pretrained weights of BERT as well as some new vars used for ops on top of BERT
	- Using this class, you can:
		- Load and Use results from a pretrained BERT, trained as described in https://arxiv.org/pdf/1810.04805v2.pdf
		- :todo: Finetune BERT weights on you own data with the same training objective as in BERT
		- Train a custom model on top of BERT with your own objective; Please manually define your ops in set_custom_model_ops() method below
	- While saving details of custom model on top of pretrained BERT, necessary flies like vocab, etc. must also be save in that destination directory
		- This is to be in compatible with relaoding bert weights upon saving custom model
	- On reloading the custom model, those new path locations are to be specified for the set_pretrained_bert_detail() method
	"""
	def __init__(self, gpu_devices):
		#
		self.gpu_devices = gpu_devices #[0,1] #[0] #[]
		if len(self.gpu_devices)>1:
			os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(device) for device in self.gpu_devices)
			for i, num in enumerate(self.gpu_devices):
				setattr(self, 'gpu_devices_n{}'.format(i), '/gpu:{}'.format(num))
		elif len(self.gpu_devices)==1:
			os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_devices[0])
			setattr(self, 'gpu_devices_n1', '/gpu:{}'.format(self.gpu_devices[0]))
		else:
			os.environ['CUDA_VISIBLE_DEVICES'] = ''
			setattr(self, 'gpu_devices_n1', '/cpu:0')
		#
		self.tf_graph = tf.Graph()
		self.tf_writer = None
		self.tf_saver = None
		self.MAX_SEQ_LENGTH = 128
		self.src_ckpt_dir = os.path.join("/".join(__file__.split("/")[:-1]), './checkpoints/BERT_PRETRAINED_MODELS/uncased_L-12_H-768_A-12') # donot put trailing / slash
		self.dest_ckpt_dir = os.path.join("/".join(__file__.split("/")[:-1]), './checkpoints/CUSTOM_MODELS/uncased_L-12_H-768_A-12_custom0')
		for key, value in self.__dict__.items():
			text = "INITIAL_SETTINGS:: {} = {}".format(key,value)
			print(text)
	#
	# DATA PROCESS for BERT
	def get_InputFeatures(self, text_a_list, text_b_list=[], guid_list=[], label_list:"can be strings too; should be present in all_unique_labels"=[], all_unique_labels:"can be label strings too"=[]):
		#
		if len(text_b_list)!=0:
			assert len(text_a_list)==len(text_b_list)
		elif len(text_b_list)!=0 and len(label_list)!=0:
			assert len(text_a_list)==len(text_b_list)==len(label_list)
		elif len(text_b_list)!=0 and len(label_list)!=0 and len(guid_list)!=0:
			assert len(text_a_list)==len(text_b_list)==len(label_list)==len(guid_list)
		#
		if not len(all_unique_labels)==0: # retaining the order
			all_unique_labels = np.asarray(all_unique_labels)
			_, idx = np.unique(all_unique_labels, return_index=True)
			all_unique_labels = all_unique_labels[np.sort(idx)]
			print("Retaining the order, unique items from all_unique_labels are now picked.")
		#
		num = len(text_a_list)
		if len(text_b_list)==0:
			text_b_list=[None]*num
		if len(label_list)==0:
			label_list=[None]*num
		if len(guid_list)==0:
			guid_list=[None]*num
		#
		inputExamples = []
		for i, _ in enumerate(text_a_list):
			inputExamples.append(run_classifier.InputExample(guid=guid_list[i], text_a=text_a_list[i], text_b=text_b_list[i], label=label_list[i]))
		inputFeatures = run_classifier.convert_examples_to_features(inputExamples, all_unique_labels, self.MAX_SEQ_LENGTH, self.tokenizer)
		# input_ids, input_mask, segment_ids, label_id, is_real_example are the attributes for a inputFeauture object
		return inputFeatures
		#
	#
	# SAVE vars
	def save_weights(self, sess, model_name='model.ckpt', ckpt_dir=None):
		if ckpt_dir==None:
			print('Using known destination ckpt dir')
			ckpt_dir = self.dest_ckpt_dir
		if self.get_cased_from_name==True:
			print('Since *cased* info was derived from a folder name, to retain this info...')
			print('Creating a folder named {} in destination ckpt dir to save the weights'.format(self.src_ckpt_dir.split("/")[-1]))
			ckpt_dir = os.path.join(ckpt_dir, self.src_ckpt_dir.split("/")[-1])
		if not os.path.exists(ckpt_dir):
			os.makedirs(ckpt_dir)			
		ckpt_path = os.path.join(ckpt_dir, model_name)
		self.tf_saver.save(sess, ckpt_path)
		self._save_pretrained_bert_config(ckpt_dir=ckpt_dir)
		#
		additional_dict = {"DO_LOWER_CASE":model.DO_LOWER_CASE, "MAX_SEQ_LENGTH":model.MAX_SEQ_LENGTH, "get_cased_from_name":self.get_cased_from_name}
		now_open = open(os.path.join(ckpt_dir, "model_dict.json"),"w")
		now_open.write(json.dumps(additional_dict))
		now_open.close()
		return
	def _save_pretrained_bert_config(self, ckpt_dir):
		print("The following files will be saved: vocab.txt, bert_config.json in the ckpt_dir {}".format(ckpt_dir))
		try:
			shutil.copyfile(self.VOCAB_FILE, os.path.join(ckpt_dir, self.VOCAB_FILE.split("/")[-1]))
		except Exception as e:
			print("Couldn't copy files because of exception: {}".format(e))
			pass
		try:
			shutil.copyfile(self.CONFIG_FILE, os.path.join(ckpt_dir, self.CONFIG_FILE.split("/")[-1]))
		except Exception as e:
			print("Couldn't copy files because of exception: {}".format(e))
			pass
		return
	#
	# RESTORE vars
	def restore_weights(self, model_name='bert_model.ckpt', ckpt_dir=None, print_tvars=False, sess=None):
		if ckpt_dir==None:
			print('Using known source ckpt dir')
			ckpt_dir = self.src_ckpt_dir
			ckpt_path = os.path.join(ckpt_dir, model_name)	
			with self.tf_graph.as_default():
				tvars = tf.trainable_variables()
				(assignment_map,initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, ckpt_path)
				tf.train.init_from_checkpoint(ckpt_path, assignment_map) 
				# Values are not loaded immediately, but when the initializer is run (typically by running a tf.global_variables_initializer op)
				if print_tvars:
					tf.logging.info("**** Trainable Variables ****")
					for var in tvars:
						init_string = ""
						if var.name in initialized_variable_names:
							init_string = ", *INIT_FROM_CKPT*"
						tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)
		else:
			ckpt_path = os.path.join(ckpt_dir, model_name)
			self.tf_saver.restore(sess, ckpt_path)
		return
	def restore_pretrained_bert_config(self, ckpt_dir=None, cased: "True/ False/ None; If none, detects automatically"=None, max_seq_len=None):
		if ckpt_dir==None:
			print('Using known source ckpt dir')
			ckpt_dir = self.src_ckpt_dir
		if cased==None:
			if (ckpt_dir.split("/")[-1].startswith('uncased') or ckpt_dir.split("/")[-1].startswith('cased')):
				self.DO_LOWER_CASE = ckpt_dir.split("/")[-1].startswith('uncased')
				self.get_cased_from_name = True
			elif (ckpt_dir.split("/")[-2].startswith('uncased') or ckpt_dir.split("/")[-2].startswith('cased')):
				self.DO_LOWER_CASE = ckpt_dir.split("/")[-2].startswith('uncased')
				self.get_cased_from_name = True
			else:
				raise Exception('The model\'s folder must start with either *cased* or *uncased* string since this arg is not set in func call')
		else:
			self.DO_LOWER_CASE = not cased
			self.get_cased_from_name = False
		if not max_seq_len==None:
			self.MAX_SEQ_LENGTH = max_seq_len 
		#
		print("The following files will be loaded: vocab.txt, bert_config.json")
		self.VOCAB_FILE = os.path.join(ckpt_dir, 'vocab.txt')
		self.CONFIG_FILE = os.path.join(ckpt_dir, 'bert_config.json')	
		self.bert_config = modeling.BertConfig.from_json_file(self.CONFIG_FILE)
		self.tokenizer = tokenization.FullTokenizer(vocab_file=self.VOCAB_FILE, do_lower_case=self.DO_LOWER_CASE) 
		print(self.tokenizer.tokenize("This is a simple example of how the BERT tokenizer tokenizes text. "))
		return
	#
	# TF GRAPH OPS
	def set_base_ops(self):
		with self.tf_graph.as_default():
			with tf.device(self.gpu_devices_n1):
				self.batch_input_ids__tensor = tf.placeholder(dtype=tf.int32, shape=(None,self.MAX_SEQ_LENGTH), name="batch_input_ids__tensor")
				self.batch_input_mask__tensor = tf.placeholder(dtype=tf.int32, shape=(None,self.MAX_SEQ_LENGTH), name="batch_input_mask__tensor")
				self.batch_token_type_ids__tensor = tf.placeholder(dtype=tf.int32, shape=(None,self.MAX_SEQ_LENGTH), name="batch_itoken_type_ids__tensor")
				self.bertModel = modeling.BertModel(self.bert_config,
													is_training=True,
													input_ids=self.batch_input_ids__tensor,
													input_mask=self.batch_input_mask__tensor,
													token_type_ids=self.batch_token_type_ids__tensor,
													scope="bert" # any different name and you have issues loading data from Checkpoint
											   )
				self.cls_output = self.bertModel.get_cls_vector()
				self.full_output = self.bertModel.get_sequence_output()
		return
	def set_custom_ops_BCELoss(self):
		with self.tf_graph.as_default():
			with tf.device(self.gpu_devices_n1):
				with tf.variable_scope("custom_ops"):
					self.true_labels = tf.placeholder(tf.float32, shape=(None, 1), name="true_labels")
					self.learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")
					#
					self.cls_dense1 = tf.layers.dense(self.cls_output,512,activation=tf.tanh,kernel_initializer=modeling.create_initializer(self.bert_config.initializer_range))
					self.cls_dense2 = tf.layers.dense(self.cls_dense1,128,activation=tf.tanh,kernel_initializer=modeling.create_initializer(self.bert_config.initializer_range))
					self.predicted_logits = tf.layers.dense(self.cls_dense2,1,activation=tf.nn.sigmoid,kernel_initializer=modeling.create_initializer(self.bert_config.initializer_range))
					#
					self.batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predicted_logits, labels=self.true_labels)
					self.loss = tf.reduce_mean(self.batch_loss)
					self.train_op = self.get_train_ops(self.loss, self.learning_rate)
					self.init_op = self.get_init_ops()
					self.set_saver_ops()
		return
	def set_custom_ops_TripletLoss(self):
		with self.tf_graph.as_default():
			with tf.device(self.gpu_devices_n1):
				with tf.variable_scope("custom_ops"):
					self.part_size = tf.placeholder(tf.int32, shape=(), name="part_size")
					self.learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")
					self.margin_gamma = tf.placeholder(tf.float32, shape=(), name="margin_gamma")
					#
					self.query_weights = tf.get_variable("query_weights", shape=(self.cls_output.get_shape().as_list()[-1], 256), initializer=modeling.create_initializer(self.bert_config.initializer_range))
					self.label_weights = tf.get_variable("label_weights", shape=(self.cls_output.get_shape().as_list()[-1], 256), initializer=modeling.create_initializer(self.bert_config.initializer_range))
					#
					self.qvec = self.cls_output[:self.part_size,:]
					self.pvec = self.cls_output[self.part_size:2*self.part_size,:]
					self.nvec = self.cls_output[2*self.part_size:3*self.part_size,:]
					self.qvec_prime = tf.matmul(self.qvec, self.query_weights)
					self.pvec_prime = tf.matmul(self.pvec, self.label_weights)
					self.nvec_prime = tf.matmul(self.nvec, self.label_weights)
					#
					# only for inference
					self.qvec_prime_inference = tf.matmul(self.cls_output, self.query_weights)
					self.pvec_prime_inference = tf.matmul(self.cls_output, self.label_weights)
					#
					self.batch_pdist = tf.norm(self.qvec_prime-self.pvec_prime, ord="euclidean", axis=-1)
					self.batch_ndist = tf.norm(self.qvec_prime-self.nvec_prime, ord="euclidean", axis=-1)
					self.pdist = tf.reduce_mean(self.batch_pdist)
					self.ndist = tf.reduce_mean(self.batch_ndist)
					#
					self.batch_loss = tf.nn.relu(self.margin_gamma+self.batch_pdist-self.batch_ndist)
					self.loss = tf.reduce_mean(self.batch_loss)
					self.train_op = self.get_train_ops(self.loss, self.learning_rate)
					self.init_op = self.get_init_ops()
					self.set_saver_ops()
		return
	def get_init_ops(self):
		init_op = tf.group([tf.global_variables_initializer(),tf.tables_initializer()])
		return init_op
	def get_train_ops(self, loss, learning_rate: "can be a tensor value too"):
		# train_op = tf.train.AdamOptimizer(2e-5).minimize(loss) # values obtained from paper reading 
		train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
		return train_op
	def set_saver_ops(self):
		with self.tf_graph.as_default():
			self.tf_saver = tf.train.Saver()
			print("tf_saver initialized; any new vars defined after this point will not be saved by attribute tf_saver")
		return
	def get_ops_devices(self):
		device_info = [n.device for n in self.tf_graph.as_graph_def().node]
		return device_info
	#
	# FINETUNE your data with BERT Objective
	def finetune_bert_weights(self, data=None):
		print("Function unavailable as of now...please run with your custom objective with pretrained bert-weights")
		return


########################################

"""
test_sentences = []
file = open('sentences.txt')
for row in file:
    test_sentences.append(row.split('\t')[0].strip())
file.close()

model = Model(gpu_devices=[0])
model.restore_pretrained_bert_config(max_seq_len=64)
inputFeatures = model.get_InputFeatures(text_a_list=test_sentences)
inputFeatures= np.asarray(inputFeatures)
# each feature object has input_ids, input_mask, segment_ids, label_id, token_len, is_real_example attributes

model.set_base_ops()
model.restore_weights()

INFER_NUM = len(test_sentences)
BATCH_SIZE = 50
cls_outputs = []
full_outputs = []
tf_config = tf.ConfigProto()
tf_config.allow_soft_placement = True
tf_config.log_device_placement = False
tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
with tf.Session(graph = model.tf_graph, config=tf_config) as sess:
	#
	sess.run(model.get_init_ops())
	#
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
		batch_cls_outputs, batch_full_outputs = \
			sess.run([model.cls_output, model.full_output],
						feed_dict={
									model.batch_input_ids__tensor:batch_input_ids,
									model.batch_input_mask__tensor:batch_input_mask,
									model.batch_token_type_ids__tensor:batch_token_type_ids
								  }
					)
		#
		if i==0:
			cls_outputs = batch_cls_outputs
			full_outputs = batch_full_outputs
		else:
			cls_outputs = np.concatenate((cls_outputs, batch_cls_outputs), axis=0)
			full_outputs = np.concatenate((full_outputs, batch_full_outputs), axis=0)
"""




# ======================================================================
# UNDER CONSTRUCTION!!!
# ======================================================================

# y_tr_raw_negatives = generate_random_negatives(data.x_tr_raw, data.y_tr_raw, train_plus_test_labels, list_size=1)
# y_va_raw_negatives = generate_random_negatives(data.x_va_raw, data.y_va_raw, train_plus_test_labels, list_size=1)

# In[ ]:
"""
weights_initializer = modeling.create_initializer(bert_config.initializer_range)
w1 = tf.get_variable(name='my_w1', shape=(batch_sent_embs__tensor.shape[1].value, 256), initializer=weights_initializer)
b1 = tf.get_variable(name='my_b1', shape=(256), initializer=tf.zeros_initializer())
w2 = tf.get_variable(name='my_w2', shape=(256*4, 64), initializer=weights_initializer)
b2 = tf.get_variable(name='my_b2', shape=(64), initializer=tf.zeros_initializer())
w3 = tf.get_variable(name='my_w3', shape=(64, 1), initializer=weights_initializer)
b3 = tf.get_variable(name='my_b3', shape=(1), initializer=tf.zeros_initializer())

def myFunc1(sent_embs):
	with tf.variable_scope("myFunc1", reuse=tf.AUTO_REUSE):
		hidden = tf.nn.tanh(tf.add(tf.matmul(sent_embs,w1),b1))
	return hidden

# as in Conneau et al. 2017 https://arxiv.org/abs/1705.02364
def myFunc2(intent_class_embs, query_embs):
	with tf.variable_scope("myFunc2", reuse=tf.AUTO_REUSE):
		first = intent_class_embs
		second = query_embs
		third = tf.abs(intent_class_embs-query_embs)
		fourth = tf.multiply(intent_class_embs,query_embs)
		vector = tf.concat([first, second, third, fourth], axis=-1)
		hidden = tf.nn.tanh(tf.add(tf.matmul(vector,w2),b2))
		logit = tf.nn.sigmoid(tf.add(tf.matmul(hidden,w3),b3))
	return logit

#  https://github.com/google-research/bert/issues/38
batch_intent_embs_, batch_pos_query_embs_, batch_neg_query_embs_ = \
				tf.split(batch_sent_embs__tensor, num_or_size_splits=3, axis=0)
assert batch_intent_embs_.shape[1].value==batch_pos_query_embs_.shape[1].value==batch_neg_query_embs_.shape[1].value

#projection scores
batch_intent_embs = myFunc1(batch_intent_embs_)
batch_pos_query_embs = myFunc1(batch_pos_query_embs_)
batch_neg_query_embs = myFunc1(batch_neg_query_embs_)
batch_pos_scores = myFunc2(batch_intent_embs, batch_pos_query_embs)
batch_neg_scores = myFunc2(batch_intent_embs, batch_neg_query_embs)

# cosine scores
#batch_intent_normed = tf.nn.l2_normalize(batch_intent_embs, axis=1)
#batch_pos_query_normed = tf.nn.l2_normalize(batch_pos_query_embs, axis=1)
#batch_neg_query_normed = tf.nn.l2_normalize(batch_neg_query_embs, axis=1)
#batch_pos_scores = tf.reduce_sum(tf.multiply(batch_intent_normed, batch_pos_query_normed), axis=1)
#batch_neg_scores = tf.reduce_sum(tf.multiply(batch_intent_normed, batch_neg_query_normed), axis=1)

batch_loss = tf.nn.relu( MAX_MARGIN_GAMMA + batch_pos_scores - batch_neg_scores )
loss = tf.reduce_mean(batch_loss)
train_op = tf.train.AdamOptimizer(2e-5).minimize(loss)


BATCH_SIZE = 20
N_EPOCHS = 100

NEW_CKPT_DIR_CUSTOM_MODEL = '../checkpoints'
if not os.path.exists(NEW_CKPT_DIR_CUSTOM_MODEL):
	os.mkdir(NEW_CKPT_DIR_CUSTOM_MODEL)

sess = tf.Session(config=tf_config)

# initializations
model_saver = tf.train.Saver() # all variables considered
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

# training
epoch_losses = []
for epoch in range(N_EPOCHS):
	print("\n")
	print("Epoch: {}".format(epoch))
	this_epoch_loss = 0.0
	#
	print("Processing data as required by BERT code...")
	y_tr_raw_negatives = generate_random_negatives(data.x_tr_raw, data.y_tr_raw, train_plus_test_labels, list_size=1)
	inputExamples = []
	for i, _ in enumerate(y_tr_raw_negatives):
		inputExamples.append(run_classifier.InputExample(guid=None,text_a=data.y_tr_raw[i],
																   text_b=data.x_tr_raw[i],
																   label=1))
		inputExamples.append(run_classifier.InputExample(guid=None,text_a=y_tr_raw_negatives[i],
																   text_b=data.x_tr_raw[i],
																   label=0))
		inputExamples.append(run_classifier.InputExample(guid=None,text_a=data.x_tr_raw[i],
																   text_b=data.y_tr_raw[i],
																   label=1))
		inputExamples.append(run_classifier.InputExample(guid=None,text_a=data.x_tr_raw[i],
																   text_b=y_tr_raw_negatives[i],
																   label=0))
	inputFeatures = run_classifier.convert_examples_to_features(inputExamples, [0,1], MAX_SEQ_LENGTH, tokenizer)
	inputFeatures = np.asarray(inputFeatures)
	#
	TRAIN_NUM = len(inputFeatures)
	n_train_batches = int(math.ceil(TRAIN_NUM/BATCH_SIZE))
	all_indices = np.arange(TRAIN_NUM)
	np.random.shuffle(all_indices) #in-place ops
	for i in range(n_train_batches):
		begin_index = i * BATCH_SIZE
		end_index = min((i + 1) * BATCH_SIZE, TRAIN_NUM)
		batch_index = all_indices[begin_index : end_index]
		#
		batch_input_ids = np.asarray([f.input_ids for f in inputFeatures[batch_index]], dtype=np.int32)
		batch_input_mask = np.asarray([f.input_mask for f in inputFeatures[batch_index]], dtype=np.int32)
		batch_token_type_ids = np.asarray([f.segment_ids for f in inputFeatures[batch_index]], dtype=np.int32)
		batch_labels = np.asarray([f.label_id for f in inputFeatures[batch_index]], dtype=np.float32).reshape(-1,1)
		#
		_, this_batch_loss, this_batch_per_sample_loss =             sess.run([train_op, batch_loss, loss], feed_dict={batch_input_ids__tensor:batch_input_ids,
															  batch_input_mask__tensor:batch_input_mask,
															  batch_token_type_ids__tensor:batch_token_type_ids,
															  batch_sent_embs__labels:batch_labels})
		this_epoch_loss+=this_batch_loss
		progressBar(i, n_train_batches,
					['this_batch_loss', 'this_batch_per_sample_loss'],
					[this_batch_loss, this_batch_per_sample_loss])
	epoch_losses.append( this_epoch_loss/len(all_indices) )
	#
	path = os.path.join(NEW_CKPT_DIR_CUSTOM_MODEL, BERT_MODEL_NAME+'_Updated'+'_epoch_{}'.format(epoch))
	model_saver.save(sess, path)
"""

"""
BATCH_SIZE = 20
N_EPOCHS = 100

NEW_CKPT_DIR_CUSTOM_MODEL = '../checkpoints'
if not os.path.exists(NEW_CKPT_DIR_CUSTOM_MODEL):
	os.mkdir(NEW_CKPT_DIR_CUSTOM_MODEL)
	
with tf.Session(config=tf_config) as sess:
	#
	model_saver = tf.train.Saver() # all variables considered
	#
	sess.run(tf.global_variables_initializer())
	sess.run(tf.tables_initializer())
	#
	for epoch in range(N_EPOCHS):
		epoch_loss = 0.0
		#
		y_tr_raw_negatives = generate_random_negatives(data.x_tr_raw, data.y_tr_raw)
		inputExamples = []
		for i, _ in enumerate(y_tr_raw_negatives):
			inputExamples.append(run_classifier.InputExample(guid=None,text_a=data.y_tr_raw[i],
																	   text_b=data.x_tr_raw[i],
																	   label=1))
			inputExamples.append(run_classifier.InputExample(guid=None,text_a=y_tr_raw_negatives[i],
																	   text_b=data.x_tr_raw[i],
																	   label=0))
			inputExamples.append(run_classifier.InputExample(guid=None,text_a=data.x_tr_raw[i],
																	   text_b=data.y_tr_raw[i],
																	   label=1))
			inputExamples.append(run_classifier.InputExample(guid=None,text_a=data.x_tr_raw[i],
																	   text_b=y_tr_raw_negatives[i],
																	   label=0))
		inputExamples = np.asarray(inputExamples)
		inputFeatures = run_classifier.convert_examples_to_features(inputExamples, [0,1], MAX_SEQ_LENGTH, tokenizer)
		inputFeatures = np.asarray(inputFeatures)
		#
		n_train_batches = int(math.ceil(TRAIN_NUM/BATCH_SIZE))
		all_indices = np.arange(TRAIN_NUM)
		np.random.shuffle(all_indices) #in-place ops
		for i in tqdm(range(n_train_batches)):
			begin_index = i * BATCH_SIZE
			end_index = min((i + 1) * BATCH_SIZE, TRAIN_NUM)
			batch_index = all_indices[begin_index : end_index]
			#
			batch_input_ids = np.asarray([f.input_ids for f in inputFeatures[batch_index]], dtype=np.int32)
			batch_input_mask = np.asarray([f.input_mask for f in inputFeatures[batch_index]], dtype=np.int32)
			batch_token_type_ids = np.asarray([f.segment_ids for f in inputFeatures[batch_index]], dtype=np.int32)
			batch_labels = np.asarray([i.label for i in inputExamples[batch_index]], dtype=np.float32)
			#
			_, epoch_batch_loss, epoch_per_sample_loss = \
				sess.run([train_op, batch_loss, loss], feed_dict={batch_input_ids__tensor:batch_input_ids,
																  batch_input_mask__tensor:batch_input_mask,
																  batch_token_type_ids__tensor:batch_token_type_ids,
																  batch_sent_embs__labels:batch_labels})
			epoch_loss+=epoch_batch_loss
		print('Total Loss and Loss per Sample in Epoch {} are : {} and {}, respectively'.format(epoch,epoch_loss,epoch_loss/len(all_indices)))
		#
		path = os.path.join(NEW_CKPT_DIR_CUSTOM_MODEL, BERT_MODEL_NAME+'_Updated'+'_epoch_{}'.format(epoch))
		model_saver.save(sess, path)
"""


# In[ ]:


"""

BATCH_SIZE = 20 # actually, it becomes batch_size*3 in sess below
N_EPOCHS = 3

NEW_CKPT_DIR_CUSTOM_MODEL = '../checkpoints'
if not os.path.exists(NEW_CKPT_DIR_CUSTOM_MODEL):
	os.mkdir(NEW_CKPT_DIR_CUSTOM_MODEL)
	
with tf.Session(config=tf_config) as sess:
	#
	model_saver = tf.train.Saver() # all variables considered
	for epoch in range(N_EPOCHS):
		epoch_loss = 0.0
		n_train_batches = int(math.ceil(TRAIN_NUM/BATCH_SIZE))
		all_indices = np.arange(TRAIN_NUM)
		np.random.shuffle(all_indices) #in-place ops
		for i in tqdm(range(n_train_batches)):
			begin_index = i * BATCH_SIZE
			end_index = min((i + 1) * BATCH_SIZE, TRAIN_NUM)
			batch_index = all_indices[begin_index : end_index]
			#
			batch_intent_input_ids = np.asarray([f.input_ids for f in all_intent_features[batch_index]], dtype=np.int32)
			batch_intent_input_mask = np.asarray([f.input_mask for f in all_intent_features[batch_index]], dtype=np.int32)
			batch_intent_token_type_ids = np.asarray([f.segment_ids for f in all_intent_features[batch_index]], dtype=np.int32)
			#
			batch_pos_query_input_ids = np.asarray([f.input_ids for f in all_pos_query_features[batch_index]], dtype=np.int32)
			batch_pos_query_input_mask = np.asarray([f.input_mask for f in all_pos_query_features[batch_index]], dtype=np.int32)
			batch_pos_query_token_type_ids = np.asarray([f.segment_ids for f in all_pos_query_features[batch_index]], dtype=np.int32)
			#
			batch_neg_query_input_ids = np.asarray([f.input_ids for f in all_neg_query_features[batch_index]], dtype=np.int32)
			batch_neg_query_input_mask = np.asarray([f.input_mask for f in all_neg_query_features[batch_index]], dtype=np.int32)
			batch_neg_query_token_type_ids = np.asarray([f.segment_ids for f in all_neg_query_features[batch_index]], dtype=np.int32)
			#
			batch_input_ids = np.vstack((batch_intent_input_ids,batch_pos_query_input_ids,batch_neg_query_input_ids))
			batch_input_mask =  np.vstack((batch_intent_input_mask,batch_pos_query_input_mask,batch_neg_query_input_mask))
			batch_token_type_ids = np.vstack((batch_intent_token_type_ids,batch_pos_query_token_type_ids,batch_neg_query_token_type_ids))
			#
			_, batch_loss, per_sample_loss = \
				sess.run([train_op, batch_loss, loss], feed_dict={batch_input_ids__tensor:batch_input_ids,
																  batch_input_mask__tensor:batch_input_mask,
																  batch_token_type_ids__tensor:batch_token_type_ids})
			epoch_loss+=batch_loss
		epoch_loss/=len(all_indices)
		#
		path = os.path.join(NEW_CKPT_DIR_CUSTOM_MODEL, BERT_MODEL_NAME+'_Updated'+'_epoch_{}'.format(epoch))
		model_saver.save(sess, path)
"""


# # Extracting Model features directly for inference

# In[ ]:


"""
BATCH_SIZE = 50
all_sent_embs = []

with tf.Session(config=tf_config) as sess:
	#
	sess.run(tf.global_variables_initializer())
	sess.run(tf.tables_initializer())
	#
	n_infer_batches = int(math.ceil(INFER_NUM/BATCH_SIZE))
	all_indices = np.arange(INFER_NUM)
	for i in tqdm(range(n_infer_batches)):   
		begin_index = i * BATCH_SIZE
		end_index = min((i + 1) * BATCH_SIZE, INFER_NUM)
		batch_index = all_indices[begin_index : end_index]
		#
		batch_input_ids = np.asarray([f.input_ids for f in all_sents_features[batch_index]], dtype=np.int32)
		batch_input_mask = np.asarray([f.input_mask for f in all_sents_features[batch_index]], dtype=np.int32)
		batch_token_type_ids = np.asarray([f.segment_ids for f in all_sents_features[batch_index]], dtype=np.int32)
		#
		batch_sent_embs = sess.run(batch_sent_embs__tensor, feed_dict={batch_input_ids__tensor:batch_input_ids,
															   batch_input_mask__tensor:batch_input_mask,
															   batch_token_type_ids__tensor:batch_token_type_ids})
		#
		if len(all_sent_embs)==0:
			all_sent_embs = batch_sent_embs
		else:
			all_sent_embs = np.concatenate((all_sent_embs, batch_sent_embs), axis=0)
			
test_intents_len = len(np.asarray([*label2id_all.keys()]))

all_x_embs = all_sent_embs[:-test_intents_len]
all_y_embs = all_sent_embs[-test_intents_len:]

from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(all_x_embs, all_y_embs)
"""
