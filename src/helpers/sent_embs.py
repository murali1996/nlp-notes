import numpy as np, os, pandas as pd, re, csv, sys
from nltk.corpus import stopwords
from tqdm import tqdm
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
tf_config = tf.ConfigProto()
tf_config.allow_soft_placement = True # Choose alternative device if specified device not found
tf_config.log_device_placement = False
tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9


import tensorflow_hub as hub
# 1.
# if you are running the script directly from helpers/ folder as "python sent_embs.py", then uncomment...
# sys.path.append('..')
# from bert_wrapper_tf.wrapper import Model as BERTModelWrapper
# 2.
# if you are calling as a package standing in codes/ folder which contains both helpers/ and bert_wrapper_tf/ folders, then uncomment...
# from bert_wrapper_tf.wrapper import Model as BERTModelWrapper
# 3.
# if you are calling from a grandparent folder or above as a package, then uncomment... 
# note, this way of calling is not applicable for 1. mode and returns __file__ not found
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
# from bert_wrapper_tf.wrapper import Model as BERTModelWrapper
# So, in totality...
try: 
	sys.path.append('..')
	from bert_wrapper_tf.wrapper import Model as BERTModelWrapper
except:
	sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
	from bert_wrapper_tf.wrapper import Model as BERTModelWrapper


class Module(object):



	def __init__(self, gpu_devices: "can be empty list or a list of device numbers",
				 use3_directory=None, 
				 elmo2_directory=None, 
				 glove_directory=None):
		
		self.gpu_devices = gpu_devices
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
		print("GPU DEVICE SETTINGS:: {}".format(self.gpu_devices))
		
		self.cachedStopWords = stopwords.words("english")
		if use3_directory:
			self.init_use3_hub_model(use3_directory, trainable=False)
		if elmo2_directory:
			self.init_elmo2_hub_model(elmo2_directory, trainable=False)
		if glove_directory:
			self.init_glove_embedding(glove_directory)




	##########################################################
	# pre-processing textual sentences
	def __get_string_tokens_for_glove(self, txt, remove_stopwords=False):
		txt = txt.lower().strip() # glove.6B is UNCASED
		# tokens =  [x.lower() for x in re.split('\W+', txt) if x]
		# tokens = re.sub(r'[^a-zA-Z0-9-]', ' ', txt).split(' ')
		tokens = re.findall(r"[-+]?\d*[.-]\d+|\d+|[A-Za-z0-9_-]+|[?!]", txt)
		if not remove_stopwords:
			tokens = [token for token in tokens if token]
		else:
			tokens = [token for token in tokens if (token and token not in self.cachedStopWords)]
		return tokens


	def __get_string_tokens_for_elmo2(self, txt, remove_stopwords=False):
		txt = txt.strip()
		# tokens = re.findall(r"[\w']+|[\\/,!?;_-]", txt) # covers all contiguous strings with only exceptional 
		#	punctuation "'" and a set of punctuation chars
		# tokens = re.findall(r"[-+]?\d*\.\d+|\d+|[A-Za-z0-9_-]+|[?!]", txt) # covers decimal numbers, integer numbers,
		#	 contiguous strings with only exceptional punctuation "_", "?!" chars
		tokens = re.findall(r"[-+]?\d*[.-]\d+|\d+|[A-Za-z0-9_-]+|[?!]", txt)
		if not remove_stopwords:
			tokens = [token for token in tokens if token]
		else:
			tokens = [token for token in tokens if (token and token not in self.cachedStopWords)]
		return tokens







	##########################################################
	# use3 embeddings
	def init_use3_hub_model(self, use3_directory, trainable=False):
		self.use3_hub_model_graph, self.use3_hub_model, self.use3_hub_model_sess = None, None, None
		self.use3_hub_model_graph = tf.Graph()
		with self.use3_hub_model_graph.as_default():
			with tf.device(self.gpu_devices_n1):
				self.use3_hub_model = hub.Module(use3_directory, trainable=trainable)
		self.use3_hub_model_sess = tf.Session(config=tf_config, graph=self.use3_hub_model_graph)
		self.use3_hub_model_sess.__enter__()
		self.use3_hub_model_sess.run(tf.global_variables_initializer())
		self.use3_hub_model_sess.run(tf.tables_initializer())
		return 
	def exit_use3_hub_model(self):
		self.use3_hub_model_sess.close()
		self.use3_hub_model_graph, self.use3_hub_model, self.use3_hub_model_sess = None, None, None
		return
	def get_use3_sentence_vector(self, list_sentences):
		print("obtaining use3 sentence vector start")
		assert(type(list_sentences)==list)
		assert(type(list_sentences[0])==str)
		use3_vecs = []
		BATCH_SIZE=50
		if len(list_sentences)<BATCH_SIZE:
			use3_vecs = self.use3_hub_model_sess.run(self.use3_hub_model(list_sentences))
		else: # make batches of BATCH_SIZE and send
			N_SAMPLES = len(list_sentences)
			N_BATCHES = int( (N_SAMPLES/BATCH_SIZE)+np.ceil((N_SAMPLES%BATCH_SIZE)/BATCH_SIZE) )
			for batch_ind in tqdm(range(N_BATCHES)):
				batch_list_sentences = list_sentences[batch_ind*BATCH_SIZE:np.min([(batch_ind+1)*BATCH_SIZE,N_SAMPLES])]
				batch_use3_vecs = self.use3_hub_model_sess.run(self.use3_hub_model(batch_list_sentences))
				if len(use3_vecs)==0:
					use3_vecs = batch_use3_vecs
				else:
					use3_vecs = np.vstack((use3_vecs,batch_use3_vecs))
		print("obtaining use3 sentence vector end")
		return use3_vecs # np array with shape [n_list_rows, 512]






	##########################################################
	# elmo2 embeddings
	def init_elmo2_hub_model(self, elmo2_directory, trainable=False):
		self.elmo2_hub_model_graph, self.elmo2_hub_model, self.elmo2_hub_model_sess = None, None, None
		self.elmo2_hub_model_graph = tf.Graph()
		with self.elmo2_hub_model_graph.as_default():
			with tf.device(self.gpu_devices_n1):
				self.elmo2_hub_model = hub.Module(elmo2_directory, trainable=trainable)
				self.input_x_elmo_tokens = tf.placeholder(tf.string, shape=[None, None])
				self.s_len = tf.placeholder(tf.int32, shape=[None])
				self.elmo2_sentlevel_embeddings_tensor = \
					self.elmo2_hub_model(inputs={"tokens": self.input_x_elmo_tokens,"sequence_len": self.s_len},
						signature="tokens",as_dict=True)["default"]
		self.elmo2_hub_model_sess = tf.Session(config=tf_config, graph=self.elmo2_hub_model_graph)
		self.elmo2_hub_model_sess.__enter__()
		self.elmo2_hub_model_sess.run(tf.global_variables_initializer())
		self.elmo2_hub_model_sess.run(tf.tables_initializer())
		return
	def exit_elmo2_hub_model(self):
		self.elmo2_hub_model_sess.close()
		self.elmo2_hub_model_graph, self.elmo2_hub_model, self.elmo2_hub_model_sess = None, None, None
		return
	def get_elmo2_wordseq(self, list_sentences, max_len=-1):
		#
		list_wordseq = [];
		s_len = [];
		replace_extra = "";
		#
		for sent in list_sentences:
			tokens = self.__get_string_tokens_for_elmo2(sent)
			s_len.append(len(tokens))
			list_wordseq.append(tokens)
		#
		max_len = np.max(np.asarray(s_len)) if max_len<0 else max_len
		for i, _ in enumerate(list_wordseq):
			if s_len[i]<max_len:
				extras = max_len-s_len[i]
				list_wordseq[i]+=[replace_extra]*extras
			if s_len[i]>max_len:
				list_wordseq[i] = list_wordseq[i][:max_len]
				s_len[i] = max_len
		return np.asarray(list_wordseq), np.asarray(s_len)
	def get_elmo2_sentence_vector(self, list_sentences):
		print("obtaining elmo2 sentence vector start")
		assert(type(list_sentences)==list)
		assert(type(list_sentences[0])==str)
		#
		input_x_elmo_tokens, s_len = self.get_elmo2_wordseq(list_sentences)
		#
		elmo2_vecs = []
		BATCH_SIZE=50
		if len(input_x_elmo_tokens)<BATCH_SIZE:
			elmo2_vecs = \
				self.elmo2_hub_model_sess.run(self.elmo2_sentlevel_embeddings_tensor,
					feed_dict={self.input_x_elmo_tokens:input_x_elmo_tokens, self.s_len:s_len})
		else: # make batches of BATCH_SIZE and send
			N_SAMPLES = len(input_x_elmo_tokens)
			N_BATCHES = int( (N_SAMPLES/BATCH_SIZE)+np.ceil((N_SAMPLES%BATCH_SIZE)/BATCH_SIZE) )
			for batch_ind in tqdm(range(N_BATCHES)):
				batch_input_x_elmo_tokens = input_x_elmo_tokens[batch_ind*BATCH_SIZE:np.min([(batch_ind+1)*BATCH_SIZE,N_SAMPLES]),:]
				batch_s_len = s_len[batch_ind*BATCH_SIZE:np.min([(batch_ind+1)*BATCH_SIZE,N_SAMPLES])]
				batch_elmo2_vecs = \
					self.elmo2_hub_model_sess.run(self.elmo2_sentlevel_embeddings_tensor,
						feed_dict={self.input_x_elmo_tokens:batch_input_x_elmo_tokens, self.s_len:batch_s_len})
				if len(elmo2_vecs)==0:
					elmo2_vecs = batch_elmo2_vecs
				else:
					elmo2_vecs = np.vstack((elmo2_vecs,batch_elmo2_vecs))
		print("obtaining elmo2 sentence vector end")
		return elmo2_vecs # np array with shape [n_list_rows, 1024]	





	##########################################################
	# BERT PRETRAINED [CLS] embeddings
	def get_bert_cls_sentence_vector(self, list_sentences):
		print("obtaining bert cls sentence vector start")
		model = BERTModelWrapper(gpu_devices=self.gpu_devices)
		model.restore_pretrained_bert_config()
		inputFeatures = model.get_InputFeatures(text_a_list=list_sentences)
		inputFeatures= np.asarray(inputFeatures)
		# each feature object has input_ids, input_mask, segment_ids, label_id, is_real_example attributes

		model.set_base_ops(is_training=False)
		model.restore_weights()

		import math
		INFER_NUM = len(list_sentences)
		BATCH_SIZE = 50
		cls_outputs = []
		full_outputs = []
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
		print("obtaining bert cls sentence vector start")
		return cls_outputs






	##########################################################
	# mean-pooled glove embeddings
	def init_glove_embedding(self, glove_directory):
		print("=======================================================================")
		print("=======================================================================")
		self._glove_unknown =  "__UNKNOWN__"
		glove_file_path = os.path.join(glove_directory, "glove.6B.{}d.txt".format(300))
		print("load glove word embedding begin")
		self._glove_dfg, self._glove_w2id = pd.DataFrame(), {}
		self._glove_dfg = pd.read_csv(glove_file_path, sep=" ", quoting=3, header=None, index_col=0)
		self._glove_dfg.loc[self._glove_unknown] = np.zeros((self._glove_dfg.shape[-1]))
		print('Glove Data Shape: {}'.format(self._glove_dfg.shape))
		print('Considering normalized embeddings')
		self.word2vec_embedding = tool.norm_matrix(self._glove_dfg.values.astype('float32'))
		for i, word in enumerate(self._glove_dfg.index.values):
			self._glove_w2id[word] = i;
		print("load word embedding end")
		return
	def get_idseq_glove(self, list_queries, max_len=-1):
		#
		list_idseq = [];
		list_wordseq = [];
		s_len = [];
		replace_extra = self._glove_unknown;
		#
		for sent in list_queries:
			tokens = self.__get_string_tokens_for_glove(sent)
			s_len.append(len(tokens))
			list_wordseq.append(tokens)
		#
		max_len = np.max(np.asarray(s_len)) if max_len<0 else max_len
		for i, _ in enumerate(list_wordseq):
			if s_len[i]<max_len:
				extras = max_len-s_len[i]
				list_wordseq[i]+=[replace_extra]*extras
			if s_len[i]>max_len:
				list_wordseq[i] = list_wordseq[i][:max_len]
				s_len[i] = max_len
		#
		for _, wordseq in enumerate(list_wordseq):
			this_idseq = [];
			for word in wordseq:
				try:
					this_idseq.append(self._glove_w2id[word])
				except KeyError as error:
					this_idseq.append(self._glove_w2id[replace_extra])
			list_idseq.append(this_idseq)
		return np.asarray(list_idseq), np.asarray(s_len)
	def get_glove_avg_sentence_vector(self, list_sentences):
		print("obtaining avg glove embeddings start")
		assert(type(list_sentences[0])==str)
		assert(type(list_sentences)==list)
		glove_avg_vecs = []
		for sent in list_sentences:
			list_sentences = self.__get_string_tokens_for_glove(sent, remove_stopwords=False);
			glove_avg_vec = [];
			for token in list_sentences:
				try:
					glove_avg_vec.append(self._glove_dfg.loc[token,:].values.tolist()) # list with size of embs
				except:
					glove_avg_vec.append(self._glove_dfg.loc[self._glove_unknown,:].values.tolist()) # list with size of embs
			glove_avg_vec = np.mean(np.asarray(glove_avg_vec),axis=0).tolist()
			glove_avg_vecs.append(glove_avg_vec)
		glove_avg_vecs = np.asarray(glove_avg_vecs)
		print("obtaining avg glove embeddings end")
		return glove_avg_vecs # np array with shape [n_list_rows, size of glove embedding]





	##########################################################
	# tensorboard projections
	def tfboard_projections(
		self,
		ckpt_dir,
		myEmbeddings: "array of vectors",
		myName: "tensor data will be saved as myName.ckpt",
		metadata: "np array of shape: [len(myEmbeddings),len(metadata_names)]" = [],
		metadata_names: "list of names which indicate corresponding data to be seen in metadata" = []
		):
		# bash commands to launch tensorboard
		# cd './projections_checkpoints/<myName>'
		# tensorboard --logdir=. --port=8870
		if not len(metadata)==0:
			assert len(myEmbeddings)==metadata.shape[0]
		else:
			metadata = np.arange(len(myEmbeddings)).reshape(-1,1)
		if not len(metadata_names)==0:
			assert len(metadata_names)==metadata.shape[1]
		else:
			metadata_names = ["column_number_{}".format(i+1) for i in range(metadata.shape[1])]
		save_dir = os.path.join(ckpt_dir,myName)
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		# create session, initialize the tensor and save in a checkpoint	
		myGraph = tf.Graph()
		with myGraph.as_default():
			with tf.variable_scope('tf_board', reuse=tf.AUTO_REUSE):
				myEmbeddingTensor = \
					tf.get_variable(name=myName, shape=myEmbeddings.shape, 
						initializer=tf.constant_initializer(myEmbeddings), trainable=False)
				print(myEmbeddingTensor)
		mySess = tf.Session(graph=myGraph,config=tf_config)
		mySess.__enter__()
		mySess.run(myEmbeddingTensor.initializer)
		with myGraph.as_default():
			mySaver = tf.train.Saver([myEmbeddingTensor]) # Save only this variable
		mySaver.save(mySess, os.path.join(save_dir, myName+'.ckpt'))
		
		# save metadata if any
		tsv_path_local = str(myName)+'_labels.tsv'
		tsv_path = os.path.join(save_dir, tsv_path_local);
		with open(tsv_path, 'w', encoding='utf8') as tsv_file:
			tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
			if len(metadata_names)>1:
				tsv_writer.writerow(metadata_names)
			for row in metadata:
				tsv_writer.writerow([i for i in row]) # [label, cnt] # [label]
		
		# launch tensorboard projector
		myProjector = projector.ProjectorConfig()
		embedding = myProjector.embeddings.add()
		embedding.tensor_name = myEmbeddingTensor.name
		embedding.metadata_path = tsv_path_local
		projector.visualize_embeddings(tf.summary.FileWriter(save_dir), myProjector) # Saves a config file that TensorBoard will read during startup.
		print('Visualization Data Saved at: {}'.format(save_dir))
		return


if __name__=="__main__":

	print("in main")

	# ======================================================================
	# Set Paths
	# ======================================================================
	DATA_FOLDER = '../../../../DATA'
	glove_directory = os.path.join(DATA_FOLDER, 'WORD_EMBEDDINGS/glove.6B/')
	use3_directory = os.path.join(DATA_FOLDER, 'TFHUB_MODELS/use3/')
	elmo2_directory = os.path.join(DATA_FOLDER, 'TFHUB_MODELS/elmo2/')

	# ======================================================================
	# Get Sentences
	# ======================================================================
	list_sentences = []
	file = open("sentences.txt", "r")
	for row in file:
		list_sentences.append(row.split("\t")[0].strip())	
	file.close()

	# ======================================================================
	# Execute required model(s)
	# ======================================================================	
	projections_ckpt_dir = './projections_checkpoints/'
	# elmo
	module = Module(gpu_devices=[0], elmo2_directory=elmo2_directory)
	myEmbs = module.get_elmo2_sentence_vector(list_sentences)
	module.tfboard_projections(ckpt_dir=projections_ckpt_dir, myEmbeddings=myEmbs, myName="elmo2_hhp_train_intents", 
									metadata=np.asarray(list_sentences).reshape(-1,1), 
									metadata_names=np.asarray(["train_labels"]))
	# bert cls
	module = Module(gpu_devices=[0])
	myEmbs = module.get_bert_cls_sentence_vector(list_sentences)
	module.tfboard_projections(ckpt_dir=projections_ckpt_dir, myEmbeddings=myEmbs, myName="bert_cls_hhp_train_intents", 
									metadata=np.asarray(list_sentences).reshape(-1,1), 
									metadata_names=np.asarray(["train_labels"]))
	

