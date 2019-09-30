# ======================================================================
# coding: utf-8
# Author: Sai Muralidhar 
# ======================================================================
# ======================================================================
# some terminology
# :todo:
# :notes:
# :examples:
# ======================================================================
# ======================================================================
# :notes:
# Changes made wrt original code of BERT at https://github.com/google-research/bert
# 1) In modeling.py 
#       In BertModel() class, made the following addition:
#           def get_cls_vector(self):
#               return self.first_token_tensor
# 2) In run_classifier.py()
#       In convert_single_example() method, made the following replacement
#           As-is:
#               label_id = label_map[example.label]
#           To-be:
#               try:
#                   label_id = label_map[example.label]
#               except:
#                   label_id = None
#               # And then comment the print lines below it
# 3) In run_classifier.py()
#       In the imports line, made the following replacement
#           As-is:
#               import modeling, optimization, tokenization
#           To-be:
#               from . import modeling, optimization, tokenization
# 4) In run_classifier.py()
#       In InputFeatures() class, made the following addition
#           self.token_len = token_len,
#           self.joined_tokens = joined_tokens,
#       In convert_single_example() method, 
#           In isinstance(example, PaddingInputExample) check,
#               In the return object, made the following addition
#                   token_len = max_seq_length,
#                   joined_tokens = "",
#           In feature = InputFeatures(...) object instantiation, made the following addition
#               token_len = len(tokens),
#               joined_tokens = " ".join(tokens),
# ======================================================================


print("/".join(__file__.split("/")[:-1]))

import os, sys, math, numpy as np, pandas as pd
import json
from tqdm import tqdm # to maintain progress bar
import shutil # to copy files
import tensorflow as tf



# ======================================================================
# Download & Load BERT codes and make changes to files as required in :notes:
# ======================================================================
#
# 1. Download codes
# !git clone https://murali1996:<password>@github.com/murali1996/nlp.git
# !git clone https://github.com/google-research/bert
#
# 2. Download pretrained weights
# BERT_PRETRAINED_MODELS_DIR = os.path.join('./checkpoints','BERT_PRETRAINED_MODELS')
# BERT_MODEL_NAME = "wwm_cased_L-24_H-1024_A-16" #"cased_L-12_H-768_A-12"
# BERT_PRETRAINED_DIR = os.path.join(BERT_PRETRAINED_MODELS_DIR,BERT_MODEL_NAME)
# if not os.path.exists(BERT_PRETRAINED_MODELS_DIR):
#   os.mkdir(BERT_PRETRAINED_MODELS_DIR)
# date = "2019_05_30" #2018_10_18
# os.system('wget https://storage.googleapis.com/bert_models/{}/{}.zip -O {}/{}.zip'   \ 
#             .format(date,BERT_MODEL_NAME,BERT_PRETRAINED_MODELS_DIR,BERT_MODEL_NAME))
# os.system('unzip {}/{}.zip -d {}/'.format(BERT_PRETRAINED_MODELS_DIR,BERT_MODEL_NAME,BERT_PRETRAINED_MODELS_DIR))







# ======================================================================
# Bert-Wrapper Model Class
# ======================================================================

from .bert_master import run_classifier, optimization, tokenization, modeling

class Model(object):
    """
    - Custom Model is the one built on top of BERT and include both pretrained weights of BERT as well as some new vars 
      used for ops on top of BERT
    - Using this class, you can:
        - Load and Use results from a pretrained BERT, trained as described in https://arxiv.org/pdf/1810.04805v2.pdf
        - :todo: Finetune BERT weights on you own data with the same training objective as in BERT
        - Train a custom model on top of BERT with your own objective; Please manually define your ops in 
          set_custom_model_ops() method below
    - While saving details of custom model on top of pretrained BERT, necessary flies like vocab, etc. must also be save
      in that destination directory
        - This is to be in compatible with relaoding bert weights upon saving custom model
    - On reloading the custom model, those new path locations are to be specified for the set_pretrained_bert_detail() 
      method
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
        # examples: uncased_L-12_H-768_A-12 or cased_L-12_H-768_A-12 or wwm_cased_L-24_H-1024_A-16
        self.src_ckpt_dir = os.path.join("/".join(__file__.split("/")[:-1]), 
                                './checkpoints/BERT_PRETRAINED_MODELS/uncased_L-12_H-768_A-12') 
        self.src_model_name = "bert_model.ckpt"
        self.dest_ckpt_dir = os.path.join("/".join(__file__.split("/")[:-1]), 
                                './checkpoints/CUSTOM_MODELS')
        self.dest_model_name = "bert_model.ckpt"
        for key, value in self.__dict__.items():
            text = "INITIAL_SETTINGS:: {} = {}".format(key,value)
            print(text)
        #
        self.DO_LOWER_CASE = None
        self.MAX_SEQ_LENGTH = 128
        self.append_cased_info_dir_name = None
        self.__init__tf()
    def __init__tf(self):
        self.tf_graph = tf.Graph()
        self.tf_writer = None
        self.tf_saver = None
    #
    # DATA PROCESS and SAVE for BERT
    def get_InputFeatures(self, text_a_list, text_b_list=[],
                          guid_list=[], label_list:"can be strings too; should be present in all_unique_labels"=[], 
                          all_unique_labels:"can be label strings too"=[]):
        #
        if len(text_b_list)!=0:
            assert len(text_a_list)==len(text_b_list)
        elif len(text_b_list)!=0 and len(label_list)!=0:
            assert len(text_a_list)==len(text_b_list)==len(label_list)
        elif len(text_b_list)!=0 and len(label_list)!=0 and len(guid_list)!=0:
            assert len(text_a_list)==len(text_b_list)==len(label_list)==len(guid_list)
        #
        if not len(all_unique_labels)==0: # retaining the order
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
            inputExamples.append(run_classifier.InputExample(guid=guid_list[i], text_a=text_a_list[i], 
                                                             text_b=text_b_list[i], label=label_list[i]))
        inputFeatures = run_classifier.convert_examples_to_features(inputExamples, all_unique_labels, 
                                                                    self.MAX_SEQ_LENGTH, self.tokenizer)
        # input_ids, input_mask, segment_ids, label_id, is_real_example are the attributes for a inputFeauture object
        return inputFeatures
        #
    def dump_tokenized_sents(self, *args : "x_tr_inputFeatures, x_tr_raw, y_tr_inputFeatures, y_tr_raw", 
                             column_names, file_title, ckpt_dir=None):
        n_args = len(args)
        same_len = -1
        for arg in args:
            same_len = len(arg) if same_len==-1 else same_len
            if same_len!=len(arg):
                raise Exception("In dump_tokenized_sents(), all arguments provided must have same length!!")
        assert n_args==len(column_names)
        #
        if ckpt_dir==None:
            ckpt_dir = self.__get_default_dest_ckpt_dir()
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        #
        excel = []
        for i in range(same_len):
            excel.append([arg[i] for arg in args])
        df = pd.DataFrame(excel, columns=column_names)
        filepath = os.path.join(ckpt_dir, file_title+'.xlsx')
        df.to_excel(filepath, index=False)
        print("Saving tokenized sents: {}".format(filepath))
        return
    def dump_json(self, dict_record, file_title, open_mode: "a->append, w->write", ckpt_dir=None):
        #
        if ckpt_dir==None:
            ckpt_dir = self.__get_default_dest_ckpt_dir()
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        #
        def default(o):
            if isinstance(o, np.int64): return int(o)  
            raise TypeError
        now_open = open(os.path.join(ckpt_dir, file_title+".json"),open_mode)
        now_open.write("\n")
        now_open.write(json.dumps(dict_record, indent=4, sort_keys=False, default=default))
        now_open.close()
        return
    #
    # SAVE and RESTORE vars
    def __get_default_src_ckpt_dir(self):
        # default
        return self.src_ckpt_dir
    def __get_default_dest_ckpt_dir(self):
        ckpt_dir = self.dest_ckpt_dir
        print('Using known destination ckpt dir')
        if not self.append_cased_info_dir_name==None:
            print('Since *cased* info was derived from a folder name, to retain this info...')
            print('Creating a folder named {} in destination ckpt dir to save the weights' \
                   .format(self.src_ckpt_dir.split("/")[-1]))
            ckpt_dir = os.path.join(ckpt_dir, self.append_cased_info_dir_name)
        return ckpt_dir
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
    def save_weights(self, sess, model_name=None, ckpt_dir=None):
        if model_name==None:
            model_name = self.dest_model_name
        if ckpt_dir==None:
            ckpt_dir = self.__get_default_dest_ckpt_dir()
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        ckpt_path = os.path.join(ckpt_dir, model_name)
        self.tf_saver.save(sess, ckpt_path)
        self._save_pretrained_bert_config(ckpt_dir=ckpt_dir)
        #
        additional_dict = { "DO_LOWER_CASE":self.DO_LOWER_CASE, 
                            "MAX_SEQ_LENGTH":self.MAX_SEQ_LENGTH, 
                            "append_cased_info_dir_name":self.append_cased_info_dir_name
                          }
        now_open = open(os.path.join(ckpt_dir, "custom_model_dict.json"),"w")
        now_open.write("\n")
        now_open.write(json.dumps(additional_dict, indent=4, sort_keys=False))
        now_open.close()
        return
    def restore_weights(self, sess, model_name=None, ckpt_dir=None, print_tvars=False):
        if model_name==None:
            model_name = self.src_model_name
        if ckpt_dir==None:
            ckpt_dir = self.__get_default_src_ckpt_dir()
            ckpt_path = os.path.join(ckpt_dir, model_name)  
            print('Using known source ckpt dir')
            with self.tf_graph.as_default():
                tvars = tf.trainable_variables()
                (ass_map,initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, ckpt_path)
                tf.train.init_from_checkpoint(ckpt_path, ass_map) 
                # Values are not loaded immediately, but when the initializer is run 
                # (typically by running a tf.global_variables_initializer op)
                if print_tvars:
                    tf.logging.info("**** Trainable Variables ****")
                    for var in tvars:
                        init_string = ""
                        if var.name in initialized_variable_names:
                            init_string = ", *INIT_FROM_CKPT*"
                        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)
            sess.run(self.init_op)
        else:
            ckpt_path = os.path.join(ckpt_dir, model_name)
            self.tf_saver.restore(sess, ckpt_path)
        return
    def restore_pretrained_bert_config(self,
                                       ckpt_dir=None, 
                                       cased: "True/ False/ None; If none, detects automatically"=None, 
                                       max_seq_len=None):
        if ckpt_dir==None:
            print('Using known source ckpt dir')
            ckpt_dir = self.__get_default_src_ckpt_dir()
        if cased==None:
            if (ckpt_dir.split("/")[-1].startswith('uncased') or ckpt_dir.split("/")[-1].startswith('cased')):
                self.DO_LOWER_CASE = ckpt_dir.split("/")[-1].startswith('uncased')
                self.append_cased_info_dir_name = ckpt_dir.split("/")[-1]
            elif (ckpt_dir.split("/")[-2].startswith('uncased') or ckpt_dir.split("/")[-2].startswith('cased')):
                self.DO_LOWER_CASE = ckpt_dir.split("/")[-2].startswith('uncased')
                self.append_cased_info_dir_name = ckpt_dir.split("/")[-2]
            else:
                raise Exception('The model\'s folder must start with either *cased* or *uncased* string \
                                since this arg is not set in func call')
        else:
            self.DO_LOWER_CASE = not cased
            self.append_cased_info_dir_name = None
        if not max_seq_len==None:
            self.MAX_SEQ_LENGTH = max_seq_len 
        #
        print("The following files will be loaded: vocab.txt, bert_config.json")
        self.VOCAB_FILE = os.path.join(ckpt_dir, 'vocab.txt')
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.VOCAB_FILE, do_lower_case=self.DO_LOWER_CASE) 
        self.CONFIG_FILE = os.path.join(ckpt_dir, 'bert_config.json')   
        self.bert_config = modeling.BertConfig.from_json_file(self.CONFIG_FILE)     
        print(self.tokenizer.tokenize("This is a simple example of how the BERT's FullTokenizer tokenizes text. "))
        return
    #
    # TF GRAPH (EXTRAS)
    def _print_ops_devices(self):
        print("=======================================================================")
        print("=======================================================================")
        print("Printing device assignment information...")
        device_info = [n.device for n in self.tf_graph.as_graph_def().node]
        for info in device_info:
            print(info)
        return
    def _print_custom_ops(self):
        print("=======================================================================")
        print("=======================================================================")
        print("Printing Custom ops information...")
        list_ = [(n.name, n.device) for n in self.tf_graph.as_graph_def().node if "custom_ops" in n.name.split('/')[0]]
        for item in list_:
            print(item)
        return      
    def _print_placeholders_list(self):
        print("=======================================================================")
        print("=======================================================================")
        print("Printing Placeholders information...")
        # [ op for op in self.tf_graph.get_operations() if op.type == "Placeholder"]
        list_ = [(n.name, n.device) for n in self.tf_graph.as_graph_def().node if n.op=="Placeholder"]
        for item in list_:
            print(item)
        return
    def _print_trainables_counts(self):
        print("=======================================================================")
        print("=======================================================================")
        print("Printing tf.trainable_variables() information...")
        trainable_weights = 0
        trainable_vars = 0
        with self.tf_graph.as_default():
            for var_ in tf.trainable_variables():
                trainable_vars+=1
                var_weights = 1
                for dim in var_.get_shape().as_list():
                    var_weights*=dim
                trainable_weights+=var_weights
            print("#trainable_vars: {}, #trainable_units: {}".format(trainable_vars, trainable_weights))
        return
    # 
    # TF GRAPH OPS (MAIN)
    def __get_init_ops(self):
        init_op = tf.group([tf.global_variables_initializer(),tf.tables_initializer()])
        return init_op
    def __get_train_ops(self, loss, bert_lr: "tf tensor or python scalar", custom_lr: "tf tensor or python scalar"):
        # The bert_lr is better kept around 2e-5 in order to avoid catastrophic forgetting and to support training set 
        # convergence. In alternative to below approach, you can find gradients first using tf.gradients(loss, vars) and
        # then use optimizer.apply_gradients()
        with self.tf_graph.as_default():
            #_vars = tf.trainable_variables()
            bert_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="bert")
            custom_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="custom_ops")
        optimizer1 = tf.train.AdamOptimizer(bert_lr)
        train_op1 = optimizer1.minimize(loss, var_list=bert_vars)
        optimizer2 = tf.train.AdamOptimizer(custom_lr)
        train_op2 = optimizer1.minimize(loss, var_list=custom_vars)
        #
        train_op = tf.group([train_op1,train_op2])
        return train_op 
    def __set_saver_ops(self):
        with self.tf_graph.as_default():
            self.tf_saver = tf.train.Saver()
            print("tf_saver initialized; any new vars defined after this point will not be saved by attribute tf_saver")
        return
    def set_base_ops(self, is_training):
        with self.tf_graph.as_default():
            with tf.device(self.gpu_devices_n1):
                self.batch_input_ids__tensor = tf.placeholder(dtype=tf.int32, shape=(None,self.MAX_SEQ_LENGTH), 
                                                              name="batch_input_ids__tensor")
                self.batch_input_mask__tensor = tf.placeholder(dtype=tf.int32, shape=(None,self.MAX_SEQ_LENGTH), 
                                                               name="batch_input_mask__tensor")
                self.batch_token_type_ids__tensor = tf.placeholder(dtype=tf.int32, shape=(None,self.MAX_SEQ_LENGTH),
                                                                   name="batch_itoken_type_ids__tensor")
                # in the scope below, add any different name and you have issues loading data from Checkpoint
                self.bertModel = modeling.BertModel(self.bert_config,
                                                    is_training=is_training,
                                                    input_ids=self.batch_input_ids__tensor,
                                                    input_mask=self.batch_input_mask__tensor,
                                                    token_type_ids=self.batch_token_type_ids__tensor,
                                                    scope="bert" 
                                               )
                self.cls_output = self.bertModel.get_cls_vector()
                self.full_output = self.bertModel.get_sequence_output()
        return
    def set_custom_ops_BCELoss(self, is_training, hidden_dropout_prob=None):
        hidden_dropout_prob = 0 if is_training else self.bert_config.hidden_dropout_prob \
                                    if hidden_dropout_prob==None else hidden_dropout_prob
        with self.tf_graph.as_default():
            with tf.device(self.gpu_devices_n1):
                self.true_labels = tf.placeholder(tf.float32, shape=(None, 1), name="true_labels")
                self.bert_lr = tf.placeholder(tf.float32, shape=(), name="bert_lr")
                self.custom_lr = tf.placeholder(tf.float32, shape=(), name="custom_lr")
                #
                with tf.variable_scope("custom_ops"):
                    self.cls_dense1 = tf.layers.dense( 
                        tf.nn.dropout(self.cls_output, keep_prob=1-hidden_dropout_prob),
                        512,
                        activation=tf.tanh,
                        kernel_initializer=modeling.create_initializer(self.bert_config.initializer_range))
                    self.cls_dense2 = tf.layers.dense(
                        tf.nn.dropout(self.cls_dense1, keep_prob=1-hidden_dropout_prob),
                        256,
                        activation=tf.tanh,
                        kernel_initializer=modeling.create_initializer(self.bert_config.initializer_range))
                    self.predicted_logits = tf.layers.dense(
                        tf.nn.dropout(self.cls_dense2, keep_prob=1-hidden_dropout_prob),
                        1,
                        activation=tf.nn.sigmoid,
                        kernel_initializer=modeling.create_initializer(self.bert_config.initializer_range))
                    #
                    self.batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predicted_logits, 
                                                                              labels=self.true_labels)
                    self.loss = tf.reduce_mean(self.batch_loss)
                #
                self.train_op = self.__get_train_ops(self.loss, self.bert_lr, self.custom_lr)
                self.init_op = self.__get_init_ops()
                self.__set_saver_ops()
        self._print_placeholders_list()
        self._print_trainables_counts()
        #self._print_custom_ops()
        return
    def set_custom_ops_TripletLoss(self, is_training, hidden_dropout_prob=None):
        hidden_dropout_prob = 0 if is_training else self.bert_config.hidden_dropout_prob \
                                    if hidden_dropout_prob==None else hidden_dropout_prob
        with self.tf_graph.as_default():
            with tf.device(self.gpu_devices_n1):
                self.part_size = tf.placeholder(tf.int32, shape=(), name="part_size")
                self.bert_lr = tf.placeholder(tf.float32, shape=(), name="bert_lr")
                self.custom_lr = tf.placeholder(tf.float32, shape=(), name="custom_lr")
                self.margin_gamma = tf.placeholder(tf.float32, shape=(), name="margin_gamma")
                #
                with tf.variable_scope("custom_ops"):
                    self.query_w1 = tf.get_variable(
                                            "query_w1", 
                                            shape=(self.cls_output.get_shape().as_list()[-1], 256), 
                                            initializer=modeling.create_initializer(self.bert_config.initializer_range))
                    self.label_w1 = tf.get_variable(
                                            "label_w1",
                                            shape=(self.cls_output.get_shape().as_list()[-1], 256),
                                            initializer=modeling.create_initializer(self.bert_config.initializer_range))
                    #
                    self.cls_output_ =  tf.nn.dropout(self.cls_output, keep_prob=1-hidden_dropout_prob)
                    self.qvec = self.cls_output_[:self.part_size,:]
                    self.pvec = self.cls_output_[self.part_size:2*self.part_size,:]
                    self.nvec = self.cls_output_[2*self.part_size:3*self.part_size,:]
                    self.qvec_prime = tf.matmul(self.qvec, self.query_w1)
                    self.pvec_prime = tf.matmul(self.pvec, self.label_w1)
                    self.nvec_prime = tf.matmul(self.nvec, self.label_w1)
                    #
                    self.batch_pdist = tf.norm(self.qvec_prime-self.pvec_prime, ord="euclidean", axis=-1)
                    self.batch_ndist = tf.norm(self.qvec_prime-self.nvec_prime, ord="euclidean", axis=-1)
                    self.batch_loss = tf.nn.relu(self.margin_gamma+self.batch_pdist-self.batch_ndist)
                    self.loss = tf.reduce_mean(self.batch_loss)
                    #
                    self.pdist = tf.reduce_mean(self.batch_pdist)
                    self.ndist = tf.reduce_mean(self.batch_ndist)
                    self.qvec_prime_infer = tf.matmul(self.cls_output, self.query_w1, name="qvec_prime_infer")
                    self.pvec_prime_infer = tf.matmul(self.cls_output, self.label_w1, name="pvec_prime_infer")
                #
                self.train_op = self.__get_train_ops(self.loss, self.bert_lr, self.custom_lr)
                self.init_op = self.__get_init_ops()
                self.__set_saver_ops()
        self._print_placeholders_list()
        self._print_trainables_counts()
        #self._print_custom_ops()
        return
    def set_custom_ops_SquashAndTripletLoss(self, is_training, hidden_dropout_prob=None):
        hidden_dropout_prob = 0 if is_training else self.bert_config.hidden_dropout_prob \
                                    if hidden_dropout_prob==None else hidden_dropout_prob
        with self.tf_graph.as_default():
            with tf.device(self.gpu_devices_n1):
                self.part_size = tf.placeholder(tf.int32, shape=(), name="part_size")
                self.bert_lr = tf.placeholder(tf.float32, shape=(), name="bert_lr")
                self.custom_lr = tf.placeholder(tf.float32, shape=(), name="custom_lr")
                self.margin_gamma = tf.placeholder(tf.float32, shape=(), name="margin_gamma")
                #
                with tf.variable_scope("custom_ops"):
                    with tf.variable_scope("weights_and_biases"):
                        query_h1 = self.cls_output.get_shape().as_list()[-1]
                        self.query_w1 = tf.get_variable(
                                                "query_w1",
                                                shape=(self.cls_output.get_shape().as_list()[-1], query_h1),
                                                initializer=modeling.create_initializer(self.bert_config.initializer_range))
                        self.query_b1 = tf.get_variable(
                                                "query_b1",
                                                shape=(query_h1), 
                                                initializer=tf.zeros_initializer())
                        query_h2 = 128
                        self.query_w2 = tf.get_variable(
                                                "query_w2",
                                                shape=(query_h1, query_h2),
                                                initializer=modeling.create_initializer(self.bert_config.initializer_range))
                        self.query_b2 = tf.get_variable(
                                                "query_b2",
                                                shape=(query_h2), 
                                                initializer=tf.zeros_initializer())
                        #
                        label_h1 = self.cls_output.get_shape().as_list()[-1]
                        self.label_w1 = tf.get_variable(
                                                "label_w1",
                                                shape=(self.cls_output.get_shape().as_list()[-1], label_h1),
                                                initializer=modeling.create_initializer(self.bert_config.initializer_range))
                        self.label_b1 = tf.get_variable(
                                                "label_b1",
                                                shape=(label_h1), 
                                                initializer=tf.zeros_initializer())
                        label_h2 = 128
                        self.label_w2 = tf.get_variable(
                                                "label_w2",
                                                shape=(label_h1, label_h2),
                                                initializer=modeling.create_initializer(self.bert_config.initializer_range))
                        self.label_b2 = tf.get_variable(
                                                "label_b2",
                                                shape=(label_h2), 
                                                initializer=tf.zeros_initializer())
                    #
                    def __get_query_h2(input_vec, hidden_dropout_prob):
                        h1 = tf.nn.dropout( tf.nn.tanh(tf.nn.xw_plus_b(input_vec, self.query_w1, self.query_b1)),
                                            keep_prob=1-hidden_dropout_prob)
                        h2 = tf.nn.dropout( tf.nn.tanh(tf.nn.xw_plus_b(h1, self.query_w2, self.query_b2)),
                                            keep_prob=1-hidden_dropout_prob)
                        return h2
                    def __get_label_h2(input_vec, hidden_dropout_prob):
                        h1 = tf.nn.dropout( tf.nn.tanh(tf.nn.xw_plus_b(input_vec, self.label_w1, self.label_b1)),
                                            keep_prob=1-hidden_dropout_prob)
                        h2 = tf.nn.dropout( tf.nn.tanh(tf.nn.xw_plus_b(h1, self.label_w2, self.label_b2)),
                                            keep_prob=1-hidden_dropout_prob)
                        return h2
                    #
                    self.cls_output_ =  tf.nn.dropout(self.cls_output, keep_prob=1-hidden_dropout_prob)
                    self.qvec = self.cls_output_[:self.part_size,:]
                    self.pvec = self.cls_output_[self.part_size:2*self.part_size,:]
                    self.nvec = self.cls_output_[2*self.part_size:3*self.part_size,:]
                    #
                    self.qvec_h2 = __get_query_h2(self.qvec, hidden_dropout_prob)
                    self.pvec_h2 = __get_label_h2(self.pvec, hidden_dropout_prob)
                    self.nvec_h2 = __get_label_h2(self.nvec, hidden_dropout_prob)
                    #
                    self.batch_pdist = tf.norm(self.qvec_h2-self.pvec_h2, ord="euclidean", axis=-1)
                    self.batch_ndist = tf.norm(self.qvec_h2-self.nvec_h2, ord="euclidean", axis=-1)
                    self.batch_loss = tf.nn.relu(self.margin_gamma+self.batch_pdist-self.batch_ndist)
                    self.loss = tf.reduce_mean(self.batch_loss)
                    #
                    self.pdist = tf.reduce_mean(self.batch_pdist)
                    self.ndist = tf.reduce_mean(self.batch_ndist)
                    self.qvec_prime_infer = __get_query_h2(self.cls_output, hidden_dropout_prob=0)
                    self.pvec_prime_infer = __get_label_h2(self.cls_output, hidden_dropout_prob=0)
                #
                self.train_op = self.__get_train_ops(self.loss, self.bert_lr, self.custom_lr)
                self.init_op = self.__get_init_ops()
                self.__set_saver_ops()
        self._print_placeholders_list()
        self._print_trainables_counts()
        #self._print_custom_ops()
        return
    def set_custom_ops_PairwiseLoss(self, is_training, hidden_dropout_prob=None):
        hidden_dropout_prob = 0 if is_training else self.bert_config.hidden_dropout_prob \
                                    if hidden_dropout_prob==None else hidden_dropout_prob
        with self.tf_graph.as_default():
            with tf.device(self.gpu_devices_n1):
                self.part_size = tf.placeholder(tf.int32, shape=(), name="part_size")
                self.bert_lr = tf.placeholder(tf.float32, shape=(), name="bert_lr")
                self.custom_lr = tf.placeholder(tf.float32, shape=(), name="custom_lr")
                self.margin_gamma = tf.placeholder(tf.float32, shape=(), name="margin_gamma")
                #
                with tf.variable_scope("custom_ops"):
                    self.query_w1 = tf.get_variable(
                                            "query_w1", 
                                            shape=(self.cls_output.get_shape().as_list()[-1], 256), 
                                            initializer=modeling.create_initializer(self.bert_config.initializer_range))
                    self.label_w1 = tf.get_variable(
                                            "label_w1",
                                            shape=(self.cls_output.get_shape().as_list()[-1], 256),
                                            initializer=modeling.create_initializer(self.bert_config.initializer_range))
                    #
                    self.cls_output_ =  tf.nn.dropout(self.cls_output, keep_prob=1-hidden_dropout_prob)
                    self.qvec = self.cls_output_[:self.part_size,:]
                    self.pvec = self.cls_output_[self.part_size:2*self.part_size,:]
                    self.nvec = self.cls_output_[2*self.part_size:3*self.part_size,:]
                    self.qvec_prime = tf.matmul(self.qvec, self.query_w1)
                    self.pvec_prime = tf.matmul(self.pvec, self.label_w1)
                    self.nvec_prime = tf.matmul(self.nvec, self.label_w1)
                    #
                    self.batch_pdist = tf.norm(self.qvec_prime-self.pvec_prime, ord="euclidean", axis=-1)
                    self.batch_ndist = tf.norm(self.qvec_prime-self.nvec_prime, ord="euclidean", axis=-1)
                    self.batch_loss = self.batch_pdist + tf.nn.relu(self.margin_gamma-self.batch_ndist)
                    self.loss = tf.reduce_mean(self.batch_loss)
                    #
                    self.pdist = tf.reduce_mean(self.batch_pdist)
                    self.ndist = tf.reduce_mean(self.batch_ndist)
                    self.qvec_prime_infer = tf.matmul(self.cls_output, self.query_w1, name="qvec_prime_infer")
                    self.pvec_prime_infer = tf.matmul(self.cls_output, self.label_w1, name="pvec_prime_infer")
                #
                self.train_op = self.__get_train_ops(self.loss, self.bert_lr, self.custom_lr)
                self.init_op = self.__get_init_ops()
                self.__set_saver_ops()
        self._print_placeholders_list()
        self._print_trainables_counts()
        #self._print_custom_ops()
        return
    '''
    def finetune_bert_weights(self, data=None):
        print("Function unavailable as of now...please run with your custom objective with pretrained bert-weights")
        return
    '''


# HOW TO RUN

"""bash
test_sentences = []
file = open('sentences.txt')
for row in file:
    test_sentences.append(row.split('\t')[0].strip())
file.close()

model = Model(gpu_devices=[0])
model.restore_pretrained_bert_config(max_seq_len=64)
inputFeatures = model.get_InputFeatures(text_a_list=test_sentences)
inputFeatures = np.asarray(inputFeatures)
# each feature object has input_ids, input_mask, segment_ids, label_id, token_len, joined_tokens, is_real_example attrs

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
    sess.run(model.__get_init_ops())
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
