# UPDATE

This file is no longer actively maintained. If you are interested in maintaining/updating it, feel free to update by raising PRs or by reaching out to `jsaimurali001 [at] gmail [dot] com`

# READINGS_NLP

- [Text Comprehension](#Comprehension)
    - [Evaluation](#Evaluation)
    - [Word and Sentence Embeddings](#Word-and-Sentence-Embeddings)
    - [Contextual Representations and Transfer Learning](#Contextual-Representations-and-Transfer-Learning)
    - [Multi-task learning](#Multi-task-learning)
    - [Multi-lingual and cross-lingual learning](#Multi-lingual-and-cross-lingual-learning)
    - [Multi-modal learning](#Multi-modal-learning)
    - [Interpretability and Ethics](#Interpretability-and-Ethics)
- [Text Generation](#Generation)
- [IR and QA](#IR-and-QA)
    - [Knowledge Graphs](#Knowledge-Graphs)
    - [Question Answering](#Question-Answering)
- [BERT & Transformers](#Bert--Transformers)
- [Active Learning](#Active-Learning )
- [Notes](#Notes)
- [Bookmarks](#Bookmarks)

# Word and Sentence Embeddings

### word-level representations

1. [Natural Language Processing (almost) from Scratch, Collobert et al. 2011](https://arxiv.org/abs/1103.0398)
1. [*Word2Vec*, Efficient Estimation of Word Representations in Vector Space, Mikolov et al. 2013a][Mikolov et al. 2013a]
1. [*Word2Vec*, Distributed Representations of Words and Phrases and their Compositionality, Mikolov et al. 2013b][Mikolov et al. 2013b]

:bulb: see Ruder's 3 parts of explanation-[p1](https://ruder.io/word-embeddings-1/), [p2](http://ruder.io/word-embeddings-softmax/index.html), [p3](http://ruder.io/secret-word2vec/) along with
his [Aylien blog](https://blog.aylien.com/overview-word-embeddings-history-word2vec-cbow-glove/), Chris McCormick's take on Negative
Sampling [here](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/) along with [resources](http://mccormickml.com/2016/04/27/word2vec-resources/) to reimplement,
see [here](http://www.claudiobellei.com/2018/01/06/backprop-word2vec/) for backprop derivations in word2vec and [here](https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz)
to download pretrained embeddings

1. [GloVe: Global Vectors for Word Representation, Pennington et al. 2014][Pennington et al. 2014]

### character-level representations

1. <https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html>(G Lample et al. 2016)
1. [*ELMo*, Deep contextualized word representations, Peters et al. 2018][Peters et al. 2018]
    - [Character-Aware Neural Language Models](https://arxiv.org/abs/1508.06615), [Exploring the Limits of Language Modeling](https://arxiv.org/abs/1602.02410)
      , [Highway Networks](https://arxiv.org/abs/1505.00387)
1. [*FLAIR*, Contextual String Embeddings for Sequence Labeling, Akbik et al. 2018][Akbik et al. 2018] [[CODE]](https://github.com/zalandoresearch/flair)
1. [Character-Level Language Modeling with Deeper Self-Attention, Rami et al. 2018](https://arxiv.org/pdf/1808.04444.pdf)

### subword-level representations

1. [*FastText*, Enriching Word Vectors with Subword Information, Bojanowski et al. 2016][Bojanowski et al. 2016]
1. [Neural Machine Translation of Rare Words with Subword Units, Sennrich et al. 2015][Sennrich et al. 2015] [also see [this](https://arxiv.org/pdf/1609.08144.pdf) and [this](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)]

### additional objectives

:bulb: de-biasing, robust to spelling errors, etc.

1. [Robsut Wrod Reocginiton via semi-Character Recurrent Neural Network, Sakaguchi et al. 2016](https://arxiv.org/abs/1608.02214)
1. [Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings, Bolukbasi et al. 2016](https://arxiv.org/abs/1607.06520)
1. [Learning Gender-Neutral Word Embeddings, Zhao et al. 2018][Zhao et al. 2018]
1. [Combating Adversarial Misspellings with Robust Word Recognition, Danish et al. 2019](https://arxiv.org/abs/1905.11268)
1. [Misspelling Oblivious Word Embeddings, Edizel et al. 2019](https://arxiv.org/abs/1905.09755) [[facebook AI]](https://ai.facebook.com/blog/-a-new-model-for-word-embeddings-that-are-resilient-to-misspellings-/)

### sentence representations

1. [Skip-Thought Vectors, Kiros et al. 2015][Kiros et al. 2015]
1. [A Structured Self-attentive Sentence Embedding, Lin et al. 2017][Lin et al. 2017]
1. [*InferSent*, Supervised Learning of Universal Sentence Representations from Natural Language Inference Data, Conneau eta al. 2017][Conneau eta al. 2017]
1. [Hierarchical Attention Networks for Document Classification, Yang et al. 2016](http://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf)
1. [DisSent: Sentence Representation Learning from Explicit Discourse Relations, Nie et al. 2017](https://arxiv.org/abs/1710.04334)
1. [*USE*, Universal Sentence Encoder, Cer et al. 2018][Cer et al. 2018] [[also see Multilingual USE]][Yinfei et al. 2019]

### Multi-lingual word embeddings

1. [[fasttext embeddings]](https://fasttext.cc/docs/en/aligned-vectors.html)
1. [Polyglot: Distributed Word Representations for Multilingual NLP, Rami et al. 2013](https://www.aclweb.org/anthology/W13-3520.pdf)
1. [Density Matching for Bilingual Word Embedding, Zhou et al. 2015](https://www.aclweb.org/anthology/N19-1161.pdf)
1. [Word Translation Without Parallel Data, Conneua et al. 2017](https://arxiv.org/abs/1710.04087) [[repo]](https://github.com/facebookresearch/MUSE)
1. [Loss in Translation: Learning Bilingual Word Mapping with a Retrieval Criterion, Joulin et al. 2018](https://arxiv.org/abs/1804.07745)
1. [Unsupervised Multilingual Word Embeddings, Chen & Cardie 2018](https://arxiv.org/abs/1808.08933) [[repo]](https://github.com/ccsasuke/umwe)

### [Go Back To Top](#Contents)

# Evaluation

:bulb: NLU and XLU

1. [GLUECoS: An Evaluation Benchmark for Code-Switched NLP, Khanuja et al. 2020](https://www.aclweb.org/anthology/2020.acl-main.329/)
1. [XGLUE: A New Benchmark Dataset for Cross-lingual Pre-training, Understanding and Generation, Liang et al. 2020](https://arxiv.org/abs/2004.01401)
1. [XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization, Hu et al. 2020](https://arxiv.org/abs/2003.11080)
1. [SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems, Wang et al. 2019](https://arxiv.org/abs/1905.00537)
1. [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding, Wang et al. 2018][Wang et al. 2018] [[Site]](https://gluebenchmark.com/leaderboard)
1. [XNLI: Evaluating Cross-lingual Sentence Representations, Conneau eta al. 2018c][Conneau eta al. 2018c]
1. [SentEval: An Evaluation Toolkit for Universal Sentence Representations, Conneau et al. 2018a][Conneau et al. 2018a] [[Site]](https://github.com/facebookresearch/SentEval)
1. [CLUE: Language Understanding Evaluation benchmark for Chinese (CLUE)](https://github.com/CLUEbenchmark/CLUE)

### [Go Back To Top](#Contents)

# Interpretability and Ethics

:bulb: inductive bias, distillation and pruning, adversarial attacks, fairness and bias  
:bulb: distillation (can thought of a MAP estimate with prior rather than MLE objective)  
:arrow_upper_right: see some papers on bias in [Word and Sentence Embeddings](#Word-and-Sentence-Embeddings) section

### inductive bias and generalization

1. [Dissecting Contextual Word Embeddings: Architecture and Representation, Peters et al. 2018b](https://www.aclweb.org/anthology/D18-1179)
1. [What you can cram into a single $&!#\* vector: Probing sentence embeddings for linguistic properties, Conneau et al 2018b][Conneau et al 2018b]
1. [Troubling Trends in Machine Learning Scholarship, Lipton and Steinhardt 2018](https://arxiv.org/pdf/1807.03341.pdf)
1. [How Much Reading Does Reading Comprehension Require? A Critical Investigation of Popular Benchmarks, Kaushik and Lipton 2018](https://arxiv.org/abs/1808.04926)
1. [Are Sixteen Heads Really Better than One?, Paul et al. 2019](https://arxiv.org/abs/1905.10650) [[Blogpost]](https://blog.ml.cmu.edu/2020/03/20/are-sixteen-heads-really-better-than-one/?/)
1. [No Training Required: Exploring Random Encoders for Sentence Classification, Wieting et al. 2019][Wieting et al. 2019]
1. [BERT Rediscovers the Classical NLP Pipeline, Tenney et al. 2019](https://arxiv.org/abs/1905.05950)
1. [Compositional Questions Do Not Necessitate Multi-hop Reasoning, Min et al. 2019](https://arxiv.org/abs/1906.02900)
1. [Probing Neural Network Comprehension of Natural Language Arguments, Niven & Kao 2019](https://arxiv.org/pdf/1907.07355.pdf)
   and [[this]](https://medium.com/syncedreview/has-bert-been-cheating-researchers-say-it-exploits-spurious-statistical-cues-b256760ded57) related article
1. [The Bottom-up Evolution of Representations in the Transformer: A Study with Machine Translation and Language Modeling Objectives, Voita et al. 2019](https://arxiv.org/abs/1909.01380)
1. [Rethinking Generalization of Neural Models: A Named Entity Recognition Case Study, Fu et al. 2019](https://arxiv.org/abs/2001.03844)

### interpreting attention

1. [Attention is not Explanation, Jain and Wallace 2019](https://arxiv.org/pdf/1902.10186.pdf)
1. [Is Attention Interpretable?, Serrano and Smith 2019](https://arxiv.org/abs/1906.03731)
1. [Attention is not not Explanation, Wiegreffe and Pinter 2019](https://arxiv.org/pdf/1908.04626.pdf)
1. [Learning to Deceive with Attention-Based Explanations, Pruthi et al. 2020](https://arxiv.org/abs/1909.07913)

### adversarial attacks

1. [Combating Adversarial Misspellings with Robust Word Recognition, Danish et al. 2019](https://arxiv.org/abs/1905.11268)
1. [Universal Adversarial Triggers for Attacking and Analyzing NLP, Wallace et al. 2019](https://arxiv.org/abs/1908.07125)
1. [Weight Poisoning Attacks on Pre-trained Models, Kurita et al. 2020](https://arxiv.org/abs/2004.06660)

### model distillation and pruning

1. [Understanding Knowledge Distillation in Non-autoregressive Machine Translation, Zhou et al. 2019](https://arxiv.org/abs/1911.02727)
1. [Distilling Task-Specific Knowledge from BERT into Simple Neural Networks, Tang et al. 2019][Tang et al. 2019]. Also a related work from
   HuggingFace [here](https://medium.com/huggingface/distilbert-8cf3380435b5), and work on quantization compression by RASA [here](https://blog.rasa.com/compressing-bert-for-faster-prediction-2/)
1. [Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes](https://arxiv.org/pdf/1904.00962.pdf) [[also see this article]](https://medium.com/syncedreview/new-google-brain-optimizer-reduces-bert-pre-training-time-from-days-to-minutes-b454e54eda1d)
1. [*RoBERTa*, A Robustly Optimized BERT Pretraining Approach, Liu et al. 2019](https://arxiv.org/abs/1907.11692)
1. [Patient Knowledge Distillation for BERT Model Compression, Sun et al. 2019](https://arxiv.org/abs/1908.09355)
1. [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations, Lan et al. 2019](https://arxiv.org/abs/1909.11942)

### fairness and bias in models

1. [GROVER, Defending Against Neural Fake News, Zellers et al. 2019](https://arxiv.org/abs/1905.12616) [[blogpost]](https://grover.allenai.org/)

### [Go Back To Top](#Contents)

# Contextual Representations and Transfer Learning

### Language modeling

:bulb: Similar works are also compiled here: [Pre-trained Language Model Papers](https://github.com/thunlp/PLMpapers)  
:bulb: Typically, these *pre-training* methods involve an self-supervised (also called semi-supervised/unsupervised in some works) learning followed by a supervised learning. This is unlike CV domain
where *pre-training* is mainly supervised learning.

1. <https://web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture13-contextual-representations.pdf>
1. [Semi-supervised Sequence Learning, Dai et al. 2015][Dai et al. 2015]
1. [Unsupervised Pretraining for Sequence to Sequence Learning, Ramachandran et al. 2016][Ramachandran et al. 2016]
1. [context2vec: Learning Generic Context Embedding with Bidirectional LSTM, Melamud et al. 2016](https://www.aclweb.org/anthology/K16-1006.pdf)
1. [*InferSent*, Supervised Learning of Universal Sentence Representations from Natural Language Inference Data, Conneau eta al. 2017][Conneau eta al. 2017]
1. [*ULM-FiT*, Universal Language Model Fine-tuning for Text Classification, Howard and Ruder 2018][Howard and Ruder 2018]
1. [*ELMo*, Deep contextualized word representations, Peters et al. 2018][Peters et al. 2018] \[also see previus works- [TagLM](https://arxiv.org/abs/1705.00108)
   and [CoVe](https://arxiv.org/abs/1708.00107) \]
1. [*GPT-1 aka OpenAI Transformer*, Improving Language Understanding by Generative Pre-Training, Radford et al. 2018][Radford et al. 2018]
1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Devlin et al. 2018][Devlin et al. 2018] [[SLIDES]](https://nlp.stanford.edu/seminar/details/jdevlin.pdf) [[also see Illustrated BERT]][Illustrated BERT]
1. [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context, Zihang et al. 2019][Zihang et al. 2019]
1. [*GPT-2*, Language Models are Unsupervised Multitask Learners, Radford et al. 2019][Radford et al. 2019] [also see [Illustrated GPT-2][Illustrated GPT-2]]
1. [ERNIE: Enhanced Language Representation with Informative Entities, Zhang et al. 2019](https://arxiv.org/abs/1905.07129)
1. [XLNet: Generalized Autoregressive Pretraining for Language Understanding, Yang et al. 2019](https://arxiv.org/abs/1906.08237)
1. [RoBERTa: A Robustly Optimized BERT Pretraining Approach, Liu et al. 2019](https://arxiv.org/abs/1907.11692)
1. [ERNIE 2.0: A Continual Pre-training Framework for Language Understanding, Sun et al. 2019](https://arxiv.org/abs/1907.12412)
1. [CTRL: A Conditional Transformer Language Model for Controllable Generation, Keskar et al. 2019](https://arxiv.org/abs/1909.05858)
1. [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations, Lan et al. 2019](https://arxiv.org/abs/1909.11942)
1. [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators, Clark et al. 2019](https://openreview.net/pdf?id=r1xMH1BtvB) [[Google Blog]](https://ai.googleblog.com/2020/03/more-efficient-nlp-model-pre-training.html?m=1)

### + supervised objectives

:bulb: Some people went ahead and thought "how about using supervised (+- self-unsupervised) tasks for pretraining?!"

1. [*InferSent*, Supervised Learning of Universal Sentence Representations from Natural Language Inference Data, Conneau eta al. 2017][Conneau eta al. 2017]
1. [*USE*, Universal Sentence Encoder, Cer et al. 2018][Cer et al. 2018] [[also see Multilingual USE]][Yinfei et al. 2019]
1. [Sentence Encoders on STILTs: Supplementary Training on Intermediate Labeled-data Tasks, Phang et al. 2018][Phang et al. 2018]

### [Go Back To Top](#Contents)

# BERT & Transformers

:bulb: see [Interpretability and Ethics](#Interpretability-and-Ethics) section for more papers

### BERTology

:arrow_forward:  [Bert related papers compilation](https://github.com/tomohideshibata/BERT-related-papers)

1. [E-BERT: Efficient-Yet-Effective Entity Embeddings for BERT, Poerner et al. 2020](https://arxiv.org/abs/1911.03681)
1. [A Primer in BERTology: What we know about how BERT works, Rogers et al. 2020](https://arxiv.org/abs/2002.12327)
1. [Comparing BERT against traditional machine learning text classification, Carvajal et al. 2020](https://arxiv.org/abs/2005.13012)
1. [Revisiting Few-sample BERT Fine-tuning, Zhang et al. 2020](https://arxiv.org/pdf/2006.05987.pdf)

### Transformers

1. [The Evolved Transformer, So et al. 2019](https://arxiv.org/abs/1901.11117v2)
1. [R-Transformer: Recurrent Neural Network Enhanced Transformer, Wang et al. 2019](https://arxiv.org/abs/1907.05572)
1. [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer, Raffel et al. 2019](https://arxiv.org/abs/1910.10683)
1. [The Bottom-up Evolution of Representations in the Transformer: A Study with Machine Translation and Language Modeling Objectives, Voita et al. 2019](https://arxiv.org/abs/1909.01380)
1. [Reformer: The Efficient Transformer, Kitaev et al. 2020](https://arxiv.org/abs/2001.04451)

# Active Learning

:bulb: dataset distillation :p

1. [Deep Active Learning for Named Entity Recognition, shen et al. 2017](https://arxiv.org/abs/1707.05928)
1. [Learning how to Active Learn: A Deep Reinforcement Learning Approach, Fang et al. 2017](https://arxiv.org/pdf/1708.02383.pdf)
1. [An Ensemble Deep Active Learning Method for Intent Classification, Zhang et al. 2019](https://dl.acm.org/doi/pdf/10.1145/3374587.3374611)

### [Go Back To Top](#Contents)

# Multi-task learning

1. [*decaNLP*, The Natural Language Decathlon: Multitask Learning as Question Answering, McCann et al. 2018][McCann et al. 2018]
1. [*HMTL*, A Hierarchical Multi-task Approach for Learning Embeddings from Semantic Tasks, Victor et al. 2018][Victor et al. 2018]
1. [*GenSen*, Learning General Purpose Distributed Sentence Representations via Large Scale Multi-task Learning, Subramanian et al. 2018][Subramanian et al. 2018]
1. [Can You Tell Me How to Get Past Sesame Street? Sentence-Level Pretraining Beyond Language Modeling, Wang et al. 2019](https://arxiv.org/abs/1812.10860)
1. [*GPT-2*, Language Models are Unsupervised Multitask Learners, Radford et al. 2019][Radford et al. 2019] [also see [Illustrated GPT-2][Illustrated GPT-2]]
1. [Unified Language Model Pre-training for Natural Language Understanding and Generation, Dong et al. 2019](https://arxiv.org/abs/1905.03197)
1. [MASS: Masked Sequence to Sequence Pre-training for Language Generation, Song et al. 2019](https://arxiv.org/abs/1905.02450)
1. [ERNIE 2.0: A Continual Pre-training Framework for Language Understanding, Sun et al. 2019](https://arxiv.org/abs/1907.12412)
1. [T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer, Raffel et al. 2019](https://arxiv.org/pdf/1910.10683v2.pdf) [[code]](https://github.com/google-research/text-to-text-transfer-transformer)

### [Go Back To Top](#Contents)

# Generation

### text generation

1. [Incorporating Copying Mechanism in Sequence-to-Sequence Learning, Jiatao Gu et al. 2016](https://arxiv.org/abs/1603.06393)
1. [Quantifying Exposure Bias for Neural Language Generation, He et al. 2019](https://arxiv.org/abs/1905.10617)
1. [CTRL: A Conditional Transformer Language Model for Controllable Generation, Keskar et al. 2019](https://arxiv.org/abs/1909.05858)
1. [Plug and Play Language Models: A Simple Approach to Controlled Text Generation, Dathathri et al. 2019](https://arxiv.org/abs/1912.02164)

### dialogue sytems

1. [Zero-shot User Intent Detection via Capsule Neural Networks, Xia et al. 2018][Xia et al. 2018]
1. [Investigating Capsule Networks with Dynamic Routing for Text Classification, Zhao et al. 2018][Zhao et al. 2018]
1. [BERT for Joint Intent Classification and Slot Filling, Chen et al. 2019][Chen et al. 2019]
1. [Few-Shot Generalization Across Dialogue Tasks, Vlasov et al. 2019][Vlasov et al. 2019] [RASA Research]
1. [Towards Open Intent Discovery for Conversational Text, Vedula et al. 2019][Vedula et al. 2019]
1. [What makes a good conversation? How controllable attributes affect human judgments](https://www.aclweb.org/anthology/N19-1170) [[also see this article]](http://www.abigailsee.com/2019/08/13/what-makes-a-good-conversation.html)

### machine translation

1. [Sequence to Sequence Learning with Neural Networks, Sutskever et al. 2014](https://arxiv.org/abs/1409.3215)
1. [Addressing the Rare Word Problem in Neural Machine Translation, Luong et al. 2014](https://arxiv.org/abs/1410.8206)
1. [Neural Machine Translation of Rare Words with Subword Units, Sennrich et al. 2015][Sennrich et al. 2015]
1. [*Transformer*, Attention Is All You Need, Vaswami et al. 2017][Vaswami et al. 2017]
1. [Understanding Back-Translation at Scale, Edunov et al. 2018](https://arxiv.org/pdf/1808.09381.pdf)
1. [Achieving Human Parity on Automatic Chinese to English News Translation, Microsoft Research 2018](https://arxiv.org/abs/1803.05567) [[Bites]](https://github.com/kweonwooj/papers/issues/98) [also see [this](https://arxiv.org/pdf/1707.00415.pdf) and [this](https://papers.nips.cc/paper/6775-deliberation-networks-sequence-generation-beyond-one-pass-decoding.pdf)]

# Knowledge Graphs

:bulb: LMs realized as diverse learners; learning more than what you thought!!

1. [Language Models as Knowledge Bases?, Petroni et al. 2019](https://arxiv.org/abs/1909.01066)

### [Go Back To Top](#Contents)

# Multi-lingual and cross-lingual learning

### multilingual

1. [Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond, Artetxe et al. 2018][Artetxe et al. 2018]
1. [How multilingual is Multilingual BERT?, Pires et al.2019](https://arxiv.org/abs/1906.01502)
1. [Multilingual Universal Sentence Encoder (USE) for Semantic Retrieval, Yinfei Yang et al. 2019](https://arxiv.org/pdf/1907.04307.pdf)
1. [How Language-Neutral is Multilingual BERT?, Libovicky et al. 2020](https://arxiv.org/abs/1911.03310)
1. [Universal Phone Recognition with a Multilingual Allophone System, Le te al. 2020](https://arxiv.org/abs/2002.11800)

### Cross-Lingual

1. <http://ruder.io/cross-lingual-embeddings/index.html>
1. [*XLM*, Cross-lingual Language Model Pretraining, Guillaume and Conneau et al. 2019](https://arxiv.org/abs/1901.07291)
1. [Cross-Lingual Ability of Multilingual BERT: An Empirical Study, karthikeyan et al. 2019](https://arxiv.org/abs/1912.07840)
1. [XQA: A Cross-lingual Open-domain Question Answering Dataset, Liu et al. 2019](https://www.aclweb.org/anthology/P19-1227.pdf)

### [Go Back To Top](#Contents)

# Multi-modal learning

1. [Representation Learning with Contrastive Predictive Coding, Oord et al. 2018](https://arxiv.org/abs/1807.03748)
1. [M-BERT: Injecting Multimodal Information in the BERT Structure, Rahman et al. 2019](https://arxiv.org/abs/1908.05787)
1. [LXMERT: Learning Cross-Modality Encoder Representations from Transformers, Tan and Bansal 2019](https://arxiv.org/abs/1908.07490)
1. [BERT Can See Out of the Box: On the Cross-modal Transferability of Text Representations, Scialom et al. 2020](https://arxiv.org/abs/2002.10832)

# Question Answering

1. [A Deep Neural Network Framework for English Hindi Question Answering](https://www.cse.iitb.ac.in/~pb/papers/tallip-qa.pdf)
1. [DrQA, Reading Wikipedia to Answer Open-Domain Questions, Chen et al. 2017](https://arxiv.org/abs/1704.00051)
1. [GoldEn Retriever, Answering Complex Open-domain Questions Through Iterative Query Generation, Qi et al 2019](https://arxiv.org/pdf/1910.07000.pdf)
1. [BREAK It Down: A Question Understanding Benchmark, Wolfson et al. 2020](https://arxiv.org/pdf/2001.11770v1.pdf)
1. [XQA: A Cross-lingual Open-domain Question Answering Dataset, Liu et al. 2019](https://www.aclweb.org/anthology/P19-1227.pdf)

### [Go Back To Top](#Contents)

# Notes

## Quick Bites

1. ```Byte Pair Encoding (BPE)``` is a data compression technique that iteratively replaces the most frequent pair of symbols (originally bytes) in a given dataset with a single unused symbol. In each
   iteration, the algorithm finds the most frequent (adjacent) pair of symbols, each can be constructed of a single character or a sequence of characters, and merged them to create a new symbol. All
   occurences of the selected pair are then replaced with the new symbol before the next iteration. Eventually, frequent sequence of characters, up to a whole word, are replaced with a single symbol,
   until the algorithm reaches the defined number of iterations (50k can be an example figure). During inference, if a word isn’t part of the BPE’s pre-built dictionary, it will be split into subwords
   that are. Code of BPE can be found [here](https://gist.github.com/ranihorev/6ba9a88c9e7401b603cd483dd767e783).
   See [Overall Idea blog-post](https://medium.com/@makcedward/how-subword-helps-on-your-nlp-model-83dd1b836f46), [BPE specific blog-post](https://leimao.github.io/blog/Byte-Pair-Encoding/)
   and [BPE Code](https://github.com/google/sentencepiece) for more details.

```
import re
words0 = [" ".join([char for char in word]+["</w>"]) for word in "in the rain in Ukraine".split()]+["i n"]+["<w> i n"]
print(words0)

eword1 = re.escape('i n')
p1 = re.compile(r'(?<!\S)' + eword1 + r'(?!\S)')
words1 = [p1.sub('in',word) for word in words0]
print(words1)


eword2 = re.escape('in </w>')
p2 = re.compile(r'(?<!\S)' + eword2 + r'(?!\S)')
words2 = [p2.sub('in</w>',word) for word in words1]
print(words2)


eword3 = re.escape('a in</w>')
p3 = re.compile(r'(?<!\S)' + eword3 + r'(?!\S)')
words3 = [p3.sub('ain</w>',word) for word in words2]
print(words3)
```

2. ```re``` library

```
re.search(), re.findall(), re.split(), re.sub()
re.escape(), re.compile()
```

[101](https://www.w3schools.com/python/python_regex.asp), [Positive and Negative Lookahead/Lookbehind](https://www.regular-expressions.info/lookaround.html)

3. Models can be trained on SNLI in two different ways: (i) sentence encoding-based models that explicitly separate the encoding of the individual sentences and (ii) joint methods that allow to use
   encoding of both sentences (to use cross-features or attention from one sentence to the other).

## Food For Thought

1. How good do ranking algorithms, the ones with pointwise/pairwise/listwise learning paradigms, perform when the no. of test classes at the infernece time grow massively? KG Reasoning using
   Translational/Bilinear/DL techniques is one important area under consideration.
1. While the chosen neural achitecture is important, the techniques used for training the problem objective e.g.[*Word2Vec*][Mikolov et al. 2013b] or the techniques used while doing loss optimization
   e.g.[*OpenAI Transformer*][Radford et al. 2018] play a significant role in both fast as well as a good convergence.
1. Commonality between Language Modelling, Machine Translation and Word2vec: All of them have a huge vocabulary size at the output and there is a need to alleviate computing of the huge sized softmax
   layer! See [Ruder's page](http://ruder.io/word-embeddings-softmax/index.html) for a quick-read.

# Bookmarks

## codes, videos and MOOCs

- nlp code examples: [nn4nlp-code](https://github.com/neubig/nn4nlp-code) and [nlp-tutorial](https://github.com/graykode/nlp-tutorial)
- [lazynlp for data crawling](https://github.com/chiphuyen/lazynlp)
- [video-list](https://github.com/CShorten/HenryAILabs-VideoList)
- [fast.ai](https://www.fast.ai/2019/07/08/fastai-nlp/), [d2l.ai](https://d2l.ai/index.html), [nn4nlp-graham](http://www.phontron.com/class/nn4nlp2020/)

## each link is either a series of blogs from an individual/organization or a conference related link or a MOOC

- [nlpprogress](http://nlpprogress.com/), [paperswithcode](https://paperswithcode.com/area/natural-language-processing), [dair-ai](https://github.com/dair-ai/nlp_newsletter)
  , [groundai](https://www.groundai.com/?=&tag=LANGUAGE), [lyrn.ai](https://www.lyrn.ai/category/nlp/)
- [BERT-related-papers](https://github.com/tomohideshibata/BERT-related-papers), [awesome-qa](https://github.com/seriousran/awesome-qa),
  [awesome-NLP](https://github.com/keon/awesome-nlp), [awesome-sentence-embedding](https://github.com/Separius/awesome-sentence-embedding)
- [Ruder's Blog-posts](http://ruder.io); [this](https://ruder.io/research-highlights-2019/) and [this](http://ruder.io/state-of-transfer-learning-in-nlp/) taking on latest trends  
  [Jay Alammar](http://jalammar.github.io/),  
  [LiLian (at OpenAI)](https://lilianweng.github.io/lil-log/),  
  [Sebastian Rraschka (at UW-Madison)](https://sebastianraschka.com/blog/index.html),  
  [Victor Sanh (at Huggingface)](https://medium.com/@victorsanh),  
  [Keita Kurita|ML & NLP Explained](http://mlexplained.com/category/nlp/),  
  [distill.pub](https://distill.pub/),  
  [Chris McCormick](https://mccormickml.com/archive/),  
  [Kavita Ganesan](https://kavita-ganesan.com/kavitas-tutorials/#.Xhk3_0dKjDc),
- [blog.feedspot.com](https://drive.google.com/file/d/15XD2c2PypVZTveezFuaJpPEJWP8Vi3ak/view?usp=sharing)

## selected blog-posts

- [Guillaume's blog for seq tagging](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html)
- [ULMFit](https://yashuseth.blog/2018/06/17/understanding-universal-language-model-fine-tuning-ulmfit/)
- [Lilian at openAI on Attention](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)
- [The Natural Language Decathlon](https://blog.einstein.ai/the-natural-language-decathlon/)
- [OpenAI GPT-1](https://openai.com/blog/language-unsupervised/) and [OpenAi GPT-2](https://openai.com/blog/better-language-models/)

## miscellaneous

- [SPACY IRL 2019 Talks](https://www.youtube.com/playlist?list=PLBmcuObd5An4UC6jvK_-eSl6jCvP1gwXc)
- [text processing](https://www.kaggle.com/sudalairajkumar/getting-started-with-text-preprocessing)
- [Regularization Techniques for NLP](http://mlexplained.com/2018/03/02/regularization-techniques-for-natural-language-processing-with-code-examples/)
- [Chat Smarter with Allo](https://ai.googleblog.com/2016/05/chat-smarter-with-allo.html)
- [word cloud](https://github.com/amueller/word_cloud)

### [Go Back To Top](#Contents)

[Mikolov et al. 2013a]: https://arxiv.org/abs/1301.3781

[Mikolov et al. 2013b]: https://arxiv.org/abs/1310.4546

[Pennington et al. 2014]: https://www.aclweb.org/anthology/D14-1162

[Bojanowski et al. 2016]: https://arxiv.org/abs/1607.04606

[Peters et al. 2018]: https://arxiv.org/abs/1802.05365

[Akbik et al. 2018]: http://alanakbik.github.io/papers/coling2018.pdf

[Lin et al. 2017]: https://arxiv.org/abs/1703.03130

[Vaswami et al. 2017]: https://arxiv.org/pdf/1706.03762.pdf

[Radford et al. 2018]: https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf

[Howard and Ruder 2018]: https://arxiv.org/abs/1801.06146

[Devlin et al. 2018]:https://arxiv.org/abs/1810.04805

[Radford et al. 2019]: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

[Xia et al. 2018]: https://arxiv.org/abs/1809.00385

[Cer et al. 2018]: https://arxiv.org/pdf/1803.11175.pdf

[Subramanian et al. 2018]: https://arxiv.org/abs/1804.00079

[McCann et al. 2018]: https://arxiv.org/abs/1806.08730

[Zihang et al. 2019]: https://arxiv.org/abs/1901.02860v2

[Victor et al. 2018]: https://arxiv.org/abs/1811.06031

[Dai et al. 2015]: https://arxiv.org/abs/1511.01432

[Chen et al. 2019]: https://arxiv.org/abs/1902.10909

[Zhao et al. 2018]: https://arxiv.org/abs/1804.00538

[Conneau eta al. 2017]: https://arxiv.org/abs/1705.02364

[Wieting et al. 2019]: https://arxiv.org/abs/1901.10444

[Beltagy et al. 2019]: https://arxiv.org/abs/1903.10676

[NeelKant et al. 2018]: https://arxiv.org/abs/1812.01207

[Wang et al. 2018]: https://arxiv.org/abs/1804.07461

[Hassan et al. 2018]: https://arxiv.org/abs/1803.05567

[Bowman et al. 2018]: https://arxiv.org/abs/1812.10860

[Tang et al. 2019]: https://arxiv.org/abs/1903.12136

[Sennrich et al. 2015]: https://arxiv.org/abs/1508.07909

[Artetxe et al. 2018]: https://arxiv.org/abs/1812.10464

[Zhao et al. 2018]: https://arxiv.org/abs/1809.01496

[Lample et al. 2019]: https://arxiv.org/abs/1901.07291

[Conneau eta al. 2018c]: https://arxiv.org/abs/1809.05053

[Kiros et al. 2015]: https://arxiv.org/abs/1506.06726

[Conneau et al. 2018a]: https://arxiv.org/abs/1803.05449

[Phang et al. 2018]: https://arxiv.org/abs/1811.01088

[Conneau et al 2018b]: https://arxiv.org/abs/1805.01070

[Vlasov et al. 2019]: https://arxiv.org/abs/1811.11707

[Zhai et at. 2017]: https://arxiv.org/abs/1701.04027

[Ramachandran et al. 2016]: https://arxiv.org/abs/1611.02683

[Vedula et al. 2019]: https://arxiv.org/abs/1904.08524

[Nie et al. 2017]: https://arxiv.org/abs/1710.04334

[Yinfei et al. 2019]: https://arxiv.org/abs/1907.04307

[Illustrated BERT]: http://jalammar.github.io/illustrated-bert/

[OpenAi GPT-1]: https://openai.com/blog/language-unsupervised/

[OpenAi GPT-2]: https://openai.com/blog/better-language-models/

[The Natural Language Decathlon]: https://blog.einstein.ai/the-natural-language-decathlon/

[ULMFit]: https://yashuseth.blog/2018/06/17/understanding-universal-language-model-fine-tuning-ulmfit/

[Important AI papers 2018 TOPBOTS]: https://www.topbots.com/most-important-ai-research-papers-2018/

[Chat Smarter with Allo]: https://ai.googleblog.com/2016/05/chat-smarter-with-allo.html

[Regularization Techniques for NLP]: http://mlexplained.com/2018/03/02/regularization-techniques-for-natural-language-processing-with-code-examples/

[Illustrated GPT-2]: https://jalammar.github.io/illustrated-gpt2/
