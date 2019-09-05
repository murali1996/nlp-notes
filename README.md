
# Contents
Hi there!<br>
I was looking for some way to categorize the massively evolving research studies surrounding NLP (and in the field of DL and CV) and here's my two cents.<br>
Obviously, the list isn't exhaustive and I'll keep adding new papers that help us in better interpreting and representing textual data. For now, let's walk through this list to find some interesting reads...

## Related to word-level representations
1. [*Word2Vec*, Efficient Estimation of Word Representations in Vector Space, Mikolov et al. 2013a][Mikolov et al. 2013a]
1. [*Word2Vec*, Distributed Representations of Words and Phrases and their Compositionality, Mikolov et al. 2013b][Mikolov et al. 2013b]
1. [GloVe: Global Vectors for Word Representation, Pennington et al. 2014][Pennington et al. 2014]
1. [Learning Gender-Neutral Word Embeddings, Zhao et al. 2018][Zhao et al. 2018]

## Related to character-level representations
1. [*FLAIR*, Contextual String Embeddings for Sequence Labeling, Akbik et al. 2018][Akbik et al. 2018] [[CODE]](https://github.com/zalandoresearch/flair)
1. [Character-Level Language Modeling with Deeper Self-Attention, Rami et al. 2018](https://arxiv.org/pdf/1808.04444.pdf)

## Related to subword-level representations
1. [*FastText*, Enriching Word Vectors with Subword Information, Bojanowski et al. 2016][Bojanowski et al. 2016]
1. [Combating Adversarial Misspellings with Robust Word Recognition, Pruthi et al. 2019](https://arxiv.org/abs/1905.11268)
1. [Misspelling Oblivious Word Embeddings, Edizel et al. 2019](https://arxiv.org/abs/1905.09755) [[facebook AI]](https://ai.facebook.com/blog/-a-new-model-for-word-embeddings-that-are-resilient-to-misspellings-/)

## Related to neural models for Sentence Encoding
1. [Skip-Thought Vectors, Kiros et al. 2015][Kiros et al. 2015]
1. [A Structured Self-attentive Sentence Embedding, Lin et al. 2017][Lin et al. 2017]
1. [Hierarchical Attention Networks for Document Classification, Yang et al. 2016](http://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf)
1. [DisSent: Sentence Representation Learning from Explicit Discourse Relations, Nie et al. 2017](https://arxiv.org/abs/1710.04334)

[Go Back To Contents](#Contents)

## Evaluation for representation learning, inductive bias analysis, knowledge distillation and pruning
1. [SentEval: An Evaluation Toolkit for Universal Sentence Representations, Conneau et al. 2018a][Conneau et al. 2018a] [[Site]](https://github.com/facebookresearch/SentEval)
1. [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding, Wang et al. 2018][Wang et al. 2018] [[Site]](https://gluebenchmark.com/leaderboard)
1. [What you can cram into a single $&!#\* vector: Probing sentence embeddings for linguistic properties, Conneau et al 2018b][Conneau et al 2018b]
1. [No Training Required: Exploring Random Encoders for Sentence Classification, Wieting et al. 2019][Wieting et al. 2019]
1. [*DistillBERT*, Distilling Task-Specific Knowledge from BERT into Simple Neural Networks, Tang et al. 2019][Tang et al. 2019] [[also see this article]](http://nlp.town/blog/distilling-bert/)
1. [Are Sixteen Heads Really Better than One?, Paul et al. 2019](https://arxiv.org/abs/1905.10650)
1. [Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes](https://arxiv.org/pdf/1904.00962.pdf) [[also see this article]](https://medium.com/syncedreview/new-google-brain-optimizer-reduces-bert-pre-training-time-from-days-to-minutes-b454e54eda1d)
1. [*RoBERTa*, A Robustly Optimized BERT Pretraining Approach, Liu et al. 2019](https://arxiv.org/abs/1907.11692)

## Related to contextual representations and transfer learning
:bulb: Typically, these *pre-training* methods involve an unsupervised (also called semi-supervised in some works) learning followed by a supervised learning. This is unlike CV domain where *pre-training* is mainly supervised learning.
1. [Semi-supervised Sequence Learning, Dai et al. 2015][Dai et al. 2015]
1. [Unsupervised Pretraining for Sequence to Sequence Learning, Ramachandran et al. 2016][Ramachandran et al. 2016]
1. [*ULM-FiT*, Universal Language Model Fine-tuning for Text Classification, Howard and Ruder 2018][Howard and Ruder 2018]
1. [*ELMo*, Deep contextualized word representations, Peters et al. 2018][Peters et al. 2018] 
1. [*GPT-1 aka OpenAI Transformer*, Improving Language Understanding by Generative Pre-Training, Radford et al. 2018][Radford et al. 2018]
1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Devlin et al. 2018][Devlin et al. 2018] [[SLIDES]](https://nlp.stanford.edu/seminar/details/jdevlin.pdf) [[also see Illustrated BERT]][Illustrated BERT]
1. [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context, Zihang et al. 2019][Zihang et al. 2019]
1. [*XLM*, Cross-lingual Language Model Pretraining, Guillaume and Conneau](https://arxiv.org/abs/1901.07291)
1. [XLNet: Generalized Autoregressive Pretraining for Language Understanding, Yang et al. 2019](https://arxiv.org/abs/1906.08237)

:bulb: Some people went ahead and thought "how about using supervised(+unsupervised) tasks for pretraining?!"
1. [*InferSent*, Supervised Learning of Universal Sentence Representations from Natural Language Inference Data, Conneau eta al. 2017][Conneau eta al. 2017]
1. [*USE*, Universal Sentence Encoder, Cer et al. 2018][Cer et al. 2018] [[also see Multilingual USE]][Yinfei et al. 2019]
1. [Sentence Encoders on STILTs: Supplementary Training on Intermediate Labeled-data Tasks, Phang et al. 2018][Phang et al. 2018]

:bulb: Some people went further ahead and thought "how about using multi-task learning?!"
1. [*decaNLP*, The Natural Language Decathlon: Multitask Learning as Question Answering, McCann et al. 2018][McCann et al. 2018]
1. [*HMTL*, A Hierarchical Multi-task Approach for Learning Embeddings from Semantic Tasks, Victor et al. 2018][Victor et al. 2018]
1. [*GenSen*, Learning General Purpose Distributed Sentence Representations via Large Scale Multi-task Learning, Subramanian et al. 2018][Subramanian et al. 2018]
1. [*GPT-2*, Language Models are Unsupervised Multitask Learners, Radford et al. 2019][Radford et al. 2019] [[also see Illustrated GPT-2]][Illustrated GPT-2]
1. [Can You Tell Me How to Get Past Sesame Street? Sentence-Level Pretraining Beyond Language Modeling, Wang et al. 2019](https://arxiv.org/abs/1812.10860)
1. [ERNIE 2.0: A Continual Pre-training Framework for Language Understanding, Sun et al. 2019](https://arxiv.org/abs/1907.12412)

## Huh...the derivatives, I reckon!
1. [Practical Text Classification With Large Pre-Trained Language Models, NeelKant et al. 2018][NeelKant et al. 2018]
1. [SciBERT: Pretrained Contextualized Embeddings for Scientific Text, Beltagy et al. 2019][Beltagy et al. 2019]

[Go Back To Contents](#Contents)

## NMT
1. [Understanding Back-Translation at Scale, Edunov et al. 2018](https://arxiv.org/pdf/1808.09381.pdf)
1. [Achieving Human Parity on Automatic Chinese to English News Translation, Microsoft Research 2018](https://arxiv.org/abs/1803.05567) [[Bites]](https://github.com/kweonwooj/papers/issues/98)
1. [Dual Supervised Learning, Xia et al. 2017](https://arxiv.org/pdf/1707.00415.pdf)
1. [Deliberation Networks: Sequence Generation Beyond One-Pass Decoding, Xia et al. 2017](https://papers.nips.cc/paper/6775-deliberation-networks-sequence-generation-beyond-one-pass-decoding.pdf)

## Natural Language Generation
1. [Unified Language Model Pre-training for Natural Language Understanding and Generation, Dong et al. 2019](https://arxiv.org/abs/1905.03197)
1. [MASS: Masked Sequence to Sequence Pre-training for Language Generation, Song et al. 2019](https://arxiv.org/abs/1905.02450)

## Neural Dialogue
1. [Zero-shot User Intent Detection via Capsule Neural Networks, Xia et al. 2018][Xia et al. 2018]
1. [Investigating Capsule Networks with Dynamic Routing for Text Classification, Zhao et al. 2018][Zhao et al. 2018]
1. [Few-Shot Generalization Across Dialogue Tasks, Vlasov et al. 2019][Vlasov et al. 2019] [RASA Research]
1. [BERT for Joint Intent Classification and Slot Filling, Chen et al. 2019][Chen et al. 2019]
1. [Towards Open Intent Discovery for Conversational Text, Vedula et al. 2019][Vedula et al. 2019]
1. [What makes a good conversation? How controllable attributes affect human judgments](https://www.aclweb.org/anthology/N19-1170) [[also see this article]](http://www.abigailsee.com/2019/08/13/what-makes-a-good-conversation.html)

[Go Back To Contents](#Contents)

## Related to XLU, Cross-lingual & Multi-lingual
1. [XNLI: Evaluating Cross-lingual Sentence Representations, Conneau eta al. 2018c][Conneau eta al. 2018c]
1. [Neural Machine Translation of Rare Words with Subword Units, Sennrich et al. 2015][Sennrich et al. 2015]
1. [*Transformer*, Attention Is All You Need, Vaswami et al. 2017][Vaswami et al. 2017]
1. [Achieving Human Parity on Automatic Chinese to English News Translation, Hassan et al. 2018][Hassan et al. 2018]
1. [Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond, Artetxe et al. 2018][Artetxe et al. 2018]
1. [Cross-lingual Language Model Pretraining, Lample et al. 2019][Lample et al. 2019]
1. [Multilingual Universal Sentence Encoder for Semantic Retrieval, Yang et al. 2019][Yinfei et al. 2019]
1. [*XLM*, Cross-lingual Language Model Pretraining, Guillaume and Conneau](https://arxiv.org/abs/1901.07291)

[Go Back To Contents](#Contents)

# Bookmarks
- [See this article from huggingface for more reading pointers :)](https://medium.com/huggingface/the-best-and-most-current-of-modern-natural-language-processing-5055f409a1d1)
- [NLP Progress Site](http://nlpprogress.com/),
  [NLP | ML Explained](http://mlexplained.com/category/nlp/),
  [Ruder's Blogs](http://ruder.io),
  [Ppaers with code](https://paperswithcode.com/area/natural-language-processing)
  </br>
- [Awesome-Sentence-Embedding](https://github.com/Separius/awesome-sentence-embedding),
  [Awesome-NLP](https://github.com/keon/awesome-nlp),
  [NLP-Tutorial](https://github.com/graykode/nlp-tutorial),
  [lazynlp](https://github.com/chiphuyen/lazynlp),
  [huggingface-transformers](https://github.com/huggingface/pytorch-transformers)
  </br>
- [Illustrated GPT-2][Illustrated GPT-2],
  [Illustrated BERT][Illustrated BERT],
  [OpenAi GPT-1](https://openai.com/blog/language-unsupervised/),
  [OpenAi GPT-2](https://openai.com/blog/better-language-models/),
  [The Natural Language Decathlon](https://blog.einstein.ai/the-natural-language-decathlon/),
  [ULMFit](https://yashuseth.blog/2018/06/17/understanding-universal-language-model-fine-tuning-ulmfit/)
  </br>
- [Regularization Techniques for NLP](http://mlexplained.com/2018/03/02/regularization-techniques-for-natural-language-processing-with-code-examples/),
  [Important AI papers 2018 TOPBOTS](https://www.topbots.com/most-important-ai-research-papers-2018/),
  [Chat Smarter with Allo](https://ai.googleblog.com/2016/05/chat-smarter-with-allo.html)
  </br>

[Go Back To Contents](#Contents)

## Food For Thought
1. How good do ranking algorithms, the ones with pointwise/pairwise/listwise learning paradigms, perform when the no. of test classes at the infernece time grow massively? KG Reasoning using Translational/Bilinear/DL techniques is one important area under consideration.
1. While the chosen neural achitecture is important, the techniques used for training the problem objective e.g.[*Word2Vec*][Mikolov et al. 2013b] or the techniques used while doing loss optimization e.g.[*OpenAI Transformer*][Radford et al. 2018] play a significant role in both fast as well as a good convergence.
1. Commonality between Language Modelling, Machine Translation and Word2vec: All of them have a huge vocabulary size at the output and there is a need to alleviate computing of the huge sized softmax layer! See [Ruder's page](http://ruder.io/word-embeddings-softmax/index.html) for a quick-read.

## Quick Bites
1. Byte Pair Encoding (BPE) is a data compression technique that iteratively replaces the most frequent pair of symbols (originally bytes) in a given dataset with a single unused symbol. In each iteration, the algorithm finds the most frequent (adjacent) pair of symbols, each can be constructed of a single character or a sequence of characters, and merged them to create a new symbol. All occurences of the selected pair are then replaced with the new symbol before the next iteration. Eventually, frequent sequence of characters, up to a whole word, are replaced with a single symbol, until the algorithm reaches the defined number of iterations (50k can be an example figure). During inference, if a word isn’t part of the BPE’s pre-built dictionary, it will be split into subwords that are. An example code of BPE can be found here. https://gist.github.com/ranihorev/6ba9a88c9e7401b603cd483dd767e783
1. Models can be trained on SNLI in two different ways: (i) sentence encoding-based models that explicitly separate the encoding of the individual sentences and (ii) joint methods that allow to use encoding of both sentences (to use cross-features or attention from one sentence to the other).

[Go Back To Contents](#Contents)

## Related to deep nets modeling
1. [An overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/)
1. [Cyclical Learning Rates for Training Neural Networks, Smith et al. 2015](https://arxiv.org/abs/1506.01186)
1. [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates, Smith et al. 2017](https://arxiv.org/abs/1708.07120)
1. [How to Escape Saddle Points Efficiently](https://bair.berkeley.edu/blog/2017/08/31/saddle-efficiency/)
1. [Ongoing Work, *RAdam*, On the variance of adaptive learning rate and beyond](https://arxiv.org/pdf/1908.03265.pdf) [[also see this article]](https://medium.com/@lessw/new-state-of-the-art-ai-optimizer-rectified-adam-radam-5d854730807b)

## Miscellaneous
1. [ResNet 2015](https://arxiv.org/abs/1512.03385)
1. [DenseNet 2017](https://arxiv.org/abs/1608.06993)
1. [Stability Based Filter Pruning for Accelerating Deep CNNs, Pravendra et al. 2018](https://arxiv.org/abs/1811.08321)
1. <https://towardsdatascience.com/latest-computer-vision-trends-from-cvpr-2019-c07806dd570b>
1. [Object-driven Text-to-Image Synthesis via Adversarial Training, Li et al. 2019](https://arxiv.org/abs/1902.10740)


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