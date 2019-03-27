#### This repository contains paper links, code links and my implementations to some of the well researched nlp problems till date.

## Tasks Overview
* [Understanding Word Embeddings](#understanding-word-embeddings)
* [BERT for Intent Classification](#bert-for-intent-classification)

## Interesting Papers
##### Word Embedding
1. [*Word2Vec*, Efficient Estimation of Word Representations in Vector Space, Mikolov et al. 2013a][Mikolov et al. 2013a]
1. [*Word2Vec*, Distributed Representations of Words and Phrases and their Compositionality, Mikolov et al. 2013b][Mikolov et al. 2013b]
1. [*GloVe*, GloVe: Global Vectors for Word Representation, Pennington et al. 2014][Pennington et al. 2014]
1. [*FastText*, Enriching Word Vectors with Subword Information, Bojanowski et al. 2016][Bojanowski et al. 2016]
1. [*ELMo*, Deep contextualized word representations, Peters et al. 2018][Peters et al. 2018] 
1. [*FLAIR*, Contextual String Embeddings for Sequence Labeling, Akbik et al. 2018][Akbik et al. 2018] [[Code]](https://github.com/zalandoresearch/flair)
---
##### Sentence Embedding
1. [A Structured Self-attentive Sentence Embedding, Lin et al. 2017][Lin et al. 2017]
1. [*InferSent*, Supervised Learning of Universal Sentence Representations from Natural Language Inference Data, Alexis et al. 2017][Alexis et al. 2017]
1. [*GenSen*, Learning General Purpose Distributed Sentence Representations via Large Scale Multi-task Learning, Sandeep et al. 2018][Sandeep et al. 2018]
1. [*USE*, Universal Sentence Encoder, Daniel et al. 2018][Daniel et al. 2018]
---
##### Transformer Networks
1. [*Transformer*, Attention Is All You Need, Vaswami et al. 2017][Vaswami et al. 2017]
1. [*Transformer-Decoder*, Generating Wikipedia by Summarizing Long Sequences, Lui et al. 2018][Lui et al. 2018]
1. [*GPT-1 a.k.a OpenAI Transformer*, Improving Language Understanding by Generative Pre-Training, Radford et al. 2018][Radford et al. 2018]
1. [*Transformer-XL*, Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context, Zihang et al. 2019][Zihang et al. 2019]
1. [*GPT-2*, Language Models are Unsupervised Multitask Learners, Radford et al. 2019][Radford et al. 2019]
---
##### General Purpose Modeling a.k.a Pre-trained Models
1. [*ULM-FiT*, Universal Language Model Fine-tuning for Text Classification, Howard and Ruder 2018][Howard and Ruder 2018]
1. [*decaNLP*, The Natural Language Decathlon: Multitask Learning as Question Answering][Bryan et al. 2018]
1. [*BERT*, BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Devlin et al. 2018][Devlin et al. 2018]
1. [*HMTL*, A Hierarchical Multi-task Approach for Learning Embeddings from Semantic Tasks, Victor et al. 2018][Victor et al. 2018]
---
##### Capsule Networks
1. [*Capsule-Nets*, Dynamic Routing Between Capsules, Sabour et al. 2017][Sabour et al. 2017]
1. [Zero-shot User Intent Detection via Capsule Neural Networks, Xia et al. 2018][Xia et al. 2018]


## Downloads
1. [*Word2Vec*](https://github.com/mmihaltz/word2vec-GoogleNews-vectors/)
1. [*Glove*](https://nlp.stanford.edu/projects/glove/)

## Food For Thought
1. While the chosen neural achitecture is important, the techniques used for training the problem objective e.g.[*Word2Vec*][Mikolov et al. 2013b] or the techniques used while doing loss optimization e.g.[*OpenAI Transformer*][Radford et al. 2018] play a significant role in both fast as well as a good convergence.
1. Commonality between Language Modelling, Machine Translation and Word2vec: All of them have a huge vocabulary size at the output and there is a need to alleviate computing of the huge sized softmax layer! See [Ruder's page](http://ruder.io/word-embeddings-softmax/index.html) for a quick-read.


[Mikolov et al. 2013a]: https://arxiv.org/abs/1301.3781
[Mikolov et al. 2013b]: https://arxiv.org/abs/1310.4546
[Pennington et al. 2014]: https://www.aclweb.org/anthology/D14-1162
[Bojanowski et al. 2016]: https://arxiv.org/abs/1607.04606
[Peters et al. 2018]: https://arxiv.org/abs/1802.05365
[Akbik et al. 2018]: http://alanakbik.github.io/papers/coling2018.pdf
[Lin et al. 2017]: https://arxiv.org/abs/1703.03130
[Vaswami et al. 2017]: https://arxiv.org/pdf/1706.03762.pdf
[Lui et al. 2018]: https://arxiv.org/abs/1801.10198
[Radford et al. 2018]: https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
[Howard and Ruder 2018]: https://arxiv.org/abs/1801.06146
[Devlin et al. 2018]:https://arxiv.org/abs/1810.04805
[Radford et al. 2019]: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
[Sabour et al. 2017]: https://arxiv.org/abs/1710.09829
[Xia et al. 2018]: https://arxiv.org/abs/1809.00385
[Daniel et al. 2018]: https://arxiv.org/pdf/1803.11175.pdf
[Alexis et al. 2017]: https://arxiv.org/abs/1705.02364
[Sandeep et al. 2018]: https://arxiv.org/abs/1804.00079
[Bryan et al. 2018]: https://arxiv.org/abs/1806.08730
[Zihang et al. 2019]: https://arxiv.org/abs/1901.02860v2
[Victor et al. 2018]: https://arxiv.org/abs/1811.06031
