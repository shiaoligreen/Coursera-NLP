**Distributional Semantics**

First order co-occurences - syntagmatic associates, reatedness (bee and honey)

second order co-occurences - paradigmatic parallels, similarity (bee and bumblebee)

Pointwise Mutual Information
$$PMI = \log\frac{p(u,v)}{p(u)p(v)} =\log\frac{n_{uv}n}{n_un_v} $$

To avoid large negative values, use positive PMI
$$pPMI = \max(0,PMI)$$


**Explicit and Implicit Matrix Factorization**

Use truncated version of Singular Value Decomposition (SVD) for approximation.

$$\hat{X}_k = U_k\sum_kV_k^T$$

Approximation of rank k in terms of Frobenius norm.

Vector space model of semantics:
* input - word-word co-occurences
* method - dimensionality reduction
* output - similarity between vector representations of words

Weighted squared loss - GloVe
$$\sum_{u \in W}\sum_{v \in W} f(n_{uv}) (\langle \phi_u,\theta_v \rangle + b_u - b'_v - \log n_{uv})^2$$

Skip-gram - predict context words given a focus word. Model each probability with a softmax. Still two matrices of parameters.

Skip-gram Negative Sampling - instead of predicting a word for another word, predict "yes" or "no" for word pairs
* use positive examples from data
* sample negative examples
* Train with SGD to find two matrices of parameters

**word2vec and doc2vec**

2 architectures:
* CBOW (Continuous Bag-of-words)
* Continuous Skip-gram

To avoid softmax:
* Negative sampling
* Hierarchical softmax

Evaluate word similarities - human judgement, Spearman's correlation.

Evaluate word analogies - relational similarity, human judgement.

Paragraph2vec / doc2vec - context given focus word and vice versa.

2 architectures
* DM (Distributed Memory) - word given context
* DBOW (Distributed Bag Of Words) - context given focus word

Evaluation - triplets from archive papers, one is dissimilar to the other two.


**Word Analogies**

Word2vec is unsupervised, but can understand some meaning of the language.

Works well for inflectional morphology (student:students, strong:stronger), derivational morphology (bake:baker, edit: editable), lexicographical semantics (peach:fruit, up:down), encyclopedic semantics (cat:kitten, dog:bark). 

Be careful about effectiveness for word analogy tasks.


**Characters to sentence embeddings**

Morphology can be used to improve word embeddings.

FastText - represent a word as a bag of character n-grams, model a word vector as a sum of sub-word vectors.

Sent2vec - learn sentence embeddings as a sum of sub-sentence units.

StarSpace - Mode 3 (sentence embeddings) - learn pairwise similarity from collections of similar objects (sentence similarity). 
* Each line is a collection of similar sentences
* Each sentence is represented as a bag of features. Embedded to predict sentence similarity.
* Similar sentence pairs are taken from the collection.
* Dissimilar sentence pairs are sampled at random.

Deep learning for sentence representations - recurrent neural networks (sequence modelling), convolutional neural networks, recursive neural networks (hierarchical structure).

Skip-thought vectors - predict next and previous sentences in the text, RNN encoder-decoder architecture.


**Topic Modelling**

Can be seen as soft bi-clustering - bi-clustering because words and documents are grouped, soft because probability distributions are used for assignment.
* provides hidden semantic representation of texts

Probabilistic Latent Semantic Analysis (PLSA)

$$p(w|d) = \sum_{t \in T} p(w|t,d)p(t|d) = \sum_{t \in T} p(w|t)p(t|d) $$

Generative models!


**Training PLSA**

Use log-likelihood optimisation while considering non-negative and and normalisation constraints.

Use Bayes' rule and product rule to estimate the topic assignment probabilities.

EM algorithm

E step:

$$p(t|d,w) = \frac{p(w|t)p(t|d)}{p(w|d)} = \frac{\phi_{wt}\theta_{td}}{\sum_{s \in T}\phi_{ws}\theta_{sd}}$$

M step:

$$\phi_{wt} = \frac{n_{wt}}{\sum_wn_{wt}}$$  $$n_{wt} = \sum_d n_{dw}p(t|d,w)$$
$$\theta_{td} = \frac{n_{td}}{\sum_tn_{td}}$$
$$n_{td} = \sum_w n_{dw} p(t|d,w)$$


**The Zoo of Topic Models**

Latent Dirichlet Allocation - dirichlet priors for $\phi_t = (\phi_{wt})_{w \in W}$ and $\theta_d = (\theta_{td})_{t \in T}$.

Inference - Variational Bayes, Gibbs Sampling.

Output - posterior probabilities for parameters (Dirichlet).

Hierarchical topic model - extension of LDA model. Topics and sub-topics.

Dynamic topic models - topics can evolve over time.

Multilingual topic models - trained on parallel data for the same topic, expressed in different terms.

Additive Regularization for Topic Models - add regularisation to likelihood for topics.
* Want diversity in pairs of topics
* Regularised EM-algorithm


Multimodal topic models - metadata can be built in using Mutli-ARTM through weighted likelihoods for each modality.
* Each topic has a few probability distributions
* More parameters, still EM


