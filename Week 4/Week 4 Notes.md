# Introduction to Machine Translation

Data - parallel corpora like Euoropean Parliament proceedings, movie subtitles, http://opus.lingfil.uu.se/.

Problems - noisy, specific domain, rare language pairs, not aligned, insufficient.

Evaluation - BLEU score, a way to compare the output to several reference translations. 

Brevity penalty - divide the length of the output by the length of the reference. min( 1, output/reference).

**BLEU Score**

$$\text{BLEU} = \text{brevity}\sqrt[4]{\text{1-gram}\times\text{2-gram}\times\text{3-gram}\times\text{4-gram}} $$ 

# Noisy Channel

Given French sentence, find English translation.

$$e* = \arg\max_{e \in E}p(e)p(f|e)$$

(fluency * adequacy of translation)


# Word Alignment Models

Alignment - which words correspond to each other.

Constrain the matrix - each target word is only allowed to have one source.

Probabilistic model (generative story)
* Likelihood maximisation for the incomplete data $p(f|e, \Theta) = \sum_a p(f,a|e, \Theta) \to \max_\Theta$
* EM-algorithm
  * E-step: estimates posterior probabilities for alignments
  * M-step: updates $\Theta$ - parameters of the model


1. Choose the length of the foreign sentence
2. Choose an alignment for each word
3. Choose the word

**IBM model 1**

$$p(f,a|e) = p(J|e) \prod_{j=1}^J p(a_j)p(f_j|a_j,e)$$
* Model is simple and does not have too many parameters
* Alignment prior does not depend on word positions

**IBM model 2**

$$p(f,a|e) = p(J|e)\prod_{j=1}^J p(a_j|j, I, J)p(f_j|a_j,e)$$

* Alignments depend on position-based prior
* Quite a lot of parameters for the alignments

Re-parametrization - model the matrix as a diagonal matrix.

**HMM for the prior**

$$p(f,a|e) = \prod_{j=1}^Jp(a_j|a_{j-1},I,J)p(f_j|a_j,e)$$

Transition probabilities, transition tables.


# Encoder-decoder architecture

Encoder - maps the source sequence to the hidden vector.

Decoder - performs langauge modeling given this vector.
 
Prediction - often using softmax.

LSTM model through RNN and LCM modules.

# Attention mechanism

Ways to compute attention weights:
* Additive attention: $$sim(h_i, s_j) = w^T \tanh(W_hh_i + W_ss_j)$$
* Multiplicative attention: $$sim(h_i,s_j) = h_i^TWs_j$$
* Dot product: $$sim(h)i,s_j) = h_i^Ts_j$$

Humans save time with attention whereas machines waste time with attention.

Local attention:
* Find the most relevant position $a_j$ in the source
* attend only positions within a window


# How to deal with a vocabulary?

Computing softmax is slow and there are often out of vocabulary words.

Develop a hierarchical softmax through a binary tree.

$$p(w_n = w | w_1^{n-1}) = \prod p(d_i|w_1^{n-1})$$

Construct a tree using semantics or a Huffman tree.

Can also implement sub-word modeling like word-character hybrid models and byte-pair encoding.


# How to implement a conversational chat-bot?

Chat-bot can be goal-oriented or have a casual conversation.

Retrieval-based mdoels
* use a repository of predefined responses
* no grammatical mistakes
* unable to handle unseen cases

Generative models
* generate new responses from scratch
* can make mistakkes
* impression you're talking to a human

Sequence to sequence
* LSTM for encoder and decoder
* Attention (or reversed input)
* Beam search for the best response

Variance in length. Can deal with this through padding or bucketing.

 Sequence to sequence learning: one-size fits all?

* Machine Translation
* Summarization
* Text simplification
*  Language to code
* Chit-chat bot
* Question answering
* Speech recognition
* Image caption generation

**Summarization**

Extract parts of the text or generate summary that is not from the text.

**Simplification**

Reduce the lexical and syntactical complexity text. Use operations like splitting, deletion, paraphrasing: reordering, lexical substitution, syntactic transformations.

Encoder-decoder framework with reinforement learning.

Measure using SARI (system against references and input) - arithmetic aveage of n0gram precision and recall of addition, copying, deletion.

# Summarization with pointer-generator networks

**Sequ2seq + attention**

Attention mechanism generates the distribution or vocabulary.

Copy distribution - tells us when to copy something from the input.

Pinter-generator network - use a mixture of a copy and generative model.

Coverage mechanism - sum all attention distributions to a coverage vector. Coverage loss is used to minimise the minimum of the attention probabilities and the coverage vector.
