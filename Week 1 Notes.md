**Linear models for sentiment analysis**

Feature Extraction - e.g. TF-IDF values.

Sentiment classification - positive or negative bag of words or logistic regression.

*   Simple methods - bag of words, logistic regression (+ 2-grams).
*   Improve - tokenization, normalization, other models (SVM, Naïve Bayes), Deep Learning.

**Hashing Trick in spam filtering**

Hash map - can not fit in memory on one machine, it is difficult to synchronize.

Hashing - take N-gram, you take a hash value of that N-gram and take that value modulo $2^{20}$, or $2^{22}$ or any other huge number.

Hash - function that converts an input into some number.

**Neural Networks for Words**

Word2vec property - words that have a similar context have collinear vectors.

1D convolution - provides high activations for 2-grams with certain meaning.

*   Can be extended to n-grams.
*   Multiple filters as track many n-grams.
*   Maximum pooling over time.

Model architecture

*   3, 4, 5-gram windows with 100 filters each.
*   MLP on top of these 300 features.

**Neural Networks for Characters**

Can also apply 1D convolutions on one-hot encoded characters.

Max pooling - provides a bit of position invariance for character n-grams.

Model Architecture


*   1014 characters of text.
*   1D convolution + max pooling 6 times.
  *   Kernel widths: 7, 7, 3, 3, 3, 3
  *   1024 filters
*   Apply MLP to 1024 x 34 matrix.
