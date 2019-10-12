# Task-oriented dialog systems

Personal assistant, solve tasks, write to a chat bot.

Intent classification - what does the user want? Which executable scenario is desired? Classification task where you can measure accuracy.

Form filling approach to dialog management. Each intent has a set of fields required to execute a request.

To fill and tag slots use BIO (beginning, inside, outside) scheme tagging.

May require context for multi-turn form filling dialog managers. Add previous utterance intent and slots filled in so far with binary features for each slot. An alternative is to use memory networks.

To track a form switch, look at when the intent switches.


# Intent classifier and slot tagger (NLU)

Intent Classifier methods include BOW with n-grams and TF-IDF, RNN (LSTM, GRU), CNN (1D convolutions).

SLot Tagger methods include handcrafted rules (regex), CRF, RNN seq2seq, CNN seq2seq, seq2seq with attention.

# Adding context to NLU

Use memory networks to store all previous utterances in memory. Encode the utterances using a Contextual Sentence Encoder (RNN). New utterance, encode and match to previous knowledge using the dot product and softmax (attention).

# Adding lexicon to NLU

A dataset might have a finite set of labels in training, how can we add a third party database?

Lexicon features
* Match every n-gram of input text against entries in our lexicon
* Successful when n-gram matches the prefix or postfix, at least half the length of the entry
* When there are multiple overlapping matches
  * prefer exact matches
  * prefer longer matches
  * prefer earlier matches

BIOES coding (Begin, Inside, Outside, End, Single). Often encoded as one-hot vectors.

**Training**

Can sample the lexicon dictionaries so that context and lexicon features are both learned. Augment the dataset by replacing slot values with values from the same lexicon.


# State tracking in DM

State tracker - queries external database, tracks the evolving state of dialog, constructs the state estimation.

Policy learner - takes the state estimation and chooses a dialog action.

DSTC 2 dataset - 3324 telephone-based dialogs for finding a restaurant in Cambridge.

Dialog state:
* Goals - distribution over values of each slot
* Method - distribution over methods 
* Requested slots - probability for each requestable slot that it has been requested by the user

State tracking - train a good NLU, make simple hand-crafted rules for dialog state change.

Neural belief tracker - uses previous system output, takes current utterance --> current state of dialog through embedding of context, utterance and candidate --> context modelling and semantic decoding -- binary decision making.

Frames dataset - frame tracking, extend state tracking to where several states are tracked simultaneously. Annotated with dialogue acts, slot types, slot values, references, to other frames for each utterance, ID of currently active frames.

# Policy optimisation in DM

Dialog policy: a mapping of dialog state --> agent act.

Optimise using machine learning
* Supervised - train to imitate observed actions of an expert, large amount of expert-labeled data, some state space may not be well-covered
* Reinforcement: given only a reward, optimise dialogue policy through interaction with users, requires many samples from an environment, need simulated users

# Final Remarks

Task-oriented dialog system: speech --> text (ASR), text --> natural language understanding (intent/slots), dialog manager - dialog state and policy (backend --> user (natural language generation)).

* Can train slot tagger and intent classifier in NLU separately
* Or jointly
* Train NLU and DM separately
* Or jointly
* Can use hand-crafted rules sometimes
* Learning from data is the best

**Evaluation**

NLU = turn-level metrics (intent accuracy, slots F1)
DM - turn-level metrics (state tracking accuracy), dialog-level metrics (task success rate, reward).

Slot tagger is more important than intent classifier.
