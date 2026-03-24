# The Transformer inference logic

At inference time, the Transformer generates tokens **autoregressively**: starting from a beginning-of-sequence token, the model predicts one token at a time, appending each prediction to the input sequence and feeding it back into the decoder for the next step. This sequential process continues until the model produces an end-of-sequence token or reaches a maximum length.

In the encoder-decoder setting, the encoder first computes a fixed context representation from the source sentence. The decoder then iteratively attends to this context through the cross-attention mechanism, while also attending to its own previously generated tokens through masked self-attention, to produce the next token in the target sequence.

The choice of **decoding strategy** determines how the next token is selected from the model's output probability distribution at each step. The two main strategies are greedy decoding and beam search.


## Greedy Decoding

The simplest decoding strategy is **greedy decoding**. At each time step $t$ in generation, the output token $\hat{w}_t$ is chosen by computing the probability for each word in the vocabulary and selecting the highest probability word (the argmax):

$$
\hat{w}_t = \text{argmax}_{w \in V} \, P(w \mid \bold{w}_{<t})
$$

Greedy decoding is straightforward and computationally efficient, but it has an important limitation: it is a **locally optimal** strategy. The token that looks most probable at time step $t$ might turn out to have been the wrong choice once the generation reaches step $t+1$. Because greedy decoding commits to each decision without reconsidering, it can miss globally better sequences.

Additionally, greedy decoding is **deterministic**: given the same context and the same model, it will always produce the exact same output. This predictability means the resulting text tends to be generic and often quite repetitive.


## Beam Search

**Beam search**[^1] addresses the limitation of greedy decoding by maintaining multiple candidate sequences (called **hypotheses**) at each time step, instead of committing to a single best token.

The key idea is to model decoding as searching the space of possible generations, represented as a **search tree** whose branches represent actions (generating a token) and nodes represent states (having generated a particular prefix). The goal is to find the sequence with the highest overall probability.

At each decoding step, beam search keeps the top $k$ most probable partial sequences, where $k$ is called the **beam width**. The algorithm proceeds as follows:

1. At the first step, compute a softmax over the entire vocabulary and select the $k$ most probable tokens. These form the initial set of hypotheses.
2. At each subsequent step, extend each of the $k$ hypotheses by considering all $|V|$ possible next tokens, producing $k \times |V|$ candidate sequences. Each candidate is scored by $P(y_i | x, y_{<i})$: the probability of the current word choice multiplied by the probability of the path that led to it. The $k \times |V|$ candidates are then pruned down to the top $k$ hypotheses.
3. This process continues until an EOS token is generated, indicating that a complete candidate output has been found. At this point, the completed hypothesis is removed from the frontier and the size of the beam is reduced by one. The search continues until the beam has been reduced to 0, resulting in $k$ completed hypotheses.

Note that when the beam width $k = 1$, beam search reduces to greedy decoding.

### Scoring hypotheses

To score each hypothesis, the chain rule of probability is used to decompose $P(y|x)$ into a product of conditional probabilities, which becomes a sum in log space (for an output string of length $t$):

$$
\text{score}(y) = \log P(y|x) = \sum_{i=1}^{t} \log P(y_i | y_1, \dots, y_{i-1}, x)
$$

Thus at each step, the probability of a partial sentence is computed by simply adding the log probability of the prefix so far to the log probability of generating the next token.

### Length normalization

One issue with this scoring method is that language models generally assign lower probabilities to longer strings. Since beam search compares completed hypotheses of potentially different lengths, a naive scoring strategy would favor shorter sequences. Note that this is not an issue during the earlier steps of decoding, since beam search is breadth-first and all hypotheses being compared have the same length.

To address this, **length normalization** is applied by dividing the log probability by the number of tokens:

$$
\text{score}(y) = \frac{1}{t} \log P(y|x) = \frac{1}{t} \sum_{i=1}^{t} \log P(y_i | y_1, \dots, y_{i-1}, x)
$$

For machine translation, beam widths $k$ between 5 and 10 are commonly used. The final result consists of $k$ hypotheses, which can either all be passed to a downstream application with their respective scores, or the most probable one can be selected as the final translation.

Beam search is particularly effective in constrained generation tasks such as machine translation, where the output is strongly conditioned on the source input and the space of valid translations is relatively narrow. By exploring multiple hypotheses simultaneously, it is more likely to find globally better sequences than greedy decoding alone.


[^1]: Lowerre, B. T. 1976. *The Harpy Speech Recognition System*
