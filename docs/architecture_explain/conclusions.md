# Conclusions

## Learned lessons


- **Vocabulary size's impact on model size.** The choice of word-level tokenization, driven by the capability to initialise embeddings with pretrained GloVe vectors, requires a large vocabulary to achieve reasonable language coverage. This is costly because the vocabulary size $V$ directly determines the number of trainable parameters in both the embedding layers ($V \times d_{model}$) and the language modeling head ($d_{model} \times V$), thus creating a tradeoff between model size and language coverage. To balance this tradeoff the vocabulary was built by selecting only words with a minimum frequency of 2 and capping $V$ at $70,000$.

    As the table below shows, the embedding and language modeling head layers dominate the parameter count across all model sizes, especially in the *Nano* and *Small* variants.
    The Encoder and Decoder layers are where the model actually learns to transform token representations through attention and feed forward sub-layers, effectively serving as its reasoning capacity. In the *Nano* and *Small* variants, however, these layers account for a small proportion of the total parameter budget, meaning that most parameters are consumed by the embedding and language modeling head layers rather than by the components that refine contextual understanding. This imbalance limits the smaller variants' ability to produce high quality translations.

    | Layers                     | Nano          | Small         | Base          | Original      |
    |:---------------------------|:--------------|:--------------|:--------------|:--------------|
    | **Encoder**                | 0.12M (1.12%) | 0.72M (3.16%) | 6.70M (8.45%) | 18.9M (12.47%)|
    | **Decoder**                | 0.16M (1.48%) | 0.95M (4.18%) | 9.55M (12.05%)| 25.2M (16.62%)|
    | **Encoder Embeddings**     | 3.50M (32.47%)| 7.00M (30.89%)| 21.0M (26.50%)| 35.8M (23.64%)|
    | **Decoder Embeddings**     | 3.50M (32.47%)| 7.00M (30.89%)| 21.0M (26.50%)| 35.8M (23.64%)|
    | **Language Modeling Head** | 3.50M (32.47%)| 7.00M (30.89%)| 21.0M (26.50%)| 35.8M (23.64%)|
    | **Total**                  | 10.8M         | 22.7M         | 79.2M         | 152M          |



- **Efficient attention module design.** An important aspect of designing a Transformer model, or broadly a Deep Learning model, is to make sure operations are executed efficiently. One of the main reasons the Transformer model has been developed is to parallelize operations in order to take advantage of parallel-optimized hardware, like GPUs and TPUs. In order to do so, dot product operations have to be grouped in matricial products as discussed in [The Attention mechanism](attention.md#scaled-dot-product-attention) where $q_i \cdot k_j \\ \forall i, j $ are grouped by considering two matrices $Q$ and $K$ containing all queries and all keys respectively, and then computing a single matricial product $Q K^\top$.
This has to happen also for the heads: since each head operates on its own independent $Q_i, K_i, V_i$ projections, a separate matrix product $Q_i K_i^\top$ would have to be computed for each head. To avoid this, the projection matrices $W_Q, W_K, W_V$ are built with dimensions $d_{model} \times A \cdot d_{head}$, so that a single matrix multiplication projects the input into unified $Q, K, V$ matrices that contain the queries, keys, and values for all $A$ heads.

    Note: hardware-level attention optimisations such as FlashAttention are considered out of scope for this project.





## Modern architectural advances

The basic encoder-decoder Transformer has inspired numerous researchers who improved many aspects of the architecture and adapted it to scenarios beyond Natural Language Processing:

- **Efficient attention variants.** The standard scaled dot product attention has $O(n^2)$ complexity with respect to the sequence length $n$, because the softmax over all key positions forces every query to interact with every key. Two major families of alternatives aim to reduce this cost[^1]. *Linear attention* replaces the softmax kernel with a decomposable feature map $\phi(\cdot)$ so that attention can be rewritten as $\phi(Q)(\phi(K)^\top V)$, reducing complexity from $O(n^2 d_{model})$ to $O(n d_{model}^2)$ since the expensive $n \times n$ attention matrix is never explicitly formed. *Sparse attention* restricts each query to attend only to a subset of keys through fixed patterns (local windows, dilated, block-sparse), routing-based selection, or clustering-based retrieval, cutting cost while retaining most of the modelling capacity. Modern architectures usually adopt *hybrid* designs that interleave dense, sparse, and local attention across layers, for example using *sliding window attention* in early layers for local context and full attention in deeper layers for global reasoning, balancing efficiency with long context modelling capacity.

- **$\mathbf{KV}$ cache.** Modern Transformer based systems need to cache $K$ and $V$ vectors to avoid recomputing them during every token generation step, which would otherwise heavily slow down inference. The problem is that memory requirements grow as $2\, n_h\, d_{model}\, n_l$, with $n_h$ the number of attention heads and $n_l$ the number of layers. As the model scales up, the $KV$ cache grows linearly with these parameters. Several techniques have been proposed to address this: *GroupedQuery Attention* lets $G$ groups of query heads attend to their respective shared pair of $K$ and $V$ vectors; *MultiQuery Attention* is the extreme variant with $G=1$, such that every query head attends to the same single pair of $K$ and $V$; *MultiHead Latent Attention* compresses $K$ and $V$ vectors into a lower dimensional latent space and stores these latents as the $KV$ cache.

    The last approach is the most interesting one since it employs the core idea of an **AutoEncoder** creating a compressed latent space that represents higher dimensional vectors with high fidelity. Experiments have also shown better performance[^2] on top of a significantly smaller KV cache.



- **Manifold constrained Hyper-Connections.** The standard residual connection owes much of its effectiveness to the *identity mapping* property: the input passes through unchanged, preserving the global mean of features across layers and providing a stable gradient highway. *Hyper-Connections* (HC)[^3] generalise this by widening the residual stream with learnable mixing matrices that increase topological complexity. However, when stacked across many layers the composite mixing no longer preserves the feature mean, leading to unbounded signal amplification or attenuation, thus causing exploding and vanishing gradients when training at scale. *Manifold-Constrained Hyper-Connections* (mHC), proposed by DeepSeek[^4], solve this by projecting each mixing matrix onto the set of *doubly stochastic matrices* (rows and columns sum to one and all values are non-negative) via the Sinkhorn-Knopp algorithm. Because a doubly stochastic matrix acts as a convex combination of the input features, the feature mean is conserved and the signal norm is regularised; and since doubly stochastic matrices are closed under multiplication, this conservation holds across arbitrarily deep stacks, restoring the stability of the original residual connection while retaining the expressiveness of HC.

- **Mixture of Experts.** Rather than activating all parameters for every input token, *Mixture of Experts* (MoE) architectures partition the feed forward layers into multiple *expert* sub networks and use a lightweight *router* module to select only a small subset of experts for each token at every layer. This decouples total model capacity from per-token computational cost: a model can have hundreds of billions of parameters while only activating a fraction of them during each forward pass. Modern MoE systems, such as those used in Mixtral[^5] and DeepSeek V2[^2], employ techniques like top-$k$ routing, load balancing losses, and shared experts to ensure training stability and an even distribution of tokens across experts. The result is a favourable scaling law: significantly better performance per FLOP compared to dense models of equivalent compute budget.

- **Relative positional encodings.** The proposed Transformer model injects *absolute positional encodings* that are added to the token embeddings in the Embedding layers. *Relative positional encodings* instead encode the distance between tokens directly into the attention computation, making the model inherently more robust to varying sequence lengths at inference time. The most widely adopted variant today is *Rotary Position Embedding* (RoPE)[^6], which encodes position by rotating the query and key vectors in pairs of dimensions. Because the dot product between a rotated query and key depends only on their relative distance, RoPE naturally captures relative position while being simple and efficient.

- **Encoder only models**. They remove the decoder entirely and stack only bidirectional encoder layers. Because every token can attend to all other tokens in the sequence, these models excel at understanding tasks, like classification, named entity recognition (NER), semantic similarity, and clustering. They are a core building block of modern semantic retrieval mostly used in *Retrieval Augmented Generation* (RAG) pipelines, where documents are embedded into a dense vector space for efficient similarity search. Most used models of this type are BERT, RoBERTa, ModernBERT.

- **Decoder only models**. They remove the encoder and rely solely on causal decoder stacks (so removing also the cross attention modules). Each token can attend only to previous tokens, making these models naturally suited for autoregressive generation. This simplicity, combined with aggressive scaling of parameters and data, has proven remarkably effective: decoder only models are the base of all modern LLMs and chat assistants, like GPT, Llama and DeepSeek.

- **Vision Transformer and Adaptive LayerNorm.** The *Vision Transformer* (ViT) [^7] demonstrated that the Transformer architecture transfers directly to computer vision: an image is split into fixed size patches, each patch is linearly projected into an embedding, and the resulting sequence is processed by a standard Transformer encoder. This patch-based tokenisation alongside positional encoding removes the need for convolutions and allows vision models to benefit from the global attention patterns and the same scaling laws observed in NLP[^8]. *Adaptive LayerNorm* (adaLN), popularised by the *Diffusion Transformer* DiT[^9], replaces the parameters of LayerNorm with parameters that are dynamically predicted from a conditioning signal (e.g. a timestep or class label). This enables a single model to modulate its internal representations based on external context, providing fine-grained control over generation without requiring separate conditioning branches.


## To wrap up

The Transformer architecture marks a fundamental shift in sequence modeling. By discarding recurrence and convolution entirely, it overcomes the sequential bottleneck of RNNs and the limited receptive fields of CNNs, enabling full parallelization during training while directly capturing long-range dependencies.

At its core, the scaled dot-product attention mechanism provides a flexible information routing system: queries, keys, and values allow each token to selectively attend to the most relevant parts of the input, with the softmax weighting ensuring stable, normalized outputs. Multi-head attention extends this by projecting into multiple subspaces in parallel, letting different heads specialize on distinct linguistic relationships.

Raw text enters the model through word level tokenization and dense embeddings, where pretrained GloVe vectors provide a rich semantic foundation. Sinusoidal positional encodings then restore sequence order by injecting position-dependent signals at multiple frequencies, preserving both fine-grained local and coarse-grained global positional information without adding learnable parameters.

These representations flow through stacked encoder layers, where self-attention builds bidirectional contextual embeddings, and through decoder layers, where causal masking enforces autoregressive generation and cross-attention bridges the source and target languages. Feed-forward networks at each layer introduce the non-linear transformations necessary to learn abstract, higher-level concepts from the attention-aggregated context.

Throughout the entire architecture, residual connections and layer normalization work together to ensure training stability: the residual stream provides a gradient highway that enables effective optimization of deep stacks, while normalization keeps activations well-conditioned across layers and training steps.

Together, these components form a cohesive and modular architecture whose design principles have proven foundational not only for machine translation but for the broader landscape of modern AI.



[^1]: Wan et al. 2025. Efficient Attention Mechanisms for Large Language Models: A Survey. <https://arxiv.org/abs/2507.19595>.
[^2]: DeepSeekAI et al. 2024. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. <https://arxiv.org/abs/2405.04434>.
[^3]: Zhu et al. 2024. HYPER-CONNECTIONS. <https://arxiv.org/pdf/2409.19606>
[^4]: DeepSeekAI et al. 2025. mHC: Manifold-Constrained Hyper-Connections. <https://arxiv.org/abs/2512.24880>
[^5]: Jiang et al. 2024. Mixtral of Experts. <https://arxiv.org/abs/2401.04088>.
[^6]: Su et al. 2023. RoFormer: Enhanced Transformer with Rotary Position Embedding. <https://arxiv.org/abs/2104.09864>.
[^7]: Dosovitskiy et al. 2021. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. <https://arxiv.org/abs/2010.11929>.
[^8]: Dehghani et al. 2023. Scaling Vision Transformers to 22 Billion Parameters.
<https://arxiv.org/abs/2302.05442>.
[^9]: Peebles & Xie. 2023. Scalable Diffusion Models with Transformers. <https://arxiv.org/abs/2212.09748>.
