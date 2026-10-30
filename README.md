# Transformer Architecture — My Learning Notes

> A practical, from-first-principles explanation of how Transformers work, why attention matters, and how GPT-style language models generate text.

## Table of Contents

1. [What is a Transformer?](#1-what-is-a-transformer)
2. [Why Transformers replaced recurrent architectures](#2-why-transformers-replaced-recurrent-architectures)
3. [The original encoder–decoder design](#3-the-original-encoderdecoder-design)
4. [Tokenization, embeddings, and position](#4-tokenization-embeddings-and-position)
5. [Self-attention and Q, K, V](#5-self-attention-and-q-k-v)
6. [Scaled dot-product attention: worked example](#6-scaled-dot-product-attention-worked-example)
7. [Multi-head attention](#7-multi-head-attention)
8. [Feed-forward networks](#8-feed-forward-networks)
9. [Residual connections and normalization](#9-residual-connections-and-normalization)
10. [Transformer blocks and stacking](#10-transformer-blocks-and-stacking)
11. [Encoder: bidirectional attention](#11-encoder-bidirectional-attention)
12. [Decoder: causal self-attention](#12-decoder-causal-self-attention)
13. [Cross-attention in encoder–decoder models](#13-cross-attention-in-encoderdecoder-models)
14. [How a model predicts the next token](#14-how-a-model-predicts-the-next-token)
15. [End-to-end translation walkthrough](#15-end-to-end-translation-walkthrough)
16. [Encoder-only vs decoder-only vs encoder–decoder](#16-encoder-only-vs-decoder-only-vs-encoderdecoder)
17. [How GPT-style inference actually works](#17-how-gpt-style-inference-actually-works)
18. [Training vs inference](#18-training-vs-inference)
19. [Why Transformers scale—and their limitations](#19-why-transformers-scaleand-their-limitations)
20. [Common misconceptions](#20-common-misconceptions)
21. [Interview-ready explanation](#21-interview-ready-explanation)
22. [Quick revision cheat sheet](#22-quick-revision-cheat-sheet)
23. [References](#23-references)

---

## 1. What is a Transformer?

A **Transformer** is a neural network architecture for processing sequences. Its defining operation is **attention**, which lets each position combine information from other permitted positions in the sequence.

Transformers were introduced in the 2017 paper **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)**. The original model was built for sequence-to-sequence tasks such as translation, with an **encoder** that processes the source and a **decoder** that generates the target.

A Transformer is not literally a word-to-word lookup machine. Text is first converted into **tokens**, then numerical vectors, then processed through many learned transformations. Transformer-based architectures also work with image, audio, and other modalities.

```text
Input text → Tokenizer → Token IDs → Embeddings + position
                                        ↓
                                Transformer blocks
                                        ↓
                              Contextual hidden states
                                        ↓
                         Output head / next-token logits
                                        ↓
                            Token selection → Text
```

**My mental model:** attention moves information *between positions*; the feed-forward network transforms information *within each position*; stacking blocks builds richer representations.

## 2. Why Transformers replaced recurrent architectures

Earlier sequence models, such as RNNs, LSTMs, and GRUs, processed tokens through a recurrent hidden state:

\[
h_t = f(W_x x_t + W_h h_{t-1} + b)
\]

This means step `t` depends on step `t−1`. Two difficulties follow:

- **Limited parallelism:** recurrent steps cannot all be computed independently at once.
- **Long-range dependencies:** information and gradients must travel across many sequential steps. LSTMs and GRUs mitigate this but do not eliminate all challenges.

Attention allows a token to directly gather information from other permitted positions, and all positions within a training sequence can be processed in parallel within a layer.

**Important nuance:** a decoder-only LLM still *generates new tokens sequentially* at inference time. Transformers parallelize computation across positions during training and prompt processing, not the dependency between successive output-token decisions.

## 3. The original encoder–decoder design

```text
Source sentence
      │
      ▼
Token embeddings + position
      │
      ▼
┌───────────────────────────┐
│ ENCODER × N               │
│ • Bidirectional attention │
│ • Feed-forward network    │
│ • Residuals + norm        │
└───────────────────────────┘
      │  Encoder representations (memory)
      │
      ├────────────────────────────────────┐
      │                                    │
      │                         Target prefix tokens
      │                                    │
      │                                    ▼
      │                         Embeddings + position
      │                                    │
      │                                    ▼
      │                         ┌───────────────────────────┐
      └────────────────────────►│ DECODER × N               │
                                │ • Causal self-attention   │
                                │ • Encoder cross-attention│
                                │ • Feed-forward network   │
                                │ • Residuals + norm       │
                                └───────────────────────────┘
                                            │
                                            ▼
                                  Logits → next token
                                            │
                                      Repeat until stop
```

The encoder does not simply output a single sentence meaning: typically it returns **one contextual vector per source-token position**. The decoder can attend to all these vectors through cross-attention.

## 4. Tokenization, embeddings, and position

### 4.1 Tokenization

A tokenizer breaks text into vocabulary units. A token might be a whole word, a word fragment, punctuation, or another unit. For example, `I love learning` might be split into `I`, ` love`, ` learning`—the exact result depends on the tokenizer.

Each token maps to an integer ID. The number itself is an identifier, **not** a semantic score.

### 4.2 Token embeddings

An embedding table maps token IDs to learned vectors. For a sequence of `n` tokens and hidden size `d_model`, the input embedding tensor has shape:

\[
X \in \mathbb{R}^{n \times d_{model}}
\]

For illustration, three tokens with four-dimensional embeddings might be:

```text
I         → [ 0.2, 0.7, -0.1, 0.4]
love      → [ 0.8, 0.1,  0.5,-0.2]
learning  → [ 0.4, 0.9,  0.2, 0.6]
```

These values are invented. Real embeddings are learned, usually high-dimensional, and later layers turn them into **context-dependent** representations.

### 4.3 Positional information

Attention alone does not encode token order. `Dog bites man` and `Man bites dog` must be distinguishable.

The original Transformer added sinusoidal positional vectors to token embeddings:

\[
X_{input} = X_{embedding} + P
\]

Its positional encoding was defined as:

\[
PE(pos, 2i) = \sin\left(pos/10000^{2i/d_{model}}\right)
\]

\[
PE(pos, 2i+1) = \cos\left(pos/10000^{2i/d_{model}}\right)
\]

Modern models may use **RoPE** (rotary positional embeddings), learned positions, or other positional schemes. RoPE incorporates relative position effects through rotations in attention rather than simply adding a position vector to token embeddings.

## 5. Self-attention and Q, K, V

Consider: **“The cat sat on the mat because it was tired.”**

The representation of `it` may benefit from information about `cat`. Attention lets token positions combine relevant context.

Each position's current hidden vector is linearly projected into:

- **Query (Q):** what this position seeks, as a useful intuition.
- **Key (K):** how this position can be matched against a query.
- **Value (V):** information contributed if the position receives attention.

The actual computation is:

\[
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V
\]

The matrices `W_Q`, `W_K`, and `W_V` are **learned parameters**. They are not hand-coded grammar rules.

For every query position, the model compares its query with keys at allowed positions, normalizes the scores, and takes a weighted sum of the values.

## 6. Scaled dot-product attention: worked example

The core formula is:

\[
\boxed{\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V}
\]

Where:

- `QKᵀ` computes query–key compatibility scores.
- `√d_k` scales scores to help avoid overly saturated softmax distributions.
- `M` is a mask (zero for allowed positions, effectively `−∞` for disallowed positions).
- `softmax` converts scores into nonnegative weights summing to 1.
- Multiplication by `V` creates weighted combinations of value vectors.

**Toy example:** Suppose the final scaled scores for the query token `sleeps` against `The`, `cat`, `sleeps` are `[1, 3, 2]`. Softmax gives approximately:

```text
The      0.09
cat      0.67
sleeps   0.24
```

So its new attention output is approximately:

\[
0.09 V_{The} + 0.67 V_{cat} + 0.24 V_{sleeps}
\]

These are illustrative values, not observed attention weights from a real model. Also, **attention weights are not a complete or guaranteed explanation of a model's reasoning**.

### Matrix dimensions

For `n` tokens, a head with key dimension `d_k` and value dimension `d_v` typically has:

```text
Q: n × d_k
K: n × d_k
V: n × d_v
QKᵀ: n × n
Attention output: n × d_v
```

The `n × n` attention matrix describes possible interactions between token positions.

## 7. Multi-head attention

A single attention head uses one set of learned projections. **Multi-head attention** computes multiple heads in parallel, then combines them:

\[
head_i = Attention(Q_i,K_i,V_i)
\]

\[
MultiHead(X) = Concat(head_1, \ldots, head_h) W_O
\]

```text
                Input hidden states
                        │
           ┌────────────┼────────────┐
           ▼            ▼            ▼
        Head 1        Head 2       Head 3 ...
           │            │            │
           └────────────┼────────────┘
                        ▼
                   Concatenate
                        ▼
                 Output projection
```

For example, if `d_model = 512` and there are 8 heads, a common design uses 64-dimensional query/key vectors per head (`8 × 64 = 512`).

Different heads can learn different patterns, but they are **not guaranteed** to specialize neatly into grammar, facts, or particular relationships.

**Heads are parallel within a layer. Layers are stacked sequentially.**

## 8. Feed-forward networks

After attention, each position passes through a feed-forward neural network (FFN/MLP). In the original Transformer:

\[
FFN(x) = \operatorname{ReLU}(xW_1+b_1)W_2+b_2
\]

A typical original configuration expands `512 → 2048` dimensions, applies a nonlinearity, then projects `2048 → 512`.

The **same FFN weights** are applied independently at each token position. It transforms representations but does not directly mix information between positions; attention does that.

Modern LLMs may use gated MLPs such as **SwiGLU** rather than the original ReLU FFN.

## 9. Residual connections and normalization

### Residual connections

A residual/skip connection adds a sublayer's input to its output:

\[
y = x + F(x)
\]

This helps preserve information and improve gradient flow through deep networks.

### Layer normalization

LayerNorm normalizes features within a token representation:

\[
LN(x) = \gamma \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
\]

Normalization helps stabilize optimization. The original Transformer used **Post-LN**:

\[
y = LN(x + F(x))
\]

Many newer LLMs use **Pre-LN** arrangements:

\[
y = x + F(LN(x))
\]

Some use **RMSNorm** instead of LayerNorm. The exact ordering and normalization method vary by model.

## 10. Transformer blocks and stacking

A simplified encoder block contains:

```text
Input
  ↓
Multi-head self-attention
  ↓
Residual + normalization
  ↓
Feed-forward network
  ↓
Residual + normalization
  ↓
Output → next block
```

A Transformer stacks blocks:

\[
H^{(1)}=F_1(X), \quad H^{(2)}=F_2(H^{(1)}), \quad \ldots, \quad H^{(N)}=F_N(H^{(N-1)})
\]

Each block has its **own learned parameters**. The original architecture had six encoder blocks and six decoder blocks. GPT-3 175B used 96 Transformer layers.

More layers provide more stages of computation, but model quality depends on many factors beyond depth.

## 11. Encoder: bidirectional attention

An encoder typically uses **bidirectional self-attention**: every token position may attend to earlier, current, and later input positions (subject to padding or other task-specific masks).

For `I love machine learning`, when processing `love`, an encoder can use information from `I`, `love`, `machine`, and `learning`.

```text
Encoder self-attention visibility (✓ = allowed)

Query \ Key      I    love   machine  learning
I                ✓      ✓       ✓        ✓
love             ✓      ✓       ✓        ✓
machine          ✓      ✓       ✓        ✓
learning         ✓      ✓       ✓        ✓
```

The encoder outputs contextual vectors for all input positions. **BERT** is a well-known encoder-only architecture, commonly adapted for classification, extraction, and other representation-focused tasks.

## 12. Decoder: causal self-attention

An autoregressive decoder uses **causal (masked) self-attention**. Each position may attend to itself and earlier positions, **not future positions**.

```text
Decoder self-attention visibility (✓ = allowed)

Query \ Key      I    love   machine  learning
I                ✓      ×       ×        ×
love             ✓      ✓       ×        ×
machine          ✓      ✓       ✓        ×
learning         ✓      ✓       ✓        ✓
```

Why mask the future? During generation, future target tokens are not available. During training, the mask prevents the model from cheating by reading target tokens that it is supposed to predict.

**Precise wording:** The future-token restriction applies to **decoder self-attention over the target sequence**. It does not mean an encoder–decoder model's decoder cannot access future *source* tokens through cross-attention.

## 13. Cross-attention in encoder–decoder models

The original Transformer decoder has **three sublayers** per block:

1. Causal self-attention over the target prefix.
2. **Cross-attention** to the encoder's output.
3. Feed-forward network.

For cross-attention:

\[
Q = H_{decoder}W_Q, \qquad K = H_{encoder}W_K, \qquad V = H_{encoder}W_V
\]

The decoder asks questions of the encoder's representations while generating output. For translation, this connects the emerging target-language sequence with the complete source-language input.

**Important:** A GPT-style decoder-only model does **not** normally include the original encoder–decoder cross-attention sublayer. Instead, its prompt and generated tokens share one causal attention stream.

## 14. How a model predicts the next token

After the final decoder block, the model has hidden vectors. A learned output projection produces **logits**, one per vocabulary token:

\[
z = hW_{out}+b
\]

Softmax converts logits into probabilities:

\[
P(t_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
\]

Example probabilities after the prefix `J'`:

```text
adore      72%
aime       15%
mange       1%
other      12%
```

These numbers are illustrative. A decoding strategy selects the next token:

- **Greedy:** choose the highest-probability token.
- **Temperature:** rescale logits before sampling; lower values generally concentrate probability, higher values flatten it.
- **Top-k:** sample among the k highest-scoring candidates.
- **Top-p:** sample from the smallest high-probability set reaching a chosen cumulative mass.

The selected token is appended to the context. The process repeats until an end token, length limit, or another stopping condition.

## 15. End-to-end translation walkthrough

Task: translate `I love learning` into French.

**Encoder:**

1. Tokenize the English source.
2. Look up embeddings and add positional information.
3. Pass all source positions through encoder blocks with bidirectional attention.
4. Return contextual source representations.

**Decoder:**

```text
Start target prefix: <bos>
Predict next token(s): J'
Updated prefix:      <bos> J'
Predict next token(s): adore
Updated prefix:      <bos> J' adore
Predict next token(s): apprendre
Updated prefix:      <bos> J' adore apprendre
Predict:             <eos>
```

At every step, the original decoder combines (a) causal self-attention over the target prefix, (b) cross-attention to the source representations, and (c) feed-forward computation.

This is a **conceptual word-level walkthrough**; actual tokenizers may split these words into smaller pieces.

## 16. Encoder-only vs decoder-only vs encoder–decoder

| Variant | Attention behavior | Representative models | Common use cases |
|---|---|---|---|
| **Encoder-only** | Bidirectional self-attention | BERT | Classification, extraction, representations |
| **Decoder-only** | Causal self-attention | GPT, Llama | Chat, code, autoregressive generation |
| **Encoder–decoder** | Bidirectional source encoder + causal target decoder + cross-attention | T5, BART | Translation, summarization, sequence-to-sequence tasks |

**Critical insight:** Both **decoder-only** and **encoder–decoder** architectures can generate output **sequentially**. Sequential generation is not unique to decoder-only models.

Many modern generative LLMs use decoder-only Transformer cores. Multimodal systems can also include separate modality-specific processing components, so calling an entire multimodal system “only a decoder” may be too simplistic.

## 17. How GPT-style inference actually works

For a decoder-only model, the prompt and generated output are part of the **same autoregressive sequence**:

```text
Prompt: "What is machine learning?"
                  ↓
       Tokenizer + embeddings
                  ↓
    Causal Transformer blocks
                  ↓
       Next-token logits
                  ↓
         Select a token
                  ↓
       Append to sequence
                  ↺ repeat
```

Two practical phases matter:

- **Prefill:** process the prompt and construct the initial attention **KV cache**.
- **Decode:** generate subsequent tokens one at a time, reusing previously computed keys and values rather than recalculating everything.

This is relevant to production metrics such as **time to first token (TTFT)**, output tokens per second, throughput, and memory use.

A causal mask applies even when the entire prompt is processed in parallel: position `i` cannot attend to later positions `j > i`.

## 18. Training vs inference

**Training:** the model learns its parameters by predicting target tokens and minimizing a loss, often next-token cross-entropy for autoregressive language models:

\[
\mathcal{L}=-\sum_t \log P(x_t\mid x_{<t})
\]

Backpropagation and an optimizer update the weights. Post-training methods can further improve instruction following and preferred behaviors.

**Inference:** weights normally stay fixed. The model computes predictions for the prompt and generates tokens using a decoding strategy.

| | Training | Inference |
|---|---|---|
| Main purpose | Learn parameters | Produce outputs |
| Weight updates | Yes | Normally no |
| Backpropagation | Yes | Normally no |
| Token processing | Parallel across positions with causal masks | Prompt prefill, then sequential decoding |

## 19. Why Transformers scale—and their limitations

**Strengths:**

- Parallel processing across sequence positions during training and prefill.
- Direct information exchange between distant permitted positions.
- Flexible architectures for language, vision, audio, and multimodal tasks.
- Effective scaling with data, parameters, and compute when appropriately trained.

**Limitations and engineering tradeoffs:**

- Standard dense attention computes an `n × n` score matrix, so attention computation generally scales **quadratically in sequence length** for full-sequence attention.
- Long-context access is possible but not equivalent to perfect long-range recall or reasoning.
- Autoregressive output has sequential dependencies and can be latency-sensitive.
- KV caches consume memory as context length and concurrent requests grow.
- Language models can hallucinate; attention does not guarantee factual accuracy.

Engineering optimizations include **FlashAttention**, **KV caching**, **grouped-query attention (GQA)**, **quantization**, and specialized long-context techniques. These address different bottlenecks rather than changing the core next-token objective.

## 20. Common misconceptions

**“The encoder sees both sides, and the decoder sees only the past.”**  
Generally correct for **self-attention** in standard encoder-only and autoregressive decoder architectures. In an encoder–decoder model, decoder **cross-attention** can access the entire encoded source.

**“GPT uses the full original encoder–decoder Transformer.”**  
No. GPT-style models are decoder-only Transformers, without the original separate source encoder and encoder–decoder cross-attention stack.

**“Decoder-only is necessary for sequential generation.”**  
No. Encoder–decoder models also generate target tokens autoregressively.

**“Attention heads are programmed to handle grammar or facts.”**  
No. Their projection weights are learned, and their functions are not necessarily cleanly interpretable.

**“Attention and layers are the same.”**  
No. Multiple heads run in parallel *inside* an attention sublayer. Complete Transformer blocks are stacked *sequentially*.

**“Transformers process every output token simultaneously.”**  
Not for ordinary autoregressive inference. Training can score many positions in parallel; generation chooses the next token before choosing the one after it.

**“An embedding is the final meaning of a word.”**  
No. The initial embedding is a learned token vector. Transformer layers make its representation context-dependent.

**“LayerNorm always happens after the residual.”**  
No. The original design used Post-LN; many modern models use Pre-LN variants or RMSNorm.

**“Every modern AI model is a Transformer.”**  
No. Transformers are central to many modern LLMs, but alternative architectures and hybrid systems also exist.

## 21. Interview-ready explanation

> A Transformer is a neural network architecture introduced in 2017 that uses attention to process sequences. Text is tokenized, mapped to embeddings, and given positional information. Each Transformer block uses multi-head attention to exchange information between token positions, a feed-forward network to transform each position's representation, and residual connections plus normalization to support deep training.
>
> The original Transformer has an encoder with bidirectional self-attention and a decoder with causal self-attention, encoder–decoder cross-attention, and feed-forward layers. Decoder-only models such as GPT use causal self-attention over the prompt and previously generated tokens to predict the next token autoregressively.
>
> Attention supports parallel computation across positions during training and direct long-range interactions, which helped Transformers become a foundation of modern language and multimodal AI.

## 22. Quick revision cheat sheet

| Concept | One-line takeaway |
|---|---|
| Tokenization | Text → vocabulary tokens and IDs |
| Embedding | Token ID → learned numerical vector |
| Position | Encodes token order |
| Q / K / V | Learned projections for matching and mixing information |
| Self-attention | Each position gathers information from permitted positions |
| Multi-head | Multiple learned attention patterns in parallel |
| FFN / MLP | Nonlinear per-position transformation |
| Residual | Adds the original sublayer input back |
| Normalization | Stabilizes representations and training |
| Encoder | Usually bidirectional self-attention |
| Decoder | Causal self-attention over target/generated tokens |
| Cross-attention | Decoder queries the encoder's source representations |
| Transformer block | Attention + FFN + residuals/norm |
| Output head | Hidden state → vocabulary logits |
| Softmax | Logits → probabilities |
| Autoregression | Generate one token, append it, repeat |
| KV cache | Reuse past attention keys/values during decoding |

### My final mental model

```text
TOKENIZATION:     Which token IDs represent the input?
EMBEDDING:        What numerical vectors represent those IDs?
POSITION:         Where are the tokens in the sequence?
ATTENTION:        Which permitted positions should exchange information?
MULTI-HEAD:       How can multiple learned projections capture different patterns?
FFN:              How should each contextual representation be transformed?
RESIDUAL + NORM:  How can many blocks be trained stably?
STACKED BLOCKS:   How do representations become more expressive?
OUTPUT HEAD:      What are the next-token scores?
DECODING:         Which token should be emitted next?
```

**Bottom line:** The Transformer is a repeated system of **context mixing (attention)** and **representation transformation (MLP)**. Its attention mask determines which positions can communicate. The original architecture separates source understanding and target generation; GPT-style models instead use one causal Transformer stack to condition on the prompt and generate the continuation.

## 23. References

- Vaswani et al. (2017), **Attention Is All You Need** — https://arxiv.org/abs/1706.03762
- Devlin et al. (2018), **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** — https://arxiv.org/abs/1810.04805
- Raffel et al. (2019), **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)** — https://arxiv.org/abs/1910.10683
- Brown et al. (2020), **Language Models are Few-Shot Learners (GPT-3)** — https://arxiv.org/abs/2005.14165
- Su et al. (2021), **RoFormer: Enhanced Transformer with Rotary Position Embedding** — https://arxiv.org/abs/2104.09864
- Dao et al. (2022), **FlashAttention** — https://arxiv.org/abs/2205.14135

---

*Personal learning notes on Transformer architecture. Examples, toy vectors, and attention probabilities are illustrative and are not measurements from a particular production model.*
