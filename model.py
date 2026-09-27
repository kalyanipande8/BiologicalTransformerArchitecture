"""
model.py -- Custom Transformer encoder with SPECIALIZED attention heads,
implemented from scratch in raw numpy (no autodiff / no PyTorch -- see
README for why: this machine had < 1GB free disk and no torch installed).

DESIGN RATIONALE ("specialized attention heads", not vanilla uniform MHA)
--------------------------------------------------------------------------
Real secondary-structure-relevant signal in a residue sequence lives at two
very different scales:
  * LOCAL patterns (e.g. a helical turn, a beta-strand pairing) that only
    involve residues a few positions apart.
  * GLOBAL / positional patterns (a motif recurring at specific anchor
    positions, or interactions between residues far apart in the sequence).

A single undifferentiated multi-head attention block *can* in principle
learn either regime, but with a handful of training sequences and no GPU
budget, it will not reliably discover this separation on its own. So this
encoder HARD-WIRES the receptive field of each head via a fixed additive
attention mask, giving each head an explicit, documented job:

  head 0 (narrow-local, window=1): can only attend to i-1, i, i+1.
      Rationale: captures tight local motifs like "HHH"/"EE" pairs.
  head 1 (wide-local, window=4): attends to i-4 .. i+4.
      Rationale: captures a full turn/short-strand neighborhood.
  head 2 (motif-anchored global): attends ONLY to a small fixed set of
      anchor positions (0, 10, 20, 30) plus itself, regardless of i.
      Rationale: many real motifs recur at roughly periodic positions;
      this head is a fixed "lookup" over those candidate anchor slots.
  head 3 (full global, unmasked): standard full self-attention.
      Rationale: a fallback / long-range channel, the "vanilla" case.

This is a genuine architectural choice (fixed structural priors baked into
the attention pattern, not learned) -- a lightweight, CPU-only analogue of
techniques like Longformer/BigBird's mixed local+global attention, adapted
here at toy scale for biological sequences.

TRAINABLE vs FIXED PARTS (documented simplification)
--------------------------------------------------------------------------
The token embedding table and the Q/K/V/O projection matrices for every
head are FIXED (structured random init, not trained by gradient descent):
training a from-scratch attention encoder end-to-end by hand-derived
backprop through softmax attention was judged not worth the added
complexity/fragility at this scale. Instead, following the task spec's own
documented fallback ("...or use scikit-learn's MLPClassifier over
attention-pooled features as a documented simplification"), this encoder
is used purely as a FORWARD-PASS feature extractor: raw sequence -> custom
specialized-head self-attention -> mean-pooled contextual embedding. All
of the actual *learning* happens downstream, in the small classifier heads
trained on top of these pooled features (see train_primary_task.py /
evaluate_downstream_tasks.py). This is analogous to a randomly-initialized
but architecturally-structured "reservoir" encoder -- the structure (which
positions each head can see) is the hand-designed part; the classifier
weights are the trained part.
"""
import numpy as np

D_MODEL = 16
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS
ANCHORS = [0, 10, 20, 30]
SEED = 7


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _sinusoidal_positions(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe


def _build_head_masks(seq_len):
    """Additive attention masks (0 = allowed, -inf = blocked) per head."""
    idx = np.arange(seq_len)
    i, j = np.meshgrid(idx, idx, indexing="ij")

    def band_mask(window):
        allowed = np.abs(i - j) <= window
        m = np.where(allowed, 0.0, -1e9)
        return m

    narrow = band_mask(1)
    wide = band_mask(4)

    anchor_allowed = np.isin(j, ANCHORS) | (i == j)
    anchor_mask = np.where(anchor_allowed, 0.0, -1e9)

    full = np.zeros((seq_len, seq_len))
    return [narrow, wide, anchor_mask, full]


class SpecializedAttentionEncoder:
    """Custom Transformer encoder with 4 heads of distinct receptive fields."""

    def __init__(self, vocab, seq_len, d_model=D_MODEL, n_heads=N_HEADS, seed=SEED):
        assert d_model % n_heads == 0
        self.vocab = vocab
        self.v2i = {c: idx for idx, c in enumerate(vocab)}
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d_model)
        self.embed = rng.normal(0, scale, size=(len(vocab), d_model))
        self.pos_enc = _sinusoidal_positions(seq_len, d_model)
        self.head_masks = _build_head_masks(seq_len)

        self.Wq = [rng.normal(0, scale, (d_model, self.d_head)) for _ in range(n_heads)]
        self.Wk = [rng.normal(0, scale, (d_model, self.d_head)) for _ in range(n_heads)]
        self.Wv = [rng.normal(0, scale, (d_model, self.d_head)) for _ in range(n_heads)]
        self.Wo = rng.normal(0, scale, (d_model, d_model))

    def _tokenize(self, seq):
        idxs = [self.v2i[c] for c in seq[: self.seq_len]]
        while len(idxs) < self.seq_len:
            idxs.append(0)
        return np.array(idxs)

    def encode_one(self, seq, return_attn=False):
        """Forward pass for a single sequence -> pooled (d_model,) vector."""
        idxs = self._tokenize(seq)
        x = self.embed[idxs] + self.pos_enc  # (seq_len, d_model)

        head_outputs = []
        attn_maps = []
        for h in range(self.n_heads):
            Q = x @ self.Wq[h]
            K = x @ self.Wk[h]
            V = x @ self.Wv[h]
            scores = (Q @ K.T) / np.sqrt(self.d_head) + self.head_masks[h]
            attn = _softmax(scores, axis=-1)
            head_outputs.append(attn @ V)
            attn_maps.append(attn)

        concat = np.concatenate(head_outputs, axis=-1)  # (seq_len, d_model)
        out = concat @ self.Wo
        out = out + x  # residual connection
        pooled = out.mean(axis=0)  # mean-pool over sequence -> (d_model,)

        if return_attn:
            return pooled, attn_maps
        return pooled

    def encode_batch(self, seqs):
        return np.stack([self.encode_one(s) for s in seqs], axis=0)


HEAD_NAMES = [
    "narrow_local (window=1)",
    "wide_local (window=4)",
    "motif_anchored_global (fixed anchors)",
    "full_global (unmasked)",
]
