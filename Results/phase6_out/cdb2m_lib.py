# cdb2m_lib.py - shared code for CD-B2M (Phase 3 onward). Written by 03_encoder_splits.
import numpy as np, pandas as pd

CORE  = ["hippocampus", "amygdala", "parahippocampal", "lateral_temporal"]
BANDS = ["theta", "alpha", "beta", "lgamma", "hgamma"]
BIN_S = 0.5
EPOCH = {"encoding": (0, 2), "maintenance": (2, 5), "probe": (5, 8)}
PAD, SEP, MAX_LEN = 0, 1, 12          # token ids and max memory-set length


# ---------------------------------------------------------------- brain features
def win_slice(name):
    lo, hi = EPOCH[name]
    return slice(int(lo / BIN_S), int(hi / BIN_S))

def region_present(X):
    "(N, R) bool: region has data in that trial's session"
    return ~np.isnan(X).all(axis=(2, 3))

def window_features(X, window="maintenance", regions=None):
    "X (N,R,T,B) -> (N,F). Window-mean features; NaN -> 0 (data are z-scored, 0 = session mean)."
    ridx = [CORE.index(r) for r in (regions or CORE)]
    Xr = X[:, ridx]
    if window == "full":
        F = Xr.reshape(len(X), -1)
    elif window == "all3":
        F = np.concatenate([Xr[:, :, win_slice(w)].mean(2).reshape(len(X), -1) for w in EPOCH], 1)
    else:
        F = Xr[:, :, win_slice(window)].mean(2).reshape(len(X), -1)
    return np.nan_to_num(F, nan=0.0).astype(np.float32)


# ---------------------------------------------------------------- splits
def make_folds(M, n_folds=5, seed=42):
    """Subject-grouped folds, balanced over acquisition group (2000 Hz vs 4000/4096 Hz).
    Per fold: train / val on loads {4,6}; test_id = new subjects, loads {4,6};
    test_ood = new subjects, load 8; seen_ood = training subjects, load 8."""
    subj = M.groupby("subject").agg(fs=("fs", "first")).reset_index()
    subj["grp"] = (subj.fs >= 3000).astype(int)
    grp_of = dict(zip(subj.subject, subj.grp))
    rng = np.random.default_rng(seed)
    fold_of, off = {}, 0
    for g, d in subj.groupby("grp"):
        s = np.array(d.subject.tolist()); rng.shuffle(s)
        for i, name in enumerate(s):
            fold_of[name] = (i + off) % n_folds
        off += len(s)
    clean = M.clean.values.astype(bool); load = M.SetSize.values; sub = np.array(M.subject.tolist())
    pick = lambda subs, loads: np.where(clean & np.isin(sub, subs) & np.isin(load, loads))[0]
    folds = []
    for f in range(n_folds):
        test_s = sorted(s for s, k in fold_of.items() if k == f)
        rest = sorted(s for s in fold_of if s not in test_s)
        val_s = []
        for g in (0, 1):                       # one validation subject per acquisition group
            cand = [s for s in rest if grp_of[s] == g]
            if cand: val_s.append(cand[int(rng.integers(len(cand)))])
        train_s = [s for s in rest if s not in val_s]
        folds.append(dict(fold=f, test_subjects=test_s, val_subjects=sorted(val_s), train_subjects=train_s,
                          train=pick(train_s, [4, 6]), val=pick(val_s, [4, 6]), val_ood=pick(val_s, [8]),
                          test_id=pick(test_s, [4, 6]), test_ood=pick(test_s, [8]), seen_ood=pick(train_s, [8])))
    return folds


# ---------------------------------------------------------------- stimuli (surrogate memory sets)
def letter_tokens(alphabet):
    return {ch: 2 + i for i, ch in enumerate(sorted(alphabet))}

def split_alphabet(alphabet, n_unseen=6, seed=0):
    a = np.array(sorted(alphabet)); np.random.default_rng(seed).shuffle(a)
    return sorted(a[n_unseen:].tolist()), sorted(a[:n_unseen].tolist())

def make_sequence(rng, size, pool, match, max_len=MAX_LEN):
    pool = np.asarray(pool)
    assert size + 1 <= len(pool), f"pool of {len(pool)} letters too small for set size {size}"
    if match:
        mem = rng.choice(pool, size, replace=False); probe = int(rng.choice(mem))
    else:
        ch = rng.choice(pool, size + 1, replace=False); mem, probe = ch[:size], int(ch[-1])
    out = np.zeros(max_len, np.int64); out[:size] = mem
    return out, probe

def to_input(mem, probe):
    "token layout: [memory letters (PAD-padded to max_len)..., SEP, probe]"
    return np.concatenate([mem, np.full((len(mem), 1), SEP, np.int64), probe[:, None]], 1).astype(np.int64)

def build_real_stimuli(M, pool, seed=0, max_len=MAX_LEN):
    "One surrogate memory set per real trial: SetSize letters, contains the probe iff Match == IN."
    N = len(M); mem = np.zeros((N, max_len), np.int64); probe = np.zeros(N, np.int64)
    y = (np.array(M.Match.tolist()) == "IN").astype(np.int64)
    for i in range(N):
        s = int(M.SetSize.iat[i])
        if s <= 0: continue
        mem[i], probe[i] = make_sequence(np.random.default_rng([seed, i]), s, pool, bool(y[i]), max_len)
    return to_input(mem, probe), y

def make_synth(n, sizes, pool, seed, max_len=MAX_LEN):
    rng = np.random.default_rng(seed)
    sz = rng.choice(sizes, n); y = np.resize([0, 1], n); rng.shuffle(y)
    mem = np.zeros((n, max_len), np.int64); probe = np.zeros(n, np.int64)
    for i in range(n):
        mem[i], probe[i] = make_sequence(rng, int(sz[i]), pool, bool(y[i]), max_len)
    return to_input(mem, probe), y.astype(np.int64), sz

def make_suite(train_pool, unseen_pool, n=1000, seed=123):
    "Evaluation-only synthetic sets (no brain data needed at test time)."
    return {"id_4_6": make_synth(n, [4, 6], train_pool, seed + 1),
            "size_8": make_synth(n, [8], train_pool, seed + 2),
            "len_10": make_synth(n, [10], train_pool, seed + 3),
            "len_12": make_synth(n, [12], train_pool, seed + 4),
            "unseen_letters": make_synth(n, [4, 5], unseen_pool, seed + 5)}


# ---------------------------------------------------------------- stats helper
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(1.0, run)
    return adj


# ---------------------------------------------------------------- torch parts
try:
    import torch, torch.nn as nn, torch.nn.functional as F
except ImportError:                                   # lets the numpy parts import without torch
    torch = None

if torch is not None:
    class BrainEncoder(nn.Module):
        """Shared encoder for real AND control brain data.
        Input  (B, 4 regions, 16 time bins, 5 bands), NaN = region not recorded in that session.
        Output (B, d_out). Missing regions are masked out of the pooling (and the mask itself is an input)."""
        def __init__(self, n_reg=4, n_t=16, n_band=5, d_hid=64, d_out=32, p_drop=0.2):
            super().__init__()
            self.reg = nn.ModuleList([nn.Sequential(nn.Linear(n_t * n_band, d_hid), nn.GELU(), nn.Dropout(p_drop))
                                      for _ in range(n_reg)])
            self.head = nn.Sequential(nn.Linear(d_hid + n_reg, d_hid), nn.GELU(), nn.Linear(d_hid, d_out))
        def forward(self, x):
            mask = (~torch.isnan(x).all(dim=3).all(dim=2)).float()          # (B, R)
            x = torch.nan_to_num(x, nan=0.0)
            h = torch.stack([m(x[:, k].flatten(1)) for k, m in enumerate(self.reg)], 1)   # (B, R, H)
            w = mask.unsqueeze(-1)
            pooled = (h * w).sum(1) / w.sum(1).clamp(min=1.0)
            return self.head(torch.cat([pooled, mask], 1))

    class SeqGRU(nn.Module):
        "Task model for the ceiling check. rep = final hidden state (the 'model representation' m_i)."
        def __init__(self, n_tokens, d_emb=32, d_hid=128):
            super().__init__()
            self.emb = nn.Embedding(n_tokens, d_emb, padding_idx=PAD)
            self.gru = nn.GRU(d_emb, d_hid, batch_first=True)
            self.out = nn.Linear(d_hid, 2)
        def forward(self, x, return_rep=False):
            h = self.gru(self.emb(x))[1][-1]
            return (self.out(h), h) if return_rep else self.out(h)

    def train_gru(Xtr, ytr, n_tokens, seed, steps=3000, bs=64, lr=3e-3, device="cpu"):
        torch.manual_seed(seed); np.random.seed(seed)
        model = SeqGRU(n_tokens).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)       # no weight decay: unseen-letter embeddings stay at init
        Xt = torch.as_tensor(Xtr, device=device); yt = torch.as_tensor(ytr, device=device)
        for _ in range(steps):
            b = torch.randint(0, len(Xt), (bs,), device=device)
            loss = F.cross_entropy(model(Xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        return model

    @torch.no_grad()
    def gru_accuracy(model, X, y, device="cpu"):
        model.eval()
        pred = model(torch.as_tensor(X, device=device)).argmax(1).cpu().numpy()
        return float((pred == y).mean())
