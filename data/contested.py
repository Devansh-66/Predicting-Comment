import os, sys, time, math, json, random
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
from pathlib import Path
import numpy as np, pandas as pd

T0 = time.time()
BUDGET = float(os.environ.get('BUDGET_S', 68*60))   # hard cap is 90 min for the whole run
FLOOR = 240        # never let training eat the last 4 min: the file must get written
def left(): return BUDGET - (time.time()-T0)
def log(*a): print('[%6.1fs]' % (time.time()-T0), *a, flush=True)

def parse_paths():
    if len(sys.argv) > 2 and not sys.argv[1].startswith('-f'):
        pub, out = Path(sys.argv[1]), Path(sys.argv[2])
    else:
        candidates = ['./dataset/public',
                      '/kaggle/input/datasets/devrajput06/predicting-contested',
                      '/kaggle/input/predicting-contested', '/kaggle/input', '.']
        pub = next((Path(c) for c in candidates if (Path(c) / 'train.csv').exists()), Path('./dataset/public'))
        out = Path('./working/submission.csv')
    os.makedirs(out.parent, exist_ok=True)
    return pub, out

PUB, OUT = parse_paths()
SEED = 42
random.seed(SEED); np.random.seed(SEED)
LBL = ['consensus', 'mixed', 'contested']            # ordinal: rising disagreement
L2I = {l: i for i, l in enumerate(LBL)}
K = 3
MAXLEN = 320
FOLDS = 4
EPOCHS = 2
TOK_BS, TOK_EBS = 4608, 16384   # padded tokens per batch, not rows: bounds peak memory
MAXBS = 64                      # cap rows/batch so short-text batches stay sane
LR, HLR = 3e-5, 1e-3
LLRD = 0.9              # layer-wise LR decay
ORD_EPS = 0.06          # ordinal label smoothing, adjacent grades only
HEAD_TOK = 192          # head tokens kept when head+tail truncating
MIN_ARCH2 = 16*60       # seconds needed before starting a 2nd architecture

train = pd.read_csv(PUB / 'train.csv')
test = pd.read_csv(PUB / 'test.csv')
try:
    samp = pd.read_csv(PUB / 'sample_submission.csv')
except Exception:
    samp = test[['comment_id']].copy()
train['text'] = train.text.fillna('').astype(str)
test['text'] = test.text.fillna('').astype(str)
y = train.contestedness.map(L2I).values.astype(np.int64)
N, NT = len(train), len(test)
PRIOR = np.bincount(y, minlength=K)/N
log('train', N, 'test', NT, 'prior', PRIOR.round(4))

def macro_f1(pred, t):
    s = 0.0
    for k in range(K):
        tp = int(((pred == k) & (t == k)).sum())
        fp = int(((pred == k) & (t != k)).sum())
        fn = int(((pred != k) & (t == k)).sum())
        s += 2*tp/max(2*tp+fp+fn, 1)
    return s/K

def fit_prior_weights(P, target=PRIOR, iters=400):
    """Macro-F1 weights every grade equally, but `mixed` is the middle of an ordinal
    scale and is almost never the argmax -> plain argmax collapses that grade and the
    macro average with it. Rescale the columns until the predicted class distribution
    matches the training prior. Validated locally: 0.380 -> 0.471 on held-out data,
    and unlike a direct macro-F1 search it has nothing to overfit."""
    w = np.ones(K)
    for _ in range(iters):
        d = np.bincount((P*w).argmax(1), minlength=K)/len(P)
        w *= (target/np.maximum(d, 1e-4))**0.25
        w /= w.mean()
    return w

def tune_weights(P, t, iters=6):
    w = np.ones(K); best = macro_f1((P*w).argmax(1), t)
    for _ in range(iters):
        for k in range(K):
            for m in np.exp(np.linspace(-1.0, 1.0, 33)):
                c = w.copy(); c[k] *= m
                s = macro_f1((P*c).argmax(1), t)
                if s > best + 1e-6: best, w = s, c
    return w/w.mean(), best

def decide(P, oof, yv, mask):
    """Pick the decision rule on OOF: argmax, prior-matched, or macro-F1-tuned."""
    cands = {'argmax': np.ones(K),
             'prior': fit_prior_weights(oof[mask]),
             'tuned': tune_weights(oof[mask], yv[mask])[0]}
    sc = {k: macro_f1((oof[mask]*w).argmax(1), yv[mask]) for k, w in cands.items()}
    for k in cands: log('  rule %-6s OOF %.4f  w=%s' % (k, sc[k], cands[k].round(3)))
    # 'tuned' is fitted on the same rows it is scored on, so its OOF number is optimistic;
    # locally it lost ~0.004 out-of-sample while 'prior' held up. Make it earn the switch.
    bk = 'prior' if sc['prior'] >= sc['argmax'] else 'argmax'
    if sc['tuned'] > sc[bk] + 0.004: bk = 'tuned'
    log('  -> using %s (OOF %.4f)' % (bk, sc[bk]))
    return (P*cands[bk]).argmax(1), sc[bk]

def write_sub(pred, tag):
    """Write a schema-valid submission NOW. Called repeatedly; each call overwrites the
    previous one with something better. A killed or crashed run must still leave a
    scoring file behind -- writing only at the end is how a timeout becomes a zero."""
    sub = pd.DataFrame({'comment_id': test.comment_id.values,
                        'label': [LBL[i] for i in pred]})
    sub = samp[['comment_id']].merge(sub, on='comment_id', how='left')
    sub['label'] = sub.label.fillna(LBL[int(PRIOR.argmax())])
    assert len(sub) == len(samp) and sub.comment_id.duplicated().sum() == 0
    assert set(sub.comment_id) == set(samp.comment_id) and sub.label.isin(LBL).all()
    sub.to_csv(OUT, index=False)
    log('wrote %s (%s), %d rows' % (OUT, tag, len(sub)))

# insurance before any model runs: prior-proportional labels, ~chance but never zero
_rs = np.random.RandomState(SEED)
write_sub(_rs.choice(K, size=NT, p=PRIOR), 'prior-random placeholder')

def make_folds(nf):
    fo = np.zeros(N, np.int64)
    order = np.random.RandomState(SEED).permutation(N)
    for k in range(K):                               # stratified
        ik = order[y[order] == k]
        fo[ik] = np.arange(len(ik)) % nf
    return fo
folds = make_folds(FOLDS)

# ---------------- linear baseline (always runs: cheap + the safety net) ----------------
lin_oof = np.full((N, K), np.nan, np.float32); lin_te = np.zeros((NT, K), np.float32)
lin_ok = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from scipy.sparse import hstack
    vw = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200000, sublinear_tf=True)
    vc = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3,
                         max_features=200000, sublinear_tf=True)
    Xa = hstack([vw.fit_transform(train.text), vc.fit_transform(train.text)]).tocsr()
    Xb = hstack([vw.transform(test.text), vc.transform(test.text)]).tocsr()
    for f in range(FOLDS):
        m = folds != f
        clf = LogisticRegression(C=2.0, max_iter=250)   # 466s at 1000; it converges well before
        clf.fit(Xa[m], y[m])
        lin_oof[~m] = clf.predict_proba(Xa[~m]).astype(np.float32)
        lin_te += clf.predict_proba(Xb).astype(np.float32)/FOLDS
    lin_ok = True
    write_sub((lin_te*fit_prior_weights(lin_oof)).argmax(1), 'linear')
    log('linear OOF argmax %.4f | prior-matched %.4f'
        % (macro_f1(lin_oof.argmax(1), y),
           macro_f1((lin_oof*fit_prior_weights(lin_oof)).argmax(1), y)))
except Exception as e:
    log('linear baseline unavailable:', repr(e))

# ---------------- transformer ----------------
# The two DeBERTas share tokenizer/objective/corpus and make correlated errors, so the
# blend's only real diversity came from roberta-BASE -- also its weakest member. Swap in
# a strong diverse model instead. Blend weights are searched on OOF, so if it does not
# earn its place its weight simply goes to 0.
# roberta-large was tested (s3) and scored 0.5641 vs 0.5653 without it -- 0.3 SE, i.e.
# no effect. Reverted to the best-measured ladder. Note s3 also changed the blend search,
# so the two changes were confounded; this keeps the arch set that actually scored best.
ARCHS = [('microsoft/deberta-v3-base', 4, 1.0),      # known-good, runs first
         ('microsoft/deberta-v3-large', 2, 0.5),     # strongest, if time allows
         ('roberta-base', 4, 1.0)]                   # diverse, cheap
ALT = [('microsoft/deberta-v3-base', 4, 1.0), ('microsoft/deberta-v3-small', 4, 1.0),
       ('distilroberta-base', 4, 1.0), ('bert-base-uncased', 4, 1.0)]
res = {}
try:
    import re as _re
    import torch, torch.nn as nn, torch.nn.functional as Fn
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
    BF16 = DEV == 'cuda' and torch.cuda.is_bf16_supported()
    AMPDT = torch.bfloat16 if BF16 else torch.float16
    log('device', DEV, '| bf16', BF16)

    # Ordinal soft targets: consensus < mixed < contested, so landing on the adjacent
    # grade is a smaller error than landing across the scale. Smooth to neighbours only.
    SOFT = np.zeros((K, K), np.float32)
    for k in range(K):
        nb = [j for j in (k-1, k+1) if 0 <= j < K]
        SOFT[k, k] = 1-ORD_EPS
        for j in nb: SOFT[k, j] = ORD_EPS/len(nb)
    SOFT_T = torch.from_numpy(SOFT)

    def run_arch(name, nfold=FOLDS, tok_scale=1.0):
        """Fine-tune `name` over as many folds as fit in the remaining budget.
        Returns (oof, te, folds_done), or None if the encoder will not load."""
        try:
            tokz = AutoTokenizer.from_pretrained(name); AutoConfig.from_pretrained(name)
        except Exception as e:
            log('  cannot load', name, repr(e)[:120]); return None
        PAD = tokz.pad_token_id or 0
        cls_id, sep_id = tokz.cls_token_id, tokz.sep_token_id
        if cls_id is None or sep_id is None:
            cls_id = tokz.bos_token_id if tokz.bos_token_id is not None else PAD
            sep_id = tokz.eos_token_id if tokz.eos_token_id is not None else PAD

        def encode(texts):
            """Head+tail truncation: the closing lines of a long comment carry the
            sign-off tone that plain head truncation throws away."""
            e = tokz(list(texts), add_special_tokens=False, truncation=True,
                     max_length=4096, padding=False)['input_ids']
            body = MAXLEN-2; out = []
            for x in e:
                if len(x) > body: x = list(x[:HEAD_TOK]) + list(x[-(body-HEAD_TOK):])
                out.append(np.asarray([cls_id]+list(x)+[sep_id], np.int64))
            return out
        ids_tr = encode(train.text.values); ids_te = encode(test.text.values)
        ln = np.array([len(x) for x in ids_tr])
        log('%s | tokens med %d p90 %d p99 %d | head+tail on %.1f%%'
            % (name, np.median(ln), np.percentile(ln, 90), np.percentile(ln, 99),
               100*np.mean(ln >= MAXLEN)))

        def batches(ids, budget, shuffle, rs=None):
            """Pack batches to a padded-TOKEN budget, not a row count.

            Length-grouping puts all the longest texts in the same batch, so a fixed
            row count makes peak memory MAXLEN*BS on that one batch -- which is what
            OOMed. A token budget gives long texts small batches and short texts big
            ones, bounding peak memory while keeping throughput up."""
            n = len(ids); idx = np.arange(n)
            if shuffle:
                idx = rs.permutation(n); mega = 4096
                idx = np.concatenate([c[np.argsort([len(ids[i]) for i in c])]
                                      for c in np.array_split(idx, max(1, n//mega))])
            else:
                idx = idx[np.argsort([len(ids[i]) for i in idx])]
            bl = []; cur = []; mx = 0
            for i in idx:
                l = len(ids[i]); nmx = max(mx, l)
                if cur and (nmx*(len(cur)+1) > budget or len(cur) >= MAXBS):
                    bl.append(np.asarray(cur)); cur = [i]; mx = l
                else:
                    cur.append(i); mx = nmx
            if cur: bl.append(np.asarray(cur))
            if shuffle: rs.shuffle(bl)
            return bl

        def collate(ids, b):
            m = max(len(ids[i]) for i in b)
            x = np.full((len(b), m), PAD, np.int64); am = np.zeros((len(b), m), np.int64)
            for j, i in enumerate(b):
                t = ids[i]; x[j, :len(t)] = t; am[j, :len(t)] = 1
            return torch.from_numpy(x), torch.from_numpy(am)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = AutoModel.from_pretrained(name)
                h = self.enc.config.hidden_size
                self.drop = nn.Dropout(0.1)
                self.fc = nn.Linear(2*h, K)              # mean + max pooling
            def forward(self, x, a):
                o = self.enc(input_ids=x, attention_mask=a).last_hidden_state
                am = a.unsqueeze(-1).float()
                mean = (o*am).sum(1)/am.sum(1).clamp(min=1)
                mx = o.masked_fill(am == 0, -1e4).max(1).values
                return self.fc(self.drop(torch.cat([mean, mx], -1)))

        def groups(model):
            """Layer-wise LR decay: lower layers hold general language features that
            53k short comments should barely move; the fresh head needs a big step."""
            nl = model.enc.config.num_hidden_layers
            gs = []
            for n_, p in model.named_parameters():
                if not p.requires_grad: continue
                wd = 0.0 if any(k in n_ for k in ('bias', 'LayerNorm')) else 0.01
                m = _re.search(r'layer\.(\d+)\.', n_)
                if not n_.startswith('enc.'): r = HLR
                elif m is not None: r = lr_enc*(LLRD**(nl-1-int(m.group(1))))
                elif 'embed' in n_: r = lr_enc*(LLRD**nl)
                else: r = lr_enc
                gs.append({'params': [p], 'lr': r, 'weight_decay': wd})
            return gs

        @torch.no_grad()
        def infer(model, ids):
            model.eval(); out = np.zeros((len(ids), K), np.float32)
            for b in batches(ids, TOK_EBS, False):
                x, a = collate(ids, b)
                with torch.autocast('cuda', dtype=AMPDT, enabled=DEV == 'cuda'):
                    z = model(x.to(DEV), a.to(DEV))
                out[b] = torch.softmax(z.float(), -1).cpu().numpy()
            return out

        oof = np.full((N, K), np.nan, np.float32); te = np.zeros((NT, K), np.float32)
        dn = 0; per = None
        st = SOFT_T.to(DEV)
        fo = make_folds(nfold)
        budget = int(TOK_BS*tok_scale)
        # large encoders (roberta-large especially) diverge at the LR that suits a base
        big = 'large' in name or 'xlarge' in name
        lr_enc = LR*0.5 if big else LR
        log('  %s: %d folds, token budget %d' % (name, nfold, budget))
        for f in range(nfold):
            if per is not None and left() < per + 90:
                log('  %s: stopping after %d folds (%.0fs left, %.0fs needed)'
                    % (name, f, left(), per+90)); break
            t1 = time.time()
            tr_i = np.where(fo != f)[0]; va_i = np.where(fo == f)[0]

            def train_fold(budget, ckpt):
                torch.manual_seed(SEED+f); np.random.seed(SEED+f)
                model = Model().to(DEV)
                if ckpt:
                    try: model.enc.gradient_checkpointing_enable()
                    except Exception: pass
                gs = groups(model)
                opt = torch.optim.AdamW(gs)
                rs = np.random.RandomState(SEED+f)
                sub = [ids_tr[i] for i in tr_i]; ysub = y[tr_i]
                plan = [batches(sub, budget, True, rs) for _ in range(EPOCHS)]
                sch = torch.optim.lr_scheduler.OneCycleLR(
                    opt, [g['lr'] for g in gs], total_steps=sum(len(e) for e in plan),
                    pct_start=0.1, anneal_strategy='linear')
                scaler = torch.amp.GradScaler('cuda', enabled=(DEV == 'cuda' and not BF16))
                for bl in plan:
                    model.train()
                    for bi, b in enumerate(bl):
                        if bi % 50 == 0 and left() < FLOOR:
                            raise TimeoutError('out of budget mid-fold')
                        x, a = collate(sub, b)
                        t = st[torch.from_numpy(ysub[b]).to(DEV)]
                        with torch.autocast('cuda', dtype=AMPDT, enabled=DEV == 'cuda'):
                            z = model(x.to(DEV), a.to(DEV))
                            loss = -(t*Fn.log_softmax(z.float(), -1)).sum(-1).mean()
                        opt.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(opt); scaler.update(); sch.step()
                return model, opt

            model = opt = None
            for attempt, (bud, ckpt) in enumerate(((budget, False), (budget//2, False),
                                                   (budget//4, True))):
                try:
                    model, opt = train_fold(bud, ckpt)
                    break
                except TimeoutError:
                    log('  %s fold %d: out of budget mid-fold, abandoning' % (name, f))
                    model = opt = None
                    if DEV == 'cuda': torch.cuda.empty_cache()
                    break
                except torch.cuda.OutOfMemoryError:
                    log('  OOM at token budget %d%s -- retrying smaller'
                        % (bud, ' +checkpointing' if ckpt else ''))
                    model = opt = None
                    import gc; gc.collect()
                    if DEV == 'cuda': torch.cuda.empty_cache()
            if model is None:
                log('  %s fold %d: no usable model, skipping' % (name, f))
                if left() < FLOOR: break
                continue
            oof[va_i] = infer(model, [ids_tr[i] for i in va_i])
            te += infer(model, ids_te); dn += 1
            log('  %s fold %d  %.0fs  val argmax %.4f  prior-matched %.4f'
                % (name, f, time.time()-t1, macro_f1(oof[va_i].argmax(1), y[va_i]),
                   macro_f1((oof[va_i]*fit_prior_weights(oof[va_i])).argmax(1), y[va_i])))
            per = time.time()-t1
            del model, opt
            if DEV == 'cuda': torch.cuda.empty_cache()
        if dn == 0: return None
        te /= dn
        m = ~np.isnan(oof[:, 0])
        log('  %s DONE %d folds | OOF prior-matched %.4f'
            % (name, dn, macro_f1((oof[m]*fit_prior_weights(oof[m])).argmax(1), y[m])))
        return oof, te, dn

    for ai, (nm, nf, tsc) in enumerate(ARCHS):
        if ai and left() < MIN_ARCH2:
            log('skipping %s: only %.0fs left' % (nm, left())); break
        r = run_arch(nm, nf, tsc)
        if r is not None:
            res[nm] = r
            # refresh the on-disk submission after every architecture, so whatever is
            # there is always the best result achieved so far
            try:
                acc = [v[1] for v in res.values()] + ([lin_te] if lin_ok else [])
                om = [v[0] for v in res.values()] + ([lin_oof] if lin_ok else [])
                Pt = sum(acc)/len(acc); Po = sum(om)/len(om)
                mm = ~np.isnan(Po[:, 0])
                write_sub((Pt*fit_prior_weights(Po[mm])).argmax(1), 'after ' + nm)
            except Exception as _e:
                log('  interim write skipped:', repr(_e)[:80])
        elif ai == 0:
            # large did not load (or would not fit): fall back down the size ladder
            for anm, anf, atsc in ALT:
                r = run_arch(anm, anf, atsc)
                if r is not None:
                    res[anm] = r; break
except Exception as e:
    import traceback; traceback.print_exc()
    log('transformer stage failed:', repr(e))

# ---------------- combine + decide ----------------
srcs = [(k, v[0], v[1]) for k, v in res.items()]
if lin_ok: srcs.append(('linear', lin_oof, lin_te))
if not srcs: raise SystemExit('no model produced predictions')
mask = np.ones(N, bool)
for _, o, _t in srcs: mask &= ~np.isnan(o[:, 0])
if mask.sum() < 0.2*N:      # an arch stopped mid-way; keep only fully-covered sources
    log('mask collapsed to %d rows -- dropping partial sources' % mask.sum())
    keep = [(k, o, t) for k, o, t in srcs if np.isnan(o[:, 0]).mean() < 0.05]
    if keep:
        srcs = keep
        mask = np.ones(N, bool)
        for _, o, _t in srcs: mask &= ~np.isnan(o[:, 0])
log('blending %d source(s) on %d OOF rows: %s' % (len(srcs), mask.sum(), [s[0] for s in srcs]))

if len(srcs) == 1:
    bw = np.array([1.0])
elif len(srcs) <= 4:
    # Exact grid, the search that produced the best measured LB (0.5653). 6^4 = 1296
    # combos is affordable; `iters` is cut from 400 because the multiplicative prior
    # update converges long before then.
    bw, bsc = None, -1
    ym = y[mask]; OM = [o[mask] for _, o, _t in srcs]
    for w in np.ndindex(*([6]*len(srcs))):
        if sum(w) == 0: continue
        ws = np.array(w, float); ws /= ws.sum()
        P = sum(x*o for x, o in zip(ws, OM))
        sc = macro_f1((P*fit_prior_weights(P, iters=150)).argmax(1), ym)
        if sc > bsc: bsc, bw = sc, ws
    log('blend weights %s -> OOF %.4f (exact grid)'
        % ({s_[0]: round(float(w), 2) for s_, w in zip(srcs, bw)}, bsc))
else:
    # 6^5 = 7776 combos over 53k rows would exceed the whole budget; approximate.
    rs2 = np.random.RandomState(SEED)
    sub_i = np.where(mask)[0]
    if len(sub_i) > 30000: sub_i = rs2.choice(sub_i, 30000, replace=False)
    ys = y[sub_i]
    OS = np.stack([o[sub_i] for _, o, _t in srcs])
    def bscore(ws):
        P = np.tensordot(ws, OS, 1)
        return macro_f1((P*fit_prior_weights(P, iters=60)).argmax(1), ys)
    k = len(srcs)
    cands = [np.eye(k)[i] for i in range(k)] + [np.ones(k)/k]
    cands += [w/w.sum() for w in rs2.dirichlet(np.ones(k), 400)]
    bw = max(cands, key=bscore); bsc = bscore(bw)
    for _ in range(3):
        for i in range(k):
            for m in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
                c = bw.copy(); c[i] = m
                if c.sum() <= 0: continue
                c = c/c.sum(); sc = bscore(c)
                if sc > bsc + 1e-5: bsc, bw = sc, c
    log('blend weights %s -> OOF %.4f (subsampled search)'
        % ({s_[0]: round(float(w), 2) for s_, w in zip(srcs, bw)}, bsc))

OOF = sum(w*o for w, (_, o, _t) in zip(bw, srcs))
TE = sum(w*t for w, (_, _o, t) in zip(bw, srcs))

pred, oof_score = decide(TE, OOF, y, mask)
log('FINAL OOF macro-F1 %.4f' % oof_score)
log('test dist %s | train prior %s'
    % ((np.bincount(pred, minlength=K)/NT).round(4), PRIOR.round(4)))

write_sub(pred, 'final')
