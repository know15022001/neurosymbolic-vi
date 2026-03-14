"""
Baseline Models — Generate từ seed câu hỏi
N-gram: trigram + backoff bigram/unigram
LSTM: word-level, numpy only
"""

import numpy as np
import re
import time
import math
from collections import defaultdict, Counter


class NGramModel:
    """Trigram với backoff → bigram → unigram. Seed = token cuối câu hỏi."""
    name = "N-Gram (trigram+backoff)"

    def __init__(self):
        self.trigrams = defaultdict(Counter)
        self.bigrams  = defaultdict(Counter)
        self.unigrams = Counter()
        self.vocab    = []

    def train(self, sentences: list):
        for sent in sentences:
            toks = self._tok(sent)
            self.unigrams.update(toks)
            padded = ['<S>', '<S>'] + toks + ['</S>']
            for i in range(len(padded) - 2):
                self.trigrams[(padded[i], padded[i+1])][padded[i+2]] += 1
                self.bigrams [(padded[i+1],)           ][padded[i+2]] += 1
        self.vocab = [w for w in self.unigrams
                      if w not in ('<S>', '</S>')]

    def _next(self, history: list) -> str:
        if len(history) >= 2:
            ctx = tuple(history[-2:])
            if self.trigrams[ctx]:
                return max(self.trigrams[ctx], key=self.trigrams[ctx].get)
        if history:
            ctx = (history[-1],)
            if self.bigrams[ctx]:
                return max(self.bigrams[ctx], key=self.bigrams[ctx].get)
        return self.unigrams.most_common(1)[0][0] if self.unigrams else '</S>'

    def answer_query(self, query: str) -> dict:
        t0 = time.perf_counter()
        q_toks = self._tok(query)
        stop   = {'gì','không','đâu','sao','nào'}
        seed   = [t for t in q_toks if t not in stop] or q_toks[-2:]

        history   = list(seed)
        generated = []
        for _ in range(15):
            nxt = self._next(history)
            if nxt in ('</S>','<S>'):
                break
            if len(generated) >= 3 and generated[-3:] == [nxt]*3:
                break
            generated.append(nxt)
            history.append(nxt)

        answer = ' '.join(generated).strip()
        if answer:
            answer = answer[0].upper() + answer[1:]
            if not answer.endswith('.'):
                answer += '.'
        elapsed = (time.perf_counter() - t0) * 1000
        return {'answer': answer or '[không generate được]',
                'confidence': 0.4,
                'inference_time_ms': round(elapsed, 3),
                'hallucination_risk': 'HIGH',
                'seed': ' '.join(seed)}

    def _tok(self, text: str) -> list:
        return [t for t in re.sub(r'[^\w\s]', ' ', text.lower()).split() if t]

    def memory_bytes(self):
        return (sum(len(v)*80 for v in self.trigrams.values()) +
                sum(len(v)*60 for v in self.bigrams.values()))


class TinyLSTM:
    """Word-level LSTM, numpy only. Generate từ seed = tokens câu hỏi."""
    name = "Tiny LSTM (word-level, numpy)"

    def __init__(self, hidden_size: int = 64):
        self.H = hidden_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.V = 0
        self.Wi = self.Wf = self.Wo = self.Wg = self.Wy = None
        self.trained = False
        self.train_loss = []

    def _init(self, V):
        self.V = V; H = self.H
        sc = np.sqrt(2.0 / (V + H))
        self.Wi = np.random.randn(H, V+H) * sc
        self.Wf = np.random.randn(H, V+H) * sc
        self.Wo = np.random.randn(H, V+H) * sc
        self.Wg = np.random.randn(H, V+H) * sc
        self.bi = np.zeros((H,1)); self.bf = np.ones((H,1))*0.5
        self.bo = np.zeros((H,1)); self.bg = np.zeros((H,1))
        self.Wy = np.random.randn(V, H) * np.sqrt(1.0/H)
        self.by = np.zeros((V,1))

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -8, 8)))

    def _step(self, idx: int, h, c):
        x = np.zeros((self.V, 1))
        if 0 <= idx < self.V: x[idx] = 1.0
        xh = np.vstack([x, h])
        i_ = self._sigmoid(self.Wi@xh + self.bi)
        f_ = self._sigmoid(self.Wf@xh + self.bf)
        o_ = self._sigmoid(self.Wo@xh + self.bo)
        g_ = np.tanh(self.Wg@xh + self.bg)
        c2 = f_*c + i_*g_; h2 = o_*np.tanh(c2)
        y  = self.Wy@h2 + self.by
        e  = np.exp(y - y.max()); p = e/e.sum()
        return p, h2, c2

    def train(self, sentences: list, epochs: int = 5, lr: float = 0.02):
        all_toks = []
        for s in sentences:
            all_toks.extend(self._tok(s))
        vocab = ['<UNK>'] + sorted(set(all_toks))
        self.word_to_idx = {w:i for i,w in enumerate(vocab)}
        self.idx_to_word = {i:w for i,w in enumerate(vocab)}
        self._init(len(vocab))

        seqs = []
        for s in sentences:
            idxs = [self.word_to_idx.get(t,0) for t in self._tok(s)]
            if len(idxs) >= 2: seqs.append(idxs)

        chunk = 8
        for epoch in range(epochs):
            total_loss = 0.0; n = 0
            np.random.shuffle(seqs)
            for seq in seqs:
                h = np.zeros((self.H,1)); c = np.zeros((self.H,1))
                for start in range(0, len(seq)-1, chunk):
                    inp = seq[start:start+chunk]
                    tgt = seq[start+1:start+chunk+1]
                    if not tgt: continue
                    hs, probs = [h], []; loss = 0.0
                    for xi, yi in zip(inp, tgt):
                        p, h, c = self._step(xi, h, c)
                        probs.append(p); hs.append(h)
                        loss -= math.log(float(p.flatten()[yi]) + 1e-9)
                    total_loss += loss/max(len(tgt),1); n += 1
                    dWy = np.zeros_like(self.Wy); dby = np.zeros_like(self.by)
                    for t, yi in enumerate(tgt):
                        dp = probs[t].copy(); dp[yi] -= 1.0
                        dWy += dp @ hs[t+1].T; dby += dp
                    np.clip(dWy,-5,5,out=dWy); np.clip(dby,-5,5,out=dby)
                    self.Wy -= lr*dWy; self.by -= lr*dby
            self.train_loss.append(total_loss/max(n,1))
        self.trained = True

    def generate(self, seed_tokens: list, max_len=15, temperature=0.8) -> str:
        h = np.zeros((self.H,1)); c = np.zeros((self.H,1))
        seed_idxs = [self.word_to_idx.get(t,0) for t in seed_tokens]
        for idx in seed_idxs:
            _, h, c = self._step(idx, h, c)

        generated = []; cur_idx = seed_idxs[-1] if seed_idxs else 0
        for _ in range(max_len):
            p, h, c = self._step(cur_idx, h, c)
            p_flat = p.flatten()
            lp = np.log(p_flat+1e-9)/max(temperature,0.1)
            lp -= lp.max(); ps = np.exp(lp); ps /= ps.sum()
            top5 = np.argsort(ps)[-5:]; tp = ps[top5]; tp /= tp.sum()
            cur_idx = int(np.random.choice(top5, p=tp))
            word = self.idx_to_word.get(cur_idx, '<UNK>')
            if word == '<UNK>': continue
            if len(generated) >= 4 and generated[-4:] == [word]*4: break
            generated.append(word)
        return ' '.join(generated)

    def answer_query(self, query: str) -> dict:
        t0 = time.perf_counter()
        if not self.trained:
            return {'answer':'[chưa train]','confidence':0.0,
                    'inference_time_ms':0.0,'hallucination_risk':'HIGH'}
        stop  = {'gì','không','đâu','sao','nào','có'}
        q_toks = self._tok(query)
        seed   = [t for t in q_toks if t not in stop] or q_toks

        gen = self.generate(seed, max_len=15, temperature=0.7)
        elapsed = (time.perf_counter()-t0)*1000
        answer = gen.strip()
        if answer:
            answer = answer[0].upper()+answer[1:]
            if not answer.endswith('.'): answer += '.'
        return {'answer': answer or '[không generate được]',
                'confidence': 0.35,
                'inference_time_ms': round(elapsed,3),
                'hallucination_risk': 'HIGH',
                'seed': ' '.join(seed)}

    def _tok(self, text: str) -> list:
        return [t for t in re.sub(r'[^\w\s]',' ',text.lower()).split() if t]

    def memory_bytes(self):
        if self.Wy is None: return 0
        return sum(getattr(self,a).nbytes
                   for a in ['Wi','Wf','Wo','Wg','Wy','bi','bf','bo','bg','by'])

    @property
    def n_params(self):
        if self.Wy is None: return 0
        return sum(getattr(self,a).size for a in ['Wi','Wf','Wo','Wg','Wy'])
