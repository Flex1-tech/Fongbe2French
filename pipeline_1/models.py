"""Architectures A2 (BiLSTM+attention), A3 (BiGRU+attention), A6 (encodeur CNN + décodeur LSTM+attention).
Le décodeur et l'attention de Bahdanau sont identiques ; on change d'architecture avec `arch`.
Hyperparamètres communs lus dans config.yaml (section model)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD, UNK, SOS, EOS = 0, 1, 2, 3


class RNNEncoder(nn.Module):
    """Encodeur bidirectionnel 1 couche (LSTM pour A2, GRU pour A3)."""
    def __init__(self, vocab, emb, hid, dropout, cell):
        super().__init__()
        self.cell = cell
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.drop = nn.Dropout(dropout)
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(emb, hid, batch_first=True, bidirectional=True)
        self.bridge_h = nn.Linear(2 * hid, hid)
        self.bridge_c = nn.Linear(2 * hid, hid) if cell == "lstm" else None

    def forward(self, src, src_len):
        x = self.drop(self.emb(src))
        packed = pack_padded_sequence(x, src_len.cpu(), batch_first=True, enforce_sorted=False)
        out, hn = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        if self.cell == "lstm":
            h, c = hn
            h = torch.cat([h[0], h[1]], -1); c = torch.cat([c[0], c[1]], -1)
            return out, (torch.tanh(self.bridge_h(h)), torch.tanh(self.bridge_c(c)))
        h = torch.cat([hn[0], hn[1]], -1)
        return out, (torch.tanh(self.bridge_h(h)), None)


class DeepEncoder(nn.Module):
    """A4 : BiLSTM a 2 couches (dropout entre les couches), un 'pont' par couche pour initialiser le decodeur."""
    def __init__(self, vocab, emb, hid, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb, hid, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.bridge_h = nn.ModuleList([nn.Linear(2 * hid, hid) for _ in range(2)])
        self.bridge_c = nn.ModuleList([nn.Linear(2 * hid, hid) for _ in range(2)])

    def forward(self, src, src_len):
        x = self.drop(self.emb(src))
        packed = pack_padded_sequence(x, src_len.cpu(), batch_first=True, enforce_sorted=False)
        out, (h, c) = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        # h, c : [4, B, hid] = (couche0 avant, couche0 arriere, couche1 avant, couche1 arriere)
        hs = [torch.tanh(self.bridge_h[l](torch.cat([h[2 * l], h[2 * l + 1]], -1))) for l in range(2)]
        cs = [torch.tanh(self.bridge_c[l](torch.cat([c[2 * l], c[2 * l + 1]], -1))) for l in range(2)]
        return out, (torch.stack(hs, 1), torch.stack(cs, 1))     # etats du decodeur : [B, 2, hid]


class CNNEncoder(nn.Module):
    """A6 : embeddings de mots + positions apprises, pile de convolutions GLU résiduelles."""
    def __init__(self, vocab, emb, hid, dropout, layers=3, kernel=3, max_pos=256):
        super().__init__()
        d = 2 * hid                                   # même dimension de sortie que l'encodeur BiRNN
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.pos = nn.Embedding(max_pos, emb)
        self.max_pos = max_pos
        self.proj = nn.Linear(emb, d)
        self.convs = nn.ModuleList([nn.Conv1d(d, 2 * d, kernel, padding=kernel // 2) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.bridge_h = nn.Linear(d, hid)
        self.bridge_c = nn.Linear(d, hid)

    def forward(self, src, src_len):
        B, S = src.shape
        mask = (src != PAD).unsqueeze(-1).float()
        pos = torch.arange(S, device=src.device).clamp(max=self.max_pos - 1).unsqueeze(0)
        x = self.drop(self.emb(src) + self.pos(pos))
        x = self.proj(x) * mask
        for conv in self.convs:
            y = F.glu(conv(self.drop(x).transpose(1, 2)), dim=1).transpose(1, 2)
            x = (x + y) * math.sqrt(0.5) * mask
        mean = (x * mask).sum(1) / mask.sum(1).clamp(min=1)    # moyenne des états de l'encodeur
        return x, (torch.tanh(self.bridge_h(mean)), torch.tanh(self.bridge_c(mean)))


class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, att_dim):
        super().__init__()
        self.We = nn.Linear(enc_dim, att_dim, bias=False)
        self.Ws = nn.Linear(dec_dim, att_dim)
        self.v = nn.Linear(att_dim, 1, bias=False)

    def forward(self, s, enc_proj, enc_out, mask):
        e = self.v(torch.tanh(enc_proj + self.Ws(s).unsqueeze(1))).squeeze(-1)
        e = e.masked_fill(~mask, -1e9)                         # masque sur le padding
        a = F.softmax(e, -1)
        return torch.bmm(a.unsqueeze(1), enc_out).squeeze(1), a


class Seq2Seq(nn.Module):
    def __init__(self, arch, src_vocab, tgt_vocab, mcfg):
        super().__init__()
        emb, hid, drop = mcfg["emb_dim"], mcfg["hid_dim"], mcfg["dropout"]
        self.arch, self.hid, self.tgt_vocab = arch, hid, tgt_vocab
        self.cell = "gru" if arch == "a3" else "lstm"          # cellule du décodeur
        if arch == "a2":   self.encoder = RNNEncoder(src_vocab, emb, hid, drop, "lstm")
        elif arch == "a3": self.encoder = RNNEncoder(src_vocab, emb, hid, drop, "gru")
        elif arch == "a4": self.encoder = DeepEncoder(src_vocab, emb, hid, drop)
        elif arch == "a6": self.encoder = CNNEncoder(src_vocab, emb, hid, drop, mcfg.get("cnn_layers", 3), mcfg.get("cnn_kernel", 3))
        else: raise ValueError(f"architecture inconnue : {arch} (a2, a3, a4, a6)")
        enc_dim = 2 * hid
        self.enc_dim = enc_dim
        self.emb = nn.Embedding(tgt_vocab, emb, padding_idx=PAD)
        self.drop = nn.Dropout(drop)
        self.att = BahdanauAttention(enc_dim, hid, hid)
        Cell = nn.GRUCell if self.cell == "gru" else nn.LSTMCell
        self.dec = Cell(emb + enc_dim, hid)
        self.deep = arch == "a4"
        if self.deep: self.dec2 = nn.LSTMCell(hid, hid)      # 2e couche du decodeur (A4)
        self.out = nn.Linear(hid + enc_dim + emb, tgt_vocab)

    # ---- encodage / un pas de décodage -------------------------------------------------
    def encode(self, src, src_len):
        enc_out, (h, c) = self.encoder(src, src_len)
        if c is None: c = torch.zeros_like(h)
        mask = src != PAD
        return {"enc": enc_out, "proj": self.att.We(enc_out), "mask": mask}, (h, c)

    def step(self, prev, state, mem):
        h, c = state
        e = self.drop(self.emb(prev))
        if self.deep: return self._step_deep(e, h, c, mem)
        ctx, a = self.att(h, mem["proj"], mem["enc"], mem["mask"])
        inp = torch.cat([e, ctx], -1)
        if self.cell == "gru":
            h = self.dec(inp, h)
        else:
            h, c = self.dec(inp, (h, c))
        logits = self.out(self.drop(torch.cat([h, ctx, e], -1)))
        return logits, (h, c), a

    def _step_deep(self, e, h, c, mem):
        """A4 : decodeur a 2 couches. h, c : [B, 2, hid]. Residuel autour de la 2e couche."""
        ctx, a = self.att(h[:, -1], mem["proj"], mem["enc"], mem["mask"])    # attention sur l'etat de la couche du haut
        h1, c1 = self.dec(torch.cat([e, ctx], -1), (h[:, 0], c[:, 0]))
        h2, c2 = self.dec2(self.drop(h1), (h[:, 1], c[:, 1]))              # dropout entre les couches
        top = h2 + h1                                                      # connexion residuelle
        logits = self.out(self.drop(torch.cat([top, ctx, e], -1)))
        return logits, (torch.stack([h1, h2], 1), torch.stack([c1, c2], 1)), a

    def forward(self, src, src_len, tgt_in, teacher_forcing=1.0):
        """tgt_in : <SOS> y1 ... yn ; renvoie des logits [B, T, V]."""
        mem, state = self.encode(src, src_len)
        outs, prev = [], tgt_in[:, 0]
        for t in range(tgt_in.size(1)):
            logits, state, _ = self.step(prev, state, mem)
            outs.append(logits)
            if t + 1 < tgt_in.size(1):
                use_gold = torch.rand(prev.size(0), device=prev.device) < teacher_forcing
                prev = torch.where(use_gold, tgt_in[:, t + 1], logits.argmax(-1))
        return torch.stack(outs, 1)

    # ---- décodage ----------------------------------------------------------------------
    @torch.no_grad()
    def greedy(self, src, src_len, max_len=100):
        mem, state = self.encode(src, src_len)
        B = src.size(0)
        prev = torch.full((B,), SOS, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        out = []
        for _ in range(max_len):
            logits, state, _ = self.step(prev, state, mem)
            logits[:, PAD] = -1e9; logits[:, SOS] = -1e9
            prev = logits.argmax(-1).masked_fill(done, PAD)
            out.append(prev); done |= prev == EOS
            if done.all(): break
        return self._strip(torch.stack(out, 1))

    @torch.no_grad()
    def beam(self, src, src_len, beam=4, max_len=100):
        """Beam search groupé (B phrases x K hypothèses), normalisation par la longueur."""
        mem, (h, c) = self.encode(src, src_len)
        B, K, dev = src.size(0), beam, src.device
        rep = lambda x: x.repeat_interleave(K, 0)
        mem = {k: rep(v) for k, v in mem.items()}
        state = (rep(h), rep(c))
        scores = torch.full((B, K), -1e9, device=dev); scores[:, 0] = 0
        seqs = torch.zeros(B, K, 0, dtype=torch.long, device=dev)
        lens = torch.zeros(B, K, device=dev)
        fin = torch.zeros(B, K, dtype=torch.bool, device=dev)
        prev = torch.full((B * K,), SOS, dtype=torch.long, device=dev)
        for _ in range(max_len):
            logits, state, _ = self.step(prev, state, mem)
            logits[:, PAD] = -1e9; logits[:, SOS] = -1e9
            logp = F.log_softmax(logits, -1).view(B, K, -1)
            V = logp.size(-1)
            # une hypothèse terminée ne peut plus qu'émettre <PAD> sans changer son score
            frozen = torch.full((V,), -1e9, device=dev); frozen[PAD] = 0
            logp = torch.where(fin.unsqueeze(-1), frozen.expand(B, K, V), logp)
            cand = (scores.unsqueeze(-1) + logp).view(B, K * V)
            scores, idx = cand.topk(K, -1)
            bi, tok = idx // V, idx % V
            seqs = torch.cat([seqs.gather(1, bi.unsqueeze(-1).expand(-1, -1, seqs.size(2))), tok.unsqueeze(-1)], -1)
            was_fin = fin.gather(1, bi)
            lens = lens.gather(1, bi) + (~was_fin).float()
            fin = was_fin | (tok == EOS)
            flat = (bi + torch.arange(B, device=dev).unsqueeze(1) * K).view(-1)
            state = (state[0][flat], state[1][flat])
            prev = tok.view(-1)
            if fin.all(): break
        best = (scores / lens.clamp(min=1)).argmax(-1)
        return self._strip(seqs[torch.arange(B), best])

    @staticmethod
    def _strip(ids):
        res = []
        for row in ids.tolist():
            r = []
            for t in row:
                if t in (EOS, PAD): break
                r.append(t)
            res.append(r)
        return res

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
