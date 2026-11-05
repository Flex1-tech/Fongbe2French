"""
A3 — BiGRU + attention de Bahdanau (variante de A2, cellules GRU)
Encodeur bidirectionnel GRU (1 couche) + attention de Bahdanau + décodeur GRU (1 couche).
Module autonome : aucune dépendance au config.yaml (hyperparamètres en arguments, valeurs par défaut = config commune).

Usage :
    from a3_bigru_attention import A3Seq2Seq, greedy_batch, beam_search
    model = A3Seq2Seq(v_src=2000, v_tgt=4000).to(device)
    model.load_target_embeddings(matrice_fasttext, freeze=False)     # P4a : init FastText côté cible (optionnel)
    logits = model(src, src_len, tgt_in)                              # (B, T, v_tgt), teacher forcing = 1.0
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD, UNK, SOS, EOS = 0, 1, 2, 3


class Bahdanau(nn.Module):
    """Attention additive : score = v^T tanh(W_enc h_enc + W_dec s). Les positions de padding sont masquées."""

    def __init__(self, enc_dim, dec_dim, att_dim):
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, att_dim, bias=False)
        self.W_dec = nn.Linear(dec_dim, att_dim)
        self.v = nn.Linear(att_dim, 1, bias=False)

    def project(self, enc_out):                      # précalculé une fois par phrase
        return self.W_enc(enc_out)

    def forward(self, h, enc_out, proj, mask):
        e = self.v(torch.tanh(proj + self.W_dec(h).unsqueeze(1))).squeeze(-1)
        a = F.softmax(e.masked_fill(~mask, -1e9), dim=-1)
        return torch.bmm(a.unsqueeze(1), enc_out).squeeze(1)


class A3Seq2Seq(nn.Module):
    def __init__(self, v_src, v_tgt, emb_dim=256, hid_dim=256, dropout=0.3):
        super().__init__()
        self.src_emb = nn.Embedding(v_src, emb_dim, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(v_tgt, emb_dim, padding_idx=PAD)
        self.enc = nn.GRU(emb_dim, hid_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.bridge_h = nn.Linear(2 * hid_dim, hid_dim)   # liaison : états finaux avant+arrière -> état initial du décodeur (pas d'état de cellule)
        self.attn = Bahdanau(2 * hid_dim, hid_dim, hid_dim)
        self.cell = nn.GRUCell(emb_dim + 2 * hid_dim, hid_dim)
        self.out = nn.Linear(hid_dim + 2 * hid_dim + emb_dim, v_tgt)   # [état ; contexte ; embedding] -> vocabulaire
        self.drop = nn.Dropout(dropout)

    def load_target_embeddings(self, matrix, freeze=False):
        """Initialise les embeddings cible (ex. FastText réduit à emb_dim). matrix : tensor/array (v_tgt, emb_dim)."""
        self.tgt_emb.weight.data.copy_(torch.as_tensor(matrix, dtype=torch.float32))
        self.tgt_emb.weight.requires_grad = not freeze

    def encode(self, src, src_len):
        mask = src != PAD
        x = self.drop(self.src_emb(src))
        packed = pack_padded_sequence(x, src_len.cpu(), batch_first=True, enforce_sorted=False)
        out, hn = self.enc(packed)
        enc_out, _ = pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        state = (torch.tanh(self.bridge_h(torch.cat([hn[0], hn[1]], -1))),)   # état = tuple à 1 élément
        return enc_out, mask, state, self.attn.project(enc_out)

    def step(self, y, state, enc_out, proj, mask):
        """Un pas de décodage. y : (B,) token précédent. Retourne (logits (B, v_tgt), nouvel état)."""
        emb = self.drop(self.tgt_emb(y))
        ctx = self.attn(state[0], enc_out, proj, mask)
        inp = torch.cat([emb, ctx], -1)
        state = (self.cell(inp, state[0]),)                # GRU : seul l'état caché
        logits = self.out(self.drop(torch.cat([state[0], ctx, emb], -1)))
        return logits, state

    def forward(self, src, src_len, tgt_in):
        enc_out, mask, state, proj = self.encode(src, src_len)
        outs = []
        for t in range(tgt_in.size(1)):
            lg, state = self.step(tgt_in[:, t], state, enc_out, proj, mask)
            outs.append(lg)
        return torch.stack(outs, 1)


@torch.no_grad()
def greedy_batch(model, src, src_len, max_len=100):
    """Décodage glouton par batch. Retourne une liste de listes d'ids (sans <eos>)."""
    model.eval()
    dev = src.device
    enc_out, mask, state, proj = model.encode(src, src_len)
    y = torch.full((src.size(0),), SOS, dtype=torch.long, device=dev)
    done = torch.zeros(src.size(0), dtype=torch.bool, device=dev)
    outs = []
    for _ in range(max_len):
        lg, state = model.step(y, state, enc_out, proj, mask)
        y = lg.argmax(-1)
        outs.append(y)
        done |= y == EOS
        if done.all():
            break
    return [s[:s.index(EOS)] if EOS in s else s for s in torch.stack(outs, 1).tolist()]


@torch.no_grad()
def beam_search(model, src, src_len, beam_size=4, max_len=100):
    """Beam search pour UNE phrase (src : (1, S)). Normalisation du score par la longueur. Retourne une liste d'ids."""
    model.eval()
    dev, k = src.device, beam_size
    enc_out, mask, state, _ = model.encode(src, src_len)
    enc_out, mask = enc_out.expand(k, -1, -1).contiguous(), mask.expand(k, -1).contiguous()
    state = tuple(s.expand(k, -1).contiguous() for s in state)
    proj = model.attn.project(enc_out)
    y = torch.full((k,), SOS, dtype=torch.long, device=dev)
    scores = torch.zeros(k, device=dev)
    scores[1:] = -1e9                                  # au départ une seule hypothèse vivante
    hist = torch.zeros(k, 0, dtype=torch.long, device=dev)
    finished = []
    for t in range(max_len):
        lg, state = model.step(y, state, enc_out, proj, mask)
        cand = (scores.unsqueeze(1) + F.log_softmax(lg, -1)).view(-1)
        top_s, top_i = cand.topk(k)
        beam_idx, tok = top_i // lg.size(1), top_i % lg.size(1)
        state = tuple(s[beam_idx] for s in state)
        hist = torch.cat([hist[beam_idx], tok.unsqueeze(1)], 1)
        scores, y = top_s.clone(), tok
        for i in (tok == EOS).nonzero().flatten().tolist():
            finished.append((scores[i].item() / (t + 1), hist[i, :-1].tolist()))
            scores[i] = -1e9
        if len(finished) >= k:
            break
    if not finished:
        finished = [(scores[i].item() / max_len, hist[i].tolist()) for i in range(k)]
    return max(finished, key=lambda x: x[0])[1]
