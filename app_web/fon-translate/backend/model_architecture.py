import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD, UNK, SOS, EOS = 0, 1, 2, 3


class RNNEncoder(nn.Module):
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
        packed = pack_padded_sequence(
            x, src_len.cpu(), batch_first=True, enforce_sorted=False
        )
        out, hidden = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        if self.cell == "lstm":
            h, c = hidden
            h = torch.cat([h[0], h[1]], -1)
            c = torch.cat([c[0], c[1]], -1)
            return out, (torch.tanh(self.bridge_h(h)), torch.tanh(self.bridge_c(c)))
        h = torch.cat([hidden[0], hidden[1]], -1)
        return out, (torch.tanh(self.bridge_h(h)), None)


class CNNEncoder(nn.Module):
    def __init__(self, vocab, emb, hid, dropout, layers=3, kernel=3, max_pos=256):
        super().__init__()
        dim = 2 * hid
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.pos = nn.Embedding(max_pos, emb)
        self.max_pos = max_pos
        self.proj = nn.Linear(emb, dim)
        self.convs = nn.ModuleList(
            [nn.Conv1d(dim, 2 * dim, kernel, padding=kernel // 2) for _ in range(layers)]
        )
        self.drop = nn.Dropout(dropout)
        self.bridge_h = nn.Linear(dim, hid)
        self.bridge_c = nn.Linear(dim, hid)

    def forward(self, src, src_len):
        _, length = src.shape
        mask = (src != PAD).unsqueeze(-1).float()
        positions = torch.arange(length, device=src.device).clamp(
            max=self.max_pos - 1
        ).unsqueeze(0)
        x = self.drop(self.emb(src) + self.pos(positions))
        x = self.proj(x) * mask
        for conv in self.convs:
            y = F.glu(conv(self.drop(x).transpose(1, 2)), dim=1).transpose(1, 2)
            x = (x + y) * math.sqrt(0.5) * mask
        mean = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        return x, (torch.tanh(self.bridge_h(mean)), torch.tanh(self.bridge_c(mean)))


class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, att_dim):
        super().__init__()
        self.We = nn.Linear(enc_dim, att_dim, bias=False)
        self.Ws = nn.Linear(dec_dim, att_dim)
        self.v = nn.Linear(att_dim, 1, bias=False)

    def forward(self, state, enc_proj, enc_out, mask):
        energy = self.v(
            torch.tanh(enc_proj + self.Ws(state).unsqueeze(1))
        ).squeeze(-1)
        energy = energy.masked_fill(~mask, -1e9)
        attention = F.softmax(energy, -1)
        context = torch.bmm(attention.unsqueeze(1), enc_out).squeeze(1)
        return context, attention


class Seq2Seq(nn.Module):
    def __init__(self, arch, src_vocab, tgt_vocab, config):
        super().__init__()
        emb, hid, dropout = config["emb_dim"], config["hid_dim"], config["dropout"]
        self.arch, self.hid, self.tgt_vocab = arch, hid, tgt_vocab
        self.cell = "gru" if arch == "a3" else "lstm"
        if arch == "a2":
            self.encoder = RNNEncoder(src_vocab, emb, hid, dropout, "lstm")
        elif arch == "a3":
            self.encoder = RNNEncoder(src_vocab, emb, hid, dropout, "gru")
        elif arch == "a6":
            self.encoder = CNNEncoder(
                src_vocab,
                emb,
                hid,
                dropout,
                config.get("cnn_layers", 3),
                config.get("cnn_kernel", 3),
            )
        else:
            raise ValueError(f"Architecture inconnue : {arch} (a2, a3, a6)")
        enc_dim = 2 * hid
        self.att_dim = enc_dim
        self.emb = nn.Embedding(tgt_vocab, emb, padding_idx=PAD)
        self.drop = nn.Dropout(dropout)
        self.att = BahdanauAttention(enc_dim, hid, hid)
        cell_type = nn.GRUCell if self.cell == "gru" else nn.LSTMCell
        self.dec = cell_type(emb + enc_dim, hid)
        self.out = nn.Linear(hid + enc_dim + emb, tgt_vocab)

    def encode(self, src, src_len):
        enc_out, (hidden, cell) = self.encoder(src, src_len)
        if cell is None:
            cell = torch.zeros_like(hidden)
        memory = {
            "enc": enc_out,
            "proj": self.att.We(enc_out),
            "mask": src != PAD,
        }
        return memory, (hidden, cell)

    def step(self, previous, state, memory):
        hidden, cell = state
        embedded = self.drop(self.emb(previous))
        context, attention = self.att(
            hidden, memory["proj"], memory["enc"], memory["mask"]
        )
        decoder_input = torch.cat([embedded, context], -1)
        if self.cell == "gru":
            hidden = self.dec(decoder_input, hidden)
        else:
            hidden, cell = self.dec(decoder_input, (hidden, cell))
        logits = self.out(self.drop(torch.cat([hidden, context, embedded], -1)))
        return logits, (hidden, cell), attention

    @torch.no_grad()
    def beam(self, src, src_len, beam=4, max_len=100):
        memory, (hidden, cell) = self.encode(src, src_len)
        batch, device = src.size(0), src.device
        repeat = lambda tensor: tensor.repeat_interleave(beam, 0)
        memory = {key: repeat(value) for key, value in memory.items()}
        state = (repeat(hidden), repeat(cell))
        scores = torch.full((batch, beam), -1e9, device=device)
        scores[:, 0] = 0
        sequences = torch.zeros(batch, beam, 0, dtype=torch.long, device=device)
        lengths = torch.zeros(batch, beam, device=device)
        finished = torch.zeros(batch, beam, dtype=torch.bool, device=device)
        previous = torch.full((batch * beam,), SOS, dtype=torch.long, device=device)

        for _ in range(max_len):
            logits, state, _ = self.step(previous, state, memory)
            logits[:, PAD] = -1e9
            logits[:, SOS] = -1e9
            log_probs = F.log_softmax(logits, -1).view(batch, beam, -1)
            vocab_size = log_probs.size(-1)
            frozen = torch.full((vocab_size,), -1e9, device=device)
            frozen[PAD] = 0
            log_probs = torch.where(
                finished.unsqueeze(-1), frozen.expand(batch, beam, vocab_size), log_probs
            )
            candidates = (scores.unsqueeze(-1) + log_probs).view(batch, beam * vocab_size)
            scores, indices = candidates.topk(beam, -1)
            beam_indices, tokens = indices // vocab_size, indices % vocab_size
            sequences = torch.cat(
                [
                    sequences.gather(
                        1,
                        beam_indices.unsqueeze(-1).expand(-1, -1, sequences.size(2)),
                    ),
                    tokens.unsqueeze(-1),
                ],
                -1,
            )
            was_finished = finished.gather(1, beam_indices)
            lengths = lengths.gather(1, beam_indices) + (~was_finished).float()
            finished = was_finished | (tokens == EOS)
            flat_indices = (
                beam_indices + torch.arange(batch, device=device).unsqueeze(1) * beam
            ).view(-1)
            state = (state[0][flat_indices], state[1][flat_indices])
            previous = tokens.view(-1)
            if finished.all():
                break

        best = (scores / lengths.clamp(min=1)).argmax(-1)
        return self._strip(sequences[torch.arange(batch), best])

    @staticmethod
    def _strip(ids):
        results = []
        for row in ids.tolist():
            result = []
            for token in row:
                if token in (EOS, PAD):
                    break
                result.append(token)
            results.append(result)
        return results
