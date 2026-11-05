"""Entraînement commun (P0 par défaut). Les autres pipelines ne changent QUE la fonction
`prepare_source` / `prepare_target` (ex. P1 : échantillonnage SentencePiece, P5 : bruitage).
Exemple : python scripts/train.py --pipeline P0 --arch a2 --seed 1
Le TEST n'est jamais évalué ici (règle : utilisé une seule fois, par le modèle final, avec --eval_test)."""
import argparse, csv, json, os, random, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn, yaml
import sentencepiece as spm
sys.path.insert(0, os.path.dirname(__file__))
from models import Seq2Seq, PAD, SOS, EOS
from evaluate import score

ap = argparse.ArgumentParser()
ap.add_argument("--pipeline", default="P0"); ap.add_argument("--arch", default="a2", choices=["a2", "a3", "a4", "a6"])
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--config", default="config.yaml")
ap.add_argument("--data_dir", default="data"); ap.add_argument("--tok_dir", default="tokenizers")
ap.add_argument("--run_dir"); ap.add_argument("--resume", action="store_true")
ap.add_argument("--max_epochs", type=int); ap.add_argument("--time_cap_min", type=float)
ap.add_argument("--max_steps", type=int, help="pour les tests rapides uniquement")
ap.add_argument("--train_limit", type=int, help="sous-échantillon du train (tests rapides)")
ap.add_argument("--eval_limit", type=int, help="sous-échantillon de la validation (tests rapides)")
ap.add_argument("--eval_test", action="store_true", help="MODÈLE FINAL UNIQUEMENT")
ap.add_argument("--device", default="auto")
ap.add_argument("--alpha", type=float, default=0.2, help="P1 : lissage de l'echantillonnage Unigram (petit = plus de variete ; 0 = pas d'echantillonnage, equivaut a P0)")
ap.add_argument("--sample_target", action="store_true", help="P1 : echantillonner aussi le francais (train seulement)")
a = ap.parse_args()

cfg = yaml.safe_load(open(a.config)); T, M, D = cfg["train"], cfg["model"], cfg["decode"]
if a.max_epochs: T["max_epochs"] = a.max_epochs
cap = a.time_cap_min or T.get("time_cap_minutes")
run = a.run_dir or f"runs/{a.pipeline}_{a.arch}_s{a.seed}"; os.makedirs(run, exist_ok=True)
dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device))
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)

# ---------- données ----------
sp_f = spm.SentencePieceProcessor(model_file=f"{a.tok_dir}/fon.model")
sp_t = spm.SentencePieceProcessor(model_file=f"{a.tok_dir}/fr.model")
MAXS, MAXT = cfg["tokenizer"]["max_src_len"], cfg["tokenizer"]["max_tgt_len"]

def prepare_source(text, epoch=0, train=False):      # P0 : segmentation déterministe
    return sp_f.encode(text)
def prepare_target(text):
    return sp_t.encode(text)

def load(split, limit=None):
    df = pd.read_csv(f"{a.data_dir}/{split}.csv")
    if limit: df = df.sample(n=min(limit, len(df)), random_state=0)
    return df
tr, va = load("train", a.train_limit), load("val", a.eval_limit)
def encode_train(epoch):
    """P1 : re-segmente tout le train au debut de chaque epoch (echantillonnage Unigram).
    Une graine par epoch => reproductible. Le filtre de longueur est refait a chaque fois,
    sur source ET cible ensemble, pour garder l'alignement."""
    if a.alpha > 0:
        spm.set_random_generator_seed(a.seed * 1000 + epoch)   # encode par LISTE : seul cas reproductible
        src = sp_f.encode(tr.fon.tolist(), enable_sampling=True, nbest_size=-1, alpha=a.alpha)
        tgt = (sp_t.encode(tr.fr.tolist(), enable_sampling=True, nbest_size=-1, alpha=a.alpha)
               if a.sample_target else sp_t.encode(tr.fr.tolist()))
    else:
        src, tgt = sp_f.encode(tr.fon.tolist()), sp_t.encode(tr.fr.tolist())
    keep = [i for i in range(len(tr)) if 0 < len(src[i]) <= MAXS and 0 < len(tgt[i]) <= MAXT]
    return [src[i] for i in keep], [tgt[i] for i in keep]
print(f"train : {len(tr)} paires | val : {len(va)} | appareil : {dev} | alpha : {a.alpha}", flush=True)

def pad(seqs):
    L = max(len(s) for s in seqs); x = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs): x[i, :len(s)] = torch.tensor(s)
    return x

def make_batch(src, tgt):
    s = pad(src); lens = torch.tensor([len(x) for x in src])
    ti = pad([[SOS] + t for t in tgt]); to = pad([t + [EOS] for t in tgt])
    return s.to(dev), lens.to(dev), ti.to(dev), to.to(dev)

def batches(src, tgt, bs, rng):
    idx = list(range(len(src))); rng.shuffle(idx)
    chunks = [sorted(idx[i:i + bs * 50], key=lambda j: len(src[j])) for i in range(0, len(idx), bs * 50)]
    out = [c[i:i + bs] for c in chunks for i in range(0, len(c), bs)]
    rng.shuffle(out); return out

va_src = [prepare_source(s) for s in va.fon]; va_tgt = [prepare_target(s) for s in va.fr]
va_idx = [i for i in range(len(va)) if len(va_tgt[i]) > 0 and len(va_src[i]) > 0]
sub_rng = random.Random(0); sub_idx = sorted(sub_rng.sample(range(len(va)), min(T["chrf_subset"], len(va))))  # sous-ensemble FIXE

# ---------- modèle ----------
model = Seq2Seq(a.arch, sp_f.get_piece_size(), sp_t.get_piece_size(), M).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=T["lr"])
crit = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=T["label_smoothing"])
crit_eval = nn.CrossEntropyLoss(ignore_index=PAD)
print(f"{a.pipeline} / {a.arch} / seed {a.seed} : {model.n_params():,} paramètres", flush=True)

@torch.no_grad()
def val_loss():
    model.eval(); tot, n = 0.0, 0
    order = sorted(va_idx, key=lambda i: len(va_src[i]))
    for i in range(0, len(order), 64):
        b = order[i:i + 64]
        s, l, ti, to = make_batch([va_src[j] for j in b], [va_tgt[j] for j in b])
        lg = model(s, l, ti, 1.0)
        tot += crit_eval(lg.reshape(-1, lg.size(-1)), to.reshape(-1)).item() * (to != PAD).sum().item(); n += (to != PAD).sum().item()
    model.train(); return tot / n

@torch.no_grad()
def translate(idx, beam):
    model.eval(); res = {}
    order = sorted(idx, key=lambda i: len(va_src[i]))
    for i in range(0, len(order), 64):
        b = order[i:i + 64]
        s = pad([va_src[j] for j in b]).to(dev); l = torch.tensor([len(va_src[j]) for j in b]).to(dev)
        out = model.greedy(s, l, D["max_len"]) if beam <= 1 else model.beam(s, l, beam, D["max_len"])
        for j, o in zip(b, out): res[j] = sp_t.decode(o)
    model.train(); return [res[i] for i in idx]

def save(path, extra=None):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "epoch": epoch, "best_val": best_val,
                "bad": bad, "elapsed": elapsed_prev + time.time() - t0, "best_elapsed": best_elapsed, "best_step": best_step,
                "config": {"pipeline": a.pipeline, "arch": a.arch, "seed": a.seed, "model": M, "train": T, "decode": D},
                "tokenizers": [f"{a.tok_dir}/fon.model", f"{a.tok_dir}/fr.model"]}, path)

# ---------- reprise ----------
step = epoch = bad = best_step = 0; best_val = float("inf"); elapsed_prev = best_elapsed = 0.0
if a.resume and os.path.exists(f"{run}/last.pt"):
    ck = torch.load(f"{run}/last.pt", map_location=dev)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
    step, epoch, best_val, bad = ck["step"], ck["epoch"], ck["best_val"], ck["bad"]
    elapsed_prev, best_elapsed, best_step = ck["elapsed"], ck["best_elapsed"], ck["best_step"]
    print(f"reprise à la mise à jour {step} (epoch {epoch})", flush=True)

logf = open(f"{run}/log.csv", "a", newline=""); logw = csv.writer(logf)
if os.path.getsize(f"{run}/log.csv") == 0: logw.writerow(["step", "epoch", "train_loss", "val_loss", "chrf_val_subset", "minutes"])
t0 = time.time(); rng = random.Random(a.seed + epoch); stop = None; run_loss, run_n = 0.0, 0; epoch_times = []
model.train()
while stop is None and epoch < T["max_epochs"]:
    te = time.time()
    tr_src, tr_tgt = encode_train(epoch)
    print(f"epoch {epoch} : {len(tr_src)}/{len(tr)} paires gardees", flush=True)
    for b in batches(tr_src, tr_tgt, T["batch_size"], rng):
        s, l, ti, to = make_batch([tr_src[j] for j in b], [tr_tgt[j] for j in b])
        lg = model(s, l, ti, T["teacher_forcing"])
        loss = crit(lg.reshape(-1, lg.size(-1)), to.reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), T["clip"]); opt.step()
        step += 1; run_loss += loss.item(); run_n += 1; elapsed = elapsed_prev + time.time() - t0
        if step % T["save_last_every"] == 0: save(f"{run}/last.pt")
        if T.get("save_step_every") and step % T["save_step_every"] == 0: save(f"{run}/step_{step:06d}.pt")
        if step % T["eval_every"] == 0:
            vl = val_loss(); chrf = ""
            if step % T["chrf_every"] == 0:
                chrf = score(translate(sub_idx, 1), [va.fr.iloc[i] for i in sub_idx])["chrf"]
            tl = run_loss / run_n; run_loss = run_n = 0
            if vl < best_val:
                best_val, bad, best_step, best_elapsed = vl, 0, step, elapsed; save(f"{run}/best.pt"); mark = " *meilleur*"
            else: bad += 1; mark = f" (patience {bad}/{T['patience']})"
            logw.writerow([step, epoch, round(tl, 4), round(vl, 4), chrf, round(elapsed / 60, 2)]); logf.flush()
            print(f"mise à jour {step} | epoch {epoch} | train {tl:.3f} | val {vl:.3f} | chrF {chrf} | {elapsed/60:.1f} min{mark}", flush=True)
            if bad >= T["patience"]: stop = "early stopping"
        if a.max_steps and step >= a.max_steps: stop = "max_steps"
        if cap and elapsed / 60 >= cap: stop = "plafond de temps"
        if stop: break
    epoch += 1; epoch_times.append(time.time() - te)
if stop is None: stop = "epochs maximum"
if not os.path.exists(f"{run}/best.pt"):            # test très court : aucune évaluation atteinte
    best_val, best_step, best_elapsed = val_loss(), step, elapsed_prev + time.time() - t0; save(f"{run}/best.pt")
save(f"{run}/last.pt")

# ---------- évaluation finale : validation complète, beam search, meilleur modèle ----------
ck = torch.load(f"{run}/best.pt", map_location=dev); model.load_state_dict(ck["model"])
hyp = translate(list(range(len(va))), D["beam_size"]); ref = va.fr.tolist()
open(f"{run}/val_hyp.txt", "w", encoding="utf-8").write("\n".join(hyp) + "\n")
open(f"{run}/val_ref.txt", "w", encoding="utf-8").write("\n".join(ref) + "\n")
sc = score(hyp, ref)
spe = float(np.mean(epoch_times)) if epoch_times else None
res = {"pipeline": a.pipeline, "arch": a.arch, "seed": a.seed, "params": model.n_params(), "arret": stop,
       "meilleure_maj": best_step, "meilleur_val_loss": round(best_val, 4), "epochs_faites": epoch,
       "bleu_val": sc["bleu"], "chrf_val": sc["chrf"], "minutes_total": round((elapsed_prev + time.time() - t0) / 60, 2),
       "minutes_pour_meilleur_point": round(best_elapsed / 60, 2), "secondes_par_epoch": round(spe, 1) if spe else None,
       "plafond_temps_propose_min": round(best_elapsed / 60 * 1.5, 1),
       "conseil_epochs": "monter à 50 (le meilleur point est proche de la fin)" if (stop == "epochs maximum") else "garder la valeur actuelle"}
if a.eval_test:
    te_df = load("test"); va_src, va_tgt, va = [prepare_source(s) for s in te_df.fon], [prepare_target(s) for s in te_df.fr], te_df
    th = translate(list(range(len(te_df))), D["beam_size"]); res["test"] = score(th, te_df.fr.tolist())
    open(f"{run}/test_hyp.txt", "w", encoding="utf-8").write("\n".join(th) + "\n")
json.dump(res, open(f"{run}/results.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(res, indent=2, ensure_ascii=False))
