import csv, json, os, time
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from translator import translate, load_registry

DATA = Path(__file__).parent / "data"
SRC, TGT = os.getenv("SRC_COL", "fon"), os.getenv("TGT_COL", "fr")

app = FastAPI(title="Fon-Français")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Req(BaseModel):
    text: str
    model: str
    direction: Literal["fon-fr", "fr-fon"] = "fon-fr"

@app.get("/api/models")
def models():
    return load_registry()

@app.post("/api/translate")
def do_translate(r: Req):
    if not r.text.strip():
        raise HTTPException(400, "Texte vide")
    try:
        out = translate(r.model, r.text.strip(), r.direction)
    except KeyError:
        raise HTTPException(404, "Modèle inconnu")
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"model": r.model, "source_text": r.text, **out}

def read_rows(name):
    p = DATA / name
    if not p.exists():
        return None
    with open(p, encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        s, t = (SRC, TGT) if SRC in cols and TGT in cols else (cols[0], cols[1]) if len(cols) > 1 else (None, None)
        return [(r[s], r[t]) for r in rd] if s else []

@app.get("/api/data/stats")
def stats():
    tr, va = read_rows("train.csv"), read_rows("val.csv")
    if tr is None:
        return {"available": False}
    n = max(len(tr), 1)
    return {"available": True, "train": len(tr), "val": len(va) if va is not None else 0,
            "avg_src_len": round(sum(len(a.split()) for a, _ in tr) / n, 1),
            "avg_tgt_len": round(sum(len(b.split()) for _, b in tr) / n, 1),
            "vocab_src": len({w for a, _ in tr for w in a.lower().split()}),
            "vocab_tgt": len({w for _, b in tr for w in b.lower().split()}),
            "sample": [{"src": a, "tgt": b} for a, b in tr[:10]]}

@app.get("/api/training")
def training():
    p = DATA / "training_log.jsonl"
    if not p.exists():
        return {"logs": [], "running": False}
    logs = [json.loads(l) for l in p.read_text("utf-8").splitlines() if l.strip()]
    return {"logs": logs, "running": time.time() - p.stat().st_mtime < 60}
