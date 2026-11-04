import json, time
from pathlib import Path
LOG = Path(__file__).parent / "data" / "training_log.jsonl"

def log_epoch(epoch, train_loss, val_loss, model="seq2seq_lstm", **extra):
    LOG.parent.mkdir(exist_ok=True)
    row = {"t": time.time(), "model": model, "epoch": epoch,
           "train_loss": float(train_loss), "val_loss": float(val_loss), **extra}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
