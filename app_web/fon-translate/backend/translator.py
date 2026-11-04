import json
from pathlib import Path

BASE = Path(__file__).parent / "models"
_cache = {}


def load_registry():
    return json.loads((BASE / "registry.json").read_text("utf-8"))


def _resolve_tokenizer(configured_path, checkpoint_path):
    path = Path(configured_path)
    candidates = [path] if path.is_absolute() else [checkpoint_path.parent / path, BASE / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Tokenizer introuvable : {configured_path}")


def _load(ckpt, model_config, direction):
    import sentencepiece as spm
    import torch

    from model_architecture import Seq2Seq

    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    saved_config = payload.get("config", {})
    architecture = saved_config.get("arch")
    architecture_config = saved_config.get("model")
    if not architecture or not isinstance(architecture_config, dict):
        raise ValueError(f"Configuration d'architecture absente du checkpoint : {ckpt}")

    configured_tokenizers = model_config.get("tokenizers", {})
    if direction in configured_tokenizers:
        configured_tokenizers = configured_tokenizers[direction]
    embedded_tokenizers = payload.get("tokenizers", [])
    if isinstance(configured_tokenizers, dict) and {
        "source", "target"
    }.issubset(configured_tokenizers):
        source_tokenizer_path = configured_tokenizers["source"]
        target_tokenizer_path = configured_tokenizers["target"]
    elif len(embedded_tokenizers) >= 2:
        source_tokenizer_path, target_tokenizer_path = embedded_tokenizers[:2]
    else:
        raise ValueError(
            f"Tokenizers source/cible non déclarés pour le modèle {model_config['id']}"
        )

    source_tokenizer_path = _resolve_tokenizer(source_tokenizer_path, ckpt)
    target_tokenizer_path = _resolve_tokenizer(target_tokenizer_path, ckpt)
    source_tokenizer = spm.SentencePieceProcessor(model_file=str(source_tokenizer_path))
    target_tokenizer = spm.SentencePieceProcessor(model_file=str(target_tokenizer_path))

    model = Seq2Seq(
        architecture,
        source_tokenizer.get_piece_size(),
        target_tokenizer.get_piece_size(),
        architecture_config,
    )
    model.load_state_dict(payload["model"])
    model.eval()

    decode_config = saved_config.get("decode", {})
    max_source_length = int(model_config.get("max_src_len", 80))
    max_target_length = int(
        model_config.get("max_decode_len", decode_config.get("max_len", 100))
    )
    beam_size = int(model_config.get("beam_size", decode_config.get("beam_size", 4)))

    def run(text):
        token_ids = source_tokenizer.encode(text, out_type=int)[:max_source_length]
        if not token_ids:
            return ""
        source = torch.tensor([token_ids], dtype=torch.long)
        source_length = torch.tensor([len(token_ids)], dtype=torch.long)
        with torch.inference_mode():
            translated_ids = model.beam(
                source, source_length, beam=beam_size, max_len=max_target_length
            )[0]
        return target_tokenizer.decode(translated_ids)

    return run


def translate(model_id, text, direction="fon-fr"):
    if direction not in ("fon-fr", "fr-fon"):
        raise ValueError(f"Sens de traduction inconnu : {direction}")

    model_config = next((item for item in load_registry() if item["id"] == model_id), None)
    if not model_config:
        raise KeyError(model_id)

    checkpoints = model_config.get("checkpoints", {})
    checkpoint = checkpoints.get(direction)
    if checkpoint is None and direction == "fon-fr":
        checkpoint = model_config.get("checkpoint")
    if checkpoint is None:
        source, target = ("Fon", "français") if direction == "fon-fr" else ("français", "Fon")
        return {"translation": f"[démo {source} → {target}] {text}", "demo": True}

    checkpoint_path = BASE / checkpoint
    if not checkpoint_path.is_file():
        source, target = ("Fon", "français") if direction == "fon-fr" else ("français", "Fon")
        return {"translation": f"[démo {source} → {target}] {text}", "demo": True}

    stat = checkpoint_path.stat()
    cache_key = (model_id, direction, stat.st_mtime_ns, stat.st_size)
    if cache_key not in _cache:
        for old_key in list(_cache):
            if old_key[:2] == (model_id, direction):
                del _cache[old_key]
        _cache[cache_key] = _load(checkpoint_path, model_config, direction)
    return {"translation": _cache[cache_key](text), "demo": False}
