import { useEffect, useRef, useState } from "react";

const API = import.meta.env.VITE_API || "/api";
async function call(url, body) {
  const r = await fetch(API + url, body && {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || "Erreur " + r.status);
  return r.json();
}

function ModelSelect({ models, value, onChange }) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}>
      {models.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
    </select>
  );
}

function DirectionSelect({ value, onChange }) {
  return (
    <div className="direction-select" role="group" aria-label="Sens de traduction">
      <button type="button" className={value === "fon-fr" ? "selected" : ""} aria-pressed={value === "fon-fr"} onClick={() => onChange("fon-fr")}>Fon <span aria-hidden="true">→</span> Français</button>
      <button type="button" className={value === "fr-fon" ? "selected" : ""} aria-pressed={value === "fr-fon"} onClick={() => onChange("fr-fon")}>Français <span aria-hidden="true">→</span> Fon</button>
    </div>
  );
}

function AskBar({ text, setText, onSubmit, busy, models, model, setModel, direction, setDirection, placeholder }) {
  const [listening, setListening] = useState(false);
  const [voiceError, setVoiceError] = useState("");
  const recognitionRef = useRef(null);

  useEffect(() => () => recognitionRef.current?.abort(), []);

  const toggleListening = () => {
    if (listening) {
      recognitionRef.current?.stop();
      return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setVoiceError("La saisie vocale n’est pas prise en charge par ce navigateur. Essayez Chrome ou Edge.");
      return;
    }

    setVoiceError("");
    const recognition = new SpeechRecognition();
    recognition.lang = "fr-FR";
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;
    const originalText = text.trimEnd();

    recognition.onstart = () => setListening(true);
    recognition.onresult = (event) => {
      const transcript = Array.from(event.results)
        .map((result) => result[0].transcript)
        .join("");
      setText([originalText, transcript].filter(Boolean).join(" "));
    };
    recognition.onerror = (event) => {
      const messages = {
        "audio-capture": "Aucun microphone n’a été détecté.",
        "language-not-supported": "Cette langue n’est pas prise en charge par le navigateur.",
        "no-speech": "Aucune voix n’a été détectée. Réessayez.",
        "not-allowed": "L’accès au microphone a été refusé.",
        "service-not-allowed": "Le service de reconnaissance vocale n’est pas autorisé.",
      };
      setVoiceError(messages[event.error] || "La reconnaissance vocale a échoué. Réessayez.");
      setListening(false);
    };
    recognition.onend = () => setListening(false);
    recognitionRef.current = recognition;

    try {
      recognition.start();
    } catch {
      setListening(false);
      setVoiceError("Impossible de démarrer le microphone. Vérifiez son autorisation.");
    }
  };

  return (
    <div className="ask-wrap">
      <div className={`ask${listening ? " listening" : ""}`}>
        <textarea rows={2} value={text} placeholder={placeholder}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); onSubmit(); } }} />
        <div className="ask-toolbar">
          <div className="ask-options">
            <DirectionSelect
              value={direction}
              onChange={(nextDirection) => {
                if (nextDirection !== "fr-fon" && listening) recognitionRef.current?.stop();
                setDirection(nextDirection);
              }}
            />
            {setModel && <ModelSelect models={models} value={model} onChange={setModel} />}
          </div>
          <div className="ask-actions">
            <button className={`mic-button${listening ? " active" : ""}`} onClick={toggleListening} disabled={busy || direction !== "fr-fon"} aria-label={listening ? "Arrêter la transcription vocale" : "Démarrer la transcription vocale en français"} title={direction !== "fr-fon" ? "Disponible uniquement pour une saisie en français" : listening ? "Arrêter l’écoute" : "Dicter en français"}>
              {listening ? (
                <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="7" y="7" width="10" height="10" rx="2" /></svg>
              ) : (
                <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="3" width="6" height="12" rx="3" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3m-4 0h8" /></svg>
              )}
            </button>
            <button className="submit-button" onClick={onSubmit} disabled={busy || !text.trim()}>{busy ? "..." : "Traduire"}<span aria-hidden="true"> ↗</span></button>
          </div>
        </div>
      </div>
      {direction === "fon-fr" && <div className="voice-hint">Dictée vocale disponible uniquement pour les textes en français.</div>}
      {listening && <div className="voice-status" role="status"><span className="listening-dot" /> Écoute en cours… parlez clairement puis arrêtez le micro.</div>}
      {!listening && voiceError && <div className="voice-error" role="status">{voiceError}</div>}
    </div>
  );
}

function Translate({ models }) {
  const [text, setText] = useState(""), [model, setModel] = useState(""), [direction, setDirection] = useState("fon-fr"), [items, setItems] = useState([]), [busy, setBusy] = useState(false);
  useEffect(() => { if (!model && models.length) setModel(models[0].id); }, [models]);
  const run = async () => {
    if (!text.trim() || busy) return;
    setBusy(true);
    const src = text, id = Date.now(), selectedDirection = direction;
    try { const r = await call("/translate", { text: src, model, direction: selectedDirection }); setItems((i) => [{ id, src, ...r }, ...i]); setText(""); }
    catch (e) { setItems((i) => [{ id, src, model, direction: selectedDirection, error: e.message }, ...i]); }
    setBusy(false);
  };
  const sourceLanguage = direction === "fon-fr" ? "Fon" : "Français";
  const targetLanguage = direction === "fon-fr" ? "Français" : "Fon";
  return (
    <div className="page">
      <h1>Que voulez-vous traduire aujourd'hui&nbsp;?</h1>
      <p className="direction-hint">Traduction {sourceLanguage} <span aria-hidden="true">→</span> {targetLanguage}</p>
      <AskBar {...{ text, setText, busy, models, model, setModel, direction, setDirection }} onSubmit={run} placeholder={`Saisissez un texte en ${sourceLanguage.toLowerCase()}`} />
      <div className="results">
        {items.map((it) => (
          <div className="card" key={it.id}>
            <div className="meta">{it.model} · {it.direction === "fr-fon" ? "Français → Fon" : "Fon → Français"}{it.demo && " · mode démo"}</div>
            <div className="src">{it.src}</div>
            {it.error ? <div className="err">{it.error}</div> : <div className="out">{it.translation}</div>}
          </div>
        ))}
      </div>
    </div>
  );
}

function Compare({ models }) {
  const [text, setText] = useState(""), [direction, setDirection] = useState("fon-fr"), [res, setRes] = useState([]), [busy, setBusy] = useState(false);
  const run = async () => {
    if (!text.trim() || busy) return;
    setBusy(true);
    setRes(await Promise.all(models.map(async (m) => {
      try { return { m, ...(await call("/translate", { text, model: m.id, direction })) }; }
      catch (e) { return { m, error: e.message }; }
    })));
    setBusy(false);
  };
  return (
    <div className="page">
      <h1>Comparer les modèles</h1>
      <AskBar {...{ text, setText, busy, models, direction, setDirection }} onSubmit={run} placeholder={`Une phrase en ${direction === "fon-fr" ? "fon" : "français"}, tous les modèles`} />
      <div className="grid">
        {res.map((r) => (
          <div className="card" key={r.m.id}>
            <div className="meta">{r.m.name} {r.m.bleu != null && `- BLEU ${r.m.bleu}`}</div>
            {r.error ? <div className="err">{r.error}</div> : <div className="out">{r.translation}</div>}
          </div>
        ))}
      </div>
    </div>
  );
}

function LossChart({ logs }) {
  if (!logs.length) return <div className="empty">En attente des premières epochs (backend/training_log.py).</div>;
  const W = 640, H = 220, P = 30;
  const maxE = Math.max(...logs.map((l) => l.epoch), 2);
  const maxL = Math.max(...logs.flatMap((l) => [l.train_loss, l.val_loss]));
  const pt = (l, k) => `${P + ((l.epoch - 1) / (maxE - 1)) * (W - 2 * P)},${H - P - (l[k] / maxL) * (H - 2 * P)}`;
  const line = (k, c) => <polyline fill="none" stroke={c} strokeWidth="2" points={logs.map((l) => pt(l, k)).join(" ")} />;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart">
      <line x1={P} y1={H - P} x2={W - P} y2={H - P} stroke="#444" /><line x1={P} y1={P} x2={P} y2={H - P} stroke="#444" />
      {line("train_loss", "#6c8cff")}{line("val_loss", "#f0a04b")}
      <text x={P + 6} y={P} fill="#6c8cff" fontSize="11">train</text>
      <text x={P + 46} y={P} fill="#f0a04b" fontSize="11">val</text>
      <text x={W - P} y={H - 8} fill="#888" fontSize="11" textAnchor="end">epoch (max loss {maxL.toFixed(2)})</text>
    </svg>
  );
}

function Training({ models }) {
  const [tr, setTr] = useState({ logs: [], running: false }), [st, setSt] = useState({ available: false });
  useEffect(() => {
    const tick = () => { call("/training").then(setTr).catch(() => {}); call("/data/stats").then(setSt).catch(() => {}); };
    tick(); const t = setInterval(tick, 3000); return () => clearInterval(t);
  }, []);
  const last = tr.logs[tr.logs.length - 1];
  return (
    <div className="page wide">
      <h1>Entraînement et données</h1>
      <div className="stats">
        <div><b>{tr.running ? "En cours" : "Inactif"}</b><span>état</span></div>
        <div><b>{last ? last.epoch : "-"}</b><span>epoch</span></div>
        <div><b>{last ? last.val_loss.toFixed(3) : "-"}</b><span>val loss</span></div>
        <div><b>{st.available ? st.train : "-"}</b><span>paires train</span></div>
        <div><b>{st.available ? st.val : "-"}</b><span>paires val</span></div>
      </div>
      <div className="panel"><LossChart logs={tr.logs} /></div>
      <div className="panel">
        <h3>Performances (BLEU sur le test)</h3>
        <table><thead><tr><th>Modèle</th><th>Architecture</th><th>BLEU</th><th>chrF</th></tr></thead>
          <tbody>{models.map((m) => <tr key={m.id}><td>{m.name}</td><td>{m.architecture}</td><td>{m.bleu ?? "-"}</td><td>{m.chrf ?? "-"}</td></tr>)}</tbody></table>
      </div>
      <div className="panel">
        <h3>Aperçu des données</h3>
        {st.available ? (<>
          <p className="muted">Longueur moyenne : {st.avg_src_len} / {st.avg_tgt_len} mots - Vocabulaire : {st.vocab_src} / {st.vocab_tgt}</p>
          <table><thead><tr><th>Fon</th><th>Français</th></tr></thead>
            <tbody>{st.sample.map((s, i) => <tr key={i}><td>{s.src}</td><td>{s.tgt}</td></tr>)}</tbody></table>
        </>) : <div className="empty">Déposer train.csv et val.csv dans backend/data/</div>}
      </div>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState("translate"), [models, setModels] = useState([]), [down, setDown] = useState(false);
  useEffect(() => { call("/models").then(setModels).catch(() => setDown(true)); }, []);
  const tabs = [["translate", "Traduire"], ["compare", "Comparer"], ["training", "Entraînement"]];
  return (
    <div className="app">
      <nav>
        <div className="brand">Fon / Français</div>
        {tabs.map(([k, l]) => <button key={k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{l}</button>)}
        {down && <div className="err small">Backend injoignable (port 8000)</div>}
      </nav>
      <main>
        {tab === "translate" && <Translate models={models} />}
        {tab === "compare" && <Compare models={models} />}
        {tab === "training" && <Training models={models} />}
      </main>
    </div>
  );
}
