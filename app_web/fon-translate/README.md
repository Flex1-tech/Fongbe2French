# Traduction Fon ↔ Français

Frontend : React + Vite. Backend : FastAPI. Prérequis : Python 3.10+ et Node.js 18+.

## Copier le projet sur un autre PC

Copiez le dossier du projet (par ZIP, clé USB ou dépôt Git). Gardez notamment
`frontend/package-lock.json` : il permet de réinstaller les mêmes dépendances
frontend. Ne copiez pas `backend/.venv`, `frontend/node_modules` ou `frontend/dist` ;
ils sont recréés sur le nouveau PC.

Les fichiers de données (`backend/data/*.csv`, `*.jsonl`) et les checkpoints
(`backend/models/*.pt`) sont exclus de Git. Si vous en avez, copiez-les séparément
et placez-les dans les dossiers indiqués ci-dessus. L'application charge les
checkpoints produits par `fonfr2/scripts/train.py` (format `best.pt`) avec leurs
tokenizers SentencePiece.
Une connexion Internet est nécessaire lors de la première installation des dépendances.

## Lancer (identique sur n'importe quel PC)

Dans PowerShell, ouvrez deux terminaux à la racine du projet.

Terminal 1 (backend) :
    cd backend
    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000

Terminal 2 (frontend ; utilisez `npm.cmd` dans PowerShell si l'exécution de scripts est bloquée) :
    cd frontend
    npm.cmd install
    npm.cmd run dev
Ouvrir http://localhost:5173 dans un navigateur.

Sur Linux/macOS, dans le terminal backend, utilisez `cd backend`,
`python3 -m venv .venv`, `source .venv/bin/activate`,
`pip install -r requirements.txt` puis `uvicorn main:app --reload --port 8000`.
Dans le terminal frontend, utilisez `cd frontend`, `npm install` puis `npm run dev`.

## Où brancher le reste
- Données : déposer train.csv et val.csv dans backend/data/ (colonnes fon, fr ; modifiable via SRC_COL / TGT_COL).
- Entraînement temps réel : dans la boucle d'epochs, ajouter
    from training_log import log_epoch ; log_epoch(epoch, train_loss, val_loss)
- Modèles : déclarer chaque modèle dans `backend/models/registry.json` (checkpoint,
  scores BLEU/chrF). Copiez le checkpoint dans `backend/models/` et les fichiers
  `fon.model` et `fr.model` dans `backend/models/tokenizers/`. Les chemins enregistrés
  dans le checkpoint sont résolus depuis son dossier ; ils peuvent être remplacés
  par `tokenizers` dans l'entrée du registre, par exemple :
  `"tokenizers": {"fon-fr": {"source": "tokenizers/fon.model", "target": "tokenizers/fr.model"}}`.
  L'architecture (`a2`, `a3` ou `a6`) et les paramètres sont lus depuis le checkpoint.
  Pour traduire dans les deux sens, entraînez aussi un checkpoint français→Fon puis
  déclarez les chemins sous `checkpoints` et `tokenizers` avec les clés `fon-fr` et `fr-fon`.
- Tant qu'aucun checkpoint n'existe pour le sens choisi, la traduction indique le mode [démo].
