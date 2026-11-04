# Rapport de Travail : Traduction Automatique Fongbé → Français (P4a)

Ce rapport présente l'analyse comparative des architectures de traduction automatique Seq2Seq basées sur un encodeur bidirectionnel et un mécanisme d'attention de Bahdanau, en évaluant l'apport de l'initialisation des plongements (embeddings) cibles (français) à l'aide de **FastText**.

---

## 1. Description du Jeu de Données et Prétraitement
- **Train** : 27 730 paires (116 écartées car trop longues).
- **Validation** : 2 242 paires.
- **Vocabulaire Source (Fongbé)** : 2 000 sous-mots (SentencePiece Unigram entraîné sur le train).
- **Vocabulaire Cible (Français)** : 4 000 sous-mots (SentencePiece pré-entraîné `spm_fr.model`).

---

## 2. Architectures & Protocoles Expérimentaux
Nous avons évalué deux architectures principales :
- **A2** : LSTM bidirectionnel + attention de Bahdanau.
- **A3** : GRU bidirectionnel + attention de Bahdanau.

Les embeddings cibles ont été initialisés selon trois stratégies :
1. **Random** (Contrôle) : Initialisation aléatoire classique.
2. **Piece** (FastText sub-word) : Vecteur FastText direct associé au sous-mot du SentencePiece.
3. **Frozen** / **Finetune** : Plongements gelés à l'entraînement ou ajustés avec un taux d'apprentissage réduit (multiplicateur de `0.1`).

---

## 3. Résultats Comparatifs (Validation - Beam Search de largeur 4)

Voici le récapitulatif des performances obtenues à la fin des cycles d'entraînement :

| Expérience (Run) | Architecture | Init. Embeddings | Mode | Paramètres (M) | Validation Loss (Min) | chrF (val) | BLEU (val) | Temps (min) |
|---|---|---|---|---|---|---|---|---|
| **A2_piece_finetune** | A2 (LSTM) | FastText (Piece) | Finetune | 8.20 M | **3.9158** | 20.20 | **3.55** | 9.7 min |
| **A3_piece_finetune** | A3 (GRU) | FastText (Piece) | Finetune | 7.54 M | 4.0068 | **20.81** | 2.63 | 5.8 min |
| **A2_piece_frozen**   | A2 (LSTM) | FastText (Piece) | Frozen   | 8.20 M | 3.9504 | 19.62 | 3.18 | 9.0 min |
| **A2_random**         | A2 (LSTM) | Aléatoire        | Finetune | 8.20 M | 3.9316 | 20.41 | 3.32 | 9.9 min |

---

## 4. Analyses & Conclusions

1. **Apport de FastText (A2_piece_finetune vs A2_random)** :
   - L'initialisation par FastText sub-word (`A2_piece_finetune`) offre la meilleure perte de validation minimale (**3.9158**) et le meilleur score **BLEU (3.55)**.
   - L'initialisation aléatoire (`A2_random`) reste très compétitive (BLEU de 3.32), prouvant la capacité du modèle à apprendre par lui-même sur cette taille de corpus, mais FastText apporte un guidage initial bénéfique.

2. **Impact du gel des plongements (A2_piece_frozen vs A2_piece_finetune)** :
   - Bloquer la mise à jour des embeddings (`A2_piece_frozen`) dégrade légèrement les performances (BLEU de 3.18 contre 3.55). Permettre un finetuning léger (`EMB_LR_MULT = 0.1`) offre une flexibilité nécessaire pour adapter l'espace de représentation FastText à la tâche de traduction.

3. **LSTM (A2) vs GRU (A3)** :
   - L'architecture **A3 (GRU)** est beaucoup plus rapide à l'entraînement (5.8 minutes contre 9.7 minutes pour A2) et obtient le meilleur score **chrF (20.81)**, bien que son score BLEU pur soit en retrait (2.63).
