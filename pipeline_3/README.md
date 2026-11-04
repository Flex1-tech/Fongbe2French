
=== P3 — CharCNN côté source, Fongbe → Français ===

Architectures testées : A2 (BiLSTM), A3 (BiGRU)
CharCNN : 107 chars, emb 32, convs 2/3/4 × 64 filtres, highway, proj 256
Cible : SentencePiece Unigram 4000 (commun au groupe)
Améliorations : weight tying (−20% params A2, −26% A3)
Scheduler warmup+inverse-sqrt testé et écarté (dégrade les perf en 30 epochs)
Seed : 1 (contrainte deadline)

Résultats sur validation (2242 phrases, beam search largeur 4) :
                   BLEU    chrF    params    temps   best_step
A2 (LSTM)          3.04   17.74   3.99M     14 min  13000 (epoch 30)
A3 (GRU)           2.60   18.49   3.34M     14 min  13000 (epoch 30)

Décomposition mots inconnus (35.4% des phrases de val) :
                   chrF_unk   chrF_kno   écart
A2 (LSTM)           15.67      19.62     3.95
A3 (GRU)            16.78      20.06     3.28

Conclusion P3 :

- A3 ≈ A2 en performance (chrF 18.49 vs 17.74, dans le bruit à 1 seed)
- A3 plus économe (−16% params) et généralise mieux sur mots inconnus
- Poids du CharCNN : 145 632 params (3.6% du total A2)
- Taux de mots inconnus : 5.0% des mots, 35.4% des phrases
