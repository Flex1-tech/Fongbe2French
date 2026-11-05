# Pipeline P1 : subword regularization (Loan)

## Méthode
- Régularisation de sous-mots avec SentencePiece Unigram, équivalent du BPE-dropout : au début de chaque epoch, tout le train est re-segmenté (`encode_train` dans `train.py`) avec `sp.encode(liste, enable_sampling=True, nbest_size=-1, alpha=...)`, puis le filtre de longueur du `config.yaml` (`max_src_len`, `max_tgt_len`) est refait sur source et cible ensemble pour garder l'alignement.
- Par défaut, seul le fongbé (source) est échantillonné. `--sample_target` échantillonne aussi le français.
- Reproductibilité : la graine est fixée à `seed*1000 + epoch` et l'encodage se fait par liste entière (phrase par phrase, l'échantillonnage n'est pas reproductible).
- Validation (et test) restent déterministes. Le test n'est jamais évalué (règle d'équipe). Données, tokenizers et `evaluate.py` ne sont pas modifiés.
- Arguments ajoutés à `train.py` : `--alpha` (défaut 0,2 ; `0` = pas d'échantillonnage, donc comportement P0) et `--sample_target`.
- A4, écrit par moi à partir du PDF (ce n'est pas la version de Bénito) : encodeur BiLSTM à 2 couches (dropout entre les couches, un « pont » par couche pour initialiser le décodeur), décodeur LSTM à 2 couches avec dropout entre les couches et connexion résiduelle autour de la seconde, attention calculée sur l'état de la couche du haut. Classe `DeepEncoder` et méthode `_step_deep` dans `models.py`. Paramètres : A2 8 199 072, A3 7 541 920, A4 10 565 024.

## Reproduire
Depuis la racine du dépôt (dépendances : sentencepiece, sacrebleu, pyyaml) :

```
unzip -oq "fonfr_P0 (2).zip"
cd fonfr_P0/fonfr2
pip install -q sentencepiece sacrebleu pyyaml
# copier train.py et models.py de ce dossier dans scripts/ (ils remplacent ceux de P0)
python scripts/train.py --pipeline P1 --arch a2 --seed 1 --alpha 0.3 --run_dir P1_a2_alpha03_s1
python scripts/train.py --pipeline P1 --arch a3 --seed 1 --alpha 0.3 --run_dir P1_a3_alpha03_s1
python scripts/train.py --pipeline P1 --arch a4 --seed 1 --alpha 0.3 --run_dir P1_a4_alpha03_s1
python scripts/train.py --pipeline P1 --arch a2 --seed 1 --alpha 0.3 --sample_target --run_dir P1_a2_alpha03_cible_s1
python scripts/train.py --pipeline P1 --arch a2 --seed 1 --alpha 0.2 --run_dir P1_a2_alpha02_s1
python scripts/train.py --pipeline P1 --arch a2 --seed 1 --alpha 0.1 --run_dir P1_a2_alpha01_s1
```

Comparaison appariée avec P0 (baseline = `val_hyp.txt` de `P0_a2_s1`, dans les résultats de Wadoud) :

```
python scripts/evaluate.py --hyp P1_a2_alpha03_s1/val_hyp.txt --ref P1_a2_alpha03_s1/val_ref.txt --baseline <P0_a2_s1/val_hyp.txt>
```

Les runs ont été exécutés sur GPU Kaggle (notebook `P1.ipynb`). La cellule qui modifie `train.py` ne doit être lancée qu'une fois par session.

## Résultats (validation, seed 1)
Référence P0 (Colab T4, seed 1) : A2 BLEU 3,77 / chrF 19,71 / val loss 3,919 ; A3 3,55 / 20,31.

| Run (dossier dans `resultats/`) | BLEU | chrF | val loss | Meilleure mise à jour | Arrêt | Epochs |
|---|---|---|---|---|---|---|
| A2, alpha 0,3 (`P1_a2_alpha03_s1`, 2e exécution) | 3,83 | 20,98 | 3,850 | 17 000 | plafond de temps | 22 |
| A2, alpha 0,3, source + cible (`P1_a2_alpha03_cible_s1`) | 3,00 | 19,92 | 4,058 | 16 000 | plafond de temps | 20 |
| A2, alpha 0,2 (`P1_a2_alpha02_s1`, 2e exécution) | 3,03 | 19,41 | 3,892 | 12 000 | early stopping | 18 |
| A2, alpha 0,1 (`P1_a2_alpha01_s1`, 2e exécution) | 2,82 | 18,77 | 3,967 | 16 000 | early stopping | 23 |
| A3, alpha 0,3 (`P1_a3_alpha03_s1`) | 3,97 | 21,09 | 3,939 | 13 000 | early stopping | 19 |
| A4, alpha 0,3 (`P1_a4_alpha03_s1`) | 3,24 | 19,93 | 3,864 | 12 000 | plafond de temps | 16 |

Premières exécutions (fichiers perdus, chiffres relevés à l'époque) : A2 alpha 0,3 : 4,26 / 21,06 / 3,840 ; alpha 0,2 : 3,91 / 20,65 / 3,863 ; alpha 0,1 : 3,44 / 19,25 / 3,957.

Temps par epoch : 46,4 s pour A2 et A3, 63,8 s pour A4.

Comparaison appariée avec P0 (A2 alpha 0,3, exécution à 3,83) : BLEU 3,83 contre 3,77, chrF 20,98 contre 19,71. Probabilité de gain sur 1 000 rééchantillonnages des phrases de validation : 0,603 (BLEU) et 1,0 (chrF). Elle mesure la variabilité due aux phrases, pas celle d'un entraînement à l'autre.

## Conclusions
- Le bruit entre deux exécutions identiques est grand (0,4 à 0,9 BLEU) : les écarts de BLEU plus petits ne sont pas interprétables.
- L'ordre des alphas se reproduit dans les deux exécutions (0,3 > 0,2 > 0,1 sur BLEU, chrF et val loss). Alpha 0,1 est sous P0 les deux fois.
- Alpha 0,3 : le chrF dépasse P0 d'environ 1,3 point dans les deux exécutions (+1,35 et +1,27) et la val loss est meilleure (3,840 et 3,850 contre 3,919). Le gain de BLEU (+0,49 et +0,06) n'atteint pas le seuil d'environ 1 point fixé par l'équipe. Tendance favorable, pas de gain net établi.
- A3 : pas d'écart net avec A2, on garde A2. A4 : moins bon que A2. Source + cible : moins bon.
- Les erreurs observées (voir `exemples.md`) sont des phrases courtes correctes ou fausses de sens, et des répétitions sur les phrases longues d'actualité (environ 8 % des hypothèses).

## Limites
- Une seule graine pour la plupart des runs.
- P0 mesuré sur Colab, P1 sur Kaggle.
- Trois runs (A2 alpha 0,3, source + cible, A4) s'arrêtent au plafond de 17 minutes avec leur meilleur point à 15,4 à 16,5 minutes : ils étaient sans doute encore en progression. La mauvaise performance d'A4 peut venir du plafond et pas de l'architecture. A3 s'arrête par early stopping, ce qui rend la comparaison A3 / A2 inégale.
- Le champ `alpha` n'est pas enregistré dans `results.json` : les runs sont identifiés par le nom de leur dossier.
- La cause du bruit (GPU non reproductible, arrêt au plafond à des endroits différents) est une hypothèse.
- A4 a été écrit à partir du PDF, pas fourni par Bénito.
- Environ 5 % des phrases de validation et de test ont une traduction française identique à une phrase du train. Le chrF d'entraînement est plus haut que le chrF final (écart non expliqué). Le tokenizer fongbé à 2 000 tokens est un choix de Wadoud.
- Une grande partie du code a été écrite avec l'aide d'une IA (Claude), puis appliquée et testée par moi.