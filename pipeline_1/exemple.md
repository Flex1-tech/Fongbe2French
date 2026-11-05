# Exemples de traductions (validation, A2, alpha 0,3, seed 1)

Run : `resultats/P1_a2_alpha03_s1` (BLEU 3,83 / chrF 20,98). Numéros de ligne dans `val_hyp.txt` / `val_ref.txt` (la ligne 1 est la première).

## Traductions correctes
| Ligne | Fongbé → hypothèse | Référence |
|---|---|---|
| 193 | je t'ai aperçu de loin | je t'ai aperçu de loin |
| 232 | ton père est à la maison | ton père est à la maison |
| 458 | cotonou est une grande ville | cotonou est une grande ville |
| 115 | je vais travailler | je m'en vais travailler |

La ligne 115 est correcte pour le sens, mais la nuance de la référence est perdue.

## Erreurs de sens sur des phrases courtes
| Ligne | Hypothèse | Référence |
|---|---|---|
| 5 | j'ai dix ans avec dix enfants | j'ai dix femmes et j'ai trente enfants. |
| 2 | chatre | papaye |
| 214 | j'ai des soucis | une abeille m'a piqué |

Le modèle produit une phrase fluide mais fausse (ligne 5), un mot inventé (ligne 2) ou une phrase sans rapport (ligne 214).

## Phrases longues d'actualité (répétitions)
| Ligne | Hypothèse | Référence |
|---|---|---|
| 106 | selon le directeur général de la maladie, les deux cas confirmés sont mortes, les deux cas confirmés sont mortes. | on compte désormais 57 personnes sont guéries, 71 sous traitement et deux (02) décès. |

Le style journalistique est reconnu, mais les chiffres et le sens se perdent, et la fin de la phrase est répétée.

## Chiffres sur les 2 242 phrases de validation
- 59 hypothèses sont identiques à la référence (comparaison hors espaces et points en début et fin de phrase).
- 172 (environ 8 %) présentent une répétition marquée (moins de 60 % de trigrammes de mots distincts).
- Lecture : les phrases courtes et fréquentes sont mieux traitées que les phrases longues, ce qui est cohérent avec un chrF (20,98) bien plus élevé que le BLEU (3,83).