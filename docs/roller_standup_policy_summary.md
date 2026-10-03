# Policy `roller_standup` — se relever sur rollers

**But** : le microduck (sur rollers) part du sol — à plat ventre ou à plat dos — et se remet **debout sur ses roues**, puis **tient** la station.

- **Tâche** : `Mjlab-RollerStandUp-Flat-MicroDuck`
- **Fichier** : `src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py`
- **Base** : dérivée de l'env roller (`velocity_rollers`) → même robot, même physique/DR, **même observation 61D** (interchangeable au runtime, chargeable via `--new-cmd-obs`).
- **Spec** : `docs/superpowers/specs/2026-08-04-roller-standup-design.md`
- **Politique aveugle** : pas de scan de terrain ; proprioception + `projected_gravity`.

## Hauteurs (mesurées, pas devinées)

| pose | modèle pieds | modèle rollers |
|---|---|---|
| debout | 0.1172 → `STAND_Z=0.115` sous charge | 0.1407 → **`ROLLER_STAND_Z=0.138`** |
| à plat ventre (repos) | 0.075 | 0.075 |
| à plat dos (repos) | 0.048 | 0.048 |

Les hauteurs de repos au sol sont identiques aux deux modèles : c'est la coque du tronc qui touche, pas les pieds.

## ⚠️ Indices de joints — les roues sont INTERCALÉES

```
0-4   jambe gauche      5-6   roues gauches
7-10  cou / tête       11-15  jambe droite      16-17  roues droites
```
Cet ordre est celui du tableau de joints d'**entité**. Mais les récompenses de pose ne l'indexent pas : `pose_target_match` / `pose_l1_penalty` passent par `_servo_joint_pos()`, qui applique déjà `find_joints("^(?!passive_).*")` et rend une vue à **14 colonnes sans les roues**. Dans cette vue le décalage disparaît et la disposition redevient celle du modèle sans roues :

```
0-4  jambe gauche      5-8  cou / tête      9-13  jambe droite
```

D'où `_LEG_JOINTS = [0-4, 9-13]` et `_NECK_JOINTS = [5-8]` — les mêmes que `standup` et `sitstand`. Écrire ici les indices d'entité (`[0-4, 11-15]`) déborde la vue servo : 14 et 15 sont hors bornes et l'env meurt au premier calcul de récompense (`IndexError` sur CPU, device-side assert CUDA sur GPU). C'est ce qui est arrivé entre le 11/08 et le correctif : la migration vers la vue servo (`fa1b2ac`) et cet env ont été développés sur des branches parallèles, et leur fusion a laissé le cfg en indices d'entité.

Verrouillé par `tests/test_roller_standup_cfg.py::test_joint_indices_are_servo_space` (les noms, pour cet env) et `tests/test_joint_index_bounds.py` (les bornes, pour tous les envs).

## Reset — départ au sol

`set_random_ground_state` : ventre (`prone_z` 0.076–0.09, plancher relevé car le ventre ne décolle du sol qu'à 0.0752) / dos / **déjà debout** (`standing_z` 0.134–0.144), ± 10° de bruit en pitch/roll. Pas de bucket « assis ». Le bucket « debout » est nécessaire : sans lui la policy monte mais ne tient pas.

**Curriculum `ground_state_mix`** (easy → hard, le dos en dernier) :

| iter | debout | ventre | dos |
|---|---|---|---|
| 0 | 0.50 | 0.50 | 0.00 |
| 600 | 0.35 | 0.45 | 0.20 |
| 1500 | 0.25 | 0.40 | 0.35 |
| 2500 | 0.20 | 0.40 | 0.40 |

## Récompenses

Dix termes repris du `standup` avec leurs poids déjà réglés : `pose_stand_legs` (+8), `pose_stand_l1` (+5), `height_stand` (+4, std 0.04), `height_stand_sharp` (+4, std 0.015), `height_stand_l1` (+30), `com_upward_velocity` (+3), `gentle_rise` (−0.02), `upright_linear` (+6), `upright_sharp` (+6), `standing_composite` (+15). Plus `joint_torque_rate_l2` (−2e-3), l'anti-jitter qui n'empêche pas le retournement.

Régularisateurs hérités : `body_ang_vel` **−0.05** (bloqueur de mouvement, à garder LÉGER), `angular_momentum` −0.02, `action_rate_l2` (rampe −0.4 → −1.0, **pas** le −2.0 du roller), `neck_action_rate_l2` −0.5, `neck_joint_pos_l2` −0.5 (tête droite), `joint_torques_l2` −1e-3, `action_over_limit` −0.5, `self_collisions` −1.0.

Retirées : toutes les récompenses de patinage, plus `feet_flat` (les lames ne sont pas à plat pendant la montée) et `hip_roll_neutral` (se relever demande d'écarter les jambes).

## ⚠️ Le point dur : les roues roulent

Aucune adhérence longitudinale pour pousser sur le sol. Le **curriculum de friction de roulement est INVERSÉ** (l'env roller la fait monter, ici elle descend) :

| iter | frictionloss | |
|---|---|---|
| 0 | 0.05 | roues quasi bloquées → se relève comme avec des pieds |
| 1000 | 0.02 | |
| 2000 | 0.008 | |
| 3000 | 0.003 | |
| 4000 | 0.0015 | la vraie valeur du roulement |

**Surveiller `Episode_Reward/standing_composite` aux paliers.** S'il s'écroule, le geste « pieds adhérents » ne transfère pas aux roues libres → il faudra guider une technique de patineur (appui genou intermédiaire, un patin à la fois). C'est un résultat, pas un échec.

**Surveiller AUSSI la dérive horizontale du robot en play**, à chaque palier de friction. `standing_composite` ne voit ni `root_link_pos_w[:2]` ni la vitesse horizontale : une policy qui se relève en glissant loin de son point de départ collecte exactement le même score qu'une qui se relève et s'arrête. Tant que cette dérive n'a pas été mesurée visuellement, le résultat du curriculum de friction (la question même que cet env existe pour trancher) n'est pas fiable.

**Sim2real** : seuls les checkpoints d'après iter 4000 sont candidats au déploiement. Avant, la policy s'appuie sur une friction qui n'existe pas sur le vrai robot.

## Commande

Slot `twist` neutralisé : `lin_vel_x`/`lin_vel_y` ± 0.01, `ang_vel_z` **± 0.05** (5× plus large — même
choix que le `standup`). Slots `head_pose` / `body_pose` **zero-paddés** (convention roller). Déploiement visé : en `--standing` face à la policy roller en `--walking`, avec la bascule automatique sur la magnitude de la commande (`infer_policy.py:262`, seuil 0.05) ; le slot twist y est laissé à zéro (`infer_policy.py:239`).

**Réserve** : `infer_policy.py` est le script de sim/clavier local. Le runtime robot est le binaire Rust `microduck_runtime`, absent du repo — il n'est pas vérifié qu'il expose un équivalent `--standing`. Le doc de passation du crouch ne liste que `--model`, `--ground-pick`, `--fold-policy`. À confirmer.

## Terminaisons

`fell_over` **supprimée** (le robot démarre tombé). `nan_state` héritée. `nan_policy="sanitize"` sur les obs actor/critic.

## Réseau / PPO

Actor et critic `(512, 256, 128)` elu, `obs_normalization=True`. PPO `lr=1e-3` adaptive, `desired_kl=0.01`, `gamma=0.99`, `lam=0.95`, `num_steps_per_env=24`, épisode 6 s, `max_iterations=15000`. **Symétrie OFF** (`SYMMETRY_CFG` est câblé pour le layout 51D).

## Commandes

```bash
uv run train Mjlab-RollerStandUp-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 15000
uv run scripts/play_latest.py        # alias md-play
uv run scripts/export_latest.py      # alias md-export
uv run --with pytest pytest tests/test_roller_standup_cfg.py -q
```

### ⚠️ Voir les départs sur le dos au play

Un play ne montre **jamais** de départ sur le dos par défaut : l'env de play est
reconstruit à neuf, donc `common_step_counter` repart à 0 et le curriculum applique son
palier 0, où `face_up_prob = 0`. On ne voit que 50 % ventre / 50 % debout, quelle que soit
la maturité du checkpoint chargé. Or le dos est le cas le plus dur, celui qu'on veut
justement inspecter.

`STANDUP_PLAY_FACE_UP` force le mélange (même motif que `SLOPE_PLAY_DIFFICULTY` dans
`roller_slope`), **uniquement sur le chemin `play=True`** — l'entraînement et son
curriculum easy → hard sont intouchés :

```bash
STANDUP_PLAY_FACE_UP=1.0 md-play    # 100 % de départs sur le dos
STANDUP_PLAY_FACE_UP=0.4 md-play    # le mélange du dernier palier du curriculum
STANDUP_PLAY_FACE_UP=none md-play   # défaut (palier 0, pas de dos)
```

Le reste (`1 - face_up`) est réparti ventre:debout dans le rapport 2:1 du dernier palier,
si bien que `0.4` reproduit exactement le mélange de fin d'entraînement (0.40 / 0.20 / 0.40).

## 🔧 Correction anti-violence (après premier test robot)

**Symptômes** sur un checkpoint 4000+ : mouvements très brusques, la tête tape le sol,
échec du relevé depuis le dos sur le robot. **Présents en simu aussi** → ce n'était donc
ni du sim2real, ni un checkpoint trop jeune, mais la conception des récompenses.

**Root cause : `gentle_rise` récompensait la violence.** `trunk_vertical_accel_penalty`
renvoie déjà `-|a_z|` (`mdp.py:2171`) ; multiplié par le poids **−0.02** hérité du
`standup`, ça faisait un double négatif, donc `+0.02·|a_z|` — **plus le tronc accélérait
brutalement, plus la policy était payée**. Confirmé par le log : `Episode_Reward/gentle_rise
= +0.0118` sur le run `vweolw91`, seul terme de pénalité loggé positif.

`mdp.py` mélange deux conventions de signe, et c'est le piège :

| terme | la fonction renvoie | poids correct |
|---|---|---|
| `height_stand_l1`, `pose_stand_l1`, `gentle_rise` | `-abs(...)`, déjà négatif | **positif** |
| `joint_torques_l2`, `joint_torque_rate_l2`, `action_rate_l2`, `body_impact_cost` | magnitude positive | **négatif** |

Verrouillé par `test_already_negative_penalties_use_positive_weights`.

⚠️ **Le `standup` du marcheur a exactement le même bug** (même fonction, même poids −0.02).
Ça explique la série de tentatives d'amortissement infructueuses documentées dans ses
commentaires (« *violent / shaky / overshoot-tip-repeat on the real robot* ») : elles
combattaient un terme qui poussait activement dans l'autre sens. **Non corrigé ici** — c'est
un autre env, à trancher séparément.

**Problème structurel associé.** À convergence les récompenses de tâche totalisaient **≈ +41.6**
saturées à 95–99 %, contre **≈ −1.2** pour tous les amortisseurs réunis — dont
`joint_torque_rate_l2` à **−0.0002/pas** et `joint_torques_l2` à **−0.0001/pas**, soit rien.
Rapport ~35:1 : aucune raison d'être doux.

**État actuel des corrections :**

| | avant | maintenant | pourquoi |
|---|---|---|---|
| `gentle_rise` | −0.02 (récompense) | **+0.02** (pénalité) | signe corrigé ; magnitude gardée PETITE exprès — `\|a_z\|` est forcément élevé pendant un retournement, un gros poids serait un bloqueur de mouvement |
| `joint_torque_rate_l2` | −2e-3 | **−0.2** | le levier SÛR : pénalise la variation de couple, pas le mouvement |
| `head_impact_penalty` | absent | **toujours absent** | essayé à −1.0, a gelé la policy — voir ci-dessous |

### ⚠️ La pénalité d'impact tête a gelé la policy — ne pas la remettre telle quelle

Tentative avec les valeurs de `velstand` (`body_impact_cost`, sous-arbre `neck`, −1.0,
seuil 2.0) : **la policy a convergé vers rester couchée, inerte.** Mesuré (run `d8rnko6p`) :

| terme | avant (violent) | avec head_impact (gelé) |
|---|---|---|
| `standing_composite` | +14.32 | **+3.26** |
| `upright_sharp` | +5.76 | +1.06 |
| `head_impact_penalty` | — | **−1.01** ← plus gros terme négatif |
| `joint_torque_rate_l2` | −0.0002 | −0.255 (donc **pas** le coupable) |

L'erreur de raisonnement : croire qu'une pénalité « ciblée » ne bride pas le mouvement.
**Faux ici — pour se relever du dos, ce robot pivote sur sa tête et ses épaules.** La tête
est le point d'appui du retournement, pas un dégât collatéral ; la pénaliser bloque le seul
mécanisme disponible, et le dos était déjà le cas qui échouait.

**L'optimum paresseux qui rend ce gel possible** : `pose_stand_legs` restait à **+7.72 sur 8**
alors que le robot était allongé — les jambes sont à HOME en position couchée, donc cette
récompense est encaissée quasi gratuitement. C'est `height_stand_l1` (poids +30) qui doit
rendre « rester au sol » net négatif ; il ne faut pas l'affaiblir.

**Hypothèse en cours de test** : taper la tête était un *symptôme* de la violence (le bug de
signe payait la brutalité, et une montée brutale finit sur la tête), pas un défaut séparé.
Si le slam revient maintenant que le signe est corrigé, la reprise doit être une pénalité
**gatée en hauteur** (comme `upright_sharp` l'est), qui épargne la phase de retournement au sol.

**Leçon de méthode** : les trois corrections ont été appliquées d'un coup, donc le gel n'a pas
pu être attribué avec certitude — seul le suspect le plus probable a pu être désigné. Une
correction à la fois, à l'avenir.

**Recalibrage si c'est encore violent** : `|Δτ|²` vaut ~0.1 à convergence, donc la
contribution de `joint_torque_rate_l2` ≈ `0.1 × |poids|`. Monter **ce** terme, **pas**
`body_ang_vel` (−0.05) ni `action_rate_l2` (rampe → −1.0) : ceux-là sont des bloqueurs de
mouvement et le `standup` documente qu'à −0.15 et −1.2 respectivement, ils **gelaient** le
relevé depuis le dos. Si au contraire le dos cesse de fonctionner, **baisser**
`joint_torque_rate_l2` en premier.

## Hors périmètre

Intégrer le relevé dans la policy de roulage (recette `velstand`) ; buckets de départ sur le côté ; variante rough ; pénalités d'impact tronc/tête.

Aucune récompense ne pénalise la vitesse horizontale du tronc (`root_link_lin_vel_w[:, :2]`) : « se relever en roulant loin » est un résultat non pénalisé et qui score à plein. Décision volontaire (pas un oubli) : une récompense d'immobilité qui ne serait pas gatée en hauteur pénaliserait aussi la translation que le relevé depuis le sol exige physiquement — le mode d'échec « bloqueur de mouvement » que le `standup` documente. Candidat si le problème se confirme : une immobilité gatée en hauteur (proche de `ROLLER_STAND_Z` seulement).
