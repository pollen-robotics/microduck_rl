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

## ⚠️ Indices de joints — vue SERVO-ONLY (14 joints)

Les récompenses de pose indexent via `mdp._servo_joint_pos`, qui sélectionne
`^(?!passive_).*` : les 14 servos, **sans** les roues ni les charnières de backlash. Les
indices s'écrivent donc dans cette vue canonique, identique au marcheur :

```
0-4   jambe gauche      5-8   cou / tête      9-13  jambe droite
```
`_LEG_JOINTS = [0-4, 9-13]`, `_NECK_JOINTS = [5-8]`. Plus de `_WHEEL_JOINTS` : les roues
n'existent pas dans cette vue, et leur DR les cible par la regex `^passive_.*wheel`.

**Ne pas « corriger » vers les positions du tableau complet** (`[0-4, 11-15]`, roues
intercalées en 5,6 et 16,17) : c'était juste avant la migration de `mdp.py`, et ça rend l'env
inentraînable depuis. Voir la régression décrite en bas de ce document. Verrouillé sur les
modèles rollers ET rollers+backlash par
`tests/test_roller_standup_cfg.py::test_joint_indices_are_in_the_canonical_servo_space`.

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

**Bloc de tâche**, aligné sur la recette évoluée du `standup` (poids à 1/4 de la version
initiale ; les ratios internes et tous les `std` sont inchangés) :

| reward | poids | |
|---|---|---|
| `height_stand_l1` | **7.5** | doit dominer le bloc — c'est lui qui rend « rester au sol » net négatif |
| `standing_composite` | 3.75 | score multiplicatif hauteur × droit × pose |
| `pose_stand_legs` | 2.0 | pose cible = HOME (std 0.5) |
| `upright_linear` / `upright_sharp` | 1.5 / 1.5 | `upright_sharp` gatée en hauteur, std 0.3 |
| `pose_stand_l1` | 1.25 | bootstrap L1 |
| `height_stand` / `height_stand_sharp` | 1.0 / 1.0 | std 0.04 (large) et 0.015 (serrée) |
| `com_upward_velocity` | 0.75 | paye la montée, coupée à `ROLLER_STAND_Z + 0.010` |
| `gentle_rise` | **+0.005** | poids POSITIF (voir bug de signe plus bas) ; 0.005 = plafond mesuré |

**Anti-violence, introduits seulement à l'itération 3000** (voir la leçon de timing plus bas) :
`arrival_damping` (0 → −0.025 → −0.05) et `joint_torque_rate_l2` (0 → −1e-3).

Régularisateurs hérités : `body_ang_vel` **−0.05** (bloqueur de mouvement, à garder LÉGER),
`angular_momentum` −0.02, `action_rate_l2` (base −0.1, rampe douce jusqu'à −1.0 **à 1500**),
`neck_action_rate_l2` −0.5, `neck_joint_pos_l2` −0.5 (tête droite), `joint_torques_l2` −1e-3,
`action_over_limit` −0.5, `self_collisions` −1.0.

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

## 🔧 Historique des échecs et de la resynchronisation

### Bug de signe : `gentle_rise` récompensait la violence

**Symptômes** (checkpoint 4000+) : mouvements très brusques, la tête tape le sol,
échec du relevé depuis le dos. **Présents en simu aussi** → ni sim2real, ni checkpoint jeune.

`trunk_vertical_accel_penalty` renvoie déjà `-|a_z|` (`mdp.py`) ; multiplié par le poids
**−0.02** hérité du `standup`, ça donnait `+0.02·|a_z|` — **plus le tronc accélérait
brutalement, plus la policy était payée**. Confirmé : `Episode_Reward/gentle_rise = +0.0118`
sur le run `vweolw91`, seul terme de pénalité loggé positif.

`mdp.py` mélange deux conventions de signe :

| terme | la fonction renvoie | poids correct |
|---|---|---|
| `height_stand_l1`, `pose_stand_l1`, `gentle_rise` | `-abs(...)`, déjà négatif | **positif** |
| `joint_torques_l2`, `joint_torque_rate_l2`, `action_rate_l2`, `body_impact_cost` | magnitude positive | **négatif** |

Verrouillé par `test_already_negative_penalties_use_positive_weights`. Le même bug existait
dans `standup` et `sitstand` (run `7ev90yd9`) ; **les deux ont été corrigés depuis**.

### Échec n°1 : pénalité d'impact tête → policy gelée

Essayée avec les valeurs de `velstand` (−1.0, seuil 2.0) : **la policy a convergé vers rester
couchée, inerte.** Mesuré (run `d8rnko6p`) : `head_impact_penalty` −1.01/pas, plus gros terme
négatif, `standing_composite` effondré de +14.3 à +3.3.

**L'optimum paresseux qui rend ce gel possible** : `pose_stand_legs` restait à **+7.72 sur 8**
alors que le robot était allongé — les jambes sont à HOME en position couchée, donc la
récompense est encaissée quasi gratuitement. `height_stand_l1` est le terme qui contrebalance
ça (verrouillé par `test_height_l1_stays_the_dominant_task_term`).

### La vraie leçon : c'est le TIMING, pas la magnitude

Le `standup` a établi la loi générale sur deux runs cassés : *« the same weights active from
step 0 prevent the flips from ever being DISCOVERED (attempt-tax on exploration) »*, et
*« the fix is timing, not magnitude »*. Toute taxe sur les tentatives pendant la phase de
découverte fait gagner « ne rien faire ». Les deux ajouts de cet env (`head_impact_penalty`
à −1.0 **et** `joint_torque_rate_l2` à −2.0) étaient actifs dès le pas 0.

### État actuel — recette resynchronisée sur `standup`

| | avant | maintenant |
|---|---|---|
| bloc de tâche entier | poids ×4 | **÷4** (`standing_composite` 3.75, `pose_stand_legs` 2.0, `height_stand_l1` 7.5…) |
| `gentle_rise` | −0.02 (récompense) | **+0.005** — plafond mesuré : 0.01 contribuait au gel |
| `com_upward_velocity` | 3.0 | **0.75** |
| `arrival_damping` | absent | **`body_ang_vel_at_height`**, gaté hauteur+inclinaison, 0 → −0.025 à 3000 → −0.05 à 4000 |
| `joint_torque_rate_l2` | −0.2 dès le pas 0 | **0** → −1e-3 à 3000 |
| rampe `action_rate_l2` | −0.4 → −1.0 dès 500 | **−0.1 → −1.0 à 1500** |
| `head_impact_penalty` | testé à −1.0 | **absent** |

Diviser la tâche plutôt que monter les amortisseurs corrige le rapport tâche/amortissement
(mesuré à ~35:1, maintenant ~9:1) **sans** transformer un amortisseur en bloqueur de mouvement.

`arrival_damping` vise la boucle d'échec réelle — monter → dépasser la verticale → basculer →
recommencer. Ses portes de hauteur sont transposées sur `ROLLER_STAND_Z` (0.113 / 0.133) et
**pas** copiées du marcheur (0.09 / 0.11), qui ouvriraient la porte alors que le robot roller
est encore 3 cm sous sa station, donc en pleine montée. La porte d'inclinaison est
indispensable : sans elle, le redressement final d'une montée pliée est lui-même une grande
rotation, et la taxer dresse un mur juste avant l'arrivée.

⚠️ **Si le relevé se dégrade après 3000, adoucir le DERNIER palier — ne pas avancer
l'introduction.**

### 🐛 Régression : indices de joints hors bornes

La migration de `mdp.py` vers `_servo_joint_pos` a rendu l'env **inentraînable** sans que rien
ne le signale. `joint_indices` s'interprète désormais dans la vue **servo-only à 14 joints**
(`^(?!passive_).*`, donc sans roues ni backlash) ; les constantes visaient le tableau complet à
18 joints, donc les indices 14 et 15 sortaient des bornes → `index out of bounds` sur GPU.

**Les 37 tests de config passaient quand même** : ils construisent `cfg` sans jamais appeler
les récompenses. Seul un vrai run le révèle — c'est la limite structurelle de ces tests, et il
faut lancer 3 itérations réelles après tout changement d'indices ou de capteur.

Bénéfice : la vue servo-only est **identique** sur le modèle rollers et sur rollers+backlash
(32 joints dont 18 passifs), donc les indices sont maintenant robustes au backlash — vérifié
sur les deux modèles par `test_joint_indices_are_in_the_canonical_servo_space`.

### Leçon de méthode

Les trois premières corrections ont été appliquées d'un coup, donc le gel n'a pas pu être
attribué avec certitude. Une correction à la fois.

## Hors périmètre

Intégrer le relevé dans la policy de roulage (recette `velstand`) ; buckets de départ sur le côté ; variante rough ; pénalités d'impact tronc/tête.

Aucune récompense ne pénalise la vitesse horizontale du tronc (`root_link_lin_vel_w[:, :2]`) : « se relever en roulant loin » est un résultat non pénalisé et qui score à plein. Décision volontaire (pas un oubli) : une récompense d'immobilité qui ne serait pas gatée en hauteur pénaliserait aussi la translation que le relevé depuis le sol exige physiquement — le mode d'échec « bloqueur de mouvement » que le `standup` documente. Candidat si le problème se confirme : une immobilité gatée en hauteur (proche de `ROLLER_STAND_Z` seulement).
