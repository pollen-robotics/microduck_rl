"""Verrou transverse : tout indice de joint d'un cfg vit dans l'espace SERVO.

Les fonctions de mdp.py qui prennent des indices (``joint_indices``,
``target_overrides``, ``sitting_joint_overrides``, ``sit_overrides``,
``tuck_overrides`` — cf. _INDEX_*_PARAMS) passent toutes par
``_servo_joint_pos`` / ``_servo_default_joint_pos``, qui appliquent déjà
``find_joints(r"^(?!passive_).*")``. Elles reçoivent donc une vue à 14 colonnes
— la disposition canonique des 14 servos — quel que soit le modèle.

Sur les modèles à joints passifs (rollers, backlash), le tableau de joints de
l'ENTITÉ est plus large et intercalé. Écrire les indices dans cet espace-là
donne, selon la valeur :

  - un IndexError / device-side assert CUDA si l'indice dépasse 14 (c'est le bug
    qu'a eu roller_standup : _LEG_JOINTS en indices d'entité, donc 14 et 15 hors
    bornes, l'env plantait au premier calcul de récompense) ;
  - pire, une récompense SILENCIEUSEMENT appliquée au mauvais joint si l'indice
    reste sous 14.

CE test ne couvre que le PREMIER cas : c'est une vérification de bornes, elle ne
peut pas savoir quel joint un terme VOULAIT viser. Le second cas se vérifie par
NOM, et donc par env — cf. test_joint_indices_are_servo_space dans
tests/test_roller_standup_cfg.py, qui compare les noms de joints attendus.
"""

import pytest

from mjlab_microduck.tasks import (
    make_microduck_ball_kick_env_cfg,
    make_microduck_ground_pick_env_cfg,
    make_microduck_roller_crouch_env_cfg,
    make_microduck_roller_slope_env_cfg,
    make_microduck_roller_standup_env_cfg,
    make_microduck_roulade_env_cfg,
    make_microduck_sitstand_env_cfg,
    make_microduck_spin_env_cfg,
    make_microduck_standup_env_cfg,
    make_microduck_velocity_env_cfg,
    make_microduck_velocity_rollers_env_cfg,
    make_microduck_velocity_swizzle_env_cfg,
    make_microduck_velstand_env_cfg,
)

# La disposition canonique : 14 servos, cf. AGENTS.md (« Joint layout »).
N_SERVO = 14

# Les paramètres de terme qui portent des indices de joints dans l'espace servo.
_INDEX_LIST_PARAMS = ("joint_indices",)
_INDEX_DICT_PARAMS = (
    "target_overrides",
    "sitting_joint_overrides",
    "sit_overrides",
    "tuck_overrides",
)

_FACTORIES = {
    "ball_kick": make_microduck_ball_kick_env_cfg,
    "ground_pick": make_microduck_ground_pick_env_cfg,
    "roller_crouch": make_microduck_roller_crouch_env_cfg,
    "roller_slope": make_microduck_roller_slope_env_cfg,
    "roller_standup": make_microduck_roller_standup_env_cfg,
    "roulade": make_microduck_roulade_env_cfg,
    "sitstand": make_microduck_sitstand_env_cfg,
    "spin": make_microduck_spin_env_cfg,
    "standup": make_microduck_standup_env_cfg,
    "velocity": make_microduck_velocity_env_cfg,
    "velocity_rollers": make_microduck_velocity_rollers_env_cfg,
    "velocity_swizzle": make_microduck_velocity_swizzle_env_cfg,
    "velstand": make_microduck_velstand_env_cfg,
}


def _indices_in(params):
    """(nom du paramètre, indices) pour chaque paramètre porteur d'indices."""
    if not params:
        return
    for key in _INDEX_LIST_PARAMS:
        value = params.get(key)
        if value:
            yield key, list(value)
    for key in _INDEX_DICT_PARAMS:
        value = params.get(key)
        if value:
            yield key, list(value.keys())


def _terms_of(cfg):
    """(famille, nom du terme, params) sur récompenses ET événements."""
    for family in ("rewards", "events"):
        container = getattr(cfg, family, None)
        if container is None:
            continue
        for name, term in container.items():
            yield family, name, getattr(term, "params", None)


@pytest.mark.parametrize("task", sorted(_FACTORIES))
def test_joint_indices_stay_in_servo_space(task):
    cfg = _FACTORIES[task]()
    for family, name, params in _terms_of(cfg):
        for key, indices in _indices_in(params):
            assert min(indices) >= 0, f"{task}.{family}.{name}.{key}: indice négatif {indices}"
            assert max(indices) < N_SERVO, (
                f"{task}.{family}.{name}.{key} = {indices} sort de la vue servo "
                f"({N_SERVO} colonnes). Les fonctions de mdp.py indexent "
                f"_servo_joint_pos(), d'où les joints passive_* sont DÉJÀ retirés : "
                f"les indices d'entité (roues/backlash intercalés) n'y valent pas."
            )


# Nombre de paramètres d'indices que le scan trouve aujourd'hui, par nom. C'est
# un PLANCHER anti-régression, pas une spec : en ajouter fait monter le chiffre
# (à mettre à jour), en perdre veut dire que le scan est devenu muet — c'est le
# cas qu'on veut attraper. Un simple "total > 0" ne suffit pas : renommer
# joint_indices en laisserait passer 6 tout en vidant 15 assertions.
_EXPECTED_MIN_PARAMS = {
    "joint_indices": 15,
    "sit_overrides": 3,
    "sitting_joint_overrides": 2,
    "tuck_overrides": 1,
}


def test_the_scan_actually_finds_indices():
    """Garde-fou : sans lui, test_joint_indices_stay_in_servo_space peut passer
    au vert en n'ayant rien vérifié du tout (paramètres renommés côté cfg)."""
    seen = {}
    for _task, factory in _FACTORIES.items():
        cfg = factory()
        for _family, _name, params in _terms_of(cfg):
            for key, _indices in _indices_in(params):
                seen[key] = seen.get(key, 0) + 1
    for key, expected in _EXPECTED_MIN_PARAMS.items():
        assert seen.get(key, 0) >= expected, (
            f"le scan ne trouve plus que {seen.get(key, 0)} '{key}' au lieu de "
            f"{expected} : paramètre renommé ? Le verrou de bornes est devenu "
            f"muet pour d'autant de termes."
        )
