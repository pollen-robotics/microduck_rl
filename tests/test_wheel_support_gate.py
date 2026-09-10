"""Unit tests for mdp.wheel_support_gate — the roller-standup support gate.

The gate is what makes "standing" mean standing ON THE WHEELS. Run fmt83tri
converged to a head tripod (head planted, trunk levered to standing height at
~55 deg of tilt) that collected 48 % of the maximum task stack without ever
standing, so these cases are the failure mode itself, written down.

Contact tensors are faked: mdp reads sensors through `_sensor_any_contact`,
which only touches `env.scene.sensors[name].data.found`.
"""

import torch

from mjlab_microduck.tasks import mdp


class _FakeSensorData:
    def __init__(self, found):
        self.found = found


class _FakeSensor:
    def __init__(self, found):
        self.data = _FakeSensorData(found)


class _FakeScene:
    def __init__(self, sensors):
        self.sensors = sensors


class _FakeEnv:
    """Minimal env exposing only what the gate reads."""

    def __init__(self, num_envs, contacts):
        self.num_envs = num_envs
        self.device = "cpu"
        self.scene = _FakeScene(
            {
                name: _FakeSensor(torch.as_tensor(v, dtype=torch.float32))
                for name, v in contacts.items()
            }
        )


def _gate(contacts, num_envs=1):
    return mdp.wheel_support_gate(_FakeEnv(num_envs, contacts))


def test_wheels_only_opens_the_gate():
    # Un pneu au sol, ni tête ni tronc : c'est la station sur roues.
    g = _gate(
        {
            "feet_ground_contact": [[1.0]],
            "head_ground_contact": [[0.0]],
            "trunk_ground_contact": [[0.0]],
        }
    )
    assert g.tolist() == [1.0]


def test_head_tripod_scores_zero():
    # LE mode d'échec : pneus au sol ET tête au sol -> aucune récompense d'état-but.
    g = _gate(
        {
            "feet_ground_contact": [[1.0]],
            "head_ground_contact": [[1.0]],
            "trunk_ground_contact": [[0.0]],
        }
    )
    assert g.tolist() == [0.0]


def test_trunk_prop_scores_zero():
    # Le hack suivant si on ne gatait que la tête : s'appuyer sur la batterie.
    g = _gate(
        {
            "feet_ground_contact": [[1.0]],
            "head_ground_contact": [[0.0]],
            "trunk_ground_contact": [[1.0]],
        }
    )
    assert g.tolist() == [0.0]


def test_airborne_scores_zero():
    # Aucun appui : pas de station, donc pas de paiement (anti-ballistique).
    g = _gate(
        {
            "feet_ground_contact": [[0.0]],
            "head_ground_contact": [[0.0]],
            "trunk_ground_contact": [[0.0]],
        }
    )
    assert g.tolist() == [0.0]


def test_any_wheel_counts_not_all():
    """Un seul slot en contact suffit.

    Exiger les deux pieds ferait de la porte un fil du rasoir qui se coupe au
    moindre instant d'appui unilatéral — un robot debout sur roues en a.
    """
    g = _gate(
        {
            "feet_ground_contact": [[1.0, 0.0]],
            "head_ground_contact": [[0.0]],
            "trunk_ground_contact": [[0.0]],
        }
    )
    assert g.tolist() == [1.0]


def test_gate_is_per_env():
    g = _gate(
        {
            # env0 : debout sur roues. env1 : trépied. env2 : à plat sur le tronc.
            "feet_ground_contact": [[1.0], [1.0], [1.0]],
            "head_ground_contact": [[0.0], [1.0], [0.0]],
            "trunk_ground_contact": [[0.0], [0.0], [1.0]],
        },
        num_envs=3,
    )
    assert g.tolist() == [1.0, 0.0, 0.0]


def test_missing_sensors_degrade_to_all_ones():
    """Un env qui ne déclare aucun de ces capteurs n'est pas neutralisé.

    Les variantes gatées ont des noms de capteurs par défaut ; si un autre env
    les appelait sans déclarer les capteurs, une porte fermée par défaut
    annulerait silencieusement ses récompenses. Elle s'ouvre donc.
    """
    g = _gate({}, num_envs=2)
    assert g.tolist() == [1.0, 1.0]


def test_gated_variants_multiply_the_base_reward():
    """La variante gatée = terme de base × porte, sans autre changement.

    Vérifié en montant une porte fermée et une porte ouverte sur les MÊMES
    données : le rapport doit être exactement 0 et l'identité.
    """
    # Porte fermée par la tête -> 0 ; porte ouverte -> valeur de base inchangée.
    closed = _gate(
        {"feet_ground_contact": [[1.0]], "head_ground_contact": [[1.0]]}
    )
    opened = _gate(
        {"feet_ground_contact": [[1.0]], "head_ground_contact": [[0.0]]}
    )
    assert closed.tolist() == [0.0]
    assert opened.tolist() == [1.0]
