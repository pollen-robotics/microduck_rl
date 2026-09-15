import pytest

from mjlab_microduck.tasks.microduck_roller_standup_env_cfg import (
    EPISODE_LENGTH_S,
    make_microduck_roller_standup_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
)

# Récompenses de PATINAGE : elles ne doivent pas survivre dans un env de relevé.
SKATING_REWARDS = (
    "wheel_speed",
    "braking",
    "skating_air_time",
    "glide",
    "single_support",
    "gait_symmetry",
    "forward_lean",
    "heading_hold",
    "feet_flat",
    "hip_roll_neutral",
    "pose",
    "com_height_target",
    "upright",
)


def test_env_builds_train_and_play():
    assert make_microduck_roller_standup_env_cfg() is not None
    assert make_microduck_roller_standup_env_cfg(play=True) is not None


def test_episode_is_short():
    # Épisode court : monter puis stabiliser, comme standup (6 s).
    cfg = make_microduck_roller_standup_env_cfg()
    assert cfg.episode_length_s == EPISODE_LENGTH_S == 6.0


def test_no_skating_rewards_survive():
    cfg = make_microduck_roller_standup_env_cfg()
    for name in SKATING_REWARDS:
        assert name not in cfg.rewards, f"reward de patinage survivante : {name}"


def test_smoothness_regularisers_kept():
    # Gardées de l'héritage roller : le relevé a besoin de douceur sim2real, mais
    # body_ang_vel doit rester LÉGER (standup documente qu'à -0.15 il gelait).
    cfg = make_microduck_roller_standup_env_cfg()
    for name in (
        "action_over_limit",
        "self_collisions",
        "body_ang_vel",
        "angular_momentum",
        "action_rate_l2",
        "neck_action_rate_l2",
        "neck_joint_pos_l2",
        "joint_torques_l2",
    ):
        assert name in cfg.rewards, f"régularisateur perdu : {name}"
    assert cfg.rewards["body_ang_vel"].weight == -0.05


def test_twist_command_is_neutralised():
    # Pas de pilotage : la policy se déploie en --standing, où le runtime laisse
    # le slot twist à zéro (cf. infer_policy.py:239).
    cfg = make_microduck_roller_standup_env_cfg()
    cmd = cfg.commands["twist"]
    assert cmd.ranges.lin_vel_x == (-0.01, 0.01)
    assert cmd.ranges.lin_vel_y == (-0.01, 0.01)
    assert cmd.ranges.ang_vel_z == (-0.05, 0.05)
    assert cmd.heading_command is False
    assert cmd.ranges.heading is None
    assert cmd.rel_standing_envs == 0.0


def test_twist_command_is_not_heading_relative():
    # L'env roller installe un RelativeHeadingVelocityCommandCfg (cmd[2] = erreur
    # de cap, calculée en interne). Ici cmd[2] doit être un vrai zéro bruité.
    from mjlab_microduck.tasks import mdp as microduck_mdp

    cfg = make_microduck_roller_standup_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, microduck_mdp.VelocityCommandCommandOnlyCfg)
    assert not isinstance(cmd, microduck_mdp.RelativeHeadingVelocityCommandCfg)


def test_obs_nan_policy_sanitize():
    # Un contact rare fait diverger le free-joint en NaN : on assainit l'obs
    # plutôt que de tuer l'entraînement (même choix que roller_slope).
    cfg = make_microduck_roller_standup_env_cfg()
    assert cfg.observations["actor"].nan_policy == "sanitize"
    assert cfg.observations["critic"].nan_policy == "sanitize"


def test_obs_parity_with_roller_env():
    # Parité 61D obligatoire : sinon l'ONNX ne se charge pas dans un slot runtime.
    standup = make_microduck_roller_standup_env_cfg()
    roller = make_microduck_velocity_rollers_env_cfg()
    for grp in ("actor", "critic"):
        assert list(standup.observations[grp].terms.keys()) == list(
            roller.observations[grp].terms.keys()
        ), f"layout d'observation divergent sur le groupe {grp}"


def test_terrain_is_plain_plane():
    # Hérité de l'env roller : sol plat, pas de générateur. Pas de variante rough
    # pour cette v1.
    cfg = make_microduck_roller_standup_env_cfg()
    assert cfg.scene.terrain.terrain_type == "plane"
    assert cfg.scene.terrain.terrain_generator is None


def test_task_is_registered():
    from mjlab.tasks.registry import list_tasks

    import mjlab_microduck.tasks  # noqa: F401  (l'import déclenche l'enregistrement)

    assert "Mjlab-RollerStandUp-Flat-MicroDuck" in list_tasks()


def _servo_joint_names():
    """Les noms de joints dans l'espace SERVO du modèle rollers.

    Reproduit _servo_joint_ids() de mdp.py : joints d'entité (donc sans le
    free-joint), moins les ``passive_*``. Pur CPU, pas de sim.
    """
    import mujoco

    from mjlab_microduck.robot.microduck_constants import get_walk_rollers_spec

    model = get_walk_rollers_spec().compile()
    articulated = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
    ]
    return [n for n in articulated if not n.startswith("passive_")]


def test_joint_indices_are_servo_space():
    """Verrou sur le bug qui crashait l'env au premier calcul de récompense.

    Les roues passives sont intercalées dans l'ordre des joints d'ENTITÉ, mais
    pose_target_match / pose_l1_penalty passent par _servo_joint_pos(), qui les
    a déjà retirées : elles indexent une vue à 14 colonnes. Les indices d'entité
    ([0-4, 11-15]) y débordent — 14 et 15 sont hors bornes, ce qui déclenche un
    device-side assert CUDA (« index out of bounds ») dès le premier pas.

    Dans la vue servo, les roues n'existent plus : la disposition est celle du
    modèle sans roues, donc les mêmes indices que standup / sitstand.
    """
    from mjlab_microduck.tasks.microduck_roller_standup_env_cfg import (
        _LEG_JOINTS,
        _NECK_JOINTS,
    )

    servo = _servo_joint_names()
    assert len(servo) == 14, "la vue servo doit avoir exactement les 14 servos"

    assert [servo[i] for i in _LEG_JOINTS] == [
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    ]
    assert [servo[i] for i in _NECK_JOINTS] == [
        "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    ]
    # Aucun recouvrement, et les deux listes couvrent toute la vue servo.
    assert len(set(_LEG_JOINTS) | set(_NECK_JOINTS)) == len(servo)


def test_entity_joint_order_matches_the_documented_layout():
    """Épingle le bloc de commentaire d'ordre d'ENTITÉ du cfg.

    Le cfg documente où tombent les roues dans le tableau de joints d'entité
    (5-6 et 16-17) pour expliquer pourquoi ces indices-là ne valent pas côté
    récompense. C'est ce commentaire qui justifie tout le reste du bloc, et plus
    rien ne le vérifiait depuis la suppression de _WHEEL_JOINTS : sans ce test
    il peut dériver silencieusement si le modèle rollers est réexporté.

    Les bornes de la vue servo, elles, sont vérifiées pour TOUS les envs par
    tests/test_joint_index_bounds.py.
    """
    import mujoco

    from mjlab_microduck.robot.microduck_constants import get_walk_rollers_spec

    model = get_walk_rollers_spec().compile()
    articulated = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
    ]
    assert len(articulated) == 18
    assert [articulated[i] for i in (5, 6, 16, 17)] == [
        "passive_LF_wheel", "passive_LR_wheel", "passive_RF_wheel", "passive_RR_wheel",
    ]
    assert [articulated[i] for i in (0, 4, 7, 10, 11, 15)] == [
        "left_hip_yaw", "left_ankle",
        "neck_pitch", "head_roll",
        "right_hip_yaw", "right_ankle",
    ]


def test_recovery_rewards_present_with_expected_weights():
    cfg = make_microduck_roller_standup_env_cfg()
    expected = {
        "pose_stand_legs":      8.0,
        "pose_stand_l1":        5.0,
        "height_stand":         4.0,
        "height_stand_sharp":   4.0,
        "height_stand_l1":     30.0,
        "com_upward_velocity":  3.0,
        # gentle_rise : poids POSITIF. trunk_vertical_accel_penalty renvoie déjà
        # -|a_z|, donc un poids négatif en faisait une RÉCOMPENSE de la violence
        # (bug mesuré : Episode_Reward/gentle_rise loggée à +0.0118).
        "gentle_rise":         +0.02,
        "upright_linear":       6.0,
        "upright_sharp":        6.0,
        "standing_composite":  15.0,
        # -2e-3 ne contribuait que -0.0002/pas face à +41.6 de tâche : nul.
        # -2.0 a mesuré -0.255/pas (run d8rnko6p) — pas le gel, mais on redescend
        # à -0.2 pour dégager le budget d'amortissement pendant qu'on isole.
        "joint_torque_rate_l2": -0.2,
    }
    for name, weight in expected.items():
        assert name in cfg.rewards, f"récompense de relevé manquante : {name}"
        assert cfg.rewards[name].weight == weight, f"poids inattendu sur {name}"


def test_recovery_rewards_use_roller_heights_not_walker_heights():
    from mjlab_microduck.tasks.microduck_roller_standup_env_cfg import (
        ROLLER_PRONE_Z,
        ROLLER_STAND_Z,
    )

    cfg = make_microduck_roller_standup_env_cfg()
    assert ROLLER_STAND_Z == 0.138  # PAS le 0.115 du modèle sans roues
    for name in ("height_stand", "height_stand_sharp", "height_stand_l1"):
        assert cfg.rewards[name].params["target_height"] == ROLLER_STAND_Z
    assert cfg.rewards["standing_composite"].params["target_height"] == ROLLER_STAND_Z
    # com_upward_velocity se coupe juste AU-DESSUS de la cible (10 mm de marge),
    # sinon la policy se gare à l'altitude de coupure sans finir la montée.
    assert cfg.rewards["com_upward_velocity"].params["max_height"] == ROLLER_STAND_Z + 0.010
    # upright_sharp est gatée entre le repos au sol et la station debout.
    assert cfg.rewards["upright_sharp"].params["height_low"] == ROLLER_PRONE_Z
    assert cfg.rewards["upright_sharp"].params["height_high"] == ROLLER_STAND_Z


def test_pose_rewards_target_legs_only_at_roller_indices():
    from mjlab_microduck.tasks.microduck_roller_standup_env_cfg import _LEG_JOINTS

    cfg = make_microduck_roller_standup_env_cfg()
    for name in ("pose_stand_legs", "pose_stand_l1", "standing_composite"):
        assert cfg.rewards[name].params["joint_indices"] == _LEG_JOINTS
        # target_overrides=None → la cible est HOME (default_joint_pos).
        assert cfg.rewards[name].params["target_overrides"] is None


def test_trunk_asset_cfgs_are_distinct_objects():
    """mjlab résout et MUTE les SceneEntityCfg en place : un objet partagé entre
    plusieurs termes provoque des indices périmés. Chaque terme doit avoir le sien.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    names = (
        "height_stand", "height_stand_sharp", "height_stand_l1",
        "com_upward_velocity", "gentle_rise", "upright_linear",
        "upright_sharp", "standing_composite",
    )
    seen = [id(cfg.rewards[n].params["asset_cfg"]) for n in names]
    assert len(set(seen)) == len(seen), "asset_cfg partagé entre plusieurs termes"


def test_starts_from_ground_states():
    # Ventre + dos + debout. Pas de bucket "assis" : il n'existait dans standup
    # que pour le hand-off depuis la policy sit, dont il n'y a pas d'équivalent
    # roller — et ses sitting_joint_overrides sont des indices du modèle SANS roues.
    cfg = make_microduck_roller_standup_env_cfg()
    assert "set_ground_state" in cfg.events
    params = cfg.events["set_ground_state"].params
    assert params["sitting_prob"] == 0.0
    assert params["sitting_joint_overrides"] is None
    assert params["face_down_prob"] > 0.0
    assert params["standing_prob"] > 0.0
    # face_up (le dos) démarre à 0 : introduit tard par le curriculum.
    assert params["face_up_prob"] == 0.0


def test_ground_state_heights_are_roller_specific():
    cfg = make_microduck_roller_standup_env_cfg()
    params = cfg.events["set_ground_state"].params
    # Ventre et dos partagent une seule plage de z, mais leurs contacts diffèrent :
    # le ventre ne décolle du sol qu'à partir de 0.0752, le dos repose à 0.0475.
    # prone_z_min = 0.076 pour éliminer toute interpénétration côté ventre.
    assert (params["prone_z_min"], params["prone_z_max"]) == (0.076, 0.09)
    # Sous 0.0752 (contact mesuré, pose HOME), le départ ventre commence DANS le
    # sol — un pushout de contact que la policy paierait via gentle_rise /
    # joint_torque_rate_l2. prone_z_min doit rester au-dessus.
    assert params["prone_z_min"] >= 0.0752
    # Debout : hauteur ROLLER (+23 mm vs le modèle sans roues, qui est à 0.11–0.12).
    assert params["standing_z_min"] == 0.134
    assert params["standing_z_max"] == 0.144
    assert params["standing_z_min"] < 0.138 < params["standing_z_max"]


def test_ground_state_event_runs_after_base_reset():
    # set_ground_state écrase la pose posée par reset_base / reset_robot_joints :
    # l'ordre des événements suit l'ordre d'insertion, il doit donc venir APRÈS.
    cfg = make_microduck_roller_standup_env_cfg()
    order = list(cfg.events.keys())
    assert order.index("set_ground_state") > order.index("reset_base")
    assert order.index("set_ground_state") > order.index("reset_robot_joints")


def test_no_fall_termination():
    # Le robot DÉMARRE tombé : une terminaison sur inclinaison tuerait l'épisode
    # au premier pas. nan_state (hérité) reste, lui.
    cfg = make_microduck_roller_standup_env_cfg()
    assert "fell_over" not in cfg.terminations
    assert "nan_state" in cfg.terminations


def test_ground_state_curriculum_ramps_easy_to_hard():
    cfg = make_microduck_roller_standup_env_cfg()
    assert "ground_state_mix" in cfg.curriculum
    stages = cfg.curriculum["ground_state_mix"].params["param_stages"]
    assert cfg.curriculum["ground_state_mix"].params["event_name"] == "set_ground_state"
    # Les steps sont croissants et démarrent à 0.
    steps = [s["step"] for s in stages]
    assert steps[0] == 0 and steps == sorted(steps) and len(set(steps)) == len(steps)
    # Le dos (face_up) est introduit tard puis croît de façon monotone.
    face_up = [s["params"]["face_up_prob"] for s in stages]
    assert face_up[0] == 0.0
    assert face_up == sorted(face_up)
    assert face_up[-1] >= 0.35
    # Chaque palier est une distribution valide, et le "déjà debout" ne disparaît
    # jamais (sinon la policy se relève puis retombe faute d'apprendre à tenir).
    for stage in stages:
        p = stage["params"]
        total = (
            p["standing_prob"] + p["sitting_prob"]
            + p["face_down_prob"] + p["face_up_prob"]
        )
        assert abs(total - 1.0) < 1e-9
        assert p["sitting_prob"] == 0.0
        assert p["standing_prob"] > 0.0


def test_wheel_friction_curriculum_is_decreasing():
    """La pièce nouvelle : roues FREINÉES → LIBRES.

    Les roues roulent, donc il n'y a aucune adhérence longitudinale pour pousser
    sur le sol. On bootstrappe avec des roulements quasi bloqués (le relevé se
    fait comme avec des pieds) puis on rampe vers la vraie valeur. L'env roller,
    lui, fait MONTER cette friction (0 → 0.0015) : le sens est bien inversé ici.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    stages = cfg.curriculum["wheel_friction"].params["ranges_stages"]
    assert cfg.curriculum["wheel_friction"].params["event_name"] == "randomize_wheel_friction"

    steps = [s["step"] for s in stages]
    assert steps[0] == 0 and steps == sorted(steps) and len(set(steps)) == len(steps)

    lows = [s["ranges"][0] for s in stages]
    assert lows == sorted(lows, reverse=True), "la friction doit DÉCROÎTRE"
    assert lows[0] >= 0.02, "départ franchement freiné pour bootstrapper le geste"
    # Arrivée sur la vraie valeur du roulement (celle de l'env roller).
    assert stages[-1]["ranges"] == (0.0015, 0.0015)
    for stage in stages:
        assert stage["ranges"][0] == stage["ranges"][1]


def test_wheel_friction_event_default_matches_stage_zero():
    # Le curriculum manager tourne AVANT les événements de reset à chaque reset
    # (y compris le tout premier), et wheel_friction_curriculum défaut lui-même
    # sur le palier 0 : cette valeur par défaut de l'événement n'est donc jamais
    # lue en pratique. On vérifie juste qu'elle reste cohérente avec le palier 0
    # du curriculum — redondance défensive utile si le curriculum disparaît un
    # jour en laissant l'événement en place.
    cfg = make_microduck_roller_standup_env_cfg()
    stage0 = cfg.curriculum["wheel_friction"].params["ranges_stages"][0]["ranges"]
    assert cfg.events["randomize_wheel_friction"].params["ranges"] == stage0


def test_action_rate_ramp_is_the_standup_one_not_the_roller_one():
    # L'env roller monte à -2.0 (gait calme) : c'est un bloqueur de mouvement,
    # il ralentit l'action rapide dont le relevé depuis le dos a besoin. On
    # reprend la rampe du standup, qui plafonne à -1.0.
    cfg = make_microduck_roller_standup_env_cfg()
    weights = [
        s["weight"] for s in cfg.curriculum["action_rate_weight"].params["weight_stages"]
    ]
    assert weights == [-0.4, -0.8, -1.0]
    assert cfg.rewards["action_rate_l2"].weight == -0.6


def test_push_curriculum_ramps_from_zero():
    # Poussées héritées (±0.2 m/s), mais rampées : une bourrade dès le pas 0
    # parasite le bootstrap du relevé.
    cfg = make_microduck_roller_standup_env_cfg()
    assert "push_robot" in cfg.events
    stages = cfg.curriculum["push_magnitude"].params["push_stages"]
    assert cfg.curriculum["push_magnitude"].params["event_name"] == "push_robot"
    assert stages[0]["velocity_range"]["x"] == (0.0, 0.0)
    assert stages[-1]["velocity_range"]["x"] == (-0.2, 0.2)
    highs = [s["velocity_range"]["x"][1] for s in stages]
    assert highs == sorted(highs), "la poussée doit CROÎTRE"


def test_inherited_dr_curricula_survive():
    # La DR héritée de l'env roller ne doit pas avoir été perdue en chemin.
    cfg = make_microduck_roller_standup_env_cfg()
    for name in ("com_range", "head_com_range"):
        assert name in cfg.curriculum, f"curriculum de DR perdu : {name}"
    for name in (
        "randomize_com",
        "randomize_head_com",
        "randomize_armature",
        "randomize_joint_friction",
        "randomize_mass_inertia",
        "randomize_wheel_friction",
        "encoder_bias",
    ):
        assert name in cfg.events, f"événement de DR perdu : {name}"


# ── Override de play : forcer les départs sur le dos ──────────────────────────
# Sans override, un play ne montre JAMAIS de départ sur le dos : l'env de play est
# reconstruit à neuf, donc common_step_counter repart à 0 et le curriculum applique
# son palier 0, où face_up_prob = 0. Or c'est justement le cas le plus dur, celui
# qu'on veut inspecter à l'œil. STANDUP_PLAY_FACE_UP force le mélange, sur le
# modèle de SLOPE_PLAY_DIFFICULTY dans roller_slope.


def test_play_face_up_override_forces_back_starts(monkeypatch):
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "1.0")
    cfg = make_microduck_roller_standup_env_cfg(play=True)
    params = cfg.events["set_ground_state"].params
    assert params["face_up_prob"] == 1.0
    assert params["face_down_prob"] == 0.0
    assert params["standing_prob"] == 0.0
    # Sans ça, le curriculum réécrirait les probabilités dès le premier reset
    # (event_param_curriculum tourne AVANT les événements de reset).
    assert "ground_state_mix" not in cfg.curriculum


def test_play_face_up_override_splits_remainder_like_final_stage(monkeypatch):
    # 0.4 doit reproduire le DERNIER palier du curriculum (0.40 ventre / 0.20
    # debout / 0.40 dos) : le reste est réparti dans le rapport 2:1 de ce palier.
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "0.4")
    params = make_microduck_roller_standup_env_cfg(play=True).events["set_ground_state"].params
    assert params["face_up_prob"] == pytest.approx(0.40)
    assert params["face_down_prob"] == pytest.approx(0.40)
    assert params["standing_prob"] == pytest.approx(0.20)
    total = params["face_up_prob"] + params["face_down_prob"] + params["standing_prob"]
    assert total == pytest.approx(1.0)


def test_play_face_up_override_is_clamped(monkeypatch):
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "3.0")
    params = make_microduck_roller_standup_env_cfg(play=True).events["set_ground_state"].params
    assert params["face_up_prob"] == 1.0


def test_play_face_up_override_ignored_during_training(monkeypatch):
    # Garde-fou : la variable ne doit JAMAIS toucher l'entraînement, sinon on
    # casserait le curriculum easy->hard sans s'en apercevoir.
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "1.0")
    cfg = make_microduck_roller_standup_env_cfg(play=False)
    assert cfg.events["set_ground_state"].params["face_up_prob"] == 0.00
    assert "ground_state_mix" in cfg.curriculum


def test_play_without_override_keeps_curriculum_mix(monkeypatch):
    # Comportement par défaut inchangé : palier 0, pas de départ sur le dos.
    monkeypatch.delenv("STANDUP_PLAY_FACE_UP", raising=False)
    cfg = make_microduck_roller_standup_env_cfg(play=True)
    assert cfg.events["set_ground_state"].params["face_up_prob"] == 0.00
    assert "ground_state_mix" in cfg.curriculum


def test_play_face_up_override_invalid_value_falls_back(monkeypatch):
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "pouet")
    cfg = make_microduck_roller_standup_env_cfg(play=True)
    assert cfg.events["set_ground_state"].params["face_up_prob"] == 0.00
    assert "ground_state_mix" in cfg.curriculum


def test_play_face_up_override_none_keyword_disables(monkeypatch):
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "none")
    cfg = make_microduck_roller_standup_env_cfg(play=True)
    assert cfg.events["set_ground_state"].params["face_up_prob"] == 0.00
    assert "ground_state_mix" in cfg.curriculum


# ── Anti-violence : corrections après test sur le robot ───────────────────────
# Symptômes observés (checkpoint 4000+, EN SIMU AUSSI donc pas du sim2real) :
# mouvements très brusques, la tête tape le sol, échec du relevé depuis le dos
# sur le vrai robot. Diagnostic mesuré dans wandb (run vweolw91, iter 7500).


def test_already_negative_penalties_use_positive_weights():
    """Verrou sur la classe de bug qui rendait la policy violente.

    mdp.py mélange DEUX conventions de signe : certaines fonctions de pénalité
    renvoient une magnitude positive (à multiplier par un poids négatif), d'autres
    renvoient déjà une valeur négative (à multiplier par un poids POSITIF).
    trunk_vertical_accel_penalty renvoie -|a_z| : avec le poids -0.02 hérité du
    standup, le double négatif RÉCOMPENSAIT l'accélération verticale — mesuré à
    Episode_Reward/gentle_rise = +0.0118, seul terme de pénalité loggé positif.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    # Ces trois termes appellent des fonctions qui renvoient déjà du négatif
    # (height_l1_penalty, pose_l1_penalty, trunk_vertical_accel_penalty).
    for name in ("height_stand_l1", "pose_stand_l1", "gentle_rise"):
        assert cfg.rewards[name].weight > 0, (
            f"{name} appelle une fonction qui renvoie déjà du négatif : "
            f"un poids négatif en ferait une récompense"
        )
    # Et ces termes renvoient une magnitude positive → poids négatif.
    for name in ("joint_torques_l2", "joint_torque_rate_l2", "action_rate_l2"):
        assert cfg.rewards[name].weight < 0, f"{name} attend un poids négatif"


def test_no_ungated_head_impact_penalty():
    """PAS de pénalité d'impact tête non gatée — elle gelait la policy.

    Essayée à -1.0 (valeurs de velstand) : la policy a convergé vers rester
    couchée, inerte. Mesuré sur le run d8rnko6p : head_impact_penalty -1.01/pas,
    le plus gros terme négatif, pendant que standing_composite s'effondrait de
    +14.3 à +3.3.

    L'erreur de raisonnement était de croire qu'une pénalité « ciblée » ne bride
    pas le mouvement. Faux ici : pour se relever du dos, ce robot PIVOTE sur sa
    tête et ses épaules. La tête est le point d'appui du retournement, pas un
    dégât collatéral — la pénaliser, c'est pénaliser le seul mécanisme disponible.

    Si le slam revient une fois le signe de gentle_rise corrigé, la reprise doit
    être une pénalité GATÉE EN HAUTEUR (comme upright_sharp l'est), qui épargne la
    phase de retournement au sol. Pas celle-ci.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    assert "head_impact_penalty" not in cfg.rewards
    assert "head_impact_contact" not in [s.name for s in cfg.scene.sensors]


def test_inherited_sensors_intact():
    # Les capteurs hérités de l'env roller sont utilisés par des récompenses
    # gardées (self_collisions) et par les observations.
    cfg = make_microduck_roller_standup_env_cfg()
    names = [s.name for s in cfg.scene.sensors]
    assert "feet_ground_contact" in names
    assert "self_collision" in names


def test_lazy_prone_optimum_is_documented_risk():
    """Le gel vient d'un optimum paresseux : couché, jambes à HOME, ça paye.

    pose_stand_legs restait à +7.72 sur 8 alors que le robot était allongé — les
    jambes sont à HOME en position couchée, donc la récompense de pose est encaissée
    quasi gratuitement. C'est le contrepoids qui rend « ne rien faire » viable dès
    qu'on ajoute un coût au mouvement. height_stand_l1 (poids +30) est le terme
    censé rendre « rester au sol » net négatif : il doit rester fort.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    assert cfg.rewards["height_stand_l1"].weight >= 30.0
    assert cfg.rewards["com_upward_velocity"].weight > 0.0


def test_damping_terms_are_not_numerically_negligible():
    """Les amortisseurs dédiés ne pesaient littéralement rien.

    Mesuré à convergence : joint_torque_rate_l2 -0.0002/pas et joint_torques_l2
    -0.0001/pas, face à ~+41.6 de récompense de tâche (rapport ~35:1 pour tous
    les amortisseurs réunis). joint_torque_rate_l2 est le levier SÛR à remonter :
    il pénalise la VARIATION de couple, pas le mouvement, donc il n'agit pas comme
    bloqueur de mouvement — le standup documente que body_ang_vel et action_rate,
    eux, gelaient le relevé depuis le dos.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    assert abs(cfg.rewards["joint_torque_rate_l2"].weight) >= 0.1
    # Les bloqueurs de mouvement restent à leurs valeurs « se relève de partout ».
    assert cfg.rewards["body_ang_vel"].weight == -0.05
    weights = [s["weight"] for s in cfg.curriculum["action_rate_weight"].params["weight_stages"]]
    assert min(weights) >= -1.0, "action_rate au-delà de -1.0 gelait le relevé (standup)"
