"""Microduck JUMP task — saut vertical sur place (legs), piloté par la phase.

Geste cyclique déclenché via le slot de commande phase (même machinerie que
spin / ground_pick / roller_crouch) :

  phase ∈ [0, HOLD_LO)       : impulsion — montée  stand -> peak
  phase ∈ [HOLD_LO, HOLD_HI) : envol     — plafond  peak  (le robot est en l'air)
  phase ∈ [HOLD_HI, 1.0)     : descente  peak -> stand puis stabilisation posée

Local addition — geste inédit dans la suite officielle :
  - physique / robot legs   ← microduck_velocity_env_cfg.py (défaut, reset z 0.12–0.13)
  - machinerie phase        ← GroundPickPhaseCommand (comme spin/crouch)
  - obs 61D unifié → interchangeable au runtime avec walk / sitstand / ground_pick
  - symétrie gauche/droite ON (saut symétrique par nature ; spin l'interdit car
    il transforme une rotation gauche en droite).

Aucun fichier upstream modifié : seuls ce module et la ligne de registre dans
tasks/__init__.py sont locaux.
"""

import math
from copy import deepcopy

import torch
from mjlab.envs import ManagerBasedRlEnv

# Un saut est bilatéralement symétrique → on *voudrait* tirer parti de la
# symétrie, mais PpoWithSymmetryCfg ne prend en pratique que None : tyro ne
# sait pas résoudre `dict | None` (symmetry.py: `symmetry_cfg: dict | None =
# None`), aucun env officiel n'a jamais passé un dict non-None (docstring de
# symmetry.py : "dead code until now"). Suivre spin → off pour l'instant.
ENABLE_SYMMETRY = False

# DR — repris de l'environnement legs/velocity (pas de roues ici).
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

COM_RANDOMIZATION_RANGE          = 0.003
HEAD_COM_RANDOMIZATION_RANGE     = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE     = (0.9, 1.1)
# Saut sur place : petites poussées suffisent pour peupler le DR (trop fort =
# la policy reste aux prises à se battre contre le vent).
VELOCITY_PUSH_INTERVAL_S         = (3.0, 6.0)
VELOCITY_PUSH_RANGE              = (-0.1, 0.1)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0
ENCODER_BIAS_RANGE               = (-0.015, 0.015)

# ==== enveloppe du saut (mètres / fractions de phase) ====
# v4 — courbes calibrées sur la PHYSIQUE RÉELLE mesurée (phys_ceiling/phys_dyn,
# Thor 2026-09-01) : legs à extension limitée, z dynamique max ≈ 0.13 m.
#   action=+1 (extension max, planté) : z ≈ 0.12–0.123
#   action=-0.8 (accroupi)            : z ≈ 0.106–0.112
#   crouch→launch dynamique           : pic z ≈ 0.1297
# Donc : stand 0.120 / crouch 0.106 / peak 0.128 — le 0.185 de v1–v3 est
# PHYSIQUEMENT INATTEIGNABLE ; il a forcé l'arnaque "one-leg-lift" que le
# gate max(air) de v1–v3 laissait passer.
JUMP_STAND_Z      = 0.120  # trunk z debout stabilisé (mesuré 0.112–0.122)
JUMP_CROUCH_Z     = 0.106  # accroupissement mesuré (action -0.8 : 0.106–0.112)
JUMP_PEAK_Z       = 0.128  # v6 : pic dynamique mesuré 0.1297 → cible sous la limite
                            # physique (0.133 restait au-dessus → gaussienne jamais 1.0)
JUMP_CROUCH_START = 0.12   # début de l'accroupissement
JUMP_CROUCH_END   = 0.28   # bas de la charge (fin 1er 1/4 de cycle)
JUMP_LAUNCH_END   = 0.34   # détente jambes → apex (0.24 s d'impulsion)
JUMP_APEX_END     = 0.44   # fin du temps de vol (0.40 s ; v3 étirait 0.36–0.58)
JUMP_HOLD_LO      = 0.32   # porte d'air : à partir de la détente
JUMP_HOLD_HI      = 0.48   # porte d'air : fin (atterrissage)
# v7 — 事件/升程窗口放宽 : HOLD_LO/HI 还被 jump_target/jump_landed 复用(改它会
# 拉扯高度曲线),所以 airtime 与 lift 用独立常量,只放宽奖励门,不动曲线。
JUMP_AIR_WIN_LO   = 0.26   # v7: 事件窗口(0.30→0.26——实测 G2 深蹲+纯腿推空@ph0.285-0.31,须盖住)
JUMP_AIR_WIN_HI   = 0.55   # v7: 事件窗口(原 0.48——跳早 0.1s 就丢 10 分)
JUMP_LIFT_LO      = 0.28   # v7: 升程梯度窗口开始(= crouch 底,起跳前)
JUMP_LIFT_HI      = 0.50   # v7: 升程梯度窗口结束(落地)
JUMP_PERIOD       = 4.0    # même période que spin/ground-pick → rien à passer au runtime
JUMP_AIR_EPS      = 0.03   # air_time (s) en dessous duquel on considère "au sol"

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROBOT_CFG
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG

NECK_PATTERN_NO_YAW = r"^(neck_pitch|head_pitch|head_roll)$"


# ==== fonctions de récompense locales (mdp.py upstream non modifié) ====

def jump_target(phase: torch.Tensor) -> torch.Tensor:
    """Cible de hauteur du tronc le long de la phase [0,1) — courbe « vrai saut » :
    station debout → accroupissement de charge (genoux pliés, 2 cm) → détente
    des jambes jusqu'à l'apex (+6 cm) → temps de vol → pose → station.

    Implémentation : interpolation linéaire par morceaux (continue en phase
    médiane, et en phase 0 == 1 pour la période).
    """
    xs = torch.tensor(
        (0.0, JUMP_CROUCH_START, JUMP_CROUCH_END, JUMP_LAUNCH_END,
         JUMP_APEX_END, JUMP_HOLD_HI, 1.0),
        device=phase.device, dtype=phase.dtype,
    )
    ys = torch.tensor(
        (JUMP_STAND_Z, JUMP_STAND_Z, JUMP_CROUCH_Z, JUMP_PEAK_Z,
         JUMP_PEAK_Z, JUMP_STAND_Z, JUMP_STAND_Z),
        device=phase.device, dtype=phase.dtype,
    )
    idx = torch.searchsorted(xs, phase).clamp(1, len(xs) - 1)
    x0, x1 = xs[idx - 1], xs[idx]
    y0, y1 = ys[idx - 1], ys[idx]
    return y0 + (phase - x0) * (y1 - y0) / (x1 - x0)


def _trunk_z(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2],
        nan=0.0,
    )


def _phase_from_command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)  # (B, 3) = [cos, sin, 0]
    return (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * math.pi)) % 1.0


def jump_height_track(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    std: float = 0.015,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
    """Récompense principale : suivre la cible de hauteur (trapèze de bond).

    v7 — SUPPRESSION de la porte aérienne v4-v6 : le gate min(air)>0.02 en
    [0.34,0.44) transformait la fenêtre de vol en zéro PERMANENT (personne ne
    volait → plus aucun gradient "vers le haut") et le gate 0/1 a même fait
    perdre un saut parfaitement exécuté mais décalé de 2 pas. v7: la courbe
    paie tout le cycle (la forme), jump_airtime paie la preuve de vol (dense).
    std 0.012 → 0.015 : l'accroupissement atteignable (action -0.8 → z mesuré
    0.111) vaut exp(-(0.005/0.015)²)=0.89 au lieu de 0.84 — moins de friction.
    """
    phase = _phase_from_command(env, command_name)
    target = jump_target(phase)
    base = torch.exp(-((_trunk_z(env, asset_cfg) - target) / std) ** 2)
    return base


def jump_height_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Bootstrap L1 : gradient constant quand la gaussienne sature loin de cible."""
    target = jump_target(_phase_from_command(env, command_name))
    return -torch.abs(_trunk_z(env, asset_cfg) - target)


def jump_crouch(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """v7f — 阶梯第 1 级: 相位 [0.10, 0.30) 蹲到 z≈0.106 的高斯梯度.

    为什么必须: v6/v7d/v7e 两轮实测, 策略全部收敛到站姿不动点(airtime
    恒 0.000, track 站姿照吃) —— 因为"窗口内真腾空"是 PPO 随机 rollout
    概率≈0 的事件, 梯度永远不出现. 这一级从站姿(0.120)出发随机扑腾就能
    碰到, 把"蹲"变成平滑可达的目标, 再给第 2 级(vz 上升)和第 3 级(腾空)
    铺路.
    """
    z = _trunk_z(env, asset_cfg)
    phase = _phase_from_command(env, command_name)
    active = (phase >= 0.10) & (phase < 0.30)
    g = torch.exp(-((z - JUMP_CROUCH_Z) / std) ** 2)
    return torch.where(active, g, torch.zeros_like(g))


def jump_launch_push(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "twist",
) -> torch.Tensor:
    """v5 — 起跳推力 : récompense la VITESSE VERTICALE vers le haut durant la
    fenêtre de détente [CROUCH_END, CROUCH_END+0.10).

    v4 教不会：le gate min(air) est discontinu (0 → 1) et la politique ne
    trouvait pas de gradient vers le saut — elle restait debout (5.19/6 sans
    jamais quitter le sol). Ce terme est DENSE et CONTINU : il paie à la fois
    la poussée (piétinement vertical) ET donne un chemin de gradient vers
    l'envol. **v7f 修正刻度**: l'ancien vz/0.5 (pic balistique ≈0.45 m/s) était
    inatteignable — les vrais sauts mesurés (probe5 G2, 2026-09-02) montent à
    vz ≈ 0.11-0.2 m/s seulement → le terme restait bloqué à 0.006. Désormais
    vz/0.12 : un vrai bond paie plein.
    """
    asset = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_vel_w[:, 2], nan=0.0)
    phase = _phase_from_command(env, command_name)
    launch = (phase >= JUMP_CROUCH_END) & (phase < JUMP_CROUCH_END + 0.10)
    return torch.where(launch, torch.clamp(vz / 0.12, min=0.0, max=1.0), torch.zeros_like(vz))


def jump_airtime(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    command_name: str = "twist",
) -> torch.Tensor:
    """v7 — RÉCOMPENSE D'ENVOL DENSE (remplace l'événement binaire v6, échec
    documenté le 2026-09-01 : airtime ep. max 0.01 sur 600 iters — jamais).

    Pourquoi v6 a échoué (chiffré) : l'événement était un 0/1 sur
    phase∈[0.32,0.48). Le vol mesuré ne dure que 0.02–0.05 s (2-3 pas) — un
    saut physiquement PARFAIT raté de 2 pas la fenêtre donnait 0 point ; le
    run docu « hero » (crouch 30→push 6→free, z=0.130, air 256/256) a produit
    air à phase 0.19–0.23 et 0.295–0.315 → l'événement N'A JAMAIS sonné. Un
    gate binaire à 2 pas près est un aiguillage sans gradient : en 600 iters
    PPO n'a jamais obtenu 0.0001 de signal.

    v7 : (1) fenêtre élargie [AIR_WIN_LO, AIR_WIN_HI) = [0.30, 0.55) —
    avant/après le vol, sans knife-edge ; (2) paiement PROPORTIONNEL
    clamp(air/0.04, 0, 1) : 0.01 s d'envol (1 pas) = 5 pts, 0.02 s (2 pas,
    la vraie porte du saut) = 10 pts pleins. Le gradient se construit dès le
    premier pas de décollement au lieu d'attendre un exploit à 2 pas près.

    v7b — preuve de MONTÉE : impulsion verticale des 5 derniers pas
    (vz_impulse = somme_vz[-5pas]) multiplie en pente douce. Pourquoi : le
    canard est sur rouleaux, ses pieds ne quittent le sol que QUAND vz est
    déjà retombé (trace 2026-09-01 : poussée vz≈+0.2 pieds au sol, vol à
    vz≈-0.15). Un gate vz>0.05 A L'INSTANT du vol ne paierait jamais. Sans
    preuve d'impulsion, une chute/roulement (air_time sur ~100 pas, les pieds
    face au ciel) encaisserait la récompense pour tomber. La pente
    clamp(vz_impulse/0.25) : envolée propre → 1.0 ; chute → 0.
    """
    air = microduck_mdp.foot_air_time_safe(env, sensor_name)  # [N,2] 两脚
    air = air.min(dim=1).values  # v4+: DEUX pieds en l'air
    phase = _phase_from_command(env, command_name)
    asset = env.scene[SceneEntityCfg("robot").name]
    vz = torch.nan_to_num(asset.data.root_link_vel_w[:, 2], nan=0.0)
    if not hasattr(env, "_jump_vz_hist"):
        env._jump_vz_hist = [torch.zeros_like(vz) for _ in range(5)]
    env._jump_vz_hist = env._jump_vz_hist[1:] + [vz]
    vz_impulse = torch.stack(env._jump_vz_hist).sum(dim=0)
    ascent = torch.clamp(vz_impulse / 0.25, min=0.0, max=1.0)
    in_window = (phase >= JUMP_AIR_WIN_LO) & (phase < JUMP_AIR_WIN_HI)
    dense = torch.clamp(air / 0.02, min=0.0, max=1.0)
    return torch.where(in_window, dense * ascent, torch.zeros_like(dense))


def jump_landed(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    command_name: str = "twist",
) -> torch.Tensor:
    """Après HOLD_HI : le robot doit être (re)touché le sol (air ≈ 0)."""
    air = microduck_mdp.foot_air_time_safe(env, sensor_name)  # [N,2] 两脚
    air = air.max(dim=1).values  # 双脚都落地才算 landed
    phase = _phase_from_command(env, command_name)
    landed_ok = air < JUMP_AIR_EPS
    return torch.where(phase >= JUMP_HOLD_HI, landed_ok.float(), torch.zeros_like(air))


def jump_feet_flat(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", site_names=("left_foot", "right_foot")
    ),
    sensor_name: str = "feet_ground_contact",
    command_name: str = "twist",
) -> torch.Tensor:
    """v7 — feet_flat seulement hors phases ACTIVES : crouch/détente/vol cassent
    l'à-plat des pieds (le microduck est sur rouleaux) — en v6 la pénalité
    -2.0 taxait les phases où un vrai saut est IMPOSSIBLE autrement, le tableau
    de décomposition (hero vs zero) montre -0.034 net contre l'envol. Fenêtre
    [0.14, 0.46) = accroupissement → vol → pose : à zéro, ailleurs la peine
    s'applique (fiabilité à l'aller / au repos)."""
    pen = microduck_mdp.feet_flat_penalty(
        env, asset_cfg=asset_cfg, sensor_name=sensor_name
    )
    phase = _phase_from_command(env, command_name)
    active = (phase >= 0.14) & (phase < 0.46)
    return torch.where(active, torch.zeros_like(pen), pen)


def make_microduck_jump_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Env saut vertical sur place (robot legs, phase command)."""

    # legs 机器人的脚底接触（ankle_l_v1/ankle_r_v1 是 rollers 变体的名字）：
    # 用 velocity 官方任务同款 geom pattern —— LEFT first, RIGHT second。
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    cfg = make_velocity_env_cfg()
    # 通用 make_velocity_env_cfg() 不带 robot entity（velocity/微鸭官方任务和
    # spin 都显式设置）—— 必须补上，否则 contact sensor 解析 'robot' 时失败。
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROBOT_CFG}
    # velocity 默认 sensors 有三个：feet_ground_contact / self_collision /
    # foot_height_scan（raycast）。plane 地形下 raycast 的 frame "robot/" 在
    # terrain 模型上找不到 body → KeyError 启动即崩；与 spin 同款处理：移除
    # raycast，只留两个 contact 传感器（obs 里的 height_scan 项已在下面删掉）。
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # === REWARDS ===
    keep = {"upright", "body_ang_vel", "action_rate_l2"}
    for name in list(cfg.rewards.keys()):
        if name not in keep:
            del cfg.rewards[name]

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["action_rate_l2"].weight = -0.4  # v7: plafond 0.4 (voir curriculum)

    # Objectif principal : hauteur du tronc le long du trapèze de bond.
    cfg.rewards["jump_height_track"] = RewardTermCfg(
        func=jump_height_track,
        weight=2.0,
        params={"command_name": "twist", "std": 0.015},  # v7: 0.012 → 0.015
    )
    # Bootstrap L1 (gaussienne saturée → gradient constant).
    cfg.rewards["jump_height_l1"] = RewardTermCfg(
        func=jump_height_l1,
        weight=0.3,
        params={"command_name": "twist"},
    )
    # Preuve d'envol : DENSE (pas l'événement binaire v6) — voir doc de la fonction.
    cfg.rewards["jump_airtime"] = RewardTermCfg(
        func=jump_airtime,
        weight=75.0,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    )
    # Atterrissage posé après HOLD_HI (contact rétabli).
    cfg.rewards["jump_landed"] = RewardTermCfg(
        func=jump_landed,
        weight=0.5,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    )
    # v5 — poussée verticale dense pendant la détente (gradient continu vers l'envol).
    # v7f: échelle corrigée (vz/0.12 au lieu de vz/0.5 — vrai bond ≈ 0.11-0.2 m/s).
    cfg.rewards["jump_launch_push"] = RewardTermCfg(
        func=jump_launch_push,
        weight=10.0,
        params={"command_name": "twist"},
    )
    # v7f — échelle 1: cramponner la cible d'accroupissement pendant [0.10, 0.30).
    # Si la politique reste debout (v7e 295 iters: airtime 0.000), ce terme est le
    # premier échelon que le bruit de PPO peut franchir (z 0.120 → 0.106 = +6 pts).
    cfg.rewards["jump_crouch"] = RewardTermCfg(
        func=jump_crouch,
        weight=10.0,
        params={"command_name": "twist", "std": 0.02},
    )
    # Stability / sim2real — v7: gateé aux phases actives (voir jump_feet_flat).
    cfg.rewards["feet_flat"] = RewardTermCfg(
        func=jump_feet_flat,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
            "sensor_name": "feet_ground_contact",
        },
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "self_collision"},
    )
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_action_rate_l2, weight=-0.5
    )
    cfg.rewards["neck_joint_pos_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_joint_pos_l2,
        weight=-0.2,
        params={"pattern": NECK_PATTERN_NO_YAW},
    )
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2, weight=-1e-3
    )

    # === TERMINATIONS ===
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan, time_out=False,
    )

    # === EVENTS ===
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history, mode="reset",
    )
    del cfg.events["foot_friction"]

    if ENABLE_VELOCITY_PUSHES:
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=VELOCITY_PUSH_INTERVAL_S,
            params={
                "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    # Saut vertical : pas d'élan d'entrée (reset propre, vitesse racine nulle).
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.12, 0.13)
    cfg.events["reset_base"].params["velocity_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )
    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia, mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )
    # BAM (mjlab_frictionloss 分支) 每步写入 per-env dof_frictionloss/dof_damping；
    # 必须先用这个 no-op startup 事件把字段注册为 per-world 展开，否则
    # BamActuator.compute 在 reset 时直接 RuntimeError（ground_pick 同款处理）。
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )
    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    # === OBSERVATIONS (layout 61D unifié) ===
    del cfg.observations["actor"].terms["base_lin_vel"]
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot")
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # === COMMAND : phase (comme spin / ground_pick) ===
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(
        **{
            **vars(command),
            "class_type": microduck_mdp.GroundPickPhaseCommand,
            "period": JUMP_PERIOD,
            "randomize_phase": False,
        }
    )

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # === CURRICULUM ===
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]
    # v7 — caps à -0.5 : la détente exige un changement d'action explosif
    # (-0.8 → +1.0 en ~6 pas, Δ≈1.8 par articulation) ; le -1.0 de v6 taxait
    # l'envol de ~0.17 pt/cycle (décomposition hero/zero), exactement le coût
    # qu'un vrai saut devait d'abord surmonter.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.4},
                {"step": 250 * 24, "weight": -0.5},
                {"step": 500 * 24, "weight": -0.5},
            ],
        },
    )
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0, "range": 0.003},
                    {"step": 500 * 24, "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0, "range": 0.003},
                    {"step": 500 * 24, "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )

    return cfg


MicroduckJumpRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="jump",
    run_name="jump",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=4_000,
)
