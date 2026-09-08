#!/usr/bin/env python3
"""Inject gearbox-backlash joints into an onshape-to-robot MJCF export.

For every actuated servo joint (``class="chosen_actuator"``) this inserts an
unactuated hinge on the same body / same axis right after it:

    <joint axis="0 0 1" name="left_hip_yaw" ... class="chosen_actuator"/>
    <joint axis="0 0 1" name="passive_left_hip_yaw_backlash" class="backlash"/>

The composite link rotation is main + backlash: the main joint is the servo
output (BAM drives it), the backlash joint is the play between the servo and
the link, free to wander within ±(backlash/2).

Naming: the ``passive_`` prefix means the new joints are automatically excluded
by every existing regex in the task configs (actuators ``^(?!passive_).*``,
joint obs, pose reward). The encoder-through-backlash handling lives on the
mjlab side (BacklashEncoderBamActuatorCfg + joint_pos/vel_rel_backlash obs).

Meant to run as the LAST post_import_command of an onshape-to-robot config
(see config_mjcf_groundcontact_backlash.json), but works standalone on any
already-exported robot xml:

    python3 add_backlash.py robot_groundcontact_backlash.xml --backlash-deg 2.0

``--backlash-deg`` is the TOTAL peak-to-peak play (what you measure wiggling
the horn with the servo held); the joint range is symmetric ±deg/2.
"""

import argparse
import math
import re
import sys

JOINT_RE = re.compile(r'^(\s*)<joint\b[^>]*/>\s*$')
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')


def build_backlash_default(half_range_rad: float, damping: float,
                           armature: float, frictionloss: float,
                           total_deg: float) -> str:
    return (
        f"  <!-- Backlash injected by add_backlash.py: {total_deg:g} deg total play"
        f" (symmetric +/-{total_deg / 2:g} deg) -->\n"
        f"  <default>\n"
        f"    <default class=\"backlash\">\n"
        f"      <!-- stiff limit constraint: with a range this small the default\n"
        f"           solref (0.02,1) lets the joint overshoot its limits ~2x under\n"
        f"           load. 0.01 = 2*sim_dt (mjlab velocity tasks run dt=0.005),\n"
        f"           the stiffest stable setting; solimp raises the impedance so\n"
        f"           the gear-teeth contact is nearly rigid. -->\n"
        f"      <joint damping=\"{damping:g}\" frictionloss=\"{frictionloss:g}\""
        f" armature=\"{armature:g}\" limited=\"true\""
        f" range=\"{-half_range_rad:.17g} {half_range_rad:.17g}\""
        f" solreflimit=\"0.01 1\" solimplimit=\"0.95 0.999 0.0001 0.5 2\"/>\n"
        f"    </default>\n"
        f"  </default>\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", help="MJCF file to modify in place")
    parser.add_argument("--backlash-deg", type=float, default=2.0,
                        help="TOTAL backlash play in degrees (peak-to-peak); "
                             "joint range is symmetric +/-deg/2 (default: 2.0)")
    parser.add_argument("--damping", type=float, default=0.01,
                        help="backlash joint damping (default: 0.01)")
    parser.add_argument("--armature", type=float, default=0.001,
                        help="backlash joint armature, kept small but non-zero "
                             "for solver conditioning (default: 0.001)")
    parser.add_argument("--frictionloss", type=float, default=0.0,
                        help="backlash joint frictionloss (default: 0)")
    parser.add_argument("--joint-class", default="chosen_actuator",
                        help="default class of the joints that get backlash "
                             "(default: chosen_actuator)")
    parser.add_argument("--exclude", default=None,
                        help="optional regex of joint names to skip "
                             "(e.g. '.*(neck|head).*')")
    args = parser.parse_args()

    half_range = math.radians(args.backlash_deg) / 2.0
    exclude = re.compile(args.exclude) if args.exclude else None

    with open(args.xml) as f:
        lines = f.readlines()

    if any('class="backlash"' in line for line in lines):
        print(f"[add_backlash] {args.xml} already contains backlash joints — aborting.")
        return 1

    out = []
    added = []
    default_inserted = False
    for line in lines:
        # Insert the defaults block right before <worldbody>.
        if not default_inserted and "<worldbody>" in line:
            out.append(build_backlash_default(
                half_range, args.damping, args.armature, args.frictionloss,
                args.backlash_deg))
            default_inserted = True

        out.append(line)

        m = JOINT_RE.match(line)
        if m is None:
            continue
        attrs = dict(ATTR_RE.findall(line))
        if attrs.get("class") != args.joint_class:
            continue
        name = attrs.get("name")
        if not name or (exclude and exclude.match(name)):
            continue
        indent = m.group(1)
        axis = attrs.get("axis", "0 0 1")
        pos = f' pos="{attrs["pos"]}"' if "pos" in attrs else ""
        out.append(
            f'{indent}<joint axis="{axis}"{pos} '
            f'name="passive_{name}_backlash" type="hinge" class="backlash"/>\n'
        )
        added.append(name)

    if not default_inserted:
        print("[add_backlash] ERROR: no <worldbody> found — is this an MJCF file?")
        return 1
    if not added:
        print(f"[add_backlash] ERROR: no joints with class=\"{args.joint_class}\" found.")
        return 1

    with open(args.xml, "w") as f:
        f.writelines(out)

    print(f"[add_backlash] added {len(added)} backlash joints "
          f"(+/-{args.backlash_deg / 2:g} deg = +/-{half_range:.5f} rad) to {args.xml}: "
          f"{', '.join(added)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
