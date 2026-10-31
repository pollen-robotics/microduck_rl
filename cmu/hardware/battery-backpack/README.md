# Battery backpack: redesign and print it yourself

The backpack is the orange box on the robot's back. It holds a Tattu 2S 650 mAh LiPo standing upright, a Swann 39
rocker switch in its floor, and a DFRobot DFR1202 5 V buck converter for the head board; it bolts on where the bench
power box does, with the same four screws, and is swapped for that box when the robot runs from the bench supply.
Decision record D-010 (2026-10-15), refitted to the Swann switch on 2026-10-17; the last numbered robot prints are V1.20.

**The STL files in `stl/` are now the battery-on-the-cap redesign, not V1.20.** The battery rides in a cradle on the cap, held by
two screwed bars, and its leads leave through the roof: see [../battery-redesign/README.md](../battery-redesign/README.md), which
replaces the pocket, cap and plug details below. The rest of this page, the design record, renders and checks here still
describe V1.20; the V1.20 frame and cap are kept in `../battery-redesign/v120-source/` and `../printing/ready/V1.20/`.

Read in this order: the renders (`renders/TEAM-sheet.png`, `renders/BP-sheet.png`, why the battery stands upright in
`renders/lead-bend-explainer.png`), then [DESIGN-D010.md](DESIGN-D010.md) (what is inside, how each part is held, the
plugs, charging, the checks), then [BATTERY-BACKPACK-TEAM.html](BATTERY-BACKPACK-TEAM.html) (open it in a browser:
the wiring and assembly steps with photos), then [POWER-PLAN.md](POWER-PLAN.md) (why 2S, why these parts, the parts
list in its section 8, the servo setting and the battery guard in sections 6 and 7).

Dates matter: POWER-PLAN.md is the 2026-10-15 plan and still names the REV-31-1387 switch; the robot got RoboClub's
Swann 39 instead (DESIGN-D010.md, 2026-10-17 status) and the printed frame carries the Swann cutout. Where the two
disagree, DESIGN-D010.md and the STL files are current.

## What is fixed and what is free

Keep these, or the part no longer fits the robot (all numbers in the robot frame, mm, x forward, y left, z up; the
bench box shares every one of them, so the two stay interchangeable):

- The front plate outline and the four fixing holes, M2 x 6 into the rear pack's posts at (y, z) = (+-16.1, 26.0) and
  (+-16.1, -36.5); the plate's outer face is at x = -45 (`cad-source/build_bp.py`, `X_IN`, `X_OUT`).
- Width |y| <= 32.05 (the shells' width), top z <= 43, the straight step at z = -15.2 to the narrow lower tier
  (|y| <= 17.7, down to z = -34). Nothing grows above the shells; a bigger battery means a deeper box, not a taller one.
- The two swap plugs (XT30 for servo power, MR30 for the 5 V head feed) leave through a window in the front plate and
  stand in the empty channel in front of the box, behind the OpenRB. The robot halves are fixed leads; the backpack
  halves must reach them (`envelopes/plugs-world.stl`, `envelopes/swap_plugs-world.stl`, `envelopes/wires.json`).
- Leg clearance: hip pitch +-30 degrees and hip roll +-10 degrees must keep every leg part 6 mm or more from the box,
  plugs and wires (`checks/bp_hip_sweep.json` is the record for the current box).
- Shawn's design rules for this box: no upward growth; windows, not wider walls; a straight step; rear edges rounded
  like the shells; no glue over a screw; every boss and hole has a named purpose.

Free to change: the battery (the current pocket is 58 x 32 x 13 mm for the 57 x 31 x 12 listing size; a 2S 930 mAh
pack is 61 x 30 x 16 and needs a wider pocket), how the cap closes, how the board and switch are held, the switch
itself, the cap screws (now 3 x M2 x 8, 3.0 mm screw-head floors; the 1.0 mm floors of V1.17/V1.18 tore), the window.

## Files

- `stl/` (redesign, owned in this fork): `battery_backpack_{frame,cap,bar}-world.stl` in the robot frame (design against
  these); `*-print.stl` the same solids in their print pose, lowest point on z = 0 (drag these into a slicer).
- `mating/`: the robot parts next to the box, robot frame: `rear_pack.stl` (the V1.9 pack the box screws onto, with
  the posts), `rear_frame.stl` and `rear_lid.stl` (the bench power box and its cap: the sibling design with the same
  interface), `shell_left.stl`, `shell_right.stl` (torso shells: the width and the top), `openrb-board-upright.stl`
  (the controller board in front of the plug channel).
- `envelopes/`: what the box was drawn around, robot frame: `battery-world.stl` (listing size), `switch_swann-world.stl`
  (Swann 39 measured shape), `switch-world.stl` (the REV-31-1387 envelope the first drafts used), `dfr1202-world.stl`
  and `dfr_pads-world.stl`, `plugs-world.stl` (the two XT30 joints beside the battery), `swap_plugs-world.stl`,
  `wires.json` (wire centrelines), `build.json` (every derived number: pocket, switch pocket, cap split x = -67.05,
  lid back x = -74.45, window, screw columns, bought-part overlaps).
- `checks/`: `bp_checks.json` (print poses with bed contact and height, insertion paths, cap path, outline, wire
  penetration, screw thread and driver lanes), `bp_hip_sweep.json`, `bp_mass.json` (masses and the balance shift),
  `release-rows-orange.json` (the rows the V1.20 print release was built from).
- `renders/`: the three sheets named above.
- `cad-source/`: the scripts that produced all of the above. `build_bp.py` is the geometry (trimesh + manifold3d +
  shapely; its header lists every constant), `bp_checks.py` the checks, `bp_mass.py` the mass, `render_bp.py` and
  `render_team.py` the renders, `swann_switch.py` the switch model, `concepts.py` the two earlier drafts and the
  shared helpers. They import private modules (`pack_lib`, `shellkit`, `build_rear_pack`, `ia_lib`, the vtk renderer)
  that are not in this repository, so they do not run here as is; they are the record of how each number was derived.
  Redesign in whatever CAD you like (Fusion, Onshape, FreeCAD, CadQuery) from the STL files and the numbers above.

## Parts list (section 8 of POWER-PLAN.md has the sources and prices)

- Tattu 2S 650 mAh 75C XT30 LiPo (57 x 31 x 12 mm, 43 g), two packs, one in use and one charging.
- Swann 39 rocker switch (RoboClub stock; 12.3 x 27.5 mm panel cutout, 30 x 13.8 mm flange, body 11.5 mm thick).
- DFRobot DFR1202 5 V 16 A buck converter (30 x 26 x 8 mm, castellated pads, no mounting holes).
- Amass XT30U pairs (battery voltage: battery to switch, switch to board, servo swap plug) and one Amass MR30 pair
  (5 V to the head board, a different shape so 8.4 V cannot reach the head by mistake).
- 22 AWG stranded silicone wire, heat shrink, Dupont female terminals for the head board header.
- Screws: 4 x M2 x 6 (fixing, shared with the bench box), 3 x M2 x 8 (cap). Two foam pads (battery, inductor).
- A 2S balance charger, a LiPo bag and a cell checker for charging off the robot (never on it).

## Masses and the simulation

`checks/bp_mass.json`: the backpack adds about 91 g (frame 16, cap 8, battery 43, switch with leads 14, board 6,
plugs and wires 10, minus the removed V1.8 cover) and moves the whole-robot balance point 6.5 mm back. Only the
battery mass is a listed value; the rest are estimates until weighed. The simulation's battery form of the robot
(`cmu/model/variants/battery.json`) carries exactly these entries, with their positions. After a redesign, or after
weighing the real parts: edit that file, run `uv run --no-sync python cmu/model/make_variants.py`, then
`uv run cmu/simcheck/sim_suite.py cmu/whole/runtime-overlay/models out.jsonl --variant battery` to see what the new
masses do to the walk (today: 10 to 15 % shorter walks, no falls; `cmu/simcheck/README.md`).

## Checks a new design must pass before it is printed

The same list the current box passed (`checks/bp_checks.json`, `DESIGN-D010.md` section "Checks"):

1. Fit: the battery, board, switch and plugs sit inside with their clearances; the only contacts are designed ones
   (switch snap wings, board clip lips).
2. Assembly paths: the battery slides out the back, the cap goes on, the switch goes in from below and the board from
   behind, each touching only what holds it.
3. Outline: nothing behind the front plate pokes outside the box outline; no wire centreline crosses a wall.
4. Screws: 5 mm of thread for each cap screw, a clear head seat, a clear 4 mm driver lane; a clear driver lane for all
   four fixing screws (the upper two through the empty battery space, the lower two under the floor).
5. Hip sweep: 6 mm or more to every leg part over hip pitch +-30 and roll +-10 degrees.
6. Print: each part a single watertight solid; a print pose with real flat contact (the frame prints rim-down, 713 mm2
   contact, 22 mm tall; the cap on its rear face, 2691 mm2, 7.4 mm tall), then the rules in
   [../printing/README.md](../printing/README.md).
7. Mass and balance rerun (section above) and the sim suite on the battery variant.

Then print it with the printing guide, and tell the person who runs the robot which plug goes where before the first
power-up (the wiring page has the meter checks).
