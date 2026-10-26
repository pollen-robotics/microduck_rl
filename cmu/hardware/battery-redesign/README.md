# Battery backpack redesign: the battery rides on the cap

A proposal, not a release. The V1.20 frame has nothing that locates the battery above the step at z -15.2, and nothing
stops it lifting 7.3 mm into the lead space in a fall: the only sideways guides are two fences on the cap. This design
moves the battery onto the back panel. The frame keeps the switch and DFR1202 exactly as V1.20; the cap carries a printed
cradle, and a small screwed bar across the battery's front face closes it. Removing the three cap screws takes the cap,
cradle, bar and battery off together.

Frames and units as in [../README.md](../README.md): millimetres, robot frame, x forward, y left, z up.

## What changes against V1.20

- Cap: a cradle grows off the inner back wall: two side walls (|y| 15.8 to 17.0, z -13.7 to 36), a 1.2 mm floor
  (|y| <= 16.2, z -23.71 to -22.51) and three roof tabs over the pack's top (y -17 to -11, -3 to 6, 11 to 17, z 34.79
  to 35.99), with the gaps between the tabs for the leads. Two bosses (diameter 5, 5 deep, M2 pilot) at the walls' front
  ends, centred at (y, z) = (+-18.6, -8). The cap prints back-face-down as today and every new feature rises off the
  bed. The cap moves 1.1 mm back (inner wall x -73.25 to -74.35).
- Retainer bar (new part): 42.2 x 5 x 1.5 mm at x -61.75 to -60.25, z -10.5 to -5.5, held by 2 x M2 x 6 into the
  cradle bosses. It sits below plug 1 and 0.3 mm behind the switch body.
- Frame: the switch and DFR1202 holders are untouched. The rim grows 1.1 mm (the prismatic rim slab copied back), and a
  1.5 mm recess is cut into the floor (x -68.15 to -61.75, |y| <= 16.5) for the cradle floor.
- Battery: x -74.05 to -62.05 (1.8 mm further back than V1.20), 0.3 mm play on every side at the 57 x 31 x 12 mm listing.
- Plugs: plug 1 rides with the cap on a looped lead S in the robot-right side space, so the cap pulls about 40 mm back
  before you unplug. Cut lead S's battery end to 75 to 80 mm. Plug 2 parks in the front bay beside the board.
- Box: 29.45 to 30.55 mm deep behind the front plate. Width, top, step and the four fixing holes are unchanged.
- Mass: about +2.5 g (cradle 1.9, bar 0.3, frame 0.5); estimates from CAD volume x 1.24 x 0.851, as `bp_mass.json`.

## Charging

Three cap screws out, pull the cap back about 40 mm, unplug plug 1 beside the box. The battery stays captured in the cap;
the two bar screws release it.

## Files

- `stl/battery_backpack_{frame,cap,bar}-world.stl`: the three parts in the robot frame. No print-pose files yet.
  `stl/battery_backpack_cradle_added-world.stl` is the cradle alone, only for the viewer's highlight.
- `checks/checks.json`: the checks below, plus the plug parking boxes and the wire centrelines they were run against.
- `build_cap_cradle.py`: builds all of the above from the V1.20 world STLs in `../battery-backpack/` with trimesh and
  manifold3d (command in its header). Unlike `../battery-backpack/cad-source/`, it runs as is.
- `viewer/index.html`: a 3D viewer (three.js) of this design against V1.20, with a section cut, the cap-removal and
  battery-removal sliders, the slack loop paying out, labels and the checks table. It reads the STLs and JSON above and
  in `../battery-backpack/` directly, so serve `cmu/hardware` over HTTP (browsers block those reads from `file://`):

  ```bash
  cd cmu/hardware && python3 -m http.server 8000
  ```

  then open http://localhost:8000/battery-redesign/viewer/.

## Checks run (`checks/checks.json`)

- Each part is one closed solid (manifold3d).
- The battery slides into the cradle from 20 mm in front without touching it.
- Cap, cradle, bar, battery and plug 1 pulled 45 mm straight back touch nothing on the frame, the switch, the DFR1202,
  its pads or the swap plugs.
- Seated, the battery touches neither the bar nor the frame. It grazes the cap by 0.07 mm3 at its bottom-rear edges:
  the box-shaped envelope against the cap's corner radius; real pack edges are rounded.
- Plugs 1, 2 and the balance plug overlap nothing. No wire route crosses a part, except two contacts at the front window
  (servo and 5 V leads to J3 and J4) that are already in the V1.20 `wires.json`.
- Lead S at 73 mm covers the 57 mm straight-line reach with the cap 40 mm back.
- 4 mm driver lanes are clear for the two bar screws and the two upper cover screws.

## Not done yet

- Measure the battery (thickness, width, length with the lead bulge, both lead exits). The cradle and the roof tabs are
  drawn around the listing and a main lead at y -8; the robot-left tab is the one most likely to meet the balance lead.
- Plug parking spots and the lead S, C routes are approximate.
- Hip sweep, print poses, slicing and the release pipeline: none rerun.
- The cradle walls are 1.2 mm thick and only tied together once the bar is on; if they feel flexible, take them to 1.6 mm
  outward (the play stays the same).
- Add the cradle and bar to `cmu/model/variants/battery.json` and rerun the sim suite on the battery variant.
