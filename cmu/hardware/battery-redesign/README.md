# Battery backpack redesign: the battery rides on the cap

The current backpack geometry; its parts are in [../battery-backpack/stl/](../battery-backpack/stl/). Not a numbered print release. The V1.20 frame has nothing that locates the battery above the step at z -15.2, and nothing
stops it lifting 7.3 mm into the lead space in a fall: the only sideways guides are two fences on the cap. This design
moves the battery onto the back panel. The frame keeps the switch and DFR1202 exactly as V1.20; the cap carries a printed
cradle, and two small screwed bars across the battery's front face close it. The stiff battery leads stand up through
openings in the roof, plug 1 sits on top of the box, and lead S comes back in through a third opening. Unplug plug 1 and
the three cap screws take the cap, cradle, bars and battery off together.

Frames and units as in [../README.md](../README.md): millimetres, robot frame, x forward, y left, z up.

## What changes against V1.20

- Cap: a cradle grows off the inner back wall: two side walls (|y| 15.8 to 17.0, z -13.7 to 34.49, stopping at the pack's
  top), a 1.2 mm floor (|y| <= 16.2, z -23.71 to -22.51) and one middle roof tab over the pack's top (y -3 to 6, z 34.79
  to 35.99). Four boss columns (diameter 5) beside the walls, centred at (y, z) = (+-18.6, -8) and (+-18.6, 24), run from
  the cap wall to the walls' front ends with a 5.5 mm M2 pilot at the front. The cap prints back-face-down as today and
  every new feature rises off the bed. The cap moves 1.1 mm back (inner wall x -73.25 to -74.35).
- Retainer bars (two identical new parts, one file): 42.2 x 5 x 1.5 mm at x -61.75 to -60.25, centred at z -8 and z 24, each
  held by 2 x M2 x 6 into its bosses. The low bar sits 0.3 mm behind the switch body.
- Roof openings: main lead x -71.5 to -64.6, y -14 to -2; balance lead x -71.5 to -64.6, y 7 to 17. Both straddle the
  cap seam (x -68.15), so each is an open notch in the frame and in the cap and the leads never trap the cap. Lead S:
  x -60 to -49, y -25 to -18.5, frame only. Each passes an XT30 or JST-XH without unplugging or cutting.
- Frame: the switch and DFR1202 holders are untouched. The rim grows 1.1 mm (the prismatic rim slab copied back), and a
  1.5 mm recess is cut into the floor (x -68.15 to -61.75, |y| <= 16.5) for the cradle floor.
- Battery: x -74.05 to -62.05 (1.8 mm further back than V1.20), 0.3 mm play on every side at the 57 x 31 x 12 mm listing.
- Plugs: plug 1 lies on top of the box, about 26 x 10 x 5 mm above the roof between the main opening and the lead S opening.
  This breaks the V1.20 rule that nothing grows above the shells. The balance plug also sits on top. Plug 2 parks in the
  front bay beside the board. Lead S: red about 70 mm to the switch, black about 95 mm to plug 2 (route lengths; a lead S
  cut for the earlier slack loop at 75 to 80 mm and 110 to 115 mm is long enough).
- Box: 29.45 to 30.55 mm deep behind the front plate. Width, top, step and the four fixing holes are unchanged.
- Mass: about +3.3 g (cradle 2.4, bars 0.6, frame 0.3); estimates from CAD volume x 1.24 x 0.851, as `bp_mass.json`.

## Charging

Unplug plug 1 on top, take out the three cap screws and pull the cap straight back. The battery stays captured in the cap;
the four bar screws release it.

## Files

- `../battery-backpack/stl/battery_backpack_{frame,cap,bar}-world.stl`: the parts in the robot frame (design against these;
  the viewer and checks use them). The bar file holds both bars. `../battery-backpack/stl/battery_backpack_{frame,cap,bar}-print.stl`: the same parts in their print pose, lowest point on z = 0,
  ready to drag into a slicer: frame rim down (23.15 mm tall, 637 mm2 contact), cap back face down (13.8 mm, 2691 mm2), both
  bars flat (1.5 mm, 203 mm2 each, 27 mm apart); the V1.20 rotation x' = y, y' = z, z' = x. All tip angles are above 10
  degrees, so no brim (`../printing/README.md`). The frame and cap each load in Bambu Studio 02.08 as one manifold part, the
  bar file as two.
  `viewer/battery_backpack_cradle_added-world.stl` is the cradle alone, only for the viewer's highlight.
- `v120-source/`: the V1.20 frame and cap in the robot frame, the starting point of the build script and the viewer's
  comparison. The V1.20 print release itself stays in `../printing/ready/V1.20/`.
- `checks/checks.json`: the checks below, plus the plug parking boxes and the wire centrelines they were run against.
- `build_cap_cradle.py`: builds all of the above from `v120-source/` and the envelopes in `../battery-backpack/` with trimesh and
  manifold3d (command in its header). Unlike `../battery-backpack/cad-source/`, it runs as is.
- `viewer/index.html`: a 3D viewer (three.js) of this design against V1.20, with a section cut, the cap-removal and
  battery-removal sliders, labels and the checks table. It reads the STLs and JSON above
  directly, so serve `cmu/hardware` over HTTP (browsers block those reads from `file://`):

  ```bash
  cd cmu/hardware && python3 -m http.server 8000
  ```

  then open http://localhost:8000/battery-redesign/viewer/.

## Checks run (`checks/checks.json`)

- Frame and cap are one closed solid each; the bar file holds two (manifold3d).
- The battery slides into the cradle from 20 mm in front without touching it.
- Cap, cradle, bars, battery and both leads pulled 45 mm straight back touch nothing on the frame, the switch, the DFR1202,
  its pads or the swap plugs.
- An XT30 passes through the main opening and through the lead S opening.
- Seated, the battery touches neither the bars nor the frame. It grazes the cap by 0.07 mm3 at its bottom-rear edges:
  the box-shaped envelope against the cap's corner radius; real pack edges are rounded.
- Plugs 1, 2 and the balance plug overlap nothing, and the plugs on top clear the shells and the rear pack. No wire route
  crosses a part, except two contacts at the front window (servo and 5 V leads to J3 and J4) that are already in the V1.20
  `wires.json`.
- 4 mm driver lanes are clear for the four bar screws and the two upper cover screws.
- The cap screens like the V1.20 cap on `orientation_check.py` (500.8 mm2 overhang); the frame screens like the V1.20 frame.

## Not done yet

- Measure the battery (thickness, width, length with the lead bulge, both lead exits). The cradle and the roof openings are
  drawn around the listing and a main lead at y -8, the balance lead at y 12.
- Plug parking spots and lead routes are approximate. Plug 1 on top is not fixed to anything yet.
- Head and neck motion against plug 1 and the leads on top of the box has not been checked.
- Hip sweep, native slicing and the release pipeline: none rerun.
- The cradle walls are 1.2 mm thick and only tied together once the bars are on; if they feel flexible, take them to 1.6 mm
  outward (the play stays the same).
- Add the cradle, bars and plug 1 on top to `cmu/model/variants/battery.json` and rerun the sim suite on the battery variant.
