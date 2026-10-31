# cmu/hardware: the parts you can redesign and print

Three things live here, all for teammates who want to change the robot's printed parts rather than only its policies:

- [battery-backpack/](battery-backpack/README.md): the 2S battery backpack: the current STL files in robot and print
  orientation, and from V1.20 the design brief and its fixed interfaces, the bought-part envelopes
  (battery, switch, 5 V board, plugs, wires) it was drawn around, the mating robot parts to design against, the mass
  and balance numbers, the wiring and assembly page, the parts list, and the CAD scripts that produced it.
- [printing/](printing/README.md): how we turn an STL into a file the two RoboClub printers accept (Bambu Lab X1 Carbon,
  Elegoo Centauri Carbon): the slicer profiles we use, the exact command lines, the orientation and brim rules, the
  Centauri preview rule, the delivery format, and the ready-to-print V1.20 files of the backpack.
- [battery-redesign/](battery-redesign/README.md): the current backpack design, which carries the battery on the cap in
  a printed cradle closed by two screwed bars: its checks, viewer and the script that builds the parts in
  `battery-backpack/stl/` from the V1.20 frame and cap kept in its `v120-source/`. Owned here, not a copy.

Everything except the `README.md` pages, `battery-backpack/stl/` and `battery-redesign/` is a copy of the private robotics working tree, refreshed by
`python3 cmu/sync_from_robotics.py` (see [cmu/README.md](../README.md)); do not edit the copies, change the source and
resync. Several copied scripts import private modules that are not in this repository (the robot's CAD library, the
interference audit, the bounded CAD runner); they are here as the authoritative record of how the numbers were
produced, and each README says what runs as is and what is reference only.

Frames and units: millimetres; the robot frame has x forward, y left, z up, origin as in the MJCF of
`src/mjlab_microduck/robot/microduck/`. The backpack sits at negative x (behind the torso).
