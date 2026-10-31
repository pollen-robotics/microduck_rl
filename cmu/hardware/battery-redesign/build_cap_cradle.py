"""Battery on the back panel: proposed redesign of the V1.20 battery backpack (not a release). The frame keeps the switch and
DFR1202 exactly as V1.20; the cap carries a printed cradle (floor, side walls, one middle roof tab) and two separate retainer
bars across the battery's front face, each held by 2 x M2 x 6. The stiff battery leads leave through two openings in the roof
that straddle the cap seam, plug 1 sits on top of the box, and lead S comes back in through a third opening in the frame roof.
Unplug plug 1 on top, then the 3 cap screws take the cap, cradle, bars and battery off together. The box grows 1.1 mm deeper
(frame rim slab copied back, cap moved back). World frame mm, x forward, y left, z up. Builds from the V1.20 world STLs in
v120-source/ next to this script, with the envelopes and mating parts in cmu/hardware/battery-backpack, and writes
battery_backpack_{frame,cap,bar}-{world,print}.stl (the current parts, kept in cmu/hardware/battery-backpack/stl; the bar files
hold both bars), battery_backpack_cradle_added-world.stl (the cradle alone, kept in viewer/ for its highlight),
removed_recess.stl and checks.json:

    uv run --with trimesh --with manifold3d python cmu/hardware/battery-redesign/build_cap_cradle.py \\
        cmu/hardware/battery-backpack OUT_DIR
"""
import json, sys, numpy as np, trimesh, manifold3d as mf
from pathlib import Path
BP, OUT = sys.argv[1], sys.argv[2]
def load(p): m = trimesh.load(p, force='mesh', process=True); m.merge_vertices(); return m
def to_mf(m): return mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32), tri_verts=np.asarray(m.faces, np.uint32)))
def to_tm(M): g = M.to_mesh(); return trimesh.Trimesh(np.asarray(g.vert_properties)[:, :3], np.asarray(g.tri_verts), process=True)
def box(lo, hi): lo, hi = np.array(lo, float), np.array(hi, float); return mf.Manifold.cube(tuple(hi - lo)).translate(tuple(lo))
def cyl_x(r, x0, x1, y, z): return mf.Manifold.cylinder(x1 - x0, r, r, 48).rotate((0, 90, 0)).translate((x0, y, z))
U = lambda ms: mf.Manifold.batch_boolean(list(ms), mf.OpType.Add)
vol = lambda M: round(float(M.volume()), 3)

X_SPLIT, Z_FLOOR, ROOF_IN, Z_STEP_TOP, HY_LO = -67.05, -22.51, 41.80, -14.0, 16.5
GROW = 1.1
SW_BACK = -59.95
GAP = 0.3
BAR_T = 1.5
BAR_X = (SW_BACK - GAP - BAR_T, SW_BACK - GAP)                         # -61.75 .. -60.25
BAT = (31.0, 12.0, 57.0)
BAT_FRONT = BAR_X[0] - GAP; BAT_BACK = BAT_FRONT - BAT[1]              # -62.05 .. -74.05
CAP_IN = -73.25 - GROW                                                 # -74.35
assert abs((BAT_BACK - GAP) - CAP_IN) < 1e-6
BAT_TOP = Z_FLOOR + BAT[2]
WALL_Y = (BAT[0] / 2 + GAP, BAT[0] / 2 + GAP + 1.2)                    # 15.8 .. 17.0
ROOF_Z = (BAT_TOP + GAP, BAT_TOP + GAP + 1.2)                          # 34.79 .. 35.99
ROOF_TABS = [(-3.0, 6.0)]                                              # middle tab only: the side tabs blocked the stiff leads
FLOOR_Z = (Z_FLOOR - 1.2, Z_FLOOR); FLOOR_HY = HY_LO - GAP             # cradle floor sits in a recess cut into the frame floor
BOSS = dict(y=18.6, zs=(-8.0, 24.0), r=2.5, x=(CAP_IN - 0.05, BAR_X[0]), pilot=5.5)   # one boss pair per bar, beside the cradle walls;
                                                                      # columns from the cap wall so they rise off the bed in the print pose
BAR = dict(y=21.1, h=5.0)
# Roof openings, cut from z 36.5 up (clear of the roof tab). Main and balance openings straddle the cap seam so a lead standing
# on the pack's top centre (x -68.05) never traps the cap, and pass an XT30 (10.2 x 5.2) or JST-XH without unplugging.
ROOF_CUT_Z = 36.5
SEAM_X = (-71.5, -64.6)
OPEN_MAIN_Y, OPEN_BAL_Y = (-14.0, -2.0), (7.0, 17.0)
OPEN_S = dict(x=(-60.0, -49.0), y=(-25.0, -18.5))                      # frame only: lead S back in, passes its XT30; long side in x so the print bridge is 6.5 mm

V120 = Path(__file__).resolve().parent / 'v120-source'
F0 = to_mf(load(f'{V120}/battery_backpack_frame-world.stl')); C0 = to_mf(load(f'{V120}/battery_backpack_cap-world.stl'))

# ---- frame: grow 1.1 at the rim, cut a 1.5 mm recess in the floor for the cradle floor. Switch and DFR untouched.
slab = F0 ^ box((X_SPLIT, -40, -40), (X_SPLIT + GROW + 0.5, 40, 50))   # 0.5 mm overlap: a face-to-face join at the old rim
                                                                     # left zero-volume slivers that slicers flag as non-manifold
recess = box((X_SPLIT - GROW - 0.1, -HY_LO, FLOOR_Z[0] - GAP), (BAR_X[0], HY_LO, Z_FLOOR + 0.5))
openings = {'main': box((SEAM_X[0], OPEN_MAIN_Y[0], ROOF_CUT_Z), (SEAM_X[1], OPEN_MAIN_Y[1], 50)),
            'balance': box((SEAM_X[0], OPEN_BAL_Y[0], ROOF_CUT_Z), (SEAM_X[1], OPEN_BAL_Y[1], 50)),
            'lead S': box((OPEN_S['x'][0], OPEN_S['y'][0], ROOF_CUT_Z), (OPEN_S['x'][1], OPEN_S['y'][1], 50))}
F1 = (F0 + slab.translate((-GROW, 0, 0))) - recess - U(openings.values())

# ---- cap + cradle (world, installed)
X_BACK = CAP_IN - 0.05
cr = []
for s in (1, -1):
    yl = sorted([s * WALL_Y[0], s * WALL_Y[1]])
    cr.append(box((X_BACK, yl[0], Z_STEP_TOP + GAP), (BAT_FRONT, yl[1], BAT_TOP)))   # walls stop at the pack's top, clear of the leads
    for z in BOSS['zs']:
        cr.append(cyl_x(BOSS['r'], BOSS['x'][0], BOSS['x'][1], s * BOSS['y'], z))
for (y0, y1) in ROOF_TABS:
    cr.append(box((X_BACK, y0, ROOF_Z[0]), (BAT_FRONT, y1, ROOF_Z[1])))
cr.append(box((X_BACK, -FLOOR_HY, FLOOR_Z[0]), (BAT_FRONT, FLOOR_HY, FLOOR_Z[1])))
cradle = U(cr)
pil = U([cyl_x(0.8, BOSS['x'][1] - BOSS['pilot'], BOSS['x'][1] + 0.1, s * BOSS['y'], z) for s in (1, -1) for z in BOSS['zs']])
C1 = (C0.translate((-GROW, 0, 0)) + cradle) - pil - openings['main'] - openings['balance']

# ---- two retainer bars (identical separate parts, one file)
def one_bar(z):
    return box((BAR_X[0], -BAR['y'], z - BAR['h'] / 2), (BAR_X[1], BAR['y'], z + BAR['h'] / 2)) - U([cyl_x(1.15, BAR_X[0] - 1, BAR_X[1] + 1, s * BOSS['y'], z) for s in (1, -1)])
bar = U([one_bar(z) for z in BOSS['zs']])

checks = {}
for nm, M in (('frame', F1), ('cap', C1), ('bar', bar)):
    t = to_tm(M)
    checks[nm] = dict(bodies=len(M.decompose()), status=str(M.status()), volume_cm3=round(M.volume() / 1000, 3), bounds=np.round(t.bounds, 2).tolist())
    t.export(f'{OUT}/battery_backpack_{nm}-world.stl')
    # print pose as V1.20 (x' = y, y' = z, z' = x): frame rim down, cap back face down, bars flat; lowest point on z = 0
    tp = t.copy(); tp.apply_transform(np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], float)); tp.apply_translation(-tp.bounds[0])
    tp.export(f'{OUT}/battery_backpack_{nm}-print.stl')
to_tm(cradle - pil).export(f'{OUT}/battery_backpack_cradle_added-world.stl')
to_tm(recess ^ F0).export(f'{OUT}/removed_recess.stl')
checks['cap']['cradle_volume_cm3'] = round((cradle - C0.translate((-GROW, 0, 0))).volume() / 1000, 3)
checks['mass_g_est'] = dict(cradle=round(checks['cap']['cradle_volume_cm3'] * 1.24 * 0.851, 1), bar=round(checks['bar']['volume_cm3'] * 1.24 * 0.851, 1),
                            frame_delta=round((F1.volume() - F0.volume()) / 1000 * 1.24 * 0.851, 1))

bat = box((BAT_BACK, -BAT[0] / 2, Z_FLOOR), (BAT_FRONT, BAT[0] / 2, BAT_TOP))
XT = (10.2, 5.2, 26.0)
def pbox(lo, hi): return box(lo, hi)
bx = BAT_FRONT - BAT[1] / 2
plugs = {
  'plug 1 (on top of the box)': ((-71.5, -21.0, 44.0), (-45.5, -11.0, 49.2)),
  'plug 2 (front bay)': ((-59.1, 19.9, 3.0), (-48.9, 25.1, 29.0)),
  'balance plug (on top of the box)': ((bx - 3.0, 8.0, 44.0), (bx + 3.0, 16.0, 48.5)),
}
PM = {k: pbox(*v) for k, v in plugs.items()}
def tube(points, r):
    a = np.asarray(points, float); s = np.concatenate([np.linspace(a[i], a[i + 1], 60) for i in range(len(a) - 1)])
    return U([mf.Manifold.sphere(r, 12).translate(tuple(p)) for p in s[::2]])
MAIN_PTS = [[bx, -8, BAT_TOP], [bx, -8, 44.5], [-70.0, -11.0, 47.0], [-73.0, -14.0, 47.5], [-71.5, -16.0, 46.6]]
BAL_PTS = [[bx, 12, BAT_TOP], [bx, 12, 44.0]]
moving = C1 + bar + bat + tube(MAIN_PTS, 1.3) + tube(BAL_PTS, 0.7) + PM['balance plug (on top of the box)']   # plug 1 is unplugged first
sweep = lambda Mv, d, n=90: U([Mv.translate((-d * i / n, 0, 0)) for i in range(0, n + 1)])
env = {f: to_mf(load(f'{BP}/envelopes/{f}-world.stl')) for f in ['dfr1202', 'dfr_pads', 'switch_swann', 'swap_plugs']}
fit = {
  'battery vs cap + cradle (seated)': vol(bat ^ C1),
  'battery vs bar': vol(bat ^ bar),
  'battery vs frame': vol(bat ^ F1),
  'bar vs frame': vol(bar ^ F1),
  'cap + cradle vs frame (seated)': vol(C1 ^ F1),
  'battery slid into cradle from 20 mm in front (cap off)': vol(U([bat.translate((20 * i / 60, 0, 0)) for i in range(1, 61)]) ^ C1),
  'cap, battery, bars and leads pulled 45 mm back vs frame': vol(sweep(moving, 45) ^ F1),
  'plug 1 passed through the main opening': vol(box((bx - 2.6, -13.1, ROOF_CUT_Z), (bx + 2.6, -2.9, 50.0)) ^ (F1 + C1)),
  'lead S XT30 passed through its frame opening': vol(box((-59.7, -24.35, ROOF_CUT_Z), (-49.5, -19.15, 50.0)) ^ F1),
}
for k, E in env.items():
    fit[f'cap pulled back vs {k}'] = vol(sweep(moving, 45) ^ E)
pl = {}
for k, P in PM.items():
    pl[k] = vol(P ^ (F1 + C1 + bar + bat))
for k, E in env.items():
    pl[f'{k} vs cradle + bar'] = vol(E ^ (cradle + bar))
for f in ('shell_left', 'shell_right', 'rear_pack'):
    pl[f'plugs on top vs {f}'] = vol(to_mf(load(f'{BP}/mating/{f}.stl')) ^ (PM['plug 1 (on top of the box)'] + PM['balance plug (on top of the box)']))
W = json.load(open(f'{BP}/envelopes/wires.json'))
routes = {k: dict(points=v['points'], r=0.9) for k, v in W.items() if k in ('servo lead to J3', '5 V lead to J4', 'robot servo inlet (from OpenRB J4)', 'robot A7Z lead (from the neck)')}
routes['battery lead (16 AWG)'] = dict(r=1.3, moves='cap', points=MAIN_PTS)
routes['balance lead'] = dict(r=0.7, moves='cap', points=BAL_PTS)
routes['lead S red: plug 1 to switch'] = dict(r=0.9, points=[[-45.5, -16.0, 46.6], [-44.0, -18.0, 45.5], [-47.0, -21.5, 44.6], [-54.5, -21.75, 44.0], [-54.5, -21.75, 41.0], [-54.5, -21.75, 10.0], [-55.0, -15.0, 2.0], [-54.0, -5.0, 1.5]])
routes['lead S black: plug 1 to plug 2'] = dict(r=0.9, points=[[-45.5, -14.0, 46.6], [-44.0, -16.0, 45.5], [-47.0, -20.0, 44.6], [-52.5, -20.2, 44.0], [-52.5, -20.2, 41.0], [-52.5, -20.2, 10.0], [-56.5, -18.0, 5.0], [-56.5, 18.0, 5.0], [-54.0, 21.5, 3.0]])
routes['lead S red: switch to plug 2'] = dict(r=0.9, points=[[-54.0, 5.0, 1.5], [-55.0, 14.0, 3.0], [-54.0, 22.5, 3.0]])
routes['lead C: plug 2 to DFR VIN/GND'] = dict(r=0.9, points=[[-54.0, 22.5, 29.0], [-54.0, 22.5, 31.5], [-52.5, 16.5, 31.5], [-52.5, 16.5, 11.2], [-50.5, 10.0, 11.0]])
wires = {}
solids = F1 + C1 + bar + bat
for k, w in routes.items():
    a = np.asarray(w['points'], float)
    T = tube(w['points'], w['r'])
    wires[k] = vol(T ^ ((F1 + C1 + bar) if w.get('moves') else solids))
    w['length_mm'] = round(float(np.sum(np.linalg.norm(np.diff(a, axis=0), axis=1))), 1)
leads = {k: routes[k]['length_mm'] for k in ('battery lead (16 AWG)', 'balance lead', 'lead S red: plug 1 to switch', 'lead S black: plug 1 to plug 2', 'lead S red: switch to plug 2', 'lead C: plug 2 to DFR VIN/GND')}
# driver lanes with the cap assembly off: upper cover screws through the frame only; bar screws from the front of the cradle
lanes = {f'upper cover screw ({y:+.1f}, 26)': vol(cyl_x(2.0, -110, -47.5, y, 26.0) ^ F1) for y in (16.1, -16.1)}
lanes.update({f'bar screw ({s * BOSS["y"]:+.1f}, {z}) from the front, cap off': vol(cyl_x(2.0, BAR_X[1], BAR_X[1] + 30, s * BOSS['y'], z) ^ (C1 + bat)) for s in (1, -1) for z in BOSS['zs']})
checks.update(fit_mm3=fit, plugs_mm3=pl, wires_mm3=wires, lead_lengths_mm=leads, driver_lanes_mm3=lanes,
              layout=dict(battery_x=[BAT_BACK, BAT_FRONT], bar_x=list(BAR_X), cap_inner_x=CAP_IN, box_deeper_mm=GROW, play_mm=GAP,
                          roof_tabs_y=ROOF_TABS, walls_y=list(WALL_Y), boss=BOSS, bar=BAR,
                          roof_openings=dict(main=dict(x=SEAM_X, y=OPEN_MAIN_Y), balance=dict(x=SEAM_X, y=OPEN_BAL_Y), lead_S=OPEN_S)))
json.dump(dict(checks=checks, routes=routes, plugs={k: dict(lo=list(v[0]), hi=list(v[1])) for k, v in plugs.items()}), open(f'{OUT}/checks.json', 'w'), indent=1)
print(json.dumps(checks, indent=1))
