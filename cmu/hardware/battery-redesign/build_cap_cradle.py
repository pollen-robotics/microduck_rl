"""Battery on the back panel: proposed redesign of the V1.20 battery backpack (not a release). The frame keeps the switch and DFR1202 exactly as V1.20; the cap
carries a printed cradle (floor, side walls, roof tabs) and a separate retainer bar across the battery's front face held
by 2 x M2 x 6. Removing the 3 cap screws takes the cap, cradle, bar and battery off together; plug 1 rides on slack so
the cap pulls ~40 mm back before you unplug. The box grows 1.1 mm deeper (frame rim slab copied back, cap moved back).
World frame mm, x forward, y left, z up. Builds from the V1.20 world STLs in cmu/hardware/battery-backpack and writes
battery_backpack_{frame,cap,bar}-world.stl, battery_backpack_cradle_added-world.stl (the cradle alone, for the viewer's
highlight), removed_recess.stl and checks.json to OUT_DIR:

    uv run --with trimesh --with manifold3d python cmu/hardware/battery-redesign/build_cap_cradle.py \
        cmu/hardware/battery-backpack OUT_DIR
"""
import json, sys, numpy as np, trimesh, manifold3d as mf
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
ROOF_TABS = [(-WALL_Y[1], -11.0), (-3.0, 6.0), (11.0, WALL_Y[1])]      # clear of the main lead at y -8; balance side to confirm
FLOOR_Z = (Z_FLOOR - 1.2, Z_FLOOR); FLOOR_HY = HY_LO - GAP             # cradle floor sits in a recess cut into the frame floor
BOSS = dict(y=18.6, z=-8.0, r=2.5, x=(BAR_X[0] - 5.0, BAR_X[0]))       # bar bosses on the wall fronts, below plug 1
BAR = dict(y=21.1, z=(BOSS['z'] - 2.5, BOSS['z'] + 2.5))

F0 = to_mf(load(f'{BP}/stl/battery_backpack_frame-world.stl')); C0 = to_mf(load(f'{BP}/stl/battery_backpack_cap-world.stl'))

# ---- frame: grow 1.1 at the rim, cut a 1.5 mm recess in the floor for the cradle floor. Switch and DFR untouched.
slab = F0 ^ box((X_SPLIT, -40, -40), (X_SPLIT + GROW, 40, 50))
recess = box((X_SPLIT - GROW - 0.1, -HY_LO, FLOOR_Z[0] - GAP), (BAR_X[0], HY_LO, Z_FLOOR + 0.5))
F1 = (F0 + slab.translate((-GROW, 0, 0))) - recess

# ---- cap + cradle (world, installed)
X_BACK = CAP_IN - 0.05
cr = []
for s in (1, -1):
    yl = sorted([s * WALL_Y[0], s * WALL_Y[1]])
    cr.append(box((X_BACK, yl[0], Z_STEP_TOP + GAP), (BAT_FRONT, yl[1], ROOF_Z[1])))
    cr.append(cyl_x(BOSS['r'], BOSS['x'][0], BOSS['x'][1], s * BOSS['y'], BOSS['z']))
for (y0, y1) in ROOF_TABS:
    cr.append(box((X_BACK, y0, ROOF_Z[0]), (BAT_FRONT, y1, ROOF_Z[1])))
cr.append(box((X_BACK, -FLOOR_HY, FLOOR_Z[0]), (BAT_FRONT, FLOOR_HY, FLOOR_Z[1])))
cradle = U(cr)
pil = U([cyl_x(0.8, BOSS['x'][0] - 0.5, BOSS['x'][1] + 0.1, s * BOSS['y'], BOSS['z']) for s in (1, -1)])
C1 = (C0.translate((-GROW, 0, 0)) + cradle) - pil

# ---- retainer bar (separate part)
bar = box((BAR_X[0], -BAR['y'], BAR['z'][0]), (BAR_X[1], BAR['y'], BAR['z'][1])) - U([cyl_x(1.15, BAR_X[0] - 1, BAR_X[1] + 1, s * BOSS['y'], BOSS['z']) for s in (1, -1)])

checks = {}
for nm, M in (('frame', F1), ('cap', C1), ('bar', bar)):
    t = to_tm(M)
    checks[nm] = dict(bodies=len(M.decompose()), status=str(M.status()), volume_cm3=round(M.volume() / 1000, 3), bounds=np.round(t.bounds, 2).tolist())
    t.export(f'{OUT}/battery_backpack_{nm}-world.stl')
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
  'plug 1 (rides with the cap)': ((-74.0, -24.0, -1.0), (-63.8, -18.8, 25.0)),
  'plug 2 (front bay)': ((-59.1, 19.9, 3.0), (-48.9, 25.1, 29.0)),
  'balance plug': ((bx + 1.0 - 3.0, 13.0, 36.55), (bx + 1.0 + 3.0, 21.0, 41.05)),
}
PM = {k: pbox(*v) for k, v in plugs.items()}
moving = C1 + bar + bat + PM['plug 1 (rides with the cap)'] + PM['balance plug']
sweep = lambda Mv, d, n=90: U([Mv.translate((-d * i / n, 0, 0)) for i in range(0, n + 1)])
env = {f: to_mf(load(f'{BP}/envelopes/{f}-world.stl')) for f in ['dfr1202', 'dfr_pads', 'switch_swann', 'swap_plugs']}
fit = {
  'battery vs cap + cradle (seated)': vol(bat ^ C1),
  'battery vs bar': vol(bat ^ bar),
  'battery vs frame': vol(bat ^ F1),
  'bar vs frame': vol(bar ^ F1),
  'cap + cradle vs frame (seated)': vol(C1 ^ F1),
  'battery slid into cradle from 20 mm in front (cap off)': vol(U([bat.translate((20 * i / 60, 0, 0)) for i in range(1, 61)]) ^ C1),
  'cap, battery, bar, plug 1 pulled 45 mm back vs frame': vol(sweep(moving, 45) ^ F1),
}
for k, E in env.items():
    fit[f'cap pulled back vs {k}'] = vol(sweep(moving, 45) ^ E)
pl = {}
for k, P in PM.items():
    pl[k] = vol(P ^ (F1 + C1 + bar + bat))
for k, E in env.items():
    pl[f'{k} vs cradle + bar'] = vol(E ^ (cradle + bar))
W = json.load(open(f'{BP}/envelopes/wires.json'))
routes = {k: dict(points=v['points'], r=0.9) for k, v in W.items() if k in ('servo lead to J3', '5 V lead to J4', 'robot servo inlet (from OpenRB J4)', 'robot A7Z lead (from the neck)')}
p1x = -68.9
routes['battery lead (16 AWG)'] = dict(r=1.3, moves='cap', points=[[bx, -8, BAT_TOP], [bx, -8, 38.0], [bx, -19, 38.0], [p1x, -21.4, 32.0], [p1x, -21.4, 25.0]])
routes['lead S red: plug 1 to switch (slack loop)'] = dict(r=0.9, slack=True, points=[[p1x, -21.4, -1.0], [-71.0, -23.0, -6.0], [-73.0, -27.0, -12.4], [-64.0, -29.5, -12.4], [-60.5, -29.5, -2.0], [-57.5, -27.0, -11.5], [-56.5, -22.0, -3.0], [-55.0, -15.0, 2.0], [-54.0, -5.0, 1.5]])
routes['lead S black: plug 1 to plug 2 (slack loop)'] = dict(r=0.9, slack=True, points=[[p1x + 1.0, -20.6, -1.0], [-71.5, -24.5, -7.0], [-72.0, -28.5, -12.0], [-65.0, -28.0, -11.0], [-61.5, -28.0, 0.0], [-58.5, -26.0, -12.0], [-57.0, -20.0, 3.5], [-57.0, 15.0, 4.0], [-54.0, 21.5, 3.0]])
routes['lead S red: switch to plug 2'] = dict(r=0.9, points=[[-54.0, 5.0, 1.5], [-55.0, 14.0, 3.0], [-54.0, 22.5, 3.0]])
routes['lead C: plug 2 to DFR VIN/GND'] = dict(r=0.9, points=[[-54.0, 22.5, 29.0], [-54.0, 22.5, 31.5], [-52.5, 16.5, 31.5], [-52.5, 16.5, 11.2], [-50.5, 10.0, 11.0]])
wires = {}
solids = F1 + C1 + bar + bat
for k, w in routes.items():
    a = np.asarray(w['points'], float); s = np.concatenate([np.linspace(a[i], a[i + 1], 60) for i in range(len(a) - 1)])
    T = U([mf.Manifold.sphere(w['r'], 12).translate(tuple(p)) for p in s[::2]])
    wires[k] = vol(T ^ ((F1 + C1 + bar) if '16 AWG' in k else solids))
    w['length_mm'] = round(float(np.sum(np.linalg.norm(np.diff(a, axis=0), axis=1))), 1)
# slack: straight-line reach from the switch lug to plug 1 with the cap 40 mm back vs the drawn lead length
lug = np.array([-54.0, -5.0, 1.5]); p1_out = np.array([p1x - 40.0, -21.4, -1.0])
slack = dict(lead_S_red_drawn_mm=routes['lead S red: plug 1 to switch (slack loop)']['length_mm'],
             straight_reach_at_40mm_mm=round(float(np.linalg.norm(p1_out - lug)), 1))
# driver lanes with the cap assembly off: upper cover screws through the frame only; bar screws from the front of the cradle
lanes = {f'upper cover screw ({y:+.1f}, 26)': vol(cyl_x(2.0, -110, -47.5, y, 26.0) ^ F1) for y in (16.1, -16.1)}
lanes.update({f'bar screw ({s * BOSS["y"]:+.1f}, {BOSS["z"]}) from the front, cap off': vol(cyl_x(2.0, BAR_X[1], BAR_X[1] + 30, s * BOSS['y'], BOSS['z']) ^ (C1 + bat)) for s in (1, -1)})
checks.update(fit_mm3=fit, plugs_mm3=pl, wires_mm3=wires, slack=slack, driver_lanes_mm3=lanes,
              layout=dict(battery_x=[BAT_BACK, BAT_FRONT], bar_x=list(BAR_X), cap_inner_x=CAP_IN, box_deeper_mm=GROW, play_mm=GAP,
                          roof_tabs_y=ROOF_TABS, walls_y=list(WALL_Y), boss=BOSS, bar=BAR))
json.dump(dict(checks=checks, routes=routes, plugs={k: dict(lo=list(v[0]), hi=list(v[1])) for k, v in plugs.items()}), open(f'{OUT}/checks.json', 'w'), indent=1)
print(json.dumps(checks, indent=1))
