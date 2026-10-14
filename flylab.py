"""
FlyLab - single-file control panel.
One window, two tabs:
  * Control Panel : live MaleCNS brain (paint neurons), the NeuroMechFly fly body, and the Fly Bridge key controls
  * MaleCNS Setup : builds a subset of the real MaleCNS v1.0 connectome (optionally aligned with navis transforms).

The fly body is the official NeuroMechFly model (FlyGym 2.x meshes). Its legs walk with a tripod gait
and the brain's descending-neuron drives steer it (forward / back / left / right).
The brain view uses coordinates from brain_subset.npz (built from MaleCNS v1.0).

Run:  python flylab.py

One-time install. FlyGym 2.x needs Python 3.12 or newer (check with:  python --version) and git.
Use the SAME python for every line and to run this file (on Windows, e.g. replace "python" with "py -3.12"):
  python -m pip install numpy scipy opencv-python mss pydirectinput mujoco
  python -m pip uninstall -y flygym
  python -m pip install "flygym @ git+https://github.com/NeLy-EPFL/flygym.git@v2.1.0"
MaleCNS Setup tab:        python -m pip install pyarrow pandas neuprint-python
In the app:  Control Panel = live brain + NeuroMechFly body (drag the fly to rotate it), 'Mode / preset' drop-down
(Dynamic environment = the fly wanders by itself, or any BadApple-fly movement program), Build tab = place obstacles and food.
Template alignment (optional, navis transforms tutorial):  python -m pip install navis flybrains h5py
"""
from collections import deque
import collections, random, ctypes, os, sys, json, time, threading, base64, math, re, queue, warnings, textwrap
# Make Windows report real pixels to everything (Tk, screen capture, clicks) so coordinates all agree.
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass
import mss, cv2, pydirectinput
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog
import numpy as np
import scipy.sparse as sp
# =====================================================================
# PART 1 - BRAIN ENGINE (leaky integrate-and-fire, CPU only)
# =====================================================================
DT = 1.0                                   # ms per simulation step
V_REST, V_RESET, V_THRESH = -52.0, -52.0, -45.0   # mV
TAU_M, TAU_SYN, T_REF = 20.0, 5.0, 2.2     # ms
W_SYN = 0.275                              # mV added to g per synapse-count unit
EXT_W = 40.0                               # weight units added by one external (painted) spike
FLASH_TAU = 120.0                          # ms, visual afterglow of a spike
RATE_TAU = 150.0                           # ms, smoothing of the DN group rates
GROUPS = ("fwd", "back", "left", "right")
_DECAY_SYN = float(np.exp(-DT / TAU_SYN))
_DECAY_FLASH = float(np.exp(-DT / FLASH_TAU))
ACT_TAU = 350.0                            # ms, slow 'recent activity' trace: smooth glow of what the network has been doing
_DECAY_ACT = float(np.exp(-DT / ACT_TAU))
_ALPHA_RATE = DT / RATE_TAU
ADAPT_TAU, ADAPT_JUMP = 300.0, 0.4          # spike-frequency adaptation: ms, mV added per spike (stops activity latching on forever)
_DECAY_ADAPT = float(np.exp(-DT / ADAPT_TAU))
# BGR colours: region 0 = central brain (purple), 1 = optic lobes (blue), 2 = nerve cord (amber)
REGION_COLORS = np.array([(255, 140, 192), (255, 196, 96), (100, 186, 250)], np.float32)
BG_BGR = np.array([39, 36, 16], np.float32)          # #102427, dark teal like the reference view
MINT_BGR, FIRE_BGR = (190, 230, 110), (110, 215, 255)  # stimulated / firing
def assign_regions(P, k=4, seed=0):
    """Visual guess of 0 = central brain, 1 = optic lobes, 2 = nerve cord from where the points sit."""
    P = np.asarray(P, np.float32)
    if len(P) < 50:
        return np.zeros(len(P), np.int8)
    rng = np.random.default_rng(seed)
    sub = P[rng.choice(len(P), min(len(P), 15000), replace=False)]
    C = [sub[rng.integers(len(sub))]]
    for _ in range(k - 1):
        d2 = np.min([((sub - c) ** 2).sum(1) for c in C], axis=0) + 1e-12
        C.append(sub[rng.choice(len(sub), p=d2 / d2.sum())])
    C = np.array(C)
    for _ in range(15):
        lab = np.argmin(((sub[:, None, :] - C[None]) ** 2).sum(2), axis=1)
        for j in range(k):
            if (lab == j).any():
                C[j] = sub[lab == j].mean(0)
    full = np.concatenate([np.argmin(((P[i:i + 20000, None, :] - C[None]) ** 2).sum(2), axis=1)
                           for i in range(0, len(P), 20000)])
    d_other = [float(np.linalg.norm(C[j] - np.delete(C, j, 0).mean(0))) for j in range(k)]
    vnc = int(np.argmax(d_other))
    rest = [j for j in range(k) if j != vnc]
    isolated = d_other[vnc] > 1.8 * float(np.median([d_other[j] for j in rest]))
    pool = rest if isolated else list(range(k))
    mid = C[pool].mean(0)
    central = pool[int(np.argmin([np.linalg.norm(C[j] - mid) for j in pool]))]
    out = np.ones(len(P), np.int8)
    out[full == central] = 0
    if isolated:
        out[full == vnc] = 2
    return out
def make_demo_anatomy(m=30000, seed=2):
    """Decorative stand-in anatomy (central brain, two optic lobes, nerve cord) for the demo network."""
    rng = np.random.default_rng(seed)
    def blob(c, r, n, rot=0.0):
        p = rng.normal(size=(n, 3))
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        p *= (0.65 + 0.35 * rng.random((n, 1)))               # mostly shell-like
        p *= np.asarray(r, np.float32)
        if rot:
            c_, s_ = math.cos(rot), math.sin(rot)
            p[:, :2] = p[:, :2] @ np.array([[c_, s_], [-s_, c_]], np.float32)
        return p + np.asarray(c, np.float32)
    parts = [blob((0, 0, 0), (0.5, 0.55, 0.4), int(m * .34)),
             blob((-0.8, 0, 0), (0.3, 0.42, 0.38), int(m * .18)),
             blob((0.8, 0, 0), (0.3, 0.42, 0.38), int(m * .18)),
             blob((1.9, 0.75, 0), (0.8, 0.26, 0.2), int(m * .30), rot=-0.2)]
    lab = np.concatenate([np.full(len(p), l, np.int8) for p, l in zip(parts, (0, 1, 1, 2))])
    return np.vstack(parts).astype(np.float32), lab
class BrainNet:
    """W[post, pre] (signed synapse counts), 3D positions, the four DN groups, and an optional
    'context' cloud of extra neurons that is only drawn (never simulated) so the anatomy has a shape."""
    def __init__(self, W, pos, groups, source="demo", ctx_pos=None, labels=None,
                 skel_pos=None, skel_edges=None, skel_grp=None):
        self.W = sp.csc_matrix(W, dtype=np.float32)
        self.n = self.W.shape[0]
        sim = np.asarray(pos, np.float32)
        ctx = np.zeros((0, 3), np.float32) if ctx_pos is None else np.asarray(ctx_pos, np.float32).reshape(-1, 3)
        allp = np.vstack([sim, ctx])
        mu = allp.mean(0)
        allp = allp - mu
        sc = float(np.abs(allp).max()) + 1e-9
        allp = allp / sc
        if skel_pos is not None and len(skel_pos) and skel_edges is not None and len(skel_edges):
            sp_ = np.nan_to_num(np.asarray(skel_pos, np.float32).reshape(-1, 3), nan=0.0)
            self.skel_pos = ((sp_ - mu) / sc).astype(np.float32)
            self.skel_edges = np.asarray(skel_edges, np.int64).reshape(-1, 2)
            self.skel_grp = np.asarray(skel_grp, np.int8).reshape(-1)
            if len(self.skel_edges) > 12000:                 # keep the redraw fast
                keep = np.linspace(0, len(self.skel_edges) - 1, 12000).astype(np.int64)
                self.skel_edges, self.skel_grp = self.skel_edges[keep], self.skel_grp[keep]
        else:
            self.skel_pos = np.zeros((0, 3), np.float32)
            self.skel_edges = np.zeros((0, 2), np.int64)
            self.skel_grp = np.zeros(0, np.int8)
        self.all_pos = allp.astype(np.float32)             # first n rows = simulated neurons
        self.pos = self.all_pos[: self.n]
        self.n_ctx = len(ctx)
        self.labels = assign_regions(self.all_pos) if labels is None else np.asarray(labels, np.int8)
        self.groups = {g: np.asarray(groups.get(g, []), dtype=np.int64) for g in GROUPS}
        self.presets = {}                                  # preset id -> {"L": rows, "R": rows}
        self.source = source
def load_net(path):
    d = np.load(path, allow_pickle=False)
    W = sp.csc_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(int(s) for s in d["shape"]))
    groups = {g: d["grp_" + g] for g in GROUPS}
    ctx = d["ctx_pos"] if "ctx_pos" in d.files else None
    sk = {}
    if "skel_pos" in d.files:
        sk = dict(skel_pos=d["skel_pos"], skel_edges=d["skel_edges"], skel_grp=d["skel_grp"])
    labels = d["labels"] if "labels" in d.files else None
    net = BrainNet(W, d["pos"], groups, source=os.path.basename(path), ctx_pos=ctx, labels=labels, **sk)
    for k in d.files:                                      # movement-program neurons: pre_<id>_<L|R>
        if k.startswith("pre_"):
            pid, side = k[4:].rsplit("_", 1)
            net.presets.setdefault(pid, {})[side] = d[k].astype(np.int64)
    return net
def make_demo_net(n=4000, k_out=25, n_dn=80, seed=1):
    """Stand-in network so the app runs before the MaleCNS subset is built.
    Neurons are laid out top-to-bottom; signals flow downward to 4 DN groups at the bottom."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(n, 3)) * np.array([1.0, 1.3, 0.7])
    pos = pos[np.argsort(-pos[:, 1])]                       # index 0 = top, last = bottom
    inhib = rng.random(n) < 0.3
    win = n // 4
    rows, cols, vals = [], [], []
    for i in range(n - n_dn):
        lo = max(0, i - win // 3) if inhib[i] else i + 1       # inhibitory cells also feed back sideways
        cand = np.arange(lo, min(n, i + 1 + win))
        cand = cand[cand != i]
        d2 = (pos[cand, 0] - pos[i, 0]) ** 2 + (pos[cand, 2] - pos[i, 2]) ** 2
        p = np.exp(-d2 / 0.1)
        p /= p.sum()
        tg = rng.choice(cand, size=min(k_out, cand.size), replace=False, p=p)
        w = rng.integers(20, 75, size=tg.size).astype(np.float32)
        w *= -2.0 if inhib[i] else 1.0
        rows.append(tg)
        cols.append(np.full(tg.size, i))
        vals.append(w)
    W = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    dn = np.arange(n - n_dn, n)
    dn = dn[np.argsort(pos[dn, 0])]                         # sort by x: left ... right
    q = n_dn // 4
    mid = dn[q:-q]
    mid = mid[np.argsort(pos[mid, 2])]                      # sort by depth (z)
    groups = {"left": dn[:q], "right": dn[-q:], "back": mid[: mid.size // 4], "fwd": mid[mid.size // 4:]}
    ctx, ctx_lab = make_demo_anatomy()
    disp = pos / np.abs(pos).max(0) * np.array([0.38, 0.42, 0.3], np.float32)   # simulated cells sit inside the central brain
    return BrainNet(W, disp, groups, source="DEMO network", ctx_pos=ctx,
                    labels=np.concatenate([np.zeros(n, np.int8), ctx_lab]))
class LIFSim:
    def __init__(self, net):
        self.net = net
        n = net.n
        self.gain = 1.0                    # global multiplier on all synapses
        self.stim_rate = 100.0             # Hz, external Poisson rate on painted neurons
        self.stim_idx = np.zeros(0, np.int64)
        self._ext = (np.zeros(0, np.int64), np.zeros(0, np.float32))   # (neurons, spike probability per step)
        self.running = True
        self.speed = 0.0                   # simulated ms per wall ms (1.0 = real time)
        self.spikes_per_s = 0.0
        self.rates = {g: 0.0 for g in GROUPS}   # smoothed Hz per neuron in each DN group
        self.gid = np.full(n, -1, np.int64)
        for k, g in enumerate(GROUPS):
            self.gid[net.groups[g]] = k
        self._gsize = np.array([max(1, net.groups[g].size) for g in GROUPS], np.float64)
        self._rng = np.random.default_rng()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.reset()
    def reset(self):
        with self._lock:
            n = self.net.n
            self.v = np.full(n, V_REST, np.float32)
            self.g = np.zeros(n, np.float32)
            self.ref = np.zeros(n, np.float32)
            self.flash = np.zeros(n, np.float32)
            self.act = np.zeros(n, np.float32)
            self.adapt = np.zeros(n, np.float32)
            self._last = np.zeros(0, np.int64)
            self._pop = 0.0
            self.t_ms = 0.0
            self.n_spk = 0                 # total spikes so far (the Graphs window reads this)
            for k in self.rates:
                self.rates[k] = 0.0
    def set_stim(self, mask):
        self.stim_idx = np.flatnonzero(mask).astype(np.int64)
    def set_ext(self, rates_hz):
        """Per-neuron Poisson input rates (Hz) from the environment / presets."""
        idx = np.flatnonzero(rates_hz > 0.5).astype(np.int64)
        self._ext = (idx, (rates_hz[idx] * DT / 1000.0).astype(np.float32))
    def _gather(self, idx):
        """Sum the weights of all outgoing synapses of the spiking neurons -> input per neuron."""
        W = self.net.W
        starts, ends = W.indptr[idx], W.indptr[idx + 1]
        counts = ends - starts
        tot = int(counts.sum())
        if tot == 0:
            return None
        sel = np.repeat(starts - (np.cumsum(counts) - counts), counts) + np.arange(tot)
        return np.bincount(W.indices[sel], weights=W.data[sel], minlength=self.net.n)
    def step(self):
        # synaptic input from last step's spikes (+ painted external Poisson input)
        self.g *= _DECAY_SYN
        if self._last.size:
            inp = self._gather(self._last)
            if inp is not None:
                self.g += (inp * (W_SYN * self.gain / (1.0 + (self._pop / POP_REF) ** 2))).astype(np.float32)
        si = self.stim_idx
        if si.size:
            hit = si[self._rng.random(si.size) < self.stim_rate * DT / 1000.0]
            if hit.size:
                self.g[hit] += EXT_W * W_SYN
        ei, ep = self._ext
        if ei.size:
            hit = ei[self._rng.random(ei.size) < ep]
            if hit.size:
                self.g[hit] += EXT_W * W_SYN
        # membrane update (clamped while refractory)
        free = self.ref <= 0
        self.v += (V_REST - self.v + self.g - self.adapt) * (DT / TAU_M) * free
        self.ref -= DT
        spk = np.flatnonzero(self.v >= V_THRESH)
        if spk.size:
            self.v[spk] = V_RESET
            self.ref[spk] = T_REF
        self.adapt *= _DECAY_ADAPT
        if spk.size:
            self.adapt[spk] += ADAPT_JUMP
        self._last = spk
        self.n_spk += int(spk.size)
        self._pop += 0.02 * (spk.size / self.net.n - self._pop)     # global activity level (about 50 ms memory)
        self.flash *= _DECAY_FLASH
        self.flash[spk] = 1.0
        self.act *= _DECAY_ACT
        self.act[spk] += 0.3
        self.t_ms += DT
        # DN group rates (Hz per neuron, exponentially smoothed)
        cnt = np.bincount(self.gid[spk][self.gid[spk] >= 0], minlength=len(GROUPS))
        inst = cnt / self._gsize / (DT / 1000.0)
        for k, g in enumerate(GROUPS):
            self.rates[g] += _ALPHA_RATE * (inst[k] - self.rates[g])
        return spk.size
    def drives(self, full_hz=40.0):
        """(fwd, back, left, right), each 0..1 (full_hz = group rate that counts as 'full drive')."""
        rf, rb, rl, rr = (self.rates[g] / full_hz for g in GROUPS)
        ny, nx = rf - rb, rl - rr                           # opposing descending groups cancel (push-pull), like antagonist muscles
        return (min(1.0, max(0.0, ny)), min(1.0, max(0.0, -ny)), min(1.0, max(0.0, nx)), min(1.0, max(0.0, -nx)))
    # ---- background thread -------------------------------------------------
    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
    def stop(self):
        self._stop.set()
    def _run(self):
        wall0, sim0, nspk = time.perf_counter(), self.t_ms, 0
        while not self._stop.is_set():
            if not self.running:
                time.sleep(0.02)
                wall0, sim0, nspk = time.perf_counter(), self.t_ms, 0
                continue
            with self._lock:
                for _ in range(5):
                    nspk += self.step()
            wall_ms = (time.perf_counter() - wall0) * 1000.0
            sim_ms = self.t_ms - sim0
            if wall_ms >= 1000.0:                      # refresh the speed read-out once a second
                self.speed = sim_ms / wall_ms
                self.spikes_per_s = nspk / (wall_ms / 1000.0)
                wall0, sim0, nspk = time.perf_counter(), self.t_ms, 0
            elif sim_ms > wall_ms:                     # never run faster than real time
                time.sleep(min((sim_ms - wall_ms) / 1000.0, 0.02))
# =====================================================================
# PART 1b - WORLD (arena with obstacles and food) + wandering + input sets
# =====================================================================
ARENA_HALF = 40.0          # mm; the arena is 80 x 80 mm
FLY_R = 1.4                # mm, body radius used for collisions
FOOD_RANGE = 22.0          # mm, how far the fly "smells" food
AVOID_RANGE = 6.0          # mm, how far ahead it "sees" obstacles and walls
EAT_RANGE = 1.6            # mm
EAT_SECONDS = 4.0          # how long one food pellet takes to eat
ODOR_PULSE = 0.35                                      # concentration a test pulse adds at an antenna
SMELL_PLUME, SMELL_RADIUS = "Plume (fluid dynamics)", "Radius (Deprecated)"
ENV_RATE_HZ = 600.0        # Poisson rate on a descending-neuron group at full command
UP_POOL = 120              # strongest upstream (premotor) neurons per DN group that carry the command into the network
POP_REF = 0.0025           # population rate (spikes per neuron per ms, = 2.5 Hz) above which global inhibition damps the synapses
UP_GAIN_POW = 1.0          # the stronger the synapse gain slider, the weaker this extra drive (keeps the network from running away)
UP_RATE_HZ = 100.0          # Poisson rate on those at full command, so activity visibly flows through the brain on the way to the DNs
def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
class Arena:
    """Flat arena: round obstacles, food pellets, walls. All units are mm, +y is 'up' on the Build tab."""
    def __init__(self):
        self.half = ARENA_HALF
        self.obstacles, self.foods, self.food_spawn = [], [], []
        self.eaten = 0
        self.smell_range = FOOD_RANGE                      # mm: how far the smell of food reaches (Build tab slider)
        self.version = 0                                   # bumped whenever the layout changes (UI redraw)
        self.plume = Plume(self.half)
    def antennae(self, x, y, h):
        c, s = math.cos(h), math.sin(h)
        return np.array([[x + c * 1.0 - s * 0.8, y + s * 1.0 + c * 0.8], [x + c * 1.0 + s * 0.8, y + s * 1.0 - c * 0.8]])
    def near_food(self, x, y, h, rng=5.0):
        best = None
        for fx, fy, _a in self.foods:
            d = math.hypot(fx - x, fy - y)
            if d < rng and (best is None or d < best[0]):
                best = (d, wrap_pi(math.atan2(fy - y, fx - x) - h))
        return None if best is None else best[1]
    def default_layout(self):
        self.obstacles = [[-14.0, 10.0, 4.0], [12.0, -8.0, 5.0], [18.0, 18.0, 3.0], [-20.0, -18.0, 4.5]]
        self.food_spawn = [[-26.0, 24.0], [26.0, 4.0], [0.0, -26.0], [-8.0, 27.0], [29.0, -22.0], [-30.0, -4.0]]
        self.foods = [[f[0], f[1], 1.0] for f in self.food_spawn]
        self.version += 1
    def add_obstacle(self, x, y, r):
        self.obstacles.append([float(x), float(y), float(r)])
        self.version += 1
    def add_food(self, x, y):
        self.foods.append([float(x), float(y), 1.0])
        self.food_spawn.append([float(x), float(y)])
        self.version += 1
    def erase(self, x, y, rad=3.0):
        self.obstacles = [o for o in self.obstacles if math.hypot(o[0] - x, o[1] - y) > max(rad, o[2])]
        keep = [f for f in self.foods if math.hypot(f[0] - x, f[1] - y) > rad]
        self.food_spawn = [f for f in self.food_spawn if math.hypot(f[0] - x, f[1] - y) > rad]
        self.foods = keep
        self.version += 1
    def clear(self):
        self.obstacles, self.foods, self.food_spawn = [], [], []
        self.plume.clear()
        self.version += 1
    def refill(self):
        self.foods = [[f[0], f[1], 1.0] for f in self.food_spawn]
        self.version += 1
    def resolve(self, x, y):
        """Push the fly out of obstacles and keep it inside the walls."""
        for ox, oy, r in self.obstacles:
            dx, dy = x - ox, y - oy
            d = math.hypot(dx, dy)
            lim = r + FLY_R
            if d < lim:
                if d < 1e-6:
                    dx, dy, d = 1.0, 0.0, 1.0
                x, y = ox + dx / d * lim, oy + dy / d * lim
        m = self.half - FLY_R
        return min(max(x, -m), m), min(max(y, -m), m)
    def feed(self, x, y, dt):
        """Eat from the nearest pellet within reach. Returns True while eating (a pellet lasts EAT_SECONDS)."""
        best = None
        for f in self.foods:
            d = math.hypot(f[0] - x, f[1] - y)
            if d < EAT_RANGE and (best is None or d < best[0]):
                best = (d, f)
        if best is None:
            return False
        f = best[1]
        f[2] -= dt / EAT_SECONDS
        if f[2] <= 0:
            self.foods = [g for g in self.foods if g is not f]
            self.eaten += 1
        self.version += 1                                  # the pellet shrinks: redraw it
        return True
    def sense(self, x, y, h):
        """What the fly 'perceives': food (strength 0..1, bearing rad, + = to the left) and obstacle strengths left / right / ahead."""
        out = {"food": None, "obst_l": 0.0, "obst_r": 0.0, "front": 0.0}
        best = None
        for fx, fy, _amt in self.foods:
            d = math.hypot(fx - x, fy - y)
            if d < self.smell_range and (best is None or d < best[0]):
                best = (d, wrap_pi(math.atan2(fy - y, fx - x) - h))
        if best:
            out["food"] = (1.0 - best[0] / self.smell_range, best[1])
        hf = self.half
        pts = [(ox, oy, r) for ox, oy, r in self.obstacles] + [(hf, y, 0.0), (-hf, y, 0.0), (x, hf, 0.0), (x, -hf, 0.0)]
        for ox, oy, r in pts:
            dx, dy = ox - x, oy - y
            edge = math.hypot(dx, dy) - r
            if edge > AVOID_RANGE:
                continue
            th = wrap_pi(math.atan2(dy, dx) - h)
            if abs(th) > 1.3:
                continue
            st = (1.0 - max(edge, 0.0) / AVOID_RANGE) * (1.0 - 0.5 * abs(th) / 1.3)
            if th >= 0:
                out["obst_l"] = max(out["obst_l"], st)
            else:
                out["obst_r"] = max(out["obst_r"], st)
            if abs(th) < 0.45:
                out["front"] = max(out["front"], st)
        return out
class Plume:
    """Odour plume as a cloud of drifting puffs (a filament / particle model of turbulent advection-diffusion).
    Every food pellet releases puffs; the wind carries them downwind, slow eddies make the stream meander and billow,
    and each puff spreads out as it ages (diffusion) while its odour gets thinner. 2-D, mm / s."""
    MAXN, LIFE, RATE, S0, DIFF = 9000, 15.0, 90.0, 0.4, 0.3
    def __init__(self, half):
        self.half = half
        self.rng = np.random.default_rng(3)
        self.clear()
        self.theta = 0.0                                   # wind direction (rad, direction the wind blows TOWARD)
        self.t = 0.0
        self._emit = 0.0
        self.phase = self.rng.uniform(0, 2 * math.pi, 3)
        self.wind_dir = 0.0
    def clear(self):
        self.pos = np.zeros((0, 2))
        self.vt = np.zeros((0, 2))                         # per-puff turbulent velocity
        self.age = np.zeros(0)
    def update(self, dt, foods, speed, direction_deg, turb=0.8, obstacles=()):
        dt = min(dt, 0.1)
        self.t += dt
        base = math.radians(direction_deg)
        self.theta += (0.0 - self.theta) * dt / 8.0 + math.sqrt(2 * dt / 8.0) * 0.3 * turb * float(self.rng.normal())
        self.theta = float(np.clip(self.theta, -0.7, 0.7))
        th = base + self.theta
        self.wind_dir = th
        wx, wy = math.cos(th), math.sin(th)
        nx, ny = -wy, wx
        if len(self.age):
            x, y = self.pos[:, 0], self.pos[:, 1]
            sa = x * wx + y * wy                           # coordinate along the wind
            u_perp = np.zeros_like(sa)
            for k, (wl, om, amp) in enumerate(((26.0, 0.9, 0.9), (13.0, 1.9, 0.55), (7.0, 3.1, 0.25))):
                u_perp += amp * np.sin(2 * math.pi * sa / wl - om * self.t + self.phase[k])      # eddies: the stream snakes and billows
            u_perp *= speed * turb * (0.4 + 0.6 * np.minimum(1.0, self.age / 3.0))
            tau = 0.7
            self.vt += -self.vt * dt / tau + math.sqrt(2 * dt / tau) * 0.12 * speed * turb * self.rng.normal(size=self.vt.shape)
            dvx = np.zeros_like(sa)
            dvy = np.zeros_like(sa)
            Ux, Uy = wx * speed, wy * speed
            for ox, oy, a in obstacles:                    # wind flows AROUND each obstacle (potential flow past a cylinder)
                rx, ry = x - ox, y - oy
                rho2 = np.maximum(rx * rx + ry * ry, a * a)
                near = rho2 < (6.0 * a) ** 2
                if not near.any():
                    continue
                k = a * a / rho2
                dot = (Ux * rx + Uy * ry) / rho2             # U . rho_hat / rho
                dvx += np.where(near, k * (Ux - 2.0 * dot * rx), 0.0)
                dvy += np.where(near, k * (Uy - 2.0 * dot * ry), 0.0)
            self.pos[:, 0] += (wx * speed + nx * u_perp + self.vt[:, 0] + dvx) * dt
            self.pos[:, 1] += (wy * speed + ny * u_perp + self.vt[:, 1] + dvy) * dt
            for ox, oy, a in obstacles:                    # smoke cannot enter an obstacle: slide it back out to the surface
                rx, ry = self.pos[:, 0] - ox, self.pos[:, 1] - oy
                rho = np.hypot(rx, ry)
                inside = rho < a * 1.03
                if inside.any():
                    rho_i = np.maximum(rho[inside], 1e-6)
                    self.pos[inside, 0] = ox + rx[inside] / rho_i * a * 1.03
                    self.pos[inside, 1] = oy + ry[inside] / rho_i * a * 1.03
                    nxr, nyr = rx[inside] / rho_i, ry[inside] / rho_i
                    vin = np.minimum(self.vt[inside, 0] * nxr + self.vt[inside, 1] * nyr, 0.0)
                    self.vt[inside, 0] -= vin * nxr
                    self.vt[inside, 1] -= vin * nyr
            self.age += dt
            h = self.half + 6.0
            keep = (self.age < self.LIFE) & (np.abs(self.pos[:, 0]) < h) & (np.abs(self.pos[:, 1]) < h)
            if not keep.all():
                self.pos, self.vt, self.age = self.pos[keep], self.vt[keep], self.age[keep]
        self._emit += dt * self.RATE
        n_new = int(self._emit)
        self._emit -= n_new
        if foods and n_new:
            for fx, fy, amt in foods:
                if len(self.age) >= self.MAXN:
                    break
                k = max(1, int(n_new * min(1.0, 0.3 + amt)))
                p = np.array([fx, fy]) + self.rng.normal(scale=0.15, size=(k, 2))
                self.pos = np.vstack([self.pos, p])
                self.vt = np.vstack([self.vt, np.zeros((k, 2))])
                self.age = np.concatenate([self.age, np.zeros(k)])
    def burst(self, x, y, n=160):
        """A sudden puff of odour at (x, y) (the 'Puff ahead' trigger)."""
        p = np.array([x, y]) + self.rng.normal(scale=0.6, size=(n, 2))
        self.pos = np.vstack([self.pos, p])
        self.vt = np.vstack([self.vt, np.zeros((n, 2))])
        self.age = np.concatenate([self.age, np.zeros(n)])
    def sigma(self):
        return np.sqrt(self.S0 ** 2 + 2.0 * self.DIFF * self.age)
    def amp(self):
        sg = self.sigma()
        return (self.S0 / sg) ** 2 * np.exp(-self.age / 14.0)
    def conc(self, pts):
        """Odour concentration at each (x, y) in pts (N, 2)."""
        pts = np.asarray(pts, float)
        if not len(self.age):
            return np.zeros(len(pts))
        sg = self.sigma()
        d2 = ((pts[:, None, :] - self.pos[None, :, :]) ** 2).sum(-1)
        return (self.amp()[None, :] * np.exp(-d2 / (2 * sg[None, :] ** 2))).sum(1) * 0.18
    def density(self, G=150):
        """Smoke picture for the Build map: (G, G) float 0..1, row 0 = top (+y)."""
        out = np.zeros((G, G), np.float32)
        if not len(self.age):
            return out
        h = self.half
        w = 0.32 * np.exp(-self.age / 18.0)
        for lo, hi, sg, gain in ((0.0, 4.0, 1.3, 1.0), (4.0, 9.0, 1.9, 1.4), (9.0, 99.0, 2.8, 2.0)):
            m = (self.age >= lo) & (self.age < hi)
            if not m.any():
                continue
            H, _, _ = np.histogram2d(self.pos[m, 1], self.pos[m, 0], bins=G, range=[[-h, h], [-h, h]], weights=w[m])
            out += cv2.GaussianBlur(H[::-1].astype(np.float32), (0, 0), sg) * gain
        return (1.0 - np.exp(-1.3 * out)).astype(np.float32)
class OdorTracker:
    """Plume tracking like a real fly: surge upwind while the odour is there, cast crosswind (wider and wider) when it is lost,
    and walk straight at the food once it is close enough to see. Returns a (strength, bearing) pair like Arena.sense()['food']."""
    THR = 0.04
    def __init__(self):
        self.c = 0.0
        self.since = 99.0
        self.cast = 1.0
        self.cast_t = 0.0
        self.cast_T = 1.0
        self.last_th = 0.0
    def update(self, dt, c_l, c_r, wind_dir, heading, near):
        c = 0.5 * (c_l + c_r)
        self.c += (c - self.c) * min(1.0, dt / 0.15)
        hit = self.c > self.THR
        if hit:
            self.since, self.cast_T, self.cast_t = 0.0, 1.0, 0.0
        else:
            self.since += dt
        upwind = wind_dir + math.pi
        if near is not None:
            return (1.0, near)
        if hit:
            grad = (c_l - c_r) / (c_l + c_r + 1e-6)         # + = stronger on the left antenna
            self.last_th = wrap_pi(upwind - heading) + 0.9 * grad
            return (min(1.0, 0.5 + 4.0 * self.c), self.last_th)
        if self.since < 0.4:
            return (0.6, self.last_th)
        if self.since < 9.0:                                # lost the plume: cast crosswind, wider each time
            self.cast_t += dt
            if self.cast_t > self.cast_T:
                self.cast, self.cast_t, self.cast_T = -self.cast, 0.0, min(3.5, self.cast_T * 1.4)
            return (0.6, wrap_pi(upwind + self.cast * 1.2 - heading))
        return None
class Explorer:
    """Slowly wandering command signals (Ornstein-Uhlenbeck noise): how fast to walk and which way to veer."""
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.u = self.v = 0.0
    def update(self, dt):
        self.u += -self.u * dt / 1.6 + math.sqrt(2.0 * dt / 1.6) * self.rng.normal()
        self.v += -self.v * dt / 5.0 + math.sqrt(2.0 * dt / 5.0) * self.rng.normal()
        return float(np.clip(0.6 + 0.6 * self.v, 0.0, 1.0)), float(math.tanh(self.u))
def env_commands(sense, wander, explore=True):
    """Combine wandering with what the fly senses -> (fwd, back, left, right) command levels 0..1 for the input neurons."""
    lvl, turn = wander if explore else (0.0, 0.0)
    back = 0.0
    food = sense["food"]
    if food is not None:
        st, th = food
        mix = min(1.0, st * 2.5)
        lvl = (1 - mix) * lvl + mix * 0.9 * max(0.0, 1.0 - abs(th) / 0.45)      # turn on the spot until the food is ahead, then walk to it
        turn = (1 - mix) * turn + mix * float(np.clip(th / 0.7, -1.0, 1.0))
    ol, orr, front = sense["obst_l"], sense["obst_r"], sense["front"]
    av = max(ol, orr)
    if av > 0:
        away = float(np.clip(orr - ol, -1.0, 1.0))
        if front > 0.35 and abs(away) < 0.25:
            away = 1.0 if ol >= orr else -1.0               # dead ahead: pick a side
        turn = float(np.clip(turn * (1 - av) + away * (0.4 + 0.9 * av), -1.0, 1.0))
        lvl *= 1.0 - 0.7 * front
        if front > 0.75:
            back, lvl = 1.0, 0.0
    return lvl, back, max(0.0, turn), max(0.0, -turn)
# =====================================================================
# PART 2 - MALECNS DATA BUILDER (used by the Setup tab)
# =====================================================================
# --- Which DN types drive which group. Edit if neuPrint reports different names. -------------
# Cell types follow the BadApple-fly evidence catalogue (published phenotypes, MaleCNS names).
FWD_TYPES = ["DNg100", "DNg97", "DNp09"]   # forward walking (DNg97 = oDN1)
BACK_TYPES = ["MDN"]                       # "moonwalker" backward walking
# Steering: DNa02 on one side shortens that side's strides (turn toward it); DNg13 projects across, so its
# soma side is the OPPOSITE of the turn it drives.
LEFT_SPEC = (("DNa02", "L"), ("DNg13", "R"))
RIGHT_SPEC = (("DNa02", "R"), ("DNg13", "L"))
# (id, label, cell types, whole family?, body animation: 'walk' = changes the walking, 'groom' = front-leg grooming, 'none')
PRESETS = [
    ("forward", "Forward walking", ["DNg100", "DNg97"], False, "walk"),
    ("backward", "Backward walking", ["MDN"], False, "walk"),
    ("turn_inside", "Steering - inner strides", ["DNa02"], False, "walk"),
    ("turn_outside", "Steering - outer strides", ["DNg13"], False, "walk"),
    ("context_walk", "Context-dependent walk / halt", ["DNp09"], False, "walk"),
    ("antenna", "Antennal grooming", ["DNg62", "DNge078"], False, "groom"),
    ("rub", "Front-leg rubbing", ["DNg11"], False, "groom"),
    ("anterior", "Anterior grooming", ["DNg12"], True, "groom"),
    ("feeding", "Proboscis extension", ["MN9"], False, "feed"),
    ("song", "Courtship wing movement", ["pIP10"], False, "none"),
    ("flight", "Flight wing amplitude", ["DNg02"], True, "none"),
    ("landing", "Landing leg extension", ["DNp07", "DNp10"], False, "none"),
    ("escape", "Escape leg extension", ["DNp01"], False, "none"),
]
INHIBITORY_NT = ("gaba", "glu")      # predicted transmitters treated as inhibitory (substring match)
# Template alignment (navis transforms tutorial): neuPrint male-CNS coordinates -> JRC2018M template.
# "JRCFIB2022Mraw" = 8 nm voxels of the male CNS FIB-SEM volume as registered in navis-flybrains.
ALIGN_SOURCE, ALIGN_TARGET = "JRCFIB2022Mraw", "JRC2018M"
def detect_columns(names):
    """Find the pre-synaptic id, post-synaptic id and weight columns in the feather file."""
    low = {n: n.lower() for n in names}
    if "body_pre" in names and "body_post" in names:
        w = next((n for n in ("weight", "count", "syn_count", "synapse_count") if n in names), None)
        if w:
            return "body_pre", "body_post", w
    pre = [n for n in names if "pre" in low[n]]
    post = [n for n in names if "post" in low[n]]
    wt = [n for n in names if any(k in low[n] for k in ("weight", "count", "syn")) and "conf" not in low[n]]
    wt = [n for n in wt if n not in pre and n not in post] or [n for n in wt]
    if len(pre) < 1 or len(post) < 1 or len(wt) < 1:
        return None
    return pre[0], post[0], wt[0]
def upstream_expand(tbl, cols, seed_ids, n_target, log=print):
    """Greedy upstream growth: repeatedly add the neurons that send the most synapses into the current set."""
    import pyarrow as pa
    import pyarrow.compute as pc
    pre_c, post_c, w_c = cols
    cur = list(dict.fromkeys(int(i) for i in seed_ids))
    have = set(cur)
    for it in range(20):
        if len(cur) >= n_target:
            break
        value_set = pa.array(cur, type=tbl[post_c].type)
        sub = tbl.filter(pc.is_in(tbl[post_c], value_set=value_set))
        agg = sub.group_by(pre_c).aggregate([(w_c, "sum")]).to_pandas()
        agg = agg[~agg[pre_c].isin(have)].sort_values(f"{w_c}_sum", ascending=False)
        take = min(n_target - len(cur), max(300, n_target // 6))
        new = [int(i) for i in agg[pre_c].head(take)]
        if not new:
            break
        cur.extend(new)
        have.update(new)
        log(f"  hop {it + 1}: {len(cur)} neurons")
    return np.array(cur, dtype=np.int64)
def build_matrix(tbl, cols, ids):
    """Sparse W[post, pre] (positive synapse counts) restricted to the chosen neurons."""
    import pyarrow as pa
    import pyarrow.compute as pc
    pre_c, post_c, w_c = cols
    vs_pre = pa.array(ids, type=tbl[pre_c].type)
    vs_post = pa.array(ids, type=tbl[post_c].type)
    mask = pc.and_(pc.is_in(tbl[pre_c], value_set=vs_pre), pc.is_in(tbl[post_c], value_set=vs_post))
    df = tbl.select([pre_c, post_c, w_c]).filter(mask).to_pandas()
    index = {int(b): i for i, b in enumerate(ids)}
    r = df[post_c].map(index).to_numpy()
    c = df[pre_c].map(index).to_numpy()
    w = df[w_c].to_numpy().astype(np.float32)
    return sp.csc_matrix((w, (r, c)), shape=(len(ids), len(ids)))
def fill_missing_positions(pos, W):
    """Neurons without a soma position get the mean position of their connected neighbours."""
    A = (abs(W) + abs(W).T).tocsr()
    A.data[:] = 1.0
    for _ in range(4):
        known = ~np.isnan(pos[:, 0])
        if known.all():
            break
        p0 = np.where(known[:, None], pos, 0.0)
        num = A @ p0
        den = A @ known.astype(np.float32)
        ok = (~known) & (den > 0)
        pos[ok] = num[ok] / den[ok, None]
    still = np.isnan(pos[:, 0])
    if still.any():
        pos[still] = np.nanmean(pos[~still], axis=0) + np.random.default_rng(0).normal(scale=1000, size=(still.sum(), 3))
    return pos
def get_positions(df):
    pos = np.full((len(df), 3), np.nan, np.float32)
    for col in ("somaLocation", "rootLocation", "tosomaLocation"):
        if col not in df.columns:
            continue
        for i, v in enumerate(df[col].tolist()):
            if np.isnan(pos[i, 0]) and isinstance(v, (list, tuple, np.ndarray)) and len(v) == 3:
                pos[i] = np.asarray(v, np.float32)
    return pos
def align_positions(pos, log=print, source=ALIGN_SOURCE, target=ALIGN_TARGET):
    """Move neuron positions into the standard JRC2018M template with navis (navis transforms tutorial).
    Never fatal: if navis/flybrains or the template transforms are missing, the raw positions are kept."""
    try:
        import navis
        import flybrains
        flybrains.register_transforms()
    except ImportError as err:
        log(f"  Alignment skipped - missing package ({err}). Run: python -m pip install navis flybrains h5py")
        return pos
    ok = ~np.isnan(pos[:, 0])
    if not ok.any():
        return pos
    try:
        xf = np.asarray(navis.xform_brain(pos[ok].astype(np.float64), source=source, target=target), dtype=np.float32)
    except Exception as err:
        log(f"  Alignment skipped - {err}")
        log("  Press 'Download template transforms' once, then build again.")
        return pos
    good = np.isfinite(xf).all(axis=1)
    log(f"  aligned {int(good.sum())} of {int(ok.sum())} positions to {target}")
    if good.sum() < 0.5 * ok.sum():
        log("  Too few positions landed inside the template (many neurons sit outside the brain) - keeping raw coordinates.")
        return pos
    out = pos.copy()
    idx = np.flatnonzero(ok)
    out[idx] = np.nan                    # positions that failed are filled in from their neighbours later
    out[idx[good]] = xf[good]
    return out
def fetch_skeleton_lines(groups, per_total, types, log=print):
    """Skeletons (navis.interfaces.neuprint.fetch_skeletons) of a few DNs per group plus any extra types.
    Returns (points, edges, edge_group): group 0-3 = fwd/back/left/right, 4 = extra types."""
    import navis
    import navis.interfaces.neuprint as neu
    from neuprint import NeuronCriteria as NC
    jobs = []
    if per_total > 0:
        per = max(1, per_total // 4)
        for gi, g in enumerate(GROUPS):
            ids = [int(b) for b in list(groups.get(g, []))[:per]]
            if ids:
                jobs.append((NC(bodyId=ids), gi))
    for t in [t.strip() for t in str(types).split(",") if t.strip()]:
        jobs.append((NC(type=t), 4))
    pos_l, edge_l, grp_l, off = [], [], [], 0
    for crit, gi in jobs:
        try:
            nl = neu.fetch_skeletons(crit)
        except Exception as err:
            log(f"  skeletons skipped for group {gi}: {err}")
            continue
        for sk in nl:
            try:
                sk = navis.downsample_neuron(sk, 12, inplace=False)
            except Exception:
                pass
            nd = sk.nodes
            idc = "node_id" if "node_id" in nd.columns else "treenode_id"
            row = {int(i): k for k, i in enumerate(nd[idc].to_numpy())}
            par = nd["parent_id"].to_numpy()
            e = np.array([[k, row[int(p)]] for k, p in enumerate(par) if int(p) in row], np.int64).reshape(-1, 2)
            pos_l.append(nd[["x", "y", "z"]].to_numpy(np.float32))
            edge_l.append(e + off)
            grp_l.append(np.full(len(e), gi, np.int8))
            off += len(nd)
        log(f"  group {gi}: {len(nl)} skeletons")
    if not pos_l:
        return np.zeros((0, 3), np.float32), np.zeros((0, 2), np.int64), np.zeros(0, np.int8)
    return np.vstack(pos_l), np.vstack(edge_l), np.concatenate(grp_l)
def fetch_context(fetch_custom, n_target, log=print):
    """Soma positions of a sample of neurons (about every 4th one) - drawn only, to give the view its shape."""
    q = ("MATCH (n:Neuron) WHERE n.somaLocation IS NOT NULL AND n.bodyId % 4 = 0 "
         f"RETURN n.somaLocation AS loc LIMIT {int(n_target)}")
    df = fetch_custom(q)
    locs = []
    for v in df["loc"]:
        if isinstance(v, dict):
            v = v.get("coordinates")
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 3:
            locs.append(v)
    return np.asarray(locs, np.float32).reshape(-1, 3)
def download_transforms(log=print):
    """One-time download of the JRC bridging transforms used by align_positions (HDF5, no CMTK needed)."""
    try:
        import flybrains
    except ImportError as err:
        raise RuntimeError(f"Missing package ({err}). Run: python -m pip install navis flybrains h5py")
    log("Downloading the JRC template transforms (one time, can take a few minutes) ...")
    flybrains.download_jrc_transforms()
    flybrains.register_transforms()
    log("Done. You can now build with 'Align to JRC2018M' ticked.")
def side_of(instance, soma_side):
    m = re.search(r"_([LR])(?:_|$|\W)", str(instance))
    if m:
        return m.group(1)
    s = str(soma_side).strip().lower()
    if s in ("l", "left"):
        return "L"
    if s in ("r", "right"):
        return "R"
    return None
def groups_from_df(df, log=print):
    """df needs bodyId, type, instance, somaSide -> dict group -> list of bodyIds."""
    typ = df["type"].astype(str)
    def ids_of(types):
        return [int(b) for b in df.loc[typ.isin(types), "bodyId"]]
    groups = {"fwd": ids_of(FWD_TYPES), "back": ids_of(BACK_TYPES), "left": [], "right": []}
    for key, spec in (("left", LEFT_SPEC), ("right", RIGHT_SPEC)):
        for t, side in spec:
            for _, row in df[typ == t].iterrows():
                if side_of(row.get("instance", ""), row.get("somaSide", "")) == side:
                    groups[key].append(int(row["bodyId"]))
    for k in ("fwd", "back"):
        log(f"  {k}: {len(groups[k])} neurons")
    log(f"  left: {len(groups['left'])}  right: {len(groups['right'])}")
    return groups
def fetch_groups(NC, fetch_neurons, log=print):
    """Look the DN groups up in neuPrint -> dict group -> list of bodyIds."""
    names = sorted(set(FWD_TYPES + BACK_TYPES + [t for t, _ in LEFT_SPEC + RIGHT_SPEC]))
    df, _ = fetch_neurons(NC(type="(" + "|".join(names) + ")", regex=True))
    return groups_from_df(df, log)
def resolve_presets(ann):
    """{(preset id, 'L'|'R'): [bodyIds]} for the BadApple-fly movement programs found in these annotations."""
    typ = ann["type"].astype(str)
    out = {}
    for pid, _label, types, fam, _kind in PRESETS:
        m = typ.isin(types)
        if fam:
            for t in types:
                m |= typ.str.startswith(t + "_")
        for _, row in ann[m].iterrows():
            side = side_of(row.get("instance", ""), row.get("somaSide", ""))
            if side:
                out.setdefault((pid, side), []).append(int(row["bodyId"]))
    return out
def find_local_tables(weights):
    """Offline mode: annotations + neurotransmitters feather files sitting next to the weights file."""
    d = os.path.dirname(os.path.abspath(weights))
    def pick(*pats):
        for f in sorted(os.listdir(d)):
            if any(p in f.lower() for p in pats) and f.lower().endswith(".feather"):
                return os.path.join(d, f)
        return None
    return pick("annotation"), pick("neurotransmitter")
def load_local_annotations(ann_path, nt_path, log=print):
    import pandas as pd
    import pyarrow.feather as feather
    ann = feather.read_table(ann_path).to_pandas()
    ann = ann.drop_duplicates("bodyId")
    if nt_path:
        nt = feather.read_table(nt_path, columns=["body", "consensus_nt"]).to_pandas().drop_duplicates("body")
        ann = ann.merge(nt.rename(columns={"body": "bodyId", "consensus_nt": "nt"}), on="bodyId", how="left")
    log(f"  local annotations: {len(ann)} neurons" + ("" if nt_path else " (no neurotransmitter file found)"))
    return ann
def groups_local(ann, log=print):
    return groups_from_df(ann, log)
def region_of(superclass):
    """0 = central brain, 1 = optic lobes, 2 = nerve cord, from the MaleCNS superclass annotation."""
    sc = np.asarray(superclass).astype(str)
    lab = np.zeros(len(sc), np.int8)
    ol = np.char.startswith(sc, "ol_") | np.char.startswith(sc, "visual")
    vnc = np.char.startswith(sc, "vnc") | np.char.startswith(sc, "sensory_ascending") | np.char.startswith(sc, "ascending") \
        | np.char.startswith(sc, "efferent_ascending")
    lab[ol] = 1
    lab[vnc] = 2
    return lab
def context_local(ann, n_target):
    """Soma positions (and region labels) of a random sample of all annotated neurons: drawn only, gives the view its shape."""
    pos = get_positions(ann)
    lab = region_of(ann["superclass"]) if "superclass" in ann.columns else np.zeros(len(pos), np.int8)
    ok = np.isfinite(pos).all(axis=1)
    pos, lab = pos[ok], lab[ok]
    if len(pos) > n_target:
        k = np.random.default_rng(0).choice(len(pos), n_target, replace=False)
        pos, lab = pos[k], lab[k]
    return pos.astype(np.float32), lab
BRAIN_VERSION = 2          # bump when the layout of brain_subset.npz changes -> the app rebuilds an older file by itself
def brain_is_current(path):
    try:
        with np.load(path, allow_pickle=False) as d:
            return "build_version" in d.files and int(d["build_version"]) >= BRAIN_VERSION
    except Exception:
        return False
def find_compiled(weights):
    """Folder with compiled MaleCNS arrays (indptr/indices/weights/signs/ids .npy + neurons.feather), if present."""
    base = os.path.dirname(os.path.abspath(weights))
    for d in (os.path.join(os.path.dirname(base), "compiled"), os.path.join(base, "compiled"), base):
        if all(os.path.exists(os.path.join(d, f)) for f in ("indptr.npy", "indices.npy", "weights.npy", "ids.npy", "neurons.feather")):
            return d
    return None
def load_compiled(cdir, log=print):
    """Whole connectome as W[post, pre] (CSR, synapse counts) with per-neuron sign (+1 / -1 / 0), plus annotations."""
    import pandas as pd
    import pyarrow.feather as feather
    ids = np.load(os.path.join(cdir, "ids.npy"))
    M = sp.csr_matrix((np.load(os.path.join(cdir, "weights.npy")), np.load(os.path.join(cdir, "indices.npy")),
                       np.load(os.path.join(cdir, "indptr.npy"))), shape=(len(ids), len(ids)))   # rows = pre, cols = post
    sign = np.load(os.path.join(cdir, "signs.npy")).astype(np.float32)
    ann = feather.read_table(os.path.join(cdir, "neurons.feather")).to_pandas().drop_duplicates("bodyId")
    ann = ann.set_index("bodyId").reindex(ids).reset_index()
    log(f"  compiled MaleCNS: {len(ids)} neurons, {M.nnz} connections")
    return ids, M.T.tocsr(), sign, ann
def upstream_expand_csr(Wr, seed_rows, n_target, log=print):
    """Greedy upstream growth on W[post, pre]: add the neurons sending the most synapses into the current set."""
    cur = list(dict.fromkeys(int(i) for i in seed_rows))
    have = np.zeros(Wr.shape[0], bool)
    have[cur] = True
    for it in range(30):
        if len(cur) >= n_target:
            break
        tot = np.asarray(Wr[cur].sum(axis=0)).ravel()
        tot[have] = 0
        take = min(n_target - len(cur), max(300, n_target // 6))
        new = np.argsort(tot)[::-1][:take]
        new = new[tot[new] > 0]
        if not len(new):
            break
        cur.extend(int(i) for i in new)
        have[new] = True
        log(f"  hop {it + 1}: {len(cur)} neurons")
    return np.array(cur, np.int64)
def build_subset(weights, token, n, out, log=print, list_dn=False, align=False, context=0, shapes=0, shape_types=""):
    """Build brain_subset.npz: ~n neurons that feed the walking/turning DNs. Returns the output path."""
    try:
        import pandas as pd
        import pyarrow.feather as feather
    except ImportError as err:
        raise RuntimeError(f"Missing package ({err}). Run:  pip install pyarrow pandas")
    cdir = find_compiled(weights) if os.path.exists(weights) or os.path.isdir(os.path.dirname(os.path.abspath(weights))) else None
    ann_path, nt_path = find_local_tables(weights) if os.path.exists(weights) else (None, None)
    use_np = bool(token)
    if use_np:
        try:
            from neuprint import Client, fetch_neurons, fetch_custom, NeuronCriteria as NC
        except ImportError as err:
            raise RuntimeError(f"Missing package ({err}). Run:  pip install neuprint-python")
        log("Connecting to neuPrint (male-cns:v1.0) ...")
        Client("https://neuprint.janelia.org", dataset="male-cns:v1.0", token=token)
    if cdir:
        log(f"Using your compiled MaleCNS arrays in {cdir} (fast, no token needed).")
        ids_all, Wr, sign_all, local = load_compiled(cdir, log)
    elif ann_path:
        log("Using the local annotation files next to the weights file (no neuPrint token needed).")
        local = load_local_annotations(ann_path, nt_path, log)
    elif not use_np:
        raise RuntimeError("Found neither compiled arrays nor annotation files next to the weights file; paste a neuPrint token.")
    else:
        local = None
    if list_dn:
        if local is not None:
            dn = local[local["superclass"].astype(str).str.startswith("descending")]
            log("DN types found (count per type):")
            log(dn["type"].value_counts().head(60).to_string())
        else:
            df, _ = fetch_neurons(NC(type="DN.*", regex=True))
            log(df["type"].value_counts().head(60).to_string())
        return None
    log("Finding the walking/turning DNs ...")
    groups = groups_local(local, log) if local is not None else fetch_groups(NC, fetch_neurons, log)
    empty = [g for g, v in groups.items() if not v]
    if empty:
        raise RuntimeError(f"No neurons found for {empty}. Press 'List DN type names' to see the real names, "
                           "then edit FWD_TYPES / BACK_TYPES / LEFT_SPEC / RIGHT_SPEC near the top of this file.")
    seeds = [b for g in ("fwd", "back", "left", "right") for b in groups[g]]
    presets = resolve_presets(local) if local is not None else {}
    preset_bodies = sorted({b for v in presets.values() for b in v})
    log(f"  movement-program presets: {len(presets)} groups, {len(preset_bodies)} neurons")
    log(f"Growing upstream from {len(seeds)} DNs to {n} neurons ...")
    if cdir:
        row_of = pd.Series(np.arange(len(ids_all)), index=ids_all)
        rows = upstream_expand_csr(Wr, row_of.loc[[b for b in seeds if b in row_of.index]].to_numpy(), n, log)
        extra_rows = row_of.reindex(preset_bodies).dropna().astype(np.int64).to_numpy()
        rows = np.concatenate([rows, np.setdiff1d(extra_rows, rows)]).astype(np.int64)
        ids = ids_all[rows]
        W = Wr[rows][:, rows].tocsc()
        sign = sign_all[rows]
        ann = local.iloc[rows].reset_index(drop=True)
        log(f"  {W.shape[0]} neurons, {W.nnz} connections")
    else:
        log("Reading the weights file (can take a minute) ...")
        tbl = feather.read_table(weights, memory_map=True)
        log(f"  columns: {tbl.column_names}   rows: {tbl.num_rows}")
        cols = detect_columns(tbl.column_names)
        if cols is None:
            raise RuntimeError("Couldn't tell which columns are pre / post / weight. Send me the column list above.")
        log(f"  using pre/post/weight = {cols}")
        ids = upstream_expand(tbl, cols, seeds, n, log=log)
        ids = np.concatenate([ids, np.setdiff1d(np.array(preset_bodies, np.int64), ids)]).astype(np.int64)
        log("Building the synapse matrix ...")
        W = build_matrix(tbl, cols, ids)
        log(f"  {W.shape[0]} neurons, {W.nnz} connections")
        log("Fetching neuron annotations (positions, transmitters) ...")
        if local is not None:
            ann = local.set_index("bodyId").reindex(ids).reset_index()
        else:
            frames = []
            for i in range(0, len(ids), 1500):
                df, _ = fetch_neurons(NC(bodyId=[int(b) for b in ids[i:i + 1500]]))
                frames.append(df)
            ann = pd.concat(frames).drop_duplicates("bodyId").set_index("bodyId").reindex(ids).reset_index()
        sign = None
    pos = get_positions(ann)
    ctx = np.zeros((0, 3), np.float32)
    ctx_lab = None
    if context > 0:
        log(f"Collecting about {context} context neurons (shape of the whole CNS) ...")
        try:
            if local is not None:
                ctx, ctx_lab = context_local(local, context)
            else:
                ctx = fetch_context(fetch_custom, context, log)
            log(f"  got {len(ctx)} context positions")
        except Exception as err:
            log(f"  context cloud skipped - {err}")
    sk_pos, sk_edges, sk_grp = np.zeros((0, 3), np.float32), np.zeros((0, 2), np.int64), np.zeros(0, np.int8)
    if (shapes > 0 or str(shape_types).strip()) and not use_np:
        log("Skeleton shapes need a neuPrint token - skipped (the density view still works).")
    elif shapes > 0 or str(shape_types).strip():
        log("Fetching neuron skeletons (shapes) ...")
        try:
            sk_pos, sk_edges, sk_grp = fetch_skeleton_lines(groups, shapes, shape_types, log)
            log(f"  {len(sk_pos)} skeleton points, {len(sk_edges)} segments")
        except Exception as err:
            log(f"  shapes skipped - {err}")
    if align:
        log(f"Aligning positions to the {ALIGN_TARGET} template with navis (brain only) ...")
        if len(sk_pos):
            sk_pos = align_positions(sk_pos, log)
            good = np.isfinite(sk_pos).all(axis=1)
            keep = good[sk_edges[:, 0]] & good[sk_edges[:, 1]]
            sk_edges, sk_grp = sk_edges[keep], sk_grp[keep]
        pos = align_positions(pos, log)
        if len(ctx):
            ctx = align_positions(ctx, log)
            good_c = np.isfinite(ctx).all(axis=1)
            ctx = ctx[good_c]
            if ctx_lab is not None:
                ctx_lab = ctx_lab[good_c]
    pos = fill_missing_positions(pos, W)
    if sign is None:
        nt_cols = [c for c in ann.columns if c.lower() in ("predictednt", "consensusnt", "nt", "predicted_nt")
                   or "neurotransmitter" in c.lower()]
        sign = np.ones(len(ids), np.float32)
        if nt_cols:
            nt = ann[nt_cols[0]].astype(str).str.lower()
            sign[nt.str.contains("|".join(INHIBITORY_NT)).to_numpy()] = -1.0
            log(f"  transmitter column '{nt_cols[0]}': {int((sign < 0).sum())} inhibitory neurons")
        else:
            log("  WARNING: no transmitter column found - all synapses treated as excitatory.")
    else:
        log(f"  transmitter signs from the compiled data: {int((sign < 0).sum())} inhibitory, {int((sign > 0).sum())} excitatory, "
            f"{int((sign == 0).sum())} with no fast synapses")
    # Descending neurons are OUTPUT units (their axons run down to the nerve cord): cut their outgoing links inside
    # the brain, otherwise they feed back into the network and inhibit each other and never behave as a readout.
    out_only = np.zeros(len(ids), bool)
    if "superclass" in ann.columns:
        out_only |= ann["superclass"].astype(str).str.startswith("descending").to_numpy()
    index0 = {int(b): i for i, b in enumerate(ids)}
    for g_ in groups:
        for b in groups[g_]:
            if b in index0:
                out_only[index0[b]] = True
    log(f"  {int(out_only.sum())} descending neurons treated as output-only")
    W = (W @ sp.diags(sign * (~out_only))).tocsc()
    W.eliminate_zeros()
    extra = {}
    if "superclass" in ann.columns and (ctx_lab is not None or len(ctx) == 0):
        extra["labels"] = np.concatenate([region_of(ann["superclass"]), ctx_lab if ctx_lab is not None else np.zeros(0, np.int8)])
    index = {int(b): i for i, b in enumerate(ids)}
    for (pid, side), bodies in presets.items():
        extra[f"pre_{pid}_{side}"] = np.array([index[b] for b in bodies if b in index], np.int64)
    grp = {f"grp_{g}": np.array([index[b] for b in groups[g] if b in index], np.int64) for g in groups}
    np.savez_compressed(out, data=W.data, indices=W.indices, indptr=W.indptr,
                        shape=np.array(W.shape), pos=pos.astype(np.float32), ctx_pos=ctx.astype(np.float32),
                        skel_pos=sk_pos.astype(np.float32), skel_edges=sk_edges, skel_grp=sk_grp,
                        body_ids=ids, build_version=np.array(BRAIN_VERSION), **extra, **grp)
    log(f"Saved {out}")
    log(str({g.replace('grp_', ''): int(v.size) for g, v in grp.items()}))
    return out
# =====================================================================
# PART 3 - FLYLAB APP (your Fly Bridge + live brain lab)
# =====================================================================
# pydirectinput sleeps 0.1s after EVERY call by default. With 4+ calls per tick that capped the
# whole loop at ~2 frames per second (the slow preview + laggy tracking). Turn it off.
pydirectinput.PAUSE = 0
# --- 1. CONFIGURATIONS & COORDINATES ---
# Default capture box. Use the "Select Fly Area" button to draw a new one; it is saved and reloaded.
DEFAULT_CAPTURE_REGION = {"top": 640, "left": 310, "width": 120, "height": 120}
REGION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flylab_region.json")
BLOCK_X, BLOCK_Y = 1183, 531
SPAWN_MENU_X, SPAWN_MENU_Y = 1005, 520
SPAWN_GROUND_X, SPAWN_GROUND_Y = 1430, 420
# Blooket game modes. Click targets are NOT built in: use "Map buttons..." in the panel to record the answer
# buttons (1-4) and the Continue/Submit button for each mode. Unmapped answers fall back to the number keys,
# and Classic (no Continue mapped) presses Enter.
GAME_MODES = ["Classic", "Gold Quest"]
BLOOKET_MAP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flylab_blooket_map.json")
MAP_STEPS = [("1", "ANSWER 1"), ("2", "ANSWER 2"), ("3", "ANSWER 3"), ("4", "ANSWER 4"), ("continue", "the CONTINUE / SUBMIT button")]
EYE_PX, EYE_FOV, EYE_YAW = 120, 110.0, 35.0                  # eye camera resolution (px), field of view (deg), outward turn of each eye (deg)
MOTOR_W = 860                                             # width of the Motor Properties picture
MINI_W, MINI_H = 980, 320                                  # window size on the "Minimized" page
FOOD_KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flylab_food_key.json")
def load_food_key():
    try:
        with open(FOOD_KEY_FILE) as f:
            return str(json.load(f).get("key", "")) or ""
    except Exception:
        return ""
def save_food_key(key):
    try:
        with open(FOOD_KEY_FILE, "w") as f:
            json.dump({"key": key}, f)
    except Exception as err:
        print(f"[!] Couldn't save the food key: {err}")
_KEYSYM_TO_KEY = {"Return": "enter", "KP_Enter": "enter", "space": "space", "Tab": "tab", "BackSpace": "backspace",
                  "Up": "up", "Down": "down", "Left": "left", "Right": "right", "Shift_L": "shift", "Shift_R": "shift",
                  "Control_L": "ctrl", "Control_R": "ctrl", "Alt_L": "alt", "Alt_R": "alt", "Delete": "delete",
                  "Home": "home", "End": "end", "Prior": "pageup", "Next": "pagedown", "Insert": "insert"}
def keysym_to_key(sym):
    """Tk keysym -> a pydirectinput key name ('' if it can't be used)."""
    if sym in _KEYSYM_TO_KEY:
        return _KEYSYM_TO_KEY[sym]
    if len(sym) == 1:
        return sym.lower()
    if sym[:1] == "F" and sym[1:].isdigit() and 1 <= int(sym[1:]) <= 12:
        return sym.lower()
    return ""
_PULSE_CACHE = {}
_PULSE_ERR = [False]
def play_pulse(hz, ms=110):
    """A short sine pulse (soft edges) played in the background. Safe to call from any thread; does nothing off Windows.
    (winsound can't play from memory asynchronously, so each pitch is written once to a small .wav in the temp folder.)"""
    try:
        import winsound, wave, tempfile
        hz = int(max(20, min(8000, round(float(hz)))))
        path = _PULSE_CACHE.get(hz)
        if path is None or not os.path.isfile(path):
            sr = 44100
            n = int(sr * ms / 1000)
            t = np.arange(n) / sr
            env = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.008)           # 8 ms fade in / out, no clicks
            pcm = (np.sin(2 * np.pi * hz * t) * env * 0.35 * 32767).astype("<i2")
            d = os.path.join(tempfile.gettempdir(), "flylab_pulses")
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, "pulse_%d.wav" % hz)
            with wave.open(path, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
                w.writeframes(pcm.tobytes())
            _PULSE_CACHE[hz] = path
        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
    except Exception as err:
        if not _PULSE_ERR[0]:
            _PULSE_ERR[0] = True
            print("[!] pulse sound:", err)
def load_blooket_map():
    """-> (list of game modes, {mode: {'1': [x, y], ..., 'continue': [x, y]}})"""
    modes, maps = list(GAME_MODES), {}
    try:
        with open(BLOOKET_MAP_FILE) as f:
            d = json.load(f)
        for m in d.get("modes", []):
            if m not in modes:
                modes.append(str(m))
        for m, pts in d.get("maps", {}).items():
            maps[m] = {k: [int(v[0]), int(v[1])] for k, v in pts.items() if k in dict(MAP_STEPS) and len(v) == 2}
    except Exception:
        pass
    return modes, maps
def save_blooket_map(modes, maps):
    try:
        with open(BLOOKET_MAP_FILE, "w") as f:
            json.dump({"modes": modes, "maps": maps}, f, indent=1)
    except Exception as err:
        print(f"[!] Couldn't save the button map: {err}")
# Auto-Replay: when the fly has been still this long, click the play/replay button here.
AUTO_REPLAY_SPOT = (436, 1104)
AUTO_REPLAY_STILL_SECONDS = 1.0
AUTO_REPLAY_RETURN_SPOT = (1655, 750)  # clicked right after replay to put focus back on Roblox
STILL_THRESHOLD = 0.08  # motion below this (same trigger as walking) counts as "not moving"
# --- Brain Lab ---
DEFAULT_WEIGHTS = "C:/Users/Nipun's Lovely PC/badapple-fly/data/raw/connections.feather"
DOWNLOAD_CMD = ("gsutil cp gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/"
                "connectome-weights-male-cns-v1.0-minconf-0.5.feather C:/FlyBrain/")
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flylab_settings.json")
LAYOUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flylab_layout.json")
SUGGEST_EMAIL = ""      # put your e-mail address here: the "Email it" button in Settings > Suggestions opens a message to it
SUGGEST_STATUS = "Available"    # shown under Suggestions: "Available" (green), "Busy" (amber) or "Away" (red)
DISCORD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flylab_discord.json")   # {"token": "<bot token>", "user_id": 123456789012345678}
CONFETTI_SOUND = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "yipeeeeeeeee.mp3"),
                  r"C:\Users\Nipun's Lovely PC\Downloads\Download\yipeeeeeeeee.mp3"]      # first one that exists is played with the confetti
SUGGEST_REPO = "https://github.com/Nipunzzzzpun/flybridge-control"
SETTING_DEFAULTS = {
    # performance
    "fps": "30", "body_hz": "30", "only_open": True, "show_fps": False,
    "draw_body": True, "draw_cloud": True, "draw_graphs": True, "draw_smoke": True,
    # body and physics
    "kp": 50.0, "adh": 3.0, "fric": 2.0, "stepmax": 2.4, "lift": 0.25, "hind": 2.2, "fmax": 5.0, "stance": 0.7, "gaze": 1.0,
    "gait": "Tripod", "grav": "Earth",
    # smell and wind
    "thr": 0.04, "diff": 0.3, "rate": 90.0, "maxn": 9000.0, "turb": 0.8, "wind_mode": "Steady", "smoke_obst": True,
    # build tab
    "show_grid": True, "show_trail": True, "snap": False,
    # display
    "topmost": True, "start_min": True,
    # advanced
    "print_timings": False,
    "shadows": False,
    "profile": "Balanced",
    # theme
    "theme": "Light", "theme_mode": "Default",
    # mapping
    "kw_f": "w", "kw_b": "s", "kw_l": "a", "kw_r": "d", "ka_f": "up", "ka_b": "down", "ka_l": "left", "ka_r": "right", "kb_jump": "space",
    "win_use": False, "win_follow": False, "win_pos": "",
}
for _k, _hz in (("kw_f", 440), ("kw_b", 330), ("kw_l", 262), ("kw_r", 392), ("ka_f", 440), ("ka_b", 330), ("ka_l", 262), ("ka_r", 392), ("kb_jump", 523), ("food", 659)):
    SETTING_DEFAULTS["hz_" + _k] = float(_hz)         # sine pulse pitch for that key
    SETTING_DEFAULTS["pl_" + _k] = False              # play a pulse when that key is pressed
GAITS = {
    "Tripod": {"lf": 0.0, "rm": 0.0, "lh": 0.0, "rf": 0.5, "lm": 0.5, "rh": 0.5},
    "Tetrapod (experimental)": {"lf": 0.0, "rm": 0.0, "lh": 1 / 3, "rf": 1 / 3, "lm": 2 / 3, "rh": 2 / 3},
    "Wave (experimental)": {"rh": 0.0, "rm": 1 / 6, "rf": 2 / 6, "lh": 3 / 6, "lm": 4 / 6, "lf": 5 / 6},
}
GRAVITY = {"Earth": 1.0, "Moon": 0.165, "Zero-g": 0.0}
PROFILES = {
    "Balanced": {"shadows": False, "fps": "30", "body_hz": "30", "only_open": True, "draw_body": True, "draw_cloud": True, "draw_graphs": True, "draw_smoke": True},
    "Performance": {"shadows": False, "fps": "20", "body_hz": "15", "only_open": True, "draw_body": True, "draw_cloud": False, "draw_graphs": False, "draw_smoke": False},
    "Quality": {"shadows": True, "fps": "60", "body_hz": "60", "only_open": True, "draw_body": True, "draw_cloud": True, "draw_graphs": True, "draw_smoke": True},
}
BRAIN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brain_subset.npz")
DYN_LABEL = "Dynamic environment (default)"
BASE_LABEL = "Baseline - no stimulation"
ARENA_PX = 560                                         # Build tab canvas size
VIEW_W, VIEW_H = 380, 320   # size of the fly-body view
NEURON_W, NEURON_H = 470, 320   # size of the neuron view
DRIVE_FULL_HZ = 40.0        # DN group rate (Hz) that counts as "full drive"
BRAIN_DEADZONE = 0.15       # drive below this is ignored when the brain drives the keys
ALL_KEYS = ['w', 'a', 's', 'd', 'up', 'down', 'left', 'right', 'space', 'q', '1', '2', '3', '4', 'enter']
PREVIEW_MAX = 240  # longest side of the preview image in pixels
GRAPH_PW, GRAPH_TH, GRAPH_DH, GRAPH_AH = 360, 215, 275, 250   # Graphs column: panel width, time-graph height, direction-circle height (stacked vertically)
GRAPH_SECONDS = 5.0            # time shown on the two time graphs
MOTION_MIN_HOLD = 0.18         # s: a direction must last at least this long before it can change (stops A-D-A-D flicker)
BLOOKET_CONFIRM = 0.25         # s: the same direction must be steady this long before an answer is chosen
BLOOKET_COOLDOWN = 1.3         # s: minimum gap between two answers
BLOOKET_REPEAT = 3.0           # s: keep heading the same way this long and the same answer is chosen again
# BGR colours for the four descending-neuron groups in the neuron view
GROUP_COLORS = {"fwd": (200, 120, 0), "back": (60, 100, 160), "left": (170, 170, 0), "right": (160, 60, 160)}
def load_region():
    try:
        with open(REGION_FILE) as f:
            r = json.load(f)
        if all(k in r for k in ("top", "left", "width", "height")) and r["width"] > 0 and r["height"] > 0:
            return {k: int(r[k]) for k in ("top", "left", "width", "height")}
    except Exception:
        pass
    return dict(DEFAULT_CAPTURE_REGION)
def to_png(img):
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return buf.tobytes() if ok else None
class NeuroMechFlyBody:
    """The real NeuroMechFly model (FlyGym 2.x meshes) simulated with real physics in MuJoCo.

    Every leg joint is a position-controlled motor, the body is a free-floating rigid body with gravity and
    ground contact, and the feet grip with adhesion pads during stance. The brain's descending-neuron
    drives set a walking command; a tripod gait (inverse kinematics) turns that command into joint targets,
    and the body is only moved by the forces the feet push against the ground, so it can tilt and slip:
        walk speed ~ (fwd - back) * V_MAX            turn rate ~ (right - left) * W_MAX
    If FlyGym / MuJoCo are missing, the panel shows the reason instead of a stand-in drawing.
    """
    LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
    LEG_NAMES = {"lf": "Left front leg", "lm": "Left middle leg", "lh": "Left hind leg",
                 "rf": "Right front leg", "rm": "Right middle leg", "rh": "Right hind leg"}
    TRIPOD = {"lf": 0.0, "rm": 0.0, "lh": 0.0, "rf": 0.5, "lm": 0.5, "rh": 0.5}
    GAZE_STAB, NECK_MAX, NECK_TAU = 1.0, math.radians(30.0), 0.06     # neck reflex gain / limit (rad) / lag (s)
    NECK_SIGN = {"roll": 1.0, "pitch": 1.0}
    _neck_dt = 0.004
    HIND_LIFT = 2.2                                # the heavy hind legs need a higher swing to really clear the ground
    STEPMAX, LIFT, STANCE = 2.4, 0.25, 0.7       # mm of foot travel per step, mm foot lift, fraction of the cycle on the ground
    V_MAX, W_MAX = 22.0, 4.5                     # commanded mm/s and rad/s at full drive (feet slip, so real motion is lower)
    F_MAX, F_MIN = 5.0, 2.5                       # Hz, step frequency limits while walking
    KP, ADHESION = 50.0, 3.0                    # joint motor stiffness, foot adhesion gain
    EFF_V, EFF_W = 0.30, 0.22                    # how much of the commanded walk / turn the body is expected to achieve
    YAW_ASSIST = 0.0                             # 0..1: how strongly a gyro-like yaw-rate reflex trims the turning (0 = legs only)
    TURN_RATE = 0.35                             # achieved body turn rate (rad/s) per rad/s of turn command
    YAW_KP, YAW_KI, YAW_TMAX, YAW_IMAX = 1.5, 6.0, 6.0, 1.0   # turning reflex gains / limits
    BIAS_W = 14.0                                # rad/s, largest turn trim the feedback may add
    K_V, K_W = 0.0, 0.0                          # feedback trim strength (1/s)
    PRESS = 0.0                                  # mm: how far the stance feet push into the ground
    FORCE_MAX = 100.0                            # motor torque limit
    FRICTION = 2.0                               # foot-ground sliding friction
    PHYS_DT = 3e-4                               # s, physics time step
    FEED_ANGLES = np.array([-1.1, -0.9])         # rad: rostrum / haustellum pitch at full proboscis extension

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.error = None
        self.renderer = None
        self.x = self.y = self.heading = 0.0
        self.pitch = self.roll = 0.0
        self.arena = None                                  # Arena: obstacles / food drawn in the 3D view and used for collisions
        self.groom = 0.0                                   # 0..1: raise the front legs into a grooming motion
        self.feed = 0.0                                    # 0..1: extend the proboscis
        self.prob_adr = np.zeros(0, int)
        self._groom_t = 0.0
        self.trail = []
        self.rt_factor = 0.0                               # simulated seconds per wall second
        self.speed_mm_s = 0.0
        self.n_contacts = 0
        self.leg_force = np.zeros(6)                       # summed ground contact force per leg (uN)
        self.contact_draw = []                             # [(pos xyz, force xyz)] for the 3D view
        self._cached, self._t_last, self._render_s = None, 0.0, 0.0
        try:
            self._build()
        except Exception as exc:
            import traceback
            self.error = f"{type(exc).__name__}: {exc}"
            print("[NeuroMechFly] could not start:\n" + traceback.format_exc())

    # ---------------------------------------------------------------- build
    def _check_install(self):
        """Raise a short, specific message if this Python can't run FlyGym 2.x."""
        import importlib.metadata as md
        py = f"{sys.version_info.major}.{sys.version_info.minor}"
        try:
            ver = md.version("flygym")
        except md.PackageNotFoundError:
            raise RuntimeError(f"FlyGym is not installed in this Python ({py}). Run the install lines at the top of the file.")
        if int(ver.split(".")[0]) < 2:
            raise RuntimeError(f"Old FlyGym {ver} is installed (need 2.x). Python here is {py}"
                               + ("; FlyGym 2.x needs 3.12+, so install Python 3.12 and run the file with it." if sys.version_info < (3, 12)
                                  else ". Run: python -m pip uninstall -y flygym, then the git install line."))

    def _build(self):
        self._check_install()
        import mujoco as mj
        from flygym.compose import NeuroMechFly, FlatGroundWorld, KinematicPosePreset, ActuatorType, ContactParams
        from flygym.anatomy import Skeleton, JointPreset, AxisOrder
        from flygym.utils.math import Rotation3D
        self.mj = mj
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fly = NeuroMechFly(name="nmf")
            try:                                           # legs + the head / proboscis joints (so it can extend to feed)
                joints = list(JointPreset.LEGS_ONLY.to_joint_list()) + [
                    j for j in JointPreset.ALL_BIOLOGICAL.to_joint_list()
                    if j.child.name in ("c_head", "c_rostrum", "c_haustellum")]
                skeleton = Skeleton(axis_order=AxisOrder.PITCH_ROLL_YAW, anatomical_joints=joints)
            except Exception:
                skeleton = Skeleton(joint_preset=JointPreset.LEGS_ONLY, axis_order=AxisOrder.PITCH_ROLL_YAW)
            fly.add_joints(skeleton, neutral_pose=KinematicPosePreset.NEUTRAL)
            fly.colorize()
            fly.add_actuators(fly.get_jointdofs_order(), ActuatorType.POSITION, neutral_input=KinematicPosePreset.NEUTRAL,
                              kp=self.KP, forcerange=(-self.FORCE_MAX, self.FORCE_MAX))
            fly.add_leg_adhesion(gain=self.ADHESION)
            fly.add_tracking_camera(name="trackcam")
            world = FlatGroundWorld()
            world.add_fly(fly, [0, 0, 0.7], Rotation3D("quat", [1, 0, 0, 0]),
                          ground_contact_params=ContactParams(sliding_friction=self.FRICTION))
            self.model, self.data = world.compile()
        m, d = self.model, self.data
        m.opt.timestep = self.PHYS_DT
        self.key = mj.mj_name2id(m, mj.mjtObj.mjOBJ_KEY, "neutral")
        bid = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n)
        jid = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)
        # --- actuator tables: which joint each motor drives, which are adhesion pads -----------------------
        self.adh = {}
        pos_a, self.act_q, self.act_v, self.act_info = [], [], [], []
        for i in range(m.nu):
            name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_ACTUATOR, i) or ""
            if name.endswith("-adhesion"):
                self.adh[name.split("/")[1][:2]] = i
                continue
            j = int(m.actuator_trnid[i][0])
            if j < 0:
                continue
            jn = (mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or "").split("/")[-1]       # e.g. lf_coxa-lf_trochanterfemur-pitch
            parent, child, axis = (jn.split("-") + ["", "", ""])[:3]
            leg = child[:2] if child[:2] in self.LEGS and child[2:3] == "_" else "head"
            part = child.split("_", 1)[1] if "_" in child else child
            pos_a.append(i)
            self.act_q.append(int(m.jnt_qposadr[j]))
            self.act_v.append(int(m.jnt_dofadr[j]))
            self.act_info.append((leg, f"{part} {axis}"))
        self.pos_act = np.array(pos_a, int)
        self.act_q, self.act_v = np.array(self.act_q, int), np.array(self.act_v, int)
        # body id -> leg index (for ground-contact forces)
        self.body_leg = np.full(m.nbody, -1, int)
        for b in range(m.nbody):
            nm = (mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b) or "").split("/")[-1]
            if nm[:2] in self.LEGS and nm[2:3] == "_":
                self.body_leg[b] = self.LEGS.index(nm[:2])
        # --- let it drop onto the ground and settle, then remember the neutral joint angles ------------------
        mj.mj_resetDataKeyframe(m, d, self.key)
        self.ctrl_neutral = d.ctrl.copy()                  # the keyframe's motor targets = the neutral standing pose
        for i in self.adh.values():
            d.ctrl[i] = 1.0
        for _ in range(int(0.3 / self.PHYS_DT)):
            mj.mj_step(m, d)
        self.root_z = float(d.qpos[2])
        self.q_neutral = d.qpos.copy()
        self.q_neutral[:7] = [0, 0, self.root_z, 1, 0, 0, 0]
        self.ik = mj.MjData(m)                       # scratch copy used to solve the legs with the body held still
        self.ik.qpos[:] = self.q_neutral
        mj.mj_forward(m, self.ik)
        self.tip_body, self.tip0, self.qadr, self.dadr = {}, {}, {}, {}
        for leg in self.LEGS:
            names = [f"nmf/c_thorax-{leg}_coxa-pitch", f"nmf/c_thorax-{leg}_coxa-roll", f"nmf/c_thorax-{leg}_coxa-yaw",
                     f"nmf/{leg}_coxa-{leg}_trochanterfemur-pitch", f"nmf/{leg}_coxa-{leg}_trochanterfemur-roll",
                     f"nmf/{leg}_trochanterfemur-{leg}_tibia-pitch"]
            ids = [jid(n) for n in names]
            self.qadr[leg] = np.array([m.jnt_qposadr[i] for i in ids])
            self.dadr[leg] = np.array([m.jnt_dofadr[i] for i in ids])
            self.tip_body[leg] = bid(f"nmf/{leg}_tarsus5")
            self.tip0[leg] = self.ik.xpos[self.tip_body[leg]].copy()
        # per-motor frequency tracking (zero crossings of each joint angle)
        na = len(self.act_q)
        self._hz = np.zeros(na)
        self._mean = d.qpos[self.act_q].copy()
        self._amp = np.zeros(na)
        self._prev_neg = np.zeros(na, bool)
        self._last_up = np.full(na, -9.0)
        self._sim_t = 0.0
        self.target = d.qpos[self.act_q].copy()      # latest commanded angle of every motor (rad)
        # free orbit camera that follows the thorax: drag to rotate, wheel to zoom
        dist = 0.0
        try:
            cid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_CAMERA, "nmf/trackcam")
            dist = float(np.linalg.norm(m.cam_pos[cid]))
        except Exception:
            pass
        thorax = bid("nmf/c_thorax")
        self.thorax = thorax
        self.I_yaw = 0.06                           # effective yaw inertia incl. foot friction (measured), for the turning reflex
        pj = [jid("nmf/c_head-c_rostrum-pitch"), jid("nmf/c_rostrum-c_haustellum-pitch")]
        self.prob_adr = np.array([m.jnt_qposadr[i] for i in pj if i >= 0], int)
        self.cam = mj.MjvCamera()
        self.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = thorax if thorax >= 0 else 1
        self.cam_default = (135.0, -25.0, dist if dist > 1e-6 else 4.0)
        self.view_reset()
        self.renderer = mj.Renderer(m, height=self.h, width=self.w)
        # --- vision: one small camera renderer shared by both compound eyes, and the neck joints used for gaze stabilisation
        self.eye_renderer = mj.Renderer(m, height=EYE_PX, width=EYE_PX)
        self.head_body = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "nmf/c_head")
        self.eye_body = {"l": mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "nmf/l_eye"), "r": mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "nmf/r_eye")}
        self.neck_adr = {}
        for ax in ("roll", "pitch"):
            jj = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, f"nmf/c_thorax-c_head-{ax}")
            self.neck_adr[ax] = int(m.jnt_qposadr[jj]) if jj >= 0 else -1
        self._in_head = np.zeros(m.nbody, bool)
        for bi in range(m.nbody):
            k = bi
            while k > 0:
                if k == self.head_body:
                    self._in_head[bi] = True
                    break
                k = int(m.body_parentid[k])
        bid = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "nmf/" + n)
        self.part_ids = {"lant": [bid("l_pedicel"), bid("l_funiculus"), bid("l_arista")],
                         "rant": [bid("r_pedicel"), bid("r_funiculus"), bid("r_arista")], "palp": [bid("c_rostrum")]}
        self.neck_cmd = {"roll": 0.0, "pitch": 0.0}
        self.neck_opt = {"roll": 0.0, "pitch": 0.0}
        self.reset()

    def _draw_arena(self, rend=None, for_eye=False):
        """Add walls, obstacles and food to the rendered scene (visual only; collisions are handled in step)."""
        sc, mj = (rend or self.renderer).scene, self.mj
        eye = np.eye(3).ravel()
        def add(kind, size, pos, rgba):
            if sc.ngeom >= sc.maxgeom:
                return
            mj.mjv_initGeom(sc.geoms[sc.ngeom], kind, np.asarray(size, float), np.asarray(pos, float), eye,
                            np.asarray(rgba, np.float32))
            sc.ngeom += 1
        if self.arena is not None:
            hf = self.arena.half
            wall = (0.55, 0.55, 0.62, 0.22)       # see-through, so the camera outside the arena never hides the fly
            add(mj.mjtGeom.mjGEOM_BOX, [hf, 0.3, 0.6], [0, hf + 0.3, 0.6], wall)
            add(mj.mjtGeom.mjGEOM_BOX, [hf, 0.3, 0.6], [0, -hf - 0.3, 0.6], wall)
            add(mj.mjtGeom.mjGEOM_BOX, [0.3, hf, 0.6], [hf + 0.3, 0, 0.6], wall)
            add(mj.mjtGeom.mjGEOM_BOX, [0.3, hf, 0.6], [-hf - 0.3, 0, 0.6], wall)
            for ox, oy, r in self.arena.obstacles:
                add(mj.mjtGeom.mjGEOM_CYLINDER, [r, 1.6, 0], [ox, oy, 1.6], (0.45, 0.5, 0.62, 1.0))
            for fx, fy, amt in self.arena.foods:
                r = 0.25 + 0.45 * max(0.0, min(1.0, amt))
                if for_eye:                                     # bigger + brighter in the eyes' greyscale view so food stands out
                    add(mj.mjtGeom.mjGEOM_SPHERE, [r * 1.8, 0, 0], [fx, fy, r * 1.8], (1.0, 0.9, 0.45, 1.0))
                else:
                    add(mj.mjtGeom.mjGEOM_SPHERE, [r, 0, 0], [fx, fy, r], (0.95, 0.55, 0.12, 1.0))
        for p, f in (): # (contact arrows removed)                       # ground-contact force arrows (magenta), like neuromechfly.org
            L = float(np.linalg.norm(f))
            if L < 1e-6 or sc.ngeom >= sc.maxgeom:
                continue
            tip = p + f / L * min(2.0, L / 5.0)
            g = sc.geoms[sc.ngeom]
            mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), eye, np.array([0.95, 0.15, 0.85, 1.0], np.float32))
            mj.mjv_connector(g, mj.mjtGeom.mjGEOM_CAPSULE, 0.04, p, tip)
            sc.ngeom += 1

    # ---------------------------------------------------------------- camera
    def view_reset(self):
        if getattr(self, "cam_default", None) is None:
            return
        self.cam.azimuth, self.cam.elevation, self.cam.distance = self.cam_default
        self._t_last = 0.0
    def view_orbit(self, d_az, d_el):
        if self.renderer is None:
            return
        self.cam.azimuth = (self.cam.azimuth - d_az * 0.5) % 360.0
        self.cam.elevation = float(np.clip(self.cam.elevation - d_el * 0.4, -89.0, 89.0))
        self._t_last = 0.0                                   # redraw right away
    def view_zoom(self, factor):
        if self.renderer is None:
            return
        base = self.cam_default[2]
        self.cam.distance = float(np.clip(self.cam.distance * factor, base * 0.3, base * 4.0))
        self._t_last = 0.0

    # ---------------------------------------------------------------- state
    def reset(self, x=0.0, y=0.0, heading=0.0):
        self.groom = 0.0
        self.feed = 0.0
        self.v = self.w_turn = 0.0
        self._bv = self._bw = self._vm = self._wm = 0.0
        self._wf = self._wi = 0.0
        self.phase = 0.0
        self.gait_hz = 0.0
        self.x, self.y, self.heading = x, y, heading
        if self.renderer is not None:
            self._place(x, y, heading)
    def _place(self, x, y, heading):
        """Put the fly upright at (x, y) facing `heading`, legs in the neutral pose, at rest."""
        mj, m, d = self.mj, self.model, self.data
        mj.mj_resetDataKeyframe(m, d, self.key)
        d.qpos[:] = self.q_neutral
        d.qpos[0], d.qpos[1], d.qpos[2] = x, y, self.root_z + 0.02
        d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6] = math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2)
        d.qvel[:] = 0.0
        self.ik.qpos[:] = self.q_neutral
        d.ctrl[:] = self.ctrl_neutral
        for i in self.adh.values():
            d.ctrl[i] = 1.0
        self.target = self.q_neutral[self.act_q].copy()
        for _ in range(int(0.12 / self.PHYS_DT)):
            mj.mj_step(m, d)
        self._read_state()
    def teleport(self, x, y):
        self.reset(x, y, self.heading)
    def _read_state(self):
        d = self.data
        w, qx, qy, qz = d.qpos[3:7]
        self.x, self.y = float(d.qpos[0]), float(d.qpos[1])
        self.heading = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        self.pitch = math.asin(max(-1.0, min(1.0, 2 * (w * qy - qz * qx))))
        self.roll = math.atan2(2 * (w * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        self.upright = 1 - 2 * (qx * qx + qy * qy)
        self.speed_mm_s = float(math.hypot(d.qvel[0], d.qvel[1]))

    # ----------------------------------------------------------------- gait
    def _solve_leg(self, leg, dx, dz, dy=0.0):
        mj, m, d = self.mj, self.model, self.ik
        target = self.tip0[leg] + np.array([dx, dy, dz])
        qa, da = self.qadr[leg], self.dadr[leg]
        q0 = self.q_neutral[qa]
        jac = np.zeros((3, m.nv))
        for _ in range(4):
            mj.mj_kinematics(m, d)
            mj.mj_comPos(m, d)
            err = target - d.xpos[self.tip_body[leg]]
            if err @ err < 1e-6:
                break
            mj.mj_jacBody(m, d, jac, None, self.tip_body[leg])
            J = jac[:, da]
            dq = J.T @ np.linalg.solve(J @ J.T + 0.02 * np.eye(3), err)
            d.qpos[qa] = np.clip(d.qpos[qa] + np.clip(dq, -0.25, 0.25), q0 - 1.0, q0 + 1.0)

    def _track_hz(self, q, dt):
        """Vectorised zero-crossing frequency estimate of every motor angle."""
        self._sim_t += dt
        a = min(1.0, dt / 0.5)
        self._mean += a * (q - self._mean)
        s = q - self._mean
        self._amp += min(1.0, dt / 0.15) * (np.abs(s) - self._amp)
        neg = s < 0
        up = self._prev_neg & ~neg
        if up.any():
            per = self._sim_t - self._last_up[up]
            ok = (per > 0.03) & (per < 1.5)
            hz = np.where(ok, 1.0 / np.maximum(per, 1e-6), 0.0)
            idx = np.flatnonzero(up)
            self._hz[idx] = np.where(self._hz[idx] > 0, 0.6 * self._hz[idx] + 0.4 * hz, hz)
            self._last_up[up] = self._sim_t
        self._prev_neg = neg
        stale = (self._sim_t - self._last_up > 0.7) | (self._amp < 0.02)
        self._hz[stale] = 0.0

    def _contacts(self):
        mj, m, d = self.mj, self.model, self.data
        f6 = np.zeros(6)
        draw, f = [], np.zeros(6)
        n = 0
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            leg = self.body_leg[b2] if b1 == 0 else (self.body_leg[b1] if b2 == 0 else -1)
            if leg < 0 and not (b1 == 0 or b2 == 0):
                continue
            mj.mj_contactForce(m, d, i, f)
            fn = float(f[0])
            n += 1
            if leg >= 0:
                f6[leg] += fn
            fw = c.frame[:3] * fn                          # normal force, world frame (points from geom1 to geom2)
            if b2 == 0:
                fw = -fw
            draw.append((np.array(c.pos), fw))
        self.leg_force, self.n_contacts, self.contact_draw = f6, n, draw

    def _apply_gait(self, vs, f, scale, gr):
        """Joint targets (inverse kinematics) + adhesion for the current gait phase -> motor commands."""
        d = self.data
        for leg in self.LEGS:
            s = (self.phase + self.TRIPOD[leg]) % 1.0
            dx = dy = dz = 0.0
            if f > 0:
                S = vs[leg] * (self.STANCE / f) * scale * (1.0 - gr)
                n = float(np.linalg.norm(S))
                if n > self.STEPMAX:
                    S = S * (self.STEPMAX / n)
                    n = self.STEPMAX
                if s < self.STANCE:                        # foot on the ground, sliding backwards under the body
                    kk, dz = 0.5 - s / self.STANCE, -self.PRESS
                else:                                      # swing: lift and carry forward
                    u = (s - self.STANCE) / (1.0 - self.STANCE)
                    kk = -0.5 + 0.5 * (1 - math.cos(math.pi * u))
                    dz = self.LIFT * (self.HIND_LIFT if leg in ('lh', 'rh') else 1.0) * math.sin(math.pi * u) * min(1.0, n / (0.5 * self.STEPMAX))
                dx, dy = S[0] * kk, S[1] * kk
            if gr > 0.02 and leg in ("lf", "rf"):          # front legs lift and rub in front of the head
                ph = 2 * math.pi * 6.0 * self._groom_t + (0.0 if leg == "lf" else math.pi)
                dx = (1 - gr) * dx + gr * (0.9 + 0.3 * math.sin(ph))
                dz = (1 - gr) * dz + gr * (0.9 + 0.25 * math.cos(ph))
                dy = (1 - gr) * dy
            self._solve_leg(leg, dx, dz, dy)
            if leg in self.adh:
                swing = f > 0 and s >= self.STANCE
                d.ctrl[self.adh[leg]] = 0.0 if (swing or (gr > 0.5 and leg in ("lf", "rf"))) else 1.0
        if self.prob_adr.size:
            self.ik.qpos[self.prob_adr] = self.FEED_ANGLES[: self.prob_adr.size] * float(min(1.0, max(0.0, self.feed)))
        if self.GAZE_STAB > 0.0:                                    # neck reflex: counter-rotate the head against the body's roll / pitch
            for ax, ang in (("roll", self.roll), ("pitch", self.pitch)):
                adr = self.neck_adr.get(ax, -1)
                if adr < 0:
                    continue
                opt = float(np.clip(-self.NECK_SIGN[ax] * ang * self.GAZE_STAB, -self.NECK_MAX, self.NECK_MAX))
                self.neck_opt[ax] = opt
                self.neck_cmd[ax] += (opt - self.neck_cmd[ax]) * min(1.0, self._neck_dt / self.NECK_TAU)
                self.ik.qpos[adr] = self.q_neutral[adr] + self.neck_cmd[ax]
        self.target = self.ik.qpos[self.act_q].copy()
        d.ctrl[self.pos_act] = self.target

    def step(self, fwd, back, left, right, dt):
        if self.renderer is None:
            return
        mj, m, d = self.mj, self.model, self.data
        wall0 = time.perf_counter()
        dt = max(1e-3, min(dt, 0.06))
        v_t = (fwd - back) * self.V_MAX
        w_t = -(right - left) * self.W_MAX                 # counter-clockwise = positive; right drive turns clockwise
        k = min(1.0, dt * 8.0)                             # smooth the commands a little
        # Spinal-cord-like feedback: the feet slip, so measure how fast the body really moves / turns (proprioception +
        # a gyro-like signal) and trim the gait command until it matches what the brain asked for.
        c_h, s_h = math.cos(self.heading), math.sin(self.heading)
        v_meas = float(d.qvel[0] * c_h + d.qvel[1] * s_h)
        w_meas = float(d.qvel[5])
        a_m = min(1.0, dt / 0.2)
        self._vm += a_m * (v_meas - self._vm)
        self._wm += a_m * (w_meas - self._wm)
        if abs(v_t) < 0.5 and abs(w_t) < 0.05:
            self._bv = self._bw = 0.0                      # told to stand still: no trim
        else:
            self._bv = float(np.clip(self._bv + self.K_V * (v_t * self.EFF_V - self._vm) * dt, -self.V_MAX, self.V_MAX))
            self._bw = float(np.clip(self._bw + self.K_W * (w_t * self.EFF_W - self._wm) * dt, -self.BIAS_W, self.BIAS_W))
        self.v += ((v_t + self._bv) - self.v) * k
        self.w_turn += ((w_t + self._bw) - self.w_turn) * k
        vs = {}
        for leg in self.LEGS:
            rx, ry = self.tip0[leg][0], self.tip0[leg][1]
            vs[leg] = np.array([self.v - self.w_turn * ry, self.w_turn * rx])   # where this foot's patch of ground has to slide
        smax = max(float(np.linalg.norm(x)) for x in vs.values())
        gr = float(min(1.0, max(0.0, self.groom)))
        if smax > 0.4:
            f_need = self.STANCE * smax / self.STEPMAX
            f = min(self.F_MAX, max(self.F_MIN, f_need))
            scale = min(1.0, f / f_need)
        else:
            f, scale = 0.0, 0.0                            # standing: legs settle back to the neutral pose
        self.gait_hz = f
        self._w_des = w_t * self.TURN_RATE
        # --- real physics: the gait targets are refreshed every few milliseconds, MuJoCo steps in between;
        #     the feet push on the ground and that is what moves / turns the body --------------------------------
        n = int(min(220, max(1, round(dt / self.PHYS_DT))))
        chunk = 12
        done = 0
        while done < n:
            c = min(chunk, n - done)
            cdt = c * self.PHYS_DT
            self.phase = (self.phase + f * cdt) % 1.0
            self._groom_t += cdt
            self._neck_dt = cdt
            self._apply_gait(vs, f, scale, gr)
            if self.YAW_ASSIST > 0.0 and self.upright > 0.5:   # gyro-like turning reflex (PI on the yaw rate)
                self._wf += min(1.0, cdt / 0.05) * (float(d.qvel[5]) - self._wf)
                err = self._w_des - self._wf
                self._wi = float(np.clip(self._wi + err * cdt, -self.YAW_IMAX, self.YAW_IMAX))
                tq = self.YAW_ASSIST * (self.YAW_KP * err + self.YAW_KI * self._wi)
                d.xfrc_applied[self.thorax, 5] = float(np.clip(tq, -self.YAW_TMAX, self.YAW_TMAX))
            for _ in range(c):
                mj.mj_step(m, d)
            done += c
            self._track_hz(d.qpos[self.act_q], cdt)
        self._contacts()
        self._read_state()
        if not np.isfinite(d.qpos).all() or self.upright < 0.35 or d.qpos[2] < 0.1 or d.qpos[2] > 6.0:
            self._place(self.x if np.isfinite(self.x) else 0.0, self.y if np.isfinite(self.y) else 0.0,
                        self.heading if np.isfinite(self.heading) else 0.0)   # fell over / blew up: stand it back up
        if self.arena is not None:
            nx, ny = self.arena.resolve(self.x, self.y)
            if abs(nx - self.x) > 1e-9 or abs(ny - self.y) > 1e-9:    # the wall / an obstacle is solid
                d.qpos[0], d.qpos[1] = nx, ny
                d.qvel[0] = d.qvel[1] = 0.0
                self.x, self.y = nx, ny
        wall = max(1e-4, time.perf_counter() - wall0)
        self.rt_factor += 0.1 * (n * self.PHYS_DT / wall - self.rt_factor)

    # ------------------------------------------------------------ read-outs
    def motor_table(self):
        """Per-leg motor read-out for the Motor Properties page: [(group title, [(label, target, actual, torque, Hz, neutral angle)])]."""
        d = self.data
        actual = d.qpos[self.act_q]
        force = d.actuator_force[self.pos_act]
        groups = {}
        for i, (leg, label) in enumerate(self.act_info):
            hz = float(self._hz[i])
            if self.gait_hz > 0.5 and hz > 1.6 * self.gait_hz:
                hz *= 0.5                                  # a joint that crosses zero twice per step: count whole cycles
            groups.setdefault(leg, []).append((label, float(self.target[i]), float(actual[i]), float(force[i]), hz,
                                               float(self.q_neutral[self.act_q[i]])))
        out = []
        for leg in self.LEGS:
            if leg in groups:
                out.append((leg, self.LEG_NAMES[leg], groups[leg]))
        if "head" in groups:
            out.append(("head", "Head / proboscis", groups["head"]))
        return out

    # --------------------------------------------------------------- render
    shadows = False                                  # shadows + reflections roughly double the cost of every 3-D frame on software OpenGL
    def _shade(self, rend):
        try:
            f = self.mj.mjtRndFlag
            rend.scene.flags[f.mjRND_SHADOW] = int(self.shadows)
            rend.scene.flags[f.mjRND_REFLECTION] = int(self.shadows)
        except Exception:
            pass
    def apply_physics(self, kp, adhesion, friction, gravity):
        """Push live settings into the running MuJoCo model: joint stiffness, foot adhesion, ground friction, gravity."""
        m = getattr(self, 'model', None)
        if m is None:
            return
        mjm = self.mj
        joint = (m.actuator_trntype == int(mjm.mjtTrn.mjTRN_JOINT)) & (m.actuator_biastype == int(mjm.mjtBias.mjBIAS_AFFINE))
        m.actuator_gainprm[joint, 0] = kp
        m.actuator_biasprm[joint, 1] = -kp
        adh = m.actuator_trntype == int(mjm.mjtTrn.mjTRN_BODY)
        m.actuator_gainprm[adh, 0] = adhesion
        gid = mjm.mj_name2id(m, mjm.mjtObj.mjOBJ_GEOM, "ground_plane")
        if gid >= 0:
            m.geom_friction[gid, 0] = friction
        if not hasattr(self, "_g0"):
            self._g0 = float(m.opt.gravity[2])
        m.opt.gravity[2] = self._g0 * gravity
    def render(self, w, h):
        if self.renderer is None:
            img = np.full((h, w, 3), 245, np.uint8)
            cv2.putText(img, "NeuroMechFly unavailable", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
            msg = (self.error or "FlyGym not installed").replace("RuntimeError: ", "")
            for i, line in enumerate(textwrap.wrap(msg, 50)[:9]):
                cv2.putText(img, line, (12, 70 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
            cv2.putText(img, "Install lines: top of flylab.py", (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1, cv2.LINE_AA)
            return img
        now = time.perf_counter()
        if self._cached is not None and now - self._t_last < max(getattr(self, 'render_dt', 0.033), 2.5 * self._render_s):
            return self._cached                            # slow GPU / software GL: reuse the last frame
        try:
            self.renderer.update_scene(self.data, camera=self.cam)
            self._shade(self.renderer)
            self._draw_arena()
            img = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
            done = time.perf_counter()
            self._render_s = 0.8 * self._render_s + 0.2 * (done - now)
            self._cached, self._t_last = img, done
            return img
        except Exception as exc:
            img = np.full((h, w, 3), 245, np.uint8)
            cv2.putText(img, "NeuroMechFly render error", (12, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
            cv2.putText(img, str(exc)[:52], (12, h // 2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (90, 90, 90), 1, cv2.LINE_AA)
            return img

    def neck_state(self):
        """(optimal roll, optimal pitch, actual roll, actual pitch) of the neck in degrees."""
        d = self.data
        out = []
        for ax in ("roll", "pitch"):
            out.append(math.degrees(self.neck_opt[ax]))
        for ax in ("roll", "pitch"):
            adr = self.neck_adr.get(ax, -1)
            out.append(math.degrees(float(d.qpos[adr] - self.q_neutral[adr])) if adr >= 0 else 0.0)
        return tuple(out)
    def render_head(self, w=480, h=400):
        """Close-up of the real 3-D head (front view). Returns (rgb image, body id of every pixel, -1 = background) or None."""
        if self.renderer is None or self.head_body < 0:
            return None
        mj, m, d = self.mj, self.model, self.data
        try:
            if getattr(self, "head_renderer", None) is None:
                self.head_renderer = mj.Renderer(m, height=h, width=w)
            r = self.head_renderer
            hm = d.xmat[self.head_body].reshape(3, 3)
            f = hm[:, 0]                                         # the fly's forward axis; the camera looks at the face
            cam = mj.MjvCamera()
            cam.type = mj.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = d.xpos[self.head_body] + f * 0.42 + hm[:, 2] * -0.02
            cam.distance = 1.45
            cam.azimuth = math.degrees(math.atan2(-f[1], -f[0]))
            cam.elevation = -14.0
            outs = []
            for seg in (False, True):
                if seg:
                    r.enable_segmentation_rendering()
                r.update_scene(d, camera=cam)
                self._shade(r)
                sc = r.scene
                for gi in range(sc.ngeom):
                    g = sc.geoms[gi]
                    if g.objtype == mj.mjtObj.mjOBJ_GEOM and 0 <= g.objid < m.ngeom and not self._in_head[m.geom_bodyid[g.objid]]:
                        g.rgba[3] = 0.0
                        if seg:
                            g.segid = -1
                outs.append(r.render().copy())
                if seg:
                    r.disable_segmentation_rendering()
            rgb, sg = outs
            body = np.full(sg.shape[:2], -1, np.int32)
            ok = sg[:, :, 1] == int(mj.mjtObj.mjOBJ_GEOM)
            gid = sg[:, :, 0].astype(np.int64)
            ok &= (gid >= 0) & (gid < m.ngeom)
            body[ok] = m.geom_bodyid[gid[ok]]
            return rgb, body
        except Exception:
            return None
    def render_eyes(self):
        """Raw (greyscale) view of each compound eye, rendered from the head. Returns (left, right) uint8 arrays, or None."""
        if self.renderer is None or getattr(self, "eye_renderer", None) is None or self.head_body < 0:
            return None
        mj, m, d = self.mj, self.model, self.data
        hm = d.xmat[self.head_body].reshape(3, 3)
        fwd, up = hm[:, 0], hm[:, 2]
        left = np.cross(up, fwd)
        old = float(m.vis.global_.fovy)
        outs = []
        try:
            m.vis.global_.fovy = EYE_FOV
            for side, sgn in (("l", 1.0), ("r", -1.0)):
                pos = d.xpos[self.head_body] + fwd * 0.35 + left * (0.35 * sgn) + up * 0.1      # (the eye bodies sit at the mesh origin, so offset from the head)
                ang = math.radians(EYE_YAW)
                f = math.cos(ang) * fwd + math.sin(ang) * sgn * left
                f = f / np.linalg.norm(f)
                u = up - f * float(up @ f)
                u = u / np.linalg.norm(u)
                cam = mj.MjvCamera()
                cam.type = mj.mjtCamera.mjCAMERA_FREE
                cam.lookat[:] = pos + f * 0.5
                cam.distance = 0.5
                cam.azimuth = math.degrees(math.atan2(f[1], f[0]))
                cam.elevation = math.degrees(math.asin(max(-1.0, min(1.0, f[2]))))
                self.eye_renderer.update_scene(d, camera=cam)
                self._shade(self.eye_renderer)
                for k in range(2):                           # roll the camera with the head (a free camera always keeps world-up)
                    gc = self.eye_renderer.scene.camera[k]
                    gc.pos[:] = pos + f * 0.12
                    gc.forward[:] = f
                    gc.up[:] = u
                sc = self.eye_renderer.scene
                for gi in range(sc.ngeom):                   # the eye looks out of the head: hide the head's own meshes
                    g = sc.geoms[gi]
                    if g.objtype == mj.mjtObj.mjOBJ_GEOM and 0 <= g.objid < m.ngeom and self._in_head[m.geom_bodyid[g.objid]]:
                        g.rgba[3] = 0.0
                self._draw_arena(self.eye_renderer, for_eye=True)
                img = self.eye_renderer.render()
                outs.append((255.0 * (img.mean(axis=2) / 255.0) ** 0.5).astype(np.uint8))      # lift the dark floor so its texture shows
        except Exception:
            return None
        finally:
            m.vis.global_.fovy = old
        return outs[0], outs[1]
    def close(self):
        if getattr(self, "head_renderer", None) is not None:
            try:
                self.head_renderer.close()
            except Exception:
                pass
            self.head_renderer = None
        if getattr(self, "eye_renderer", None) is not None:
            try:
                self.eye_renderer.close()
            except Exception:
                pass
            self.eye_renderer = None
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
            self.renderer = None
class MotionFilter:
    """Turns the jittery per-frame motion (dx, dy) into steady directions.
    1. exponential smoothing   -> a fly wobbling left/right every frame averages out instead of flipping A-D-A-D
    2. hysteresis              -> a direction switches on above `on` and only off below `off` (about half of it)
    3. minimum hold time       -> a direction can't be flipped or dropped again for MOTION_MIN_HOLD seconds
    Axis: dx > 0 = left (A), dy > 0 = forward (W), as in the rest of the program."""
    def __init__(self):
        self.vx = self.vy = 0.0
        self.sx = self.sy = 0                # current direction on each axis: -1, 0, +1
        self.tx = self.ty = 0.0              # time of the last change on each axis
    def reset(self):
        self.__init__()
    def _axis(self, v, s, t, now, on, off):
        if now - t < MOTION_MIN_HOLD:
            return s, t
        if s == 0:
            n = 1 if v > on else (-1 if v < -on else 0)
        elif s > 0:
            n = -1 if v < -on else (0 if v < off else 1)
        else:
            n = 1 if v > on else (0 if v > -off else -1)
        return (n, now) if n != s else (s, t)
    def update(self, dx, dy, now, on, smoothing):
        a = 1.0 - min(0.95, max(0.0, smoothing))             # weight of the newest frame
        self.vx += a * (dx - self.vx)
        self.vy += a * (dy - self.vy)
        off = on * 0.5
        self.sx, self.tx = self._axis(self.vx, self.sx, self.tx, now, on, off)
        self.sy, self.ty = self._axis(self.vy, self.sy, self.ty, now, on, off)
        return self.sx, self.sy
    def dominant(self, on):
        """The single strongest direction as 'W','S','A','D' (or None) - for modes that must pick exactly one."""
        cx = abs(self.vx) / on if self.sx else 0.0
        cy = abs(self.vy) / on if self.sy else 0.0
        if cx == 0.0 and cy == 0.0:
            return None
        if cy >= cx:
            return "W" if self.sy > 0 else "S"
        return "A" if self.sx > 0 else "D"
class FlyControllerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FlyLab Control Panel")
        self.root.attributes("-topmost", True)
        self.root.resizable(False, False)
        # State Variables (Tk objects: ONLY touch these from the main/UI thread)
        self.walk_enabled = tk.BooleanVar(value=False)
        self.macro_enabled = tk.BooleanVar(value=False)
        self.jump_enabled = tk.BooleanVar(value=False)
        self.input_mode = tk.StringVar(value="wasd")
        self.game_mode = tk.StringVar(value="Classic")
        self.replay_enabled = tk.BooleanVar(value=False)
        self.source = tk.StringVar(value="")   # "brain" = simulation drives the keys, "screen" = optical flow
        self.sens_var = tk.DoubleVar(value=0.10)       # motion needed to count as a direction
        self.smooth_var = tk.DoubleVar(value=0.65)     # 0 = raw, higher = steadier but slower to react
        self.food_key = load_food_key()                     # Advanced: tapped each time the fly finishes a food pellet ('' = off)
        self.food_seen = 0
        self.directional_var = tk.BooleanVar(value=False)   # Advanced: keys follow the fly's compass heading, not its body axes
        # Plain-Python mirrors of the Tk variables. Background threads read ONLY these.
        self.cfg = {
            "walk": self.walk_enabled.get(),
            "macro": self.macro_enabled.get(),
            "jump": self.jump_enabled.get(),
            "mode": self.input_mode.get(),
            "game": self.game_mode.get(),
            "replay": self.replay_enabled.get(),
            "source": self.source.get(),
            "sens": self.sens_var.get(),
            "smooth": self.smooth_var.get(),
            "directional": self.directional_var.get(),
        }
        for key, var in (("walk", self.walk_enabled), ("macro", self.macro_enabled),
                         ("jump", self.jump_enabled), ("mode", self.input_mode),
                         ("game", self.game_mode), ("replay", self.replay_enabled),
                         ("source", self.source), ("sens", self.sens_var), ("smooth", self.smooth_var),
                         ("directional", self.directional_var)):
            var.trace_add("write", lambda *_, k=key, v=var: self.cfg.__setitem__(k, v.get()))
        self.capture_region = load_region()
        self.blooket_modes, self.blooket_map = load_blooket_map()
        # Latest preview frame (PNG bytes) handed from the vision thread to the UI thread
        self.latest_png = None
        self.preview_seq = 0
        self.shown_seq = -1
        self.preview_img = None
        self.stop_event = threading.Event()
        self.terminating = False
        self.selecting = False
        self.macro_in_progress = False
        self.blooket_action_lock = False
        self.replay_in_progress = False
        # Keys currently held down (so we only send a key event when something actually changes)
        self.held = set()
        self.key_lock = threading.Lock()
        self.last_motion, self.last_dirs = (0.0, 0.0), (0, 0)
        self.hist = deque(maxlen=900)                      # (time, spikes/s, drive, loop ms, motion x, motion y) for the graphs
        self._spk_prev, self.graph_win, self.graph_label = 0, None, None
        self.keylog_q = queue.Queue()                      # key presses / clicks sent to the game, shown in the log box
        # Live brain + body
        self.init_brain()
        # UI: one window. Tab 1 = Brain Lab (left) + Fly Bridge controls (right). Tab 2 = MaleCNS setup.
        bottom_bar = ttk.Frame(root, padding=(15, 6, 15, 10))        # always visible, whichever tab is open
        bottom_bar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(bottom_bar, text="Terminate Process", command=self.terminate).pack(fill=tk.X)
        self.nb = ttk.Notebook(root)
        self.nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        tab_ctrl = ttk.Frame(self.nb)
        tab_build = ttk.Frame(self.nb)
        tab_setup = ttk.Frame(self.nb)
        self.nb.add(tab_ctrl, text="   Control Panel   ")
        self.tab_ctrl = tab_ctrl
        self.nb.add(tab_build, text="   Build (obstacles & food)   ")
        self.nb.add(tab_setup, text="   MaleCNS Setup   ")
        tab_motor = ttk.Frame(self.nb)
        self.nb.add(tab_motor, text="   Motor Properties   ")
        self.tab_motor = tab_motor
        self.build_motors(tab_motor)
        tab_vis = ttk.Frame(self.nb)
        self.nb.add(tab_vis, text="   Miscellaneous   ")
        self.tab_vision = tab_vis
        self.eye_hist = collections.deque(maxlen=900)
        self.odor_hist = collections.deque(maxlen=900)
        self.odor_pulse = {"l": 0.0, "r": 0.0}
        self.odor_now = (0.0, 0.0)
        self._eye_prev = None
        self.build_vision(tab_vis)
        tab_set = ttk.Frame(self.nb)
        self.nb.add(tab_set, text="   Settings   ")
        self.build_settings(tab_set)
        tab_min = ttk.Frame(self.nb)
        self.nb.add(tab_min, text="   Minimize   ")
        self.build_minimized(tab_min)
        self.tab_min = tab_min
        self.nb.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        self.nb.select(tab_min if self.S("start_min") else tab_ctrl)      # start parked on the Minimize page (Settings > Display)
        self.tab_build = tab_build
        self.build_arena_tab(tab_build)
        self.build_setup(tab_setup)
        self.root.after(600, self.auto_build)

        content_frame = ttk.Frame(tab_ctrl)
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        graph_frame = ttk.Frame(content_frame, padding=(15, 15, 0, 15))      # the graphs live in this window, on the left
        graph_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.graph_label = tk.Label(graph_frame, borderwidth=0, highlightthickness=1, highlightbackground="#bbbbbb", bg="white")
        self.graph_label.pack(anchor=tk.N)
        self.graph_win = graph_frame
        lab_frame = ttk.Frame(content_frame, padding=(15, 15, 0, 15))
        lab_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.build_lab(lab_frame)
        main_frame = ttk.Frame(content_frame, padding="15")
        main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(main_frame, text="FlyLab Bridge Control", font=("Arial", 12, "bold")).pack(pady=(0, 10))
        preview_box = ttk.LabelFrame(main_frame, text=" Live Tracking Preview ", padding="5")
        preview_box.pack(fill=tk.X, pady=(0, 5))
        self.preview_label = ttk.Label(preview_box, text="Waiting for capture...", anchor="center")
        self.preview_label.pack()
        self.select_region_btn = ttk.Button(
            preview_box, text="Select Fly Area (drag a box)",
            command=self.select_region, state="disabled")
        self.select_region_btn.pack(fill=tk.X, pady=(5, 0))
        columns_frame = ttk.Frame(main_frame)
        columns_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        left_col = ttk.LabelFrame(columns_frame, text=" Roblox Layouts ", padding="10")
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        ttk.Radiobutton(left_col, text="WASD Layout", variable=self.input_mode, value="wasd", command=self.update_mode_states).pack(anchor=tk.W, pady=6)
        ttk.Radiobutton(left_col, text="Arrows Layout", variable=self.input_mode, value="arrows", command=self.update_mode_states).pack(anchor=tk.W, pady=6)
        adv_frame = ttk.LabelFrame(left_col, text=" Advanced ", padding="6")
        adv_frame.pack(fill=tk.X, pady=(8, 0))
        dir_cb = ttk.Checkbutton(adv_frame, text="Directional", variable=self.directional_var)
        dir_cb.pack(anchor=tk.W)
        food_row = ttk.Frame(adv_frame)
        food_row.pack(anchor=tk.W, pady=(6, 0))
        ttk.Label(food_row, text="Map food to").pack(side=tk.LEFT)
        self.food_box = tk.Frame(food_row, bg="white", highlightthickness=1, highlightbackground="#888888", takefocus=1)
        self.food_box.pack(side=tk.LEFT, padx=(6, 0))
        self.food_lbl = tk.Label(self.food_box, bg="white", width=7, anchor="center", font=("Segoe UI", 9, "bold"))
        self.food_lbl.pack(side=tk.LEFT, ipady=2)
        self.food_trash = tk.Label(self.food_box, text="\U0001F5D1", bg="white", fg="#888888", cursor="hand2", font=("Segoe UI Emoji", 9))
        self.food_trash.pack(side=tk.RIGHT, padx=(0, 3))
        self.food_trash.bind("<Button-1>", lambda e: self.set_food_key(""))
        self.food_trash.bind("<Enter>", lambda e: self.food_trash.configure(fg="#cc0000"))
        self.food_trash.bind("<Leave>", lambda e: self.food_trash.configure(fg="#888888"))
        self.food_listening = False
        self.food_box.bind("<Enter>", lambda e: self.food_listen(True))     # hover = listen, then just press a key
        self.food_box.bind("<Leave>", lambda e: self.food_listen(False))
        self.food_lbl.bind("<Enter>", lambda e: self.food_listen(True))
        self.food_box.bind("<KeyPress>", self.food_keypress)
        self.refresh_food_box()
        right_col = ttk.LabelFrame(columns_frame, text=" Blooket Layouts ", padding="10")
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        ttk.Radiobutton(right_col, text="Activate Blooket", variable=self.input_mode, value="blooket", command=self.update_mode_states).pack(anchor=tk.W, pady=(0, 4))
        ttk.Label(right_col, text="Game Mode Routing:").pack(anchor=tk.W, pady=(4, 0))
        self.game_combo = ttk.Combobox(right_col, textvariable=self.game_mode, values=self.blooket_modes, state="readonly")
        self.game_combo.pack(fill=tk.X, pady=(2, 4))
        map_row = ttk.Frame(right_col)
        map_row.pack(fill=tk.X)
        self.map_btn = ttk.Button(map_row, text="Map buttons...", command=self.map_blooket_buttons)
        self.map_btn.pack(side=tk.LEFT)
        self.add_mode_btn = ttk.Button(map_row, text="+ Mode", command=self.add_blooket_mode)
        self.add_mode_btn.pack(side=tk.LEFT, padx=(4, 0))
        self.map_note = tk.StringVar(value="")
        ttk.Label(right_col, textvariable=self.map_note, wraplength=210, foreground="#555555").pack(anchor=tk.W, pady=(4, 0))
        self.game_mode.trace_add("write", lambda *_: self.refresh_map_note())
        self.refresh_map_note()
        source_frame = ttk.LabelFrame(main_frame, text=" Fly Source (what drives the keys) ", padding="10")
        source_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Radiobutton(source_frame, text="Brain simulation (direct, no screen needed)", variable=self.source, value="brain",
                          command=self.update_source_states).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(source_frame, text="Screen (optical flow on the captured fly area)", variable=self.source, value="screen",
                          command=self.update_source_states).pack(anchor=tk.W, pady=2)
        self.make_slider(source_frame, "Sensitivity", self.sens_var, 0.03, 0.40, "{:.2f}", label_w=12)
        self.make_slider(source_frame, "Smoothing", self.smooth_var, 0.0, 0.90, "{:.2f}", label_w=12)
        master_settings_frame = ttk.LabelFrame(main_frame, text=" Master Engine Settings ", padding="10")
        master_settings_frame.pack(fill=tk.X, pady=(10, 5))
        ttk.Checkbutton(master_settings_frame, text="Enable Fly Controller", variable=self.walk_enabled, command=self.clear_keys).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(master_settings_frame, text="Enable Jump Inputs (Space)", variable=self.jump_enabled).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(master_settings_frame, text="Enable Block Placement Macro (WSW Combo)", variable=self.macro_enabled).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(master_settings_frame, text="Auto-Replay Mode", variable=self.replay_enabled).pack(anchor=tk.W, pady=2)
        self.update_mode_states()
        self.vision_thread = threading.Thread(target=self.run_vision, daemon=True)
        self.vision_thread.start()
        self.update_preview()
        self.lab_t = time.perf_counter()
        self._place_window()
        self.root.after(80, self._place_window)          # again once the start-up tab/size changes have settled
        self.root.after(400, self._place_window)
        self.apply_settings()
        self.update_lab()
        self.root.protocol("WM_DELETE_WINDOW", self.terminate)
    # ------------------------------------------------------------------
    # BRAIN LAB (live simulation + neuron view + fly body view)
    # ------------------------------------------------------------------
    def init_brain(self):
        # The real NeuroMechFly model (FlyGym meshes), walking; no hand-drawn stand-in.
        self.body = NeuroMechFlyBody(VIEW_W, VIEW_H)
        self.tool = tk.StringVar(value="rotate")
        self.gain_var = tk.DoubleVar(value=1.0)
        self.rate_var = tk.DoubleVar(value=100.0)
        self.brush_var = tk.DoubleVar(value=22.0)
        self.net_note = tk.StringVar(value="")
        self.show_skel = tk.BooleanVar(value=True)
        self.blur_var = tk.DoubleVar(value=0.0)           # fixed at 0 = crisp (no slider any more)
        self.yaw, self.pitch, self.zoom = 0.0, 0.0, 1.7
        self.mouse = None
        self._imgs = {}
        self.sim = None
        self.arena_px = ARENA_PX
        self._smell_hover = self._smell_drag = False
        self._smell_img, self._smell_key = None, None
        self.smell_show = False
        self.arena = Arena()
        self.arena.default_layout()
        self.body.arena = self.arena
        self.explorer = Explorer()
        self.tracker = OdorTracker()
        self.user_run = False                              # the simulation starts paused
        self.preset_var = tk.StringVar(value=DYN_LABEL)
        self.side_var = tk.StringVar(value="Both")
        self.preset_ids = {}
        self.preset_combo = None
        self.load_brain()
    @staticmethod
    def cut_pool_to_dn(net, pools):
        """The premotor pool lights up the brain view, but the DN drives (which steer the body) must come only from the
        explicit commands - so the pool's own links onto the DNs are removed (in memory only, nothing is saved)."""
        try:
            W = net.W.tocsc()
            dn = np.zeros(net.n, bool)
            for g in GROUPS:
                dn[net.groups[g]] = True
            for g in GROUPS:
                for c in pools[g]:
                    a, b = W.indptr[c], W.indptr[c + 1]
                    rows = W.indices[a:b]
                    W.data[a:b][dn[rows]] = 0.0
            W.eliminate_zeros()
            net.W = W
        except Exception as err:
            print("[!] could not cut pool->DN links:", err)
    @staticmethod
    def build_up_pools(net):
        """For each DN group: the neurons that feed it most strongly (excluding the DNs themselves)."""
        pools = {}
        try:
            Wr = net.W.tocsr()                      # W[post, pre]: row = receiving neuron
            alld = np.concatenate([net.groups[g] for g in GROUPS])
            S = {}
            for g in GROUPS:
                idx = net.groups[g]
                st = np.zeros(net.n, np.float64)
                if idx.size:
                    st = np.asarray(Wr[idx, :].sum(axis=0)).ravel().astype(np.float64)
                    st[alld] = 0.0
                S[g] = st
            for g in GROUPS:
                others = np.max([S[h] for h in GROUPS if h != g], axis=0)
                sel = S[g] - 2.0 * np.maximum(others, 0.0)      # feeds this group and (almost) none of the others
                sel[sel < 0] = 0.0
                k = min(UP_POOL, int((sel > 0).sum()))
                pools[g] = np.argsort(sel)[::-1][:k].astype(np.int64) if k else np.zeros(0, np.int64)
        except Exception as err:
            print("[!] upstream pools unavailable:", err)
            pools = {g: np.zeros(0, np.int64) for g in GROUPS}
        return pools
    def load_brain(self):
        """(Re)load brain_subset.npz if it exists, otherwise the demo network, and start a fresh simulation."""
        try:
            net = load_net(BRAIN_FILE) if os.path.exists(BRAIN_FILE) else make_demo_net()
        except Exception as err:
            print(f"[!] Couldn't load {BRAIN_FILE} ({err}); using the demo network instead.")
            net = make_demo_net()
        print(f"[✓] Brain: {net.source}, {net.n} neurons, "
              + ", ".join(f"{g}={net.groups[g].size}" for g in GROUPS))
        sim = LIFSim(net)
        sim.running = False                                # starts paused; Run button + Brain source turn it on
        sim.start()
        if self.sim is not None:
            self.sim.stop()
        self.net = net
        self.up_pool = self.build_up_pools(net)
        self.stim_mask = np.zeros(net.n, bool)
        self._proj = (np.zeros(net.n),) * 3
        self.region_cols = REGION_COLORS[net.labels]
        self.body.reset()
        self.sim = sim
        self.groom_sets = [np.concatenate([v for v in net.presets.get(pid, {}).values()] or [np.zeros(0, np.int64)])
                           for pid, _l, _t, _f, kind in PRESETS if kind == "groom"]
        self.groom_sets = [a for a in self.groom_sets if a.size]
        self.feed_set = np.concatenate([v for v in net.presets.get("feeding", {}).values()] or [np.zeros(0, np.int64)])
        self.update_preset_list()
        self.net_note.set("DEMO network - use the MaleCNS Setup tab to load real data"
                          if net.source.startswith("DEMO") else f"Loaded {net.source} ({net.n} neurons)")
    def update_preset_list(self):
        labels, self.preset_ids = [DYN_LABEL, BASE_LABEL], {}
        for pid, label, _types, _fam, kind in PRESETS:
            if pid in self.net.presets:
                name = label + ("" if kind != "none" else "  (neurons only)")
                labels.append(name)
                self.preset_ids[name] = pid
        if self.preset_combo is not None:
            self.preset_combo.configure(values=labels)
        if self.preset_var.get() not in labels:
            self.preset_var.set(DYN_LABEL)
    def build_lab(self, parent):
        ttk.Label(parent, text="Fly Brain Lab  (MaleCNS)", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 6))
        views = ttk.Frame(parent)
        views.pack()
        body_box = ttk.LabelFrame(views, text=" Fly body: drag to rotate, wheel zoom, double-click reset ", padding=4)
        body_box.pack(side=tk.LEFT, padx=(0, 6))
        self.body_label = tk.Label(body_box, borderwidth=0, highlightthickness=0, bg="#102524")
        self.body_label.pack()
        self.body_label.configure(cursor="fleur")
        self.body_label.bind("<ButtonPress-1>", lambda e: setattr(self, "_body_drag", (e.x, e.y)))
        self.body_label.bind("<B1-Motion>", self.body_drag)
        self.body_label.bind("<MouseWheel>", lambda e: self.body.view_zoom(0.9 if e.delta > 0 else 1.1))
        self.body_label.bind("<Double-Button-1>", lambda e: self.body.view_reset())
        cloud_box = ttk.LabelFrame(views, text=" Neurons: drag to rotate / paint ", padding=4)
        cloud_box.pack(side=tk.LEFT)
        self.cloud_label = tk.Label(cloud_box, borderwidth=0, highlightthickness=0, cursor="crosshair", bg="#102524")
        self.cloud_label.pack()
        self.cloud_label.bind("<ButtonPress-1>", self.cloud_press)
        self.cloud_label.bind("<B1-Motion>", self.cloud_drag)
        self.cloud_label.bind("<Motion>", lambda e: setattr(self, "mouse", (e.x, e.y)))
        self.cloud_label.bind("<Leave>", lambda e: setattr(self, "mouse", None))
        self.cloud_label.bind("<MouseWheel>", self.cloud_wheel)
        row1 = ttk.Frame(parent)
        row1.pack(fill=tk.X, pady=(8, 0))
        self.run_btn = ttk.Button(row1, text="Run", command=self.toggle_run)
        self.run_btn.pack(side=tk.LEFT)
        ttk.Button(row1, text="Reset brain", command=self.reset_brain).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(row1, text="Clear painted", command=self.clear_paint).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(row1, text="Reset view", command=self.reset_view).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Checkbutton(row1, text="Shapes", variable=self.show_skel).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(row1, text="   Tool:").pack(side=tk.LEFT)
        for text, val in (("Rotate", "rotate"), ("Paint", "paint"), ("Erase", "erase")):
            ttk.Radiobutton(row1, text=text, variable=self.tool, value=val).pack(side=tk.LEFT, padx=(4, 0))
        row2 = ttk.Frame(parent)
        row2.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(row2, text="Mode / preset:").pack(side=tk.LEFT)
        self.preset_combo = ttk.Combobox(row2, textvariable=self.preset_var, state="readonly", width=38)
        self.preset_combo.pack(side=tk.LEFT, padx=(6, 10))
        ttk.Label(row2, text="Side:").pack(side=tk.LEFT)
        ttk.Combobox(row2, textvariable=self.side_var, state="readonly", width=7, values=("Both", "Left", "Right")).pack(side=tk.LEFT, padx=(6, 0))
        self.update_preset_list()
        sliders = ttk.LabelFrame(parent, text=" Simulation ", padding=6)
        sliders.pack(fill=tk.X, pady=(8, 0))
        self.make_slider(sliders, "Synapse gain", self.gain_var, 0.5, 8.0, "{:.2f}")
        self.make_slider(sliders, "Stim rate (Hz)", self.rate_var, 10, 300, "{:.0f}")
        self.make_slider(sliders, "Brush (px)", self.brush_var, 6, 70, "{:.0f}")
        brow = ttk.Frame(sliders)
        brow.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(brow, text="Fly body speed").pack(side=tk.LEFT)
        self.body_speed_var = tk.StringVar(value="1x")
        ttk.Combobox(brow, textvariable=self.body_speed_var, state="readonly", width=7,
                     values=("0.1x", "0.25x", "0.5x", "1x")).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(brow, text="(slow motion for the body / legs)", foreground="#666666").pack(side=tk.LEFT, padx=(8, 0))
        self.status = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self.status, font=("Consolas", 9), justify=tk.LEFT).pack(anchor=tk.W, pady=(8, 0))
        ttk.Label(parent, textvariable=self.net_note, foreground="#666666").pack(anchor=tk.W)
        log_box = ttk.LabelFrame(parent, text=" Key presses sent to the game ", padding=4)
        log_box.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        log_row = ttk.Frame(log_box)
        log_row.pack(fill=tk.BOTH, expand=True)
        self.key_text = tk.Text(log_row, height=7, font=("Consolas", 9), state="disabled", wrap="none",
                                bg="#fafafa", relief="flat", borderwidth=0)
        log_sb = ttk.Scrollbar(log_row, orient=tk.VERTICAL, command=self.key_text.yview)
        self.key_text.configure(yscrollcommand=log_sb.set)
        log_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.key_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Button(log_box, text="Clear log", command=self.clear_keylog).pack(anchor=tk.E, pady=(4, 0))
    def make_slider(self, parent, text, var, lo, hi, fmt, label_w=14, command=None):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text=text, width=label_w).pack(side=tk.LEFT)
        val = ttk.Label(row, text=fmt.format(var.get()), width=6)
        val.pack(side=tk.RIGHT)
        def changed(_=None):
            val.configure(text=fmt.format(var.get()))
            if command:
                command()
        scale = ttk.Scale(row, from_=lo, to=hi, variable=var, orient=tk.HORIZONTAL, command=changed)
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        return scale
    def make_dial(self, parent, var, size=118):
        """Circular version of the plain Windows slider: the thin grey groove of the plain Windows slider bent into a ring, with its blue 11 x 23 thumb (no fill, no animation).
        Drag it, click the ring, scroll (5 deg per notch) or use the arrow keys when it has focus. The number is the compass bearing (0 = north, 90 = east, clockwise); the stored value is still the map angle (0 = east, counter-clockwise)."""
        bg = ttk.Style().lookup("TFrame", "background") or "#f0f0f0"
        cv = tk.Canvas(parent, width=size, height=size, bg=bg, highlightthickness=0, bd=0, takefocus=1)
        c0, R = size / 2.0, size / 2.0 - 14
        try:                                                      # same font as the other slider values (the Windows default UI font), just bigger
            import tkinter.font as tkfont
            num_font = tkfont.nametofont("TkDefaultFont").copy()
            num_font.configure(size=20)
        except Exception:
            num_font = ("Segoe UI", 20)
        def draw(*_a):
            try:
                d = float(var.get()) % 360.0
            except Exception:
                d = 0.0
            cv.delete("all")
            cv.create_oval(c0 - R, c0 - R, c0 + R, c0 + R, outline="#d2d2d2", width=3)       # thin groove: grey edge ...
            cv.create_oval(c0 - R, c0 - R, c0 + R, c0 + R, outline="#e8e8e8", width=1)       # ... light line inside
            a = math.radians(d)
            ux, uy = math.cos(a), -math.sin(a)                       # outward direction on screen
            tx, ty = -uy, ux                                          # along the ring
            kx, ky = c0 + R * ux, c0 + R * uy
            pts = []
            for sr, st in ((1, 1), (1, -1), (-1, -1), (-1, 1)):       # the 11 x 23 px Windows slider thumb, standing across the groove
                pts += [kx + ux * 11.5 * sr + tx * 5.5 * st, ky + uy * 11.5 * sr + ty * 5.5 * st]
            cv.create_polygon(*pts, fill="#0078d7", outline="#0078d7")
            cv.create_text(c0, c0, text=f"{(90.0 - d) % 360.0:.0f}\u00b0", font=num_font, fill="#000000")      # compass bearing: 0 = north, clockwise
        def drag(e):
            deg = math.degrees(math.atan2(c0 - e.y, e.x - c0)) % 360.0
            var.set(float(round(deg)) % 360.0)
            cv.focus_set()
        def step(delta):
            var.set((float(var.get()) - delta) % 360.0)                # +delta = clockwise (bearing goes up)
        cv.bind("<Button-1>", drag)
        cv.bind("<B1-Motion>", drag)
        cv.bind("<MouseWheel>", lambda e: step(5.0 if e.delta > 0 else -5.0))
        for key, dlt in (("<Right>", 1.0), ("<Up>", 1.0), ("<Left>", -1.0), ("<Down>", -1.0), ("<Prior>", 15.0), ("<Next>", -15.0)):
            cv.bind(key, lambda _e, d=dlt: step(d))
        var.trace_add("write", draw)
        draw()
        return cv
    def clear_keylog(self):
        self.key_text.configure(state="normal")
        self.key_text.delete("1.0", "end")
        self.key_text.configure(state="disabled")
    def pump_keylog(self):
        """UI thread: move queued key events into the log box (keeps the last 300 lines)."""
        lines = []
        try:
            while len(lines) < 60:
                lines.append(self.keylog_q.get_nowait())
        except queue.Empty:
            pass
        if not lines:
            return
        t = self.key_text
        t.configure(state="normal")
        t.insert("end", "\n".join(lines) + "\n")
        extra = int(t.index("end-1c").split(".")[0]) - 301
        if extra > 0:
            t.delete("1.0", f"{extra + 1}.0")
        t.configure(state="disabled")
        t.see("end")
    # ------------------------------------------------------------------
    # GRAPHS POPUP: spikes, ping, and the direction (WASD) circle
    # ------------------------------------------------------------------
    def draw_graphs(self):
        if self.graph_win is not None and self.graph_label is not None:
            self.set_image(self.graph_label, self.render_graphs())
    @staticmethod
    def _nice_max(v, floor):
        for c in (1, 2, 5):
            for e in range(-1, 8):
                if c * 10 ** e >= max(v * 1.1, floor):
                    return c * 10 ** e
        return max(v, floor)
    def _time_panel(self, title, ylabel, ts, ys, color, ymax, extra=None, note=""):
        """One white 'oscilloscope' panel: black axes, title on top, coloured line(s) over the last GRAPH_SECONDS."""
        W, H = GRAPH_PW, GRAPH_TH
        img = np.full((H, W, 3), 255, np.uint8)
        x0, x1, y0, y1 = 56, W - 16, 38, H - 42
        cv2.putText(img, title, (x0 + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
        def X(t):
            return int(x0 + (t + GRAPH_SECONDS) / GRAPH_SECONDS * (x1 - x0))
        def Y(v):
            return int(y1 - min(max(v, 0.0), ymax) / ymax * (y1 - y0))
        for k in range(0, int(GRAPH_SECONDS) + 1):          # x ticks
            xx = X(-GRAPH_SECONDS + k)
            cv2.line(img, (xx, y1), (xx, y1 + 4), (0, 0, 0), 1)
            cv2.putText(img, str(-int(GRAPH_SECONDS) + k), (xx - 6, y1 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "s", (x1 - 6, y1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        for k in range(0, 5):                               # y ticks
            v = ymax * k / 4.0
            yy = Y(v)
            cv2.line(img, (x0 - 4, yy), (x0, yy), (0, 0, 0), 1)
            lab = f"{v:.0f}" if ymax >= 10 else f"{v:.1f}"
            cv2.putText(img, lab, (x0 - 8 - 7 * len(lab), yy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, ylabel, (6, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.line(img, (x0, y0), (x0, y1), (0, 0, 0), 1)     # axes
        cv2.line(img, (x0, y1), (x1, y1), (0, 0, 0), 1)
        if extra:
            for (et, ev, ecol, edash, elabel) in extra:
                if not len(et):
                    continue
                pts = np.array([(X(t), Y(v)) for t, v in zip(et, ev)], np.int32)
                if edash:
                    for i in range(0, len(pts) - 1, 2):
                        cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), ecol, 1, cv2.LINE_AA)
                else:
                    cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, ecol, 1, cv2.LINE_AA)
                cv2.putText(img, elabel, (x1 - 8 * len(elabel) - 6, y0 + 14 + 14 * extra.index((et, ev, ecol, edash, elabel))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, ecol, 1, cv2.LINE_AA)
        if len(ts) > 1:
            pts = np.array([(X(t), Y(v)) for t, v in zip(ts, ys)], np.int32)
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)
        if note:
            cv2.putText(img, note, (x0 + 8, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (W - 1, H - 1), (190, 190, 190), 1)
        return img
    def _direction_panel(self, tips, dirs, mag_txt, dead, full):
        """Circular 'direction' graph: the W / A / S / D regions, the dead zone, the motion arrow and its recent trail."""
        W, H = GRAPH_PW, GRAPH_DH
        img = np.full((H, W, 3), 255, np.uint8)
        c = (W // 2, H // 2 + 10)
        R = min(W, H) // 2 - 40
        cv2.putText(img, "Direction (W A S D)", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
        sx, sy = dirs
        active = {"W": sy > 0, "S": sy < 0, "A": sx > 0, "D": sx < 0}
        # sector centres on screen: W up, D right, S down, A left (cv2 angles: 0 = right, clockwise)
        spans = {"D": (-45, 45), "S": (45, 135), "A": (135, 225), "W": (225, 315)}
        soft = {"W": (214, 238, 208), "D": (246, 226, 205), "S": (212, 214, 246), "A": (200, 236, 246)}
        strong = {"W": (120, 210, 110), "D": (235, 170, 90), "S": (120, 130, 240), "A": (70, 200, 240)}
        for k, (a0, a1) in spans.items():
            cv2.ellipse(img, c, (R, R), 0, a0, a1, strong[k] if active[k] else soft[k], -1, cv2.LINE_AA)
        for ang in (45, 135, 225, 315):                      # region borders
            a = math.radians(ang)
            cv2.line(img, c, (int(c[0] + R * math.cos(a)), int(c[1] + R * math.sin(a))), (150, 150, 150), 1, cv2.LINE_AA)
        cv2.circle(img, c, R, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.line(img, (c[0] - R - 8, c[1]), (c[0] + R + 8, c[1]), (0, 0, 0), 1)
        cv2.line(img, (c[0], c[1] - R - 8), (c[0], c[1] + R + 8), (0, 0, 0), 1)
        dr = int(R * min(1.0, dead))
        if dr > 2:
            cv2.circle(img, c, dr, (120, 120, 120), 1, cv2.LINE_AA)
        for k, (lx, ly) in {"W": (c[0] - 7, c[1] - R + 26), "S": (c[0] - 7, c[1] + R - 14), "A": (c[0] - R + 14, c[1] + 6),
                            "D": (c[0] + R - 28, c[1] + 6)}.items():
            cv2.putText(img, k, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0) if active[k] else (110, 110, 110), 2 if active[k] else 1, cv2.LINE_AA)
        for txt, (tx, ty) in (("0", (c[0] - 4, c[1] - R - 12)), ("90", (c[0] + R + 10, c[1] - 6)),
                              ("180", (c[0] - 12, c[1] + R + 20)), ("270", (c[0] - R - 34, c[1] - 6))):
            cv2.putText(img, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "1.0", (c[0] + R - 24, c[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1, cv2.LINE_AA)
        def unit(p):
            m = math.hypot(*p)
            return (p[0] / m, p[1] / m) if m > 1.0 else p
        tips = [unit(t) for t in tips]
        n = len(tips)
        for i, (px, py) in enumerate(tips):                  # trail, oldest = faintest
            q = (int(c[0] + R * max(-1, min(1, px))), int(c[1] - R * max(-1, min(1, py))))
            g = int(235 - 190 * (i + 1) / max(1, n))
            cv2.circle(img, q, 2, (255, g, g), -1, cv2.LINE_AA)
        if tips:
            px, py = tips[-1]
            tip = (int(c[0] + R * max(-1, min(1, px))), int(c[1] - R * max(-1, min(1, py))))
            cv2.arrowedLine(img, c, tip, (200, 60, 20), 2, cv2.LINE_AA, tipLength=0.14)
        cv2.circle(img, c, 3, (0, 0, 0), -1, cv2.LINE_AA)
        keys = "+".join(k for k in "WASD" if active[k]) or "-"
        cv2.putText(img, f"{mag_txt}   keys: {keys}", (12, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (W - 1, H - 1), (190, 190, 190), 1)
        return img
    def _heading_panel(self, heading):
        """Compass-style dial showing the fly's body angle (0 deg = up on the Build map, clockwise)."""
        W, H = GRAPH_PW, GRAPH_AH
        img = np.full((H, W, 3), 255, np.uint8)
        c = (W // 2, H // 2 + 8)
        R = min(W, H) // 2 - 44
        deg = (90.0 - math.degrees(heading)) % 360.0
        cv2.putText(img, "Fly angle (heading)", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(img, c, R, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(img, c, R - 14, (170, 170, 170), 1, cv2.LINE_AA)
        for a in range(0, 360, 10):
            r = math.radians(a - 90)
            ln = 12 if a % 30 == 0 else 6
            p0 = (int(c[0] + (R - ln) * math.cos(r)), int(c[1] + (R - ln) * math.sin(r)))
            p1 = (int(c[0] + R * math.cos(r)), int(c[1] + R * math.sin(r)))
            cv2.line(img, p0, p1, (0, 0, 0), 1, cv2.LINE_AA)
            if a % 30 == 0:
                t = str(a)
                (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
                q = (int(c[0] + (R + 17) * math.cos(r)) - tw // 2, int(c[1] + (R + 17) * math.sin(r)) + th // 2)
                cv2.putText(img, t, q, cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.line(img, (c[0] - R + 14, c[1]), (c[0] + R - 14, c[1]), (200, 200, 200), 1)
        cv2.line(img, (c[0], c[1] - R + 14), (c[0], c[1] + R - 14), (200, 200, 200), 1)
        r = math.radians(deg - 90)
        tip = (int(c[0] + (R - 16) * math.cos(r)), int(c[1] + (R - 16) * math.sin(r)))
        tail = (int(c[0] - 0.35 * R * math.cos(r)), int(c[1] - 0.35 * R * math.sin(r)))
        cv2.line(img, tail, c, (150, 150, 150), 2, cv2.LINE_AA)
        cv2.arrowedLine(img, c, tip, (200, 60, 20), 3, cv2.LINE_AA, tipLength=0.2)
        cv2.circle(img, c, 4, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.putText(img, f"{deg:5.1f} deg", (W - 100, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 60, 20), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (W - 1, H - 1), (190, 190, 190), 1)
        return img
    def render_graphs(self):
        h = list(self.hist)
        tnow = h[-1][0] if h else 0.0
        h = [r for r in h if r[0] >= tnow - GRAPH_SECONDS - 0.05]
        ts = [r[0] - tnow for r in h]
        sp = np.array([r[1] for r in h], np.float32)
        if sp.size >= 4:                                     # ~100 ms smoothing
            sp = np.convolve(np.pad(sp, (3, 0), mode="edge"), np.ones(4, np.float32) / 4, mode="valid")
        smax = self._nice_max(float(np.percentile(sp, 97)) if sp.size else 0.0, 100.0)   # one burst can't squash the graph
        drive = [r[2] * smax * 0.25 for r in h]               # stimulus drawn as a quarter-height trace, like the reference
        g1 = self._time_panel("Spikes vs Time", "spikes/s", ts, sp, (40, 40, 220), smax,
                              extra=[(ts, drive, (60, 170, 40), False, "Stimulus (drive)")],
                              note=f"now {sp[-1]:.0f}/s" if sp.size else "")
        ping = [r[3] for r in h]
        pmax = self._nice_max(float(np.percentile(ping, 97)) if ping else 0.0, 50.0)   # one slow frame can't squash the graph
        g2 = self._time_panel("Ping vs Time", "ms", ts, ping, (40, 40, 220), pmax,
                              extra=[([-GRAPH_SECONDS, 0.0], [33.3, 33.3], (60, 170, 40), True, "33 ms target")],
                              note=(f"now {ping[-1]:.0f} ms   avg {sum(ping) / len(ping):.0f} ms" if ping else ""))
        full = 1.0 if self.cfg.get("source") == "brain" else 0.8
        tips = [(-r[4] / full, r[5] / full) for r in h[-30:]]
        mag = min(1.0, math.hypot(*self.last_motion) / full)
        g3 = self._direction_panel(tips, self.last_dirs, f"|v| {mag:.2f}", float(self.cfg.get("sens", 0.1)) / full, full)
        g4 = self._heading_panel(float(getattr(self.body, 'heading', 0.0)) if getattr(self, 'body', None) is not None else 0.0)
        return np.vstack([g1, g2, g3, g4])
    def toggle_run(self):
        self.user_run = not self.user_run
        self.run_btn.configure(text="Pause" if self.user_run else "Run")
    def reset_brain(self):
        self.sim.reset()
        self.body.reset()
        self.explorer = Explorer()
        self.tracker = OdorTracker()
    def clear_paint(self):
        self.stim_mask[:] = False
        self.sim.set_stim(self.stim_mask)
    def reset_view(self):
        self.yaw, self.pitch, self.zoom = 0.0, 0.0, 1.7      # brain view
        self.body.view_reset()                              # fly view
    def body_drag(self, e):
        lx, ly = getattr(self, "_body_drag", (e.x, e.y))
        self.body.view_orbit(e.x - lx, e.y - ly)
        self._body_drag = (e.x, e.y)
    def cloud_press(self, e):
        self.mouse = (e.x, e.y)
        self._last_drag = (e.x, e.y)
        if self.tool.get() != "rotate":
            self.apply_tool(e.x, e.y)
    def cloud_drag(self, e):
        self.mouse = (e.x, e.y)
        if self.tool.get() == "rotate":
            lx, ly = self._last_drag
            self.yaw += (e.x - lx) * 0.01
            self.pitch = float(np.clip(self.pitch + (e.y - ly) * 0.01, -1.5, 1.5))
        else:
            self.apply_tool(e.x, e.y)
        self._last_drag = (e.x, e.y)
    def cloud_wheel(self, e):
        self.zoom = float(np.clip(self.zoom * (1.1 if e.delta > 0 else 0.9), 0.4, 4.0))
    def apply_tool(self, x, y):
        """Paint or erase the neurons under the brush (front layer only)."""
        sx, sy, z = self._proj
        r = self.brush_var.get()
        cand = np.flatnonzero((sx - x) ** 2 + (sy - y) ** 2 <= r * r)
        if cand.size == 0:
            return
        zc = z[cand]
        cand = cand[zc <= np.quantile(zc, 0.4)]       # nearest 40% in depth = the front layer
        self.stim_mask[cand] = (self.tool.get() == "paint")
        self.sim.set_stim(self.stim_mask)
    def brain_motion(self):
        """Brain drives -> the same (dx, dy) numbers the optical-flow tracker produces.
        dy > 0 = forward (W), dy < 0 = back (S), dx > 0 = left (A), dx < 0 = right (D)."""
        fwd, back, left, right = self.sim.drives(DRIVE_FULL_HZ)
        dy = fwd - back
        dx = left - right
        if abs(dy) < BRAIN_DEADZONE:
            dy = 0.0
        if abs(dx) < BRAIN_DEADZONE:
            dx = 0.0
        if self.cfg.get("directional"):
            # Turn the body-frame push (forward dy, leftward dx) into a map-frame push using the fly's heading
            # (0 = +x / east, counter-clockwise, +y = north): north -> W, south -> S, west -> A, east -> D.
            h = float(getattr(self.body, "heading", 0.0))
            ch, sh = math.cos(h), math.sin(h)
            wx = dy * ch - dx * sh                        # east component
            wy = dy * sh + dx * ch                        # north component
            dy, dx = wy, -wx                              # north = "forward" (W), west = "left" (A)
        return dx, dy
    def render_cloud(self):
        """Density-splat view: every neuron adds a soft glow, coloured by region, over a dark teal background."""
        W, H = NEURON_W, NEURON_H
        net, n = self.net, self.net.n
        P = net.all_pos
        cy, sy_ = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp_ = math.cos(self.pitch), math.sin(self.pitch)
        x = P[:, 0] * cy + P[:, 2] * sy_
        z = -P[:, 0] * sy_ + P[:, 2] * cy
        y = P[:, 1] * cp - z * sp_
        z2 = P[:, 1] * sp_ + z * cp
        scale = 0.42 * min(W, H) * self.zoom
        sx = x * scale + W / 2.0
        sy = -y * scale + H / 2.0
        self._proj = (sx[:n], sy[:n], z2[:n])              # painting only touches the simulated neurons
        ix, iy = sx.astype(np.int32), sy.astype(np.int32)
        ok = (ix >= 2) & (ix < W - 2) & (iy >= 2) & (iy < H - 2)
        flat = (iy * W + ix)

        blur = float(self.blur_var.get())                  # smaller = crisper dots (Sharpness slider)

        def splat(idx, weights=None, sigma=None):
            sigma = blur if sigma is None else sigma
            idx = idx[ok[idx]]
            if idx.size == 0:
                return np.zeros((H, W), np.float32)
            m = np.bincount(flat[idx], weights=weights[idx] if weights is not None else None,
                            minlength=H * W).astype(np.float32).reshape(H, W)
            return m if sigma < 0.05 else cv2.GaussianBlur(m, (0, 0), sigma)

        allidx = np.arange(len(P))
        dens = splat(allidx)
        col = np.zeros((H, W, 3), np.float32)
        for c in range(3):
            col[..., c] = splat(allidx, self.region_cols[:, c])
        col /= np.maximum(dens, 1e-6)[..., None]
        pos_d = dens[dens > 1e-3]
        ref = float(np.percentile(pos_d, 98)) if pos_d.size else 1.0
        alpha = np.clip(dens / max(ref, 1e-6), 0, 1) ** 0.85
        img = BG_BGR * (1 - alpha[..., None]) + col * alpha[..., None]

        def glow(idx, weights, color, gain, sigma=None):
            nonlocal img
            g = np.clip(splat(idx, weights, max(0.6, blur) if sigma is None else sigma) * gain, 0, 1)
            img = img * (1 - g[..., None]) + np.asarray(color, np.float32) * g[..., None]

        simidx = np.arange(n)
        painted = np.flatnonzero(self.stim_mask)
        if painted.size:
            glow(simidx, self.stim_mask.astype(np.float32), MINT_BGR, 2.5)
        act = np.minimum(self.sim.act, 1.0)
        if act.max() > 0.02:
            glow(simidx, act, FIRE_BGR, 2.6, sigma=1.6)                    # slow glow: where the network has been active lately
        hot = np.flatnonzero(self.sim.flash > 0.5)
        if hot.size:
            glow(simidx, (self.sim.flash > 0.5).astype(np.float32), (255, 255, 255), 0.8, sigma=0.7)   # white flecks: spikes right now
        if self.sim.running:                                               # the command pathway: sensory/upstream pool -> descending neurons
            for g in GROUPS:                                               # coloured by the direction it commands
                pool = self.up_pool.get(g)
                if pool is not None and pool.size:
                    wv = np.zeros(n, np.float32)
                    wv[pool] = act[pool]
                    if wv.max() > 0.05:
                        glow(simidx, wv, GROUP_COLORS[g], 9.0, sigma=2.4)
        img = img.clip(0, 255).astype(np.uint8)
        if self.show_skel.get() and len(net.skel_edges):
            S = net.skel_pos
            xs = S[:, 0] * cy + S[:, 2] * sy_
            zs = -S[:, 0] * sy_ + S[:, 2] * cy
            ys = S[:, 1] * cp - zs * sp_
            px = np.clip(xs * scale + W / 2.0, -4000, 4000).astype(np.int32)
            py = np.clip(-ys * scale + H / 2.0, -4000, 4000).astype(np.int32)
            e = net.skel_edges
            segs = np.stack([np.stack([px[e[:, 0]], py[e[:, 0]]], 1), np.stack([px[e[:, 1]], py[e[:, 1]]], 1)], 1)
            for gi in range(5):
                sel = net.skel_grp == gi
                if sel.any():
                    cv2.polylines(img, list(segs[sel]), False,
                                  GROUP_COLORS[GROUPS[gi]] if gi < 4 else (235, 235, 235), 1, cv2.LINE_AA)
        drives = dict(zip(GROUPS, self.sim.drives(DRIVE_FULL_HZ)))
        for g in GROUPS:                                   # DN groups: markers that swell with their firing rate (= what the motors get)
            idx = net.groups[g]
            idx = idx[ok[idx]]
            r = 2 + int(round(3 * drives[g]))
            for px, py in zip(ix[idx], iy[idx]):
                cv2.circle(img, (int(px), int(py)), r, GROUP_COLORS[g], -1, cv2.LINE_AA)
                if drives[g] > 0.15:
                    cv2.circle(img, (int(px), int(py)), r + 1, (255, 255, 255), 1, cv2.LINE_AA)
        if self.mouse is not None and self.tool.get() != "rotate":
            cv2.circle(img, (int(self.mouse[0]), int(self.mouse[1])), int(self.brush_var.get()), (220, 220, 220), 1, cv2.LINE_AA)
        return img
    
    def set_image(self, label, bgr):
        png = to_png(bgr)
        if png is None:
            return
        try:
            img = tk.PhotoImage(data=base64.b64encode(png))
            label.configure(image=img)
            self._imgs[id(label)] = img          # keep a reference so Tk doesn't drop it
        except tk.TclError:
            pass
    def update_lab(self):
        """UI thread, ~30 fps: push slider values to the brain, move the body, redraw both views."""
        if self.terminating:
            return
        self.pump_keylog()
        now = time.perf_counter()
        t_start = now
        dt = min(now - self.lab_t, 0.1)
        self.lab_t = now
        self.sim.gain = float(self.gain_var.get())
        self.sim.stim_rate = float(self.rate_var.get())
        brain_selected = self.source.get() == "brain"
        self.sim.running = self.user_run and brain_selected      # paused unless Run is on AND the Brain source is selected
        mode = self.preset_var.get()
        dynamic = mode == DYN_LABEL
        feeding = False
        plume_on = self.smell_model.get() == SMELL_PLUME
        if plume_on:
            wspd = float(self.wind_speed_var.get())
            if self.S("wind_mode") == "Gusting":
                wspd *= max(0.0, 1.0 + 0.6 * math.sin(now * 0.7) * math.sin(now * 0.23 + 1.0))
            self.arena.plume.update(dt, self.arena.foods, wspd, float(self.wind_dir_var.get()), turb=float(self.S("turb")),
                                    obstacles=self.arena.obstacles if self.S("smoke_obst") else ())
        if self.sim.running and dynamic:
            feeding = self.arena.feed(self.body.x, self.body.y, dt)
        cl = cr = 0.0                                      # odour at the two antennae (plume + any test pulse from the Miscellaneous page)
        if plume_on:
            cl, cr = (float(v) for v in self.arena.plume.conc(self.arena.antennae(self.body.x, self.body.y, self.body.heading)))
        t_now = time.perf_counter()
        if self.odor_pulse["l"] > t_now:
            cl += ODOR_PULSE
        if self.odor_pulse["r"] > t_now:
            cr += ODOR_PULSE
        self.odor_now = (cl, cr)
        self.odor_hist.append((t_now, cl, cr))
        if self.sim.running:
            ext = np.zeros(self.net.n, np.float32)
            if dynamic:                                    # wander + react to food / obstacles
                sense = self.arena.sense(self.body.x, self.body.y, self.body.heading)
                if plume_on or cl > 0.0 or cr > 0.0:       # smell = the plume (or a test pulse): antennae sample the odour, the tracker steers
                    near = self.arena.near_food(self.body.x, self.body.y, self.body.heading)
                    sense["food"] = self.tracker.update(dt, float(cl), float(cr), self.arena.plume.wind_dir, self.body.heading, near)
                lvl, bk, lt, rt = env_commands(sense, self.explorer.update(dt))
                # stuck-escape: if the fly has hardly moved for a few seconds (pinned at a wall / obstacle), turn away and walk off
                self._stuck_clock = getattr(self, "_stuck_clock", 0.0) + dt
                if self._stuck_clock >= 2.0:
                    p0 = getattr(self, "_stuck_pos", (self.body.x, self.body.y))
                    if math.hypot(self.body.x - p0[0], self.body.y - p0[1]) < 1.5 and not feeding:
                        self._escape_t = 1.6
                        self._escape_turn = random.choice((-1.0, 1.0))
                    self._stuck_pos = (self.body.x, self.body.y)
                    self._stuck_clock = 0.0
                if getattr(self, "_escape_t", 0.0) > 0.0:
                    self._escape_t -= dt
                    lvl, bk, lt, rt = 0.85, 0.0, max(0.0, self._escape_turn), max(0.0, -self._escape_turn)
                for key, val in (("fwd", lvl), ("back", bk), ("left", lt), ("right", rt)):
                    # the command enters the network through the upstream (premotor) pool and reaches the descending neurons
                    # through the real synapses; only 'back' also gets a direct push because its pool barely reaches its DNs
                    ext[self.up_pool[key]] = np.maximum(ext[self.up_pool[key]], UP_RATE_HZ * val / max(0.5, self.sim.gain) ** UP_GAIN_POW)
                    if key == "back":
                        ext[self.net.groups[key]] = ENV_RATE_HZ * val
            elif mode != BASE_LABEL:                       # a BadApple-fly movement program: stimulate its neurons
                sides = {"Both": ("L", "R"), "Left": ("L",), "Right": ("R",)}[self.side_var.get()]
                for sd in sides:
                    idx = self.net.presets.get(self.preset_ids.get(mode), {}).get(sd)
                    if idx is not None:
                        ext[idx] = float(self.rate_var.get())
            if feeding:                                    # eating: the proboscis motor neurons (MN9) are driven
                for idx in self.net.presets.get("feeding", {}).values():
                    ext[idx] = ENV_RATE_HZ
            self.sim.set_ext(ext)
        fwd, back, left, right = self.sim.drives(DRIVE_FULL_HZ)
        if self.sim.running:
            if feeding:
                fwd = back = left = right = 0.0            # stand still while eating
            gt = max([float(self.sim.flash[a].mean()) for a in self.groom_sets] or [0.0])
            self.body.groom += (min(1.0, gt * 1.6) - self.body.groom) * min(1.0, dt * 6.0)
            if self.feed_set.size:
                ft = min(1.0, float(self.sim.flash[self.feed_set].mean()) * 1.6)
            else:
                ft = 1.0 if feeding else 0.0               # (old brain file without MN9: extend while eating)
            self.body.feed += (ft - self.body.feed) * min(1.0, dt * 5.0)
            try:
                bs = float(self.body_speed_var.get().rstrip("x"))
            except Exception:
                bs = 1.0
            self.body.step(fwd, back, left, right, dt * bs)
        eaten = self.arena.eaten
        if eaten > self.food_seen and self.food_key and self.cfg["walk"]:
            threading.Thread(target=self.tap, args=(self.food_key,), daemon=True).start()   # one tap per pellet eaten
        self.food_seen = eaten
        self.frame_i = getattr(self, "frame_i", 0) + 1
        try:
            sel = self.nb.select()                         # draw only the page that is open (the others cost time and are invisible)
        except Exception:
            sel = str(self.tab_ctrl)
        if self.frame_i % 3 == 0:
            self.update_arena_view()
        prime = not getattr(self, "_primed", False)            # first frame: draw every page once so the window knows its full size
        self._primed = True
        on_ctrl = sel == str(self.tab_ctrl) or not self.S("only_open")
        if prime or (on_ctrl and self.S("draw_body")):
            self.set_image(self.body_label, self.body.render(VIEW_W, VIEW_H))
        if prime or (on_ctrl and self.S("draw_cloud")):
            self.set_image(self.cloud_label, self.render_cloud())
        self.status.set(
            f"t {self.sim.t_ms / 1000:6.1f}s  speed {self.sim.speed:4.2f}x  "
            f"spikes/s {self.sim.spikes_per_s:7.0f}  painted {int(self.stim_mask.sum())}\n"
            f"drive  fwd {fwd:.2f}  back {back:.2f}  left {left:.2f}  right {right:.2f}   food eaten {self.arena.eaten}"
            + ("" if self.sim.running else ("\nPAUSED - press Run" if brain_selected else "\nPAUSED - choose 'Brain simulation' as the source (Fly Bridge section)")))
        n_spk = self.sim.n_spk
        rate = max(0, n_spk - self._spk_prev) / max(dt, 0.01) if (self.sim.running and self.hist) else 0.0
        self._spk_prev = n_spk
        self.hist.append((now, rate, max(fwd, back, left, right), (time.perf_counter() - t_start) * 1000.0,
                          self.last_motion[0], self.last_motion[1]))
        if self.graph_win is not None and ((self.frame_i % 2 == 0 and on_ctrl and self.S("draw_graphs")) or prime):
            self.draw_graphs()
        if self.frame_i % 3 == 2:
            try:
                if self.nb.select() == str(self.tab_vision):
                    self.draw_vision()
            except Exception as err:
                print("[!] Vision page:", err)
        if self.frame_i % 4 == 0:
            try:
                if self.nb.select() == str(self.tab_motor):         # only draw it while the page is open
                    self.draw_motors()
            except Exception as err:
                if not getattr(self, "_motor_err", False):
                    self._motor_err = True
                    print("[!] Motor Properties page:", err)
        self._fps = 0.9 * getattr(self, "_fps", 20.0) + 0.1 / max(dt, 0.001)
        if self.S("show_fps"):
            self.status.set(self.status.get() + f"\nfps {self._fps:4.1f}   frame {(time.perf_counter() - t_start) * 1000.0:4.0f} ms")
        if self.S("print_timings") and self.frame_i % max(1, int(2000 / max(1, getattr(self, "frame_ms", 33)))) == 0:
            print(f"[timing] fps {self._fps:4.1f}  frame {(time.perf_counter() - t_start) * 1000.0:5.1f} ms  filaments {len(self.arena.plume.age)}")
        self.root.after(getattr(self, "frame_ms", 33), self.update_lab)
    # ------------------------------------------------------------------
    # BUILD TAB: place obstacles and food for the fly
    # ------------------------------------------------------------------
    def build_arena_tab(self, parent):
        parent.configure(padding=12)
        side = ttk.Frame(parent)
        side.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        ttk.Label(side, text="Build the fly's world", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        ttk.Label(side, wraplength=190, justify=tk.LEFT, text=(
            "Click on the map to place things. In 'Dynamic environment' mode the fly wanders on its own, "
            "smells food, steers around obstacles and walls, and stops to eat.")).pack(anchor=tk.W, pady=(4, 10))
        self.build_tool = tk.StringVar(value="obstacle")
        for text, val in (("Place obstacle", "obstacle"), ("Place food", "food"), ("Erase", "erase"), ("Move fly here", "fly")):
            ttk.Radiobutton(side, text=text, variable=self.build_tool, value=val).pack(anchor=tk.W, pady=1)
        self.obst_r_var = tk.DoubleVar(value=4.0)
        self.make_slider(side, "Obstacle size (mm)", self.obst_r_var, 1.5, 9.0, "{:.1f}", label_w=17)
        self.smell_model = tk.StringVar(value=SMELL_PLUME)
        ttk.Label(side, text="Smell model").pack(anchor=tk.W, pady=(6, 0))
        mb = ttk.Menubutton(side, textvariable=self.smell_model, width=24)
        mn = tk.Menu(mb, tearoff=0)
        mn.add_radiobutton(label=SMELL_PLUME, variable=self.smell_model, value=SMELL_PLUME)
        mn.add_radiobutton(label=SMELL_RADIUS + " - not available", variable=self.smell_model, value=SMELL_RADIUS, state="disabled")
        mb["menu"] = mn
        mb.pack(anchor=tk.W)
        self.wind_speed_var = tk.DoubleVar(value=8.0)
        self.wind_dir_var = tk.DoubleVar(value=90.0)           # default: the knob (and the wind) points north
        self.make_slider(side, "Wind speed (mm/s)", self.wind_speed_var, 0.0, 25.0, "{:.0f}", label_w=17)
        drow = ttk.Frame(side)
        drow.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(drow, text="Wind direction", width=17).pack(side=tk.LEFT, anchor=tk.N, pady=(6, 0))
        self.make_dial(drow, self.wind_dir_var).pack(side=tk.RIGHT)
        self.smell_var = tk.DoubleVar(value=self.arena.smell_range)      # (no slider any more: the old radius mode keeps its default reach)
        ttk.Button(side, text="Refill food", command=self.arena.refill).pack(fill=tk.X, pady=(12, 2))
        ttk.Button(side, text="Reset fly", command=self.reset_fly).pack(fill=tk.X, pady=2)
        ttk.Button(side, text="Clear everything", command=self.arena.clear).pack(fill=tk.X, pady=2)
        ttk.Button(side, text="Default layout", command=self.arena.default_layout).pack(fill=tk.X, pady=2)
        self.arena_info = tk.StringVar(value="")
        ttk.Label(side, textvariable=self.arena_info, justify=tk.LEFT).pack(anchor=tk.W, pady=(12, 0))
        panel_bg = ttk.Style().lookup("TFrame", "background") or "#f0f0f0"
        # bottom: live wind compass (black = direction, blue = velocity), fills the free space
        self.wind_canvas = tk.Canvas(side, width=270, height=184, bg=panel_bg, highlightthickness=0, bd=0)
        self.wind_canvas.pack(side=tk.BOTTOM, anchor=tk.W, pady=(8, 0))
        # column between the sidebar and the arena: Individualized Simulation = close-up of the live odour plume with numbers
        lf = ttk.LabelFrame(parent, text="Individualized Simulation")
        lf.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        self.wsim_label = tk.Label(lf, bd=0, bg="#10181a")
        self.wsim_label.pack(padx=6, pady=(2, 4))
        self.wsim_paused = True                                       # starts empty and paused
        self.wsim_label.bind("<Button-1>", self.wsim_click)         # the pause / clear icons are drawn into the picture itself
        self.wsim_label.bind("<Motion>", self.wsim_hover)
        self.wsim_info = tk.StringVar(value="")
        ttk.Label(lf, textvariable=self.wsim_info, justify=tk.LEFT, font=("Consolas", 9)).pack(anchor=tk.W, padx=8, pady=(0, 6))
        self.arena_px = ARENA_PX
        holder = ttk.Frame(parent)                         # the arena grows to fill all the free space, staying square
        holder.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.arena_canvas = tk.Canvas(holder, width=ARENA_PX, height=ARENA_PX, bg="#1d2a2c", highlightthickness=1,
                                      highlightbackground="#888888", cursor="crosshair")
        self.arena_canvas.place(relx=0.5, rely=0.5, anchor="center", width=ARENA_PX, height=ARENA_PX)
        holder.bind("<Configure>", self.arena_resize)
        self.arena_canvas.bind("<ButtonPress-1>", self.arena_click)
        self.arena_canvas.bind("<B1-Motion>", lambda e: self.arena_click(e, drag=True))
        self._arena_ver = -1
        self.trail = []
        self.update_arena_view()
    def smell_overlay(self, sc):
        """Translucent smell circles for every food: 70% transparent fill and a 3 px border that is 40% transparent."""
        n = int(self.arena_px)
        r = self.arena.smell_range * sc
        pts = tuple((int(round(self._w2c(fx, fy)[0])), int(round(self._w2c(fx, fy)[1]))) for fx, fy, _a in self.arena.foods)
        key = (n, round(r, 1), pts)
        if getattr(self, "_smell_key", None) == key and self._smell_img is not None:
            return self._smell_img
        fill = np.zeros((n, n), np.uint8)
        ring = np.zeros((n, n), np.uint8)
        for cx, cy in pts:
            cv2.circle(fill, (cx, cy), int(round(r)), 255, -1, cv2.LINE_AA)
            cv2.circle(ring, (cx, cy), int(round(r)), 255, 3, cv2.LINE_AA)
        af = fill.astype(np.float32) / 255.0 * 0.30           # fill opacity 0.30  (= 0.7 transparent)
        ab = ring.astype(np.float32) / 255.0 * 0.60           # border opacity 0.60 (= 0.4 transparent)
        cf = np.array([31, 140, 240], np.float32)             # orange (BGR)
        cb = np.array([110, 190, 255], np.float32)            # lighter orange for the border
        a_out = ab + af * (1.0 - ab)
        col = (cb * ab[..., None] + cf * (af * (1.0 - ab))[..., None]) / np.maximum(a_out, 1e-6)[..., None]
        bgra = np.dstack([col, a_out * 255.0]).clip(0, 255).astype(np.uint8)
        png = to_png(bgra)
        if png is None:
            return None
        try:
            self._smell_img = tk.PhotoImage(data=base64.b64encode(png))
            self._smell_key = key
        except tk.TclError:
            self._smell_img = None
        return self._smell_img
    def on_smell(self):
        self.arena.smell_range = float(self.smell_var.get())
        self.update_arena_view()
    def set_smell_state(self, hover=None, drag=None):
        """The smell circles show while the pointer is over the slider or the slider is being dragged."""
        if hover is not None:
            self._smell_hover = hover
        if drag is not None:
            self._smell_drag = drag
        self.smell_show = self._smell_hover or self._smell_drag
        self.update_arena_view()
    def arena_resize(self, e):
        n = int(max(240, min(e.width, e.height) - 12))
        if n == getattr(self, "arena_px", 0):
            return
        self.arena_px = n
        self.arena_canvas.place(relx=0.5, rely=0.5, anchor="center", width=n, height=n)
        self._arena_ver = -1                               # redraw obstacles / food / grid at the new scale
        if hasattr(self, "trail"):
            self.update_arena_view()
    def reset_fly(self):
        self.body.reset()
        self.trail = []
    def _w2c(self, x, y):
        sc = self.arena_px / (2 * ARENA_HALF)
        return self.arena_px / 2 + x * sc, self.arena_px / 2 - y * sc
    def arena_click(self, e, drag=False):
        sc = self.arena_px / (2 * ARENA_HALF)
        x, y = (e.x - self.arena_px / 2) / sc, (self.arena_px / 2 - e.y) / sc
        if abs(x) > ARENA_HALF or abs(y) > ARENA_HALF:
            return
        if self.S("snap"):
            x, y = round(x / 5.0) * 5.0, round(y / 5.0) * 5.0
        tool = self.build_tool.get()
        if tool == "erase":
            self.arena.erase(x, y, 2.5)
        elif drag:
            return
        elif tool == "obstacle":
            self.arena.add_obstacle(x, y, float(self.obst_r_var.get()))
        elif tool == "food":
            self.arena.add_food(x, y)
        elif tool == "fly":
            rx, ry = self.arena.resolve(x, y)
            self.body.teleport(rx, ry)
            self.trail = []
    def draw_gauges(self):
        """Wind compass on the Build tab: small black arrow = wind direction, thinner blue arrow = velocity (length scales with speed)."""
        wc = getattr(self, "wind_canvas", None)
        if wc is None:
            return
        wc.delete("all")
        speed = float(self.wind_speed_var.get()) if hasattr(self, "wind_speed_var") else 0.0
        th = getattr(self.arena.plume, "wind_dir", 0.0)
        ox, oy, r = 186, 98, 64
        wc.create_text(4, 4, anchor="nw", text="Wind", font=("Arial", 10, "bold"))
        wc.create_text(4, 20, anchor="nw", text=f"{speed:.0f} mm/s   {(90.0 - math.degrees(th)) % 360:.0f} deg", font=("Arial", 9, "bold"))
        wc.create_text(4, 35, anchor="nw", text="black = direction\nblue = velocity", font=("Arial", 8), fill="#555555")
        wc.create_oval(ox - r, oy - r, ox + r, oy + r, outline="#888888", width=2)
        wc.create_oval(ox - r * 0.5, oy - r * 0.5, ox + r * 0.5, oy + r * 0.5, outline="#cccccc")
        for k in range(0, 360, 30):
            a = math.radians(k)
            l = 9 if k % 90 == 0 else 5
            wc.create_line(ox + (r - l) * math.cos(a), oy - (r - l) * math.sin(a), ox + r * math.cos(a), oy - r * math.sin(a), fill="#888888")
        for lab, dx, dy in (("N", 0, -1), ("E", 1, 0), ("S", 0, 1), ("W", -1, 0)):
            wc.create_text(ox + dx * (r + 14), oy + dy * (r + 12), text=lab, font=("Arial", 10, "bold"), fill="#555555")
        c, sn = math.cos(th), math.sin(th)          # arrows point where the wind blows TO (map: +y is up)
        L = r * 0.95 * min(1.0, speed / 25.0)
        if L > 2:
            wc.create_line(ox - c * 6, oy + sn * 6, ox + c * L, oy - sn * L, fill="#1f6fb2", width=2, arrow=tk.LAST, arrowshape=(10, 12, 4))
        wc.create_line(ox - c * 14, oy + sn * 14, ox + c * 30, oy - sn * 30, fill="#000000", width=4, arrow=tk.LAST, arrowshape=(10, 12, 5))
        wc.create_oval(ox - 3, oy - 3, ox + 3, oy + 3, fill="#000000")
        self._wk = getattr(self, "_wk", 0) + 1
        if self._wk % 2 == 0:
            self.draw_wind_sim()
    def wsim_toggle(self):
        self.wsim_paused = not self.wsim_paused
        self._wsim_force = True
    def _wsim_hit(self, e):
        W, H = 300, getattr(self, "_wsim_H", 520)
        if H - 40 <= e.y <= H - 8:
            if W - 62 <= e.x <= W - 38:
                return "pause"
            if W - 32 <= e.x <= W - 8:
                return "clear"
        return None
    def wsim_click(self, e):
        hit = self._wsim_hit(e)
        if hit == "pause":
            self.wsim_toggle()
        elif hit == "clear":
            self.wsim_clear()
    def wsim_hover(self, e):
        try:
            self.wsim_label.configure(cursor="hand2" if self._wsim_hit(e) else "")
        except Exception:
            pass
    def wsim_clear(self):
        wp = getattr(self, "_wp", None)
        if wp is not None:
            wp.clear()
        self._wsim_force = True
    def draw_wind_sim(self):
        """Individualized Simulation: an 'item viewer' for ONE odour source. A single pellet emits into a tall wind tunnel (wind drawn blowing down the screen
        at the real wind speed), so you can see how the fluid carries, snakes and thins the odour before placing food on the map."""
        lab = getattr(self, "wsim_label", None)
        if lab is None:
            return
        W, XH = 300, 30.0                                      # image width px, half-width of the view in mm  (5 px per mm)
        PX = W / (2 * XH)
        H = 520
        try:
            hh = int(lab.master.winfo_height())
            if hh > 300:
                H = int(min(800, max(300, hh - 225)))
        except Exception:
            pass
        paused = getattr(self, "wsim_paused", False)
        if paused and not getattr(self, "_wsim_force", False) and getattr(self, "_wsim_drawn", False) and H == getattr(self, "_wsim_H", None):
            return                                              # frozen: keep the last picture (but still follow window resizes)
        YH = XH * H / W                                        # half-height of the view in mm (<= 80)
        sp = float(self.wind_speed_var.get()) if hasattr(self, "wind_speed_var") else 0.0
        paused = getattr(self, "wsim_paused", False)
        self._wsim_force = False
        self._wsim_drawn = True
        ys = YH - 6.0                                          # the source sits near the top; the wind blows downwards
        if abs(getattr(self, "_wp_ys", ys) - ys) > 0.5 and getattr(self, "_wp", None) is not None:
            self._wp.clear()                                    # window was resized: the source moved, restart the emission
        self._wp_ys = ys
        wp = getattr(self, "_wp", None)
        if wp is None or abs(wp.half - 80.0) > 1e-6:
            wp = self._wp = Plume(80.0)
            wp.rng = np.random.default_rng(11)
            self._wp_t = time.monotonic()
        now = time.monotonic()
        dt = min(0.2, max(0.0, now - getattr(self, "_wp_t", now)))
        self._wp_t = now
        if not paused:
            for _ in range(2):
                wp.update(dt / 2.0, [(0.0, ys, 1.0)], sp, 270.0)
        gx, gy = int(2 * XH / 0.5), int(2 * YH / 0.5)
        img = np.zeros((gy, gx), np.float32)
        if len(wp.age):
            w = 0.25 * np.exp(-wp.age / 18.0)
            Hh, _, _ = np.histogram2d(wp.pos[:, 1], wp.pos[:, 0], bins=[gy, gx], range=[[-YH, YH], [-XH, XH]], weights=w)
            img = cv2.GaussianBlur(Hh[::-1].astype(np.float32), (0, 0), 0.7)
        v8 = (np.clip(1.0 - np.exp(-2.2 * img), 0, 1) * 255).astype(np.uint8)
        big = cv2.resize(cv2.applyColorMap(v8, cv2.COLORMAP_INFERNO), (W, H), interpolation=cv2.INTER_LINEAR)
        big[cv2.resize(v8, (W, H), interpolation=cv2.INTER_NEAREST) < 4] = (26, 24, 16)
        px = lambda x, y: (int((x + XH) * PX), int((YH - y) * PX))
        for d in range(10, int(2 * YH), 10):                    # distance-from-source rings every 10 mm
            y = ys - d
            if y < -YH:
                break
            cv2.line(big, px(-XH, y), px(XH, y), (60, 52, 40), 1)
            if px(0, y)[1] > H - 48:
                continue                                        # keep the bottom-left corner free for the scale bar
            cv2.putText(big, f"{d} mm", (4, px(0, y)[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170, 170, 170), 1, cv2.LINE_AA)
        cv2.line(big, px(0, YH), px(0, -YH), (60, 52, 40), 1)
        cv2.circle(big, px(0, ys), 6, (31, 140, 240), -1, cv2.LINE_AA)
        cv2.circle(big, px(0, ys), 9, (110, 190, 255), 1, cv2.LINE_AA)
        cv2.putText(big, "source", (px(0, ys)[0] + 13, px(0, ys)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 190, 255), 1, cv2.LINE_AA)
        self._wsim_H = H
        ic = (235, 238, 240)                                    # icons drawn straight into the picture: pause / play and clear
        cy0 = H - 24
        px0 = W - 50
        if paused:
            cv2.fillPoly(big, [np.array([[px0 - 5, cy0 - 7], [px0 - 5, cy0 + 7], [px0 + 7, cy0]], np.int32)], ic, cv2.LINE_AA)
        else:
            cv2.rectangle(big, (px0 - 5, cy0 - 7), (px0 - 2, cy0 + 7), ic, -1)
            cv2.rectangle(big, (px0 + 2, cy0 - 7), (px0 + 5, cy0 + 7), ic, -1)
        cx0 = W - 20
        cv2.line(big, (cx0 - 6, cy0 - 6), (cx0 + 6, cy0 + 6), ic, 2, cv2.LINE_AA)
        cv2.line(big, (cx0 - 6, cy0 + 6), (cx0 + 6, cy0 - 6), ic, 2, cv2.LINE_AA)
        bar = int(10 * PX)
        cv2.rectangle(big, (10, H - 18), (10 + bar, H - 13), (255, 255, 255), -1)
        cv2.putText(big, "10 mm", (10, H - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
        self.set_image(lab, big)
        thr = OdorTracker.THR
        c = wp.conc([(0.0, ys - d) for d in (10.0, 20.0, 40.0)])
        age = float(wp.age.mean()) if len(wp.age) else 0.0
        reach = float(ys - wp.pos[:, 1].min()) if len(wp.age) else 0.0
        sel = wp.pos[(wp.pos[:, 1] < ys - 20) & (wp.pos[:, 1] > ys - 30)] if len(wp.age) else np.zeros((0, 2))
        wid = float(sel[:, 0].std() * 2) if len(sel) > 5 else 0.0
        mark = lambda x: "smelled" if x > thr else "faint"
        self.wsim_info.set(
            f"wind speed   {sp:5.1f} mm/s  (shown downward)\n"
            f"map wind dir {(90.0 - (float(self.wind_dir_var.get()) if hasattr(self, 'wind_dir_var') else 0.0)) % 360:5.0f} deg (0 = N)\n"
            f"filaments    {len(wp.age):5d}   mean age {age:4.1f} s\n"
            f"plume reach  {reach:5.1f} mm   width@25mm {wid:4.1f} mm\n"
            f"odour  10 mm {c[0]:5.3f}  {mark(c[0])}\n"
            f"odour  20 mm {c[1]:5.3f}  {mark(c[1])}\n"
            f"odour  40 mm {c[2]:5.3f}  {mark(c[2])}\n"
            f"detect threshold {thr:.2f}")
    def plume_overlay(self):
        """Grey smoke picture of the odour plume, drawn over the map (like a smoke-plume visualisation)."""
        n = int(self.arena_px)
        d = self.arena.plume.density(150)
        d = cv2.resize(d, (n, n), interpolation=cv2.INTER_CUBIC)
        a = np.clip(d, 0, 1) ** 0.8 * 235.0
        col = np.empty((n, n, 3), np.float32)
        col[:] = (232, 234, 236)
        bgra = np.dstack([col, a]).clip(0, 255).astype(np.uint8)
        png = to_png(bgra)
        if png is None:
            return None
        try:
            self._plume_img = tk.PhotoImage(data=base64.b64encode(png))
        except tk.TclError:
            self._plume_img = None
        return self._plume_img
    def update_arena_view(self):
        cv = self.arena_canvas
        sc = self.arena_px / (2 * ARENA_HALF)
        if self._arena_ver != self.arena.version:
            self._arena_ver = self.arena.version
            cv.delete("static")
            for g in (range(-30, 40, 10) if self.S("show_grid") else ()):
                a, _ = self._w2c(g, 0)
                _, b = self._w2c(0, g)
                cv.create_line(a, 0, a, self.arena_px, fill="#27393b", tags="static")
                cv.create_line(0, b, self.arena_px, b, fill="#27393b", tags="static")
            cv.create_rectangle(2, 2, self.arena_px - 2, self.arena_px - 2, outline="#8a9aa0", width=3, tags="static")
            for ox, oy, r in self.arena.obstacles:
                cx, cy = self._w2c(ox, oy)
                cv.create_oval(cx - r * sc, cy - r * sc, cx + r * sc, cy + r * sc, fill="#7d8aa6", outline="#a9b4cc", tags="static")
            for fx, fy, amt in self.arena.foods:
                cx, cy = self._w2c(fx, fy)
                rr = 2.0 + 3.5 * max(0.0, min(1.0, amt))
                cv.create_oval(cx - rr, cy - rr, cx + rr, cy + rr, fill="#f08c1f", outline="#ffc27a", tags="static")
            cv.tag_lower("static")
        b = self.body
        if not self.trail or math.hypot(b.x - self.trail[-1][0], b.y - self.trail[-1][1]) > 0.6:
            self.trail.append((b.x, b.y))
            self.trail = self.trail[-250:]
        try:
            on_build = self.nb.select() == str(self.tab_build)
        except Exception:
            on_build = True
        if not on_build and getattr(self, "_arena_primed", False):
            return                                          # map page is hidden: skip all drawing
        self._arena_primed = True
        cv.delete("smell")
        cv.delete("plume")
        if on_build and self.S("draw_smoke") and self.smell_model.get() == SMELL_PLUME and len(self.arena.plume.age):
            img = self.plume_overlay()
            if img is not None:
                cv.create_image(0, 0, anchor="nw", image=img, tags="plume")
                cv.tag_lower("plume")
                cv.tag_lower("static")
        if self.smell_show and self.arena.foods and self.smell_model.get() == SMELL_RADIUS:
            img = self.smell_overlay(sc)
            if img is not None:
                cv.create_image(0, 0, anchor="nw", image=img, tags="smell")
        cv.delete("fly")
        if len(self.trail) > 1 and self.S("show_trail"):
            pts = [c for x, y in self.trail for c in self._w2c(x, y)]
            cv.create_line(*pts, fill="#e5533d", width=1, tags="fly")
        cx, cy = self._w2c(b.x, b.y)
        ux, uy = math.cos(b.heading), -math.sin(b.heading)          # heading on screen (+y is up on the map)
        px, py = -uy, ux
        L, Wd = 3.2 * sc, 1.5 * sc
        cv.create_polygon(cx + ux * L, cy + uy * L, cx - ux * L * 0.6 + px * Wd, cy - uy * L * 0.6 + py * Wd,
                          cx - ux * L * 0.6 - px * Wd, cy - uy * L * 0.6 - py * Wd, fill="#e7b44c", outline="#fff3d0", tags="fly")
        self.draw_gauges()
        self.arena_info.set(f"Food eaten: {self.arena.eaten}\nFood left: {len(self.arena.foods)}\nObstacles: {len(self.arena.obstacles)}\n"
                            f"Fly at x {b.x:5.1f}  y {b.y:5.1f} mm")
    # ------------------------------------------------------------------
    # MALECNS SETUP TAB
    # ------------------------------------------------------------------
    def build_setup(self, parent):
        parent.configure(padding=15)
        self.setup_q = queue.Queue()
        self.setup_busy = False
        ttk.Label(parent, text="Build the real MaleCNS brain (one time)", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        ttk.Label(parent, justify=tk.LEFT, wraplength=900, text=(
            "Your downloaded MaleCNS files are used directly: leave the token EMPTY and point Weights file at connections.feather "
            "(annotations.feather and neurotransmitters.feather next to it are read automatically). A token is only needed for skeleton shapes. "
            "Press Build. When it finishes, the new brain "
            "loads straight into the live simulation.   'Context neurons' only gives the view its anatomy shape (they are drawn, "
            "not simulated).   Optional: 'Align to JRC2018M' moves positions into the standard male brain template with navis "
            "(press 'Download template transforms' once first); it keeps only the brain, so leave it off to see the nerve cord too.")).pack(anchor=tk.W, pady=(4, 8))
        cmd = ttk.Entry(parent)
        cmd.insert(0, DOWNLOAD_CMD)
        cmd.configure(state="readonly")
        cmd.pack(fill=tk.X)
        form = ttk.LabelFrame(parent, text=" Settings ", padding=10)
        form.pack(fill=tk.X, pady=10)
        form.columnconfigure(1, weight=1)
        self.weights_var = tk.StringVar(value=DEFAULT_WEIGHTS)
        self.token_var = tk.StringVar()
        self.n_var = tk.IntVar(value=8000)
        self.align_var = tk.BooleanVar(value=False)
        self.ctx_var = tk.IntVar(value=40000)
        self.shapes_var = tk.IntVar(value=24)
        self.shape_types_var = tk.StringVar(value="DNge104")
        ttk.Label(form, text="Weights file").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(form, textvariable=self.weights_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(form, text="Browse...", command=self.browse_weights).grid(row=0, column=2)
        ttk.Label(form, text="neuPrint token").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(form, textvariable=self.token_var, show="*").grid(row=1, column=1, sticky="ew", padx=6)
        ttk.Label(form, text="Neurons to keep").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Spinbox(form, from_=1000, to=20000, increment=1000, textvariable=self.n_var, width=8).grid(row=2, column=1, sticky="w", padx=6)
        ttk.Label(form, text="Context neurons").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Spinbox(form, from_=0, to=100000, increment=10000, textvariable=self.ctx_var, width=8).grid(row=3, column=1, sticky="w", padx=6)
        ttk.Label(form, text="Shape neurons").grid(row=4, column=0, sticky="w", pady=3)
        ttk.Spinbox(form, from_=0, to=200, increment=4, textvariable=self.shapes_var, width=8).grid(row=4, column=1, sticky="w", padx=6)
        ttk.Label(form, text="Extra shape types").grid(row=5, column=0, sticky="w", pady=3)
        ttk.Entry(form, textvariable=self.shape_types_var).grid(row=5, column=1, sticky="ew", padx=6)
        ttk.Checkbutton(form, text="Align to JRC2018M template (navis + flybrains) - brain only, drops the nerve cord", variable=self.align_var).grid(row=6, column=1, sticky="w", padx=6, pady=3)
        btns = ttk.Frame(parent)
        btns.pack(fill=tk.X)
        self.build_btn = ttk.Button(btns, text="Build brain_subset.npz", command=lambda: self.start_setup("build"))
        self.build_btn.pack(side=tk.LEFT)
        self.list_btn = ttk.Button(btns, text="List DN type names", command=lambda: self.start_setup("list"))
        self.list_btn.pack(side=tk.LEFT, padx=6)
        self.xform_btn = ttk.Button(btns, text="Download template transforms", command=lambda: self.start_setup("transforms"))
        self.xform_btn.pack(side=tk.LEFT)
        ttk.Button(btns, text="Reload brain now", command=self.load_brain).pack(side=tk.LEFT, padx=6)
        logbox = ttk.Frame(parent)
        logbox.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.setup_text = tk.Text(logbox, height=16, width=110, font=("Consolas", 9), state="disabled", wrap="word")
        scroll = ttk.Scrollbar(logbox, command=self.setup_text.yview)
        self.setup_text.configure(yscrollcommand=scroll.set)
        self.setup_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.append_log("Ready. Until you build the real data, the Control Panel uses the demo network.")
        self.poll_setup()
    def browse_weights(self):
        p = filedialog.askopenfilename(title="Pick the MaleCNS weights .feather file",
                                       filetypes=[("Feather files", "*.feather"), ("All files", "*.*")])
        if p:
            self.weights_var.set(p)
    def append_log(self, msg):
        self.setup_text.configure(state="normal")
        self.setup_text.insert(tk.END, str(msg) + "\n")
        self.setup_text.see(tk.END)
        self.setup_text.configure(state="disabled")
    def set_setup_buttons(self, enabled):
        for b in (self.build_btn, self.list_btn, self.xform_btn):
            b.state(["!disabled" if enabled else "disabled"])
    def start_setup(self, mode):
        """mode: 'build' | 'list' (DN type names) | 'transforms' (download navis template transforms)."""
        if self.setup_busy:
            return
        weights, token, n = self.weights_var.get().strip(), self.token_var.get().strip(), int(self.n_var.get())
        align = bool(self.align_var.get())
        ctx_n = int(self.ctx_var.get())
        shapes_n, shape_types = int(self.shapes_var.get()), self.shape_types_var.get()
        if mode != "transforms" and not token and find_local_tables(weights)[0] is None and find_compiled(weights) is None:
            self.append_log("No annotations file next to the weights file, so paste your neuPrint token.")
            return
        if mode == "build" and not os.path.exists(weights):
            self.append_log(f"Weights file not found: {weights}")
            return
        self.setup_busy = True
        self.set_setup_buttons(False)
        log = lambda m: self.setup_q.put(("log", m))
        def work():
            try:
                if mode == "transforms":
                    download_transforms(log=log)
                    self.setup_q.put(("done", None))
                else:
                    out = build_subset(weights, token, n, BRAIN_FILE, log=log, list_dn=(mode == "list"), align=align, context=ctx_n,
                                       shapes=shapes_n, shape_types=shape_types)
                    self.setup_q.put(("done", out))
            except Exception as err:
                self.setup_q.put(("error", str(err)))
        threading.Thread(target=work, daemon=True).start()
    def auto_build(self):
        """First launch: if your compiled MaleCNS arrays are found and no brain_subset.npz exists, build it right away."""
        if self.setup_busy or (os.path.exists(BRAIN_FILE) and brain_is_current(BRAIN_FILE)):
            return
        if find_compiled(DEFAULT_WEIGHTS) is None:
            return
        self.append_log("Building the brain from your MaleCNS data (a few seconds) ...")
        self.start_setup("build")
    def poll_setup(self):
        """UI thread: show messages from the build thread, and load the new brain when it finishes."""
        if self.terminating:
            return
        try:
            while True:
                kind, val = self.setup_q.get_nowait()
                if kind == "log":
                    self.append_log(val)
                else:
                    self.setup_busy = False
                    self.set_setup_buttons(True)
                    if kind == "error":
                        self.append_log(f"ERROR: {val}")
                    elif val:
                        self.append_log("Loading the new brain into the simulation ...")
                        self.load_brain()
                        self.append_log("Done - switch to the Control Panel tab.")
        except queue.Empty:
            pass
        self.root.after(100, self.poll_setup)
    # ------------------------------------------------------------------
    # UI HELPERS
    # ------------------------------------------------------------------
    def update_mode_states(self):
        self.clear_keys()
        if self.input_mode.get() == "blooket":
            self.game_combo.config(state="readonly")
            self.map_btn.state(["!disabled"])
            self.add_mode_btn.state(["!disabled"])
        else:
            self.game_combo.config(state="disabled")
            self.map_btn.state(["disabled"])
            self.add_mode_btn.state(["disabled"])
    # ------------------------------------------------------------------
    # MOTOR PROPERTIES TAB: every joint motor of the fly (target vs. actual angle, step rate) + ground contact forces
    # ------------------------------------------------------------------
    def build_motors(self, parent):
        self.motor_canvas = tk.Canvas(parent, width=900, height=520, highlightthickness=0, bd=0, bg="#f0f0f0")
        self.motor_canvas.pack(fill=tk.BOTH, expand=True)
        self.motor_label = tk.Label(self.motor_canvas, bd=0, bg="#f0f0f0")
        self.motor_canvas.create_window(0, 0, window=self.motor_label, anchor="nw")
        def wheel(e):
            self.motor_canvas.yview_scroll(-1 if e.delta > 0 else 1, "units")
        for w in (self.motor_canvas, self.motor_label):
            w.bind("<MouseWheel>", wheel)
    @staticmethod
    def _deg_text(img, val, x, y, color, scale=0.4):
        t = f"{val:+.1f}"
        cv2.putText(img, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.circle(img, (x + tw + 3, y - int(7 * scale / 0.4)), 2, color, 1, cv2.LINE_AA)            # the degree sign
    def render_motors(self, W=MOTOR_W, H=900):
        """Windows-style light panel: a group box per leg that fills the page, a slider-like bar per joint motor."""
        body = self.body
        bg, edge, zebra = (240, 240, 240), (205, 205, 205), (232, 232, 232)
        ink, dim = (20, 20, 20), (95, 95, 95)
        blue, lblue, green, groove = (215, 120, 0), (240, 205, 150), (60, 160, 40), (200, 200, 200)
        F = cv2.FONT_HERSHEY_SIMPLEX
        if body.renderer is None:
            img = np.full((H, W, 3), bg, np.uint8)
            cv2.putText(img, "NeuroMechFly unavailable - no motors to show", (20, 60), F, 0.6, ink, 1, cv2.LINE_AA)
            return img
        tab = body.motor_table()
        byid = {g: (name, rows) for g, name, rows in tab}
        order = [g for g in ("lf", "lm", "lh", "rf", "rm", "rh") if g in byid]
        ncol = 4 if W >= 1500 else (3 if W >= 1050 else 2)
        if ncol == 4:
            items = ["lf", "lm", "lh", "head", "rf", "rm", "rh", "contact"]
        else:
            items = order[:3] + (["head"] if "head" in byid else []) + order[3:] + ["contact"]
            items = [it for it in items if it in byid or it == "contact"]
            if ncol == 2:
                items = [x for pair in zip(order[:3], order[3:]) for x in pair] + [i for i in ("head", "contact") if i == "contact" or "head" in byid]
        items = [it for it in items if it in byid or it == "contact"]
        nrows_of = lambda it: 6 if it == "contact" else len(byid[it][1])
        grid = [items[i:i + ncol] for i in range(0, len(items), ncol)]
        PAD, TOP = 12, 66
        units = sum(max(nrows_of(i) for i in r) + 2.0 for r in grid)
        ROW = max(15.0, min(34.0, (H - TOP - PAD * (len(grid) + 1)) / units))
        sc = max(0.8, min(1.5, ROW / 22.0))
        BW = (W - PAD * (ncol + 1)) // ncol
        need = int(TOP + sum((max(nrows_of(i) for i in r) + 2.0) * ROW + PAD for r in grid) + PAD)
        img = np.full((max(H, need), W, 3), bg, np.uint8)
        d = body.data
        cv2.putText(img, "Motor properties", (PAD, 24), F, 0.7, ink, 1, cv2.LINE_AA)
        l1 = (f"speed {body.speed_mm_s:4.1f} mm/s    heading {math.degrees(body.heading) % 360:5.1f}    pitch {math.degrees(body.pitch):+5.1f}"
              f"    roll {math.degrees(body.roll):+5.1f}    height {float(d.qpos[2]):.2f} mm")
        l2 = (f"{body.n_contacts} ground contacts    step rate {body.gait_hz:4.1f} Hz    physics {body.rt_factor:3.1f}x real time"
              f"    ground force {float(body.leg_force.sum()):4.1f} uN (fly weighs about 10 uN)")
        cv2.putText(img, l1, (PAD, 66 - 24), F, 0.45, dim, 1, cv2.LINE_AA)
        cv2.putText(img, l2, (PAD, 66 - 8), F, 0.45, dim, 1, cv2.LINE_AA)
        lx = W - 330
        cv2.rectangle(img, (lx, 12), (lx + 8, 28), blue, -1)
        cv2.putText(img, "target (command)", (lx + 14, 26), F, 0.4, dim, 1, cv2.LINE_AA)
        cv2.fillPoly(img, [np.array([[lx + 160, 14], [lx + 170, 14], [lx + 165, 24]], np.int32)], green, cv2.LINE_AA)
        cv2.putText(img, "actual angle", (lx + 176, 26), F, 0.4, dim, 1, cv2.LINE_AA)
        FMAX = 8.0
        leg_col = {"lf": (200, 110, 150), "lm": (205, 130, 90), "lh": (110, 170, 215),
                   "rf": (80, 160, 235), "rm": (95, 200, 140), "rh": (205, 190, 80), "head": (150, 150, 150)}
        def group(x0, y0, h, title):
            cv2.rectangle(img, (x0, y0 + 9), (x0 + BW, y0 + h), edge, 1)
            (tw, _), _ = cv2.getTextSize(title, F, 0.5 * sc ** 0.5, 1)
            cv2.rectangle(img, (x0 + 8, y0), (x0 + 20 + tw, y0 + 18), bg, -1)
            cv2.putText(img, title, (x0 + 14, y0 + 14), F, 0.5 * sc ** 0.5, ink, 1, cv2.LINE_AA)
        fs = min(0.38 * sc, 0.29 * BW / 265.0)          # keep the longest label clear of the bars
        SPAN = math.radians(100.0)
        y = TOP
        for r in grid:
            nr = max(nrows_of(i) for i in r)
            h = int((nr + 2.0) * ROW)
            for ci, it in enumerate(r):
                x0 = PAD + ci * (BW + PAD)
                if it == "contact":
                    group(x0, y, h, "Foot contact + adhesion pads")
                    for i, leg in enumerate(body.LEGS):
                        cy = int(y + 1.6 * ROW + i * ROW + ROW / 2)
                        fz = float(body.leg_force[i])
                        grip = float(d.ctrl[body.adh[leg]]) if leg in body.adh else 0.0
                        cv2.rectangle(img, (x0 + 12, cy - 5), (x0 + 22, cy + 5), leg_col[leg], -1)
                        cv2.putText(img, body.LEG_NAMES[leg], (x0 + 30, cy + 4), F, fs, dim, 1, cv2.LINE_AA)
                        t0, t1 = x0 + int(BW * 0.34), x0 + int(BW * 0.70)
                        cv2.rectangle(img, (t0, cy - 3), (t1, cy + 3), groove, -1)
                        cv2.rectangle(img, (t0, cy - 3), (t0 + int((t1 - t0) * min(1.0, fz / FMAX)), cy + 3), (200, 60, 220), -1)
                        cv2.putText(img, f"{fz:4.1f} uN", (x0 + int(BW * 0.72), cy + 4), F, fs, ink, 1, cv2.LINE_AA)
                        cv2.putText(img, "grip" if grip > 0.5 else "free", (x0 + int(BW * 0.88), cy + 4), F, fs,
                                    green if grip > 0.5 else (140, 140, 140), 1, cv2.LINE_AA)
                    continue
                name, rows = byid[it]
                group(x0, y, h, f"{name} ({len(rows)})")
                if it in body.LEGS:
                    fz = float(body.leg_force[body.LEGS.index(it)])
                    bx = x0 + BW - 120
                    cv2.putText(img, f"foot {fz:4.1f} uN", (bx - 70, y + 15), F, 0.38, dim, 1, cv2.LINE_AA)
                    cv2.rectangle(img, (bx + 20, y + 8), (bx + 108, y + 13), groove, -1)
                    cv2.rectangle(img, (bx + 20, y + 8), (bx + 20 + int(88 * min(1.0, fz / FMAX)), y + 13), (200, 60, 220), -1)
                tx0, tx1 = x0 + int(BW * 0.34), x0 + int(BW * 0.70)
                for i, (label, tgt, act, tq, hz, neu) in enumerate(rows):
                    top = int(y + 1.6 * ROW + i * ROW)
                    cy = int(top + ROW / 2)
                    if i % 2 == 0:
                        cv2.rectangle(img, (x0 + 4, top), (x0 + BW - 4, int(top + ROW)), zebra, -1)
                    cv2.putText(img, label, (x0 + 10, cy + 4), F, fs, ink, 1, cv2.LINE_AA)
                    f = lambda a: int(tx0 + (tx1 - tx0) * (min(SPAN, max(-SPAN, a - neu)) + SPAN) / (2 * SPAN))
                    cv2.line(img, (tx0, cy), (tx1, cy), groove, 3, cv2.LINE_AA)
                    cv2.line(img, (f(neu), cy), (f(act), cy), lblue, 3, cv2.LINE_AA)
                    cv2.line(img, (f(neu), cy - 4), (f(neu), cy + 4), (150, 150, 150), 1)
                    tg = f(tgt)
                    cv2.rectangle(img, (tg - 3, cy - 7), (tg + 3, cy + 7), blue, -1)
                    ax = f(act)
                    cv2.fillPoly(img, [np.array([[ax - 4, cy - 10], [ax + 4, cy - 10], [ax, cy - 4]], np.int32)], green, cv2.LINE_AA)
                    self._deg_text(img, math.degrees(act), x0 + int(BW * 0.73), cy + 4, ink, fs)
                    hzt = f"{hz:4.1f} Hz" if hz > 0.05 else "- Hz"
                    cv2.putText(img, hzt, (x0 + int(BW * 0.88), cy + 4), F, fs, dim if hz < 0.05 else blue, 1, cv2.LINE_AA)
            y += h + PAD
        return img
    def draw_motors(self):
        W = max(700, self.motor_canvas.winfo_width() - 2)
        H = max(420, self.motor_canvas.winfo_height() - 2)
        img = self.render_motors(W, H)
        self.set_image(self.motor_label, img)
        try:
            self.motor_canvas.configure(scrollregion=(0, 0, img.shape[1], img.shape[0]))
        except tk.TclError:
            pass
    # ------------------------------------------------------------------
    # VISION TAB: what each compound eye sees + the neck's gaze-stabilising actuation (optimal vs. actual)
    # ------------------------------------------------------------------
    def build_vision(self, parent):
        bar = ttk.Frame(parent)
        bar.pack(fill=tk.X, padx=8, pady=(6, 2))
        ttk.Label(bar, text="Smelling trigger:").pack(side=tk.LEFT)
        for text, key in (("Left antenna", "l"), ("Right antenna", "r"), ("Both antennae", "lr")):
            ttk.Button(bar, text=text, command=lambda k=key: self.trigger_smell(k)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(bar, text="Puff of odour ahead", command=self.puff_ahead).pack(side=tk.LEFT, padx=(14, 0))
        ttk.Label(bar, text="(1.5 s test odour at the chosen antenna, or a burst released in front of the fly)", foreground="#666666").pack(side=tk.LEFT, padx=(10, 0))
        self.vision_canvas = tk.Canvas(parent, width=900, height=520, highlightthickness=0, bd=0, bg="#f0f0f0")
        self.vision_canvas.pack(fill=tk.BOTH, expand=True)
        self.vision_label = tk.Label(self.vision_canvas, bd=0, bg="#f0f0f0")
        self.vision_canvas.create_window(0, 0, window=self.vision_label, anchor="nw")
    def trigger_smell(self, which):
        t = time.perf_counter() + 1.5
        for k in which:
            self.odor_pulse[k] = t
    def puff_ahead(self):
        b = self.body
        self.arena.plume.burst(b.x + math.cos(b.heading) * 6.0, b.y + math.sin(b.heading) * 6.0)
    @staticmethod
    def _eye_mosaic(gray, size):
        """Draw a greyscale camera image the way a compound eye samples it: one grey dot per ommatidium on a hex grid."""
        bgc = 240
        out = np.full((size, size, 3), bgc, np.uint8)
        if gray is None:
            return out
        g = gray.astype(np.float32)
        lo, hi = float(g.min()), float(g.max())
        n = 30
        step = size / n
        gh, gw = g.shape
        rad = max(1, int(step * 0.56))
        for r in range(int(n / 0.866) + 1):
            cy = r * step * 0.866 + step * 0.5
            if cy > size:
                break
            for c in range(n + 1):
                cx = (c + 0.5 * (r % 2)) * step
                if cx > size:
                    continue
                if ((cx - size / 2) / (size * 0.5)) ** 2 + ((cy - size / 2) / (size * 0.47)) ** 2 > 1.0:
                    continue                                       # eye-shaped patch
                v = int(g[min(gh - 1, int(cy / size * gh)), min(gw - 1, int(cx / size * gw))])
                cv2.circle(out, (int(cx), int(cy)), rad, (v, v, v), -1, cv2.LINE_AA)
        return out
    def _eye_motion(self, eyes):
        """Image motion (optic flow) of each eye in deg/s since the last sample: phase correlation of two greyscale frames."""
        now = time.perf_counter()
        prev = self._eye_prev
        self._eye_prev = (now, [e.astype(np.float32) for e in eyes])
        if prev is None or now - prev[0] > 1.0 or now - prev[0] < 1e-3:
            return
        dt = now - prev[0]
        win = cv2.createHanningWindow((eyes[0].shape[1], eyes[0].shape[0]), cv2.CV_32F)
        deg_px = EYE_FOV / float(eyes[0].shape[1])
        vals = []
        for k in range(2):
            try:
                (dx, dy), resp = cv2.phaseCorrelate(prev[1][k], self._eye_prev[1][k], win)
            except Exception:
                dx = dy = resp = 0.0
            if resp < 0.05:                                   # nothing to track (flat image)
                dx = dy = 0.0
            vals += [dx * deg_px / dt, -dy * deg_px / dt]       # + = image moves right / up
        self.eye_hist.append((now, vals[0], vals[1], vals[2], vals[3]))
    def _plot(self, img, rect, title, series, hist_t, now, ylim, unit, hline=None, WIN=25.0):
        """A light-themed line plot. series = [(values, bgr colour, label, thickness)], hist_t = sample times."""
        F = cv2.FONT_HERSHEY_SIMPLEX
        ink, dim, grid = (20, 20, 20), (95, 95, 95), (215, 215, 215)
        px0, py0, px1, py1 = rect
        if px1 - px0 < 100 or py1 - py0 < 50:
            return
        (tw, _), _ = cv2.getTextSize(title, F, 0.5, 1)
        cv2.putText(img, title, ((px0 + px1 - tw) // 2, py0 - 8), F, 0.5, ink, 1, cv2.LINE_AA)
        cv2.rectangle(img, (px0, py0), (px1, py1), (255, 255, 255), -1)
        lo, hi = ylim
        ymap = lambda v: int(py1 - (max(lo, min(hi, v)) - lo) / (hi - lo) * (py1 - py0))
        for k in range(5):
            gv = lo + (hi - lo) * k / 4.0
            cv2.line(img, (px0, ymap(gv)), (px1, ymap(gv)), (170, 170, 170) if abs(gv) < 1e-9 else grid, 1)
            cv2.putText(img, (f"{gv:.0f}" if hi >= 20 else f"{gv:.2f}"), (px0 - 40, ymap(gv) + 4), F, 0.36, dim, 1, cv2.LINE_AA)
        for k in range(6):
            gx = int(px0 + (px1 - px0) * k / 5.0)
            cv2.line(img, (gx, py0), (gx, py1), grid, 1)
            cv2.putText(img, f"{-(25 - 5 * k):d}", (gx - 8, py1 + 15), F, 0.36, dim, 1, cv2.LINE_AA)
        cv2.putText(img, "Time [s]", ((px0 + px1) // 2 - 26, py1 + 29), F, 0.4, dim, 1, cv2.LINE_AA)
        cv2.putText(img, unit, (px0 - 40, py0 - 8), F, 0.36, dim, 1, cv2.LINE_AA)
        cv2.rectangle(img, (px0, py0), (px1, py1), (170, 170, 170), 1)
        if hline is not None:
            y = ymap(hline)
            for x in range(px0, px1, 10):
                cv2.line(img, (x, y), (min(px1, x + 5), y), (120, 120, 120), 1)
        for vals, col, label, th in series:
            pts = []
            for t, v in zip(hist_t, vals):
                tt = t - now + WIN
                if tt >= 0:
                    pts.append((int(px0 + (px1 - px0) * tt / WIN), ymap(v)))
            if len(pts) > 1:
                cv2.polylines(img, [np.array(pts, np.int32)], False, col, th, cv2.LINE_AA)
        lx = px1 - 215
        for i, (vals, col, label, th) in enumerate(series):
            cv2.line(img, (lx, py0 + 13 + 17 * i), (lx + 22, py0 + 13 + 17 * i), col, 2, cv2.LINE_AA)
            cv2.putText(img, label, (lx + 28, py0 + 17 + 17 * i), F, 0.38, ink, 1, cv2.LINE_AA)
    def _smell_diagram(self, img, x0, y0, w, h, cl, cr, head=None):
        """The real 3-D head (front view) flattened to 2-D, with the smell organs highlighted and their odour numbers."""
        F = cv2.FONT_HERSHEY_SIMPLEX
        ink, dim = (20, 20, 20), (95, 95, 95)
        cv2.putText(img, "Smell receptors (head, front view)", (x0, y0 - 8), F, 0.5, ink, 1, cv2.LINE_AA)
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (255, 255, 255), -1)
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (170, 170, 170), 1)
        vl, vr = min(1.0, cl / 0.4), min(1.0, cr / 0.4)
        vp = 0.5 * (vl + vr) * 0.6                                # palps: the average, a bit weaker
        for k, (lab_t, val, col) in enumerate((("Left antenna", cl, (200, 110, 30)), ("Right antenna", cr, (30, 140, 230)), ("Palps (est.)", 0.5 * (cl + cr) * 0.6, (60, 150, 60)))):
            yy = y0 + h - 48 + 17 * k
            cv2.putText(img, lab_t, (x0 + 12, yy), F, 0.45, col, 1, cv2.LINE_AA)
            vt = f"{val:.2f}"
            (tw, _), _ = cv2.getTextSize(vt, F, 0.45, 1)
            cv2.putText(img, vt, (x0 + 150 - tw, yy), F, 0.45, col, 1, cv2.LINE_AA)      # values right-aligned in one column
        for i in range(70):                                       # colour bar: no odour -> strong
            t = i / 69.0
            c = tuple(int(u + (v - u) * t) for u, v in zip((255, 255, 255), (0, 0, 0)))
            cv2.line(img, (x0 + w - 100 + i, y0 + h - 18), (x0 + w - 100 + i, y0 + h - 10), c, 1)
        cv2.rectangle(img, (x0 + w - 101, y0 + h - 19), (x0 + w - 30, y0 + h - 9), (170, 170, 170), 1)       # outline so the white end is visible
        cv2.putText(img, "odour", (x0 + w - 100, y0 + h - 24), F, 0.36, dim, 1, cv2.LINE_AA)
        if head is None:
            cv2.putText(img, "(no 3-D head available)", (x0 + 20, y0 + h // 2), F, 0.45, dim, 1, cv2.LINE_AA)
            return
        rgb, body = head
        ids = self.body.part_ids
        ys, xs = np.nonzero(body >= 0)
        if not len(xs):
            return
        pad = 14
        bx0, bx1 = max(0, xs.min() - pad), min(body.shape[1], xs.max() + pad)
        by0, by1 = max(0, ys.min() - pad), min(body.shape[0], ys.max() + pad)
        rgb, body = rgb[by0:by1, bx0:bx1], body[by0:by1, bx0:bx1]
        area_h = h - 82
        sc = min((w - 20) / body.shape[1], area_h / body.shape[0])
        tw, th = max(8, int(body.shape[1] * sc)), max(8, int(body.shape[0] * sc))
        rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
        body = cv2.resize(body.astype(np.int16), (tw, th), interpolation=cv2.INTER_NEAREST)
        lum = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        base = np.array([200, 205, 210], np.float32)              # flat light-grey head, shaded by the 3-D render
        tile = np.empty((th, tw, 3), np.float32)
        tile[:] = (base * (0.55 + 0.6 * lum[..., None])).clip(0, 255)
        tile[body < 0] = 255
        def lerp(v):
            a = np.array([255, 255, 255], np.float32)                 # odour scale: white (none) -> black (strong)
            bb = np.array([0, 0, 0], np.float32)
            return a + (bb - a) * max(0.0, min(1.0, v))
        marks = []
        for key, v, col, name in (("lant", vl, (200, 110, 30), "L"), ("rant", vr, (30, 140, 230), "R"), ("palp", vp, (60, 150, 60), "palps")):
            m = np.isin(body, ids[key])
            if not m.any():
                continue
            tile[m] = tile[m] * 0.35 + lerp(0.12 + 0.88 * v) * 0.65
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            tile = np.ascontiguousarray(tile)
            cv2.drawContours(tile, cnts, -1, col, 2, cv2.LINE_AA)
            yy, xx = np.nonzero(m)
            marks.append((key, name, col, int(xx.mean()), int(yy.mean()), v))
        ox, oy = x0 + (w - tw) // 2, y0 + 8
        out = tile.clip(0, 255).astype(np.uint8)
        img[oy:oy + th, ox:ox + tw] = out
        for key, name, col, mx, my, v in marks:
            px, py = ox + mx, oy + my
            if key in ("lant", "rant"):                      # label on the side where that antenna appears in the picture
                lab = f"{'L' if key == 'lant' else 'R'} antenna {(cl if key == 'lant' else cr):.2f}"
                lx = ox + 4 if mx < tw // 2 else ox + tw - 118
                ly = oy + 12
            else:
                lab, lx, ly = f"palps {0.5 * (cl + cr) * 0.6:.2f}", ox + tw // 2 - 36, oy + th - 4
            cv2.line(img, (px, py), (min(max(lx + 50, ox), ox + tw), ly + 3), col, 1, cv2.LINE_AA)
            cv2.circle(img, (px, py), 3, col, -1, cv2.LINE_AA)
            cv2.putText(img, lab, (lx, ly), F, 0.42, col, 1, cv2.LINE_AA)
    def render_vision(self, W, H, eyes=None):
        bg, ink, dim = (240, 240, 240), (20, 20, 20), (95, 95, 95)
        blue, orange = (200, 110, 30), (30, 140, 230)
        F = cv2.FONT_HERSHEY_SIMPLEX
        img = np.full((H, W, 3), bg, np.uint8)
        cv2.putText(img, "Miscellaneous", (14, 22), F, 0.65, ink, 1, cv2.LINE_AA)
        cv2.putText(img, "vision and smell", (150, 22), F, 0.45, dim, 1, cv2.LINE_AA)
        top, pad = 52, 14
        half = (H - top - pad) // 2
        # ---------------- top half: eyes (right eye on the left of the left eye) + optic-flow graphs
        side = int(max(80, min(half - 40, W * 0.235)))
        ey = top + 22
        for i, (name, key) in enumerate((("Right eye raw vision", 1), ("Left eye raw vision", 0))):
            ex = 14 + i * (side + 14)
            cv2.putText(img, name, (ex, ey - 6), F, 0.45, ink, 1, cv2.LINE_AA)
            img[ey:ey + side, ex:ex + side] = self._eye_mosaic(eyes[key] if eyes else None, side)
            cv2.rectangle(img, (ex, ey), (ex + side, ey + side), (205, 205, 205), 1)
        gx0 = 14 + 2 * side + 14 + 62
        gx1 = W - 24
        hist = list(self.eye_hist)
        now = time.perf_counter()
        ht = [h[0] for h in hist]
        gh = (half - 12) // 2 - 56
        for row, (name, hi, vi) in enumerate((("Right eye - image motion (optic flow)", 3, 4), ("Left eye - image motion (optic flow)", 1, 2))):
            py0 = top + 24 + row * (gh + 56)
            vals = [abs(h[hi]) for h in hist] + [abs(h[vi]) for h in hist] + [10.0]
            YR = float(np.ceil(max(vals) * 1.15 / 10.0) * 10.0)
            self._plot(img, (gx0, py0, gx1, py0 + gh), name,
                       [([h[hi] for h in hist], blue, "Horizontal (+ = right)", 2), ([h[vi] for h in hist], orange, "Vertical (+ = up)", 2)],
                       ht, now, (-YR, YR), "deg/s")
        # ---------------- bottom half: smell receptor diagram + odour graphs
        y0 = top + half + 24
        hh = H - y0 - 14
        dw = int(min(W * 0.30, hh * 1.1))
        cl, cr = self.odor_now
        tnow = time.monotonic()                                   # the head close-up takes ~0.6 s to render and barely changes: refresh it every 3 s
        if getattr(self, "_head_cache", None) is None or tnow - getattr(self, "_head_t", 0.0) > 3.0:
            self._head_cache = self.body.render_head() if getattr(self.body, 'renderer', None) is not None else None
            self._head_t = tnow
        head = self._head_cache
        self._smell_diagram(img, 14, y0 + 24, dw, hh - 24, cl, cr, head)
        oh = list(self.odor_hist)
        ot = [h[0] for h in oh]
        qx0, qx1 = 14 + dw + 62, W - 24
        gh2 = (hh - 12) // 2 - 56
        top_c = max([h[1] for h in oh] + [h[2] for h in oh] + [0.4])
        yc = float(np.ceil(top_c * 1.15 * 10.0) / 10.0)
        self._plot(img, (qx0, y0 + 24, qx1, y0 + 24 + gh2), "Odour at the antennae",
                   [([h[1] for h in oh], blue, "Left antenna", 2), ([h[2] for h in oh], orange, "Right antenna", 2)],
                   ot, now, (0.0, yc), "conc.", hline=OdorTracker.THR)
        dd = [(h[1] - h[2]) for h in oh]
        yd = float(np.ceil(max([abs(v) for v in dd] + [0.2]) * 1.15 * 10.0) / 10.0)
        self._plot(img, (qx0, y0 + 24 + gh2 + 56, qx1, y0 + 24 + 2 * gh2 + 56), "Left - right odour (steering cue)",
                   [(dd, (60, 140, 60), "Left minus right", 2)], ot, now, (-yd, yd), "conc.")
        return img
    def draw_vision(self):
        W = max(700, self.vision_canvas.winfo_width() - 2)
        H = max(420, self.vision_canvas.winfo_height() - 2)
        eyes = self.body.render_eyes() if getattr(self.body, "renderer", None) is not None else None
        if eyes:
            self._eye_motion(eyes)
        self.set_image(self.vision_label, self.render_vision(W, H, eyes))
    def build_minimized(self, parent):
        """4th page: a tiny 'parked' state. Selecting it shrinks the whole window; any other tab restores it."""
        try:
            import tkinter.font as tkfont
            fams = set(tkfont.families(self.root))
        except Exception:
            fams = set()
        medium = next((f for f in ("Montserrat Medium", "Montserrat") if f in fams), "Segoe UI")
        normal = "Montserrat" if "Montserrat" in fams else "Segoe UI"
        big = 22
        small = max(7, round(big * 0.4))                       # smaller text = 0.4 x the middle text
        bg = ttk.Style().lookup("TFrame", "background") or "#f0f0f0"
        box = tk.Frame(parent, bg=bg)
        box.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(box, text="Minimized", bg=bg, font=(medium, big)).pack()
        # Inset ("pressed in") pixel-style note, top left, like an old Windows pop-up: light copy offset 1 px under a dark copy.
        pix = next((f for f in ("Terminal", "Fixedsys", "MS Sans Serif") if f in fams), "Consolas")
        note = "Neeche tool isn't it?"
        cv = tk.Canvas(parent, width=230, height=22, bg=bg, highlightthickness=0, bd=0)
        cv.place(x=10, y=8)
        cv.create_text(2, 12, text=note, anchor="w", font=(pix, 9), fill="#ffffff")
        cv.create_text(1, 11, text=note, anchor="w", font=(pix, 9), fill="#7a7a7a")
        tk.Label(box, text="Select a page to start", bg=bg, fg="#555555", font=(normal, small)).pack(pady=(2, 0))
    # ------------------------------------------------------------------
    # SETTINGS TAB
    # ------------------------------------------------------------------
    def S(self, key):
        """Current value of a setting (falls back to the default if the control is missing or holds junk)."""
        try:
            return self.sv[key].get()
        except Exception:
            return SETTING_DEFAULTS[key]
    def _sched_apply(self):
        if getattr(self, "_apply_job", None) is not None:
            try:
                self.root.after_cancel(self._apply_job)
            except Exception:
                pass
        self._apply_job = self.root.after(150, self.apply_settings)
    def save_settings(self):
        try:
            data = {k: v.get() for k, v in self.sv.items()}
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1)
        except Exception as err:
            print("[!] could not save settings:", err)
    def apply_settings(self):
        """Push every setting into the running program (called on any change, and once at start)."""
        self._apply_job = None
        try:
            fps = max(5, min(120, int(float(self.S("fps")))))
        except Exception:
            fps = 30
        self.frame_ms = max(5, int(1000 / fps))
        b = getattr(self, "body", None)
        try:
            self.render_dt = 1.0 / max(5, int(float(self.S("body_hz"))))
        except Exception:
            self.render_dt = 1.0 / 30
        if b is not None:
            b.render_dt = self.render_dt
            b.shadows = bool(self.S("shadows"))
            b.STEPMAX, b.LIFT, b.HIND_LIFT = float(self.S("stepmax")), float(self.S("lift")), float(self.S("hind"))
            b.F_MAX = float(self.S("fmax"))
            b.F_MIN = min(2.5, b.F_MAX)
            b.STANCE = float(self.S("stance"))
            b.GAZE_STAB = float(self.S("gaze"))
            b.TRIPOD = dict(GAITS.get(self.S("gait"), GAITS["Tripod"]))
            b.KP, b.ADHESION, b.FRICTION = float(self.S("kp")), float(self.S("adh")), float(self.S("fric"))
            try:
                b.apply_physics(b.KP, b.ADHESION, b.FRICTION, GRAVITY.get(self.S("grav"), 1.0))
            except Exception as err:
                print("[!] physics settings:", err)
        kb = {"wasd": tuple(self.S(k) or d for k, d in (("kw_f", "w"), ("kw_b", "s"), ("kw_l", "a"), ("kw_r", "d"))),
              "arrows": tuple(self.S(k) or d for k, d in (("ka_f", "up"), ("ka_b", "down"), ("ka_l", "left"), ("ka_r", "right"))),
              "jump": self.S("kb_jump") or "space"}
        pm = {}
        for k_, d_ in (("kw_f", "w"), ("kw_b", "s"), ("kw_l", "a"), ("kw_r", "d"), ("ka_f", "up"), ("ka_b", "down"), ("ka_l", "left"), ("ka_r", "right"), ("kb_jump", "space")):
            try:
                if self.S("pl_" + k_):
                    pm[self.S(k_) or d_] = max(20.0, min(8000.0, float(self.S("hz_" + k_))))
            except Exception:
                pass
        try:
            if self.S("pl_food") and getattr(self, "food_key", ""):
                pm[self.food_key] = max(20.0, min(8000.0, float(self.S("hz_food"))))
        except Exception:
            pass
        self.cfg["pulse"] = pm
        old_kb = self.cfg.get("kb")
        self.cfg["kb"] = kb                                  # plain Python copy: the vision thread reads this, never the Tk variables
        if old_kb is not None and old_kb != kb:
            try:
                self.clear_keys()                            # let go of keys held under the old mapping
            except Exception:
                pass
        OdorTracker.THR = float(self.S("thr"))
        Plume.DIFF, Plume.RATE, Plume.MAXN = float(self.S("diff")), float(self.S("rate")), int(float(self.S("maxn")))
        try:
            self.root.attributes("-topmost", bool(self.S("topmost")))
        except Exception:
            pass
        grid = (bool(self.S("show_grid")), bool(self.S("show_trail")))
        if grid != getattr(self, "_grid_state", None):
            self._grid_state = grid
            if hasattr(self, "arena"):
                self.arena.version += 1                           # redraw the map's static layer
        try:
            self.apply_theme()
        except Exception as err:
            print("[!] theme:", err)
        self.save_settings()
    # ---- status light (linked to a Discord bot)
    def _start_discord(self):
        try:
            with open(DISCORD_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            token, uid = str(cfg["token"]).strip(), int(cfg["user_id"])
        except Exception:
            self._dc_state["note"] = "Discord not linked (create %s)" % DISCORD_FILE
            return
        try:
            import discord, asyncio
        except Exception:
            self._dc_state["note"] = "Discord not linked (pip install discord.py)"
            return
        st = self._dc_state
        def run():
            intents = discord.Intents.default()
            intents.presences = True
            intents.members = True
            client = discord.Client(intents=intents)
            def look():
                for g in client.guilds:
                    m = g.get_member(uid)
                    if m is not None:
                        st["status"] = str(m.status)
                        st["note"] = ""
                        return
                st["note"] = "Discord: you are not in a server with the bot"
            @client.event
            async def on_ready():
                look()
            @client.event
            async def on_presence_update(before, after):
                if after.id == uid:
                    st["status"] = str(after.status)
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(client.start(token))
            except Exception as err:
                st["note"] = "Discord error: %s" % err
        threading.Thread(target=run, daemon=True).start()
    def _poll_status(self):
        if self._dc_state.get("sent"):
            m = self._dc_state.pop("sent")
            if m == "OK":
                self.sg_msg.set("Thank you for your suggestion! Your suggestion has been sent.")
                self._play_yipee()
                self._confetti()
            else:
                self.sg_msg.set(m)
        st = self._dc_state.get("status")
        if st is None:
            col, txt = "#8a8a8a", self._dc_state.get("note") or SUGGEST_STATUS
            if not self._dc_state.get("note"):
                col = {"available": "#2ea043", "busy": "#e3a008", "away": "#d13438"}.get(SUGGEST_STATUS.lower(), "#8a8a8a")
        else:
            col, txt = {"online": ("#2ea043", "Available"), "idle": ("#e3a008", "Away"), "dnd": ("#d13438", "Busy")}.get(st, ("#8a8a8a", "Offline"))
        try:
            self._led.itemconfigure(self._led_dot, fill=col)
            self._led_txt.set(txt)
            self.root.after(1500, self._poll_status)
        except Exception:
            pass
    # ---- suggestions
    def _sg(self):
        return self.sg_text.get("1.0", "end").strip()
    def sg_email(self):
        import webbrowser, urllib.parse
        t = self._sg()
        if not t:
            self.sg_msg.set("Write something first.")
        elif not SUGGEST_EMAIL:
            self.root.clipboard_clear(); self.root.clipboard_append(t)
            self.sg_msg.set("No e-mail address set yet (SUGGEST_EMAIL at the top of the file), so it was copied to the clipboard instead.")
        else:
            webbrowser.open("mailto:%s?subject=%s&body=%s" % (SUGGEST_EMAIL, urllib.parse.quote("FlyLab suggestion"), urllib.parse.quote(t)))
            self.sg_msg.set("Opened your mail program.")
    def sg_discord(self):
        t = self._sg()
        if not t:
            self.sg_msg.set("Write something first.")
            return
        try:
            with open(DISCORD_FILE, "r", encoding="utf-8") as f:
                url = str(json.load(f).get("webhook_url", "")).strip()
        except Exception:
            url = ""
        if not url:
            self.sg_msg.set('Add "webhook_url" to flylab_discord.json first (Discord: channel settings > Integrations > Webhooks).')
            return
        self.sg_msg.set("Sending...")
        st = self._dc_state
        def send():
            import urllib.request
            try:
                body = json.dumps({"content": ("**FlyLab suggestion**\n" + t)[:1900]}).encode("utf-8")
                req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", "User-Agent": "FlyLab/1.0"})
                urllib.request.urlopen(req, timeout=10).read()
                st["sent"] = "OK"
            except Exception as err:
                st["sent"] = "Discord send failed: %s" % err
        threading.Thread(target=send, daemon=True).start()
    def _place_window(self):
        """Open the window at the saved position (Settings > Mapping), kept on screen."""
        try:
            if not self.S("win_use"):
                return
            x, y = [int(float(v)) for v in str(self.S("win_pos")).split(",")]
            sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
            self.root.geometry("+%d+%d" % (max(-20, min(x, sw - 120)), max(0, min(y, sh - 80))))
        except Exception:
            pass
    def _play_yipee(self):
        try:
            path = next((p for p in CONFETTI_SOUND if os.path.isfile(p)), None)
            if not path:
                return
            mci = ctypes.windll.winmm.mciSendStringW
            mci("close yipee", None, 0, None)
            rc = mci('open "%s" type mpegvideo alias yipee' % path, None, 0, None)
            if rc == 0:
                rc = mci("play yipee", None, 0, None)      # returns at once, plays in the background
            if rc != 0:
                buf = ctypes.create_unicode_buffer(256)
                ctypes.windll.winmm.mciGetErrorStringW(rc, buf, 255)
                print("[!] yipee sound: MCI error %d: %s" % (rc, buf.value))
        except Exception as err:
            print("[!] sound:", err)
    def _confetti(self):
        """Confetti shoots from the bottom-left and bottom-right corners of the window, arcs up and falls. Drawn on a click-through transparent overlay."""
        root = self.root
        try:
            root.update_idletasks()
            x0, y0, w, h = root.winfo_rootx(), root.winfo_rooty(), root.winfo_width(), root.winfo_height()
            top = tk.Toplevel(root)
            top.overrideredirect(True)
            top.attributes("-topmost", True)
            top.attributes("-transparentcolor", "#010101")
            top.geometry("%dx%d+%d+%d" % (w, h, x0, y0))
            cv = tk.Canvas(top, width=w, height=h, bg="#010101", highlightthickness=0, bd=0)
            cv.pack()
        except Exception:
            return
        try:
            self.sg_text.delete("1.0", "end")
        except Exception:
            pass
        cols = ["#f94144", "#f3722c", "#f9c74f", "#90be6d", "#43aa8b", "#4cc9f0", "#b5179e", "#ffffff"]
        parts = []
        for side in (0, 1):
            for _ in range(45):
                sz = random.randint(5, 10)
                px, py = (6 if side == 0 else w - 6), h - 6
                ang = math.radians(random.uniform(52, 82))
                sp = random.uniform(0.9, 1.5) * h * 1.25              # px/s, scaled to the window height
                vx = math.cos(ang) * sp * (1 if side == 0 else -1)
                vy = -math.sin(ang) * sp
                it = cv.create_rectangle(px, py, px + sz, py + sz * 0.6, fill=random.choice(cols), outline="")
                parts.append([it, px, py, vx, vy, random.uniform(0, 0.25)])      # id, x, y, vx, vy, launch delay
        t0 = time.perf_counter()
        last = [t0]
        def step():
            now = time.perf_counter()
            dt = min(0.05, now - last[0])
            last[0] = now
            alive = 0
            for p in parts:
                if now - t0 < p[5]:
                    alive += 1
                    continue
                p[4] += 900 * dt                      # gravity
                p[3] *= (1 - 0.8 * dt)                # air drag
                p[4] *= (1 - 0.35 * dt)
                dx, dy = p[3] * dt, p[4] * dt
                p[1] += dx
                p[2] += dy
                if p[2] < h + 20:
                    cv.move(p[0], dx, dy)
                    alive += 1
            if alive and now - t0 < 4.0:
                root.after(24, step)
            else:
                try:
                    top.destroy()
                except Exception:
                    pass
        step()
    def sg_github(self):
        import webbrowser, urllib.parse
        t = self._sg()
        if not t:
            self.sg_msg.set("Write something first.")
            return
        webbrowser.open("%s/issues/new?title=%s&body=%s" % (SUGGEST_REPO, urllib.parse.quote("FlyLab suggestion"), urllib.parse.quote(t[:1500])))
        self.sg_msg.set("Opened a new GitHub issue in your browser.")
    def sg_copy(self):
        t = self._sg()
        if t:
            self.root.clipboard_clear(); self.root.clipboard_append(t)
        self.sg_msg.set("Copied." if t else "Write something first.")
    def sg_save(self):
        t = self._sg()
        if not t:
            self.sg_msg.set("Write something first.")
            return
        p = os.path.join(os.path.dirname(os.path.abspath(SETTINGS_FILE)), "flylab_suggestions.txt")
        try:
            with open(p, "a", encoding="utf-8") as f:
                f.write(time.strftime("[%Y-%m-%d %H:%M] ") + t + "\n\n")
            self.sg_msg.set("Saved to " + p)
        except Exception as err:
            self.sg_msg.set("Could not save: %s" % err)
    # ---- themes
    def _theme_now(self):
        mode, th = self.S("theme_mode"), self.S("theme")
        if mode == "Randomize":
            if not hasattr(self, "_theme_rand"):
                self._theme_rand = random.choice(["Light", "Dark"])
            if getattr(self, "_theme_user_set", False):
                return th if th in ("Light", "Dark") else self._sys_theme()
            return self._theme_rand
        return self._sys_theme() if th == "System" else ("Dark" if th == "Dark" else "Light")
    def _sys_theme(self):
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
            return "Light" if winreg.QueryValueEx(k, "AppsUseLightTheme")[0] else "Dark"
        except Exception:
            return "Light"
    def apply_theme(self):
        want = self._theme_now()
        sel = (self.S("theme"), self.S("theme_mode"))
        if getattr(self, "_theme_sel", None) is not None and sel != self._theme_sel:
            self._theme_user_set = True               # the user touched the controls: stop using the random pick
            want = self._theme_now()
        self._theme_sel = sel
        if want == getattr(self, "_theme_on", None):
            return
        self._theme_on = want
        dark = want == "Dark"
        st = ttk.Style()
        if not hasattr(self, "_theme_orig"):
            self._theme_orig = st.theme_use()
            self._lite_bg = self.root.winfo_rgb(st.lookup("TFrame", "background") or "#f0f0f0")
            self._lite_fg = self.root.winfo_rgb("#555555")
        bg, fg, fld, sel_bg = ("#202020", "#e6e6e6", "#2d2d2d", "#0067c0") if dark else (None, None, None, None)
        if dark:
            st.theme_use("clam")
            st.configure(".", background=bg, foreground=fg, fieldbackground=fld, bordercolor="#555555", lightcolor=bg, darkcolor=bg,
                         troughcolor="#2d2d2d", selectbackground=sel_bg, selectforeground="white", insertcolor=fg)
            st.configure("TButton", background="#3a3a3a", foreground=fg, bordercolor="#555555", padding=3)
            st.map("TButton", background=[("active", "#4a4a4a"), ("pressed", "#2a2a2a")])
            st.configure("TNotebook", background=bg, bordercolor="#555555")
            st.configure("TNotebook.Tab", background="#2d2d2d", foreground=fg, padding=(8, 3))
            st.map("TNotebook.Tab", background=[("selected", "#3c3c3c")], foreground=[("selected", "white")])
            st.configure("TCombobox", fieldbackground=fld, background="#3a3a3a", foreground=fg, arrowcolor=fg)
            st.map("TCombobox", fieldbackground=[("readonly", fld)], foreground=[("readonly", fg)])
            st.configure("TLabelframe", background=bg, bordercolor="#555555")
            st.configure("TLabelframe.Label", background=bg, foreground=fg)
            st.map("TCheckbutton", background=[("active", bg)], indicatorcolor=[("selected", sel_bg), ("!selected", fld)])
            st.map("TRadiobutton", background=[("active", bg)], indicatorcolor=[("selected", sel_bg), ("!selected", fld)])
            st.configure("TScrollbar", background="#3a3a3a", troughcolor="#2d2d2d", arrowcolor=fg, bordercolor=bg)
            st.configure("Horizontal.TScale", background=bg, troughcolor="#3a3a3a")
            self.root.option_add("*TCombobox*Listbox.background", fld)
            self.root.option_add("*TCombobox*Listbox.foreground", fg)
        else:
            try:
                st.theme_use(self._theme_orig)
            except Exception:
                pass
            self.root.option_add("*TCombobox*Listbox.background", "white")
            self.root.option_add("*TCombobox*Listbox.foreground", "black")
        new_bg = self.root.winfo_rgb(bg) if dark else self._lite_bg
        new_fg = self.root.winfo_rgb("#aaaaaa") if dark else self._lite_fg
        cur_bg = self.root.winfo_rgb("#202020") if not dark else self._lite_bg
        cur_fg = self.root.winfo_rgb("#aaaaaa") if not dark else self._lite_fg
        hex_bg = "#202020" if dark else "#%02x%02x%02x" % tuple(c // 257 for c in self._lite_bg)
        hex_fg = "#aaaaaa" if dark else "#555555"
        mapping = {"#ffffff": "#000000", "#7a7a7a": "#9a9a9a"} if dark else {"#000000": "#ffffff", "#9a9a9a": "#7a7a7a"}
        def walk(w):
            cls = w.winfo_class()
            if cls in ("Frame", "Label", "Canvas", "Text", "Listbox", "Toplevel", "Tk", "Labelframe", "Checkbutton", "Radiobutton", "Scale"):
                for opt in ("background", "foreground"):
                    try:
                        val = w.cget(opt)
                    except Exception:
                        continue
                    try:
                        rgb = self.root.winfo_rgb(val)
                    except Exception:
                        continue
                    if opt == "background" and rgb == cur_bg:
                        w.configure(background=hex_bg)
                    elif opt == "foreground" and rgb == cur_fg:
                        w.configure(foreground=hex_fg)
                if cls == "Text":
                    w.configure(background="#2d2d2d" if dark else "white", foreground="#e6e6e6" if dark else "black", insertbackground="#e6e6e6" if dark else "black")
                if cls == "Listbox":
                    w.configure(background="#2d2d2d" if dark else "white", foreground="#e6e6e6" if dark else "black",
                                highlightbackground="#555555" if dark else "#bbbbbb")
                if cls == "Canvas":
                    try:
                        if int(float(w.cget("height"))) == 22:                 # the inset notes
                            for it in w.find_all():
                                if w.type(it) == "text":
                                    c = w.itemcget(it, "fill").lower()
                                    if c in mapping:
                                        w.itemconfigure(it, fill=mapping[c])
                    except Exception:
                        pass
            for ch in w.winfo_children():
                walk(ch)
        self.root.configure(background=hex_bg)
        walk(self.root)
        if hasattr(self, "arena"):
            self.arena.version += 1
    def reset_settings(self, keys=None):
        for k, d in SETTING_DEFAULTS.items():
            if keys is None or k in keys:
                try:
                    self.sv[k].set(d)
                except Exception:
                    pass
    def apply_profile(self):
        pr = PROFILES.get(self.S("profile"))
        if pr:
            for k, v in pr.items():
                self.sv[k].set(v)
    def build_settings(self, parent):
        """Settings page: categories on the left, the controls of the chosen category on the right. Everything is saved to flylab_settings.json."""
        saved = {}
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
        except Exception:
            pass
        self.sv = {}
        for key, d in SETTING_DEFAULTS.items():
            val = saved.get(key, d)
            if isinstance(d, bool):
                v = tk.BooleanVar(value=bool(val))
            elif isinstance(d, (int, float)):
                try:
                    v = tk.DoubleVar(value=float(val))
                except Exception:
                    v = tk.DoubleVar(value=float(d))
            else:
                v = tk.StringVar(value=str(val))
            v.trace_add("write", lambda *_: self._sched_apply())
            self.sv[key] = v
        try:
            import tkinter.font as tkfont
            fams = set(tkfont.families(self.root))
        except Exception:
            fams = set()
        bg = ttk.Style().lookup("TFrame", "background") or "#f0f0f0"
        pix = next((f for f in ("Terminal", "Fixedsys", "MS Sans Serif") if f in fams), "Consolas")
        note = "Recommended to not touchy"
        cv = tk.Canvas(parent, width=260, height=22, bg=bg, highlightthickness=0, bd=0)     # inset note, same style as the Minimize page
        cv.place(x=10, y=8)
        cv.create_text(2, 12, text=note, anchor="w", font=(pix, 9), fill="#ffffff")
        cv.create_text(1, 11, text=note, anchor="w", font=(pix, 9), fill="#7a7a7a")
        body = ttk.Frame(parent)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=(38, 12))
        names = ["Performance", "Body & physics", "Smell & wind", "Build tab", "Display", "Recording", "Mapping", "Profiles", "Theme", "Advanced", "Suggestions"]
        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, anchor=tk.N, padx=(0, 14))
        lb = tk.Listbox(left, exportselection=False, activestyle="none", width=18, height=len(names), font=("Segoe UI", 10),
                        relief=tk.FLAT, highlightthickness=1, highlightbackground="#bbbbbb", selectbackground="#0067c0", selectforeground="white")
        for n in names:
            lb.insert(tk.END, "  " + n)
        lb.pack(side=tk.TOP, anchor=tk.N)
        # one long scrollable page; the list on the left jumps to a heading
        area = ttk.Frame(body)
        area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sc_cv = tk.Canvas(area, highlightthickness=0, bd=0, bg=bg, width=720, height=420)
        sbar = ttk.Scrollbar(area, orient=tk.VERTICAL, command=sc_cv.yview)
        sc_cv.configure(yscrollcommand=sbar.set)
        sbar.pack(side=tk.RIGHT, fill=tk.Y)
        sc_cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        holder = ttk.Frame(sc_cv)
        win = sc_cv.create_window(0, 0, window=holder, anchor="nw")
        filler = ttk.Frame(holder, height=1)
        pages = {}
        def spin_wheel(w, var, lo, hi, step):
            """The mouse wheel over a number box changes ONLY that value (the page stays put)."""
            def on_wheel(e):
                try:
                    try:
                        cur = float(var.get())
                    except Exception:
                        cur = lo
                    var.set(max(lo, min(hi, cur + (step if e.delta > 0 else -step))))
                except Exception:
                    pass
                return "break"
            w.bind("<MouseWheel>", on_wheel)
        def page(name):
            f = ttk.Frame(holder)
            f.pack(fill=tk.X, expand=True, padx=(0, 10), pady=(0, 18))
            pages[name] = f
            ttk.Label(f, text=name, font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 6))
            return f
        def box(f, title):
            b = ttk.LabelFrame(f, text=" " + title + " ", padding=8)
            b.pack(fill=tk.X, pady=(0, 8))
            return b
        def check(f, text, key):
            ttk.Checkbutton(f, text=text, variable=self.sv[key]).pack(anchor=tk.W, pady=1)
        def combo(f, text, key, values, width=24):
            r = ttk.Frame(f)
            r.pack(fill=tk.X, pady=2)
            ttk.Label(r, text=text, width=26).pack(side=tk.LEFT)
            ttk.Combobox(r, textvariable=self.sv[key], values=values, state="readonly", width=width).pack(side=tk.LEFT)
        def slide(f, text, key, lo, hi, fmt):
            self.make_slider(f, text, self.sv[key], lo, hi, fmt, label_w=26)
        # --- Performance
        f = page("Performance")
        b = box(f, "Speed")
        combo(b, "Program frame rate (fps)", "fps", ("10", "15", "20", "30", "45", "60"))
        combo(b, "Body view refresh (Hz)", "body_hz", ("10", "15", "20", "30", "60"))
        check(b, "Only draw the page that is open", "only_open")
        check(b, "Show fps and frame time on the Control Panel", "show_fps")
        check(b, "Shadows and reflections in the 3-D views (slower)", "shadows")
        b = box(f, "What to draw")
        check(b, "Fly body view", "draw_body")
        check(b, "Neuron cloud", "draw_cloud")
        check(b, "Time graphs", "draw_graphs")
        check(b, "Smoke on the Build map", "draw_smoke")
        # --- Body & physics
        f = page("Body & physics")
        b = box(f, "Motors and contact (applied live)")
        slide(b, "Joint stiffness", "kp", 10.0, 120.0, "{:.0f}")
        slide(b, "Foot adhesion", "adh", 0.0, 8.0, "{:.1f}")
        slide(b, "Ground friction", "fric", 0.3, 4.0, "{:.1f}")
        combo(b, "Gravity", "grav", tuple(GRAVITY))
        b = box(f, "Walking")
        combo(b, "Gait", "gait", tuple(GAITS))
        slide(b, "Step length (mm)", "stepmax", 0.8, 4.0, "{:.1f}")
        slide(b, "Step lift (mm)", "lift", 0.05, 0.6, "{:.2f}")
        slide(b, "Hind-leg lift (x)", "hind", 1.0, 3.5, "{:.1f}")
        slide(b, "Max step rate (Hz)", "fmax", 2.0, 9.0, "{:.1f}")
        slide(b, "Foot-on-ground time", "stance", 0.5, 0.85, "{:.2f}")
        slide(b, "Head stabilisation", "gaze", 0.0, 1.5, "{:.1f}")
        ttk.Button(f, text="Reset body & physics to defaults",
                   command=lambda: self.reset_settings(("kp", "adh", "fric", "grav", "gait", "stepmax", "lift", "hind", "fmax", "stance", "gaze"))).pack(anchor=tk.W)
        # --- Smell & wind
        f = page("Smell & wind")
        b = box(f, "Plume")
        slide(b, "Source strength (puffs/s)", "rate", 20.0, 200.0, "{:.0f}")
        slide(b, "Spreading (diffusion)", "diff", 0.05, 1.0, "{:.2f}")
        slide(b, "Max particles", "maxn", 2000.0, 20000.0, "{:.0f}")
        slide(b, "Wind turbulence", "turb", 0.0, 2.0, "{:.1f}")
        combo(b, "Wind mode", "wind_mode", ("Steady", "Gusting"))
        check(b, "Smoke flows around obstacles", "smoke_obst")
        b = box(f, "Fly's nose")
        slide(b, "Detection threshold", "thr", 0.005, 0.3, "{:.3f}")
        ttk.Button(f, text="Reset smell & wind to defaults",
                   command=lambda: self.reset_settings(("rate", "diff", "maxn", "turb", "wind_mode", "smoke_obst", "thr"))).pack(anchor=tk.W)
        # --- Build tab
        f = page("Build tab")
        b = box(f, "Map")
        check(b, "Show grid", "show_grid")
        check(b, "Show the fly's trail", "show_trail")
        check(b, "Snap placing to a 5 mm grid", "snap")
        b = box(f, "Layout")
        row = ttk.Frame(b)
        row.pack(anchor=tk.W)
        ttk.Button(row, text="Save layout", command=self.save_layout).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row, text="Load layout", command=self.load_layout).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row, text="Randomize", command=self.randomize_layout).pack(side=tk.LEFT)
        self.layout_msg = tk.StringVar(value="")
        ttk.Label(b, textvariable=self.layout_msg, foreground="#555555").pack(anchor=tk.W, pady=(4, 0))
        # --- Display
        f = page("Display")
        b = box(f, "Window")
        check(b, "Keep this window on top", "topmost")
        check(b, "Start on the Minimize page", "start_min")
        # --- Recording
        f = page("Recording")
        b = box(f, "Save")
        row = ttk.Frame(b)
        row.pack(anchor=tk.W)
        ttk.Button(row, text="Screenshot of the fly view", command=self.save_screenshot).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row, text="Export graph data (CSV)", command=self.export_history).pack(side=tk.LEFT)
        self.rec_msg = tk.StringVar(value="Files go next to flylab.py")
        ttk.Label(b, textvariable=self.rec_msg, foreground="#555555", wraplength=420, justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 0))
        # --- Mapping
        f = page("Mapping")
        b = box(f, "Where the window opens")
        check(b, "Open the window at the saved position", "win_use")
        check(b, "Update the saved position when I close the program", "win_follow")
        pos_msg = tk.StringVar(value="")
        def pos_text():
            p = str(self.S("win_pos"))
            pos_msg.set("Saved position: " + (p.replace(",", ", ") if p else "none (Windows decides)"))
        def save_pos():
            self.sv["win_pos"].set("%d,%d" % (self.root.winfo_x(), self.root.winfo_y()))
            pos_text()
        def forget_pos():
            self.sv["win_pos"].set("")
            pos_text()
        row = ttk.Frame(b)
        row.pack(anchor=tk.W, pady=(4, 0))
        ttk.Button(row, text="Save current position", command=save_pos).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row, text="Forget position", command=forget_pos).pack(side=tk.LEFT)
        ttk.Label(b, textvariable=pos_msg, foreground="#555555").pack(anchor=tk.W, pady=(4, 0))
        pos_text()
        def pulse_ctl(r, hzk, plk):
            sp = ttk.Spinbox(r, from_=20, to=8000, increment=10, width=6, textvariable=self.sv[hzk], format="%.0f")
            sp.pack(side=tk.LEFT, padx=(10, 2))
            spin_wheel(sp, self.sv[hzk], 20, 8000, 10)
            ttk.Label(r, text="Hz").pack(side=tk.LEFT)
            ttk.Checkbutton(r, text="pulse", variable=self.sv[plk]).pack(side=tk.LEFT, padx=(8, 0))
            def preview(*_a):
                try:
                    if self.sv[plk].get():
                        play_pulse(self.sv[hzk].get())
                except Exception:
                    pass
            self.sv[plk].trace_add("write", preview)
        def keyrow(fr, text, key):
            r = ttk.Frame(fr)
            r.pack(fill=tk.X, pady=2)
            ttk.Label(r, text=text, width=26).pack(side=tk.LEFT)
            var = self.sv[key]
            btn = ttk.Button(r, width=14, text=(var.get() or "-").upper())
            btn.pack(side=tk.LEFT)
            pulse_ctl(r, "hz_" + key, "pl_" + key)
            def refresh(*_a):
                btn.configure(text=(var.get() or "-").upper())
            var.trace_add("write", refresh)
            def start():
                btn.configure(text="Press a key...")
                btn.focus_set()
                def on_key(e):
                    btn.unbind("<KeyPress>")
                    if e.keysym != "Escape":
                        k = keysym_to_key(e.keysym)
                        if k:
                            var.set(k)
                    refresh()
                    return "break"
                btn.bind("<KeyPress>", on_key)
            btn.configure(command=start)
        b = box(f, "Keybinds - click a key to change it (Esc cancels). Tick pulse to play a sine pulse at that Hz whenever the key is pressed")
        ttk.Label(b, text="WASD layout", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        for t, k in (("Forward", "kw_f"), ("Back", "kw_b"), ("Left", "kw_l"), ("Right", "kw_r")):
            keyrow(b, t, k)
        ttk.Label(b, text="Arrows layout", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(8, 0))
        for t, k in (("Forward", "ka_f"), ("Back", "ka_b"), ("Left", "ka_l"), ("Right", "ka_r")):
            keyrow(b, t, k)
        ttk.Label(b, text="Other", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(8, 0))
        keyrow(b, "Jump", "kb_jump")
        r = ttk.Frame(b)
        r.pack(fill=tk.X, pady=2)
        ttk.Label(r, text="Food key (tapped per pellet)", width=26).pack(side=tk.LEFT)
        fbtn = ttk.Button(r, width=14, text=(self.food_key or "-").upper())
        fbtn.pack(side=tk.LEFT)
        self._food_btn = fbtn                        # kept in step with the Control Panel's "Map food to" box
        def food_start():
            fbtn.configure(text="Press a key...")
            fbtn.focus_set()
            def on_key(e):
                fbtn.unbind("<KeyPress>")
                if e.keysym != "Escape":
                    k = keysym_to_key(e.keysym)
                    if k:
                        self.set_food_key(k)
                fbtn.configure(text=(self.food_key or "-").upper())
                return "break"
            fbtn.bind("<KeyPress>", on_key)
        fbtn.configure(command=food_start)
        def food_clear():
            self.set_food_key("")
            fbtn.configure(text="-")
        ttk.Button(r, text="Clear", width=7, command=food_clear).pack(side=tk.LEFT, padx=(6, 0))
        pulse_ctl(r, "hz_food", "pl_food")
        ttk.Button(b, text="Reset keybinds", command=lambda: self.reset_settings(["kw_f", "kw_b", "kw_l", "kw_r", "ka_f", "ka_b", "ka_l", "ka_r", "kb_jump"])).pack(anchor=tk.W, pady=(8, 0))
        b = box(f, "Screen select")
        reg_msg = tk.StringVar(value="")
        def reg_text():
            r_ = self.capture_region
            reg_msg.set("Capture region: x %d, y %d, %d x %d px" % (r_["left"], r_["top"], r_["width"], r_["height"]))
            try:
                self.root.after(1000, reg_text)
            except Exception:
                pass
        def reset_region():
            self.capture_region = dict(DEFAULT_CAPTURE_REGION)
            try:
                with open(REGION_FILE, "w") as fh:
                    json.dump(self.capture_region, fh)
            except Exception:
                pass
        ttk.Label(b, textvariable=reg_msg, foreground="#555555").pack(anchor=tk.W)
        row = ttk.Frame(b)
        row.pack(anchor=tk.W, pady=(6, 0))
        ttk.Button(row, text="Select capture region", command=lambda: self.select_region(force=True)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row, text="Start mapping (aligned)", command=self.align_region).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row, text="Reset region", command=reset_region).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row, text="Map Blooket buttons", command=self.map_blooket_buttons).pack(side=tk.LEFT)
        ttk.Label(b, text="Select: drag a box around the fly. Start mapping (aligned): a right-angle corner marker sits on the pointer with thin red guide lines across the screen, to line up a pop-up\'s corner exactly. Blooket buttons are mapped for the game mode chosen on the Control Panel.", foreground="#555555",
                  wraplength=460, justify=tk.LEFT).pack(anchor=tk.W, pady=(6, 0))
        reg_text()
        # --- Profiles
        f = page("Profiles")
        b = box(f, "Presets")
        combo(b, "Profile", "profile", tuple(PROFILES))
        row = ttk.Frame(b)
        row.pack(anchor=tk.W, pady=(4, 0))
        ttk.Button(row, text="Apply profile", command=self.apply_profile).pack(side=tk.LEFT, padx=(0, 6))
        self.red_button(row, "Reset ALL settings", lambda: self.reset_settings()).pack(side=tk.LEFT)
        ttk.Label(b, text="Settings are saved automatically to flylab_settings.json and loaded next time.", foreground="#555555",
                  wraplength=420, justify=tk.LEFT).pack(anchor=tk.W, pady=(6, 0))
        # --- Theme
        f = page("Theme")
        b = box(f, "Appearance")
        combo(b, "Theme", "theme", ("Light", "Dark", "System"))
        ncv = tk.Canvas(b, width=260, height=22, bg=bg, highlightthickness=0, bd=0)       # inset note, same style as the top-left one
        ncv.pack(anchor=tk.W, pady=(4, 2))
        ncv.create_text(2, 12, text="More themes coming...", anchor="w", font=(pix, 9), fill="#ffffff")
        ncv.create_text(1, 11, text="More themes coming...", anchor="w", font=(pix, 9), fill="#7a7a7a")
        b = box(f, "When the program opens")
        ttk.Radiobutton(b, text="Randomize Themes everytime when opening", variable=self.sv["theme_mode"], value="Randomize").pack(anchor=tk.W, pady=1)
        ttk.Radiobutton(b, text="Default", variable=self.sv["theme_mode"], value="Default").pack(anchor=tk.W, pady=1)
        ttk.Label(b, text="Default = the theme chosen above. Randomize = Light or Dark at random each start.", foreground="#555555",
                  wraplength=420, justify=tk.LEFT).pack(anchor=tk.W, pady=(6, 0))
        # --- Advanced
        f = page("Advanced")
        b = box(f, "Debug")
        check(b, "Print frame timings to the console (every 2 s)", "print_timings")
        ttk.Button(b, text="Copy diagnostic report", command=self.copy_report).pack(anchor=tk.W, pady=(4, 0))
        self.adv_msg = tk.StringVar(value="")
        ttk.Label(b, textvariable=self.adv_msg, foreground="#555555").pack(anchor=tk.W, pady=(4, 0))
        # --- Suggestions
        f = page("Suggestions")
        b = box(f, "Tell me what to add or fix")
        ttk.Label(b, text="Send suggestions or bug reports", foreground="#555555").pack(anchor=tk.W)
        self.sg_text = tk.Text(b, width=62, height=6, wrap="word", font=("Segoe UI", 10), relief=tk.FLAT, highlightthickness=1, highlightbackground="#bbbbbb")
        self.sg_text.pack(anchor=tk.W, pady=(4, 6))
        self.sg_msg = tk.StringVar(value="")
        row = ttk.Frame(b)
        row.pack(anchor=tk.W)
        for t, c in (("Send to Discord", self.sg_discord),):
            ttk.Button(row, text=t, command=c).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(b, textvariable=self.sg_msg, foreground="#555555", wraplength=460, justify=tk.LEFT).pack(anchor=tk.W, pady=(6, 0))
        srow = ttk.Frame(b)
        srow.pack(anchor=tk.W, pady=(10, 0))
        self._led = tk.Canvas(srow, width=16, height=16, bg=bg, highlightthickness=0, bd=0)
        self._led.pack(side=tk.LEFT, padx=(0, 6))
        self._led_dot = self._led.create_oval(2, 2, 14, 14, fill="#8a8a8a", outline="")
        self._led.create_oval(4, 3, 8, 7, fill="#ffffff", outline="", stipple="gray50")       # little highlight
        self._led_txt = tk.StringVar(value=SUGGEST_STATUS)
        ttk.Label(srow, textvariable=self._led_txt).pack(side=tk.LEFT)
        self._dc_state = {"status": None, "note": ""}              # written by the Discord thread, read here (plain strings only)
        self._start_discord()
        self._poll_status()
        filler.pack(fill=tk.X)
        def relayout(_e=None):
            sc_cv.itemconfigure(win, width=max(50, sc_cv.winfo_width()))
            last = pages[names[-1]]
            filler.configure(height=max(1, sc_cv.winfo_height() - last.winfo_height() - 24))      # lets the last heading scroll to the top
            sc_cv.configure(scrollregion=sc_cv.bbox("all"))
        sc_cv.bind("<Configure>", relayout)
        holder.bind("<Configure>", lambda _e: sc_cv.configure(scrollregion=sc_cv.bbox("all")))
        def jump(_e=None):
            sel = lb.curselection()
            if not sel:
                return
            relayout()
            total = max(1, holder.winfo_height())
            sc_cv.yview_moveto(max(0.0, (pages[names[sel[0]]].winfo_y() - 2) / total))
        lb.bind("<<ListboxSelect>>", jump)
        def wheel(e):
            sc_cv.yview_scroll(int(-e.delta / 120) * 3, "units")
            sync()
        def sync(*_a):
            # highlight the category whose heading is nearest the top of the view
            try:
                top = sc_cv.canvasy(0)
                cur = 0
                for i, n in enumerate(names):
                    if pages[n].winfo_y() <= top + 30:
                        cur = i
                lb.selection_clear(0, tk.END)
                lb.selection_set(cur)
            except Exception:
                pass
        area.bind("<Enter>", lambda _e: sc_cv.bind_all("<MouseWheel>", wheel))
        area.bind("<Leave>", lambda _e: sc_cv.unbind_all("<MouseWheel>"))
        sbar.bind("<B1-Motion>", lambda _e: sync())
        sbar.bind("<ButtonRelease-1>", lambda _e: sync())
        lb.selection_set(0)
    def red_button(self, parent, text, command):
        """A red, white-text button that has exactly the size, font and rounded corners of the neighbouring ttk buttons (native ttk buttons can't be recoloured)."""
        import tkinter.font as tkfont
        font = tkfont.nametofont("TkDefaultFont")
        w, h = font.measure(text) + 26, 26
        try:                                                      # measure a real ttk button so the size matches to the pixel
            ref = ttk.Button(parent, text=text)
            ref.update_idletasks()
            w, h = max(w, int(ref.winfo_reqwidth())), max(h, int(ref.winfo_reqheight()))
            ref.destroy()
        except Exception:
            pass
        bg = ttk.Style().lookup("TFrame", "background") or "#f0f0f0"
        cv = tk.Canvas(parent, width=w, height=h, bg=bg, highlightthickness=0, bd=0, cursor="hand2")
        state = {"over": False, "down": False}
        def draw():
            cv.delete("all")
            fill = "#a52215" if state["down"] else ("#b82819" if state["over"] else "#c42b1c")
            x0, y0, x1, y1, r = 1, 1, w - 1, h - 1, 4
            cv.create_polygon(x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0,
                              smooth=True, fill=fill, outline=fill)
            cv.create_text(w / 2.0, h / 2.0, text=text, fill="#ffffff", font=font)
        def press(_e):
            state["down"] = True
            draw()
        def release(e):
            was = state["down"]
            state["down"] = False
            draw()
            if was and 0 <= e.x <= w and 0 <= e.y <= h:
                command()
        cv.bind("<ButtonPress-1>", press)
        cv.bind("<ButtonRelease-1>", release)
        cv.bind("<Enter>", lambda _e: (state.__setitem__("over", True), draw()))
        cv.bind("<Leave>", lambda _e: (state.__setitem__("over", False), draw()))
        draw()
        return cv
    def _here(self, name):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    def save_layout(self):
        try:
            with open(LAYOUT_FILE, "w", encoding="utf-8") as f:
                json.dump({"obstacles": self.arena.obstacles, "foods": self.arena.foods, "food_spawn": self.arena.food_spawn}, f)
            self.layout_msg.set("Saved to flylab_layout.json")
        except Exception as err:
            self.layout_msg.set(f"Could not save: {err}")
    def load_layout(self):
        try:
            with open(LAYOUT_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
            self.arena.obstacles = [list(map(float, o)) for o in d["obstacles"]]
            self.arena.foods = [list(map(float, o)) for o in d["foods"]]
            self.arena.food_spawn = [list(map(float, o)) for o in d.get("food_spawn", [o[:2] for o in d["foods"]])]
            self.arena.version += 1
            self.layout_msg.set("Loaded flylab_layout.json")
        except Exception as err:
            self.layout_msg.set(f"Nothing loaded: {err}")
    def randomize_layout(self):
        rng = random.Random()
        obs, foods = [], []
        for _ in range(rng.randint(4, 7)):
            for _try in range(40):
                x, y, r = rng.uniform(-32, 32), rng.uniform(-32, 32), rng.uniform(2.5, 6.0)
                if math.hypot(x, y) > r + 7 and all(math.hypot(x - o[0], y - o[1]) > r + o[2] + 3 for o in obs):
                    obs.append([x, y, r])
                    break
        for _ in range(6):
            for _try in range(60):
                x, y = rng.uniform(-34, 34), rng.uniform(-34, 34)
                if math.hypot(x, y) > 10 and all(math.hypot(x - o[0], y - o[1]) > o[2] + 2.5 for o in obs):
                    foods.append([x, y, 1.0])
                    break
        self.arena.obstacles, self.arena.foods = obs, foods
        self.arena.food_spawn = [[f[0], f[1]] for f in foods]
        self.arena.version += 1
        self.layout_msg.set(f"Random layout: {len(obs)} obstacles, {len(foods)} food")
    def save_screenshot(self):
        try:
            img = self.body.render(VIEW_W, VIEW_H)
            path = self._here(time.strftime("flylab_shot_%Y%m%d_%H%M%S.png"))
            cv2.imwrite(path, img)
            self.rec_msg.set("Saved " + os.path.basename(path))
        except Exception as err:
            self.rec_msg.set(f"Could not save: {err}")
    def export_history(self):
        try:
            rows = list(self.hist)
            path = self._here(time.strftime("flylab_graph_%Y%m%d_%H%M%S.csv"))
            t0 = rows[0][0] if rows else 0.0
            with open(path, "w", encoding="utf-8") as f:
                f.write("time_s,spikes_per_s,drive,frame_ms\n")
                for r in rows:
                    f.write(f"{r[0] - t0:.3f},{r[1]:.1f},{r[2]:.3f},{r[3]:.2f}\n")
            self.rec_msg.set(f"Saved {os.path.basename(path)} ({len(rows)} rows)")
        except Exception as err:
            self.rec_msg.set(f"Could not save: {err}")
    def copy_report(self):
        lines = [f"python {sys.version.split()[0]}", f"platform {sys.platform}", f"opencv {cv2.__version__}", f"numpy {np.__version__}"]
        for mod in ("mujoco", "flygym"):
            try:
                lines.append(f"{mod} {__import__(mod).__version__}")
            except Exception:
                lines.append(f"{mod} not available")
        lines.append(f"fps now {getattr(self, '_fps', 0.0):.1f}   body error: {getattr(self.body, 'error', None)}")
        lines += [f"{k} = {v.get()}" for k, v in self.sv.items()]
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append("\n".join(lines))
            self.adv_msg.set("Copied to the clipboard")
        except Exception as err:
            self.adv_msg.set(f"Could not copy: {err}")
    def on_tab_changed(self, _e=None):
        try:
            if self.nb.select() == str(self.tab_min):
                self.root.geometry(f"{MINI_W}x{MINI_H}")
            else:
                self.root.geometry("")                         # back to the natural (full) size
        except Exception:
            pass
    def refresh_food_box(self):
        try:
            self._food_btn.configure(text=(self.food_key or "-").upper())
        except Exception:
            pass
        if self.food_key:
            self.food_lbl.configure(text=self.food_key.upper(), fg="black")
        else:
            self.food_lbl.configure(text="none", fg="#999999")
        self.food_trash.configure(state="normal")
    def food_listen(self, on):
        self.food_listening = on
        if on:
            self.food_box.focus_set()
        else:
            self.root.focus_set()
        self.refresh_food_box()
    def food_keypress(self, event):
        if event.keysym == "Escape":
            return
        k = keysym_to_key(event.keysym)
        if k:
            self.set_food_key(k)
    def set_food_key(self, key):
        self.food_key = key
        save_food_key(key)
        self.refresh_food_box()
        try:
            self._sched_apply()
        except Exception:
            pass
    def update_source_states(self):
        """Only allow selecting a capture area when Screen is the active fly source."""
        screen_selected = self.source.get() == "screen"
        if hasattr(self, "select_region_btn"):
            self.select_region_btn.configure(state=("normal" if screen_selected else "disabled"))
        # Force the next preview update to repaint even if no new capture frame arrived.
        self.shown_seq = -1

    def update_preview(self):
        """Show captured video only for Screen source; otherwise show a black notice."""
        if self.terminating:
            return

        if self.source.get() != "screen":
            frame = np.zeros((PREVIEW_MAX, PREVIEW_MAX, 3), dtype=np.uint8)
            text = "Source not selected."
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale, thickness = 0.55, 1
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            cv2.putText(frame, text, ((PREVIEW_MAX - tw) // 2,
                                      (PREVIEW_MAX + th) // 2),
                        font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
            png = to_png(frame)
            if png is not None:
                try:
                    self.preview_img = tk.PhotoImage(data=base64.b64encode(png))
                    self.preview_label.configure(
                        image=self.preview_img, text="", background="black")
                except tk.TclError:
                    pass
            self.root.after(33, self.update_preview)
            return

        png, seq = self.latest_png, self.preview_seq
        if png is not None and seq != self.shown_seq:
            try:
                self.preview_img = tk.PhotoImage(data=base64.b64encode(png))
                self.preview_label.configure(image=self.preview_img, text="", background="black")
                self.shown_seq = seq
            except tk.TclError:
                pass
        self.root.after(33, self.update_preview)

    def refresh_map_note(self):
        mode = self.game_mode.get()
        pts = self.blooket_map.get(mode, {})
        if pts:
            got = [("Continue" if k == "continue" else f"answer {k}") for k, _ in MAP_STEPS if k in pts]
            self.map_note.set(f"{mode}: mapped " + ", ".join(got))
        elif mode == "Classic":
            self.map_note.set("Classic: not mapped - uses the 1-4 keys and Enter.")
        else:
            self.map_note.set(f"{mode}: not mapped - answers use the 1-4 keys and nothing is clicked for Continue. "
                              "Press 'Map buttons...'.")
    def add_blooket_mode(self):
        name = simpledialog.askstring("New game mode", "Name of the Blooket game mode:", parent=self.root)
        name = (name or "").strip()
        if not name or name in self.blooket_modes:
            if name in self.blooket_modes:
                self.game_mode.set(name)
            return
        self.blooket_modes.append(name)
        self.game_combo.configure(values=self.blooket_modes)
        save_blooket_map(self.blooket_modes, self.blooket_map)
        self.game_mode.set(name)
        self.map_blooket_buttons()
    def map_blooket_buttons(self):
        """Dim the screen and record where the answer buttons and Continue are for the chosen game mode.
        Left click = record this button, right click = skip it (clears it), Esc = cancel without saving."""
        if self.selecting or self.terminating:
            return
        mode = self.game_mode.get()
        self.selecting = True
        self.clear_keys()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        ov = tk.Toplevel(self.root)
        ov.overrideredirect(True)
        ov.geometry(f"{sw}x{sh}+0+0")
        ov.attributes("-topmost", True)
        ov.attributes("-alpha", 0.35)
        ov.configure(bg="black")
        canvas = tk.Canvas(ov, bg="black", highlightthickness=0, cursor="crosshair")
        canvas.pack(fill=tk.BOTH, expand=True)
        title = canvas.create_text(sw // 2, 50, fill="white", font=("Arial", 20, "bold"), text="")
        canvas.create_text(sw // 2, 88, fill="#cccccc", font=("Arial", 13),
                                 text="Left click = record   |   Right click = skip (leave unmapped)   |   Esc = cancel")
        st = {"i": 0, "new": {}}
        def say():
            key, label = MAP_STEPS[st["i"]]
            canvas.itemconfigure(title, text=f"{mode}: click {label}   (step {st['i'] + 1} of {len(MAP_STEPS)})")
        def finish(save):
            if save:
                self.blooket_map[mode] = dict(st["new"])
                save_blooket_map(self.blooket_modes, self.blooket_map)
                print(f"[✓] Button map for {mode}: {self.blooket_map[mode]}")
            ov.destroy()
            self.selecting = False
            self.refresh_map_note()
        def advance():
            st["i"] += 1
            if st["i"] >= len(MAP_STEPS):
                canvas.itemconfigure(title, text=f"{mode}: saved")
                ov.after(350, lambda: finish(True))
            else:
                say()
        def on_left(e):
            if st["i"] >= len(MAP_STEPS):
                return
            key, _label = MAP_STEPS[st["i"]]
            st["new"][key] = [int(e.x_root), int(e.y_root)]
            canvas.create_oval(e.x - 12, e.y - 12, e.x + 12, e.y + 12, outline="#44ff88", width=3)
            canvas.create_text(e.x, e.y - 24, fill="#44ff88", font=("Arial", 14, "bold"), text="Continue" if key == "continue" else key)
            advance()
        def on_right(e):
            if st["i"] < len(MAP_STEPS):
                advance()
        for key, pt in self.blooket_map.get(mode, {}).items():     # show what is mapped now, faintly
            x, y = pt[0] - ov.winfo_rootx(), pt[1] - ov.winfo_rooty()
            canvas.create_oval(x - 8, y - 8, x + 8, y + 8, outline="#777777", width=2)
        canvas.bind("<ButtonPress-1>", on_left)
        canvas.bind("<ButtonPress-3>", on_right)
        ov.bind("<Escape>", lambda e: finish(False))
        say()
        ov.focus_force()
    def select_region(self, force=False):
        """Dim the screen and let the user drag a box around the fly. Esc cancels."""
        if self.selecting or self.terminating or (self.source.get() != "screen" and not force):
            return
        self.selecting = True
        self.clear_keys()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        ov = tk.Toplevel(self.root)
        ov.overrideredirect(True)
        ov.geometry(f"{sw}x{sh}+0+0")
        ov.attributes("-topmost", True)
        ov.attributes("-alpha", 0.3)
        ov.configure(bg="black")
        canvas = tk.Canvas(ov, bg="black", highlightthickness=0, cursor="crosshair")
        canvas.pack(fill=tk.BOTH, expand=True)
        canvas.create_text(sw // 2, 50, fill="white", font=("Arial", 18, "bold"),
                           text="Drag a box around the fly's body and legs.  Esc = cancel")
        st = {"x0": 0, "y0": 0, "rect": None}
        def finish():
            ov.destroy()
            self.selecting = False
        def on_press(e):
            st["x0"], st["y0"] = e.x_root, e.y_root
            st["rect"] = canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="red", width=3)
        def on_drag(e):
            if st["rect"] is not None:
                canvas.coords(st["rect"], st["x0"], st["y0"], e.x_root, e.y_root)
        def on_release(e):
            left, top = min(st["x0"], e.x_root), min(st["y0"], e.y_root)
            w, h = abs(e.x_root - st["x0"]), abs(e.y_root - st["y0"])
            if w >= 20 and h >= 20:
                self.capture_region = {"top": int(top), "left": int(left), "width": int(w), "height": int(h)}
                try:
                    with open(REGION_FILE, "w") as f:
                        json.dump(self.capture_region, f)
                except Exception as err:
                    print(f"[!] Couldn't save region: {err}")
                print(f"[✓] New capture region: {self.capture_region}")
            finish()
        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        ov.bind("<Escape>", lambda e: finish())
        ov.focus_force()
    def align_region(self):
        """Aligned region select. A right-angle corner marker (arms pointing down and to the right) has its tip exactly on the mouse, and thin red lines
        run across the whole screen through the tip, so a pop-up's top-left corner can be lined up exactly. Press = set that corner, drag, release = bottom-right.
        Esc cancels."""
        if self.selecting or self.terminating:
            return
        self.selecting = True
        self.clear_keys()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        ov = tk.Toplevel(self.root)
        ov.overrideredirect(True)
        ov.geometry(f"{sw}x{sh}+0+0")
        ov.attributes("-topmost", True)
        ov.attributes("-alpha", 0.35)
        ov.configure(bg="black")
        canvas = tk.Canvas(ov, bg="black", highlightthickness=0, cursor="crosshair")
        try:
            canvas.configure(cursor="none")                    # the marker's tip IS the pointer
        except Exception:
            pass
        canvas.pack(fill=tk.BOTH, expand=True)
        canvas.create_text(sw // 2, 50, fill="white", font=("Arial", 18, "bold"),
                           text="Put the corner's tip on the top-left of the area, press, drag to the bottom-right, release.  Esc = cancel")
        RED, ARM = "#ff3030", 70
        hl = canvas.create_line(0, 0, sw, 0, fill=RED, width=1)             # thin red guides across the whole screen
        vl = canvas.create_line(0, 0, 0, sh, fill=RED, width=1)
        sh_b = canvas.create_line(0, 0, 0, 0, 0, 0, fill="black", width=7)  # corner marker: dark outline under a white right angle
        sh_w = canvas.create_line(0, 0, 0, 0, 0, 0, fill="white", width=3)
        lab = canvas.create_text(0, 0, text="", fill="white", font=("Consolas", 11, "bold"), anchor="nw")
        rect = canvas.create_rectangle(0, 0, 0, 0, outline=RED, width=1, state="hidden")
        hl2 = canvas.create_line(0, 0, sw, 0, fill=RED, width=1, state="hidden")       # guides through the dragged-to corner
        vl2 = canvas.create_line(0, 0, 0, sh, fill=RED, width=1, state="hidden")
        st = {"x0": None, "y0": None, "xr": 0, "yr": 0, "open": True}
        def corner(x, y):
            pts = (x + ARM, y, x, y, x, y + ARM)
            canvas.coords(sh_b, *pts)
            canvas.coords(sh_w, *pts)
        def show(xr, yr):
            """Draw everything for pointer position (xr, yr) in screen pixels."""
            x, y = xr - ov.winfo_rootx(), yr - ov.winfo_rooty()
            canvas.coords(hl, 0, y, sw, y)
            canvas.coords(vl, x, 0, x, sh)
            corner(x, y)
            if st["x0"] is None:
                canvas.itemconfigure(lab, text="%d, %d" % (xr, yr))
            else:
                x0, y0 = st["x0"], st["y0"]
                canvas.itemconfigure(rect, state="normal")
                canvas.coords(rect, x0, y0, x, y)
                canvas.coords(hl2, 0, y0, sw, y0)
                canvas.coords(vl2, x0, 0, x0, sh)
                canvas.itemconfigure(hl2, state="normal")
                canvas.itemconfigure(vl2, state="normal")
                canvas.itemconfigure(lab, text="%d x %d" % (abs(xr - st["xr"]), abs(yr - st["yr"])))
            canvas.coords(lab, x + 14, y + 14)
        def tick():
            # follow the pointer by polling, so the marker is there at once (before any mouse movement) and never stalls
            if not st["open"]:
                return
            try:
                px, py = ov.winfo_pointerxy()
                show(px, py)
                ov.after(15, tick)
            except Exception:
                pass
        def press(e):
            st["x0"], st["y0"] = e.x_root - ov.winfo_rootx(), e.y_root - ov.winfo_rooty()
            st["xr"], st["yr"] = e.x_root, e.y_root
        def release(e):
            if st["x0"] is None:
                return
            left, top = min(st["xr"], e.x_root), min(st["yr"], e.y_root)
            w, h = abs(e.x_root - st["xr"]), abs(e.y_root - st["yr"])
            if w >= 20 and h >= 20:
                self.capture_region = {"top": int(top), "left": int(left), "width": int(w), "height": int(h)}
                try:
                    with open(REGION_FILE, "w") as fh:
                        json.dump(self.capture_region, fh)
                except Exception as err:
                    print(f"[!] Couldn't save region: {err}")
                print(f"[✓] New capture region: {self.capture_region}")
                self.root.after(1, lambda: None)
            finish()
        def finish(_e=None):
            st["open"] = False
            try:
                ov.destroy()
            except Exception:
                pass
            self.selecting = False
        canvas.bind("<ButtonPress-1>", press)
        canvas.bind("<ButtonRelease-1>", release)
        for w_ in (ov, canvas):
            w_.bind("<Escape>", finish)
        ov.lift()
        ov.focus_force()
        canvas.focus_set()
        tick()
    # ------------------------------------------------------------------
    # INPUT HELPERS
    # ------------------------------------------------------------------
    def klog(self, text):
        """Record one input event (thread-safe; the UI thread shows it)."""
        t = time.time()
        self.keylog_q.put(time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1000):03d}  {text}")
    def hw_click(self, tx, ty):
        self.klog(f"click   ({int(tx)}, {int(ty)})")
        pydirectinput.moveTo(int(tx), int(ty))
        time.sleep(0.08)
        pydirectinput.mouseDown()
        time.sleep(0.05)
        pydirectinput.mouseUp()
        time.sleep(0.05)
    def tap(self, key, hold_time=0.04):
        """A clean key tap that games register reliably."""
        self.klog(f"tap     {key}")
        hz = (self.cfg.get("pulse") or {}).get(key)
        if hz:
            play_pulse(hz)
        pydirectinput.keyDown(key)
        time.sleep(hold_time)
        pydirectinput.keyUp(key)
    def hold(self, key, down):
        """Press/release a key only when its state actually changes."""
        with self.key_lock:
            if down and key not in self.held:
                self.klog(f"down    {key}")
                hz = (self.cfg.get("pulse") or {}).get(key)
                if hz:
                    play_pulse(hz)
                pydirectinput.keyDown(key)
                self.held.add(key)
            elif not down and key in self.held:
                self.klog(f"up      {key}")
                pydirectinput.keyUp(key)
                self.held.discard(key)
    def clear_keys(self):
        with self.key_lock:
            if self.held:
                self.klog("release " + " ".join(sorted(self.held)))
            for k in list(self.held):
                pydirectinput.keyUp(k)
            self.held.clear()
    def fire_blooket(self, num_key):
        # Lock BEFORE the thread starts so two ticks can't launch two answers
        self.blooket_action_lock = True
        threading.Thread(target=self.execute_blooket_sequence, args=(num_key,), daemon=True).start()
    def execute_blooket_sequence(self, num_key):
        """Picks answer num_key (click its mapped button, else press the number key), then advances."""
        try:
            mode_choice = self.cfg["game"]
            pts = self.blooket_map.get(mode_choice, {})
            if num_key in pts:
                print(f"[➔] Blooket ({mode_choice}): clicking answer {num_key} at {tuple(pts[num_key])}")
                self.hw_click(pts[num_key][0], pts[num_key][1])
            else:
                print(f"[➔] Blooket ({mode_choice}): pressing '{num_key}'")
                self.tap(num_key)
            time.sleep(0.350)
            if "continue" in pts:
                self.hw_click(pts["continue"][0], pts["continue"][1])
                print(f"[✓] {mode_choice}: clicked Continue at {tuple(pts['continue'])}")
            elif mode_choice == "Classic":
                self.tap('enter')
                print("[✓] Classic Submission: Enter Pressed.")
            else:
                print(f"[!] {mode_choice}: no Continue button mapped - press 'Map buttons...' to set it.")
            time.sleep(0.500)
        except Exception as e:
            print(f"[!] Blooket submission failure: {e}")
        finally:
            self.blooket_action_lock = False
    def execute_auto_replay(self):
        try:
            self.clear_keys()
            original_pos = None
            try:
                original_pos = pydirectinput.position()  # where the mouse is right now
            except Exception:
                pass
            print(f"[⟳] Fly stopped for {AUTO_REPLAY_STILL_SECONDS:.1f}s: clicking replay at {AUTO_REPLAY_SPOT}")
            self.hw_click(AUTO_REPLAY_SPOT[0], AUTO_REPLAY_SPOT[1])
            time.sleep(0.15)
            self.hw_click(AUTO_REPLAY_RETURN_SPOT[0], AUTO_REPLAY_RETURN_SPOT[1])  # back to Roblox
            time.sleep(0.05)
            if original_pos is not None:
                pydirectinput.moveTo(int(original_pos[0]), int(original_pos[1]))
            print("[✓] Auto-Replay click sent, focus returned, mouse restored.")
        except Exception as e:
            print(f"[!] Auto-Replay failure: {e}")
        finally:
            self.replay_in_progress = False
    def execute_building_macro(self):
        try:
            print("[⚡] Executing Block Placement Macro Sequence...")
            self.clear_keys()
            pydirectinput.keyUp('space')
            time.sleep(0.05)
            self.tap('q')
            time.sleep(0.2)
            self.hw_click(SPAWN_MENU_X, SPAWN_MENU_Y)
            time.sleep(0.2)
            self.hw_click(BLOCK_X, BLOCK_Y)
            time.sleep(0.2)
            self.tap('q')
            time.sleep(0.2)
            self.hw_click(SPAWN_GROUND_X, SPAWN_GROUND_Y)
            print("[✓] Block sequence completed cleanly.")
        except Exception as e:
            print(f"[!] Macro Failure: {e}")
        finally:
            self.macro_in_progress = False
    # ------------------------------------------------------------------
    # CLEAN SHUTDOWN
    # ------------------------------------------------------------------
    def terminate(self):
        """Step 1: tell the vision thread to stop. Never blocks the UI thread."""
        if self.terminating:
            return
        self.terminating = True
        try:
            if self.S("win_follow"):
                self.sv["win_pos"].set("%d,%d" % (self.root.winfo_x(), self.root.winfo_y()))
                self.save_settings()
        except Exception:
            pass
        self.stop_event.set()
        self.sim.stop()
        try:
            if hasattr(self, "body"):
                self.body.close()
        except Exception:
            pass
        self.root.after(30, self.finish_terminate, 0)
    def finish_terminate(self, waited_ms):
        """Step 2: once the vision thread has exited, release keys and close the window."""
        if self.vision_thread.is_alive() and waited_ms < 2000:
            self.root.after(30, self.finish_terminate, waited_ms + 30)
            return
        extra = []
        try:
            kb = self.cfg.get("kb") or {}
            extra = list(kb.get("wasd", ())) + list(kb.get("arrows", ())) + [kb.get("jump", "space")]
        except Exception:
            pass
        for k in list(ALL_KEYS) + [e for e in extra if e not in ALL_KEYS]:
            try:
                pydirectinput.keyUp(k)
            except Exception:
                pass
        self.root.destroy()
    # ------------------------------------------------------------------
    # VISION LOOP (runs in its own thread; never touches Tk objects)
    # ------------------------------------------------------------------
    def run_vision(self):
        sct = mss.mss()
        prev_gray, jump_cooldown, move_history, last_action, debounce_ticks = None, 0, [], "NONE", 0
        still_since, replay_armed = None, True
        filt = MotionFilter()
        bk_dir, bk_since, bk_last, bk_armed, bk_neutral, bk_fired = None, 0.0, 0.0, True, 0.0, None
        try:
            while not self.stop_event.is_set():
                try:
                    screenshot = np.array(sct.grab(self.capture_region))
                    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)
                    if prev_gray is not None and prev_gray.shape != gray.shape:
                        prev_gray = None  # the capture box was changed
                    dx = dy = 0.0
                    if self.cfg["source"] not in ("brain", "screen"):
                        prev_gray = gray
                        time.sleep(0.03)
                        continue
                    if prev_gray is not None:
                        if self.cfg["source"] == "brain":
                            dx, dy = self.brain_motion()
                        else:
                            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                            dx, dy = float(np.mean(flow[..., 0])), float(np.mean(flow[..., 1]))
                        current_action = "NONE"
                        busy = (self.macro_in_progress or self.blooket_action_lock
                                or self.selecting or self.replay_in_progress)
                        now_t = time.monotonic()
                        on_th = float(self.cfg["sens"])
                        sdx, sdy = filt.update(dx, dy, now_t, on_th, float(self.cfg["smooth"]))
                        self.last_motion, self.last_dirs = (filt.vx, filt.vy), (sdx, sdy)   # for the Graphs window
                        # --- AUTO-REPLAY: click play once the fly has been still long enough ---
                        moving = max(abs(filt.vx), abs(filt.vy)) >= on_th * 0.8
                        if not self.cfg["replay"]:
                            still_since, replay_armed = None, True
                        elif busy:
                            still_since = None
                        elif moving:
                            still_since, replay_armed = None, True  # motion re-arms the next replay
                        elif replay_armed:
                            now = time.monotonic()
                            if still_since is None:
                                still_since = now
                            elif now - still_since >= AUTO_REPLAY_STILL_SECONDS:
                                replay_armed, still_since = False, None
                                self.replay_in_progress = True
                                threading.Thread(target=self.execute_auto_replay, daemon=True).start()
                        if self.cfg["jump"] and not busy:
                            if dy > 1.2 and jump_cooldown == 0:
                                self.tap((self.cfg.get("kb") or {}).get("jump", 'space'))
                                jump_cooldown = 15
                        if jump_cooldown > 0:
                            jump_cooldown -= 1
                        if self.cfg["walk"] and not busy:
                            mode = self.cfg["mode"]
                            if mode == "blooket":
                                # Pick ONE answer from the steadiest direction, once, then wait (no more 2222222...)
                                d = filt.dominant(on_th)
                                key = {"W": "2", "A": "1", "S": "4", "D": "3"}.get(d)
                                if key is None:
                                    if bk_neutral == 0.0:
                                        bk_neutral = now_t
                                    elif now_t - bk_neutral >= 0.25:
                                        bk_armed = True                      # back to neutral: ready for the next answer
                                    bk_dir = None
                                else:
                                    bk_neutral = 0.0
                                    if key != bk_dir:
                                        bk_dir, bk_since = key, now_t
                                        if key != bk_fired:
                                            bk_armed = True
                                    if (not bk_armed and key == bk_fired and now_t - bk_last >= BLOOKET_REPEAT):
                                        bk_armed = True                      # same heading for a long time: answer again
                                    if bk_armed and now_t - bk_since >= BLOOKET_CONFIRM and now_t - bk_last >= BLOOKET_COOLDOWN:
                                        self.fire_blooket(key)
                                        bk_armed, bk_last, bk_fired = False, now_t, key
                            else:
                                kbm = self.cfg.get("kb") or {}
                                if mode == "wasd":
                                    uk, dk, lk, rk = kbm.get("wasd", ('w', 's', 'a', 'd'))
                                else:
                                    uk, dk, lk, rk = kbm.get("arrows", ('up', 'down', 'left', 'right'))
                                if sdy > 0:
                                    self.hold(uk, True); self.hold(dk, False); current_action = "W"
                                elif sdy < 0:
                                    self.hold(dk, True); self.hold(uk, False); current_action = "S"
                                else:
                                    self.hold(uk, False); self.hold(dk, False)
                                if sdx > 0:
                                    self.hold(lk, True); self.hold(rk, False)
                                elif sdx < 0:
                                    self.hold(rk, True); self.hold(lk, False)
                                else:
                                    self.hold(lk, False); self.hold(rk, False)
                                if debounce_ticks > 0:
                                    debounce_ticks -= 1
                                if current_action != "NONE" and current_action != last_action and debounce_ticks == 0:
                                    move_history.append(current_action)
                                    last_action = current_action
                                    if len(move_history) > 3:
                                        move_history.pop(0)
                                    if move_history == ["W", "S", "W"]:
                                        move_history.clear(); last_action = "NONE"; debounce_ticks = 120
                                        if self.cfg["macro"]:
                                            self.macro_in_progress = True
                                            self.clear_keys()
                                            threading.Thread(target=self.execute_building_macro, daemon=True).start()
                                        prev_gray = None
                                        continue
                        else:
                            if not busy:
                                self.clear_keys()
                    # --- LIVE PREVIEW: captured area + arrow showing the detected motion ---
                    # Keep the preview canvas a constant square size. If the captured
                    # area is wide or tall, stretching it here prevents the window
                    # geometry from changing when the source is switched to Brain.
                    frame = cv2.resize(
                        cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR),
                        (PREVIEW_MAX, PREVIEW_MAX),
                    )
                    cx, cy = PREVIEW_MAX // 2, PREVIEW_MAX // 2
                    ex = int(np.clip(cx + filt.vx * 60, 0, PREVIEW_MAX - 1))
                    ey = int(np.clip(cy + filt.vy * 60, 0, PREVIEW_MAX - 1))
                    cv2.arrowedLine(frame, (cx, cy), (ex, ey), (0, 255, 0), 2, tipLength=0.3)
                    cv2.putText(frame, f"dx {filt.vx:+.2f}  dy {filt.vy:+.2f}  {'W' if filt.sy > 0 else 'S' if filt.sy < 0 else '-'}{'A' if filt.sx > 0 else 'D' if filt.sx < 0 else '-'}", (5, 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                    ok, buf = cv2.imencode(".png", frame)
                    if ok:
                        self.latest_png = buf.tobytes()
                        self.preview_seq += 1
                    prev_gray = gray
                    time.sleep(0.03)
                except Exception as e:
                    if self.stop_event.is_set():
                        break
                    print(f"[Tracking Error]: {e}")
                    time.sleep(0.1)
        finally:
            try:
                sct.close()
            except Exception:
                pass
if __name__ == "__main__":
    root = tk.Tk()
    app = FlyControllerApp(root)
    root.mainloop()
