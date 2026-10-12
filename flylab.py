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
import random, ctypes, os, sys, json, time, threading, base64, math, re, queue, warnings, textwrap
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
_ALPHA_RATE = DT / RATE_TAU
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
        self.v += (V_REST - self.v + self.g) * (DT / TAU_M) * free
        self.ref -= DT
        spk = np.flatnonzero(self.v >= V_THRESH)
        if spk.size:
            self.v[spk] = V_RESET
            self.ref[spk] = T_REF
        self._last = spk
        self.n_spk += int(spk.size)
        self._pop += 0.02 * (spk.size / self.net.n - self._pop)     # global activity level (about 50 ms memory)
        self.flash *= _DECAY_FLASH
        self.flash[spk] = 1.0
        self.t_ms += DT
        # DN group rates (Hz per neuron, exponentially smoothed)
        cnt = np.bincount(self.gid[spk][self.gid[spk] >= 0], minlength=len(GROUPS))
        inst = cnt / self._gsize / (DT / 1000.0)
        for k, g in enumerate(GROUPS):
            self.rates[g] += _ALPHA_RATE * (inst[k] - self.rates[g])
        return spk.size
    def drives(self, full_hz=40.0):
        """(fwd, back, left, right), each 0..1 (full_hz = group rate that counts as 'full drive')."""
        return tuple(min(1.0, self.rates[g] / full_hz) for g in GROUPS)
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
FLY_R = 0.9                # mm, body radius used for collisions
FOOD_RANGE = 22.0          # mm, how far the fly "smells" food
AVOID_RANGE = 6.0          # mm, how far ahead it "sees" obstacles and walls
EAT_RANGE = 1.6            # mm
EAT_SECONDS = 4.0          # how long one food pellet takes to eat
ENV_RATE_HZ = 600.0        # Poisson rate on a descending-neuron group at full command
UP_POOL = 120              # strongest upstream (premotor) neurons per DN group that carry the command into the network
POP_REF = 0.0025           # population rate (spikes per neuron per ms, = 2.5 Hz) above which global inhibition damps the synapses
UP_GAIN_POW = 1.0          # the stronger the synapse gain slider, the weaker this extra drive (keeps the network from running away)
UP_RATE_HZ = 24.0          # Poisson rate on those at full command, so activity visibly flows through the brain on the way to the DNs
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
        lvl = (1 - mix) * lvl + mix * (0.35 + 0.65 * max(0.0, 1.0 - abs(th) / 1.0))
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
MINI_W, MINI_H = 660, 320                                  # window size on the "Minimized" page
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
    """The real NeuroMechFly model (FlyGym 2.x meshes) walking on a ground plane.

    The legs follow a tripod gait solved with inverse kinematics, and the body is moved
    kinematically (no contact physics), so it stays stable and is steered directly by the
    brain's descending-neuron drives:
        speed = (fwd - back) * V_MAX            turn rate = (right - left) * W_MAX
    If FlyGym / MuJoCo are missing, the panel shows the reason instead of a stand-in drawing.
    """
    LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
    TRIPOD = {"lf": 0.0, "rm": 0.0, "lh": 0.0, "rf": 0.5, "lm": 0.5, "rh": 0.5}
    STEP_LEN, LIFT, STANCE = 0.8, 0.35, 0.6      # mm, mm, fraction of the cycle spent on the ground
    V_MAX, W_MAX, HALF_TRACK = 9.0, 3.0, 1.2     # mm/s, rad/s, mm
    F_MAX = 14.0                                 # Hz, step frequency cap
    FEED_ANGLES = np.array([-1.1, -0.9])         # rad: rostrum / haustellum pitch at full proboscis extension

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.error = None
        self.renderer = None
        self.x = self.y = self.heading = 0.0
        self.arena = None                                  # Arena: obstacles / food drawn in the 3D view and used for collisions
        self.groom = 0.0                                   # 0..1: raise the front legs into a grooming motion
        self.feed = 0.0                                    # 0..1: extend the proboscis
        self.prob_adr = np.zeros(0, int)
        self._groom_t = 0.0
        self.trail = []
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
        from flygym.compose import NeuroMechFly, FlatGroundWorld, KinematicPosePreset
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
            fly.add_tracking_camera(name="trackcam")
            world = FlatGroundWorld()
            world.add_fly(fly, [0, 0, 0.7], Rotation3D("quat", [1, 0, 0, 0]))
            self.model, self.data = world.compile()
        m, d = self.model, self.data
        key = mj.mj_name2id(m, mj.mjtObj.mjOBJ_KEY, "neutral")
        mj.mj_resetDataKeyframe(m, d, key)
        mj.mj_forward(m, d)
        bid = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n)
        jid = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)
        # lift the fly so its lowest foot just touches the ground plane
        tips = {leg: bid(f"nmf/{leg}_tarsus5") for leg in self.LEGS}
        lowest = min(d.xpos[b][2] for b in tips.values())
        d.qpos[2] += 0.12 - lowest
        mj.mj_forward(m, d)
        self.q_neutral = d.qpos.copy()
        self.root_z = float(d.qpos[2])
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
            self.tip_body[leg] = tips[leg]
            self.tip0[leg] = self.ik.xpos[tips[leg]].copy()
        # free orbit camera that follows the thorax: drag to rotate, wheel to zoom
        dist = 0.0
        try:
            cid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_CAMERA, "nmf/trackcam")
            dist = float(np.linalg.norm(m.cam_pos[cid]))
        except Exception:
            pass
        thorax = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "nmf/c_thorax")
        pj = [jid("nmf/c_head-c_rostrum-pitch"), jid("nmf/c_rostrum-c_haustellum-pitch")]
        self.prob_adr = np.array([m.jnt_qposadr[i] for i in pj if i >= 0], int)
        self.cam = mj.MjvCamera()
        self.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = thorax if thorax >= 0 else 1
        self.cam_default = (135.0, -25.0, dist if dist > 1e-6 else 4.0)
        self.view_reset()
        self.renderer = mj.Renderer(m, height=self.h, width=self.w)
        self.reset()

    def _draw_arena(self):
        """Add walls, obstacles and food to the rendered scene (visual only; collisions are handled in step)."""
        if self.arena is None:
            return
        sc, mj = self.renderer.scene, self.mj
        eye = np.eye(3).ravel()
        def add(kind, size, pos, rgba):
            if sc.ngeom >= sc.maxgeom:
                return
            mj.mjv_initGeom(sc.geoms[sc.ngeom], kind, np.asarray(size, float), np.asarray(pos, float), eye,
                            np.asarray(rgba, np.float32))
            sc.ngeom += 1
        hf = self.arena.half
        wall = (0.55, 0.55, 0.6, 1.0)
        add(mj.mjtGeom.mjGEOM_BOX, [hf, 0.3, 0.6], [0, hf, 0.6], wall)
        add(mj.mjtGeom.mjGEOM_BOX, [hf, 0.3, 0.6], [0, -hf, 0.6], wall)
        add(mj.mjtGeom.mjGEOM_BOX, [0.3, hf, 0.6], [hf, 0, 0.6], wall)
        add(mj.mjtGeom.mjGEOM_BOX, [0.3, hf, 0.6], [-hf, 0, 0.6], wall)
        for ox, oy, r in self.arena.obstacles:
            add(mj.mjtGeom.mjGEOM_CYLINDER, [r, 1.6, 0], [ox, oy, 1.6], (0.45, 0.5, 0.62, 1.0))
        for fx, fy, amt in self.arena.foods:
            r = 0.25 + 0.45 * max(0.0, min(1.0, amt))
            add(mj.mjtGeom.mjGEOM_SPHERE, [r, 0, 0], [fx, fy, r], (0.95, 0.55, 0.12, 1.0))

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
    def reset(self):
        self.groom = 0.0
        self.feed = 0.0
        self.x = self.y = 0.0
        self.heading = 0.0
        self.v = self.w_turn = 0.0
        self.phase = 0.0
        if self.renderer is not None:
            self.ik.qpos[:] = self.q_neutral
            self.data.qpos[:] = self.q_neutral
            self._pose_root()
            self.mj.mj_forward(self.model, self.data)

    def _pose_root(self):
        q = self.data.qpos
        q[0], q[1], q[2] = self.x, self.y, self.root_z
        q[3], q[4], q[5], q[6] = math.cos(self.heading / 2), 0.0, 0.0, math.sin(self.heading / 2)

    # ----------------------------------------------------------------- gait
    def _foot_offset(self, s, step):
        """(dx, dz) of the foot relative to its neutral spot; phase s in [0,1), signed step length."""
        if s < self.STANCE:                                      # foot on the ground, sliding backwards
            u = s / self.STANCE
            return step * (0.5 - u), 0.0
        u = (s - self.STANCE) / (1.0 - self.STANCE)              # swing: lift and move forward
        lift = self.LIFT * min(1.0, abs(step) / (0.5 * self.STEP_LEN))
        return step * (-0.5 + 0.5 * (1 - math.cos(math.pi * u))), lift * math.sin(math.pi * u)

    def _solve_leg(self, leg, dx, dz):
        mj, m, d = self.mj, self.model, self.ik
        target = self.tip0[leg] + np.array([dx, 0.0, dz])
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

    def step(self, fwd, back, left, right, dt):
        if self.renderer is None:
            return
        v_t = (fwd - back) * self.V_MAX
        w_t = (right - left) * self.W_MAX                  # positive = turn right (clockwise)
        k = min(1.0, dt * 8.0)                             # smooth the commands a little
        self.v += (v_t - self.v) * k
        self.w_turn += (w_t - self.w_turn) * k
        sL = self.v + self.w_turn * self.HALF_TRACK        # left legs speed up when turning right
        sR = self.v - self.w_turn * self.HALF_TRACK
        smax = max(abs(sL), abs(sR))
        if smax > 0.15:
            f = min(self.F_MAX, self.STANCE * smax / self.STEP_LEN)
            stepL, stepR = self.STEP_LEN * sL / smax, self.STEP_LEN * sR / smax
            self.phase = (self.phase + f * dt) % 1.0
        else:
            stepL = stepR = 0.0                            # standing: legs settle back to the neutral pose
        self.heading -= self.w_turn * dt
        self.x += self.v * math.cos(self.heading) * dt
        self.y += self.v * math.sin(self.heading) * dt
        if self.arena is not None:
            self.x, self.y = self.arena.resolve(self.x, self.y)
        gr = float(min(1.0, max(0.0, self.groom)))
        self._groom_t += dt
        for leg in self.LEGS:
            s = (self.phase + self.TRIPOD[leg]) % 1.0
            dx, dz = self._foot_offset(s, (stepL if leg[0] == "l" else stepR) * (1.0 - gr))
            if gr > 0.02 and leg in ("lf", "rf"):          # front legs lift and rub in front of the head
                ph = 2 * math.pi * 6.0 * self._groom_t + (0.0 if leg == "lf" else math.pi)
                dx = (1 - gr) * dx + gr * (0.9 + 0.3 * math.sin(ph))
                dz = (1 - gr) * dz + gr * (0.9 + 0.25 * math.cos(ph))
            self._solve_leg(leg, dx, dz)
        if self.prob_adr.size:
            self.ik.qpos[self.prob_adr] = self.FEED_ANGLES[: self.prob_adr.size] * float(min(1.0, max(0.0, self.feed)))
        self.data.qpos[:] = self.ik.qpos
        self._pose_root()
        self.mj.mj_forward(self.model, self.data)

    # --------------------------------------------------------------- render
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
        if self._cached is not None and now - self._t_last < max(0.033, 2.5 * self._render_s):
            return self._cached                            # slow GPU / software GL: reuse the last frame
        try:
            self.renderer.update_scene(self.data, camera=self.cam)
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

    def close(self):
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
        self.nb.add(tab_build, text="   Build (obstacles & food)   ")
        self.nb.add(tab_setup, text="   MaleCNS Setup   ")
        tab_min = ttk.Frame(self.nb)
        self.nb.add(tab_min, text="   Minimized   ")
        self.build_minimized(tab_min)
        self.tab_min = tab_min
        self.nb.bind("<<NotebookTabChanged>>", self.on_tab_changed)
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
        self.update_lab()
        self.root.protocol("WM_DELETE_WINDOW", self.terminate)
    # ------------------------------------------------------------------
    # BRAIN LAB (live simulation + neuron view + fly body view)
    # ------------------------------------------------------------------
    def init_brain(self):
        # The real NeuroMechFly model (FlyGym meshes), walking; no hand-drawn stand-in.
        self.body = NeuroMechFlyBody(VIEW_W, VIEW_H)
        self.tool = tk.StringVar(value="rotate")
        self.gain_var = tk.DoubleVar(value=1.5)
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
        self.cut_pool_to_dn(net, self.up_pool)
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
        hot = np.flatnonzero(self.sim.flash > 0.05)
        if hot.size:
            glow(simidx, self.sim.flash, FIRE_BGR, 3.0, sigma=max(1.3, blur * 1.6))   # firing neurons get a wider halo so activity is easy to see
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
        for g in GROUPS:                                   # DN groups: small always-visible markers
            idx = net.groups[g]
            idx = idx[ok[idx]]
            for px, py in zip(ix[idx], iy[idx]):
                cv2.circle(img, (int(px), int(py)), 2, GROUP_COLORS[g], -1, cv2.LINE_AA)
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
        if self.sim.running and dynamic:
            feeding = self.arena.feed(self.body.x, self.body.y, dt)
        if self.sim.running:
            ext = np.zeros(self.net.n, np.float32)
            if dynamic:                                    # wander + react to food / obstacles
                sense = self.arena.sense(self.body.x, self.body.y, self.body.heading)
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
                    ext[self.net.groups[key]] = ENV_RATE_HZ * val
                    ext[self.up_pool[key]] = np.maximum(ext[self.up_pool[key]], UP_RATE_HZ * val / max(0.5, self.sim.gain) ** UP_GAIN_POW)
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
            self.body.step(fwd, back, left, right, dt)
        eaten = self.arena.eaten
        if eaten > self.food_seen and self.food_key and self.cfg["walk"]:
            threading.Thread(target=self.tap, args=(self.food_key,), daemon=True).start()   # one tap per pellet eaten
        self.food_seen = eaten
        self.frame_i = getattr(self, "frame_i", 0) + 1
        if self.frame_i % 3 == 0:
            self.update_arena_view()
        self.set_image(self.body_label, self.body.render(VIEW_W, VIEW_H))
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
        if self.graph_win is not None and self.frame_i % 2 == 0:
            self.draw_graphs()
        self.root.after(33, self.update_lab)
    # ------------------------------------------------------------------
    # BUILD TAB: place obstacles and food for the fly
    # ------------------------------------------------------------------
    def build_arena_tab(self, parent):
        parent.configure(padding=12)
        side = ttk.Frame(parent)
        side.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        ttk.Label(side, text="Build the fly's world", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        ttk.Label(side, wraplength=210, justify=tk.LEFT, text=(
            "Click on the map to place things. In 'Dynamic environment' mode the fly wanders on its own, "
            "smells food, steers around obstacles and walls, and stops to eat.")).pack(anchor=tk.W, pady=(4, 10))
        self.build_tool = tk.StringVar(value="obstacle")
        for text, val in (("Place obstacle", "obstacle"), ("Place food", "food"), ("Erase", "erase"), ("Move fly here", "fly")):
            ttk.Radiobutton(side, text=text, variable=self.build_tool, value=val).pack(anchor=tk.W, pady=1)
        self.obst_r_var = tk.DoubleVar(value=4.0)
        self.make_slider(side, "Obstacle size (mm)", self.obst_r_var, 1.5, 9.0, "{:.1f}", label_w=17)
        self.smell_var = tk.DoubleVar(value=self.arena.smell_range)
        smell_scale = self.make_slider(side, "Smell radius (mm)", self.smell_var, 3.0, 60.0, "{:.0f}", label_w=17,
                                       command=self.on_smell)
        smell_scale.bind("<Enter>", lambda e: self.set_smell_state(hover=True), add="+")
        smell_scale.bind("<Leave>", lambda e: self.set_smell_state(hover=False), add="+")
        smell_scale.bind("<ButtonPress-1>", lambda e: self.set_smell_state(drag=True), add="+")
        smell_scale.bind("<ButtonRelease-1>", lambda e: self.set_smell_state(drag=False), add="+")
        ttk.Label(side, wraplength=210, justify=tk.LEFT, foreground="#666666", text=(
            "How far the smell of food reaches. Hover or drag this slider to see each food's range as a circle; "
            "the fly only turns toward food inside it.")).pack(anchor=tk.W, pady=(0, 4))
        ttk.Button(side, text="Refill food", command=self.arena.refill).pack(fill=tk.X, pady=(12, 2))
        ttk.Button(side, text="Reset fly", command=self.reset_fly).pack(fill=tk.X, pady=2)
        ttk.Button(side, text="Clear everything", command=self.arena.clear).pack(fill=tk.X, pady=2)
        ttk.Button(side, text="Default layout", command=self.arena.default_layout).pack(fill=tk.X, pady=2)
        self.arena_info = tk.StringVar(value="")
        ttk.Label(side, textvariable=self.arena_info, justify=tk.LEFT).pack(anchor=tk.W, pady=(12, 0))
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
            self.body.x, self.body.y = self.arena.resolve(x, y)
            self.trail = []
    def update_arena_view(self):
        cv = self.arena_canvas
        sc = self.arena_px / (2 * ARENA_HALF)
        if self._arena_ver != self.arena.version:
            self._arena_ver = self.arena.version
            cv.delete("static")
            for g in range(-30, 40, 10):
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
        cv.delete("smell")
        if self.smell_show and self.arena.foods:
            img = self.smell_overlay(sc)
            if img is not None:
                cv.create_image(0, 0, anchor="nw", image=img, tags="smell")
        cv.delete("fly")
        if len(self.trail) > 1:
            pts = [c for x, y in self.trail for c in self._w2c(x, y)]
            cv.create_line(*pts, fill="#4f9f93", width=1, tags="fly")
        cx, cy = self._w2c(b.x, b.y)
        ux, uy = math.cos(b.heading), -math.sin(b.heading)          # heading on screen (+y is up on the map)
        px, py = -uy, ux
        L, Wd = 3.2 * sc, 1.5 * sc
        cv.create_polygon(cx + ux * L, cy + uy * L, cx - ux * L * 0.6 + px * Wd, cy - uy * L * 0.6 + py * Wd,
                          cx - ux * L * 0.6 - px * Wd, cy - uy * L * 0.6 - py * Wd, fill="#e7b44c", outline="#fff3d0", tags="fly")
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
        tk.Label(box, text="Select a page to start", bg=bg, fg="#555555", font=(normal, small)).pack(pady=(2, 0))
    def on_tab_changed(self, _e=None):
        try:
            if self.nb.select() == str(self.tab_min):
                self.root.geometry(f"{MINI_W}x{MINI_H}")
            else:
                self.root.geometry("")                         # back to the natural (full) size
        except Exception:
            pass
    def refresh_food_box(self):
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
    def select_region(self):
        """Dim the screen and let the user drag a box around the fly. Esc cancels."""
        if self.selecting or self.terminating or self.source.get() != "screen":
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
        pydirectinput.keyDown(key)
        time.sleep(hold_time)
        pydirectinput.keyUp(key)
    def hold(self, key, down):
        """Press/release a key only when its state actually changes."""
        with self.key_lock:
            if down and key not in self.held:
                self.klog(f"down    {key}")
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
        for k in ALL_KEYS:
            pydirectinput.keyUp(k)
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
                                self.tap('space')
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
                                if mode == "wasd":
                                    uk, dk, lk, rk = 'w', 's', 'a', 'd'
                                else:
                                    uk, dk, lk, rk = 'up', 'down', 'left', 'right'
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
