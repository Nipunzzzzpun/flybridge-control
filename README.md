# FlyLab

A live, lightweight fly simulation in one Tk control panel: a spiking model built from the
**MaleCNS v1.0 connectome** drives a **NeuroMechFly** body (FlyGym 2.x) in a small arena.
CPU only, no GPU. Includes an optional "Fly Bridge" that turns screen optical flow into key
presses, with a Blooket helper mode.

## What it does
- **Control Panel:** live brain view (paint neurons to stimulate them), the NeuroMechFly fly (drag to rotate),
  presets taken from the BadApple-fly evidence catalogue, and a pause/run switch (starts paused).
- **Dynamic environment (default):** the fly wanders on its own, avoids walls and obstacles, finds food and eats
  it with proboscis extension.
- **Build tab:** place obstacles and food in the arena.
- **Smell plumes (Build tab):** food releases odour puffs that the wind carries downwind, with eddies that make the plume snake and spread (a 2-D filament model, drawn as smoke on the map). The fly's two antennae sample the odour: it surges upwind while it smells it, casts crosswind when it loses it, and walks straight to the food once it is close. Wind speed and direction are sliders, with a large live wind compass (black arrow = direction, thinner blue arrow = velocity); the old 'Radius' smell model is deprecated and greyed out.
- **Miscellaneous tab:** the raw view from each compound eye (right eye left of the left eye, a grey dot per ommatidium) with optic-flow graphs; the real 3-D head flattened to 2-D with the antennae and palps highlighted and their odour numbers; odour graphs for both antennae; and smelling triggers (a 1.5 s test odour at the left / right / both antennae, or a puff released in front of the fly).
- **Settings tab:** categories on the left (Performance, Body & physics, Smell & wind, Build tab, Display, Recording, Mapping, Profiles, Theme, Advanced, Suggestions). Frame rate and what gets drawn, joint stiffness / adhesion / friction / gravity (applied live to the running model), gait and step settings, plume strength / spreading / turbulence / gusts, snap-to-grid and layout save/load/randomize, screenshots and CSV export, presets. Everything is saved to `flylab_settings.json` automatically.
- Every Settings category has its own **Reset ... to defaults** button, so a messed-up category can be put back without touching the others (Profiles also keeps **Reset ALL settings**).
- **Uranium smell:** uranium rods give off no smoke; the radiation particles flying out of them are their smell, so the fly tracks them to the rod.
- **Food skins:** Build tab sidebar > Food skin (applies to food you place from then on; existing food keeps its look): normal food, glowing green uranium rods (lying at random angles; single radiation ticks at random (more often when closer), then radiation.mp3 loops and the Control Panel camera gets TV static (checkbox) and a glitch-displacement slider (0-500%) in Settings > Display, louder the closer the fly gets, until the rod is eaten), or apple slices (a halved apple: red rim, pale flesh, star-shaped core with five seeds) (in the map and the 3-D view).
- **Mapping:** where the window opens (saved position), remappable movement / jump / food keys for both layouts (each with an optional sine pulse at its own Hz when pressed), and the screen-select tools (capture region, Blooket buttons).
- **Theme and suggestions:** Light / Dark / System (follows Windows) or a random Light/Dark each start; Settings > Suggestions posts a note to the maintainer's Discord channel through a small relay (see `relay/README.md`), so the real webhook is never in this repository.
- **Motor Properties tab:** every joint motor of the fly with its target and actual angle, step rate (Hz) and the ground-contact force of each foot, like the NeuroMechFly web simulator.
- **Minimize tab:** shrinks the window to a small box.
- **MaleCNS Setup tab:** builds a subset of the connectome from local data (about 3 s from compiled arrays).
- **Fly Bridge / Blooket:** optical-flow-to-keys, with a **Map buttons...** tool to map answer/continue buttons per game mode.

## Honest limitations
This is **not** a faithful whole-brain simulation.
- The neurons are a leaky integrate-and-fire model with invented parameters, using the real connectivity and signs.
- Wandering and sensing are injected into upstream (premotor) neuron pools and reach the descending neurons through the real synapses (the reverse command also gets a small direct push). Descending neurons are treated as output-only, and opposing groups (forward/back, left/right) are subtracted to give the command.
- The body is simulated with real physics in MuJoCo (position-controlled leg motors, gravity, ground contact, adhesion feet), but the leg movements come from a hand-written tripod-gait controller. The brain only sets the walking speed and turning command, not individual muscles.
- Only walking, grooming and feeding presets change the body; other presets just light up neurons.
- Skeleton shapes in the brain view need a neuPrint token.

## Storage requirements
Plan for **about 3 GB of free disk space** in total (5 GB leaves comfortable room). Almost all of it is the Python packages and the connectome data; FlyLab itself is under 1 MB.
Sizes of the packages were measured by installing the pinned `requirements.txt` into a clean folder (on Linux, Python 3.13). Windows wheels are the same order of size but can differ by a few tens of percent, so treat them as approximate.

### 1. The program (this repository)
| Component | Size |
|---|---|
| `flylab.py` (the whole app) | 350 KB |
| `radiation.mp3` (uranium sound) | 155 KB |
| `geiger-counter-clicking.mp3` (settings radiation preview) | 55 KB |
| `README.md`, `LICENSE`, `requirements.txt`, `setup.ps1`, `.gitignore` | under 10 KB |
| **Total** | **about 0.6 MB** |

### 2. Software you need installed
| Component | Size |
|---|---|
| Python 3.12 or newer (with Tk, which the window uses) | about 100 MB |
| Git | about 300 MB |

### 3. Python packages (installed into the `venv` folder)
| Package | What it is used for | Size |
|---|---|---|
| NumPy | arrays, the neuron maths | 68 MB |
| SciPy | sparse synapse matrices | 138 MB |
| OpenCV (`opencv-python`) | drawing the camera / graph / head views | 196 MB |
| MuJoCo (+ glfw, PyOpenGL, absl, etils, fsspec) | body physics and 3-D rendering | 45 MB |
| PyArrow | reading the connectome `.feather` files | 172 MB |
| pandas | annotations tables | 68 MB |
| FlyGym 2.x (code + bundled NeuroMechFly fly model, meshes and textures) | the fly body | 6 MB |
| Numba + llvmlite | FlyGym dependency | 211 MB |
| Matplotlib (+ Pillow, fonttools, kiwisolver, ...) | FlyGym dependency | 93 MB |
| imageio + imageio-ffmpeg (bundled ffmpeg) + mediapy | FlyGym dependency | 83 MB |
| IPython and friends (Jedi, Pygments, prompt-toolkit, ...) | pulled in by mediapy | 41 MB |
| neuprint-python (+ networkx, requests, tqdm, ujson) | optional skeleton shapes | 21 MB |
| mss + pydirectinput | screen capture and key presses (Fly Bridge) | under 1 MB |
| Everything else (PyYAML, jaxtyping, loguru, ...) | small helpers | 6 MB |
| **Total (the `venv` folder)** | | **about 1.15 GB** |

Pip also keeps a download cache (`pip cache purge` frees it), which can temporarily take a few hundred MB more.

### 4. Connectome data (downloaded separately, not included)
You only need the three files below from the [MaleCNS download page](https://male-cns.janelia.org/download/). The other files listed there (synapse points and partners, T-bar neurotransmitters, body stats) are **not needed** and are huge.
| File | Size |
|---|---|
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (who connects to whom) | 1.1 GB |
| `body-neurotransmitters-male-cns-v1.0.feather` (signs: excitatory / inhibitory) | 42 MB |
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` (cell types, regions) | 13 MB |
| **Total** | **about 1.16 GB** |

If you use the pre-compiled arrays instead (`ids.npy`, `indptr.npy`, `indices.npy`, `weights.npy`, `signs.npy`, `neurons.feather`), you do not need the raw weights file. Their size depends on how you compiled them and was not measured here.

### 5. Files FlyLab creates while running
| File | Size |
|---|---|
| `brain_subset.npz` (the small brain FlyLab actually simulates, built in the MaleCNS Setup tab) | roughly 5-15 MB, estimated: it depends on the neuron count you pick (default 8000) |
| `flylab_settings.json`, `flylab_layout.json`, `flylab_region.json`, `flylab_food_key.json`, `flylab_blooket_map.json`, `flylab_discord.json` | a few KB each |
| Screenshots and graph CSV exports (Recording settings) | a few hundred KB each, only when you press the buttons |

### Total
| What | Size |
|---|---|
| Program | 0.6 MB |
| Python + Git | about 0.4 GB |
| Python packages | about 1.15 GB |
| Connectome data (3 files) | about 1.16 GB |
| Generated brain + settings | about 10 MB |
| **Total** | **about 2.7 GB** |

You can delete the raw connectome files after `brain_subset.npz` is built (about 1.16 GB back), but keep them if you want to rebuild with a different neuron count.

## Install (Windows)
Requires **Python 3.12 or newer** and **Git**.
```powershell
git clone https://github.com/<you>/FlyLab.git
cd FlyLab
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```
That creates a venv, installs everything and adds a desktop shortcut. Or by hand:
```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python flylab.py
```
FlyGym 2.x is installed from GitHub (`flygym @ git+https://github.com/NeLy-EPFL/flygym.git@v2.1.0`); an old PyPI FlyGym 1.x will not work.

## Updating
If you cloned the repo, pull the latest version and refresh the dependencies:
```powershell
cd FlyLab
git pull
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt --upgrade
python flylab.py
```
Your saved screen region and key map (`flylab_*.json`) are git-ignored, so `git pull` will not overwrite them.
If `git pull` complains about local changes, run `git stash` first (or `git checkout flylab.py` to discard your edits).
If you downloaded a ZIP instead, download the new ZIP and replace `flylab.py` (and `requirements.txt`).
Check the [Releases](../../releases) page for what changed.

## Get the data (not included)
The connectome is **not** redistributed here. Download **MaleCNS v1.0** yourself, follow its licence and citation
requirements, and point the *MaleCNS Setup* tab at the folder. Either:
- the raw files (`connections.feather`, `annotations.feather`, `neurotransmitters.feather`), or
- pre-compiled arrays (`ids.npy`, `indptr.npy`, `indices.npy`, `weights.npy`, `signs.npy`, `neurons.feather`).

Skeleton shapes optionally need a [neuPrint](https://neuprint.janelia.org) token.

## Using the Fly Bridge / Blooket mode responsibly
The bridge sends real mouse clicks and key presses. Only use it where automation is allowed;
using it in games may break their rules. Click mappings are stored in `flylab_blooket_map.json`
(per mode, personal to your screen, and git-ignored).

## Credits
- [FlyGym / NeuroMechFly](https://github.com/NeLy-EPFL/flygym) (EPFL NeLy)
- MaleCNS v1.0 connectome (Janelia / collaborators, see its release for citation)
- BadApple-fly preset catalogue
- MuJoCo, NumPy, SciPy, OpenCV

## Licence
MIT, see [LICENSE](LICENSE). The data and third-party packages keep their own licences.
