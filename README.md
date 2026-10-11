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
- **MaleCNS Setup tab:** builds a subset of the connectome from local data (about 3 s from compiled arrays).
- **Fly Bridge / Blooket:** optical-flow-to-keys, with a **Map buttons...** tool to map answer/continue buttons per game mode.

## Honest limitations
This is **not** a faithful whole-brain simulation.
- The neurons are a leaky integrate-and-fire model with invented parameters, using the real connectivity and signs.
- Wandering and sensing are injected directly into descending-neuron groups, and descending neurons are treated as output-only.
- The body is kinematic (a tripod-gait controller), not a physics-driven one.
- Only walking, grooming and feeding presets change the body; other presets just light up neurons.
- Skeleton shapes in the brain view need a neuPrint token.

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
