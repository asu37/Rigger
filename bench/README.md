# Deformation benchmark

Rigs real human bodies with Rigger, poses them and measures how the skin
deforms: volume loss at joints, surface stretch, twist spread, candy-wrapper
pinching and weights that leak between body parts. Use it to compare rig and
skinning changes with numbers instead of eyeballing.

## Run

```
python bench/fetch_assets.py                                    # once: ~50 MB, CC0
blender -b --factory-startup --python bench/run_bench.py -- --render
```

About 6 seconds per body and setup without renders (2 minutes for everything).
With `--render`, each pose also writes a heat-map image and `gallery.html`
shows them side by side; the full run then takes about 15 minutes.

Options after the `--`:

| Option | Meaning |
| --- | --- |
| `--bodies a,b` | bodies to run (default all, see `bodies.py`) |
| `--configs a,b` | rig/bind setups to compare (default all, see `CONFIGS` in `run_bench.py`) |
| `--poses a,b` | poses to run (default all, see `POSES`) |
| `--render` | heat-map render per pose (Cycles on the CPU, works headless) |
| `--out DIR` | output folder, default `bench/results/latest` |
| `--compare FILE` | `results.json` of an earlier run: adds a before/after table |

Output: `results.json` (every number), `report.md` (tables), and with `--render`
`images/` and `gallery.html`.

To check a change: run once and keep the folder, make the change, run again
with `--compare <first run>/results.json`.

`python bench/compare.py A/results.json:config B/results.json:config` prints a
win/loss/tie scoreboard per body set and per joint, and the measures A loses,
worst first.

`bench/ai_markers_eval.py` measures how far AI-guessed markers land from the
frozen ones (about 2 cm on average) and saves what the model saw to
`results/ai_markers/`.

`python -I bench/test_metrics.py` checks the measures themselves on synthetic
tubes (rigid motion reads clean, a squash reads 0.5, a linear-blend twist
pinches...).

## Bodies

`--bodies all` runs the eight bodies, `--bodies heldout` the four held-out ones.
Rigger's defaults were tuned on the four Human Base Meshes; the four primitive
mannequins (about 50 separate low-poly pieces each, joined into one mesh) are
held out to check that changes carry over. Their markers were placed with the
AI and checked on renders.


Blender Studio's [Human Base Meshes](https://www.blender.org/download/demo/asset-bundles/human-base-meshes/)
(CC0): realistic male and female, stylized male and female, 10-12k vertices,
A-pose. The multires levels are dropped and the eyeballs left out.

The six Smart Marker positions of each body are frozen in `landmarks.json`, so
the benchmark measures rigging and skinning, not marker placement. They were
derived with `derive_landmarks.py` (mesh cross-sections: crotch, throat
profile, armpit, thinnest wrist cut, lower shin) and checked on the renders it
writes to `results/landmarks/`. Run it again only when adding bodies.

## Setups

| Config | What it is |
| --- | --- |
| `default` | Rigger out of the box: joints fitted to the mesh, bendy-bone secondaries, 2 lower and 1 upper twist bones, tightened elbow/knee fans, Preserve Volume on (30% at the shoulders) |
| `full_preserve_volume` | Default with full Preserve Volume on the shoulders too |
| `no_fans` | Default without the elbow/knee fans |
| `ai_markers` | Default with markers guessed by the AI pose model instead of the frozen ones (needs `RIGGER_AI_DIR`) |
| `loose_fans` | Default with fan weights left as heat weighting makes them (not tightened) |
| `no_joint_fit` | Default with template proportions for the elbows, knees, hips, spine and toes |
| `old_weight_defaults` | Default with the earlier weight corrections: hip cut on, no head/shoulder or finger cleanup |
| `no_preserve_volume` | Same rig with plain linear blend skinning |
| `twist_secondary` | Default with twist secondaries instead of bendy bones, no fans |
| `quality_bendy` | 3 lower + 2 upper twist bones, fans at all joints, refined head weights |
| `quality_twist` | The same with twist secondaries |

## Poses

All on the character's left side, through the FK controls, starting from the
rest A-pose. Bends turn a limb toward a direction about the axis perpendicular
to it; twists turn a control about its limb's axis. 58 poses in 3D:

| Group | Poses |
| --- | --- |
| Shoulder | raised up, forward, back, out, across the body, up-forward, up-back and out-forward, two amounts each |
| Elbow / knee | 45, 90, 135 and 150 degrees; elbow bent while the hand twists |
| Wrist | bent forward, back and to both sides |
| Hip | forward 45/90/120, back, out, in, out-forward |
| Ankle | toes up, toes down, tilted in and out |
| Spine | forward, back, sideways, twisted |
| Neck | nod, back, tilt, turn |
| Twists | forearm, upper arm, neck, spine |
| Combined | squat, squat with a twist, reach up-forward, arm across the chest, crouch and reach |

`bend_deg` in `results.json` confirms the joint really turned by the asked
angle (child segment measured relative to its parent).

## Measures

Every rig is measured against the same **measuring skeleton** taken from the
mesh: the frozen markers plus the elbow, knee and hip that `rigger.fitting`
finds. Each measuring point follows one deform bone rigidly when posed. Rigs
with differently placed bones are then compared with the same yardstick: if a
rig's elbow pivot is off the arm's centre, the skin around the real elbow
distorts and the numbers show it.

A joint's **region** is the skin within a radius (a fraction of body height)
of the measuring joint and close to the limb's measuring segments. It is chosen
from geometry only, so it depends neither on the weights nor on the rig's bones.

| Measure | Meaning | Good |
| --- | --- | --- |
| **collapse p05** | 5th percentile, over the region, of each vertex's distance to its own bone (the one closest at rest), posed / at rest. Low = the joint loses volume (elbow folding flat). A rigid bend reads exactly 1. | close to 1 |
| **bulge p95** | 95th percentile of the same ratio. High = skin balloons or pokes out. | close to 1 |
| **crushed** | share of the region's triangles squashed below half their area | low |
| **stretch rms** | RMS of log2(triangle area posed / rest) in the region | low |
| **volume** | whole-body volume change | close to 0 |
| **twist error** | RMS gap between the measured twist along the limb and an even spread (0 at the start, full at the end), as a fraction of the angle | low |
| **min radius** | thinnest cross-section / rest along the twisted limb. Below 1 = candy-wrapper pinch | close to 1 |
| **end twist** | twist reached in the last slice / the ideal there | close to 1 |
| **elbow / knee off-centre** | distance of the rig's joint from the centre of the limb section through it, as a share of the limb radius | close to 0 |
| **elbow along arm** | where the rig's elbow falls between shoulder and wrist (a human skeleton: about 0.54) | |
| **bleed** | share of vertices with 5%+ weight on a bone 3+ bones away in the deform hierarchy from their main bone (twist and fan bones count as their limb bone). Worst pairs are listed. | low |
| **finger bleed** | the same between fingers (fingers sticking together) | low |
| **cross side** | share of vertices weighted 5%+ to both a left and a right bone | low |

Thickness and stretch can disagree: a bend that keeps the distance to the bone
but folds skin into a crease shows as crushed triangles, not collapse.

The images colour each vertex by its surface stretch: white unchanged, blue
squashed, red stretched (saturates at 2.8x either way).

## Limits

* Four bodies in one A-pose. Different topology or proportions may behave
  differently; add bodies to `bodies.py` and `landmarks.json`.
* No self-intersection check: a deep knee bend can push skin through itself
  without lowering any number.
* Measures compare against simple ideals (rigid limb, even twist). They catch
  regressions and gross errors well; the last word on looks is still the
  images.

## Video to Animation golden clips

`video_bench.py` scores the video motion capture against motion it knows
exactly: a benchmark body performs a scripted 5-second routine (arm raises,
reaching at the camera, a deep squat, twist, kick, turn and step, finger curls),
rendered to an MP4 with Cycles (about 8 minutes), then the pipeline runs on that
video and its result is compared with the truth.

```
blender -b --factory-startup --python bench/video_bench.py -- make --body realistic_male
blender -b --factory-startup --python bench/video_bench.py -- eval --body realistic_male --focal 26
```

The pose model must be in `bench/assets/ai` (or `RIGGER_AI_DIR`). Scores: joint
error with the root aligned (MPJPE) and after the best rigid alignment
(PA-MPJPE), limb angle error, root path and height error, jitter relative to the
truth, foot sliding during true contacts, finger curl error and the pose model's
own 2D error. `calibrate_sites.py` fits where the model's keypoints sit on the
skeleton from a clip; check any change on a second body, a fit to one body
did not carry over.

Results (1.8 m man tuned on, woman held out; camera 35 mm, solved with the 26 mm default and with 35 mm):

| | man 26 | man 35 | woman 26 | woman 35 |
| --- | --- | --- | --- | --- |
| MPJPE (root aligned) | 57.5 mm | 59.3 mm | 68.5 mm | 64.5 mm |
| PA-MPJPE | 42.5 mm | 42.9 mm | 50.4 mm | 46.5 mm |
| root height error | 6 mm | 7 mm | 7 mm | 5 mm |
| foot sliding in contacts | 0.8 cm/s | 0.8 cm/s | 8.2 cm/s | 9.2 cm/s |
| jitter vs truth | 1.29x | 1.42x | 1.57x | 1.59x |

First version on the man (26 mm): MPJPE 70.7 mm, PA-MPJPE 52.0 mm, root
height 260 mm off, sliding 6.9 cm/s. The gains came from a keypoint site model
(COCO shoulders and hips sit above the joints; the shoulder marker moves with the
collarbone, not the arm), contacts judged between the feet instead of against a
floor height, backward-swing limits at hips and shoulders, a second pass that
holds planted feet on the floor plane, and IK foot locking. The woman's sliding
comes from one-frame touches at the bottom of the squat; her remaining error is
mostly the right elbow while both arms point straight at the camera.
