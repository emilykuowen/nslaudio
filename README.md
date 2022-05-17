# NSL Audio

Codebase for Six Degrees of Freedom Audio Rendering

## Installation

```bash
pip install -r requirements.txt
```

# Realtime Audio Simulation
Run
```bash
python3 overlapping_realtime_sim.py
```

Line 396 of overlapping_realtime_sim.py can be edited to change the sound object audio files and positions within the Scene
ex) Source(-1, 2, -3, "audio_sources/source_audio.wav") creates an object in the scene at x=-1, y=2, z=-3 that reads audio from audio_sources/source_audio.wav

Controls
- Left arrow key  = translational movement in -x direction
- Right arrow key = translational movement in +x direction
- Up arrow key    = translational movement in +y direction
- Down arrow key  = translational movement in -y direction
- Space key       = translational movement in +z direction
- Shift key       = translational movement in -z direction
- 'A' key         = rotational movement in azimuth leftward
- 'D' key         = rotational movement in azimuth rightward
- 'W' key         = rotational movement in elevation upward
- 'S' key         = rotational movement in elevation downward