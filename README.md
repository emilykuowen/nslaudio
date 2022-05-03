# nslaudio

Audio Code for USC Networked Systems Lab

## Installation

```bash
pip install -r requirements.txt
```

# Realtime Audio Simulation

py overlapping_realtime_sim.py

Controls

Left arrow key  - translational movement in the -x direction

Right arrow key - translational movement in the +x direction

Up arrow key    - translational movement in the +y direction

Down arrow key  - translational movement in the -y direction

Space key       - translational movement in the +z direction

Shift key       - translational movement in the -z direction

'A' key         - rotational movement in the azimuth to the left

'D' key         - rotational movement in the azimuth to the right

'W' key         - rotational movement in the elevation upwards

'S' key         - rotational movement in the elevation downwards

Line 396 of overlapping_realtime_sim.py can be edited to change the sound object audio files and positions within the Scene

ex) Source(-1, 2, -3, "audio_sources/source_audio.wav") creates an object in the scene at x=-1, y=2, z=-3 that reads audio from audio_sources/source_audio.wav