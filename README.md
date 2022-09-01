# NSL Audio

Codebase for Six Degrees of Freedom Audio Rendering

## Installation

```bash
pip install -r requirements.txt
```

## Realtime Audio Simulation
Run
```bash
python3 overlapping_realtime_sim.py
```

Line 396 of overlapping_realtime_sim.py can be edited to change the audio files and positions of the sound objects in the Scene
```bash
Source(-1, 2, -3, "audio_sources/source_audio.wav")
# creates an object in the scene at x=-1, y=2, z=-3 that reads audio from audio_sources/source_audio.wav
```

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

## Non-Realtime Audio Simulation
1. Edit the lines in the main function to set the HRTF set, WAV file of audio source, CVS file of audio source positions, and the output filename.
```bash
HRTF_path = "hrtf/dtf_nh2.sofa"
source_file = "audio_sources/sin_440.wav"
cvs_file = "csv/sin_source_circular_xy.csv"
output_filename = "audio_output/sin_440_circular_movement_xy_precise.wav"
output_flag = True
generate_audio_for_single_source(HRTF_path, source_file, cvs_file, output_filename, output_flag)
```

2. Run
```bash
python3 spatial_movement_sim.py
```
