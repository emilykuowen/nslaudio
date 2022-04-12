import keyboard
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sofa
from pydub import AudioSegment
import io
import scipy.io.wavfile
import csv
import math

"""
Python SOFA API: https://python-sofa.readthedocs.io/en/latest/index.html
SOFA specifications: https://www.sofaconventions.org/mediawiki/index.php/SOFA_specifications
x = forward, y = right, z = up
pitch = tilting on x axis
yaw = tilting on z axis
roll = tilting on y axis
"""

"""
To-Dos:
- Create demos of interesting shapes
- Try different HRIR sets with more elevation variations
- Add in listener movements
- Make find_measurement() more efficient
"""

def find_measurement(azimuth, elevation, spherical_positions):
    best_fit = 0
    best_error = abs(azimuth - spherical_positions[0][0]) + abs(elevation - spherical_positions[0][1])
    for i in range(1, len(spherical_positions)):
        diff_az = abs(azimuth - spherical_positions[i][0])
        diff_ele = abs(elevation - spherical_positions[i][1])
        new_error = diff_az + diff_ele
        if(new_error < best_error):
            best_fit = i
            best_error = new_error
    return best_fit


def plot_coordinates(coords, title):
    x0 = coords
    n0 = coords
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    q = ax.quiver(x0[:, 0], x0[:, 1], x0[:, 2], n0[:, 0],
                  n0[:, 1], n0[:, 2], length=0.1)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(title)
    return q


def normalize(num_array):
    norm = np.linalg.norm(num_array)
    normalized_array = num_array / norm
    return normalized_array


def get_angle_and_attenuation(source_file):
    listenerX = 0
    listenerY = 0
    listenerZ = 0
    azimuth_list = []
    elevation_list = []
    attenuation_list = []

    with open(source_file, newline='') as csvfile:
        # get number of columns
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            sourceX = float(row[0])
            sourceY = float(row[1])
            sourceZ = float(row[2])
            
            diffX = sourceX - listenerX
            diffY = sourceY - listenerY
            diffZ = sourceZ - listenerZ
    
            # calculate azimuth
            if diffX == 0:
                if sourceY >= listenerY:
                    azimuth = 0
                else:
                    azimuth = 180
            elif diffY == 0:
                if sourceX >= listenerX:
                    azimuth = 90
                else:
                    azimuth = 270
            else:
                if listenerY > sourceY:
                    azimuth = math.degrees(math.atan(diffY / diffX) - math.pi)
                else:
                    azimuth = math.degrees(math.atan(diffY / diffX))

            if azimuth < 0:
                azimuth = 360 + azimuth
            print("raw azimuth = ", azimuth)
            
            # calculate elevation
            horizontal_distance = math.sqrt(diffX**2 + diffY**2)
            if diffZ == 0:
                elevation = 0
            elif horizontal_distance == 0:
                if sourceZ < listenerZ:
                    elevation = -90
                else:
                    elevation = 90
            else:
                elevation = math.degrees(math.atan(diffZ / horizontal_distance))
            print("raw atan elevation = ", elevation)

            if(elevation > 90):
                elevation = 180 - elevation
            if(elevation < -90):
                elevation = -180 - elevation
            print("elevation after flipping = ", elevation)

            # calculate 3D distance
            distance = math.sqrt(diffX**2 + diffY**2 + diffZ**2)
            if distance == 0:
                attenuation = 1.0
            else:
                attenuation = 1.0 / (distance**2)
            print("attenuation = ", attenuation)
            
            azimuth_list.append(azimuth)
            elevation_list.append(elevation)
            attenuation_list.append(attenuation)

    return np.column_stack((azimuth_list, elevation_list,attenuation_list))


def generate_audio_for_single_source(HRTF_path, source_file, cvs_file, output_filename="", output_flag=False):
    HRTF = sofa.Database.open(HRTF_path)
    HRTF.Metadata.dump()
    spherical_source_positions = HRTF.Source.Position.get_values(system="spherical")
    emitter = 0

    # read audio data from source file
    segment = AudioSegment.from_file(source_file)
    segment_channels = segment.split_to_mono()
    audio_samples = [s.get_array_of_samples() for s in segment_channels]
    audio_np_array = np.array(audio_samples).T
    audio_len = audio_np_array.shape[0]

    # calculate azimuth angle, elevation angle, and attenuation constant for each position
    audio_info = get_angle_and_attenuation(cvs_file)
    print("processing ", source_file)
    print(audio_info)
    position_num = len(audio_info)
    chunk_len = audio_len // position_num

    convolved_channel_left = np.zeros(1)
    convolved_channel_right = np.zeros(1)

    for i in range(position_num):
        azimuth = audio_info[i][0]
        elevation = audio_info[i][1]
        attenuation = audio_info[i][2]
        measurement = find_measurement(azimuth, elevation, spherical_source_positions)
        print("found measurement's azimuth = ", spherical_source_positions[measurement][0])
        print("found measurement's elevation = ", spherical_source_positions[measurement][1])
        hrtf_left = HRTF.Data.IR.get_values(indices={"M":measurement, "R":0, "E":emitter})
        hrtf_right = HRTF.Data.IR.get_values(indices={"M":measurement, "R":1, "E":emitter})

        start_index = i*chunk_len
        end_index = (i+1)*chunk_len-1
        # TODO: make this compatible with both stereo and mono files
        audio_chunk = audio_np_array[start_index:end_index, 0]

        convolved_left = np.array(signal.convolve(audio_chunk, hrtf_left, mode='full'))
        convolved_right = np.array(signal.convolve(audio_chunk, hrtf_right, mode='full'))

        # attenuate the signal by the inverse square law
        convolved_left = np.trim_zeros(convolved_left) * attenuation
        convolved_right = np.trim_zeros(convolved_right) * attenuation

        convolved_channel_left = np.concatenate((convolved_channel_left, convolved_left))
        convolved_channel_right = np.concatenate((convolved_channel_right, convolved_right))

    convolved_stereo = np.array([convolved_channel_left, convolved_channel_right]).T
   
    if output_flag == True:
        convolved_normalized = np.array(normalize(convolved_stereo))
        num_bit = 16
        bit_depth = 2 ** (num_bit-1)
        convolved_final = np.int16(convolved_normalized / np.max(np.abs(convolved_normalized)) * bit_depth-1)
        scipy.io.wavfile.write(output_filename, int(segment.frame_rate), convolved_final)

    return convolved_stereo


def generate_audio_for_multiple_sources(HRTF_path, source_array, cvs_array, output_filename):
    convolved_sum = generate_audio_for_single_source(HRTF_path, source_array[0], cvs_array[0])
    num_sources = len(source_array)

    for i in range(1, num_sources):
        convolved_new = generate_audio_for_single_source(HRTF_path, source_array[i], cvs_array[i])

        old_source_len = len(convolved_sum)
        new_source_len = len(convolved_new)
        pad_len = abs(old_source_len - new_source_len)
    
        if old_source_len < new_source_len:
            convolved_padded = np.pad(convolved_sum, ((0,pad_len),(0,0)), 'constant')
            convolved_sum = np.add(convolved_padded, convolved_new)
        else:
            convolved_padded = np.pad(convolved_new, ((0,pad_len),(0,0)), 'constant')
            convolved_sum = np.add(convolved_padded, convolved_sum)
    
    convolved_normalized = np.array(normalize(convolved_sum))
    num_bit = 16
    bit_depth = 2 ** (num_bit-1)
    convolved_final = np.int16(convolved_normalized / np.max(np.abs(convolved_normalized)) * bit_depth-1)
    scipy.io.wavfile.write(output_filename, 44100, convolved_final)

"""
MIT KEMAR HRTF

Coordinate system:
- Azimuth: 360 degrees (counterclockwise)
- Elevation range: -40 to +90 degrees

The spherical space around the KEMAR was sampled at elevations from
-40 degrees (40 degrees below the horizontal plane) to +90 degrees
(directly overhead).  At each elevation, a full 360 degrees of azimuth
was sampled in equal sized increments.  The increment sizes were
chosen to maintain approximately 5 degree great-circle increments.
The table below shows the number of samples and azimuth increment at
each elevation (all angles in degrees).  A total of 710 locations were
sampled.
"""

if __name__ == '__main__':
    # Example 1: generating audio file for square movement of sine tone
    HRTF_path = "hrtf/dtf_nh2.sofa"
    source_file = "audio_sources/sin_440.wav"
    # cvs_file = "csv/sin_source_square.csv"
    # output_filename = "audio_output/sin_440_square_movement.wav"
    output_flag = True
    # generate_audio_for_single_source(HRTF_path, source_file, cvs_file, output_filename, output_flag)

    # # Example 2: generating audio file for circular movements of sine tone and piano tune
    # source_array = ["audio_sources/sin_440.wav", "audio_sources/piano.wav"]
    # cvs_array = ["csv/sin_source_circular_xy.csv", "csv/piano_source_circular.csv"]
    # output_filename = "audio_output/sin_440_and_piano_circular_movement.wav"
    # generate_audio_for_multiple_sources(HRTF_path, source_array, cvs_array, output_filename)

    # # Example 3: generating audio file for circular movement of sine tone
    # source_file = "audio_sources/sin_440.wav"
    # cvs_file = "csv/sin_source_circular_xy.csv"
    # output_filename = "audio_output/sin_440_circular_movement_xy.wav"
    # generate_audio_for_single_source(HRTF_path, source_file, cvs_file, output_filename, output_flag)

    cvs_file = "csv/sin_source_circular_xz.csv"
    output_filename = "audio_output/sin_440_circular_movement_xz_ARI.wav"
    generate_audio_for_single_source(HRTF_path, source_file, cvs_file, output_filename, output_flag)    

    # cvs_file = "csv/sin_source_circular_yz.csv"
    # output_filename = "audio_output/sin_440_circular_movement_yz_ARI.wav"
    # generate_audio_for_single_source(HRTF_path, source_file, cvs_file, output_filename, output_flag)