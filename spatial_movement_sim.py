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

"""
Python SOFA API: https://python-sofa.readthedocs.io/en/latest/index.html
SOFA specifications: https://www.sofaconventions.org/mediawiki/index.php/SOFA_specifications
x = forward, y = right, z = up
pitch = tilting on x axis
yaw = tilting on z axis
roll = tilting on y axis
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

if __name__ == '__main__':
    """
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

    HRTF_path = "mit_kemar_normal_pinna.sofa"
    HRTF = sofa.Database.open(HRTF_path)
    HRTF.Metadata.dump()

    # plot source positions
    # cartesian_source_positions = HRTF.Source.Position.get_values(system="cartesian")
    # with open('cartesian_source_positions.csv', 'w', newline='') as file:
    #     mywriter = csv.writer(file, delimiter=',')
    #     mywriter.writerows(cartesian_source_positions)
    
    spherical_source_positions = HRTF.Source.Position.get_values(system="spherical")
    # with open('spherical_source_positions.csv', 'w', newline='') as file:
    #     mywriter = csv.writer(file, delimiter=',')
    #     mywriter.writerows(spherical_source_positions)
    
    # hrtf_sampling_rate = HRTF.Data.SamplingRate.get_values(indices={"M": measurement})
    # print("hrtf sampling rate =", hrtf_sampling_rate)
    # plot_coordinates(source_positions, 'Source positions')

    emitter = 0

    # sin_440.wav -> mono
    # piano.wav / .mp3 -> stereo
    # don't_start_now.m4a -> stereo

    filename = 'piano.wav'
    segment = AudioSegment.from_file(filename)
    channel_sounds = segment.split_to_mono()
    
    """
    AudioSegment(…).get_array_of_samples()
    Returns the raw audio data as an array of (numeric) samples.
    Note: if the audio has multiple channels, the samples for each channel will be serialized
    – for example, stereo audio would look like [sample_1_L, sample_1_R, sample_2_L, sample_2_R, …]
    """
    samples = [s.get_array_of_samples() for s in channel_sounds]
    audio_np_array = np.array(samples).T
    print('audio shape = ', audio_np_array.shape)

    hrtf_len = len(HRTF.Data.IR.get_values(indices={"M":0, "R":0, "E":emitter}))
    print('hrtf_len = ', hrtf_len)

    azimuth_resolution = 5
    num_angles = 360 // azimuth_resolution
    
    audio_len = audio_np_array.shape[0]
    chunk_len = audio_len // num_angles
    print('chunk_len = ', chunk_len)

    convolved_channel1 = np.zeros(1)
    convolved_channel2 = np.zeros(1)
    convolved_channel3 = np.zeros(1)
    convolved_channel4 = np.zeros(1)

    for i in range(num_angles):
        azimuth = i * (360 // num_angles)
        elevation = 0
        measurement = find_measurement(azimuth, elevation, spherical_source_positions)
        hrtf1 = HRTF.Data.IR.get_values(indices={"M":measurement, "R":0, "E":emitter})
        hrtf2 = HRTF.Data.IR.get_values(indices={"M":measurement, "R":1, "E":emitter})

        start_index = i*chunk_len
        end_index = (i+1)*chunk_len-1
        # print('start_index = ', start_index)
        # print('end_index = ', end_index)
        
        # win = signal.windows.hann(chunk_len)
        audio_chunk = audio_np_array[start_index:end_index, 0]
        # filtered_chunk = signal.convolve(audio_chunk, win, mode='same') / sum(win)

        convolved1 = np.array(signal.convolve(audio_chunk, hrtf1, mode='full'))
        convolved2 = np.array(signal.convolve(audio_chunk, hrtf2, mode='full'))
        # print('convolved1_len = ', len(convolved1))
        # print('convolved2_len = ', len(convolved2))
        convolved1 = np.trim_zeros(convolved1)
        convolved2 = np.trim_zeros(convolved2)
        # print('trimmed convolved1_len = ', len(convolved1))
        # print('trimmed convolved2_len = ', len(convolved2))
        convolved_channel1 = np.concatenate((convolved_channel1, convolved1))
        convolved_channel2 = np.concatenate((convolved_channel2, convolved2))
    
    filename = 'sin_440.wav'
    segment = AudioSegment.from_file(filename)
    channel_sounds = segment.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]
    audio_np_array = np.array(samples).T
    print('audio shape = ', audio_np_array.shape)

    azimuth_resolution = 5
    num_angles = 360 // azimuth_resolution
    
    audio_len = audio_np_array.shape[0]
    chunk_len = audio_len // num_angles
    
    for i in range(num_angles):
        azimuth2 = 360 - i * (360 // num_angles)
        measurement2 = find_measurement(azimuth2, elevation, spherical_source_positions)
        hrtf3 = HRTF.Data.IR.get_values(indices={"M":measurement2, "R":0, "E":emitter})
        hrtf4 = HRTF.Data.IR.get_values(indices={"M":measurement2, "R":1, "E":emitter})

        start_index = i*chunk_len
        end_index = (i+1)*chunk_len-1
        audio_chunk = audio_np_array[start_index:end_index, 0]

        convolved3 = np.array(signal.convolve(audio_chunk, hrtf3, mode='full'))
        convolved4 = np.array(signal.convolve(audio_chunk, hrtf4, mode='full'))
        convolved3 = np.trim_zeros(convolved3)
        convolved4 = np.trim_zeros(convolved4)

        convolved_channel3 = np.concatenate((convolved_channel3, convolved3))
        convolved_channel4 = np.concatenate((convolved_channel4, convolved4))

    # write to a wav file
    # left_len = len(convolved_channel1)
    # right_len = len(convolved_channel2)
    # pad_len = abs(left_len - right_len)
    # print(pad_len)

    # if left_len < right_len: 
    #     convolved_channel1 = np.pad(convolved_channel1, (0, pad_len), 'constant')
    # else: 
    #     convolved_channel2 = np.pad(convolved_channel2, (0, pad_len), 'constant')
    
    # print("left convolved len = ", len(convolved_channel1))
    # print("right convolved len = ", len(convolved_channel2))

    source_len1 = len(convolved_channel1)
    source_len2 = len(convolved_channel3)
    pad_len = abs(source_len1 - source_len2)
    
    if source_len1 < source_len2:
        convolved_channel1 = np.pad(convolved_channel1, (0, pad_len), 'constant')
        convolved_channel2 = np.pad(convolved_channel2, (0, pad_len), 'constant')
    else:
        convolved_channel3 = np.pad(convolved_channel3, (0, pad_len), 'constant')
        convolved_channel4 = np.pad(convolved_channel4, (0, pad_len), 'constant')

    convolved_sum1 = np.add(convolved_channel1, convolved_channel3)
    convolved_sum2 = np.add(convolved_channel2, convolved_channel4)
    comb = np.array([convolved_sum1, convolved_sum2]).T
    norm = np.array(normalize(comb))
    num_bit = 16
    bit_depth = 2 ** (num_bit-1)
    comb2 = np.int16(norm/np.max(np.abs(norm)) * bit_depth-1)
    filename = "test_audio_files/" + filename.partition('.')[0] + "_surround_2source_" + str(bit_depth-1) + "_trim.wav"
    scipy.io.wavfile.write(filename, int(segment.frame_rate), comb2)

