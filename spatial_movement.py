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

"""
# pitch = 0
# yaw = 0
# roll = 0
# x = 0
# y = 0
# z = 0

# keyboard listeners
def on_up():
    global pitch
    pitch = pitch + 10
    if(pitch == 190):
        pitch = -170
def on_down():
    global pitch
    pitch = pitch - 10
    if(pitch == -190):
        pitch = 170
def on_right():
    global yaw
    yaw = yaw + 10
    if(yaw == 190):
        yaw = -170
def on_left():
    global yaw
    yaw = yaw - 10
    if(yaw == -190):
        yaw = 170 
def on_period():
    global roll
    roll = roll + 10
    if(roll == 190):
        roll = -170
def on_comma():
    global roll
    roll = roll - 10
    if(roll == -190):
        roll = 170
def on_w():
    global x
    x = x + 10
def on_s():
    global x
    x = x - 10
def on_d():
    global y
    y = y + 10
def on_a():
    global y
    y = y - 10
def on_space():
    global z
    z = z + 10
def on_shift():
    global z
    z = z - 10
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

    segment = AudioSegment.from_mp3('piano.mp3')
    channel_sounds = segment.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]
    audio_np_array = np.array(samples).T
    audio_len = audio_np_array.shape[0]
    hrtf_len = len(HRTF.Data.IR.get_values(indices={"M":0, "R":0, "E":emitter}))
    print('hrtf_len = ', hrtf_len)

    num_angles = audio_len // hrtf_len
    print('num_angles = ', num_angles)
    
    chunk_len = audio_len // num_angles
    print('chunk_len = ', chunk_len)

    convolved_channel1 = np.zeros(1)
    convolved_channel2 = np.zeros(1)

    for i in range(num_angles):
        azimuth = i * (360 / num_angles)
        print('azimuth = ', azimuth)
        elevation = 0
        measurement = find_measurement(azimuth, elevation, spherical_source_positions)
        hrtf1 = HRTF.Data.IR.get_values(indices={"M":measurement, "R":0, "E":emitter})
        hrtf2 = HRTF.Data.IR.get_values(indices={"M":measurement, "R":1, "E":emitter})
        len_diff = chunk_len - len(hrtf1)
        print('len_diff = ', len_diff)

        # zeropad hrtf at the end to make it the same length as the chunk
        hrtf1_padded = np.pad(hrtf1, (0, len_diff), 'constant')
        hrtf2_padded = np.pad(hrtf2, (0, len_diff), 'constant')
        print('hrtf1_padded_len = ', len(hrtf1_padded))
        print('hrtf2_padded_len = ', len(hrtf2_padded))

        convolved1 = np.array(signal.convolve(audio_np_array[i*chunk_len:(i+1)*chunk_len-1, 0], hrtf1_padded, mode='same'))
        convolved2 = np.array(signal.convolve(audio_np_array[i*chunk_len:(i+1)*chunk_len-1, 1], hrtf2_padded, mode='same'))
        print('convolved1_len = ', len(convolved1))
        print('convolved2_len = ', len(convolved2))

        convolved_channel1 = np.concatenate((convolved_channel1, convolved1))
        convolved_channel2 = np.concatenate((convolved_channel2, convolved2))
    
    # write to a wav file
    comb = np.array([convolved_channel1, convolved_channel2]).T
    norm = np.array(normalize(comb))
    comb2 = np.int16(norm/np.max(np.abs(norm)) * 32767)
    filename = "test_audio_files/piano_test_surround.wav"
    # scipy.io.wavfile.write(filename, int(segment.frame_rate), norm)
    scipy.io.wavfile.write(filename, int(segment.frame_rate), comb2)

    # global pitch, roll, yaw, x, y, z
    # audiofile = load_file("piano2.wav")
    # keyboard.add_hotkey('up', on_up)
    # keyboard.add_hotkey('down', on_down)
    # keyboard.add_hotkey('right', on_right)
    # keyboard.add_hotkey('left', on_left)
    # keyboard.add_hotkey('.', on_period)
    # keyboard.add_hotkey(',', on_comma)
    # keyboard.add_hotkey('w', on_w)
    # keyboard.add_hotkey('a', on_a)
    # keyboard.add_hotkey('s', on_s)
    # keyboard.add_hotkey('d', on_d)
    # keyboard.add_hotkey('space', on_space)
    # keyboard.add_hotkey('shift', on_shift)

    # HRTF_path = "mit_kemar_normal_pinna.sofa"
    # HRTF = sofa.Database.open(HRTF_path)
    # HRTF.Metadata.dump()

    # while True:
    #     print("PITCH - ",pitch," YAW - ", yaw, " ROLL - ", roll, "X - ",x," Y - ", y, " Z - ", z, )
