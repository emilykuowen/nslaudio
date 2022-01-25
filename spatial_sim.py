import keyboard
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sofa
from pydub import AudioSegment
import io
import scipy.io.wavfile

"""
Python SOFA API: https://python-sofa.readthedocs.io/en/latest/index.html
SOFA specifications: https://www.sofaconventions.org/mediawiki/index.php/SOFA_specifications
"""

"""
# x = forward, y = right, z = up
# pitch = tilting on x axis
# yaw = tilting on z axis
# roll = tilting on y axis

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

def find_measurement(az, ele, positions):
    best_fit = 0
    best_error = abs(az - positions[0][0]) + abs(ele - positions[0][1])
    for i in range(len(positions)):
        diff_az = abs(az - positions[i][0])
        diff_ele = abs(az - positions[i][1])
        if((diff_az + diff_ele) < best_error):
            best_fit = i
            best_error = (diff_az + diff_ele)

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
    HRTF_path = "mit_kemar_normal_pinna.sofa"
    HRTF = sofa.Database.open(HRTF_path)
    HRTF.Metadata.dump()

    emitter = 0

    print("Azimuth: ")
    azimuth = int(input(""))
    print("Elevation: ")
    elevation = int(input(""))
    positions = HRTF.Source.Position.get_values(system="spherical")
    measurement = find_measurement(azimuth, elevation, positions)

    # plot source positions
    source_positions = HRTF.Source.Position.get_values(system="cartesian")
    print(source_positions[measurement])
    hrtf_sampling_rate = HRTF.Data.SamplingRate.get_values(indices={"M": measurement})
    print("hrtf sampling rate =", hrtf_sampling_rate)
    # plot_coordinates(source_positions, 'Source positions')

    channel1 = HRTF.Data.IR.get_values(indices={"M":measurement, "R":0, "E":emitter})
    channel2 = HRTF.Data.IR.get_values(indices={"M":measurement, "R":1, "E":emitter})
    
    segment = AudioSegment.from_mp3('piano.mp3')
    print('audio file channel count =', segment.channels)
    print('audio file sample rate =', segment.frame_rate)

    # source: https://github.com/jiaaro/pydub/blob/master/API.markdown
    channel_sounds = segment.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]
    audio_np_array = np.array(samples).T
    print('audio np array shape = ', audio_np_array.shape)
    audio_len = audio_np_array.shape[0]
    print("audio length", audio_len)

    # zeropad hrtf to be the same length as the audio file
    hrtf1 = np.pad(channel1, (audio_len-len(channel1))//2)
    hrtf2 = np.pad(channel2, (audio_len-len(channel2))//2)
    # print(channel1)
    print("channel 1 shape", channel1.shape)
    # print(hrtf1)
    print("hrtf 1 shape", hrtf1.shape)

    convolved1 = signal.convolve(audio_np_array[:, 0], hrtf1, mode='same')
    convolved2 = signal.convolve(audio_np_array[:, 1], hrtf2, mode='same')
    comb = np.array([convolved1, convolved2]).T
    print("comb shape", comb.shape)
    norm = np.array(normalize(comb))
    
    # write to a wav file
    scipy.io.wavfile.write('piano_test.wav', int(hrtf_sampling_rate), norm)
    
    # plot Data.IR at M=5 for E=0
    # legend = []
    # t = np.arange(0,HRTF.Dimensions.N)*HRTF.Data.SamplingRate.get_values(indices={"M":measurement})
    # plt.figure(figsize=(15, 5))
    # for receiver in np.arange(HRTF.Dimensions.R):
    #     plt.plot(t, HRTF.Data.IR.get_values(indices={"M":measurement, "R":receiver, "E":emitter}))
    #     legend.append('Receiver {0}'.format(receiver))
    # plt.title('HRIR at M={0} for emitter {1}'.format(measurement, emitter))
    # plt.legend(legend)
    # plt.xlabel('$t$ in s')
    # plt.ylabel(r'$h(t)$')
    # plt.grid()
    # plt.show()

    # HRTF.close()

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
