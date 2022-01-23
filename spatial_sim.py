import numpy
import scipy
from scipy.io import wavfile
import keyboard
from scipy import signal
#https://python-sofa.readthedocs.io/en/latest/index.html
#https://www.sofaconventions.org/mediawiki/index.php/SOFA_specifications

import sofa
import sys

pitch = 0
yaw = 0
roll = 0
x = 0
y = 0
z = 0

def load_file(filename):
    samplerate, data = wavfile.read(filename)
    return data

#KEY LISTENERS
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

def main():
    global pitch, roll, yaw, x, y, z
    audiofile = load_file("piano2.wav")
    keyboard.add_hotkey('up', on_up)
    keyboard.add_hotkey('down', on_down)
    keyboard.add_hotkey('right', on_right)
    keyboard.add_hotkey('left', on_left)
    keyboard.add_hotkey('.', on_period)
    keyboard.add_hotkey(',', on_comma)
    keyboard.add_hotkey('w', on_w)
    keyboard.add_hotkey('a', on_a)
    keyboard.add_hotkey('s', on_s)
    keyboard.add_hotkey('d', on_d)
    keyboard.add_hotkey('space', on_space)
    keyboard.add_hotkey('shift', on_shift)

    HRTF_path = "mit_kemar_normal_pinna.sofa"
    HRTF = sofa.Database.open(HRTF_path)
    HRTF.Metadata.dump()

    while True:
        print("PITCH - ",pitch," YAW - ", yaw, " ROLL - ", roll, "X - ",x," Y - ", y, " Z - ", z, )
    
main()


