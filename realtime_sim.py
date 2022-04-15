import csv
import wave
import pyaudio
import wave
import time
import math
import sofa
import scipy
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.io.wavfile import write
from pynput import keyboard
import pandas as pd

outputData = np.array([])

#Instead of using blocking read/write in pyaudio, use a callback function in place to generate audio when needed
# https://stackoverflow.com/questions/62618934/pyaudio-how-to-access-stream-read-data-in-callback-non-blocking-mode

class AudioStream:
    def __init__(self, file, numchannels=1):
        """ Initialize """
        self.wf = wave.open(file, 'rb')
        self.p = pyaudio.PyAudio()

        self.stream = self.p.open(
            format = self.p.get_format_from_width(self.wf.getsampwidth()), # 16-bit int
            channels = numchannels, 
            rate = self.wf.getframerate(), # 44100 Hz
            output = True
        )

    def play(self):
        """ Play entire file """
        data = self.wf.readframes(1024)

        while data != b'':
            self.stream.write(data)
            data = self.wf.readframes(1024)

    def close(self):
        """ Close stream """
        self.stream.close()
        self.p.terminate()

class HRTFFile:
    emitter = 0

    def __init__(self, file):
        """ Initialize """
        self.HRTFPath = file
        self.HRTF = sofa.Database.open(self.HRTFPath)
        self.HRTF.Metadata.dump()
        self.sphericalPositions = self.HRTF.Source.Position.get_values(system="spherical")
        self.measurement = 0

    def findMeasurement(self, azimuth, elevation):
        """ Find closest IR measurement to target azimuth and elevation """
        # TODO try hashmap or binary search
        bestIndex = 0
        bestError = abs(azimuth - self.sphericalPositions[0][0]) + abs(elevation - self.sphericalPositions[0][1])
        for i in range(1, len(self.sphericalPositions)):
            azDiff = abs(azimuth - self.sphericalPositions[i][0])
            eleDiff = abs(elevation - self.sphericalPositions[i][1])
            currError = azDiff + eleDiff
            if(currError < bestError):
                bestIndex = i
                bestError = currError
        return bestIndex

    def getIR(self, azimuth, elevation):
        """ Access IR measurements """
        self.measurement = self.findMeasurement(azimuth, elevation)
        hrtf1 = self.HRTF.Data.IR.get_values(indices={"M":self.measurement, "R":0, "E":self.emitter})
        hrtf2 = self.HRTF.Data.IR.get_values(indices={"M":self.measurement, "R":1, "E":self.emitter})
        return [hrtf1, hrtf2]

class Listener:
    def __init__(self):
        """ Initialilze """
        self.xPos = 0
        self.yPos = 0
        self.zPos = 0
        self.azimuthTilt = 0
        self.elevationTilt = 0

    def getAngles(self):
        """ Access head tilt info """
        return[self.azimuthTilt, self.elevationTilt]

    def getPos(self):
        """ Access position info """
        return [self.xPos, self.yPos, self.zPos]

    def update(self, x, y, z, az, el):
        """ Update position """
        self.xPos = self.xPos + x
        self.yPos = self.yPos + y
        self.zPos = self.zPos + z

        self.azimuthTilt = self.azimuthTilt + az
        if(self.azimuthTilt < 0):
            self.azimuthTilt = 360 + self.azimuthTilt
        if(self.azimuthTilt >=360):
            self.azimuthTilt = self.azimuthTilt - 360

        self.elevationTilt = self.elevationTilt + el
        if(self.elevationTilt < -90):
            self.elevationTilt = -180 - self.elevationTilt
        if(self.elevationTilt >90):
            self.elevationTilt = 180 - self.elevationTilt

class Scene:
    def __init__(self, sourceFilename, HRTFFilename, global_listener):
        """ Initialize """
        self.listener = global_listener
        self.HRTF = HRTFFile(HRTFFilename)
        #self.sources = [Source(0, 0, 0, "sin_440.wav"), Source(5, 0, 0, "sweep.wav"), Source(-3, -3, 0, "sin_600Hz.wav")]
        #self.sources = [Source(-5, -5, 0, "sin_300.wav"), Source(5, 5, 0, "sin_500.wav")]
        self.sources = [Source(0, 0, -5, "audio_sources/piano.wav")]
        self.stream = AudioStream("sin_300.wav", 2)
        self.chunkSize = 4096
        self.timeIndex = 0
        self.fs = 44100
        self.exit = False

    def begin(self):
        """ Continuously generate and queue next chunk """
        while self.exit==False:
            [x, y, z] = self.listener.getPos()
            [az, el] = self.listener.getAngles()
            #print("POSITION x=", x, " y=", y, " z=", z)
            #print("ANGLES az = ", az, " el = ", el)

            convolved = self.generateChunk()
            if convolved == 'flag':
                continue

            self.stream.stream.write(convolved)

    def quit(self):
        """ Exit the Scene """
        self.exit = True
        global outputData
        pd.DataFrame(outputData).to_csv("realtimeCheck.csv")
        self.stream.close()
        scipy.io.wavfile.write('realtime_output.wav', 44100, outputData)
        

    def generateChunk(self):
        """" Generate an audio chunk """
        global outputData
        flag = 0
        for currSource in self.sources:
            data = currSource.getNextChunk(self.chunkSize)
            
            if data == b'':
                self.quit()
                return 'flag'

            data_np = np.frombuffer(data, dtype=np.int16)

            [azimuth, elevation, attenuation] = self.getAngles(currSource)
            [hrtf1, hrtf2] = self.HRTF.getIR(azimuth, elevation)
            
            #TODO attenuation/distance scaling doesn't work with one source
            convolved1 = np.array(signal.fftconvolve(data_np, hrtf1, mode='full')) * attenuation
            convolved2 = np.array(signal.fftconvolve(data_np, hrtf2, mode='full')) * attenuation
            
            convolved = np.array([convolved1, convolved2]).T

            if(flag==0):
                summed = convolved
                flag = 1
            else:
                summed = summed + convolved
        
        norm = np.linalg.norm(summed)
        convolved_normalized = summed / norm
        num_bit = 16
        bit_depth = 2 ** (num_bit-1)
        convolved_final = np.int16(convolved_normalized / np.max(np.abs(convolved_normalized)) * (bit_depth-1))
        interleaved = convolved_final.flatten()
        
        #Handle wav file output
        if outputData.size == 0:
            outputData = convolved_final
        else:
            outputData = np.append(outputData, convolved_final, axis=0)
            
        return interleaved

    def getAngles(self, source):
        """ Calculate azimuth and elevation angle from listener to object """
        [sourceX, sourceY, sourceZ] = source.getPos()
        [listenerX, listenerY, listenerZ] = self.listener.getPos()
        [listenerAz, listenerEl] = self.listener.getAngles()
        
        numerator = sourceY - listenerY
        denominator = sourceX - listenerX
        
        #Calculate Azimuth
        if(denominator == 0):
            if(sourceY >= listenerY):
                azimuth = 0
            else:
                azimuth = 180
        elif(numerator == 0):
            if(sourceX >= listenerX):
                azimuth = 90
            else:
                azimuth = 270
        else:
            if(listenerY > sourceY):
                azimuth = math.degrees(math.atan(numerator / denominator) - math.pi)
            else:
                azimuth = math.degrees(math.atan(numerator / denominator))
        if (azimuth < 0):
            azimuth = 360 + azimuth
        azimuth = azimuth - listenerAz

        #Calculate Elevation
        numerator = sourceZ - listenerZ
        denominator = math.sqrt( ((sourceX - listenerX)**2) + ((sourceY - listenerY)**2) )
        if(numerator == 0):
            elevation = 0
        elif(denominator == 0):
            if(sourceZ<listenerZ):
                elevation = -90
            else:
                elevation = 90
        else:
            elevation = math.degrees(math.atan(numerator / denominator))

        if(elevation > 90):
            elevation = 180 - elevation
        if(elevation <-90):
            elevation = -180 - elevation
        elevation = elevation - listenerEl

        distance = math.sqrt((sourceX - listenerX)**2 + (sourceY - listenerY)**2 + (sourceZ - listenerZ)**2)
        if distance == 0:
            attenuation = 1.0
        else:
            attenuation = 1.0 / (distance**2)

        return [azimuth, elevation, attenuation]

class Source:
    def __init__(self, x, y, z, filename):
        """ Initialize """
        self.xPos = x
        self.yPos = y
        self.zPos = z
        self.index = 0
        segment = AudioSegment.from_file(filename)
        channel_sounds = segment.split_to_mono()
        samples = [s.get_array_of_samples() for s in channel_sounds]
        self.audioArray = np.array(samples).T
        self.stream = AudioStream(filename)
    
    def getPos(self):
        """ Access position data """
        return [self.xPos, self.yPos, self.zPos]
        
    def getSound(self):
        """ Access audio info """
        return self.audioArray
    
    def getNextChunk(self, chunkSize):
        """ Access next chunk """
        return self.stream.wf.readframes(chunkSize)

def on_press(key):
    """ Add key listeners to main """

    """ 
    Arrow Keys move user around the horizontal plane (X and Y directions)
    Space moves user up in space, Shift moves user down in space (Z direction)
    W tilts users head up, S tilts users head down (Pitch)
    A tilts users head to the left, D tilts users head to the right (Yaw)
    """
    global global_listener
    try:    
        if(key.char == 'w'):
            global_listener.update(0, 0, 0, 0, 10)
        if(key.char == 'a'):
            global_listener.update(0, 0, 0, -10, 0)
        if(key.char == 's'):
            global_listener.update(0, 0, 0, 0, -10)
        if(key.char == 'd'):
            global_listener.update(0, 0, 0, 10, 0)
        else:
            print("unknown input")
    except:
        if(key == keyboard.Key.up):
            global_listener.update(0, 1, 0, 0, 0)
        elif(key == keyboard.Key.right):
            global_listener.update(-1, 0, 0, 0, 0)
        elif(key == keyboard.Key.left):
            global_listener.update(1, 0, 0, 0, 0)
        elif(key == keyboard.Key.down):
            global_listener.update(0, -1, 0, 0, 0)
        elif(key == keyboard.Key.space):
            global_listener.update(0, 0, 1, 0, 0)
        elif(key == keyboard.Key.shift):
            global_listener.update(0, 0, -1, 0, 0)
        else:
            print("unknown input")

#Azimuth - 0 to 360 counterclockwise, 0 in front
#Elevation - -90 to 0 to 90
if __name__ == "__main__":
    global_listener = Listener()
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    currentScene = Scene("sin_440.wav", "hrtf/mit_kemar_normal_pinna.sofa", global_listener)
    currentScene.begin()

## TODO Scene(), Source() Figure out format for Source object files.
    ## Each source object should have some kind of txt or csv file containing info on its audio file and position data
    ## The sourceFilename string should be used to open a file or folder where we can parse that info, create a set of Source() objects, and place them in the Scene() self.sources() array

