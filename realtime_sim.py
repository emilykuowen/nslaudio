import keyboard
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import sofa
from pydub import AudioSegment
from scipy.io.wavfile import write
import csv
import wave
import pyaudio
import wave
import time
import math
from pynput import keyboard

class AudioStream:
    def __init__(self, file):
        """ Initialize """
        self.wf = wave.open(file, 'rb')
        self.p = pyaudio.PyAudio()

        # http://people.csail.mit.edu/hubert/pyaudio/docs/
        print(self.wf.getsampwidth()) # 2 --> 16-bit int
        print(self.p.get_format_from_width(self.wf.getsampwidth())) # 8 --> 16-bit int
        print(self.wf.getnchannels()) # 1
        print(self.wf.getframerate()) # 44100

        self.stream = self.p.open(
            format = self.p.get_format_from_width(self.wf.getsampwidth()), # 16-bit int
            channels = 1, # 1 channel
            rate = self.wf.getframerate(), # 44100 Hz
            output = True
        )

    def play(self):
        """ Play entire file """
        data = self.wf.readframes(1024)

        while data != b'':
            self.stream.write(data)
            data = self.wf.readframes(1024)

    def queueChunk(self, chunk):
        """ Write audio to stream """
        self.stream.write(chunk)

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

    def getPos(self):
        """ Access position info """
        return [self.xPos, self.yPos, self.zPos]

    def update(self, x, y, z, az, el):
        """ Update position """
        self.xPos = self.xPos + x
        self.yPos = self.yPos + y
        self.zPos = self.zPos + z
        self.azimuthTilt = self.azimuthTilt + az
        self.elevationTilt = self.elevationTilt + el


class Source:
    def __init__(self, x, y, z, filename):
        """ Initialize """
        self.xPos = x
        self.yPos = y
        self.zPos = z
        
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
    
    def getStream(self):
        return self.stream
    
    def getNextChunk(self, chunkSize):
        return self.stream.wf.readframes(chunkSize)


class Scene:
    def __init__(self, sourceFilename, HRTFFilename, global_listener):
        """ Initialize """
        self.listener = global_listener
        self.HRTF = HRTFFile(HRTFFilename)
        self.sources = [Source(0, 0, 0, "audio_sources/sin_440.wav"), 
                        Source(5, 0, 0, "audio_sources/sweep.wav"),
                        Source(-3, -3, 0, "audio_sources/sin_600Hz.wav")]
        #TODO change self.stream to initialize NOT using a normal file, but instead set parameters to match what we want the output to be as
        self.stream = AudioStream("audio_sources/piano.wav")
        self.chunkSize = 4096 # the larger the chunk size, the less noise / pauses
        self.timeIndex = 0
        self.fs = 44100
        self.exit = False
        

    def begin(self):
        # chunkTime = 1.0 / self.fs * self.chunkSize
        # print("chunk time = ", chunkTime)

        # print(list(data))
        # print("data len = ", len(list(data)))
        # print(data_np)
        # print("data np len = ", len(data_np))

        #while data != b'':
        while ~self.exit:
            [x, y, z] = self.listener.getPos()
            print("POSITION x = ", x, " y = ", y, " z = ", z)
            convolved = self.generateChunk()
            self.stream.stream.write(convolved)
            
            #time.sleep(2)

    def quit(self):
        self.exit = True

    def generateChunk(self):
        """" Generate an audio chunk """
        flag = 0

        for currSource in self.sources:
            data = currSource.getNextChunk(self.chunkSize) # returns byte string
            data_np = np.frombuffer(data, dtype=np.uint16) # returns int array of chunk_size if mono and chunk_size * 2 if stereo

            [azimuth, elevation] = self.getAngles(currSource)
            [hrtf1, hrtf2] = self.HRTF.getIR(azimuth, elevation)
            
            convolved1 = np.array(signal.fftconvolve(data_np, hrtf1, mode='full'))
            convolved2 = np.array(signal.fftconvolve(data_np, hrtf2, mode='full'))
            
            #print("data len = ", len(data))
            #print("hrtf1 len = ", len(hrtf1))
            #print("azimuth = ", azimuth)
            #print("elevation = ", elevation)
            #print("convolved1 size = ", len(convolved1))
            #print("convolved2 size = ", len(convolved2)) 

            #TODO adjust gain for inverse squared distance relationship

            convolved = np.array([convolved1, convolved2]).T
            #print(convolved.shape)
            norm = np.linalg.norm(convolved)
            convolved_normalized = convolved / norm
            #print(convolved_normalized.shape)
            num_bit = 16
            bit_depth = 2 ** (num_bit-1)
            convolved_final = np.int16(convolved_normalized / np.max(np.abs(convolved_normalized)) * (bit_depth-1))
            interleaved = convolved_final.flatten()
            out_data = interleaved.tobytes()

            if(flag == 0):
                summed = convolved_final
                flag = 1
            else:
                # gives me array size mismatch error sometimes
                # do we not need to normalize this after summing different sources?
                summed = summed + convolved_final
        
        # what is the goal of dividing?
        summed = summed/len(self.sources)
        interleaved = summed.flatten()
        out_data = interleaved.tobytes()

        return out_data

    def getAngles(self, source):
        """ Calculate azimuth and elevation angle from listener to object """
        [sourceX, sourceY, sourceZ] = source.getPos()
        [listenerX, listenerY, listenerZ] = self.listener.getPos()
        
        ydiff = sourceY - listenerY
        xdiff = sourceX - listenerX
        
        # calculate azimuth
        if(xdiff == 0):
            if(sourceY >= listenerY):
                azimuth = 0
            else:
                azimuth = 180
        elif(ydiff == 0):
            if(sourceX >= listenerX):
                azimuth = 90
            else:
                azimuth = -90
        else:
            if((listenerY > sourceY)):
                azimuth = math.degrees(math.atan(ydiff / xdiff) - math.pi)
            else:
                azimuth = math.degrees(math.atan(ydiff / xdiff))
        if(azimuth > 180):
            azimuth = -1* (360-abs(azimuth))
        if(azimuth < -180):
            azimuth = 360-abs(azimuth)

        #TODO CALCULATE ELEVATION
        #zdiff = sourceZ - listenerZ
        #xdiff = math.sqrt( ((sourceX - listenerX)**2) + ((sourceY - listenerY)**2) )
        #elevation = math.atan(numerator / xdiff)

        elevation = 0

        return [azimuth, elevation]


def on_press(key):
    global global_listener
    try:    
        if(key.char == 'w'):
            global_listener.update(0, 0, 0, 10, 0)
        if(key.char == 'a'):
            global_listener.update(0, 0, 0, 0, -10)
        if(key.char == 's'):
            global_listener.update(0, 0, 0, -10, 0)
        if(key.char == 'd'):
            global_listener.update(0, 0, 0, 0, 10)
        else:
            print("unknown input")
    except:
        if(key == keyboard.Key.up):
            global_listener.update(0, 1, 0, 0, 0)
        elif(key == keyboard.Key.right):
            global_listener.update(1, 0, 0, 0, 0)
        elif(key == keyboard.Key.left):
            global_listener.update(-1, 0, 0, 0, 0)
        elif(key == keyboard.Key.down):
            global_listener.update(0, -1, 0, 0, 0)
        elif(key == keyboard.Key.space):
            global_listener.update(0, 0, 1, 0, 0)
        elif(key == keyboard.Key.shift):
            global_listener.update(0, 0, -1, 0, 0)
        else:
            print("unknown input")


if __name__ == "__main__":
    global_listener = Listener()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    currentScene = Scene("audio_sources/sin_440.wav", "hrtf/mit_kemar_normal_pinna.sofa", global_listener)
    currentScene.begin()


## TODO Scene(), Source() Figure out format for Source object files.
    ## Each source object should have some kind of txt or csv file containing info on its audio file and position data
    ## The sourceFilename string should be used to open a file or folder where we can parse that info, create a set of Source() objects, and place them in the Scene() self.sources() array
## TODO Scene() add gain adjustment and normalization to change volume with distance
## TODO Scene() Check how stereo audio can be written to a stream
## TODO Scene() Finish getAngles() function


# Example - process frame by frame
#a = AudioFile("test.wav")
#data = a.wf.readframes(a.chunk)
#while data != b'':
#    data_processed = data #do some convolution here
#    a.stream.write(data_processed)
#    data = a.wf.readframes(a.chunk)
#a.close()

