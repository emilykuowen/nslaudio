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
            channels = 2, # 1 channel
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
        keyboard.add_hotkey('up', self.update(0, 1, 0, 0, 0))
        keyboard.add_hotkey('down', self.update(0, -1, 0, 0, 0))
        keyboard.add_hotkey('right', self.update(1, 0, 0, 0, 0))
        keyboard.add_hotkey('left', self.update(-1, 0, 0, 0, 0))
        keyboard.add_hotkey('w', self.update(0, 0, 0, 0, 1))
        keyboard.add_hotkey('a', self.update(0, 0, 0, 1, 0))
        keyboard.add_hotkey('s', self.update(0, 0, 0, 0, -1))
        keyboard.add_hotkey('d', self.update(0, 0, 0, -1, 0))
        # keyboard.add_hotkey('space', self.update(0, 0, 1, 0, 0))
        # keyboard.add_hotkey('shift', self.update(0, 0, -1, 0, 0))
        self.update(0, 0, 0, 0, 0) # reset the positions

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


class Scene:
    def __init__(self, sourceFilename, HRTFFilename):
        """ Initialize """
        self.listener = Listener()
        self.HRTF = HRTFFile(HRTFFilename)
        #TODO read a text file that lists the source objects
        #TODO use sourceFilename to open a .txt file and create an array of source objects to store in self.sources[] array 
        self.sources = [Source(1, 0, 0, sourceFilename)]
        self.stream = AudioStream(sourceFilename)
        # the larger the chunk size, the less noise / pauses
        self.chunkSize = 8192
        self.timeIndex = 0
        self.fs = 44100
        self.exit = False

    def begin(self):
        # TODO make adding hotkey not call self.quit()
        # keyboard.add_hotkey('q', self.quit())
        # chunkTime = 1.0 / self.fs * self.chunkSize
        # print("chunk time = ", chunkTime)

        chunk_size = 4096
        data = self.stream.wf.readframes(chunk_size) # returns byte string
        data_np = np.frombuffer(data, dtype=np.uint16) # returns int array of chunk_size if mono and chunk_size * 2 if stereo
        
        # print(list(data))
        # print("data len = ", len(list(data)))
        # print(data_np)
        # print("data np len = ", len(data_np))

        while data != b'':
            data_np = np.frombuffer(data, dtype=np.uint16)
            convolved = self.generateChunk(data_np, chunk_size)
            self.stream.stream.write(convolved)
            data = self.stream.wf.readframes(chunk_size)

        # while(~self.exit):
        #     start_time = time.time()
        #     # TODO use a thread to separate generating and playing chunk
        #     # TODO test code with writing audio data to an array instead of outputting right away
        #     # TODO check how much time needs to be reduced / optimized
        #     convolved_final = self.generateChunk(self.timeIndex, self.chunkSize)
        #     print("chunk size = ", 1.0 / self.fs * len(convolved_final))
        #     print("processing time = ", time.time() - start_time)
        #     self.stream.queueChunk(convolved_final)
        #     # while((time.time() - start_time) < chunkTime):
        #     #     pass
        #     self.timeIndex = self.timeIndex + self.chunkSize

    def quit(self):
        self.exit = True
        # self.stream.close()

    def generateChunk(self, data, chunkSize):
        """" Generate an audio chunk """
        for currSource in self.sources:
            # [azimuth, elevation] = self.getAngles(currSource)
            # print("azimuth = ", azimuth)
            # print("elevation = ", elevation)
            [hrtf1, hrtf2] = self.HRTF.getIR(0, 0)
            print("data len = ", len(data))
            print("hrtf1 len = ", len(hrtf1))

            # TODO test using fft instead to see if it's faster
            convolved1 = np.array(signal.fftconvolve(data, hrtf1, mode='full'))
            convolved2 = np.array(signal.fftconvolve(data, hrtf2, mode='full'))
            print("convolved1 size = ", len(convolved1))
            print("convolved2 size = ", len(convolved2)) 

            # start_index = min(np.flatnonzero(convolved1)[0], np.flatnonzero(convolved2)[0])
            # end_index = max(np.flatnonzero(convolved1)[len(np.flatnonzero(convolved1))-1], np.flatnonzero(convolved2)[len(np.flatnonzero(convolved2))-1])
            # convolved1 = convolved1[start_index:end_index]
            # convolved2 = convolved2[start_index:end_index]   

            #TODO adjust gain for inverse squared distance relationship

            convolved = np.array([convolved1, convolved2]).T
            print(convolved.shape)
            norm = np.linalg.norm(convolved)
            convolved_normalized = convolved / norm
            print(convolved_normalized.shape)
            num_bit = 16
            bit_depth = 2 ** (num_bit-1)
            convolved_final = np.int16(convolved_normalized / np.max(np.abs(convolved_normalized)) * (bit_depth-1))
            interleaved = convolved_final.flatten()
            out_data = interleaved.tobytes()
            return out_data
    
    # def generateChunk(self, timeIndex, chunkSize):
    #     """" Generate an audio chunk """
    #     for currSource in self.sources:
    #         [azimuth, elevation] = self.getAngles(currSource)
    #         print("azimuth = ", azimuth)
    #         print("elevation = ", elevation)
    #         [hrtf1, hrtf2] = self.HRTF.getIR(azimuth, elevation)

    #         soundFile = currSource.getSound()
    #         # TODO make this work for both mono and stereo files
    #         soundChunk = soundFile[timeIndex:timeIndex+chunkSize, 0]
    #         print("sound chunk size = ", len(soundChunk))

    #         # TODO test using fft instead to see if it's faster
    #         convolved1 = np.array(signal.fftconvolve(soundChunk, hrtf1, mode='valid'))
    #         convolved2 = np.array(signal.fftconvolve(soundChunk, hrtf2, mode='valid'))
    #         print("convolved1 size = ", len(convolved1))
    #         print("convolved2 size = ", len(convolved2))

    #         # start_index = min(np.flatnonzero(convolved1)[0], np.flatnonzero(convolved2)[0])
    #         # end_index = max(np.flatnonzero(convolved1)[len(np.flatnonzero(convolved1))-1], np.flatnonzero(convolved2)[len(np.flatnonzero(convolved2))-1])
    #         # convolved1 = convolved1[start_index:end_index]
    #         # convolved2 = convolved2[start_index:end_index]   

    #         #TODO adjust gain for inverse squared distance relationship

    #         #TODO check if this is the right way to add stereo audio data to stream
    #         convolved = np.array([convolved1, convolved2])
    #         norm = np.linalg.norm(convolved)
    #         convolved_normalized = convolved / norm
    #         num_bit = 16
    #         bit_depth = 2 ** (num_bit-1)
    #         convolved_final = np.int16(convolved_normalized / np.max(np.abs(convolved_normalized)) * (bit_depth-1))
    #         return convolved_final.tobytes()

    # will have to take in the listener object as an argument
    def getAngles(self, source):
        """ Calculate azimuth and elevation angle from listener to object """
        [sourceX, sourceY, sourceZ] = source.getPos()
        [listenerX, listenerY, listenerZ] = self.listener.getPos()
        print("listenerX = ", listenerX)
        print("listenerY = ", listenerY)
        print("listenerZ = ", listenerZ)
        
        numerator = sourceY - listenerY
        denominator = sourceX - listenerX
        azimuth = math.atan(numerator / denominator)

        numerator = sourceZ - listenerZ
        denominator = math.sqrt( ((sourceX - listenerX)**2) + ((sourceY - listenerY)**2) )
        elevation = math.atan(numerator / denominator)

        return [azimuth, elevation]


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
    
    def getPos(self):
        """ Access position data """
        return [self.xPos, self.yPos, self.zPos]
        
    def getSound(self):
        """ Access audio info """
        return self.audioArray


if __name__ == "__main__":
    currentScene = Scene("sin_440.wav", "mit_kemar_normal_pinna.sofa")
    # currentScene.stream.play()
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

