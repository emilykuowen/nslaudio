import keyboard
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import sofa
from pydub import AudioSegment
import pydub
import io
import scipy.io.wavfile
from scipy.io.wavfile import write
import csv
import pandas as pd
import wave
import pyaudio
import wave
import sys

class AudioFile:
    chunk = 1024

    def __init__(self, file):
        """ Initialize """
        self.wf = wave.open(file, 'rb')
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format = self.p.get_format_from_width(self.wf.getsampwidth()),
            channels = self.wf.getnchannels(),
            rate = self.wf.getframerate(),
            output = True
        )

    def play(self):
        """ Play entire file """
        data = self.wf.readframes(self.chunk)
        while data != b'':
            self.stream.write(data)
            data = self.wf.readframes(self.chunk)

    def close(self):
        """ Close stream """
        self.stream.close()
        self.p.terminate()

class hrtfFile:
    emitter = 0

    def __init__(self, file):
        """ Initialize """
        self.HRTFPath = file
        self.HRTF = sofa.Database.open(HRTFPath)
        self.HRTF.Metadata.dump()
        self.sphericalPositions = self.HRTF.Source.Position.get_values(system="spherical")
        self.measurement = 0

    def getIR(self, azimuth, elevation):
        """ Access IR measurements """
        self.measurement = findMeasurement(azimuth, elevation)
        hrtf1 = HRTF.Data.IR.get_values(indices={"M":self.measurement, "R":0, "E":self.emitter})
        hrtf2 = HRTF.Data.IR.get_values(indices={"M":self.measurement, "R":1, "E":self.emitter})
        return [hrtf1, hrtf2]

    def findMeasurement(self, azimuth, elevation):
        """ Find closest IR measurement to target azimuth and elevation """
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

class Listener:
    
    def __init__(self):
        """ Initialilze """
        self.xPos = 0
        self.yPos = 0
        self.zPos = 0

    def getPos(self):
        """ Access position info """
        return [self.xPos, self.yPos, self.zPos]

    def update(self, x, y, z):
        """ Update position """
        self.xPos = x
        self.yPos = y
        self.zPos = z


class Scene:

    def __init__(self, sourceFilename, HRTFFilename):
        """ Initialize """
        self.listener = Listener()
        self.HRTF = hrtfFile(HRTFFilename)
        #TODO use sourceFilename to open a .txt file (or something else) and create an array of source objects to store in self.sources[] array 
        self.sources = []
        self.stream = AudioStream()

    def generateChunk(self):
        """" Generate and queue an audio chunk """
        #TODO as this is, it will spatialize an entire source object audio file every time this is called. Next, this needs to be split to play a smaller chunk at a time
        for currSource in self.sources:
            [azimuth, elevation] = getAngles(currSource)
            [hrtf1, hrtf2] = self.HRTF.getIR(azimuth, elevation)

            convolved1 = np.array(signal.convolve(currSource.getSound(), hrtf1, mode='full'))
            convolved2 = np.array(signal.convolve(currSource.getSound(), hrtf2, mode='full'))

            start_index = min(np.flatnonzero(convolved1)[0], np.flatnonzero(convolved2)[0])
            end_index = max(np.flatnonzero(convolved1)[len(np.flatnonzero(convolved1))-1], np.flatnonzero(convolved2)[len(np.flatnonzero(convolved2))-1])

            convolved1 = convolved1[start_index:end_index]
            convolved2 = convolved2[start_index:end_index]

            #TODO adjust gain for inverse squared distance relationship

            #TODO check if this is the right way to add stereo audio data to stream
            self.stream.queueChunk([convolved1, convolved2])

    
    def getAngles(self, source):
        """ Calculate azimuth and elevation angle from listener to object """
        [sourceX, sourceY, sourceZ] = source.getPos()
        # TODO calculate azimuth and elevation angles from a given source object and listener object
        azimuth = 0
        elevation = 0
        return [azimuth, elivation]

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

class AudioStream:

    def __init__(self):
        """ Initialize """
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(channels = 2, rate = 44100, output = True)
    
    def queueChunk(self, chunk):
        """ Write audio to stream """
        self.stream.write(chunk)


""" MAIN """

## TODO Scene() create function to call generate chunk every "x" seconds
## TODO Scene() Add keylisteners to update listener position
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
