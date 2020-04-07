# Inspiration at: https://medium.com/@LeonFedden/comparative-audio-analysis-with-wavenet-mfccs-umap-t-sne-and-pca-cb8237bfce2f

import os
import librosa
import librosa.display
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt

file_path = "INPUTSAMPLE.wav"

""" Loading a whole folder:
directory = './path/to/my/audio/folder/'

for file in os.listdir(directory):
    if file.endswith('.wav'):
        file_path = os.path.join(directory, file)
        audio_data, _ = librosa.load(file_path)
"""


# LIBROSA:
# https://librosa.github.io/librosa/master/feature.html?highlight=features

# 1 - MFCC
print("1 - MFCC")
sample_rate = 44100
sample_rate = 22050
mfcc_size = 13

# Load the audio
pcm_data, _ = librosa.load(file_path, duration=1) # load 1 sec!

_ = librosa.feature.mfcc(pcm_data, sample_rate, n_mfcc=mfcc_size)

start = timer()

# Compute a vector of n * 13 mfccs
mfccs = librosa.feature.mfcc(pcm_data, sample_rate, n_mfcc=mfcc_size)

end = timer()
time = (end - start)
print("mfccs run took " + str(time) + "s (" + str(time / 60.0) + "min)")

print("np.asarray(mfccs).shape", np.asarray(mfccs).shape) # 1sec = (13, 44) = 572 features! // 2sec = (13, 87)
# CPU - mfccs run took 0.004294439999284805s (7.157399998808008e-05min)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()



############################

start = timer()
chroma_cq = librosa.feature.chroma_cqt(pcm_data, sample_rate)
end = timer()
time = (end - start)
print("chroma_cq run took " + str(time) + "s (" + str(time / 60.0) + "min)")
print("np.asarray(chroma_cq).shape", np.asarray(chroma_cq).shape)
# chroma_cq run took 0.20499863999793888s (0.003416643999965648min)
# np.asarray(chroma_cq).shape (12, 44)

plt.figure()
librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time')
plt.title('chroma_cqt')
plt.colorbar()
plt.tight_layout()
plt.show()


############################



start = timer()

#S = np.abs(librosa.stft(pcm_data))
#chroma = librosa.feature.chroma_stft(S=S, sr=sample_rate)
S = np.abs(librosa.stft(pcm_data, n_fft=4096))**2
chroma = librosa.feature.chroma_stft(S=S, sr=sample_rate)

end = timer()
time = (end - start)
print("chroma run took " + str(time) + "s (" + str(time / 60.0) + "min)")
print("np.asarray(chroma).shape", np.asarray(chroma).shape)
# chroma run took 0.010794457000883995s (0.0001799076166813999min)
# np.asarray(chroma).shape (12, 22)

plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagram')
plt.tight_layout()
plt.show()



######################################################################################################################################################


# 2 - WaveNet
# install magenta -> https://github.com/tensorflow/magenta
# source activate magenta
print("2 - WaveNet")

from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet import fastgen

def wavenet_encode(file_path):
    
    # Load the model weights.
    checkpoint_path = './wavenet-ckpt/model.ckpt-200000'
    
    # Load and downsample the audio.
    neural_sample_rate = 16000
    audio = utils.load_audio(file_path, 
                             sample_length=400000, 
                             sr=neural_sample_rate)
    
    # Pass the audio through the first half of the autoencoder,
    # to get a list of latent variables that describe the sound.
    # Note that it would be quicker to pass a batch of audio
    # to fastgen. 
    encoding = fastgen.encode(audio, checkpoint_path, len(audio))
    
    # Reshape to a single sound.
    return encoding.reshape((-1, 16))
  

_ = wavenet_encode(file_path)

# An array of n * 16 frames. 
start = timer()
wavenet_z_data = wavenet_encode(file_path)
end = timer()
time = (end - start)
print("wavenet_encode run took " + str(time) + "s (" + str(time / 60.0) + "min)")
# Magenta-CPU wavenet_encode run took 30.804440063000584s (0.5134073343833431min)
# Magenta-GPU wavenet_encode run took 12.997072504000243s (0.2166178750666707min) <<<< STILL VERY SLOW!


# visualize with:
# 16 colors each one as one line in plot?
# reconstruct with:
# fastgen.synthesize (like in enc > z < dec architecture)



print("np.asarray(wavenet_z_data).shape", np.asarray(wavenet_z_data).shape) # np.asarray(wavenet_z_data).shape (781, 16)



