import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

y, sr = librosa.load('./archive/Data/genres_original/blues/blues.00000.wav')

d = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(d, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Graphical representation of the audio signal')
plt.show()