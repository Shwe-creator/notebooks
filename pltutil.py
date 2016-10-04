import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import matplotlib.colors as colors
import scipy.signal

def sound_preview(wav, fs, nfft=1024, noverlap=512):
	if wav.ndim > 1:
		wav = np.mean(wav, axis=1)
	dt = 1.0 / fs
	wavlen = wav.shape[0]
	wavlens = wavlen * dt

	# plot
	fig, axarr = plt.subplots(2)#, sharex=True)
	waveax, specax = axarr

	# plot waveform
	timex = np.linspace(0.0, wavlens, wavlen, endpoint=False)
	waveax.plot(timex, wav)
	waveax.set_xlim([timex[0], timex[-1]])
	waveax.set_ylim([-1.0, 1.0])
	waveax.set_xlabel('Time (s)')
	waveax.set_ylabel('Amplitude')

	# compute spectrogram
	specy, specx, spec = scipy.signal.spectrogram(wav, fs, nperseg=nfft, noverlap=noverlap)
	specx = np.linspace(0.0, wavlens, spec.shape[1], endpoint=False)

	# plot spectrogram
	specax.pcolormesh(specx, specy, spec, norm=colors.LogNorm(vmin=spec.min(), vmax=spec.max()), cmap='magma', shading='gouraud')
	specax.set_xlabel('Time (s)')
	specax.set_ylabel('Frequency (Hz)')
	specax.set_xlim([specx[0], specx[-1]])
	specax.set_ylim([100, 20000])
	specax.set_yscale('log')

	plt.show()