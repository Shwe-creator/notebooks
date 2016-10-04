import math
import numpy as np

import scipy.io.wavfile

def spow2_gt(x):
    return int(math.pow(2.0, (math.ceil(math.log(x, 2.0)))))

def wav_read(wav_fp):
	fs, wav = scipy.io.wavfile.read(wav_fp)
	if wav.dtype == np.int16:
		wav_f = wav.astype(np.float64) / 32768.0
	elif wav.dtype == np.float32:
		wav_f = wav
	else:
		raise NotImplementedError()
	return wav_f, fs

def wav_write(wav_fp, fs, wav_f, normalize=False):
	if normalize:
		wav_f_max = wav_f.max()
		if wav_f_max != 0.0:
			wav_f /= wav_f.max()
	wav_f = np.clip(wav_f, -1.0, 1.0)
	wav = (wav_f * 32767.0).astype(np.int16)
	scipy.io.wavfile.write(wav_fp, fs, wav)

# zero pads
def zero_pad_to_n(data, n, axis=-1, leftpad=False):
	axis = axis % data.ndim
	padding = []
	pad_amount = n - data.shape[axis]
	assert pad_amount >= 0
	for dim in xrange(data.ndim):
		if dim == axis:
			if leftpad:
				padding.append((pad_amount, 0))
			else:
				padding.append((0, pad_amount))
		else:
			padding.append((0, 0))
	return np.pad(data, padding, mode='constant', constant_values=0)

# chunk along axis, inserting new axis to left
def chunk(data, n, hop=None, axis=-1):
	hop = hop if hop else n
	assert n > 0
	assert hop > 0
	axis = axis % data.ndim
	data_len = data.shape[axis]
	lo = 0
	chunks = []
	while lo < data_len:
		hi = min(lo + n, data_len)
		chunk_ = np.take(data, range(lo, hi), axis=axis)
		if chunk_.shape[axis] < n:
			chunk_ = zero_pad_to_n(chunk_, n, axis=axis)
		chunks.append(chunk_)
		lo += hop
	return np.stack(chunks, axis=axis)

# (length, val) pairs
# pre, atk, dcy, sus, rel, empty
def linterp(val_start, pts, env_len):
	pt_lens = [pt[0] for pt in pts]
	pt_vals = [pt[1] for pt in pts]
	pt_lens = [int(env_len * (pt_len / sum(pt_lens))) for pt_len in pt_lens]
	pt_lens[-1] -= sum(pt_lens) - env_len
	env = []
	val_curr = val_start
	for pt_len, pt_val in zip(pt_lens, pt_vals):
		env.append(np.linspace(val_curr, pt_val, pt_len, endpoint=False))
		val_curr = pt_val
	return np.concatenate(env)