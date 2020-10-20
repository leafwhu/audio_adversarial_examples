'''
!curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.python.sh | bash

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x
!python -m pip install tensorflow-gpu==1.14.0 progressbar numpy scipy pandas python_speech_features tables attrdict pyxdg pydub
!git clone https://github.com/mozilla/DeepSpeech.git
!cd DeepSpeech && git checkout tags/v0.6.1 && python -m pip install $(python util/taskcluster.py --decoder)
!cd DeepSpeech && git checkout tags/v0.4.1

!wget https://github.com/mozilla/DeepSpeech/releases/download/v0.4.1/deepspeech-0.4.1-checkpoint.tar.gz
!tar -xzf deepspeech-0.4.1-checkpoint.tar.gz
'''

import numpy as np
import tensorflow as tf
import argparse
from shutil import copyfile
import scipy.io.wavfile as wav
from tqdm import tqdm, trange

from sklearn.metrics import roc_auc_score
import struct
import time
import os
import sys
from collections import namedtuple

sys.path.append("DeepSpeech")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

try:
    import pydub
except:
    print("pydub was not loaded, MP3 compression will not work")

import DeepSpeech

from tensorflow.python.keras.backend import ctc_label_dense_to_sparse
toks = " abcdefghijklmnopqrstuvwxyz'-"


def get_cer(r: list, h: list):
    """
    Calculation of CER with Levenshtein distance.
    """
    # initialisation
    import numpy
    d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint16)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

# computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)] / float(len(h))


def compute_mfcc(audio, **kwargs):
    """
    Compute the MFCC for a given audio waveform. This is
    identical to how DeepSpeech does it, but does it all in
    TensorFlow so that we can differentiate through it.
    """

    batch_size, size = audio.get_shape().as_list()
    audio = tf.cast(audio, tf.float32)

    # 1. Pre-emphasizer, a high-pass filter
    audio = tf.concat(
        (audio[:, :1], audio[:, 1:] - 0.97 * audio[:, :-1], np.zeros((batch_size, 512), dtype=np.float32)), 1)

    # 2. windowing into frames of 512 samples, overlapping
    windowed = tf.stack([audio[:, i:i + 512] for i in range(0, size - 320, 320)], 1)

    window = np.hamming(512)
    windowed = windowed * window

    # 3. Take the FFT to convert to frequency space
    ffted = tf.spectral.rfft(windowed, [512])
    ffted = 1.0 / 512 * tf.square(tf.abs(ffted))

    # 4. Compute the Mel windowing of the FFT
    energy = tf.reduce_sum(ffted, axis=2) + np.finfo(float).eps
    filters = np.load("filterbanks.npy").T
    feat = tf.matmul(ffted, np.array([filters] * batch_size, dtype=np.float32)) + np.finfo(float).eps

    # 5. Take the DCT again, because why not
    feat = tf.log(feat)
    feat = tf.spectral.dct(feat, type=2, norm='ortho')[:, :, :26]

    # 6. Amplify high frequencies for some reason
    _, nframes, ncoeff = feat.get_shape().as_list()
    n = np.arange(ncoeff)
    lift = 1 + (22 / 2.) * np.sin(np.pi * n / 22)
    feat = lift * feat
    width = feat.get_shape().as_list()[1]

    # 7. And now stick the energy next to the features
    feat = tf.concat((tf.reshape(tf.log(energy), (-1, width, 1)), feat[:, :, 1:]), axis=2)

    return feat


def get_logits(new_input, length, first=[]):
    """
    Compute the logits for a given waveform.

    First, preprocess with the TF version of MFC above,
    and then call DeepSpeech on the features.
    """

    batch_size = new_input.get_shape()[0]

    # 1. Compute the MFCCs for the input audio
    # (this is differentable with our implementation above)
    empty_context = np.zeros((batch_size, 9, 26), dtype=np.float32)
    new_input_to_mfcc = compute_mfcc(new_input)
    features = tf.concat((empty_context, new_input_to_mfcc, empty_context), 1)

    # 2. We get to see 9 frames at a time to make our decision,
    # so concatenate them together.
    features = tf.reshape(features, [new_input.get_shape()[0], -1])
    features = tf.stack([features[:, i:i + 19 * 26] for i in range(0, features.shape[1] - 19 * 26 + 1, 26)], 1)
    features = tf.reshape(features, [batch_size, -1, 19, 26])

    # 3. Finally we process it with DeepSpeech
    # We need to init DeepSpeech the first time we're called
    if first == []:
        first.append(False)

        DeepSpeech.create_flags()
        tf.app.flags.FLAGS.alphabet_config_path = "DeepSpeech/data/alphabet.txt"
        DeepSpeech.initialize_globals()

    logits, _ = DeepSpeech.BiRNN(features, length, [0] * 10)

    return logits


def classify(args):
    # print(args.input.split(".")[-1])
    if args.input.split(".") == 'mp3':
        raw = pydub.AudioSegment.from_mp3(args.input)
        audio = np.array([struct.unpack("<h", raw.raw_data[i:i+2])[0] for i in range(0,len(raw.raw_data),2)])
    elif args.input.split(".")[-1] == 'wav':
        _, audio = wav.read(args.input)
    elif args.input.split(".")[-1] == 'flac':
        raw = pydub.AudioSegment.from_file(args.input)
        # print(raw.frame_rate)
        audio = np.array([struct.unpack("<h", raw.raw_data[i:i + 2])[0] for i in range(0, len(raw.raw_data), 2)])
    else:
        raise Exception("Unknown file format")

    if args.split > 0:
        split = int(len(audio) * args.split)
        audio_lst = [audio[:split]]
    else:
        audio_lst = [audio]
    result=""

    for audio in audio_lst:
        # print("length of audio_list:",len(audio_lst))
        tf.reset_default_graph()
        with tf.Session() as sess:

            N = len(audio)

            new_input = tf.placeholder(tf.float32, [1, N])
            lengths = tf.placeholder(tf.int32, [1])

            with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                logits = get_logits(new_input, lengths)

            saver = tf.train.Saver()
            saver.restore(sess, args.restore_path)

            decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=500)

            # print('logits shape', logits.shape)

            length = (len(audio)-1)//320
            l = len(audio)


            r = sess.run(decoded, {new_input: [audio],
                                lengths: [length]})
            for x in r[0].values:
                result += toks[x]

            # print("-"*80)
            # print("-"*80)

            # print("Classification:")
            # print("".join([toks[x] for x in r[0].values]))
            # print("-"*80)
            # print("-"*80)
    return result


restore = "DeepSpeech/deepspeech-0.4.1-checkpoint/model.v0.4.1"

def cal_cerauc(cleanfile_path,advfile_path):
    class Args():
        def __init__(self,input,restore_path,split):
            self.input = input
            # self.input = "/home/liuye/code/audio_adversarial_examples/audio/adv.wav1"
            self.restore_path = restore_path
            self.split = split

    """ 启动检测"""

    cleanaudio = os.listdir(cleanfile_path)
    len_cleanaudio = len(cleanaudio)
    cleancer3=[]
    cleancer5=[]
    cleancer7=[]
    for file in tqdm(cleanaudio):
        if file.split(".")[-1] not in ["mp3", "flac", "wav"]:
            len_cleanaudio = len_cleanaudio - 1
            continue
        print("\n")
        print(cleanfile_path+file)
        args = Args(cleanfile_path+file, restore, 0)
        true_y=classify(args)
        print("true_y: {}".format(true_y))

        args= Args(cleanfile_path+file, restore, 0.3)
        clsf=classify(args)
        true_y3 = true_y[:int(0.3 * len(true_y))]
        cer3=get_cer(clsf,true_y3)
        print("true_y0.3: {}".format(true_y3))
        print("split 0.3 cer: {:.2f} trans: {}".format(cer3, clsf))

        args= Args(cleanfile_path+file, restore, 0.5)
        clsf=classify(args)
        true_y5 = true_y[:int(0.5 * len(true_y))]
        cer5=get_cer(clsf,true_y5)
        print("true_y0.5: {}".format(true_y5))
        print("split 0.5 cer: {:.2f} trans: {}".format(cer5, clsf))


        args= Args(cleanfile_path+file, restore, 0.7)
        clsf=classify(args)
        true_y7 = true_y[:int(0.7 * len(true_y))]
        cer7=get_cer(clsf,true_y7)
        print("true_y0.7: {}".format(true_y7))
        print("split 0.7 cer: {:.2f} trans: {}".format(cer7, clsf))

        cleancer3.append(cer3)
        cleancer5.append(cer5)
        cleancer7.append(cer7)

    advaudio = os.listdir(advfile_path)
    len_advaudio = len(advaudio)
    advcer3 = []
    advcer5 = []
    advcer7 = []

    for file in tqdm(advaudio):
        if file.split(".")[-1] not in ["mp3", "flac", "wav"]:
            len_advaudio = len_advaudio - 1
            continue
        print("\n")
        print(advfile_path + file)
        args = Args(advfile_path + file, restore, 0)
        adv_true_y = classify(args)
        print("adv_true_y: {}".format(adv_true_y))

        args3 = Args(advfile_path + file,restore,0.3)
        clsf = classify(args3)
        adv_true_y3 = adv_true_y[:int(0.3 * len(adv_true_y))]
        cer3 = get_cer(clsf, adv_true_y3)
        print("adv_true_y0.3: {}".format(adv_true_y3))
        print("split 0.3 cer: {:.2f} trans: {}".format(cer3, clsf))


        args5 = Args(advfile_path + file,restore,0.5)
        clsf = classify(args5)
        adv_true_y5 = adv_true_y[:int(0.5 * len(adv_true_y))]
        cer5 = get_cer(clsf, adv_true_y5)
        print("adv_true_y0.5: {}".format(adv_true_y5))
        print("split 0.5 cer: {:.2f} trans: {}".format(cer5, clsf))


        args7 = Args(advfile_path + file,restore,0.7)
        clsf = classify(args7)
        adv_true_y7 = adv_true_y[:int(0.7 * len(adv_true_y))]
        cer7 = get_cer(clsf, adv_true_y7)
        print("adv_true_y0.7: {}".format(adv_true_y7))
        print("split 0.7 cer: {:.2f} trans: {}".format(cer7, clsf))

        advcer3.append(cer3)
        advcer5.append(cer5)
        advcer7.append(cer7)

    advlabel=np.ones(len_advaudio)
    cleanlabel=np.zeros(len_cleanaudio)
    print("clean cer0.3: ", cleancer3)
    print("clean cer0.5: ", cleancer5)
    print("clean cer0.7: ", cleancer7)

    print("adv cer0.3: ", advcer3)
    print("adv cer0.5: ", advcer5)
    print("adv cer0.7: ", advcer7)

    print("cer0.3: ",np.concatenate([advcer3, cleancer3]))
    print("cer0.5: ", np.concatenate([advcer5, cleancer5]))
    print("cer0.7: ", np.concatenate([advcer7, cleancer7]))
    print("sklearn auc:(0.3):", roc_auc_score(np.concatenate([advlabel,cleanlabel]),np.concatenate([advcer3,cleancer3])))
    print("sklearn auc:(0.5):", roc_auc_score(np.concatenate([advlabel,cleanlabel]),np.concatenate([advcer5,cleancer5])))
    print("sklearn auc:(0.7):", roc_auc_score(np.concatenate([advlabel,cleanlabel]),np.concatenate([advcer7,cleancer7])))

# cleanfile = "/mnt/data/audio_adversarial_examples_deepspeech/Data/wav/"
# cleanfile = "/mnt/data/audio_adversarial_examples_deepspeech/Data/from_librispeech/"
cleanfile = "/mnt/data/audio_adversarial_examples_deepspeech/Data/Datasets/librispeech/LibriSpeech_test_clean/test-clean/61/70968/"
advfile = "/mnt/data/audio_adversarial_examples_deepspeech/Data/Expriment_Data/cw_final/"
np.set_printoptions(precision=2, suppress=True)

cal_cerauc(cleanfile, advfile)

