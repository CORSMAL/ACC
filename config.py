# Default paths
#--------------------------------------------------------------------------------
# Authors: 
#   - Santiago Donaher
#   - Alessio Xompero: a.xompero@qmul.ac.uk
#
# MIT License

# Copyright (c) 2021 CORSMAL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#--------------------------------------------------------------------------------

# 2020
DATASET_PATH = 'D:/CCM_dataset/'

# 2021
NDATASET_PATH      = 'D:/CCM_dataset/CCM_2021/'
NANNOTAT_PATH_JSON = 'D:/CCM_dataset/CCM_2021/ccm_train_annotation.json'
NANNOTAT_PATH_CSV  = 'D:/CCM_dataset/CCM_2021/ccm_train_annotation.csv'

OUTPUT_PATH  = 'outputs'

# Audio features and general parameters
SAMPL_RATE = 22050
AUDIO_LEN  = 10
IMG_SIZE   = 96

# MODEL PARAMETERS
LEARNING_RATE = 0.001
EPOCHS        = 10
BATCH_SIZE    = 16

# DEFAULT TRAINED MODEL NAMES
TR_ACC     = 'acc'

# ResNet (for direct classifier)
RESNET_DEPTH = 14

# Best Chunk parameters
LOW_THRESH  = 0.001
MID_THRESH  = 0.01
HIGH_THRESH = 0.03
LOUD_AUDIO  = 1500
SOFT_AUDIO  = 750

# Temporal parser
CAT2FIL = {
    0: 'empty',
    1: 'pasta_50',
    2: 'pasta_90',
    3: 'rice_50',
    4: 'rice_90',
    5: 'water_50',
    6: 'water_90'
}

FIL2CAT = {
    'empty':    0,
    'pasta_50': 1,
    'pasta_90': 2,
    'rice_50':  3,
    'rice_90':  4,
    'water_50': 5,
    'water_90': 6
}

# Samples for demo
DEMO_SAMPLES = [
        's1_fi0_fu0_b1_l0_audio.wav',
        's0_fi1_fu1_b0_l1_audio.wav',
        's0_fi1_fu2_b0_l0_audio.wav',
        's0_fi2_fu1_b0_l1_audio.wav',
        's1_fi2_fu2_b0_l1_audio.wav',
        's0_fi3_fu1_b0_l1_audio.wav',
        's1_fi3_fu2_b1_l0_audio.wav']