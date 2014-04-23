#!/usr/bin/env python
import process_wav as pw
from svmutil import *
import numpy as np

if __name__ == "__main__":
    m = svm_load_model('svm.model')
    all_features1 = pw.gen_features_from_wave_file("my_grandfather-large_room.wav")
    all_features2 = pw.gen_features_from_wave_file("my_grandfather-small_room.wav")
     
    len1 = all_features1.shape[0]
    len2 = all_features2.shape[0]
    all = np.squeeze(np.concatenate((all_features1, all_features2)))
    label1 = np.ones((len1, 1)) 
    label2 = -1*np.ones((len2, 1))
    labels = np.squeeze(np.concatenate((label1, label2)))

    p_label, p_acc, p_val = svm_predict(labels.tolist(), all.tolist(), m)
    print p_label[0:len1-1]
    print p_label[len1:-1]
    print p_val[0:len1-1]
    print p_val[len1:-1]
