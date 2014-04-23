#!/usr/bin/env python
import process_wav as pw
from svmutil import *
import numpy as np

if __name__ == "__main__":
    all_features11 = pw.gen_features_from_wave_file("rainbow_passage-large_room.wav")
    all_features12 = pw.gen_features_from_wave_file("arthur_the_rat-large_room.wav")
    all_features1 = np.squeeze(np.concatenate((all_features11, all_features12)))
    all_features21 = pw.gen_features_from_wave_file("rainbow_passage-small_room.wav")
    all_features22 = pw.gen_features_from_wave_file("arthur_the_rat-small_room.wav")
    all_features2 = np.squeeze(np.concatenate((all_features21, all_features22)))
    print all_features1
    print all_features2

    len1 = all_features1.shape[0]
    len2 = all_features2.shape[0]
    all = np.squeeze(np.concatenate((all_features1, all_features2)))
    label1 = np.ones((len1, 1)) 
    label2 = -1*np.ones((len2, 1))
    labels = np.squeeze(np.concatenate((label1, label2)))
    print all.shape, labels.shape
    prob = svm_problem(labels.tolist(), all.tolist())
    param = svm_parameter('-c 4')
    m = svm_train(prob, param)

    p_label, p_acc, p_val = svm_predict(labels.tolist(), all.tolist(), m)
    svm_save_model('svm.model', m)
    
    all_features1 = pw.gen_features_from_wave_file("my_grandfather-large_room.wav")
    all_features2 = pw.gen_features_from_wave_file("my_grandfather-small_room.wav")
     
    len1 = all_features1.shape[0]
    len2 = all_features2.shape[0]
    all = np.squeeze(np.concatenate((all_features1, all_features2)))
    label1 = np.ones((len1, 1)) 
    label2 = -1*np.ones((len2, 1))
    labels = np.squeeze(np.concatenate((label1, label2)))

    p_label, p_acc, p_val = svm_predict(labels.tolist(), all.tolist(), m)
#    print p_label
#    print p_val
    # print m.get_sv_coef()

    #all_features1_test = pw.gen_features_from_wave_file("in1.wav")
    #all_features2_test = pw.gen_features_from_wave_file("in2.wav")
    #len1 = all_features1_test.shape[0]
    #len2 = all_features2_test.shape[0]
    #all = np.squeeze(np.concatenate((all_features1_test, all_features2_test)))
    #label1 = np.ones((len1, 1)) 
    #label2 = -1*np.ones((len2, 1))
    #labels = np.squeeze(np.concatenate((label1, label2)))
    #p_label, p_acc, p_val = svm_predict(labels.tolist(), all.tolist(), m)
