import numpy as np

def position_encoding_for_individual_residue(residue_index):
    d = 20 # collected from "sAMPpred-GAT" paper
    b = 10000 # collected from "Structural Learning of Proteins Using Graph Convolutional Neural Networks" paper

    tmp = []
    for i in range(d//2):
        tmp.append(residue_index / (b ** (2 * i / d)))
    tmp = np.array(tmp)

    pos_encoding = np.zeros(d)
    pos_encoding[0::2] = np.sin(tmp[:])
    pos_encoding[1::2] = np.cos(tmp[:])
    return pos_encoding