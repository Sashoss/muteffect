
import tensorflow as tf
import numpy as np
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

############################## Feature functions #########################

"""Kidera factors"""
data_MatrixKF = {
    "A": [-1.56, -1.67, -0.97, -0.27, -0.93, -0.78, -0.2, -0.08, 0.21, -0.48],
    "R": [0.22, 1.27, 1.37, 1.87, -1.7, 0.46, 0.92, -0.39, 0.23, 0.93],
    "N": [1.14, -0.07, -0.12, 0.81, 0.18, 0.37, -0.09, 1.23, 1.1, -1.73],
    "D": [0.58, -0.22, -1.58, 0.81, -0.92, 0.15, -1.52, 0.47, 0.76, 0.7],
    "C": [0.12, -0.89, 0.45, -1.05, -0.71, 2.41, 1.52, -0.69, 1.13, 1.1],
    "Q": [-0.47, 0.24, 0.07, 1.1, 1.1, 0.59, 0.84, -0.71, -0.03, -2.33],
    "E": [-1.45, 0.19, -1.61, 1.17, -1.31, 0.4, 0.04, 0.38, -0.35, -0.12],
    "G": [1.46, -1.96, -0.23, -0.16, 0.1, -0.11, 1.32, 2.36, -1.66, 0.46],
    "H": [-0.41, 0.52, -0.28, 0.28, 1.61, 1.01, -1.85, 0.47, 1.13, 1.63],
    "I": [-0.73, -0.16, 1.79, -0.77, -0.54, 0.03, -0.83, 0.51, 0.66, -1.78],
    "L": [-1.04, 0, -0.24, -1.1, -0.55, -2.05, 0.96, -0.76, 0.45, 0.93],
    "K": [-0.34, 0.82, -0.23, 1.7, 1.54, -1.62, 1.15, -0.08, -0.48, 0.6],
    "M": [-1.4, 0.18, -0.42, -0.73, 2, 1.52, 0.26, 0.11, -1.27, 0.27],
    "F": [-0.21, 0.98, -0.36, -1.43, 0.22, -0.81, 0.67, 1.1, 1.71, -0.44],
    "P": [2.06, -0.33, -1.15, -0.75, 0.88, -0.45, 0.3, -2.3, 0.74, -0.28],
    "S": [0.81, -1.08, 0.16, 0.42, -0.21, -0.43, -1.89, -1.15, -0.97, -0.23],
    "T": [0.26, -0.7, 1.21, 0.63, -0.1, 0.21, 0.24, -1.15, -0.56, 0.19],
    "W": [0.3, 2.1, -0.72, -1.57, -1.16, 0.57, -0.48, -0.4, -2.3, -0.6],
    "Y": [1.38, 1.48, 0.8, -0.56, 0, -0.68, -0.31, 1.03, -0.05, 0.53],
    "V": [-0.74, -0.71, 2.04, -0.4, 0.5, -0.81, -1.07, 0.06, -0.46, 0.65]
}

"""Grantham Scores"""
Grantham_dict = {
                'A': {'A': 0.0, 'R': 112.0, 'N': 111.0, 'D': 126.0, 'C': 195.0, 'Q': 91.0, 'E': 107.0, 'G': 60.0, 'H': 86.0, 'I': 94.0, 'L': 96.0, 'K': 106.0, 'M': 84.0, 'F': 113.0, 'P': 27.0, 'S': 99.0, 'T': 58.0, 'W': 148.0, 'Y': 112.0, 'V': 64.0},
                'R': {'A': 112.0, 'R': 0.0, 'N': 86.0, 'D': 96.0, 'C': 180.0, 'Q': 43.0, 'E': 54.0, 'G': 125.0, 'H': 29.0, 'I': 97.0, 'L': 102.0, 'K': 26.0, 'M': 91.0, 'F': 97.0, 'P': 103.0, 'S': 110.0, 'T': 71.0, 'W': 101.0, 'Y': 77.0, 'V': 96.0},
                'N': {'A': 111.0, 'R': 86.0, 'N': 0.0, 'D': 23.0, 'C': 139.0, 'Q': 46.0, 'E': 42.0, 'G': 80.0, 'H': 68.0, 'I': 149.0, 'L': 153.0, 'K': 94.0, 'M': 142.0, 'F': 158.0, 'P': 91.0, 'S': 46.0, 'T': 65.0, 'W': 174.0, 'Y': 143.0, 'V': 133.0},
                'D': {'A': 126.0, 'R': 96.0, 'N': 23.0, 'D': 0.0, 'C': 154.0, 'Q': 61.0, 'E': 45.0, 'G': 94.0, 'H': 81.0, 'I': 168.0, 'L': 172.0, 'K': 101.0, 'M': 160.0, 'F': 177.0, 'P': 108.0, 'S': 65.0, 'T': 85.0, 'W': 181.0, 'Y': 160.0, 'V': 152.0},
                'C': {'A': 195.0, 'R': 180.0, 'N': 139.0, 'D': 154.0, 'C': 0.0, 'Q': 154.0, 'E': 170.0, 'G': 159.0, 'H': 174.0, 'I': 198.0, 'L': 198.0, 'K': 202.0, 'M': 196.0, 'F': 205.0, 'P': 169.0, 'S': 112.0, 'T': 149.0, 'W': 215.0, 'Y': 194.0, 'V': 192.0},
                'Q': {'A': 91.0, 'R': 43.0, 'N': 46.0, 'D': 61.0, 'C': 154.0, 'Q': 0.0, 'E': 29.0, 'G': 87.0, 'H': 24.0, 'I': 109.0, 'L': 113.0, 'K': 53.0, 'M': 101.0, 'F': 116.0, 'P': 76.0, 'S': 68.0, 'T': 42.0, 'W': 130.0, 'Y': 99.0, 'V': 96.0},
                'E': {'A': 107.0, 'R': 54.0, 'N': 42.0, 'D': 45.0, 'C': 170.0, 'Q': 29.0, 'E': 0.0, 'G': 98.0, 'H': 40.0, 'I': 134.0, 'L': 138.0, 'K': 56.0, 'M': 126.0, 'F': 140.0, 'P': 93.0, 'S': 80.0, 'T': 65.0, 'W': 152.0, 'Y': 122.0, 'V': 121.0},
                'G': {'A': 60.0, 'R': 125.0, 'N': 80.0, 'D': 94.0, 'C': 159.0, 'Q': 87.0, 'E': 98.0, 'G': 0.0, 'H': 98.0, 'I': 135.0, 'L': 138.0, 'K': 127.0, 'M': 127.0, 'F': 153.0, 'P': 42.0, 'S': 56.0, 'T': 59.0, 'W': 184.0, 'Y': 147.0, 'V': 109.0},
                'H': {'A': 86.0, 'R': 29.0, 'N': 68.0, 'D': 81.0, 'C': 174.0, 'Q': 24.0, 'E': 40.0, 'G': 98.0, 'H': 0.0, 'I': 94.0, 'L': 99.0, 'K': 32.0, 'M': 87.0, 'F': 100.0, 'P': 77.0, 'S': 89.0, 'T': 47.0, 'W': 115.0, 'Y': 83.0, 'V': 84.0},
                'I': {'A': 94.0, 'R': 97.0, 'N': 149.0, 'D': 168.0, 'C': 198.0, 'Q': 109.0, 'E': 134.0, 'G': 135.0, 'H': 94.0, 'I': 0.0, 'L': 5.0, 'K': 102.0, 'M': 10.0, 'F': 21.0, 'P': 95.0, 'S': 142.0, 'T': 89.0, 'W': 61.0, 'Y': 33.0, 'V': 29.0},
                'L': {'A': 96.0, 'R': 102.0, 'N': 153.0, 'D': 172.0, 'C': 198.0, 'Q': 113.0, 'E': 138.0, 'G': 138.0, 'H': 99.0, 'I': 5.0, 'L': 0.0, 'K': 107.0, 'M': 15.0, 'F': 22.0, 'P': 98.0, 'S': 145.0, 'T': 92.0, 'W': 61.0, 'Y': 36.0, 'V': 32.0},
                'K': {'A': 106.0, 'R': 26.0, 'N': 94.0, 'D': 101.0, 'C': 202.0, 'Q': 53.0, 'E': 56.0, 'G': 127.0, 'H': 32.0, 'I': 102.0, 'L': 107.0, 'K': 0.0, 'M': 95.0, 'F': 102.0, 'P': 103.0, 'S': 121.0, 'T': 78.0, 'W': 110.0, 'Y': 85.0, 'V': 97.0},
                'M': {'A': 84.0, 'R': 91.0, 'N': 142.0, 'D': 160.0, 'C': 196.0, 'Q': 101.0, 'E': 126.0, 'G': 127.0, 'H': 87.0, 'I': 10.0, 'L': 15.0, 'K': 95.0, 'M': 0.0, 'F': 28.0, 'P': 87.0, 'S': 135.0, 'T': 81.0, 'W': 67.0, 'Y': 36.0, 'V': 21.0},
                'F': {'A': 113.0, 'R': 97.0, 'N': 158.0, 'D': 177.0, 'C': 205.0, 'Q': 116.0, 'E': 140.0, 'G': 153.0, 'H': 100.0, 'I': 21.0, 'L': 22.0, 'K': 102.0, 'M': 28.0, 'F': 0.0, 'P': 114.0, 'S': 155.0, 'T': 103.0, 'W': 40.0, 'Y': 22.0, 'V': 50.0},
                'P': {'A': 27.0, 'R': 103.0, 'N': 91.0, 'D': 108.0, 'C': 169.0, 'Q': 76.0, 'E': 93.0, 'G': 42.0, 'H': 77.0, 'I': 95.0, 'L': 98.0, 'K': 103.0, 'M': 87.0, 'F': 114.0, 'P': 0.0, 'S': 74.0, 'T': 38.0, 'W': 147.0, 'Y': 110.0, 'V': 68.0},
                'S': {'A': 99.0, 'R': 110.0, 'N': 46.0, 'D': 65.0, 'C': 112.0, 'Q': 68.0, 'E': 80.0, 'G': 56.0, 'H': 89.0, 'I': 142.0, 'L': 145.0, 'K': 121.0, 'M': 135.0, 'F': 155.0, 'P': 74.0, 'S': 0.0, 'T': 58.0, 'W': 177.0, 'Y': 144.0, 'V': 124.0},
                'T': {'A': 58.0, 'R': 71.0, 'N': 65.0, 'D': 85.0, 'C': 149.0, 'Q': 42.0, 'E': 65.0, 'G': 59.0, 'H': 47.0, 'I': 89.0, 'L': 92.0, 'K': 78.0, 'M': 81.0, 'F': 103.0, 'P': 38.0, 'S': 58.0, 'T': 0.0, 'W': 128.0, 'Y': 92.0, 'V': 69.0},
                'W': {'A': 148.0, 'R': 101.0, 'N': 174.0, 'D': 181.0, 'C': 215.0, 'Q': 130.0, 'E': 152.0, 'G': 184.0, 'H': 115.0, 'I': 61.0, 'L': 61.0, 'K': 110.0, 'M': 67.0, 'F': 40.0, 'P': 147.0, 'S': 177.0, 'T': 128.0, 'W': 0.0, 'Y': 37.0, 'V': 88.0},
                'Y': {'A': 112.0, 'R': 77.0, 'N': 143.0, 'D': 160.0, 'C': 194.0, 'Q': 99.0, 'E': 122.0, 'G': 147.0, 'H': 83.0, 'I': 33.0, 'L': 36.0, 'K': 85.0, 'M': 36.0, 'F': 22.0, 'P': 110.0, 'S': 144.0, 'T': 92.0, 'W': 37.0, 'Y': 0.0, 'V': 55.0},
                'V': {'A': 64.0, 'R': 96.0, 'N': 133.0, 'D': 152.0, 'C': 192.0, 'Q': 96.0, 'E': 121.0, 'G': 109.0, 'H': 84.0, 'I': 29.0, 'L': 32.0, 'K': 97.0, 'M': 21.0, 'F': 50.0, 'P': 68.0, 'S': 124.0, 'T': 69.0, 'W': 88.0, 'Y': 55.0, 'V': 0.0}
}

"""SeqStruc substitution scores"""
PROTSUB = {
                    "A": [4, -2, -2, -2, 1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, 1, 0, -3, -3,  0, -2, -1, 0, -4],
                    "R": [-2, 6, 0, -2, -4, 1, 0, -2, 1, -3, -2, 2, -1, -2, -2, 0, -1, -2, -2, -2, -1, 0, -1, -4],
                    "N": [-2, 0, 7, 1, -3, 2, -1, 0, 1, -3, -4, 0, -2, -3, -2, 1, 0, -4, -2, -3, 3, 0, -1, -4],
                    "D": [-2, -2, 1, 9, -3, 0, 3, -1, 1, -3, -4, -1, -3, -3, -1, 1, -2, -4, -3, -3, 4, 1, -1, -4],
                    "C": [1, -4, -3, -3, 14, -2, -3, -3, -3, -2, -1, -3, -1, -2, -3, -1, -1, -1, -2, -1, -3, -3, -2, -4],
                    "Q": [-1, 1, 2, 0, -2, 4, 1, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, 0, -2, 0, 3, -1, -4],
                    "E": [0, 0, -1, 3, -3, 1, 4, -3, 0, -4, -3, 0, -2, -4, -1, -1, -4, -3, -2, -2, 1, 4, -1, -4],
                    "G": [0, -2, 0, -1, -3, -2, -3, 9, -3, -4, -4, -2, -3, -4, -2, 0, -2, -2, -3, -3, -1, -2, -1, -4],
                    "H": [-1, 1, 1, 1, -3, 0, 0, -3, 9, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1, -4],
                    "I": [-1, -3, -3, -3, -2, -3, -4, -4, -3, 4, 3, -4, 1, 0, -3, -2, -1, -3, -1, 3, -3, -3, -1, -4],
                    "L": [-1, -2, -4, -4, -1, -2, -3, -4, -3, 3, 6, -3, 2, 0, -3, -3, -1, -2, -1, 3, -4, -3, -1, -4],
                    "K": [-1, 2, 0, -1, -3, 1, 0, -2, -1, -4, -3, 6, -2, -3, -1, 0, -1, -3, -2, -2, 0, 1, -1, -4],
                    "M": [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -2, 8, 1, -3, -1, -1, -3, 0, 1, -3, -1, -1, -4],
                    "F": [-1, -2, -3, -3, -2, -3, -4, -4, -1, 0, 0, -3, 1, 6, -4, -2, -2, 1, 3, 0, -3, -3, -1, -4],
                    "P": [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -3, -4, 8, -1, -1, -4, -3, -3, -2, -1, -2, -4],
                    "S": [1, 0, 1, 1, -1, 0, -1, 0, -1, -2, -3, 0, -1, -2, -1, 4, 2, -3, -1, -2, 0, 0, 0, -4],
                    "T": [0, -1, 0, -2, -1, -1, -4, -2, -2, -1, -1, -1, -1, -2, -1, 2, 7, -2, -2, 0, -1, -1, 0, -4],
                    "W": [-3, -2, -4, -4, -1, -2, -3, -2, -2, -3, -2, -3, -3, 1, -4, -3, -2, 13, 3, -3, -4, -3, -2, -4],
                    "Y": [-3, -2, -2, -3, -2, 0, -2, -3, 2, -1, -1, -2, 0, 3, -3, -1, -2, 3, 8, -1, -3, -2, -1, -4],
                    "V": [0, -2, -3, -3, -1, -2, -2, -3, -3, 3, 3, -2, 1, 0, -3, -2, 0, -3, -1, 4, -3, -2, -1, -4],
                    "B": [-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, 1, -1, -4],
                    "Z": [-1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4],
                    "X": [0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1, -4],
                    "*": [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1]
                    }
                    
def composition(seq):
    """
        Calculates frequency of all 20 amino acids in the input sequence (seq variable)
    """
    com = {'A':0,'R':0,'N':0,'D':0,'C':0,'E':0,'Q':0,'G':0,'H':0,'I':0,'L':0,'K':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}
    for i in seq:
        com[i] = com[i] + 1
    for key,val in com.items():
        com[key] = float(com[key])/len(seq)
    return list(com.values())


################################# Generate features for NN ##################################

def load_data():
    """Create features
    """
    
    store = dict()
    file=open("thermomutdb.tsv",encoding="utf8") # Read thermomut database
    for lines in file:
        if "MUTATION_uniprot" in lines:#to skip first line
            continue
        try:
            lobj = lines.strip().split("\t")#slipt lines with tabs  and removes new line character
            unip_id = lobj[4]# extracting specific columns identified by the index number ie protein uniprot id
            mut_id = lobj[7]#ie mutation eg:'E49M'
            mut_position = int(mut_id[1:-1])#int because readingfrom a file gives string #mut_position is the position of mutation ie 49
            if "," in mut_id:#if there are more than one mutation separated by commas skip line eg:"V64L,M72Q,G91V"
                continue
            if float(lobj[15])< 0.5 and float(lobj[15])> -0.5: # Remove approximately neutral mutations with ddg values between -0.5 and 0.5
                continue
                
            #ddg_sign = float(lobj[15])
            if float(lobj[15]) < 0:#check if ddg value is negative ie unstable
                ddg_sign = 0
            else:
                ddg_sign = 1

            PROTSUB_KEYS = list(PROTSUB.keys())
            mut_index = PROTSUB_KEYS.index(mut_id[-1])#index of mutant amino acid in protsub keys.
            #following is  nested dictionary
            store[unip_id + "_" + mut_id] = {"ph":float(lobj[12]),#experimental ph set for mutation induced change in free energy calculation
                                             "temp":float(lobj[11]),#experimental temperature set for mutation induced change in free energy calculation
                                             "ddg":float(lobj[15]),# mutation induced change in free energy value
                                             "seq":lobj[-1],#amino acid sequence of the protein
                                             "nativeKF":data_MatrixKF[mut_id[0]],#10 kidera factors of native amino acid
                                             "mutantKF":data_MatrixKF[mut_id[-1]],#10 kidear factors for mutnat amino
                                             "ddg_sign":ddg_sign,
                                             "KF_N-1" :data_MatrixKF[lobj[-1][mut_position-1]],#extracted seq then extracted the mutation positon substracted by 1.
                                            #lobj[-1] gives a sequence ,the final output for the brackets is a amino acid.KF_n-1 gives the kidera factors of amino acid at position n-1 which is [a position before the mutation position ].
                                             "KF_N+1" :data_MatrixKF[lobj[-1][mut_position+1]],
                                             "KF_N+2" :data_MatrixKF[lobj[-1][mut_position+2]],
                                             "KF_N-2" :data_MatrixKF[lobj[-1][mut_position-2]],
                                             "KF_N+3" :data_MatrixKF[lobj[-1][mut_position+3]],
                                             "KF_N-3" :data_MatrixKF[lobj[-1][mut_position-3]],
                                             "G-score":Grantham_dict[mut_id[0]][mut_id[-1]],#this is a grantham score from nested dictionary
                                             "protsub-score":PROTSUB[mut_id[0]][mut_index],
                                             "seq_com":composition(lobj[-1]),
                                             "seq_com_partition":composition(lobj[-1][mut_position-10:mut_position+10]),
                                             "seq_len":len(lobj[-1])
                                             }

        except:
            continue

    data_all = list()
    label_all = list()
    
    #Create data array
    for key,val in store.items():
        
        label_all.append(val["ddg_sign"])
        obj = [[val["ph"]]+[val["temp"]]+\
              val["nativeKF"]+val["mutantKF"]+\
              val["KF_N+1"]+val["KF_N-1"]+\
              val["KF_N+2"]+val["KF_N-2"]+\
              val["KF_N+3"]+val["KF_N-3"]+\
              [val["G-score"]]+[val["protsub-score"]]+\
              val["seq_com_partition"]+val["seq_com"]+\
              [val["seq_len"]]]
        
        data_all.append(obj)
        
    data_all = np.array(data_all)
    data_all = data_all.reshape(data_all.shape[0], data_all.shape[-1])
    data_df = pd.DataFrame(data_all, columns = ['pH','Temperature','NatK1', 'NatK2', 'NatK3', 'NatK4', 'NatK5', 'NatK6', 'NatK7', 'NatK8', 'NatK9', 'NatK10', 'mutK1', 'mutK2', 'mutK3', 'mutK4', 'mutK5', 'mutK6', 'mutK7', 'mutK8', 'mutK9', 'mutK10', 'N+1K1', 'N+1K2', 'N+1K3', 'N+1K4', 'N+1K5', 'N+1K6', 'N+1K7', 'N+1K8', 'N+1K9', 'N+1K10', 'N-1K1', 'N-1K2', 'N-1K3', 'N-1K4', 'N-1K5', 'N-1K6', 'N-1K7', 'N-1K8', 'N-1K9', 'N-1K10', 'N+2K1', 'N+2K2', 'N+2K3', 'N+2K4', 'N+2K5', 'N+2K6', 'N+2K7', 'N+2K8', 'N+2K9', 'N+2K10', 'N-2K1', 'N-2K2', 'N-2K3', 'N-2K4', 'N-2K5', 'N-2K6', 'N-2K7', 'N-2K8', 'N-2K9', 'N-2K10', 'N+3K1', 'N+3K2', 'N+3K3', 'N+3K4', 'N+3K5', 'N+3K6', 'N+3K7', 'N+3K8', 'N+3K9', 'N+3K10', 'N-3K1', 'N-3K2', 'N-3K3', 'N-3K4', 'N-3K5', 'N-3K6', 'N-3K7', 'N-3K8', 'N-3K9', 'N-3K10', 'G-score', 'seqstruc', 'partA', 'partR', 'partN', 'partD', 'partC', 'partE', 'partQ', 'partG', 'partH', 'partI', 'partL', 'partK', 'partM', 'partF', 'partP', 'partS', 'partT', 'partW','partY','partV', 'totA', 'totR', 'totN', 'totD', 'totC', 'totE', 'totQ', 'totG', 'totH', 'totI', 'totL', 'totK', 'totM', 'totF','totP','totS','totT','totW','totY','totV', 'seqlen'])
    
    get_correl(data_df)
        
    label_all = np.array(label_all)
    data_all = np.array(data_all)
    return data_all,label_all
    
def get_correl(data_df):
    corr_matrix = data_df.corr()
    plt.figure(figsize=(35, 30))
    sns.heatmap(corr_matrix, annot=False)
    plt.savefig("out1.png")
    data_df.hist(bins=10, figsize=(35, 30), color='b')
    plt.savefig("out2.png")
    return True


###################### Functions to scale, split and shuffle data #####################


def split_data_obj(X_inp, y_inp):
    """
    Split data into 80% training and 20% test
    """
    X_train, X_test, y_train, y_test = train_test_split(X_inp, y_inp, test_size= 0.20)
    return X_train, X_test, y_train, y_test

def scale_data(data):
    scaler=StandardScaler()
    X = scaler.fit_transform(data)
    return(X)
    
def shuffle_data(X,y):
    """
    Shuffles data randomly such that the indices are rearranged
    """
    indices = tf.range(start=0,limit=tf.shape(X)[0],dtype=tf.int32)
    idx=tf.random.shuffle(indices)
    X_shuffle = tf.gather(X,idx)
    y_shuffle = tf.gather(y,idx)
    return(X_shuffle,y_shuffle)
    
def train_xgb(X_train, X_test, y_train, y_test):
    """
    Code for gradient boost classifier training and testing
    """
    weights = compute_sample_weight(class_weight='balanced', y=y_train)
    xgb = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, max_depth=2, random_state=1).fit(X_train, y_train, sample_weight=weights)

    out = xgb.score(X_test, y_test)
    
    y_pred = xgb.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    mat = matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    print(out, f1, mat, auc) # Print model performance
    print(classification_report(y_test, y_pred))
    
    plt.figure(figsize=(35, 30))
    cm = confusion_matrix(y_pred, y_test)
    sns.heatmap(cm, annot = True,fmt = '.2f')
    plt.ylabel('Predicted class')
    plt.xlabel('Actual class')
    plt.savefig("out3.png")
    return out
    
def train_rf(X_train, X_test, y_train, y_test):
    
    """
    Random forest classifier training and testing
    """
    weight1 = float(list(y_train).count(0))/list(y_train).count(1)
    rf = RandomForestClassifier(max_features='auto', class_weight={1:weight1, 0:1.0}, random_state=1).fit(X_train, y_train)
    out = rf.score(X_test, y_test)
    print(rf.feature_importances_)
    y_pred = rf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    mat = matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    print(out, f1, mat, auc)
    return out
    

def run():
    """
    Execute main function
    """
    X,y = load_data()
    X,y = shuffle(X,y)
    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[-1])
    X = scale_data(X)
    X_train, X_test, y_train, y_test = split_data_obj(X, y)
    out = train_xgb(X_train, X_test, y_train, y_test)
    


    return True
    
run()

