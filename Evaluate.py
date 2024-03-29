import numpy as np

class Evaluate:
    def __init__(self, predicted_structure_set, structure_set):
        self.predicted_structure_set = predicted_structure_set
        self.structure_set = structure_set

    def getscore(self):
        num_correct_pair = 0
        num_true_pair = 0
        num_predicted_pair = 0
        for predicted_structure, true_structure in zip(self.predicted_structure_set, self.structure_set):
            #print(predicted_structure,true_structure)
            num_predicted_pair += predicted_structure.shape[0]
            num_true_pair += true_structure.shape[0]
            for predicted_pair in predicted_structure:
                for true_pair in true_structure:
                    if (predicted_pair == true_pair).all():
                        #print(predicted_pair,true_pair)
                        num_correct_pair+=1
        #print(num_correct_pair,num_predicted_pair,num_true_pair)
        Sensitivity = num_correct_pair/num_true_pair
        PPV = num_correct_pair/num_predicted_pair
        try:
            F_value = 2 * (Sensitivity * PPV) / (Sensitivity + PPV)
        except ZeroDivisionError:
            F_value = 0
            print("kotti??")
            print("ZeroDivisionError!!")
        return Sensitivity,PPV,F_value
