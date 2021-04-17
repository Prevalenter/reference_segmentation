import sys
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.append('../')
from utils import split, para_io
if __name__ == '__main__':
    positions = [i for i in range(1,17)]
    np.random.shuffle(positions)
    print(positions)


    for i in range(4):
        test = positions[4*i:4*(i+1)]
        train = [i for i in positions if i not in test]
        
        train, valid, _, _ =\
                train_test_split(train, train, test_size=0.2, random_state=i)
        print(test, train, valid)
    #     test, valid, _, _ =\
    #             train_test_split(test, test, test_size=0.33, random_state=i)

    #     print(train, test, valid)

        root = '../data/Drosophila_256_False/para%d.yaml'%i
        para = {}
        para['train'] = train
        para['test'] = test
        para['valid'] = valid

        para_io.para_write(para, root)

