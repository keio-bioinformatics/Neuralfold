import numpy as np
from .. import Config

def base_represent(base):
    if Config.gpu == True:
        print("gpu")
        # xp = cuda.cupy
        # if base in ['A' ,'a']:
        #     return Variable(xp.array([[1,0,0,0]] , dtype=np.float32))
        # elif base in ['U' ,'u']:
        #     return Variable(xp.array([[0,1,0,0]] , dtype=np.float32))
        # elif base in ['G' ,'g']:
        #     return Variable(xp.array([[0,0,1,0]] , dtype=np.float32))
        # elif base in ['C' ,'c']:
        #     return Variable(xp.array([[0,0,0,1]] , dtype=np.float32))
        # else:
        #     print("error")
        #     return Variable(xp.array([[0,0,0,0]] , dtype=np.float32))
    else:
        if base in ['A' ,'a']:
            return np.array([[1,0,0,0,0]] , dtype=np.float32)
        elif base in ['U' ,'u']:
            return np.array([[0,1,0,0,0]] , dtype=np.float32)
        elif base in ['G' ,'g']:
            return np.array([[0,0,1,0,0]] , dtype=np.float32)
        elif base in ['C' ,'c']:
            return np.array([[0,0,0,1,0]] , dtype=np.float32)
        else:
            # print(base)
            return np.array([[0,0,0,0,0]] , dtype=np.float32)

    # else:
    #     if base in ['A' ,'a']:
    #         return Variable(np.array([[1,0,0,0]] , dtype=np.float32))
    #     elif base in ['U' ,'u']:
    #         return Variable(np.array([[0,1,0,0]] , dtype=np.float32))
    #     elif base in ['G' ,'g']:
    #         return Variable(np.array([[0,0,1,0]] , dtype=np.float32))
    #     elif base in ['C' ,'c']:
    #         return Variable(np.array([[0,0,0,1]] , dtype=np.float32))
    #     else:
    #         # print(base)
    #         return Variable(np.array([[0,0,0,0]] , dtype=np.float32))
