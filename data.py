import os
import numpy as np
import torch

class MSDATA():
    ### initialize
    def __init__(self, type, fname=None):
        if type == 'load':
            self.type = 0
        elif type == 'save':
            if fname == None:
                print("Error::MSDATA save type need file name")
                return
            self.type = 1
            pth = os.path.abspath(os.getcwd()) + '/data'
            print('**Data save path:', pth)
            self.f = open(pth+'/'+fname, 'w')
        else:
            print("ERROR:MSDATA type='l' for loader / 's' for saver")
            return

    ### data loader
    def load(self, fpath, fname=None):
        if self.type == 1:
            print('ERROR::MSDATA is assigned as save type')

        files = []
        data = []
        max_dlen = 0
        min_dlen = np.infty
        cnt = 0
        if fname == None:
            fnames = os.listdir(fpath)
            fnames.sort()
            for f in fnames:
                if os.path.splitext(f)[1] == '.txt':
                    files.append(fpath+'/'+f)
        else:
            fnames = fname
            files.append(fpath+'/'+fname)
        
        dlens = []
        for file in files:
            cnt+= 1
            with open(file, 'r') as f:
                c = f.readlines()
                v = list(r.strip().split("\t") for r in c)
                data.append(np.array(v))
                dlen = len(v)
                dlens.append(dlen)
                if max_dlen < dlen:
                    max_dlen = dlen
                if min_dlen > dlen:
                    min_dlen = dlen
                    
        col = len(data[0][0])
        np_data = np.zeros((cnt, max_dlen, col))

        for i in range(cnt):
            dlen = len(data[i])
            np_data[i][0:dlen, :] = data[i][0:dlen, :]

        self.device = 'cpu' 
        if torch.cuda.is_available():
            self.device='cuda'
        self.data = np_data
        self.Tdata = torch.tensor(np_data, dtype=torch.float32, device=self.device)
        self.files = fnames
        self.maxlen = max_dlen
        self.minlen = min_dlen
        self.dlens = dlens
        self.nfiles = cnt
            
    ### data saver
    def save(self, data):
        if self.type == 0:
            print('ERROR::MSDATA is assigned as load type')
            return
        if isinstance(data, str):
            self.f.write(data)
            return
        
        

if __name__ == '__main__':
    A = MSDATA('l')
    A.load('data/230804_ref_data',)
    