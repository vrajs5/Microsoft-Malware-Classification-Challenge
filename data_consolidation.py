# -*- coding: utf-8 -*-
"""
Created on Mon Feb 09 16:50:14 2015

@author: VishnuC
@email: vrajs5@gmail.com
Beating the benchmark for Microsoft Malware Classification Challenge (BIG 2015)
"""
from multiprocessing import Pool
import os
import gzip
from csv import writer
import six

read_mode, write_mode = ('r','w') if six.PY2 else ('rt','wt')

path = '' #Path to project 
os.chdir(path)

if six.PY2:
    from itertools import izip
    zp = izip
else:
    zp = zip

# Give path to gzip of asm files
paths = ['train','test']

 
def consolidate(path):
    ''' A consolidation of given train or test files

        This function reads each asm files (stored in gzip format)
        and prepare summary. asm gzip files are stored in train_gz 
        and test_gz locating.
    '''
    
    s_path = path + '_gz/'
    Files = os.listdir(s_path)
    byteFiles = [i for i in Files if '.bytes.gz' in i]
    consolidatedFile = path + '_consolidation.gz'
    
    with gzip.open(consolidatedFile, write_mode) as f:
        # Preparing header part
        fw = writer(f)
        colnames = ['filename', 'no_que_mark']
        colnames += ['TB_'+hex(i)[2:] for i in range(16**2)]
        fw.writerow(colnames)
        
        # Creating row set
        consolidation = []
        for t, fname in enumerate(byteFiles):
            f = gzip.open(s_path+fname, read_mode)
            twoByte = [0]*16**2
            no_que_mark = 0
            for row in f:
                codes = row[:-2].split()[1:]
                # Finding number of times ?? appears
                no_que_mark += codes.count('??')
                
                # Conversion of code to to two byte
                twoByteCode = [int(i,16) for i in codes if i != '??']
                                                    
                # Frequency calculation of two byte codes
                for i in twoByteCode:
                    twoByte[i] += 1
                
            # Row added
            consolidation.append([fname[:fname.find('.bytes.gz')], no_que_mark] \
                                    + twoByte)
                                    
            # Writing rows after every 100 files processed
            if (t+1)%100==0:
                print(t+1, 'files loaded for ', path)
                fw.writerows(consolidation)
                consolidation = []
                
        # Writing remaining files
        if len(consolidation)>0:
            fw.writerows(consolidation)
            consolidation = []
    
    del Files, byteFiles, colnames, s_path, consolidation, f, fw, \
        twoByte, twoByteCode, consolidatedFile

if __name__ == '__main__':
    p = Pool(2)
    p.map(consolidate, paths)
