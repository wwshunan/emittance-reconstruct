import struct
import numpy as np
from scipy.constants import c
import matplotlib.pyplot as plt
partran_file = "RFQ.dst"

def generate_new_dis(partran_dist, I, mass):
    freq = 162.5
    n = partran_dist.shape[0]
    m0c2 = mass
    f = open(partran_file, 'wb')
    data = struct.pack('<B', 125)
    f.write(data)
    data = struct.pack('<B', 100)
    f.write(data)
    data = struct.pack('<i', n)
    f.write(data)
    data = struct.pack('<d', I)
    f.write(data)
    data = struct.pack('<d', freq)
    f.write(data)
    data = struct.pack('<B', 125) 
    f.write(data)

    for i in range(n):
        data = struct.pack('<dddddd', partran_dist[i, 0], partran_dist[i, 1], partran_dist[i, 2], partran_dist[i, 3], partran_dist[i, 4], partran_dist[i, 5])
        f.write(data)
    data = struct.pack('<d', m0c2)
    f.write(data)

    f.close()

def readDis(fname):
    f = open(fname, 'rb')
    data = struct.unpack("<B", f.read(1))
    data = struct.unpack("<B", f.read(1))

    data = struct.unpack("<i", f.read(4) )
    n = int(data[0])

    data = struct.unpack("<d", f.read(8) )
    ib   = float(data[0])

    data  = struct.unpack("<d", f.read(8))
    fbase = float(data[0])

    data = struct.unpack("<B", f.read(1))

    # read particle state data 

    partran_dist = np.arange(6*n, dtype='float64').reshape(n, 6)

    for i in range(n) :
        data = f.read(48)
        data = struct.unpack("<dddddd", data )
        partran_dist[i, 0] =  data[0]     
        partran_dist[i, 1] =  data[1]     
        partran_dist[i, 2] =  data[2]     
        partran_dist[i, 3] =  data[3]     
        partran_dist[i, 4] =  data[4]     
        partran_dist[i, 5] =  data[5]     

    data = struct.unpack("<d", f.read(8))
    m0c2 = data[0]
    f.close()
    return partran_dist

def exitDis(fname):
    f = open(fname, 'rb')

    iChar1 = struct.unpack("<B", f.read(1))
    iChar2 = struct.unpack("<B", f.read(1))

    iNe = struct.unpack("<i", f.read(4))
    Ne = int(iNe[0])

    iNp = struct.unpack("<i", f.read(4))
    parNum = int(iNp[0])

    iIb = struct.unpack("<d", f.read(8))
    mA   = float(iIb[0])

    iMHz = struct.unpack("<d", f.read(8))
    MHz   = float(iMHz[0])

    iMc2MEV= struct.unpack("<d", f.read(8))
    mc2MeV=float(iMc2MEV[0])


    for j in range(2):
        data = struct.unpack("<B", f.read(1))
        data = struct.unpack("<i", f.read(4))
        iZgen = struct.unpack("<d", f.read(8))
        data = struct.unpack("<d", f.read(8))
        data = struct.unpack("<d", f.read(8))
        if j == 1:
            partran_dist = np.arange(6*parNum, dtype='float64').reshape(parNum, 6)
            for i in range(parNum):
                partran_dist[i, 0] = float(struct.unpack("<f", f.read(4))[0])
                partran_dist[i, 1] = float(struct.unpack("<f", f.read(4))[0])
                partran_dist[i, 2] = float(struct.unpack("<f", f.read(4))[0])
                partran_dist[i, 3] = float(struct.unpack("<f", f.read(4))[0])
                partran_dist[i, 4] = float(struct.unpack("<f", f.read(4))[0])
                partran_dist[i, 5] = float(struct.unpack("<f", f.read(4))[0])
                iLoss = struct.unpack("<f", f.read(4))
        else:
            f.read(4 * 7 * parNum)
    f.close()
    return partran_dist
