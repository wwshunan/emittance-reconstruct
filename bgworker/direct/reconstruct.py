import os

latticeFile = 'MEBT_emittace.dat'
txt = '''
THIN_MATRIX 0 1 0 0 0 0 0 0 -1 0 0 0 0 0 0 1 0 0 0 0 0 0 -1 0 0 0 0 0 0 -1 0 0 0 0 0 0 1 
DRIFT 122.5 100 0 0 0
QUAD 80 0 100 0 0 0 0 0 0
DRIFT 77 100 0 0 0
DRIFT 190 100 0 0 0
SUPERPOSE_MAP 5.55112e-014 0 0 0 0 0
MAP_FIELD 90 300 0 100 {0} 0 0 0 quad1
SUPERPOSE_MAP 187 0 0 0 0 0
MAP_FIELD 90 300 0 100 -{1} 0 0 0 quad2
SUPERPOSE_MAP 379 0 0 0 0 0
MAP_FIELD 90 300 0 100 {2} 0 0 0 quad1
DRIFT 161.65 100 0 0 0
THIN_MATRIX 0 1 0 0 0 0 0 0 -1 0 0 0 0 0 0 1 0 0 0 0 0 0 -1 0 0 0 0 0 0 -1 0 0 0 0 0 0 1 
END

'''

class DirectReconstruction(object):
    def __init__(self, I, m, q, n, energy, emitx, alpx, betx, emity, alpy, bety,
                 emitz, alpz, betz, q1, q2, q3):
        self.I, self.n = I, n
        self.q1, self.q2, self.q3 = q3, q2, q1
        self.mass, self.charge, self.energy = m * 1e6, q, energy
        self.emitx, self.alpx, self.betx = emitx, alpx, betx
        self.emity, self.alpy, self.bety = emity, alpy, bety
        self.emitz, self.alpz, self.betz = emitz, alpz, betz

    def backTrack(self):
        lattice = txt.format(self.q1, self.q2, self.q3)
        with open('{}'.format(latticeFile), 'w') as f:
            f.write(lattice)
        twStr = './TraceWin MEBT.ini current1={} energy1={} mass1={} charge1={} etnx1={} alpx1={} betx1={} '
        twStr += 'etny1={} alpy1={} bety1={} eln1={} alpz1={} betz1={} nbr_part1={}'
        print(twStr.format(self.I, self.energy, self.mass, self.charge, self.emitx, self.alpx, self.betx,
                               self.emity, self.alpy, self.bety, self.emitz, self.alpz,
                               self.betz, self.n))

        os.system(twStr.format(self.I, self.energy, self.mass, self.charge, self.emitx, self.alpx, self.betx,
                               self.emity, self.alpy, self.bety, self.emitz, self.alpz,
                               self.betz, self.n))

if __name__ == '__main__':
    drst = DirectReconstruction(1, 938.27203, 1, 100000, 1.5210049, 0.25, -0.3, 0.4,
                                0.25, 0.3, 0.4, 0.2048176, 2.9512, 5.8157454, 95, 70, 35)
    drst.backTrack()
