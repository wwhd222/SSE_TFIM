import numpy as np
import random

class Configuration:
    def __init__(self):
        self.lx = 0
        self.ly = 0
        self.nn = 0
        self.nb = 0
        self.nh = 0
        self.mm = 0
        self.beta = 0.0
        self.aprob = 0.0
        self.dprob = 0.0
        self.spin = None
        self.bsites = None
        self.opstring = None
        self.frstspinop = None
        self.lastspinop = None
        self.vertexlist = None

class MeasurementData:
    def __init__(self):
        self.enrg1 = 0.0
        self.enrg2 = 0.0
        self.amag1 = 0.0
        self.amag2 = 0.0
        self.asusc = 0.0
        self.stiff = 0.0
        self.ususc = 0.0
        self.data1 = np.zeros(7)
        self.data2 = np.zeros(7)

config = Configuration()
meas_data = MeasurementData()

def ran():
    return random.random()

def initran(w):
    random.seed(w)

def makelattice():
    config.nn = config.lx * config.ly
    config.nb = 2 * config.nn
    config.bsites = np.zeros((2, config.nb), dtype=int)

    for y1 in range(config.ly):
        for x1 in range(config.lx):
            s = 1 + x1 + y1 * config.lx
            x2 = (x1 + 1) % config.lx
            y2 = y1
            config.bsites[0, s-1] = s
            config.bsites[1, s-1] = 1 + x2 + y2 * config.lx
            x2 = x1
            y2 = (y1 + 1) % config.ly
            config.bsites[0, s+config.nn-1] = s
            config.bsites[1, s+config.nn-1] = 1 + x2 + y2 * config.lx

def initconfig():
    config.spin = np.array([2*int(2*ran())-1 for _ in range(config.nn)])
    config.mm = 20
    config.opstring = np.zeros(config.mm, dtype=int)
    config.nh = 0
    config.frstspinop = np.zeros(config.nn, dtype=int)
    config.lastspinop = np.zeros(config.nn, dtype=int)
    config.vertexlist = np.zeros(4*config.mm, dtype=int)

def adjustcutoff(step):
    mmnew = config.nh + config.nh // 3
    if mmnew <= config.mm:
        return

    stringcopy = config.opstring.copy()
    config.opstring = np.zeros(mmnew, dtype=int)
    config.opstring[:config.mm] = stringcopy
    config.mm = mmnew
    config.vertexlist = np.zeros(4*config.mm, dtype=int)

    print(f" Step: {step}  Cut-off L: {config.mm}")

def diagonalupdate():
    for i in range(config.mm):
        op = config.opstring[i]
        if op == 0:
            b = min(int(ran() * config.nb) + 1, config.nb) - 1
            if config.spin[config.bsites[0, b] - 1] != config.spin[config.bsites[1, b] - 1]:
                if config.aprob >= float(config.mm - config.nh) or config.aprob >= ran() * (config.mm - config.nh):
                    config.opstring[i] = 2 * (b + 1)
                    config.nh += 1
        elif op % 2 == 0:
            p = config.dprob * (config.mm - config.nh + 1)
            if p >= 1.0 or p >= ran():
                config.opstring[i] = 0
                config.nh -= 1
        else:
            b = op // 2 - 1
            config.spin[config.bsites[0, b] - 1] *= -1
            config.spin[config.bsites[1, b] - 1] *= -1

def loopupdate():
    config.frstspinop.fill(-1)
    config.lastspinop.fill(-1)

    for v0 in range(0, 4 * config.mm, 4):
        op = config.opstring[v0 // 4]
        if op != 0:
            b = op // 2 - 1
            s1 = config.bsites[0, b] - 1
            s2 = config.bsites[1, b] - 1
            v1 = config.lastspinop[s1]
            v2 = config.lastspinop[s2]
            if v1 != -1:
                config.vertexlist[v1] = v0
                config.vertexlist[v0] = v1
            else:
                config.frstspinop[s1] = v0
            if v2 != -1:
                config.vertexlist[v2] = v0 + 1
                config.vertexlist[v0 + 1] = v2
            else:
                config.frstspinop[s2] = v0 + 1
            config.lastspinop[s1] = v0 + 2
            config.lastspinop[s2] = v0 + 3
        else:
            config.vertexlist[v0:v0+4] = 0

    for s1 in range(config.nn):
        v1 = config.frstspinop[s1]
        if v1 != -1:
            v2 = config.lastspinop[s1]
            config.vertexlist[v2] = v1
            config.vertexlist[v1] = v2

    for v0 in range(0, 4 * config.mm, 2):
        if config.vertexlist[v0] < 1:
            continue
        v1 = v0
        if ran() < 0.5:
            while True:
                config.opstring[v1 // 4] ^= 1
                config.vertexlist[v1] = -1
                v2 = v1 ^ 1
                v1 = config.vertexlist[v2]
                config.vertexlist[v2] = -1
                if v1 == v0:
                    break
        else:
            while True:
                config.vertexlist[v1] = 0
                v2 = v1 ^ 1
                v1 = config.vertexlist[v2]
                config.vertexlist[v2] = 0
                if v1 == v0:
                    break

    for i in range(config.nn):
        if config.frstspinop[i] != -1:
            if config.vertexlist[config.frstspinop[i]] == -1:
                config.spin[i] *= -1
        elif ran() < 0.5:
            config.spin[i] *= -1

def measureobservables():
    am = sum(config.spin[i] * (-1)**(i % config.lx + i // config.lx) for i in range(config.nn)) // 2
    am1 = am2 = ax1 = 0.0
    jj = [0, 0]

    for i in range(config.mm):
        op = config.opstring[i]
        if op == 0:
            continue
        elif op % 2 == 1:
            b = op // 2 - 1
            s1 = config.bsites[0, b] - 1
            s2 = config.bsites[1, b] - 1
            config.spin[s1] *= -1
            config.spin[s2] *= -1
            jj[b // config.nn] += config.spin[s2]
            am += 2 * config.spin[s1] * (-1)**(s1 % config.lx + s1 // config.lx)
        ax1 += float(am)
        am1 += float(abs(am))
        am2 += float(am)**2

    if config.nh != 0:
        ax1 = (ax1**2 + am2) / (float(config.nh) * float(config.nh + 1))
        am1 /= config.nh
        am2 /= config.nh
    else:
        am1 = float(abs(am))
        am2 = float(am)**2
        ax1 = am2

    meas_data.enrg1 += float(config.nh)
    meas_data.enrg2 += float(config.nh)**2
    meas_data.amag1 += am1
    meas_data.amag2 += am2
    meas_data.asusc += ax1
    meas_data.stiff += 0.5 * (float(jj[0])**2 + float(jj[1])**2)
    meas_data.ususc += float(sum(config.spin) // 2)**2

def writeresults(msteps, bins):
    meas_data.enrg1 /= msteps
    meas_data.enrg2 /= msteps
    meas_data.amag1 /= msteps
    meas_data.amag2 /= msteps
    meas_data.asusc /= msteps
    meas_data.stiff /= msteps
    meas_data.ususc /= msteps

    meas_data.enrg2 = (meas_data.enrg2 - meas_data.enrg1 * (meas_data.enrg1 + 1.0)) / config.nn
    meas_data.enrg1 = meas_data.enrg1 / (config.beta * config.nn) - 0.5
    meas_data.amag1 /= config.nn
    meas_data.amag2 /= config.nn
    meas_data.asusc = config.beta * meas_data.asusc / config.nn
    meas_data.ususc = config.beta * meas_data.ususc / config.nn
    meas_data.stiff /= (config.beta * config.nn)

    meas_data.data1 += [meas_data.enrg1, meas_data.enrg2, meas_data.amag1, meas_data.amag2, meas_data.asusc, meas_data.stiff, meas_data.ususc]
    meas_data.data2 += np.array([meas_data.enrg1, meas_data.enrg2, meas_data.amag1, meas_data.amag2, meas_data.asusc, meas_data.stiff, meas_data.ususc])**2

    wdata1 = meas_data.data1 / bins
    wdata2 = np.sqrt(np.abs(meas_data.data2 / bins - wdata1**2) / bins)

    print(f" Cut-off L : {config.mm}")
    print(f" Number of bins completed : {bins}")
    print(" =========================================")
    print(f" -E/N       : {wdata1[0]:.8f} {wdata2[0]:.8f}")
    print(f"  C/N       : {wdata1[1]:.8f} {wdata2[1]:.8f}")
    print(f"  <|m|>     : {wdata1[2]:.8f} {wdata2[2]:.8f}")
    print(f"  S(pi,pi)  : {wdata1[3]:.8f} {wdata2[3]:.8f}")
    print(f"  X(pi,pi)  : {wdata1[4]:.8f} {wdata2[4]:.8f}")
    print(f"  rho_s     : {wdata1[5]:.8f} {wdata2[5]:.8f}")
    print(f"  X(0,0)    : {wdata1[6]:.8f} {wdata2[6]:.8f}")
    print(" =========================================")

    meas_data.enrg1 = meas_data.enrg2 = meas_data.amag1 = meas_data.amag2 = meas_data.asusc = meas_data.stiff = meas_data.ususc = 0.0

def main():
    #parameters
    config.lx = 4  # x-dimension
    config.ly = 1  # y-dimension
    config.beta = 10.0  # beta
    nbins = 100  # number of bins
    msteps = 1000  # Monte Carlo steps in each bin
    isteps = 1000  # balance step

    initran(1)
    makelattice()
    initconfig()

    config.aprob = 0.5 * config.beta * config.nb
    config.dprob = 1.0 / (0.5 * config.beta * config.nb)

    for i in range(1, isteps + 1):
        diagonalupdate()
        loopupdate()
        adjustcutoff(i)

    print("Finished equilibration, M =", config.mm)

    for j in range(1, nbins + 1):
        for i in range(msteps):
            diagonalupdate()
            loopupdate()
            measureobservables()
        writeresults(msteps, j)

if __name__ == "__main__":
    main()
