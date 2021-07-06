import re
import numpy as np
import os

def readlineSplitFloat(fin):
    return [float(x) for x in fin.readline().split()]

def paserBundleFirstLine(fin):
    """
    first_line[0] = #Cameras
    first_line[1] = #Points
    """
    line = fin.readline()
    first_line = [int(x) for x in line.split()]
    return first_line


def paserBundleCamera(fin):
    """
    cf = focal length
    ck1 = distortion coeffs
    ck2 = distortion coeffs
    cR = 3*3 R matrix
    ct = 1*3 t array
    """
    cf, ck1, ck2 = [float(x) for x in fin.readline().split()]
    rr1 = [float(x) for x in fin.readline().split()]
    rr2 = [float(x) for x in fin.readline().split()]
    rr3 = [float(x) for x in fin.readline().split()]
    cR = np.asarray([rr1,rr2,rr3])
    t1 = [float(x) for x in fin.readline().split()]
    ct = np.asarray(t1)
    return cf, ck1, ck2, cR, ct
    
    
def paserBundlePoints(fin):
    """
    pPos = 1*3 postion
    pColor = 1*3 color
    pView = array, view list
    pViewL = loctions
    PViewN = number of view
    """
    pPos = np.asarray(readlineSplitFloat(fin))
    pColor = np.asarray(readlineSplitFloat(fin))
    
    pView = []
    pViewL = []
    pViewN = 0
    pKey = []
    
    line = fin.readline().split()
    pViewN = int(line[0])
    for idx in range(pViewN):
        pView.append(int(line[1 + idx*4]))
        pKey.append(int(line[2 + idx * 4]))
        pViewL.append((float(line[3+idx*4]), float(line[idx*4+4])))
    
    return pPos, pColor, pViewN, pView, pKey, pViewL 
    
    # FIXED name of function
def parserBundle(filename):
    assert filename != None, "The filename cannot be None"
    
    cRs = []
    cts = []
    cfks = []
    
    pPs = []
    pCs = []
    pVNs = []
    pVs = []
    pVLs = []
    pKs = []
    
    with open(filename, "r") as fin:
        fin.readline()
        nC,nP = paserBundleFirstLine(fin)
        for iter_c in range(nC):
            cf, ck1, ck2, cR, ct = paserBundleCamera(fin)
            cR = np.asarray(cR)
            cRs.append(cR)
            ct = np.matmul(-cR.T, ct.reshape(3))
            cts.append(ct)
            cfks.append((cf,ck1,ck2))
        
        for iter_p in range(nP):
            pPos, pColor, pViewN, pView, pKey, pViewL = paserBundlePoints(fin)
            pPs.append(pPos)
            pCs.append(pColor)
            pVNs.append(pViewN)
            pVs.append(pView)
            pVLs.append(pViewL)
            pKs.append(pKey)
            
    return nC, nP, cfks, cRs, cts, pPs, pCs, pVNs, pVs, pKs, pVLs 


def buildCovisableMatrix(nC, nP, pVNs, pVs):
    """
    mvis = [nC,nC], (i,j) is the number of points on both i j
    mnvis = [nC, nC] (i,j) is the number of points on i but not on j
    arrvis = [nC], # points on i
    cpl = [nC] list, the point id view on image
    """
    
    arrvis = [0 for x in range(nC)]
    cpl = [ [] for x in range(nC)]
    mvis = np.zeros([nC, nC])
    mnvis = np.zeros([nC, nC])
    
    for iter_p in range(nP):
        for c in pVs[iter_p]:
            arrvis[c] = arrvis[c] + 1
            cpl[c].append(iter_p)
            
    for i in range(nC):
        for j in range(nC):
            si = set(cpl[i])
            sj = set(cpl[j])
            mvis[i][j] = len(si.intersection(sj))
            mnvis[i][j] = len(si.difference(sj))
    return arrvis, cpl, mvis, mnvis

def parserFirstLine(fin):
    """
    %c %d %d %d
    pair numberOfMatch FrameId FrameId
    """
    line = fin.readline()
    if len(line) == 0:
        return None
    args = line.split()
    return [str(args[0]), int(args[1]), int(args[2]), int(args[3])]

def parserR(fin):
    """
    Read the rotation matrix from the file input
    """
    R = np.zeros([3,3])
    line1 = [float(x) for x in fin.readline().split()]
    line2 = [float(x) for x in fin.readline().split()]
    line3 = [float(x) for x in fin.readline().split()]
    R[0][0] = line1[0]
    R[0][1] = line1[1]
    R[0][2] = line1[2]
    R[1][0] = line2[0]
    R[1][1] = line2[1]
    R[1][2] = line2[2]
    R[2][0] = line3[0]
    R[2][1] = line3[1]
    R[2][2] = line3[2]
    return R

def parserT(fin):
    """
    Read Translation from the file input
    """
    t = np.zeros(3)
    line1 = [float(x) for x in fin.readline().split()]
    t[0] = line1[0]
    t[1] = line1[1]
    t[2] = line1[2]
    return t

def readAmbiEGWithMatch(filename):
    """
    Read the EGs file of the dataset.
    
    Return Value:
    pairs: the EG pair with image id
    R: the rotation matrix of EG pair
    t: the translation matrix of EG pair
    match_number: # of match features
    matches: the list of match features
    """
    pairs = []
    R = []
    t = []
    match_number = []
    matches = []
    
    assert filename != None, "Filename cannot be None"
    
    with open(filename, "r") as fin:
        while True:            
            first_args = parserFirstLine(fin)
            if first_args == None:
                break
            
            pairs.append((first_args[2], first_args[3]))
            match_number.append(first_args[1])
            R.append(parserR(fin))
            t.append(parserT(fin))
            matches.append([])
            for idx in range(first_args[1]):
                line = fin.readline()
                line = [float(x) for x in line.split()]
                line[0] = int(line[0])
                line[5] = int(line[5])
                matches[-1].append(line)

    return pairs, R, t, match_number, matches

def parserRandomPickMatches2Loc(matches, rand_pick_num):
    matches = numpy.random.choice(matches, rand_pick_num)
    pt1, pt2 = parserMatches2Loc(matches)
    return pt1, pt2

def parserMatches2Loc(match):
    pt1 = [[line[3], line[4]] for line in match]
    pt2 = [[line[8], line[9]] for line in match]
    return pt1, pt2

if __name__ == "__main__":
    base_dir = '/mnt/Tango/pg/Ambi/cup'
    file_name = "EGs.txt"
    filename = os.path.join(base_dir, file_name)
    pairs, R, t, match_number, matches = readAmbiEGWithMatch(filename)