# Main collection of functions for TALLEM and base mappings.

# Standard packages:
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes as orth_proc
from scipy.linalg import null_space as null
from sklearn import manifold
import networkx as nx

# Attached Toolkits:
import Data_Gen as DG
import Data_Rep as DR

# If you have ripser, uncomment this line to use it!
import ripser 


# Manifold learning calls for local coords.
def LLEM(n_neighbors = 10,n_components = 3):
    return manifold.LocallyLinearEmbedding(n_neighbors = n_neighbors,
        n_components = n_components,method = 'modified')
def LTSA(n_neighbors = 10,n_components = 3):
    return manifold.LocallyLinearEmbedding(n_neighbors = n_neighbors,
        n_components = n_components,method = 'ltsa')
(MLM,MLM['iso'],MLM['lle'],MLM['lta']) = ({},manifold.Isomap,LLEM,LTSA)

### ---------- Minor Helper Functions ----------###

## Set functions on lists ##
def intersect(a, b):
    return list(set(a) & set(b))
def union(a, b):
    return list(set(a) | set(b))
def setdiff(a, b):
    return list(set(a)-set(b))
def unique(a):
    return list(set(a))

## misc ##
def tent_kernel(dist,pow = 1):
    if pow == 1:
        return ((1-dist) + abs(1-dist))/2
    else:
        return max(0,(1 - dist)**pow)
# forces a column matrix for dimension 1
def rep_pca(pts,vec,shift):
    X = DR.repeat_pca(pts,vec,shift)
    if len(X.shape) == 1:
        X = X.reshape(X.shape[0],1)
    return X
# maps KB pts in [-pi,pi] into R4.
# note that KB CANNOT be mapped into RP3 (rotation =/= reflect) see paper:
# "Nonembedness of the Klein Bottle in RP3 and the Lawson's conjecture"
def KB_RP3_embed(pts,r = 0.5, a = 1.5):
    X = []
    for [x,y] in pts:
        x1 = (r*np.cos(y)+a)*np.cos(x)
        x2 = (r*np.cos(y)+a)*np.sin(x)
        x3 = r*np.sin(y)*np.cos(x/2)
        x4 = r*np.sin(y)*np.sin(x/2)
        X.append([x1,x2,x3,x4])
    return np.array(X)
def Flatten_Torus(pts,per=1): # R2k down to Tk inverse of C*(cos,sin) Xform
    numpts = len(pts)
    k = int(pts.shape[1]/2)
    newpts = np.zeros([numpts,k])
    for j in range(k):
        angles = (np.arctan2(-pts[:,2*j+1],-pts[:,2*j])+np.pi)*per/(2*np.pi)
        newpts[:,j] = angles
    if k == 1: # flatten coords if they are circular.
        newpts = newpts[:,0]
    return newpts
def circ_mean(pts,per=1):
    numpts = pts.shape[0]
    m1 = np.mean(pts)
    stops = min([numpts,1000]) # max precision
    mvec = np.zeros(stops)
    
    for i in tqdm(range(stops)):
        mi = m1 + per*i/stops
        for j in range(numpts):
            mvec[i] = mvec[i] + peri_dist(mi,pts[j],per=per,extra=False)**2
    slot = np.argmin(mvec)
    mean = (m1 + per*slot/stops)%per
    
    if stops == numpts:
        var = mvec[slot]/(numpts-1)
        return (mean,var*per**2)
    
    # extra precision! (for very large datasets, assuming the mean is stable)
    m2 = m1 + per*int(slot*numpts/stops)/numpts
    
    minind = -5*int(1+numpts/stops)
    maxind = 5*int(1+numpts/stops)
    mvec = np.zeros(maxind-minind)
    for i in tqdm(range(minind,maxind)):
        mi = m2 + per*i/numpts
        for j in range(numpts):
            mvec[i] = mvec[i] + peri_dist(mi,pts[j],per=per,extra=False)**2
    slot = np.argmin(mvec)
    mean = (m1 + per*slot/stops)%per
    var = mvec[slot]/(numpts-1)
    return (mean,var*per**2)

## metrics ##
def proj_dist(pt1,pt2):
    return np.arccos(min(np.abs(np.sum(pt1*pt2)),1))
def eucl_dist(p1,p2,extra = False):
    dist = np.linalg.norm(p2-p1)
    if not extra:
        return dist
    elif extra == True:
        return (dist,dist)
    elif extra == 2:
        flips = np.zeros(len(p1))
        return (dist,flips)
def circ_dist(a0,b0,per=2*np.pi): # periodic distance
    diff = abs(a0-b0)    
    if diff > per:
        diff = diff - per*int(diff/per)
    if diff > per/2:
      diff = per - diff
    return diff
def Sk_dist(pt1,pt2): # distance on a sphere
    return np.arccos(max(min(np.sum(pt1*pt2),1),-1))
def KB_dist(pt1,pt2,extra = False): # distance on a [-pi,pi]^2 klein bottle
    (x1,y1) = pt1
    (x2,y2) = pt2
    d1 = np.linalg.norm(pt1-pt2)
    d2 = np.linalg.norm(np.array([x1-x2,2*np.pi-np.abs(y1-y2)]))
    d3 = np.linalg.norm(np.array([2*np.pi-np.abs(x1-x2),y1+y2]))
    d4 = np.linalg.norm(np.array([2*np.pi-np.abs(x1-x2),2*np.pi-np.abs(y1+y2)]))
    if not extra:
        return np.min([d1,d2,d3,d4])
    else:
        return (np.min([d1,d2,d3,d4]),d1)
# stereographic distance. "Extra" Used to find "local" graph on RPd.
def stereo_dist(p1,p2,extra = False):
    p1up = np.concatenate([p1,np.array([np.sqrt(1-np.sum(p1*p1))])])
    p2up = np.concatenate([p2,np.array([np.sqrt(1-np.sum(p2*p2))])])
    if not extra:
        return proj_dist(p1up,p2up)
    else:
        return (proj_dist(p1up,p2up),np.arccos(max(min(np.sum(p1up*p2up),1),-1)))
def peri_dist(p1,p2,per=1,extra=False):
    p1 = (p1%per)/per
    p2 = (p2%per)/per
    if not extra:
        return min(abs(p1-p2),1-abs(p1-p2))
    else:
        return (min(abs(p1-p2),1-abs(p1-p2)),abs(p1-p2))
def Tk_dist(p1,p2): # distance on torus Tk in R2k
    k = int(len(p1)/2)
    R = np.sqrt(k) # makes unit vectors!
    ds = np.zeros(k)
    for j in range(k):
        ds[j] = Sk_dist(p1[2*j:2*j+2],p2[2*j:2*j+2])
    return np.linalg.norm(ds)/R
def tor_dist(p1,p2,per=1,extra=False): #distance on (RmZ^k) #extra=2 for flips
    k = len(p1)
    ds = np.zeros(k) # distances
    ns = np.zeros(k) # naives
    flips = np.zeros(k)
    p1 = (p1%per)/per
    p2 = (p2%per)/per
    for j in range(k):
        (d1,d2) = (abs(p1[j]-p2[j]),1-abs(p1[j]-p2[j]))
        ds[j] = np.min([d1,d2])
        ns[j] = d1
        if ds[j] != ns[j]:
            flips[j] = 1
    if not extra:
        return np.linalg.norm(ds)
    elif extra == True:
        return (np.linalg.norm(ds),np.linalg.norm(ns))
    elif extra == 2: 
        return (np.linalg.norm(ds),flips)

# fiber-bundle metrics (sorting for minimal geodesic)
# (S1,S1,...,F); F = (-a,a); S1 = [0,ell]/{0~ell}
def TkB_dist(p1,p2,orient=[-1,-1],ls = [1,1]):
    dim = len(orient)+1
    dists = np.zeros(2**(dim-1))
    for index in range(2**(dim-1)):
        pow = 1
        dvec = np.zeros(dim)
        for slot in range(dim-1):
            powj = int(index/2**slot)%2
            pow = pow*orient[slot]**powj
            if powj == 0:
                dvec[slot] = np.abs(p1[slot]-p2[slot])
            else:
                dvec[slot] = ls[slot] - np.abs(p1[slot]-p2[slot])
        dvec[dim-1] = np.abs(p1[dim-1] - pow*p2[dim-1])
        dists[index] = np.linalg.norm(dvec)
    return np.min(dists)
# finds dists on a RPk bundle, treated as a "bundle over Sk"
def RPB_dist(p1,p2,orient=-1):
    d11 = Sk_dist(p1[:-1],p2[:-1])
    d12 = np.abs(p1[-1]-p2[-1])
    d21 = Sk_dist(p1[:-1],-p2[:-1])
    d22 = np.abs(p1[-1]+p2[-1])
    d1 = np.sqrt(d11**2+d12**2)
    d2 = np.sqrt(d21**2+d22**2)
    return min(d1,d2)
# finds dists on a StereoProjBundle by lifting to the sphere.
def SPB_dist(p1,p2,orient=-1):
    r1 = 2/(1+np.linalg.norm(p1[:-1])**2)
    r2 = 2/(1+np.linalg.norm(p2[:-1])**2)
    s1 = np.zeros(len(p1)+1)
    s2 = np.zeros(len(p2)+1)
    s1[:-2] = r1*p1[:-1]
    s2[:-2] = r2*p2[:-1]
    s1[-2] = r1-1
    s2[-2] = r2-1
    s1[-1] = p1[-1]
    s2[-1] = p2[-1]
    return RPB_dist(s1,s2,orient=orient)

## feature selectors (for projective and circular coords) ##
def variance_selector(vec,maxsize):
    maxsize = min(len(vec),maxsize)
    ordering = np.argsort(vec)
    newvec = vec[ordering]
    splitvars = np.zeros(maxsize+1)
    splitvars[0] = np.inf
    for i in range(1,maxsize+1):
        varlow = np.var(newvec[:-i])
        varhigh = np.var(newvec[-i:])
        splitvars[i] = varlow + varhigh
    

    if maxsize == len(vec): # (replace nan with inf)
        splitvars[maxsize] = np.inf
    
    numtop = np.argmin(splitvars)
    return ordering[-numtop:]
def big_gap_selector(vec,maxsize):
    maxsize = min(len(vec),maxsize)
    ordering = np.argsort(vec)
    newvec = vec[ordering]
    gaps = np.zeros(maxsize)
    for i in range(maxsize):
        gaps[i] = newvec[-(i+1)] - newvec[-(i+2)]
    numtop = np.argmax(gaps)
    return ordering[-(numtop+1):]
def trivial_selector(vec,maxsize):
    maxsize = min(len(vec),maxsize)
    ordering = np.argsort(vec)
    newvec = vec[ordering]
    return ordering[-maxsize:]

# subset selector
def relSS(cover,Lgp,indexes):
    SSs = {}
    for j in indexes:
        SSs[j] = np.nonzero(np.in1d(Lgp,cover[j]))[0]
    return SSs
# obtain relatable subset indices for (Xi cap Xj) within Xi and Xj.
# ie, X1_(SS1[k]) = X2_(SS2[k]) are the same point. 
def intSS(cover,i1,i2):
    SS1 = np.nonzero(np.in1d(cover[i1],intersect(cover[i1],cover[i2])))[0]
    SS2 = np.nonzero(np.in1d(cover[i2],intersect(cover[i1],cover[i2])))[0]
    return (SS1,SS2)
    
### ---------- simplex and complex functions ---------- ###

# simplex sort ([<] [=] or [>])
def simp_comp(simp1,simp2):
    dim = len(simp1)-1
    for spot in range(dim+1):
        if simp1[spot] < simp2[spot]:
            return 0
        elif simp2[spot] > simp1[spot]:
            return 1
    return -1

# binary search on ordered simp list:
def simp_search(simp,simplist,verbose = False):
    numsimps = len(simplist)
    low = 0
    high = numsimps-1
    mid = int((low+high)/2)
    while high - low > 2:
        val = simp_comp(simp,simplist[mid])
        if val == -1:
            return mid
        elif val == 0:
            high = mid
        elif val == 1:
            low = mid
        if verbose:
            print("("+str(low)+","+str(high)+")")
        mid = int((low+high)/2)
    for index in [low,mid,high]:
        val = simp_comp(simp,simplist[index])
        if val == -1:
            return index
    return -1

# define 0-coboundary and weighted/multi-D variants
def delta0(simps):
    (verts,edges) = (simps[0],simps[1])
    (NV,NE) = (len(verts),len(edges))
    delta = np.zeros([NE,NV])
    
    for (index,edge) in enumerate(edges):
        (v1,v2) = edge
        delta[index,[v1,v2]] = [-1,1]
    return delta
def delta0A(simps,w): # 1-d altered version
    (verts,edges) = (simps[0],simps[1])
    (NV,NE) = (len(verts),len(edges))
    delta = np.zeros([NE,NV])
    
    for (index,edge) in enumerate(edges):
        (v1,v2) = edge
        delta[index,v2] = w[v1,v2]
        delta[index,v1] = -1
    return delta
def delta0D(simps,orths): # multi-D altered version
    dim = orths[list(orths)[0]].shape[0]
    (verts,edges) = (simps[0],simps[1])
    (NV,NE) = (dim*len(verts),dim*len(edges)) # its a (dim x dim) block matrix
    delta = np.zeros([NE,NV])
    
    for (index,edge) in enumerate(edges):
        (v1,v2) = edge
        delta[dim*index:dim*(index+1),dim*v2:dim*(v2+1)] = orths[(v1,v2)].T
        delta[dim*index:dim*(index+1),dim*v1:dim*(v1+1)] = -np.eye(dim)
    return delta
### ---------- covers/nerve/PoU/simpcomp functions ---------- ###

# max-min point search. (Furthest Point Sampling)
def max_min(input,landmarks=np.zeros(0),target_rad=0,min_landmarks=20,
    distfcn = eucl_dist,verbose=0):
    
    numpts = len(input)
    ns = numpts #min(numpts,4*min_landmarks)
    subset = np.array(list(range(numpts)))
    
    if distfcn == 'precomputed':
        Dmat = input[:,:]
    else:
        Dmat = np.zeros([ns,ns])
        def DF(ii,jj):
            return distfcn(input[ii,:],input[jj,:])
    
    if landmarks.shape[0] == 0:
        landmarks = np.array([0])
    if not distfcn == 'precomputed':
        for slot in landmarks:
            Dmat[slot,:] = [DF(slot,jj) for jj in range(ns)]
        
    if target_rad == 0: # target radius agnostic; rely on min_landmarks
        for j in range(1,ns):
            target_rad = np.max([target_rad,DF(0,j)])
    
    while True:
        slotnext = np.argmax(np.min(Dmat[landmarks,:],axis=0))
        cover_rad = np.min(Dmat[landmarks,slotnext])
        if verbose and len(landmarks)%verbose == 0:
            print(str(len(landmarks)) + ":"+str(cover_rad))
        # immediately after cover rad is computed.
        if (landmarks.shape[0] >= min_landmarks) and (cover_rad <= target_rad):
            break
        landmarks = np.concatenate([landmarks,np.array([slotnext])])
        if not distfcn == 'precomputed':
            Dmat[slotnext,:] = [DF(slotnext,jj) for jj in range(ns)]
        
    return (landmarks,cover_rad)
    
# max-min sampler which also yields geodesic distances from points to landmarks.
def MM_geodesic(pts,nn=10,landmarks=np.zeros(0),NL=20,
    distfcn = eucl_dist,verbose=0):
    numpts = len(pts)
    preDmat = DR.KNN_Graph(pts,nn=nn,mode='distance')
    # downside: graph distance is expensive with a lot of points!
    G = nx.Graph(preDmat)
    
    Dmat = np.zeros([numpts,numpts])
    def DFvec(ii):
        ddict = nx.shortest_path_length(G,source=ii,weight='weight')
        dvec = np.zeros(numpts)
        for jj in range(numpts): # how to handle missing?
            dvec[jj] = ddict[jj]
        return dvec
    
    if landmarks.shape[0] == 0:
        landmarks = np.array([0])
    if not distfcn == 'precomputed':
        for slot in landmarks:
            Dmat[slot,:] = DFvec(slot)
    
    nl = len(landmarks)
    slotnext = np.argmax(np.min(Dmat[landmarks,:],axis=0))
    cover_rad = np.min(Dmat[landmarks,slotnext])

    print("Finding Geodesic Distances and Landmarks.")
    for index in tqdm(range(nl,NL)):
        landmarks = np.concatenate([landmarks,np.array([slotnext])])
        Dmat[slotnext,:] = DFvec(slotnext)
        slotnext = np.argmax(np.min(Dmat[landmarks,:],axis=0))
        cover_rad = np.min(Dmat[landmarks,slotnext])
        if verbose and len(landmarks)%verbose == 0:
            print(str(len(landmarks)) + ":"+str(cover_rad))
    
    # make symmetric, but leave out distances between pairs of non-landmarks.
    Dmat[:,landmarks] = Dmat[landmarks,:].T
    
    return (landmarks,cover_rad,Dmat)
    
# yields (cover,assign,dists) correspondence between landmarks and points.
def landmark_cover(input,landmarks,cover_rad,distfcn=eucl_dist):
    numpts = input.shape[0]
    if distfcn == 'precomputed':
        def DF(ii,jj):
            return input[ii,jj]
    else:
        def DF(ii,jj):
            return distfcn(input[ii,:],input[jj,:])
    
    # local information!
    cover = [[] for x in range(len(landmarks))]
    assign = [[] for x in range(numpts)]
    dists = [{} for x in range(numpts)]
    
    # find cover set residence and point landmark neighbors (transposes)
    for (index,ii) in tqdm(enumerate(landmarks)):
        for jj in range(numpts):
            dist = DF(ii,jj)
            if dist < cover_rad:
              cover[index] = cover[index]+[jj]
              assign[jj] = assign[jj]+[index]
              dists[jj][index] = dist
        cover[index] = np.array(cover[index])
    return (cover,assign,dists)
    
# yields (cover,assign,dists,cover_rad) for cover on the circle.
# may also yield reduced cover given a reduced THR. 
def circle_cover(numcov,THR,coords,offset = 1/2,THRR = None):
    if not THRR == None:
        cover2 = [[] for x in range(numcov)]
        assign2 = [[] for x in range(len(coords))]
        dists2 = [{} for x in range(len(coords))]
        cover_rad_2 = np.pi*THRR/numcov

    cover = [[] for x in range(numcov)]
    assign = [[] for x in range(len(coords))]
    dists = [{} for x in range(len(coords))]
    cover_rad = np.pi*THR/numcov
    
    for index in range(numcov):
        center = 2*np.pi*(index+offset)/numcov
        for (slot,angle) in enumerate(coords):
            dist = circ_dist(center,angle)
            if dist < cover_rad:
                cover[index] = cover[index]+[slot]
                assign[slot] = assign[slot]+[index]
                dists[slot][index] = dist
            if dist < cover_rad_2:
                cover2[index] = cover2[index]+[slot]
                assign2[slot] = assign2[slot]+[index]
                dists2[slot][index] = dist
        cover[index] = np.array(cover[index])
        cover2[index] = np.array(cover2[index])
    if THRR == None:
        return (cover,assign,dists,cover_rad)
    else:
        return ((cover,assign,dists,cover_rad),(cover2,assign2,dists2,cover_rad_2))

def point_nerve(cover,maxdim=2):
    # startup
    numcov = len(cover)
    simps = [[] for x in range(maxdim+1)]
    
    # accrue vertices
    for i in range(numcov):
        simps[0] = simps[0] + [[i]]
    
    # accrue higher order simps up to maxdim.
    for j in range(maxdim):
        for presimp in simps[j]:
            ik = presimp[0]
            precov = cover[ik]
            for ik in presimp[1:]:
                precov = intersect(precov,cover[ik])
            for vert in range(ik+1,numcov):
                cov = intersect(precov,cover[vert])
                if len(cov) > 0:
                    simps[j+1] = simps[j+1] + [presimp + [vert]]
    
    return simps
    
# give entry value for things in the nerve
def point_nerve_plus(cover,dists,maxdim=2):
    # cover is a list of index lists, referring to points.
    # each cover index corresponds to a group of center landmarks.
    # dists is (numpts x numcovers) with matching indexes to cover.
    
    numpts = len(dists)
    numcov = len(cover)
    simps = [[] for x in range(maxdim+1)]
    
    # accrue vertices
    for i in range(numcov):
        simps[0] = simps[0] + [[i]]
    
    # accrue higher order simps up to maxdim.
    for j in range(maxdim):
        for presimp in simps[j]:
            ik = presimp[0]
            precov = cover[ik]
            for ik in presimp[1:]:
                precov = intersect(precov,cover[ik])
            for vert in range(ik+1,numcov):
                cov = intersect(precov,cover[vert])
                if len(cov) > 0:
                    simps[j+1] = simps[j+1] + [presimp + [vert]]
    
    # gather entry radii for max dim'l simplices (assume use in cocycle)
    entry_radii = {}
    for simp in simps[maxdim]:
        ss = cover[simp[0]]
        for index in simp[1:]:
            ss = intersect(ss,cover[index])
        value = min([max([dists[s][ii] for ii in simp]) for s in ss])
        entry_radii[tuple(simp)] = value
    
    return (simps,entry_radii)

# turn landmark distances into bary coords via a kernel.
def bary_coords(assign,dists,cover_rad):
    def altmax(this): # covers empty lists
        if len(this) == 0:
            return 0
        else:
            return max(this)
    
    numpts = len(assign)
    numcov = int(max([altmax(x) for x in assign])+1)
    barys = np.zeros([numpts,numcov])
    for slot in range(numpts):
        for (index,dist) in zip(assign[slot],[dists[slot][x] for x in assign[slot]]):
            wgt = tent_kernel(dist/cover_rad)
            barys[slot,index] = wgt
        barys[slot,:] = barys[slot,:]/np.sum(barys[slot,:])
    return barys

## Special Graphs:: Used to aid in special presentations.
# Define a graph (1-skeleton) from a simplicial complex.
def make_graph(simps):
    verts = simps[0]
    edges = simps[1]
    graph = nx.Graph()
    for vert in verts:
        graph.add_node(vert[0])
    for edge in edges:
        graph.add_edge(edge[0],edge[1],weight=1)
    return graph
    
# define (inverse) similarity weighted graph (e.g., distances)
# and the minimal spanning tree (MST) for this weighted graph.
def sim_MST_graph(cover):
    numcov = len(cover)
    graph = nx.Graph()
    for vert in range(numcov):
        graph.add_node(vert)
    
    Csizes = [len(cover[x]) for x in range(numcov)]
    for ii in range(numcov):
        for jj in range(ii+1,numcov):
            sim = (len(intersect(cover[ii],cover[jj]))
                 / np.sqrt(Csizes[ii]*Csizes[jj]))
            if not (dist == 0):
                graph.add_edge(ii,jj,weight=1/sim)
    graphMST = nx.minimum_spanning_tree(graph)
    distmat = nx.floyd_warshall_numpy(graphMST,nodelist=list(range(numcov)))
    root = np.argmin(np.max(distmat,axis=0))
    return (root,graphMST,graph)

# Uses altcoords to obtain a contractible subset and gluing-cross rules.
# Requires a special distance function on the altCoords (B').
def sim_MST_graph(altcoords,landmarks,cover,distfcn = stereo_dist):
    # coordinate sorting methods:
    if distfcn == tor_dist:
        k = altcoords.shape[1]
    else:
        k = 1
    cross_edges = [[] for x in range(k)]

    
    numcov = len(cover)
    graph = nx.Graph()
    
    for vert in range(numcov):
        graph.add_node(vert)
        
    Csizes = [len(cover[x]) for x in range(numcov)]
    for ii in range(numcov):
        for jj in range(ii+1,numcov):
            sim = (len(intersect(cover[ii],cover[jj]))
                 / np.sqrt(Csizes[ii]*Csizes[jj]))
            if (sim > 0):
            # check if edge represents geodesic within stereographic proj.
                if k == 1:
                    (d1,d2) = distfcn(altcoords[landmarks[ii],:],
                        altcoords[landmarks[jj],:],extra=1)
                    if d1 != d2:
                        indexes = [0]
                    else:
                        indexes = []
                else:
                    (d1,flips) = distfcn(altcoords[landmarks[ii],:],
                        altcoords[landmarks[jj],:],extra = 2)
                    indexes = np.where(flips == 1)[0]
                # append to cross_index or the graph. 
                if len(indexes) == 1:
                    cross_edges[indexes[0]].append((ii,jj))
                elif len(indexes) == 0:
                    graph.add_edge(ii,jj,weight=1/sim)
    # Graph to MST, root chosen most centrally.
    graphMST = nx.minimum_spanning_tree(graph)
    distmat = nx.floyd_warshall_numpy(graphMST,nodelist=list(range(numcov)))
    root = np.argmin(np.max(distmat,axis=0))
    
    return (root,graphMST,graph,cross_edges)
    
# weights used for circular coordinates.
def nerve_weights(cover,edges,adjust = 0):
    #numpts = max([max(x) for x in cover])+1
    numcov = len(cover)
    Wv = np.zeros(numcov)
    We = np.zeros(len(edges))
    Wm = np.zeros([numcov,numcov])
    
    for vert in range(numcov):
        Wv[vert] = len(cover[vert])
    
    for (k,edge) in enumerate(edges):
        (v1,v2) = edge
        val = len(intersect(cover[v1],cover[v2])) + adjust
        We[k] = val
        Wm[v1,v2] = val
        Wm[v2,v1] = val
    return (Wv,We,Wm)
def kernel_weights(points,edges,radius,distfcn=eucl_dist,Dmat=None):
    numcov = points.shape[0]
    Wv = np.zeros(numcov)
    We = np.zeros(len(edges))
    Wm = np.zeros([numcov,numcov])
    
    for (k,edge) in enumerate(edges):
        (v1,v2) = edge
        if distfcn == 'precomputed':
            dist = Dmat[v1,v2]
        else:
            dist = distfcn(points[v1,:],points[v2,:])
        val = tent_kernel(dist/(2*radius))
        We[k] = val
        Wm[v1,v2] = val
        Wm[v2,v1] = val
        Wv[v1] = Wv[v1] + val
        Wv[v2] = Wv[v2] + val
    return (Wv,We,Wm)
def unit_weights(edges):
    numcov = max([max(x) for x in edges]) + 1
    Wv = np.ones(numcov)
    We = np.ones(len(edges))
    Wm = np.zeros([numcov,numcov])
    
    for (k,edge) in enumerate(edges):
        (v1,v2) = edge
        Wm[v1,v2] = 1
        Wm[v2,v1] = 1
    return (Wv,We,Wm)
    
### ---------- Vector Bundle Functions ---------- ###
    
# obtain PCA models. models[i][0]=offset & models[i][1]= Proj. direction
# Additionally, models[i][2] is a rescale factor. 
def pca_models(pts,cover,PCAdim,verbose = False,Cvec=None,rescale=False):
    numcov = len(cover)
    models = [[] for x in range(numcov)]
    topvars = [0 for x in range(numcov)]
    remvars = [0 for x in range(numcov)]
    for index in tqdm(range(numcov)):
        X = pts[np.array(cover[index]),:]
        (Xpca,vars,comp,mean) = DR.pca(X,outdim=0,verbose = True)
        
        # visualize some data fibers
        if verbose and type(verbose) == type(0) and index%verbose == 0:
            if type(Cvec) == type(None):
                colors = 'r'
            else:
                colors = DR.colorprep(Cvec[cover[index]],symm=False)
            DR.scatter3(Xpca,colors=colors)
            plt.show()
        
        topvars[index] = vars[PCAdim-1]
        remvars[index] = np.sum(vars[PCAdim:])
        vec = comp[0:PCAdim,:]/np.linalg.norm(comp[0:PCAdim,:])
        if rescale:
            models[index] = (mean,vec,np.sqrt(vars[0]))
        else:
            models[index] = (mean,vec,np.ones(PCAdim))
            
    if not verbose:
        return models
    else:
        return (models,topvars,remvars)

# oversimplified presentation for manifold learning models.
# these models use nearest neighbor graphs. 
# (remembers outputs rather than parameters, to avoid rebuilding kernel evals)
def mani_models(pts,cover,outdim,nn=10,MLtype = 'iso',rescale = False,verbose = False,
    Cvec = None):
    if MLtype in ['iso','lle','lta']:
        ML = MLM[MLtype](n_neighbors=nn,n_components=10)
    
    numcov = len(cover)
    models = [[] for x in range(numcov)]
    varlist = [[] for x in range(numcov)]
    if verbose:
        print([len(x) for x in cover])
    for index in tqdm(range(numcov)):
        X = pts[np.array(cover[index]),:]
        if MLtype in ['iso','lle','lta']:
            Xnew = ML.fit_transform(X)
        else: 
            Xnew = X
        (Xnew,vars,no,no) = DR.pca(Xnew,outdim=10,verbose=True)
        if verbose and type(verbose) == type(0) and index%verbose == 0:
            if type(Cvec) == type(None):
                colors = 'r'
            else:
                colors = DR.colorprep(Cvec[cover[index]],symm=False)
            DR.scatter3(Xnew,colors=colors)
            plt.show()
        if rescale:
            models[index] = Xnew[:,:outdim]/np.sqrt(vars[:outdim])
        else:
            models[index] = Xnew[:,:outdim]
            varlist[index] = vars
    if verbose:
        return (models,np.array(varlist))
    else:
        return models
        
# General Model alignment for d >= 1. (currently unused)
def model_align_gen(cover,orths,v):
    # main task: orient things, assuming orths are (very nearly?) a cocycle.
    dim = orths[list(orths)[0]].shape[0]
    simps = point_nerve(cover,maxdim=1)
    delta = delta0D(simps,orths)
    deltaX = np.linalg.pinv(delta) #MP pseudo-inverse of augmented coboundary
    
    # define a long vector in C1(Ncover,Rdim):
    shiftE = np.zeros(dim*len(simps[1]))
    for (index,(i1,i2)) in enumerate(simps[1]):
        shiftE[index*dim:(index+1)*dim] = v[(i1,i2)]
    shiftV = np.matmul(deltaX,shiftE)
    offsets = {}
    for index in range(len(cover)):
        offsets[index] = shiftV[index*dim:(index+1)*dim]
    return offsets
    
# Model offset alignment for the case d=1 (so that orths = eta)
def model_align_1d(cover,w,v):
    # main task: orient things, assuming w=orths is a cocycle
    simps = point_nerve(cover,maxdim=1)
    delta = delta0A(simps,w)
    deltaX = np.linalg.pinv(delta) #MP pseudo-inverse of augmented coboundary
    shiftE = np.zeros(len(simps[1]))
    for (index,(i1,i2)) in enumerate(simps[1]):
        shiftE[index] = v[(i1,i2)]
    return np.matmul(deltaX,shiftE)

# Align pca models via identity or reflection. Yields a cocycle in Z2.
def pca_align(pts,cover,models):    
    if len(models[0][1].shape) == 1:
        dim = 1
    else:
        dim = models[0][1].shape[1]
    numcov = len(cover)
    w = np.zeros([numcov,numcov])
    v = {}
    orths = {}
    RMSE = np.zeros([numcov,numcov])
    
    # a little concerned about how this will act for d > 1...?
    for i1 in range(numcov):
        for i2 in range(numcov):
            subset = np.array(intersect(cover[i1],cover[i2]))
            if len(subset) > 0:
                cov = pts[subset,:]
                # map and center.   # cov, P1, meanshift, SDs
                X1 = rep_pca(cov,models[i1][1],models[i1][0])/models[i1][2]
                X2 = rep_pca(cov,models[i2][1],models[i2][0])/models[i2][2]
                
                m1 = np.mean(X1,axis=0)
                m2 = np.mean(X2,axis=0)
                X1 = X1 - m1
                X2 = X2 - m2
                
                M = orth_proc(X1,X2)[0]
                w[i2,i1] = np.linalg.det(M)
                orths[(i2,i1)] = M
                v[(i2,i1)] = m2 - np.matmul(m1,M)
                Es = np.matmul(X1,M)-X2
                err = np.linalg.norm(Es)
                RMSE[i2,i1] = err/np.sqrt(len(subset))
    return (w,v,orths,RMSE)

# Align R^1 models via identity or reflection. Yields a cocycle in Z2.
def line_align(pts,cover,models):
    if len(models[0][1].shape) == 1:
        dim = 1
    else:
        dim = models[0][1].shape[1]

    numcov = len(cover)
    w = np.zeros([numcov,numcov])
    v = {}
    orths = {}
    RMSE = np.zeros([numcov,numcov])
    
    for i1 in tqdm(range(numcov)):
        for i2 in range(numcov):
            (SS1,SS2) = intSS(cover,i1,i2)
            if len(SS1) > 0:
                X1 = models[i1][SS1,:] # should have the proper shape
                X2 = models[i2][SS2,:] # should have the proper shape
                
                m1 = np.mean(X1,axis=0)
                m2 = np.mean(X2,axis=0)
                X1 = X1 - m1
                X2 = X2 - m2
                                
                M = orth_proc(X1,X2)[0]
                w[i2,i1] = np.linalg.det(M)
                orths[(i2,i1)] = M
                v[(i2,i1)] = m2 - np.matmul(m1,M)
                Es = np.matmul(X1,M)-X2
                err = np.linalg.norm(Es)
                RMSE[i2,i1] = err/np.sqrt(len(SS1))
    return (w,v,orths,RMSE) # (Z2, shifts, mats, RMSE)

# alignment of pca/ML models along a spanning tree.
# should work for coords in Tk or RPk (flat or stereos)
# edits cocycle and models in place.
def tree_line_align(coords,landmarks,cocycle,models,root,spantree,
    crosses,modeltype = 'pca'):

    # number of independent 1-cocycles considered in the base space. 
    k = len(crosses)
    num_fibers = len(models)
    
    # step 2: flip models according to cocycle alignment so that (nearly)
    # everything is oriented along the spantree (but not over the whole graph!!)
    allverts = list(range(num_fibers))
    vert = root
    visited = [root]
    margin = [root]
    while not len(setdiff(allverts,visited)) == 0:
        v1 = margin.pop()
        branches = spantree[v1]
        branches = setdiff(branches,visited)
        for v2 in branches:
            f1 = cocycle[v1,v2]
            if f1 == -1:
                if modeltype == 'pca':
                    models[v2] = (models[v2][0],-models[v2][1],models[v2][2])
                else:
                    models[v2][:,0] = -models[v2][:,0]
                cocycle[v2,:] = -cocycle[v2,:]
                cocycle[:,v2] = -cocycle[:,v2]
        visited = union(visited,branches)
        margin = union(margin,branches)
    slips = [[] for j in range(k)]
    orient = np.zeros(k)
    for j in range(k):
        for (ii,jj) in crosses[j]:
            slips[j].append(cocycle[ii,jj])
        orient[j] = np.mean(slips[j])
    print("Orientation String: " + str(orient))
    
    return np.sign(orient)
    
### ---------- Presentation Functions ---------- ###
## Assembly Frame Presentations ##
# In current version of PyTALLEM, we assume d = 1.

# General presentation function for localdim = 1.
# The case d>1, which must handle non-cocycles, is handled in MatTALLEM.
# Note that the frame vectors live in R^k for some k >= d = 1.
# In most cases, gen_present does not yield the ideal embedding dimension,
# so it is not used to create figures for the paper.
def gen_present(pts,base,frames,models,assign,cover,barys,cocycle,
    modeltype='pca',offsets = None):
    
    if type(offsets) == type(None):
        offsets = np.zeros(len(models))
    numpts = len(assign)
    basedim = base.shape[1]
    framedim = frames.shape[1]
    params = np.zeros([numpts,basedim+framedim])
    
    for index in range(numpts):
        ii = assign[index][0]

        Bi = base[index,:]
        Fi = frames[index,:]
        pt = pts[index,:]
        Hi = 0
        
        # PoU averaging of local coordinates:
        for jj in assign[index]:
            modelj = models[jj]
            if modeltype == 'pca': # Recreate from model.
                meanj = modelj[0]
                vecj = modelj[1]
                varj = modelj[2]
                hij = cocycle[ii,jj]*(np.sum((pt-meanj)*vecj/varj)+offsets[jj])
            else: # Slot finding.
                (SSi,SSj) = intSS(cover,ii,jj)
                slot = np.searchsorted(cover[ii][SSi],index)
                hij = cocycle[ii,jj]*(modelj[SSj[slot]]+offsets[jj])
            Hi = Hi + barys[index,jj]*hij
            
        params[index,:basedim] = Bi
        params[index,basedim:] = Hi*Fi
        
    return params

# circle-line presentation (lives in R^3)
def CL_present(pts,RP1s,CCs,models,assign,cover,barys,corr,modeltype = 'pca',
        tracksep = 0, trackvar = 0.4, doubletrack = False,offsets = None):
    def circ_param(a):
        return np.array([np.cos(a),np.sin(a),0])
    def adjust_param(a,b,h):
        b1 = b[0]
        b2 = b[1]
        p1 = circ_param(a)
        p2 = circ_param(a)*b1*h + np.array([0,0,b2*h])
        return p1+p2
    
    if type(offsets) == type(None):
        offsets = np.zeros(len(models))
    
    numpts = len(assign)
    if doubletrack:
        params = np.zeros([2*numpts,3])
    else:
        params = np.zeros([numpts,3])
    
    for index in range(numpts):
        ai = CCs[index]
        bi = RP1s[index,:]
        pt = pts[index,:]
        hi = 0
        ii = assign[index][0]
        
        # PoU averaging of local coordinates:
        for jj in assign[index]:
            modelj = models[jj]
            if modeltype == 'pca': # Recreate from model.
                meanj = modelj[0]
                vecj = modelj[1]
                varj = modelj[2]
                hij = corr[ii,jj]*(np.sum((pt-meanj)*vecj/varj)+offsets[jj])
            else: # Slot finding.
                (SSi,SSj) = intSS(cover,ii,jj)
                slot = np.searchsorted(cover[ii][SSi],index)
                hij = corr[ii,jj]*(modelj[SSj[slot]]+offsets[jj])
            hi = hi + barys[index,jj]*hij
        hi = tracksep+trackvar*hi
        params[index,:] = adjust_param(ai,bi,hi).flatten()
        if doubletrack:
            params[numpts+index,:] = adjust_param(ai,bi,-hi).flatten()
    return params

# projective-line presentation. (lives in R^4)
def PL_present(pts,RP1s,RP2s,models,assign,cover,barys,corr,modeltype = 'pca',
        trackvar = 0.2,offsets = None):
    def adjust_param(v,c,h): # S2, S1, [0,1]
        return np.array([(h*c[0]+1)*v[0],(h*c[0]+1)*v[1],(h*c[0]+1)*v[2],h*c[1]])
        
    if type(offsets) == type(None):
        offsets = np.zeros(len(models))

    numpts = len(assign)
    params = np.zeros([numpts,4])
    
    for index in range(numpts):
        vi = RP2s[index,:]
        ci = RP1s[index,:]
        pt = pts[index,:]
        hi = 0
        ii = assign[index][0]
        
        # PoU averaging of PC1 coordinates:
        for jj in assign[index]:
            modelj = models[jj]
            if modeltype == 'pca': # Recreate from model. (oft reused)
                meanj = modelj[0]
                vecj = modelj[1]
                varj = modelj[2]
                hij = corr[ii,jj]*(np.sum((pt-meanj)*vecj/varj)+offsets[jj])
            else: # Slot finding.
                (SSi,SSj) = intSS(cover,ii,jj)
                slot = np.searchsorted(cover[ii][SSi],index)
                hij = corr[ii,jj]*(modelj[SSj[slot]]+offsets[jj])
            hi = hi + barys[index,jj]*hij
        params[index,:] = adjust_param(vi,ci,trackvar*hi).flatten()
    return params

## Almost-Global-Alignment Presentations ##

# presents a line bundle over a circle on a square with different orient types.
def CL_square_present(pts,CCs,models,assign,cover,barys,cocycle,flip_index,
    orient = 1,modeltype = 'pca',offsets = None):
    
    if type(offsets) == type(None):
        offsets = np.zeros(len(models))
    
    numpts = len(assign)
    params = np.zeros([numpts,2])
    
    for index in range(numpts):
        ai = CCs[index]
        pt = pts[index,:]
        ii = assign[index][0]
        if ii >= flip_index:    # Intrinsic Replacement for classifying map.
            f0 = orient
        else:
            f0 = 1
        
        # PoU averaging of PC1 coordinates:
        hi = 0
        for jj in assign[index]:
            modelj = models[jj]
            if modeltype == 'pca': # Recreate from model.
                meanj = modelj[0]
                vecj = modelj[1]
                varj = modelj[2]
                hij = cocycle[ii,jj]*(np.sum((pt-meanj)*vecj/varj)+offsets[jj])
            else: # Slot finding.
                (SSi,SSj) = intSS(cover,ii,jj)
                slot = np.searchsorted(cover[ii][SSi],index)
                hij = cocycle[ii,jj]*(modelj[SSj[slot]]+offsets[jj])
            hi = hi + barys[index,jj]*hij
        hi = f0*hi #cocycle swap factoring
        params[index,:] = [ai,hi]
    return params

# presents a line bundle over RP2 on a cylinder via stereographic coords and dists.
def RP2L_cyl_present(pts,SCs,models,assign,cover,barys,cocycle,landmarks,
    orient = 1,modeltype = 'pca',offsets = None):
    
    if type(offsets) == type(None):
        offsets = np.zeros(len(models))
    
    numpts = len(assign)
    params = np.zeros([numpts,3])
    
    for index in range(numpts):
        Sk = SCs[index,:]
        pt = pts[index,:]
        params[index,:2] = Sk
        for ii in assign[index]: # attempt to have landmark nearby in stereo
            Si = SCs[landmarks[ii],:]
            (d1,d2) = stereo_dist(Sk,Si,extra = True)
            if d1 == d2:
                break
        if d1!=d2:      # check cocycle if needed
            f0 = orient
        else:
            f0 = 1
        
        # PoU averaging of PC1 coordinates:
        hi = 0
        for jj in assign[index]:
            modelj = models[jj]
            if modeltype == 'pca': # Recreate from model.
                meanj = modelj[0]
                vecj = modelj[1]
                varj = modelj[2]
                hij = cocycle[ii,jj]*(np.sum((pt-meanj)*vecj/varj)+offsets[jj])
            else: # Slot finding.
                (SSi,SSj) = intSS(cover,ii,jj)
                slot = np.searchsorted(cover[ii][SSi],index)
                hij = cocycle[ii,jj]*(modelj[SSj[slot]]+offsets[jj])
            hi = hi + barys[index,jj]*hij
        params[index,2] = f0*hi
    return params

def TL_cube_present(pts,TCs,models,assign,cover,barys,cocycle,landmarks,
    orient = None,modeltype = 'pca',offsets = None):

    k = TCs.shape[1]
    if type(orient) == type(None):
        orient = [1 for j in range(k)]
    
    if type(offsets) == type(None):
        offsets = np.zeros(len(models))
    
    numpts = len(assign)
    params = np.zeros([numpts,k+1])
    
    for index in range(numpts):
        Tk = TCs[index,:]
        pt = pts[index,:]
        params[index,:k] = Tk
        # Anchor Landmark:
        ii = assign[index][0]
        Ti = TCs[landmarks[ii],:]
        (d1,flips) = tor_dist(Tk,Ti,extra=2)
        
        # flip orientation of fiber coords as needed:
        f0 = 1
        for j in range(k):
            if flips[j] == 1:
                f0 = f0*orient[j]
        
        # PoU averaging of PC1 coordinates:
        hi = 0
        for jj in assign[index]:
            modelj = models[jj]
            if modeltype == 'pca': # Recreate from model.
                meanj = modelj[0]
                vecj = modelj[1]
                varj = modelj[2]
                hij = cocycle[ii,jj]*(np.sum((pt-meanj)*vecj/varj)+offsets[jj])
            else: # Slot finding.
                (SSi,SSj) = intSS(cover,ii,jj)
                slot = np.searchsorted(cover[ii][SSi],index)
                hij = cocycle[ii,jj]*(modelj[SSj[slot]]+offsets[jj])
            hi = hi + barys[index,jj]*hij
        params[index,k] = f0*hi
    return params

# presents coordinates (arbitrarily?) over R^k.
# Requires oriented models (b/c why not?)
def Flat_present(pts,base,models,assign,cover,barys,landmarks,
    modeltype = 'pca',offsets = None):

    k = base.shape[1]
    if type(offsets) == type(None):
        offsets = np.zeros(len(models))
    
    numpts = len(assign)
    params = np.zeros([numpts,k+1])
    
    for index in range(numpts):
        Bk = base[index,:]
        pt = pts[index,:]
        params[index,:k] = Bk
        # Anchor Landmark:
        ii = assign[index][0]
        
        # PoU averaging of PC1 coordinates:
        hi = 0
        for jj in assign[index]:
            modelj = models[jj]
            if modeltype == 'pca': # Recreate from model.
                meanj = modelj[0]
                vecj = modelj[1]
                varj = modelj[2]
                hij = cocycle[ii,jj]*(np.sum((pt-meanj)*vecj/varj)+offsets[jj])
            else: # Slot finding.
                (SSi,SSj) = intSS(cover,ii,jj)
                slot = np.searchsorted(cover[ii][SSi],index)
                hij = modelj[SSj[slot]]+offsets[jj]
            hi = hi + barys[index,jj]*hij
        params[index,k] = hi
    return params

### ---------- TALLEM analysis functions ---------- ###

# Compare adjusted local models for points in cover intersection (edge).
# edges input may be null, a vertex (neighbors), or explicit list of edges.
def fiber_model_compare(pts,models,cover,barys,offsets,cocycle,
    edges=None,modeltype='pca'):
    # preprocess edge list:
    if edges == None:
        edges = 0
    if type(edges) == type(0):
        v = edges
        edges = []
        numcov = len(cover)
        for ii in range(numcov):
            Lij = intersect(cover[v],cover[ii])
            if len(Lij) > 0:
                edges.append([v,ii])
    
    print("Edges to view: " + str(edges))
    
    # View series of edges' models
    for edge in edges:
        print("Next Viewing covers "+str(edge[0])+" and "+str(edge[1])+".")
        (ii,jj) = edge
        print("Cocyle is: " + str(cocycle[ii,jj]))
        X0 = []
        X1 = []
        BS = []
        if modeltype == 'pca': # Recreate from model.
            # prep subset and models:
            SS = intersect(cover[ii],cover[jj])
            (meani,meanj) = (models[ii][0],models[jj][0])
            (veci,vecj)  =  (models[ii][1],models[jj][1])
            (vari,varj)  =  (models[ii][2],models[jj][2])
            # eval points in subset
            for index in SS:
                pt = pts[index,:]
                x0i = np.sum((pt-meani)*veci/vari)
                x0j = np.sum((pt-meanj)*vecj/varj)
                x1i = x0i+offsets[ii]
                x1j = x0j+offsets[jj]
                bi = barys[index,ii]
                bj = barys[index,jj]
                X0.append([x0i,x0j])
                X1.append([x1i,x1j])
                BS.append([bi,bj])
        else: # Slot finding.
            (SSi,SSj) = intSS(cover,ii,jj)
            for (slot,index) in enumerate(cover[ii][SSi]):
                x0i = models[ii][SSi[slot]]
                x0j = models[jj][SSj[slot]]
                x1i = x0i+offsets[ii]
                x1j = x0j+offsets[jj]
                bi = barys[index,ii]
                bj = barys[index,jj]
                X0.append([x0i,x0j])
                X1.append([x1i,x1j])
                BS.append([bi,bj])
        X0 = np.array(X0)
        X1 = np.array(X1)
        BS = np.array(BS)
        
        #DR.scatter2(X0,colors = DR.two_bary_color(BS))
        #DR.scatter2(X1,colors = DR.two_bary_color(BS))
        XT = np.concatenate([X0,X1],axis=0)
        XC = np.concatenate([np.zeros(len(X0)),np.ones(len(X1))],axis=0)
        DR.scatter2(XT,DR.colorprep(XC,symm=False))
        plt.show()
    return
    
# Uses Thm 2 from the paper to precheck existence of a cocycle
# Trial checkings suggest that many matchups are quite volatile,
# probably due to the intersection size. This is confirmed by creating
# "oversized" nbhds, suggesting benefits from large intersection.
def check_pca_consistency(pts,cover,cover2 = None):
    # allocate dictionaries for storing necessary info indexed by pairs (i,j)
    # since some Sigma_ij will be reused, while others will be unneeded.
    
    if cover2 == None:
        cover2 = cover
    
    means = {}
    covars = {}
    maxEVs = {}
    subEVs = {}
    vars = {}
    
    num_fibers = len(cover)
    # first, preallocate for all single entry members.
    for ii in range(num_fibers):
        X = pts[cover[ii],:]
        numpts = len(cover[ii])
        mean = np.mean(X,axis = 0)
        Xc = X - np.repeat(np.array([mean]),numpts,axis=0)
        covar = np.matmul(Xc.T,Xc)/numpts
        EVs = np.flip(np.sort(np.linalg.eig(covar)[0]),axis=0)
        maxEV = EVs[0]
        subEV = EVs[1]
        var = np.sum(EVs)
        
        means[ii] = mean      # mean(Xi)
        covars[ii] = covar    # covariance(Xi)
        maxEVs[ii] = maxEV    # lambda_{1,i}
        subEVs[ii] = subEV    # lambda_{2,i}
        vars[ii] = var        # sum_j(lambda_{j,i})
    
    RandBs = {} # ratios and bounds, this one is indexed by triples (i,j,k)
    epses = {} # possible errors. This one is indexed by pairs of things...
    
    for ii in range(num_fibers):
        for jj in range(ii+1,num_fibers):
            Lij = intersect(cover2[ii],cover2[jj])
            for kk in range(jj+1,num_fibers):
                Lijk = intersect(Lij,cover2[kk])
                if Lijk == []:
                    continue
                Ls = {}
                Ls[(ii,jj)] = intersect(cover[ii],cover[jj])
                Ls[(ii,kk)] = intersect(cover[ii],cover[kk])
                Ls[(jj,kk)] = intersect(cover[jj],cover[kk])
                
                J = [ii,jj,kk,(ii,jj),(ii,kk),(jj,kk)]
                # remove matchups of type (aa,bb)--(bb,dd) or (aa)--(bb,cc)
                JJ = [(ii,jj),(ii,kk),(jj,kk),(ii,(ii,jj)),(ii,(ii,kk)),
                    (jj,(ii,jj)),(jj,(jj,kk)),(kk,(ii,kk)),(kk,(jj,kk))] #9/15
                
                # add info as necessary:
                for entry in J[3:]:
                    if not entry in means:
                        X = pts[Ls[entry],:]
                        numpts = len(Ls[entry])
                        mean = np.mean(X,axis = 0)
                        Xc = X - np.repeat(np.array([mean]),numpts,axis=0)
                        covar = np.matmul(Xc.T,Xc)/numpts
                        EVs = np.flip(np.sort(np.linalg.eig(covar)[0]),axis=0)
                        maxEV = EVs[0]
                        subEV = EVs[1]
                        var = np.sum(EVs)
                        
                        means[entry] = mean      # mean(Xi)
                        covars[entry] = covar    # covariance(Xi)
                        maxEVs[entry] = maxEV    # lambda_{1,i}
                        subEVs[entry] = subEV    # lambda_{2,i}
                        vars[entry] = var        # sum_j(lambda_{j,i})
                
                # Find necessary eigenvalue ranges
                EVmax = np.inf
                EVsub = 0
                ratiomin = np.inf
                for entry in J:
                    EVmax = np.min([EVmax,maxEVs[entry]])
                    EVsub = np.max([EVsub,subEVs[entry]])
                for entry in J[3:]:
                    ratiomin = np.min([ratiomin,maxEVs[entry]/vars[entry]])
                
                Bound = np.sqrt(ratiomin/24)
                Numer = 0

                for (entry1,entry2) in JJ:
                    if not (entry1,entry2) in epses:
                        eps=np.max(np.linalg.eig(covars[entry1]-covars[entry2])[0])
                        epses[(entry1,entry2)] = eps
                    else:
                        eps = epses[(entry1,entry2)]
                    print((entry1,entry2,eps))
                    Numer = np.max([Numer,eps])
                
                print((Numer,EVmax,EVsub))
                Ratio = Numer / (EVmax - EVsub)
                RandBs[(ii,jj,kk)] = (Ratio,Bound)
    return RandBs
    
# linear correlation (*between fiber coordinates*)
# the idea is to do correlation per-fiber, then consider the dist'n of coeffs.
# base type is obtained from "orient". (list = Tk, single is RPk)
# not working correctly?
def local_fiber_corr(coords,params,cover,orient,distfcn=stereo_dist,
        verbose = False):
    # align via the first point of each.
    corrs = np.zeros(len(cover))
    slopes = np.zeros(len(cover))
    inters = np.zeros(len(cover))
    for (index,set) in enumerate(cover):
        vals = np.zeros([len(set),2])
        vals[:,0] = params[set,-1]
        vals[:,1] = coords[set,-1]
        P0 = params[set[0],:-1]
        C0 = coords[set[0],:-1]
        for (i,slot) in enumerate(set):
            Pi = params[slot,:-1]
            Ci = coords[slot,:-1]
            #print(Pi - Ci) # WHY are these the same??
            if distfcn == stereo_dist:
                (dp1,dp2) = distfcn(Pi,P0,extra=1)
                (dc1,dc2) = distfcn(Ci,C0,extra=1)
                if not dp1 == dp2:
                    vals[i,0] = float(orient)*vals[i,0]
                if not dc1 == dc2:
                    vals[i,1] = float(orient)*vals[i,1]
            else:
                Pflip = distfcn(Pi,P0,extra=2)[1]
                Cflip = distfcn(Ci,C0,extra=2)[1]
                for (j,fac) in enumerate(orient):
                    vals[i,0] = (fac**Pflip[j])*vals[i,0]
                    vals[i,1] = (fac**Cflip[j])*vals[i,1]
        if verbose == True:
            print("In coverset number " + str(index) + ":")
            print("There are " + str(Pflips) + " Pflips.")
            print("There are " + str(Cflips) + " Cflips.")
        
        # derive best fit line information.
        (varP,varC) = np.var(vals,axis=0)
        (muP,muC) = np.mean(vals,axis=0)
        corrs[index] = np.corrcoef(vals.T)[0,1]
        slopes[index] = corrs[index]*np.sqrt(varC/varP)
        inters[index] = muC - slopes[index]*muP
        
        if type(verbose) == type(0) and verbose > 0:
            if index%verbose == 0:
                print(corrs[index])
                DR.scatter2(vals)
                plt.show()
    return (corrs,slopes,inters)

# distance correlation over the entire dataset.
# we may want to be able to adjust the length/coord ratios?
# defaults may be set for the distances before input into the fcn:
# this version may use a lot of memory for large datasets (SS?)
def dist_cor(coords,params,dC = TkB_dist,dP = TkB_dist):
    numpts = coords.shape[0]
    Cmat = np.zeros([numpts,numpts])
    Pmat = np.zeros([numpts,numpts])
    
    # enable distance matrix input by index mapping pullback.
    # Still need to push into a new matrix, as it will be edited.
    if dC == "given":
        def distC(i,j):
            return coords[i,j]
    else:
        def distC(i,j):
            return dC(coords[i,:],coords[j,:])
    if dP == "given":
        def distP(i,j):
            return params[i,j]
    else:
        def distP(i,j):
            return dP(params[i,:],params[j,:])
    
    for i in range(numpts):
        for j in range(i+1,numpts):
            DC = distC(i,j)
            DP = distP(i,j)
            (Cmat[i,j],Cmat[j,i]) = (DC,DC)
            (Pmat[i,j],Pmat[j,i]) = (DP,DP)
    
    (Cvec,Pvec) = (np.mean(Cmat,axis=0),np.mean(Pmat,axis=0))
    (Cbar,Pbar) = (np.mean(Cvec,axis=0),np.mean(Pvec,axis=0))
    
    Cmat2 = np.zeros([numpts,numpts])
    Pmat2 = np.zeros([numpts,numpts])
    
    for i in range(numpts):
        for j in range(numpts):
            Cmat[i,j] = Cmat[i,j] - Cvec[i] - Cvec[j] + Cbar
            Pmat[i,j] = Pmat[i,j] - Pvec[i] - Pvec[j] + Pbar
        
    (dcovC,dcovP) = (np.sqrt(np.mean(Cmat*Cmat)),np.sqrt(np.mean(Pmat*Pmat)))
    dcov2 = np.mean(Cmat*Pmat)
    dcor2 = dcov2/(dcovC*dcovP)
    
    return np.sqrt(dcor2)
    
        
### ---------- Cocycle Functions ---------- ###

# Grab a cocycle from Landmarks. min_cover_rad is necessary for PoU.
def cocycle_prep(pts,landmarks,min_cover_rad,type = 'circular',
        maxfeatures = 2,TL = 1/4,TU=3/4,verbose = True,distfcn = eucl_dist,
        selector = big_gap_selector,Dmat=None,fullcheck=False):
    # define coeff space based on usage:
    if type in ['circular','toroidal']:
        q = 97
    elif type == 'projective':
        q = 2
    
    # get rep cocycles in C^1(K_d,Z_q)
    if distfcn == 'precomputed': # better have Dmat!
        Dmat_L = Dmat[landmarks,:][:,landmarks]
        if fullcheck:
            RR = ripser.ripser(Dmat_L,coeff=q,maxdim=2,do_cocycles=True,
                distance_matrix=True)
            if type == 'projective':
                RR2 = ripser.ripser(Dmat_L,coeff=97,maxdim=2,
                    distance_matrix=True)
        else:
            RR = ripser.ripser(Dmat_L,coeff=q,maxdim=1,do_cocycles=True,
                distance_matrix=True)
    else:
        points = pts[landmarks,:]
        if fullcheck:
            RR = ripser.ripser(points,coeff=q,maxdim=2,do_cocycles=True,
                metric=distfcn)
            if type == 'projective':
                RR2 = ripser.ripser(points,coeff=97,maxdim=2,
                    metric=distfcn)
        else:
            RR = ripser.ripser(points,coeff=q,maxdim=1,do_cocycles=True,
                metric=distfcn)
        
    PD1 = RR['dgms'][1]
    persistences = PD1[:,1] - PD1[:,0]
    # grab *most* persistent cocycle for circular coords
    if type == 'circular':
        featured = np.argmax(persistences)
        cocycle = RR['cocycles'][1][featured]
        (birth,death) = PD1[featured]
        
        # choose radius within shrunken (birth,death) interval, usually min_cover_rad?
        alphaL = birth + TL*(death-birth)
        alphaU = birth + TU*(death-birth)
        radius = min(max(alphaL,2*min_cover_rad),alphaU)/2
    # grab outlier persistent cocycles for proj coords and sum.
    # (after H^1 component of the characteristic class.)
    
    # grab desired number of most persistent cocycles for circular coords.
    if type == 'toroidal':
        featured = np.flip(np.argsort(persistences),axis=0)[:maxfeatures]
        BDs = PD1[featured,:]
        birth = np.max(BDs[:,0])
        death = np.min(BDs[:,1])
        if birth < death:
            print("toroidal cycles are present simultaneously")
        else:
            print("toroidal cycles are NOT present simultaneously")
        
        cocycles = []
        for index in featured:
            cocycles.append(RR['cocycles'][1][index])
        alphaL = birth + TL*(death-birth)
        alphaU = birth + TU*(death-birth)
        radius = min(max(alphaL,2*min_cover_rad),alphaU)/2
        
    if type == 'projective':
        birth = np.inf
        death = 0
        featured = selector(persistences,maxfeatures)
        for iter in range(maxfeatures):
            birth = np.max(PD1[featured,0])
            death = np.min(PD1[featured,1])
            if birth < death:
                break
            print("the "+str(len(featured))+" most persistent generators do not coexist.")
            featured = featured[1:] # remove the least persistent feature.
        # sum the cocycles (turn to mat, sum in Z2) we leave unused as 1.
        cocycle = np.ones([len(landmarks),len(landmarks)])
        for index in featured:
            for (b,d,val) in RR['cocycles'][1][index]:
                    cocycle[b,d] = ((-1)**val)*cocycle[b,d]
                    cocycle[d,b] = ((-1)**val)*cocycle[d,b]
        alphaL = birth + TL*(death-birth)
        alphaU = birth + TU*(death-birth)
        print("low = "+str(alphaL/2)+", base = "+str(min_cover_rad)
            +", high = "+str(alphaU/2))
        radius = min(max(alphaL,2*min_cover_rad),alphaU)/2 # rips param is double
        # For a 1-cocycle, the birth occurs simultaneously in rips/cech
        # but the death in cech may occur later (b/c 1 simplices are matched, but
        # 2 simplices may require more than half the rips param as radius)
        # A small extra buffer may be required for the birth, for *finite data*
    # If verbose, look at the diagram and give info.
    if verbose:
        print("There were " + str(len(featured)) + " cocycle(s) chosen.")
        print("The birth-death interval is: (" + str(birth) +","+ str(death)+")")
        print("The chosen radius is: " + str(radius))
    if fullcheck and type == 'projective':
        print("Showing persistence diagram with q = 2")
    if verbose or fullcheck:
        ripser.plot_dgms(RR['dgms'],show=True)
    if fullcheck and type=='projective':
        print("Showing persistence diagram with q = 97")
        ripser.plot_dgms(RR2['dgms'],show=True)
    
    if type == 'toroidal':
        return (cocycles,radius)
    else:
        return (cocycle,radius)
# check cocycle (in Z = {...,-1,0,1,...})
def cocycle_check_Z(triangles,cochain,verbose = True):
    if verbose:
        bad_tris = []
    for tri in triangles:
        (i1,i2,i3) = tri
        w1 = cochain[i1,i2]
        w2 = cochain[i2,i3]
        w3 = cochain[i1,i3]
        if w1 + w2 != w3:
            if verbose:
                bad_tris = bad_tris + [tri]
            else:
                return "No, its not a cocycle."
    if verbose:
        return bad_tris
    else:
        return "Yes, its a cocycle."
# check cocycle (in Z_2 = {+1,-1})
def cocycle_check_Z2(triangles,cochain,verbose = True):
    if verbose:
        bad_tris = []
    for tri in triangles:
        (i1,i2,i3) = tri
        w1 = cochain[i1,i2]
        w2 = cochain[i2,i3]
        w3 = cochain[i1,i3]
        if w1*w2 != w3:
            if verbose:
                bad_tris = bad_tris + [tri]
            else:
                return "No, its not a cocycle."
    if verbose:
        return (bad_tris)
    else:
        return "Yes, its a cocycle."

### ---------- Circle Coords and Bundle Functions ---------- ###

# Primary call for circular coords.
def circular_coords(pts,landmarks,min_cover_rad,TL = 1/2,TU=7/8,
        wgts = "dist",distfcn = eucl_dist,verbose = True,Dmat=None):
    
    points = pts[landmarks,:]
    q = 97
    
    (cocycle,radius) = cocycle_prep(pts,landmarks,min_cover_rad,
        type='circular',TL=TL,TU=TU,verbose=verbose,Dmat=Dmat)
    
    # Finding coverset assignments, and barycentric coords for later.
    print("Finding Cover Data for Circle Coords")
    if distfcn == 'precomputed':
        (cover,assign,dists) = landmark_cover(Dmat,landmarks,radius,
            distfcn=distfcn)
    else:
        (cover,assign,dists) = landmark_cover(pts,landmarks,radius,
            distfcn=distfcn)
    barys = bary_coords(assign,dists,radius)
    simps = point_nerve(cover,maxdim=2)
    #simps = rips(pts[landmarks],radius)
    
    # induce cocycle from Zq to Z and express as matrix over landmarks.
    NL = len(landmarks)
    CCmat = np.zeros([NL,NL]) # [[circle-cocycle matrix]]
    for edge in cocycle:
        if edge[2] <= (q-1)/2:
            CCmat[edge[0],edge[1]] = edge[2]
            CCmat[edge[1],edge[0]] = -edge[2]
        else:
            CCmat[edge[0],edge[1]] = edge[2] - q
            CCmat[edge[1],edge[0]] = q - edge[2]
    
    # check if its a cocycle on the necessary triangles given radius.
    # (Typically, this should not be a problem)
    if verbose:
        print(cocycle_check_Z(simps[2],CCmat))
    
    # Rephrase the cocycle as function on the simplices *as ordered in simps*.
    # This simplex-index-based ordering will guide the linear algebra.
    eta = np.zeros(len(simps[1]))
    for (index,edge) in enumerate(simps[1]):
        eta[index] = CCmat[edge[0],edge[1]]
    
    # weight-adjusted cobdry inverse (informs the notion of "smooth" over nerve)
    delta = delta0(simps)
    
    if distfcn == 'precomputed':
        Dmat_L = Dmat[landmarks,:][:,landmarks]
    else:
        Dmat_L = None
    del_pinv = weighted_coboundary_inverse(simps,cover,points,radius,
        wgts=wgts,distfcn=distfcn,Dmat=Dmat_L)
    
    # harmonic function on simplicial complex
    tau = -np.matmul(del_pinv,eta)
    
    # theta is eta adjusted by a coboundary (induces same cohom generator)
    thetavec = eta + np.matmul(delta,tau)
    theta = np.zeros([NL,NL])
    for (index,edge) in enumerate(simps[1]):
        theta[edge[0],edge[1]] = thetavec[index]
        theta[edge[1],edge[0]] = thetavec[index]
        
    return cocycle2circle(tau,theta,radius,assign,barys)

# Primary call for toroidal coords.
def torus_coords(pts,landmarks,min_cover_rad,outdim=2,TL = 1/2,TU=7/8,
        wgts = "dist",distfcn = eucl_dist,verbose = True,Dmat=None):
    
    points = pts[landmarks,:]
    q = 97
    # may need to update with multiple radii? Preferably not?
    (cocycles,radius) = cocycle_prep(pts,landmarks,min_cover_rad,
        type='toroidal',maxfeatures=outdim,TL=TL,TU=TU,
        verbose=verbose,Dmat=Dmat)
    
    # Finding coverset assignments, and barycentric coords for later.
    print("Finding Cover Data for Torus Coords")
    if distfcn == 'precomputed':
        (cover,assign,dists) = landmark_cover(Dmat,landmarks,radius,
            distfcn=distfcn)
    else:
        (cover,assign,dists) = landmark_cover(pts,landmarks,radius,
            distfcn=distfcn)
    barys = bary_coords(assign,dists,radius)
    simps = point_nerve(cover,maxdim=2)
    #simps = rips(pts[landmarks],radius)
    
    # induce each cocycle from Zq to Z and express as matrix over landmarks.
    NL = len(landmarks)
    CCstack = np.zeros([outdim,NL,NL]) # [[circle-cocycle matrices]]
    for (slot,cocycle) in enumerate(cocycles):
        for edge in cocycle:
            if edge[2] <= (q-1)/2:
                CCstack[slot,edge[0],edge[1]] = edge[2]
                CCstack[slot,edge[1],edge[0]] = -edge[2]
            else:
                CCstack[slot,edge[0],edge[1]] = edge[2] - q
                CCstack[slot,edge[1],edge[0]] = q - edge[2]
    
    # check if its a cocycle on the necessary triangles given radius.
    # (Typically, this should not be a problem)
    if verbose:
        for slot in range(outdim):
            print(cocycle_check_Z(simps[2],CCstack[slot]))
    
    # Rephrase the cocycle as function on the simplices *as ordered in simps*.
    # This simplex-index-based ordering will guide the linear algebra.
    eta = np.zeros([outdim,len(simps[1])])
    for (index,edge) in enumerate(simps[1]):
        eta[:,index] = CCstack[:,edge[0],edge[1]]
    
    # weight-adjusted cobdry inverse (informs the notion of "smooth" over nerve)
    delta = delta0(simps)
    if distfcn == 'precomputed':
        Dmat_L = Dmat[landmarks,:][:,landmarks]
    else:
        Dmat_L = None

    del_pinv = weighted_coboundary_inverse(simps,cover,points,radius,
        wgts=wgts,distfcn=distfcn,Dmat=Dmat_L)
    
    # harmonic functions on simplicial complex tied to each cocycle
    tau = -np.matmul(del_pinv,eta.T).T
    
    # theta is eta adjusted by a coboundary (induces same cohom generator)
    thetavec = eta + np.matmul(delta,tau.T).T
    theta = np.zeros([outdim,NL,NL])
    
    #print('')
    #print(thetavec.shape)
    #print(theta.shape)
    #print('')
    
    for (index,edge) in enumerate(simps[1]):
        theta[:,edge[0],edge[1]] = thetavec[:,index]
        theta[:,edge[1],edge[0]] = thetavec[:,index]
    
    X = np.zeros([len(pts),2*outdim])
    F = np.zeros([len(pts),outdim]) 
    for slot in range(outdim):
        (CCs,RmZs) = cocycle2circle(tau[slot],theta[slot],radius,assign,barys)
        X[:,2*slot:2*(slot+1)] = CCs
        F[:,slot] = RmZs
        
    return (X,F)
    
# Naively defines circular coords via arctan of a mapping in R^2. 
def ML_circ_coords(pts,MLtype = 'iso',nn = 10,verbose = True):
    ML = MLM[MLtype](n_neighbors=nn,n_components=2)
    Xnew = ML.fit_transform(pts)
    angles = np.arctan2(Xnew[:,0],Xnew[:,1]) + np.pi
    if verbose:
        DR.scatter2(Xnew,colors = DR.colorprep(angles,symm=False))
        DR.plt.show()
    RmZs = angles/(2*np.pi)
    CCs = np.array([[np.cos(a),np.sin(a)] for a in angles])
    return (CCs,RmZs)
    
# Helper function which finds the coboundary inverse under adjusted weights.
# The adjusted weights are based on a kernel or coverset intersection size.
def weighted_coboundary_inverse(simps,cover,points,radius,
        wgts = "dist",distfcn=eucl_dist,Dmat=None):
    # weighted cobdry on ONBs for C^0->C^1 via vertex/edge weights.
    # So, error in the M-P p-inverse is w.r.t. this inner product/norm.
    
    # Wm[i,j] = |Ci n Cj| Wv[i] = |Ci| OR Wm[i,j] =  d(li,lj), Wv[i] = sum_j Wm[i,j]
    if wgts == "nerve":
        (Wv,We,Wm) = nerve_weights(cover,simps[1],adjust=0.1)
    elif wgts == "dist":
        (Wv,We,Wm) = kernel_weights(points,simps[1],radius
            ,distfcn=distfcn,Dmat=Dmat)
    else:
        (Wv,We,Wm) = unit_weights(simps[1])
    
    # define basic cobdry, rework into weighted (ON) bases, take inverse,
    # then return on the usual (non-normalized) bases again:
    delta = delta0(simps)
    delW = np.matmul(np.diag(1/np.sqrt(We)),np.matmul(delta,np.diag(np.sqrt(Wv))))
    delWpinv = np.linalg.pinv(delW)
    return np.matmul(np.diag(np.sqrt(Wv)),np.matmul(delWpinv,np.diag(1/np.sqrt(We))))
    
	
# theta is the adjusted cocycle and tau is its harmonic integral in RmZ
def cocycle2circle(tau,theta,radius,assign,barys):
    numpts = len(assign)
    CCs = np.zeros([numpts,2])
    RmZs = np.zeros([numpts])
    for ii in range(numpts):
        jj = assign[ii][0]
        val = tau[jj]
        for kk in assign[ii][1:]:
            val = val + barys[ii][kk]*theta[jj][kk]
        RmZs[ii] = val%1
        val = 2*np.pi*val
        CCs[ii,:] = [np.cos(val),np.sin(val)]
    return (CCs,RmZs)

### ---------- Projective Coords ---------- ###
def projective_coords(pts,landmarks,cover_rad,maxfeatures=2,
    TL = 1/10,TU=9/10,verbose=True,selector = big_gap_selector,distfcn = eucl_dist,
    get_cocycle = False,Dmat = None,fullcheck=False):
    
    # prep cocycle and choose cover radius based on cocycle.
    (cocycle,new_cover_rad) = cocycle_prep(pts,landmarks,cover_rad,
        type='projective',maxfeatures=maxfeatures,TL=TL,TU=TU,
        verbose=verbose,selector=selector,Dmat=Dmat,fullcheck=fullcheck)
    
    if verbose:
        print("New cover Radius = " + str(new_cover_rad))
    
    # Find Cover Data for chosen radius
    print("Finding Cover Data for Projective Coords")
    if distfcn == 'precomputed':
        (cover,assign,dists) = landmark_cover(Dmat,
            landmarks,new_cover_rad,distfcn=distfcn)
    else:
        (cover,assign,dists) = landmark_cover(pts,
            landmarks,new_cover_rad,distfcn=distfcn)

    
    # Testing for problem points (pre barys)
    #if verbose == 2:
    #    problems = []
    #    for (i,x) in enumerate(assign):
    #        if len(x) == 0:
    #            problems.append(i)
    #    problems = np.array(problems)
    #    if verbose:
    #        print("Problem points: " + str(problems))
    #    Dmat = np.zeros([len(landmarks),len(problems)])
    #    for (i,x) in enumerate(landmarks):
    #        for (j,y) in enumerate(problems):
    #            Dmat[i,j] = distfcn(pts[x,:],pts[y,:])
    #    print("Example landmark distances:")
    #    print(Dmat[:,0])
    #    print(Dmat[:,1])
    #    print(Dmat[:,2])
    #    print("min Landmark distances:")
    #    print(np.min(Dmat,axis=0))
    
    # Determine projective coordinates.
    barys = bary_coords(assign,dists,new_cover_rad)
    if get_cocycle:
        return (projective_param(barys,cocycle,assign),cocycle)
    else:
        return projective_param(barys,cocycle,assign)

# Produces representatives on the sphere.
# Utilizes assign[slot][0] as the reference index; this choice must be
# matched in TALLEM to produce the proper plot. 
def projective_param(barys,cocycle,assign):
    (numpts,numcov) = barys.shape
    PJCs = np.zeros([numpts,numcov])
    for slot in range(numpts):
        for index in assign[slot]:
            wgt = np.sqrt(barys[slot,index])
            orient = cocycle[assign[slot][0],index]
            PJCs[slot,index] = orient*wgt
    return PJCs

# Principal Projective Coordinates from high dim to low dim. last is RP0.
def principal_projection(PJCs,verbose=False,dims2save = 5):
    (numpts,numcov) = PJCs.shape
    dims2save = min(dims2save,numcov)
    PPJCs = [np.zeros([numpts,dims2save-j]) for j in range(dims2save)]
    distorts = np.zeros(numcov)
    perps = [np.zeros(numcov-j) for j in range(numcov-1)]
    PJprev = PJCs[:,:]
    if numcov == dims2save:           # If needed, save top dim.
        PPJCs[0][:,:] = PJCs[:,:]     # Assign
    
    for dimdown in tqdm(range(1,numcov)):
        #if verbose and dimdown%verbose == 0:
        #    print("Starting PPJC for dim = " + str(numcov-dimdown))
        cov = np.matmul(np.transpose(PJprev),PJprev) # covariance mat
        (vals,vecs) = np.linalg.eigh(cov)            # eigen
        order = np.argsort(vals)                     # get small
        vec = vecs[:,order[0]]                       # best perp
        perps[dimdown-1][:] = vec[:]                 # record.
        mat = np.array([vec])                        # reshape
        proj = np.matmul(np.transpose(mat),mat)      # proj_vec
        displace = np.matmul(PJprev,proj)            # vars
        distortion = np.sum(displace*displace)       # var (approximate)
        distorts[dimdown] = distortion/numpts        # assign
        ONB = null(proj)                             # new basis
        PJnext = np.matmul(PJprev,ONB)               # new coords
        norms = np.sqrt(np.sum(PJnext*PJnext,axis=1))   #\
        for slot in range(len(norms)):                  # rescale to sphere
            PJnext[slot,:] = PJnext[slot,:]/norms[slot] #/        
        if numcov - dimdown <= dims2save:                # save small dims
            PPJCs[dimdown-numcov][:,:] = PJnext[:,:]     # Assign
        PJprev = PJnext[:,:]
    angles = np.arctan2(PPJCs[-2][:,0],PPJCs[-2][:,1])
    distorts[-1] = circ_mean(angles,per=np.pi)[1]
    np.set_printoptions(precision = 3)
    print("Principal Projection Distortions: ")
    print(distorts[-9:])
    np.set_printoptions(precision = 8)
    return (PPJCs,perps,distorts)

# wrapper for stereographic projection from PPJCs
def stereo_wrapper(PPJCs,perps,dim):
    numcov = len(PPJCs)
    coords = PPJCs[-(dim+1)]
    vec = perps[-dim]
    return stereo_proj(coords,vec)

def stereo_proj(coords,vec):
    # Take reps in upper hemisphere.
    mat = np.array([vec])
    proj = np.matmul(np.transpose(mat),mat)
    verticals = np.matmul(coords,np.transpose(mat))
    flips = np.sign(verticals)
    for index in range(len(coords)):
        coords[index,:] = flips[index,0]*coords[index,:]
    # Linearly Project down
    ONB = null(proj)
    prestereo = np.matmul(coords,ONB)
    # rescale to obtain the final projection:
    rescales = 1/(1+verticals*flips)
    stereo = np.matmul(np.diag(rescales[:,0]),prestereo)
    return stereo

# undoes stereographic projection  mapping the disc to the upper hemisphere.
def stereo_pop(coords):
    (numpts,dim) = coords.shape
    Xnew = np.zeros([numpts,dim+1])
    for i in range(numpts):
        x = coords[i,:]
        r = 2/(1+np.linalg.norm(x)**2)
        Xnew[i,:-1] = r*x
        Xnew[i,-1] = r-1
    return Xnew
