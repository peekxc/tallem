import numpy as np

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

## TODO: remove
def eucl_dist(p1,p2,extra = False):
	dist = np.linalg.norm(p2-p1)
	if not extra:
		return dist
	elif extra == True:
		return (dist,dist)
	elif extra == 2:
		flips = np.zeros(len(p1))
		return (dist,flips)

# TODO: remove
# yields (cover,assign,dists) correspondence between landmarks and points.
# def landmark_cover(input,landmarks,cover_rad,distfcn=eucl_dist):
# 	numpts = input.shape[0]
# 	if distfcn == 'precomputed':
# 			def DF(ii,jj):
# 					return input[ii,jj]
# 	else:
# 			def DF(ii,jj):
# 					return distfcn(input[ii,:],input[jj,:])
	
# 	# local information!
# 	cover = [[] for x in range(len(landmarks))]
# 	assign = [[] for x in range(numpts)]
# 	dists = [{} for x in range(numpts)]
	
# 	# find cover set residence and point landmark neighbors (transposes)
# 	for (index,ii) in tqdm(enumerate(landmarks)):
# 			for jj in range(numpts):
# 					dist = DF(ii,jj)
# 					if dist < cover_rad:
# 						cover[index] = cover[index]+[jj]
# 						assign[jj] = assign[jj]+[index]
# 						dists[jj][index] = dist
# 			cover[index] = np.array(cover[index])
# 	return (cover,assign,dists)