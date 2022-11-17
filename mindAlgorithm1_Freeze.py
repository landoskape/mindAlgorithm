# code to implement the mind algorithm from the following paper:
# paper: https://www.biorxiv.org/content/10.1101/418939v2.full.pdf

###############################################################################
# Discussion of goals and ideas (in Andrew speak rather than methods speak)
# Creating a Forest PPCA model
# - We start with all the data as a root node
# ------- Note: they preprocess with PCA, using the same 95% trick to reduce computation time (then project everything back at the end...)
# - At each node, we subdivide data into left/right child nodes using a decision hyperplane that maximizes net likelihood of the PPCA models. For each of these children, we fit a PPCA model.    
# ------- choosing a hyperplane: we start with nDir possible splitting directions on a unit sphere (serving to decorrelate ensemble)
# -------                        then we choose a split threshold such that "the mean successor state for each child node best predicts the actual successor states, as measured by the squared error"
# -------                        once a threshold has been chosen, full PPCA models are fit to each leaf. 
# -------                        finally, we compare the total PPCA likelihood for each nDir, evaluated at the best threshold, and choose the direction with the highest likelihood
# -------                        --> we continue this process until each leaf node has no fewer than nLeaf data points
# ------- choosing nDir and nLeaf: they chose nLeaf to be 40. They did it in an ad-hoc manner. (nLeaf must be large enough to fit a PPCA model at least...)
# -------                          they chose nDir to be 2... same thing, ad-hoc 
# - Ensembles: 
# ------- they use an ensemble approach where they fit S trees, and then combine the results of each tree by using the median probability of the output for each network transition
#
# optimizeHyperplane:
# -----    as input, this function requires a data array of the current neural activity along with an aligned successor array of the activity in the following timestep 
# -----    first, we split the data array in two, using one of the proposed hyperplane directions, then pick the best decision threshold for that hyperplane using an isotropic gaussian model
# -----    -----, note that this is constrained by the minimum number of datapoints in each leaf, set by the variable nLeaf (therefore, a split is not required if the number of datapoints is <2*nLeaf
# -----    -----, can either do this with efficient running sums (Torgo, 1999), or by choosing 10-20 candidates (divided along quantiles, set by nQuant) and choosing the best one 
# -----    second, once a threshold is chosen for each direction, we fit a full ppca model for the relevant successor states
# -----    (we do this for each of the proposed hyperplane directions, where the number of proposals is set by the variable nDir)
# -----    third, we sum the likelihoods of the PPCA models for each pair of left and right leafs
# -----    finally, we pick the hyperplane with the highest sum likelihood for the PPCA models of left and right leafs 
# -----    --------------------------
# -----    the output of the function is the following:
# -----    - hypDir,hypThreshold: the direction and threshold defining a decision hyperplane
# -----    - uLeft,covLeft,uRight,covRight: the mean, covariance, and noise of the PPCA model for left and right leaf nodes
# -----    - ----- NOTE: we don't really need the eigenvalues and eigenvectors of the PPCA model because they aren't required to measure the probability of a data point
# -----    - leftLikelihood, rightLikelihood: the likelihood of the PPCA model for the left and right leaf in the model
# -----    - the number of datapoints in each leaf

# inclusions
import numpy as np
import numba as nb
import scipy
from scipy.spatial.distance import squareform, pdist, cdist
from sklearn.manifold import MDS as skMDS
from sklearn_extra.cluster import KMedoids
from tqdm.notebook import tqdm, trange

# --------------------------------------------
# function library: random forest construction
# --------------------------------------------
def constructForest(currentData, successorData, numTrees=5, nDir=2, nLeaf=40, nQuant=10, keepSummaryData=False):
    forest = {}
    forest['numTrees'] = numTrees
    forest['nDir'] = nDir
    forest['nLeaf'] = nLeaf
    forest['nQuant'] = nQuant
    forest['tree'] = []
    forest['nDims'] = currentData.shape[0] # dimensionality of data
    
    progressBar = tqdm(range(numTrees))
    for tt in progressBar:
        progressBar.set_description(f'Fitting PPCA Tree {tt+1}/{numTrees}')
        if keepSummaryData:
            forest['tree'].append(splitForestNodeWithSummary(currentData,successorData,ppcaModel={},nDir=nDir,nLeaf=nLeaf,nQuant=nQuant))
        else:
            forest['tree'].append(splitForestNode(currentData,successorData,ppcaModel={},nDir=nDir,nLeaf=nLeaf,nQuant=nQuant))
    
    return forest

def addTrees(currentData, successorData, forest, numTrees):
    if currentData.shape[0] != forest['nDims']:
        raise ValueError("Current data doesn't have same number of dimensions as other data in previous trees")
        
    forest['numTrees'] = forest['numTrees'] + numTrees
    
    progressBar = tqdm(range(numTrees))
    for newTree in progressBar:
        progressBar.set_description(f'Adding PPCA Tree {newTree+1}/{numTrees}')
        forest['tree'].append(splitForestNode(currentData,successorData,ppcaModel={},nDir=forest['nDir'],nLeaf=forest['nLeaf'],nQuant=forest['nQuant']))
    return forest

def splitForestNode(currentData, successorData, ppcaModel={}, nDir=2, nLeaf=40, nQuant=10):
    # split a node in the dataset
    # Nodes are defined as:
    # - an ndarray of currentData and successorData (same shape and aligned)
    # - a dictionary with the PPCA model for the current & successor data
    # - ---- - if the dictionary is empty, assume it's the root node. if it is also a terminal node (by nLeaf), then generate an error!
    # Return node dictionary
    
    # Define node dictionary to be returned
    node = {}
    
    # if ppcaModel is empty, then this is the root node
    rootNode = (ppcaModel=={})
    
    # important variables for function
    N,D = currentData.shape # N=number of neurons, D=number of datapoints (in this node)
    
    # If this can't be split according to parameters, then return node!
    if D < 2*nLeaf:
        # If it has too few points to be split, and no PPCA model is provided, this function assumes it's a root node. But these are incompatible!
        if rootNode:
            raise ValueError("Too few data points to split, but no PPCA model provided (it is a terminal and a root node)")
        # If it has too few points to be split, but has a PPCA model, then it is a terminal node. Return PPCA model, identify it as a terminal node, and stop.
        if not checkValidPPCA(ppcaModel):
            raise ValueError("Terminal node identified, but no PPCA model provided")
        # Add PPCA model to node, then return
        node['terminalNode'] = True
        node['mean'] = ppcaModel['mean']
        node['covariance'] = ppcaModel['covariance']
        node['invcov'] = np.linalg.inv(ppcaModel['covariance'])
        node['logdetcov'] = 2*np.log(np.prod(np.diag(np.linalg.cholesky(ppcaModel['covariance']))))
        node['likelihood'] = ppcaModel['likelihood']
        return node
    
    # If this node is being split, find the best split direction
    node['terminalNode'] = False
    hyperplane, leftNode, rightNode, leftData, leftSuccessor, rightData, rightSuccessor = optimizeHyperplane(currentData, successorData, nDir=nDir, nLeaf=nLeaf, nQuant=nQuant)
    
    # Add hyperplane to node dictionary
    node['hyperplane'] = hyperplane
    
    # Return left and right leaves of node (recursively create tree, this function will call itself until it returns a terminal node)
    node['left'] = splitForestNode(leftData, leftSuccessor, ppcaModel=leftNode, nDir=nDir, nLeaf=nLeaf, nQuant=nQuant)
    node['right'] = splitForestNode(rightData, rightSuccessor, ppcaModel=rightNode, nDir=nDir, nLeaf=nLeaf, nQuant=nQuant)
    return node

def optimizeHyperplane(currentData, successorData, nDir=2, nLeaf=40, nQuant=10):
    # choosing a hyperplane to create a decision boundary
    # we aim to model the successor states of our left and right leaf nodes with a multivariate gaussian.
    # outputs hyperplane separating data (as a dictionary) and left/right current and successor data
    
    if currentData.ndim != 2 or successorData.ndim != 2:
        raise ValueError("data must be a matrix")
    
    if currentData.shape != successorData.shape:
        raise ValueError("current data and successor data must have same shape")
    
    # important variables for function
    N,D = currentData.shape # N=number of neurons, D=number of datapoints (in this node)
    
    # Confirm that this data can be split (shouldn't even make it here if it can't, but always good to check!)
    if D < 2*nLeaf:
        raise ValueError("optimizeHyperplane received data with too few datapoints to split!!")
    
    # Prepare splitting procedure (use quantile speedup trick for decision thresholds)
    quantPoints = np.linspace(0,1,nQuant+2)[1:-1]
    if (D < 3*nLeaf) and (0.5 not in quantPoints): 
        # make sure we include the middle in case a 50/50 split is the only way
        quantPoints = np.concatenate((quantPoints,[0.5]))
    
    # Generate candidate hyperplane directions on unit hypersphere
    hypDirs = np.zeros((N,nDir)) # do it this way to avoid annoyance, it's very much not the bottleneck of the pipeline 
    while np.any(np.sum(hypDirs,axis=0) == 0): hypDirs = np.random.normal(0,1,(N,nDir))
    hypDirs = hypDirs / np.sqrt(np.sum(hypDirs**2,axis=0))
    
    # Preallocate variables
    hypThresholds = np.zeros(nDir)
    llCandidate = np.zeros((nDir,2))
    uCandidate = np.zeros((nDir,N,2))
    covCandidate = np.zeros((nDir,N,N,2))
    for ndir in range(nDir):
        cProjection = currentData.T @ hypDirs[:,ndir]
        cQuantiles = np.quantile(cProjection, quantPoints)
        ssError = []
        for cThreshold in cQuantiles:
            # Do isotropic gaussian model first
            idxLeft = np.where(cProjection <= cThreshold)[0]
            idxRight = np.where(cProjection > cThreshold)[0]
            if len(idxLeft)<nLeaf or len(idxRight)<nLeaf:
                ssError.append(np.inf)
                continue
            cMeanLeft = np.mean(successorData[:,idxLeft],axis=1,keepdims=True)
            cMeanRight = np.mean(successorData[:,idxRight],axis=1,keepdims=True)
            cDevLeft = np.sum((successorData[:,idxLeft] - cMeanLeft)**2)
            cDevRight = np.sum((successorData[:,idxRight] - cMeanRight)**2)
            ssError.append(cDevLeft+cDevRight)
            if (len(idxLeft) < nLeaf) or (len(idxRight) < nLeaf):
                print(f"Left: {len(idxLeft)}, Right: {len(idxRight)}")
                print(f"Quantiles: {quantPoints}, quantPoints * numData: {quantPoints*D}")
                raise ValueError("Quantization wrong in optimizeHyperplane, one leaf ended up with too few datapoints")
        
        # Then, for the best isotropic fit, compute a full ppca model
        idxBestThreshold = np.argmin(ssError)
        idxLeft = np.where(cProjection <= cQuantiles[idxBestThreshold])[0]
        idxRight = np.where(cProjection > cQuantiles[idxBestThreshold])[0]
        
        hypThresholds[ndir] = cQuantiles[idxBestThreshold]
        llCandidate[ndir,0],uCandidate[ndir,:,0],covCandidate[ndir,:,:,0] = ppca(successorData[:,idxLeft].T)[0:3]
        llCandidate[ndir,1],uCandidate[ndir,:,1],covCandidate[ndir,:,:,1] = ppca(successorData[:,idxRight].T)[0:3]
    
    # Find optimal direction, return indices for left and right data
    totalLikelihood = np.sum(llCandidate,axis=1)
    idxHyperplane = np.argmax(totalLikelihood)
    bestProjection = currentData.T @ hypDirs[:,idxHyperplane]
    idxLeft = np.where(bestProjection <= hypThresholds[idxHyperplane])[0]
    idxRight = np.where(bestProjection > hypThresholds[idxHyperplane])[0]
    
    # Save optimal hyperplane to dictionary
    hyperplane = {}
    hyperplane['direction'] = hypDirs[:,idxHyperplane]
    hyperplane['threshold'] = hypThresholds[idxHyperplane]
    # Save left node parameters to dictionary
    leftNode = {}
    leftNode['mean'] = uCandidate[idxHyperplane,:,0]
    leftNode['covariance'] = covCandidate[idxHyperplane,:,:,0]
    leftNode['likelihood'] = llCandidate[idxHyperplane,0]
    # Save right node parameters to dictionary
    rightNode = {}
    rightNode['mean'] = uCandidate[idxHyperplane,:,1]
    rightNode['covariance'] = covCandidate[idxHyperplane,:,:,1]
    rightNode['likelihood'] = llCandidate[idxHyperplane,1]
    
    return hyperplane, leftNode, rightNode, currentData[:,idxLeft], successorData[:,idxLeft], currentData[:,idxRight], successorData[:,idxRight]


def checkValidPPCA(node,withInverse=False):
    # Take in dictionary, check if it contains a valid PPCA model
    if withInverse:
        # First check if keys exist
        if ('mean' in node) and ('covariance' in node) and ('likelihood' in node) and ('invcov' in node):
            # Then check if they have the right dimensions 
            if (node['mean'].ndim == 1) and (node['covariance'].ndim == 2) and (node['likelihood'].ndim==0) and (node['invcov'].ndim==2) and (node['covariance'].shape == node['invcov'].shape):
                # Then check if the shapes match
                if node['mean'].shape[0] == node['covariance'].shape[0] == node['covariance'].shape[1] == node['invcov'].shape[0] == node['invcov'].shape[1]:
                    # Only then, return True
                    return True
    else: 
        # First check if keys exist
        if ('mean' in node) and ('covariance' in node) and ('likelihood' in node):
            # Then check if they have the right dimensions 
            if (node['mean'].ndim == 1) and (node['covariance'].ndim == 2) and (node['likelihood'].ndim==0):
                # Then check if the shapes match
                if node['mean'].shape[0] == node['covariance'].shape[0] == node['covariance'].shape[1]:
                    # Only then, return True
                    return True
    # Otherwise, this isn't a valid PPCA model
    return False

def summarizeForest(forest, removeOriginal=False):
    """
    This function supports optimization by summarizing a random PPCA forest into easily accessible arrays. 
    The original forest of hyperplanes will remain the same (don't know how to optimize), but at the terminal node, the PPCA model will be replaced with a predefined index.
    Data will be passed through the reduced forest and each datapoint will be associated to an index. 
    Then, the data will be passed through the appropriate PPCA model using it's associated index.
    
    removeOriginal (default=False) removes the PPCA model from the original tree structures. They usually take up little memory, so I think it's not necessary to remove them. 
    """
    if removeOriginal: print("NOTE: removeOriginal was set to True, but it hasn't been coded yet!!!")
    
    numTrees = len(forest['tree'])
    forest['treeSummary'] = []
    forest['ppcaMeans'] = []
    forest['ppcaCovs'] = []
    forest['ppcaInvCovs'] = []
    forest['ppcaLogDets'] = []
    
    for tt, tree in enumerate(forest['tree']):
        treeStructure = returnTreeStructure(tree) # Measure tree structure (all possible paths to terminal nodes)
        numPaths = len(treeStructure) 
        cPpcaMeans = np.zeros((numPaths,forest['nDims']))
        cPpcaCovs = np.zeros((numPaths,forest['nDims'],forest['nDims']))
        cPpcaInvCovs = np.zeros((numPaths,forest['nDims'],forest['nDims']))
        cPpcaLogDets = np.zeros(numPaths)
        for pp in range(numPaths):
            cPathKeys = returnDecPath(treeStructure[pp]) # For each path, create a list of strings describing path
            updateNestedTree(tree, cPathKeys, 'pathIdx', pp) # Add a pathIdx value corresponding to this path
            cPpcaMeans[pp,:] = returnNestedTree(tree, cPathKeys, 'mean') # Return the mean, covariance, and inverse covariance for this path along the tree
            cPpcaCovs[pp,:,:] = returnNestedTree(tree, cPathKeys, 'covariance')
            cPpcaInvCovs[pp,:,:] = returnNestedTree(tree, cPathKeys, 'invcov')
            cPpcaLogDets[pp] = returnNestedTree(tree, cPathKeys, 'logdetcov')
            
        # Add ppca model for each path to the top of the forest, along with the tree structure
        forest['treeSummary'].append(treeStructure)
        forest['ppcaMeans'].append(cPpcaMeans)
        forest['ppcaCovs'].append(cPpcaCovs)
        forest['ppcaInvCovs'].append(cPpcaInvCovs)
        forest['ppcaLogDets'].append(cPpcaLogDets)

def returnTreeStructure(tree):
    # Returns tree structure - in form of list of lists, indicating all possible paths within tree
    # 0:left, 1:right
    treeIdx = 0
    treeStructure = []
    # initialize with all left 
    cTreePath = []
    cNode = tree
    while True:
        if cNode['terminalNode']: break
        cNode = cNode['left']
        cTreePath.append(0)
    treeStructure.append(cTreePath)
    # continue
    while True:
        # The code is setup so the last path is all right choices...
        if all([c==1 for c in treeStructure[-1]]): break
        # Otherwise, reset current node, initialize cTreePath, and extend path
        cNode = tree
        leftSwitch = False
        cTreePath = []
        while True:
            if cNode['terminalNode']: break # leave if we've reached the end

            if leftSwitch:
                cNode = cNode['left']
                cTreePath.append(0)
            
            elif len(cTreePath) < len(treeStructure[-1])-1:
                # If we're more than 2 away from end of previous, copy it's next decision
                #if treeStructure[-1][len(cTreePath)+1]==0:
                if all([t==1 for t in treeStructure[-1][1+len(cTreePath):]]):
                    cNode = cNode['right']
                    leftSwitch = True
                    cTreePath.append(1)
                else:
                    if treeStructure[-1][len(cTreePath)]==0:
                        cNode = cNode['left']
                        cTreePath.append(0)
                    else:
                        cNode = cNode['right']
                        cTreePath.append(1)
            elif len(cTreePath) == len(treeStructure[-1])-1 and (cTreePath==treeStructure[-1][:len(cTreePath)]):
                # If we're one away from previous and all else is the same, go right
                cNode = cNode['right']
                cTreePath.append(1)
            elif len(cTreePath) >= len(treeStructure[-1])-1:
                # If we're past the previous one, just go left
                cNode = cNode['left']
                cTreePath.append(0)
            else:
                raise ValueError("there was an error in the logic... sorry")
        treeStructure.append(cTreePath)
    return treeStructure


def returnDecPath(decValues):
    decPath = [{0:'left', 1:'right'}[cdecision] for cdecision in decValues]
    return decPath

def returnPathIndexLoop(cdata, tree):
    ND = cdata.shape[1]
    pathIndices = np.empty(ND,dtype=int)
    for nd in range(ND):
        pathIndices[nd] = returnPathIndex(cdata[:,nd],tree)
    return pathIndices

def returnPathIndex(cdata, tree):
    if not tree['terminalNode']: 
        # Determine appropriate leaf and go to next level
        cproj = tree['hyperplane']['direction'].reshape(1,-1) @ cdata.reshape(-1,1)
        if cproj <= tree['hyperplane']['threshold']:
            return returnPathIndex(cdata,tree['left'])
        else:
            return returnPathIndex(cdata,tree['right'])
    else:
        if not "pathIdx" in tree:
            raise ValueError("TerminalNode found, but didn't contain an index to this tree path. Summarize tree first!")
        return tree['pathIdx']

def updateNestedTree(tree, decPath, key, value):
    # used to dynamically set value of nested dictionary (here, tree=dictioanry, decPath=list of keys,key=final key, value=value to change or set)
    for dec in decPath:
        tree = tree.setdefault(dec)
    tree[key] = value
    
def returnNestedTree(tree, decPath, key):
    # used to dynamically return value of nested dictionary (here, tree=dictioanry, decPath=list of keys, key=final key)
    for dec in decPath:
        tree = tree.setdefault(dec)
    return tree.setdefault(key)


# ----------------------------------------------
# function library: measure probability with forest (using summarized trees!!!)
# ----------------------------------------------
def probabilityToDistance(probVector,minCutoff=0):
    # Convert probability to distance (with dist = sqrt(-log(p)))
    # But avoiding nonsensical computation on p=0 (set this distance to infinity)
    distVector = np.where(probVector > minCutoff, probVector, np.inf)
    np.log(distVector, out=distVector, where=distVector<np.inf)
    np.sqrt(-distVector, out=distVector, where=distVector<np.inf)
    return np.abs(distVector)
    
def smartGridProbability(data, forest):
    # Compute probability of transition from each point in data to each point in data using PPCA models in forest
    # Automatically use log switch for computing probability due to numerical stability
    # Uses summarized tree structure 
    if (data.ndim != 2): raise ValueError("Data must be a matrix")
    D,N = data.shape
    
    numTrees = forest['numTrees']
    probability = -1*np.ones((N,N,numTrees))
    progressBar = tqdm(range(numTrees))
    for tt in progressBar:
        progressBar.set_description(f"Measuring probability in tree {tt+1}/{numTrees} -- comparing {N} points to {N} points.")
        tPathIdx = returnPathIndexLoop(data, forest['tree'][tt])
        tPpcaMean = forest['ppcaMeans'][tt][tPathIdx,:]
        tPpcaInvCov = forest['ppcaInvCovs'][tt][tPathIdx,:,:]
        tPpcaLogDet = forest['ppcaLogDets'][tt][tPathIdx]
        for n in range(N):
            cudata = data - tPpcaMean[n,:].reshape(D,1)
            cexparg = (-1/2) * np.sum((tPpcaInvCov[n,:,:] @ cudata) * cudata,axis=0)
            cloglikelihood = -D/2*np.log(2*np.pi) - 1/2*tPpcaLogDet[n] + cexparg
            probability[n,:,tt] = np.exp(cloglikelihood)
    return np.median(probability,axis=2)

def numbaGridProbability(data, forest):
    # Compute probability of transition from each point in data to each point in data using PPCA models in forest
    # Automatically use log switch for computing probability due to numerical stability
    # Uses summarized tree structure 
    if (data.ndim != 2): raise ValueError("Data must be a matrix")
    D,N = data.shape
    
    cdata = np.ascontiguousarray(data.T)
    
    numTrees = forest['numTrees']
    probability = -1*np.ones((N,N,numTrees))
    progressBar = tqdm(range(numTrees))
    for tt in progressBar:
        progressBar.set_description(f"Measuring probability in tree {tt+1}/{numTrees} -- comparing {N} points to {N} points.")
        tPathIdx = returnPathIndexLoop(data, forest['tree'][tt])
        tPpcaMean = forest['ppcaMeans'][tt][tPathIdx,:]
        tPpcaInvCov = forest['ppcaInvCovs'][tt][tPathIdx,:,:]
        tPpcaLogDet = forest['ppcaLogDets'][tt][tPathIdx]
        probability[:,:,tt] = parallelLikelihoodForGrid(cdata, tPpcaMean, tPpcaInvCov, tPpcaLogDet, D)
    return np.median(probability,axis=2)

@nb.njit(nb.float64[:,::1](nb.float64[:,::1],nb.float64[:,::1],nb.float64[:,:,::1],nb.float64[::1],nb.float64), nogil=True, parallel=True)
def parallelLikelihoodForGrid(data, u, iS, logDet, D): 
    N = data.shape[0] 
    probability = np.zeros((N,N))
    for n in nb.prange(N):
        udata = np.zeros_like(data)
        # subtract mean
        for n1 in range(N):
            udata[n1,:] = data[n1,:] - u[n,:]
        cexpargmat = (udata @ iS[n]) * udata
        cexparg = np.zeros_like(logDet)
        for n1 in range(N):
            cexparg[n1] = np.sum(cexpargmat[n1,:])
        cloglikelihood = -D/2*np.log(2*np.pi) - logDet[n]/2 - cexparg/2
        probability[n,:] = np.exp(cloglikelihood)
    return probability

def smartForestLikelihood(cdata, sdata, forest):
    N = cdata.shape[1]
    D = forest['nDims']
    T = forest['numTrees']
    probability = np.empty((N,T))
    for tt in range(T):
        # For each tree, start by returning path index of each datapoint
        tPathIdx = returnPathIndexLoop(cdata, forest['tree'][tt])
        probability[:,tt] = fastLikelihood(sdata, forest['ppcaMeans'][tt][tPathIdx,:], forest['ppcaInvCovs'][tt][tPathIdx,:,:], forest['ppcaLogDets'][tt][tPathIdx], D)
    return np.median(probability,axis=1)

def fastLikelihood(data, u, iS, logDet, D): 
    udata = data - u.T
    exparg = -1/2 * np.einsum('md,mnd,nd->d',udata, iS.transpose(1,2,0), udata)
    logLikelihood = -D/2*np.log(2*np.pi) - logDet/2 + exparg
    return np.exp(logLikelihood)

def scalarLikelihood(data, u, iS, logDet, D): 
    if data.ndim != 1 or iS.ndim != 2: 
        raise ValueError("Only works for single data vector (produces single scalar output)")
    udata = data - u
    expArgument = -(1/2) * (udata.reshape(1,-1) @ iS @ udata.reshape(-1,1))
    logLikelihood = -D/2*np.log(2*np.pi) - 1/2*logDet + expArgument
    return np.exp(logLikelihood)


# --------------------------------------------
# function library: multidimensional scaling
# --------------------------------------------
def initMDS(distmat,dims,method='fast'):
    if method=='fast':
        # Create double-centered squared distance matrix
        N = distmat.shape[0]
        D2 = distmat**2
        C = np.identity(N) - (1/N)*np.ones((N,N))
        B = (-1/2) * (C @ D2 @ C)
        
        # Do eigendecomposition on B
        w,v = scipy.linalg.eigh(B)
        idx = np.argsort(-w) # return index of descending sort
        w = w[idx[:dims]] # sort eigenvalues, only keep requested ones
        v = v[:,idx[:dims]] # sort eigenvectors, only keep requested ones
        return v @ np.diag(np.sqrt(w))

    elif method=='metric':
        # Return cMDS solution
        mdsEmbedding = skMDS(n_components=dims, dissimilarity='precomputed')
        return mdsEmbedding.fit_transform(distmat)
    
    else:
        raise ValueError("Didn't recognize method, must be method='fast' or method='metric'")

def sammonMapping(distmat, coord, iterations=1000, alpha=0.1):
    # distmat is an NxN matrix of distances
    # coord is an NxD matrix of initial coordinates to be aligned to the distmat using sammon mapping.
    # iterations is the number of iterations to use (might adjust if I change to conjugate gradient descent)
    # alpha is how much to scale each update each iteration
    N,D = coord.shape
    error = -1*np.ones(iterations)
    distvec = squareform(distmat)
    
    errorConstant = 1/np.sum(distvec)
    
    progressBar = tqdm(range(iterations))
    for it in progressBar:
        progressBar.set_description(f"Sammon Mapping Optimization: Iteration {it+1}/{iterations}")

        cdistvec = pdist(coord)
        cdistmat = squareform(cdistvec)

        gradient = getGradient(distmat, cdistmat, coord, errorConstant)
        dblgrad = getDoubleGradient(distmat, cdistmat, coord, errorConstant)
        coord = coord - alpha * gradient / np.linalg.norm(dblgrad)
        cdistvec = pdist(coord)
        error[it] = sammonError(distvec, cdistvec)        
    
    if coord.ndim==1:
        coord = coord.reshape(-1,1)
    return coord, error

@nb.njit(nb.float64[:,:](nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64), nogil=True, parallel=True)
def getGradient(distmat, cdistmat, coord, errorConstant):
    # Fast method for acquiring gradient in sammonMapping
    gradient = np.zeros_like(coord)
    for n in nb.prange(coord.shape[0]):
        for d in range(coord.shape[1]):
            for j in range(coord.shape[0]):
                if j!=n:
                    gradient[n,d] += -2/errorConstant * (distmat[n,j] - cdistmat[n,j])/(distmat[n,j]*cdistmat[n,j])*(coord[n,d] - coord[j,d])
    return gradient

@nb.njit(nb.float64[:,:](nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64), nogil=True, parallel=True)
def getDoubleGradient(distmat, cdistmat, coord, errorConstant):
    # Fast method for acquiring double gradient in sammonMapping
    doubleGradient = np.zeros_like(coord)
    for n in nb.prange(coord.shape[0]):
        for d in range(coord.shape[1]):
            for j in range(coord.shape[0]):
                if j!=n:
                    doubleGradient[n,d] += -2/errorConstant * (1/(distmat[n,j]*cdistmat[n,j]))*((distmat[n,j]-cdistmat[n,j]) - \
                        (coord[n,d] - coord[j,d])**2/cdistmat[n,j]*(1+(distmat[n,j]-cdistmat[n,j])/cdistmat[n,j]))
    return doubleGradient

@nb.njit(nb.float64(nb.float64[:],nb.float64[:]), nogil=True, parallel=True)
def sammonError(distvec, cdistvec):
    return np.sum((distvec - cdistvec)**2/distvec) / np.sum(distvec)

def centerSammonCoordinates(sammonCoord):
    N,D = sammonCoord.shape
    if D==1: return sammonCoord
    uCoord = np.mean(sammonCoord, axis=0)
    msCoord = sammonCoord - uCoord
    covCoord = np.cov(msCoord.T, bias=True)
    w,v = scipy.linalg.eigh(covCoord)
    idx = np.argsort(-w) # return index of descending sort
    w = w[idx] # sort eigenvalues
    v = v[:,idx] # sort eigenvectors
    return sammonCoord @ v


# --------------------------------------------
# function library: forward and reverse mapping
# --------------------------------------------
def mapping(data, scaffold, target, k, lam):
    # data is an N x D1 array of datapoints to acquire weights for
    # scaffold is an M x D1 array of scaffold points to approximate with
    # target is an M x D2 array of target points associated with each scaffold point on a different manifold
    if data.ndim == 1: data = data.reshape(1,-1)
    ND,DD = data.shape
    MS,DS = scaffold.shape
    MT,DT = target.shape
    assert DD==DS, "data and scaffold must have same number of dimensions"
    assert MS==MT, "scaffold and target must contain the same number of datapoints"
    idx = np.argsort(cdist(data, scaffold), axis=1)[:,:k]
    datamap = np.empty((ND,DT))
    for nd in range(ND):
        cgvec = data[nd,:].reshape(1,-1) - scaffold[idx[nd,:],:]
        cw = mapWeights(cgvec,k,lam)
        datamap[nd,:] = np.dot(cw,target[idx[nd,:],:])
    return datamap

@nb.njit(fastmath=True)
def mapWeights(gvec, k, lam):
    gmat = gvec @ gvec.T
    w = np.linalg.inv(gmat + lam*np.identity(k)) @ np.ones(k)
    w = w / np.sum(w)
    return w
    
# --------------------------------------------
# function library: ppca model
# --------------------------------------------
def ppca(data, minVariance=0.95, weights=None):
    # probabilistic ppca - using description in Methods Section 1.7 of the following: https://www.biorxiv.org/content/10.1101/418939v2.full.pdf
    # data is a (observations x dimensions) array 
    # minVariance defines minimum fraction of variance required for fitting the PPCA model
    # if weights provided, then perform weighted ppca 
    
    if data.ndim != 2:
        raise ValueError("data must be a matrix")
    
    N,D = data.shape
    
    useEig = False
    if weights is None:
        useEig = True
        weights = np.ones(N)
    else:
        if (weights.ndim != 1) or (len(weights) != N):
            raise ValueError("weights must be a 1D vector with the same number of elements as the rows of data!")
    
    # Return ML estimate of mean
    uML = np.average(data,axis=0,weights=weights)
    cdata = data - uML # use centered data for computations
    
    # If weights are provided, use eigendecomposition because weighted SVD exists but isn't fast
    # Otherwise, pick method based on computational speed (logic inherited from Low/Lewallen, haven't tested yet!)
    if useEig or (N > D):
        # do eigendecomposition
        covData = np.cov(cdata.T, bias=True, aweights = weights)
        w,v = scipy.linalg.eigh(covData)
        w[w<=np.finfo(float).eps]=np.finfo(float).eps # don't allow weird tiny numbers (or negatives, it's a symmetric positive semidefinite matrix)
        idx = np.argsort(-w) # return index of descending sort
        w = w[idx] # sort eigenvalues
        v = v[:,idx] # sort eigenvectors
        s = np.sqrt(N*w) # singular values
        
    else:
        # do svd instead
        _,s,v = np.linalg.svd(cdata)
        v = v.T
        w = s**2 / N # eigenvalues
    
    varExplained = np.cumsum(w / np.sum(w))
    q = int(np.where(varExplained >= minVariance)[0][0])+1
    
    # Return ML estimate of noise variance
    nvML = np.mean(w[q:]) 
    
    # Keep q eigenvalues & eigenvectors
    w = w[:q]
    v = v[:,:q]
    
    # Compute ML estimate of covariance
    covML = nvML*np.identity(D) + (v @ (np.diag(w) - nvML*np.identity(q)) @ v.T)
    invCovML = np.linalg.inv(covML)
    
    # Return likelihood
    smartLogDet = 2*np.log(np.prod(np.diag(np.linalg.cholesky(covML))))
    likelihood = -N*D/2*np.log(2*np.pi) - N/2*smartLogDet - (1/2)*np.sum(np.array([cdata[n,:] @ invCovML @ cdata[n,:].T for n in range(N)]))

    return likelihood,uML,covML,nvML,w,v


# --------------------------------------------
# function library: data management
# --------------------------------------------
def returnLandmarkPoints(data, numLandmark=2000, numSeed=10, algorithm='greedy'):    
    # K-Medioids Algorithm (takes too long for big datasets!)
    if algorithm=='kmedioids':
        kmedoids = KMedoids(n_clusters=numLandmark).fit(data.T)
        return kmedoids.cluster_centers_.T, kmedoids.medoid_indices_
    
    # Greedy algorithm: http://graphics.stanford.edu/courses/cs468-05-winter/Papers/Landmarks/Silva_landmarks5.pdf
    if algorithm=='greedy':
        N = data.shape[1]
        l = np.zeros(numLandmark,dtype=int)
        l[:numSeed] = np.random.choice(N,numSeed,replace=False)
        m = np.min(scipy.spatial.distance.cdist(data[:,l[:numSeed]].T,data.T,metric='Euclidean'),axis=0)

        for ii in range(numSeed,numLandmark):
            # Set next landmark to datapoint furthest from other datapoints
            l[ii] = np.argmax(m)
            m = np.minimum(m, np.sum((data[:,l[ii]].reshape(-1,1) - data)**2,axis=0))
        return data[:,l],l

    raise ValueError("algorithm didn't match any of the available options...")


# --------------------------------------------
# function library: debugging type functions (not efficient, but useful for checking the algorithm after running it) 
# --------------------------------------------
def splitForestNodeWithSummary(currentData, successorData, ppcaModel={}, nDir=2, nLeaf=40, nQuant=10):
    # split a node in the dataset
    # Nodes are defined as:
    # - an ndarray of currentData and successorData (same shape and aligned)
    # - a dictionary with the PPCA model for the current & successor data
    # - ---- - if the dictionary is empty, assume it's the root node. if it is also a terminal node (by nLeaf), then generate an error!
    # Return node dictionary
    
    # Define node dictionary to be returned
    node = {}
    
    # if ppcaModel is empty, then this is the root node
    rootNode = (ppcaModel=={})
    
    # important variables for function
    N,D = currentData.shape # N=number of neurons, D=number of datapoints (in this node)
    
    # If this can't be split according to parameters, then return node!
    if D < 2*nLeaf:
        # If it has too few points to be split, and no PPCA model is provided, this function assumes it's a root node. But these are incompatible!
        if rootNode:
            raise ValueError("Too few data points to split, but no PPCA model provided (it is a terminal and a root node)")
        # If it has too few points to be split, but has a PPCA model, then it is a terminal node. Return PPCA model, identify it as a terminal node, and stop.
        if not checkValidPPCA(ppcaModel):
            raise ValueError("Terminal node identified, but no PPCA model provided")
        node['terminalNode'] = True
        node['mean'] = ppcaModel['mean']
        node['covariance'] = ppcaModel['covariance']
        node['invcov'] = np.linalg.inv(ppcaModel['covariance'])
        node['likelihood'] = ppcaModel['likelihood']
        return node
    
    # If this node is being split, find the best split direction
    node['terminalNode'] = False
    hyperplane, leftNode, rightNode, leftData, leftSuccessor, rightData, rightSuccessor, meanModelError, ppcaLikelihoodQuants = optimizeHyperplaneWithSummary(currentData, successorData, nDir, nLeaf, nQuant)
    
    # Add hyperplane to node dictionary
    node['hyperplane'] = hyperplane
    node['meanModelError'] = meanModelError
    node['ppcaLikelihoodQuants'] = ppcaLikelihoodQuants
    
    # Return left and right leaves of node (recursively create tree, this function will call itself until it returns a terminal node)
    node['left'] = splitForestNodeWithSummary(leftData, leftSuccessor, ppcaModel=leftNode, nDir=nDir, nLeaf=nLeaf, nQuant=nQuant)
    node['right'] = splitForestNodeWithSummary(rightData, rightSuccessor, ppcaModel=rightNode, nDir=nDir, nLeaf=nLeaf, nQuant=nQuant)
    return node


def optimizeHyperplaneWithSummary(currentData, successorData, nDir=2, nLeaf=40, nQuant=4):
    # choosing a hyperplane to create a decision boundary
    # we aim to model the successor states of our left and right leaf nodes with a multivariate gaussian.
    # outputs hyperplane separating data (as a dictionary) and left/right current and successor data
    
    if currentData.ndim != 2 or successorData.ndim != 2:
        raise ValueError("data must be a matrix")
    
    if currentData.shape != successorData.shape:
        raise ValueError("current data and successor data must have same shape")
    
    # important variables for function
    N,D = currentData.shape # N=number of neurons, D=number of datapoints (in this node)
    
    # Confirm that this data can be split (shouldn't even make it here if it can't, but always good to check!)
    if D < 2*nLeaf:
        raise ValueError("optimizeHyperplane received data with too few datapoints to split!!")
    
    # Prepare splitting procedure (use quantile speedup trick for decision thresholds)
    numDatapointsPerQuant = D/nQuant
    quantOffset = int(np.ceil(nLeaf / (D/nQuant)))
    quantPoints = np.linspace(0,1,nQuant+1)[quantOffset:-quantOffset]
    if len(quantPoints)==0:
        # If there are two few points, just slice the data in half
        quantPoints = 0.5
    elif quantOffset>1:
        # If the offset is greater than 1, still query nQuant points! 
        quantPoints = np.linspace(quantPoints[0],quantPoints[-1],nQuant) 

    # Generate candidate hyperplane directions on unit hypersphere
    hypDirs = np.zeros((N,nDir)) # do it this way to avoid annoyance, it's very much not the bottleneck of the pipeline 
    while np.any(np.sum(hypDirs,axis=0) == 0): hypDirs = np.random.normal(0,1,(N,nDir))
    hypDirs = hypDirs / np.sqrt(np.sum(hypDirs**2,axis=0))
    
    # Preallocate variables
    hypThresholds = np.zeros(nDir)
    llCandidate = np.zeros((nDir,2))
    uCandidate = np.zeros((nDir,N,2))
    covCandidate = np.zeros((nDir,N,N,2))
    meanModelError = np.zeros((nDir, len(quantPoints)))
    ppcaLikelihoodQuants = np.zeros((nDir, len(quantPoints)))
    for ndir in range(nDir):
        cProjection = currentData.T @ hypDirs[:,ndir]
        cQuantiles = np.quantile(cProjection, quantPoints)
        ssError = []
        for ii,cThreshold in enumerate(cQuantiles):
            # Do isotropic gaussian model first
            idxLeft = np.where(cProjection <= cThreshold)[0]
            idxRight = np.where(cProjection > cThreshold)[0]
            cMeanLeft = np.mean(successorData[:,idxLeft],axis=1,keepdims=True)
            cMeanRight = np.mean(successorData[:,idxRight],axis=1,keepdims=True)
            cDevLeft = np.sum((successorData[:,idxLeft] - cMeanLeft)**2)
            cDevRight = np.sum((successorData[:,idxRight] - cMeanRight)**2)
            ssError.append(cDevLeft+cDevRight)
            meanModelError[ndir,ii] = cDevLeft + cDevRight
            ppcaLikelihoodQuants[ndir,ii] += ppca(successorData[:,idxLeft].T)[0]
            ppcaLikelihoodQuants[ndir,ii] += ppca(successorData[:,idxRight].T)[0]
        
        # Then, for the best isotropic fit, compute a full ppca model
        idxBestThreshold = np.argmin(ssError)
        idxLeft = np.where(cProjection <= cQuantiles[idxBestThreshold])[0]
        idxRight = np.where(cProjection > cQuantiles[idxBestThreshold])[0]
        
        hypThresholds[ndir] = cQuantiles[idxBestThreshold]
        llCandidate[ndir,0],uCandidate[ndir,:,0],covCandidate[ndir,:,:,0] = ppca(successorData[:,idxLeft].T)[0:3]
        llCandidate[ndir,1],uCandidate[ndir,:,1],covCandidate[ndir,:,:,1] = ppca(successorData[:,idxRight].T)[0:3]
    
    # Find optimal direction, return indices for left and right data
    totalLikelihood = np.sum(llCandidate,axis=1)
    idxHyperplane = np.argmax(totalLikelihood)
    bestProjection = currentData.T @ hypDirs[:,idxHyperplane]
    idxLeft = np.where(bestProjection <= cThreshold)[0]
    idxRight = np.where(bestProjection > cThreshold)[0]
    
    # Save optimal hyperplane to dictionary
    hyperplane = {}
    hyperplane['direction'] = hypDirs[:,idxHyperplane]
    hyperplane['threshold'] = hypThresholds[idxHyperplane]
    # Save left node parameters to dictionary
    leftNode = {}
    leftNode['mean'] = uCandidate[idxHyperplane,:,0]
    leftNode['covariance'] = covCandidate[idxHyperplane,:,:,0]
    leftNode['likelihood'] = llCandidate[idxHyperplane,0]
    # Save right node parameters to dictionary
    rightNode = {}
    rightNode['mean'] = uCandidate[idxHyperplane,:,1]
    rightNode['covariance'] = covCandidate[idxHyperplane,:,:,1]
    rightNode['likelihood'] = llCandidate[idxHyperplane,1]
    
    
    # -- just added index !!! --
    return hyperplane, leftNode, rightNode, currentData[:,idxLeft], successorData[:,idxLeft], currentData[:,idxRight], successorData[:,idxRight], meanModelError[idxHyperplane,:], ppcaLikelihoodQuants[idxHyperplane,:]