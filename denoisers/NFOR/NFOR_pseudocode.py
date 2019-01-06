# Pseudo-code of our algorithm, in python-like syntax.
# This code is for demonstration purposes only; our actual implementation is in C++11

# Input: Image half buffers `colorA`, `colorB`, with samples evenly split between them
#        Array of auxiliary feature half buffers `featuresA`, `featuresB`, with samples evenly split between them
#        Sample mean variance of combined color image (colorA + colorB)/2, `colorVar`
#        Array of sample mean variances of combined auxiliary feature buffers, `featureVar`
#
# All of these inputs are images of the same size.
# Arithmetic operations are assumed to operate on a per-pixel basis, i.e.
# (A + B)/2 computes the per-pixel average of two images A and B and returns another image
def filterMain(colorA, colorB, featuresA, featuresB, colorVar, featureVars):
    color = (colorA + colorB)/2

    # Feature cross-prefiltering (section 5.1)
    filteredFeaturesA = []
    filteredFeaturesB = []
    for i in range(len(featureVars)):
        # Cross filtering
        filteredFeaturesA.append(nlMeans(featuresA[i], featuresB[i], 2*featureVars[i], F=3, R=5, k=0.5)
        filteredFeaturesB.append(nlMeans(featuresB[i], featuresA[i], 2*featureVars[i], F=3, R=5, k=0.5)

    # Main regression (section 5.2)
    ks = [0.5, 1.0]
    filteredColorsA = []
    filteredColorsB = []
    mses = []
    for k in ks:
        # Regression pass
        filteredColorA = collaborativeRegression(colorA, filteredFeaturesB, 2*colorVar, k)
        filteredColorB = collaborativeRegression(colorB, filteredFeaturesA, 2*colorVar, k)
        filteredColorsA.append(filteredColorA)
        filteredColorsB.append(filteredColorB)

        # MSE estimation (section 5.3)
        mseA = (colorB - filteredColorA)^2 - 2*colorVar
        mseB = (colorA - filteredColorB)^2 - 2*colorVar
        residualColorVariance = (filteredColorB - filteredColorA)^2/4
        noisyMse = (mseA + mseB)/2 - residualColorVariance

        # MSE filtering
        mses.append(nlMeans(noisyMse, color, colorVar, F=1, R=9, k=1.0))
        
    # Bandwidth selection (section 5.3)
    resultA = 0
    resultB = 0
    for i in range(len(ks)):
        # Generate selection map
        noisySelection = (mses[i] == min(mses))
        # Filter selection map
        selection = nlMeans(noisySelection, color, colorVar, F=1, R=9, k=1.0)
        
        # Apply selection map
        resultA += filteredColorsA[i]*selection
        resultB += filteredColorsB[i]*selection

    # Second filter pass (section 5.4)
    finalFeatures = []
    for i in range(len(featureVars)):
        combinedFeature = (filteredFeaturesA[i] + filteredFeaturesB[i])/2
        combinedFeatureVar = (filteredFeaturesB[i] - filteredFeaturesA[i])^2/4
        
        finalFeatures.append(nlMeans(combinedFeature, combinedFeature, combinedFeatureVar, F=3, R=2, k=0.5))

    combinedResult = (resultA + resultB)/2
    combinedResultVar = (resultB - resultA)^2/4
    return collaborativeRegression(combinedResult, finalFeatures, combinedResultVar, 1.0)

def collaborativeRegression(image, features, imageVar, k):
    # notation:
    #   P:          list of pixel indices in a patch. N = len(P)
    #   image_P:    returns N x 3 matrix of RGB image pixels contained in patch P
    #   features_P: returns N x len(features) matrix of auxiliary feature vectors contained in patch P
    
    result = 0
    weights = 0
    for P in patches_19x19(image):
        w = nlmWeights(image_P, imageVar_P, F=3, R=9, k=k)
        W = diagMatrix(w)
        Y = buffer_P
        X = features_P
        reconstruction = X (X^T W X)^-1 X^T W Y
        
        result_P += w*reconstruction;
        weights_P += w
    
    return result/weights

# Standard NL-Means filter, filtering input image `image`, using distances computed
# on guide image `guide`, with guide variance `variance`
# with patch radius `F`, filter radius `R`, and bandwidth `k`
# Please see Rousselle et al., "Robust Denoising using Feature and Color Information", for details
def nlMeans(image, guide, variance, F, R, k):
    # [omitted for brevity]
    