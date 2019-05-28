import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time
import sys
import warnings
warnings.filterwarnings('ignore')


def waitforEnter(fstop=True):
        if fstop:
                if sys.version_info[0] == 2:
                        raw_input("Press ENTER to continue.")
                else:
                        input("Press ENTER to continue.")
# 1
# Function that does a plot based on a data array and a name
def plot(data,name,data1,name1):
	plt.subplot(3,1,1)
	plt.plot(data)
	plt.title(name)
	plt.subplot(3,1,2)
	plt.plot(data1)
	plt.title(name1)
	'''
	plt.subplot(3,1,3)
	plt.plot(data2)
	plt.title(name2)
	'''
	plt.show()
	waitforEnter()

# 2
# Breaks data into trains
def breakTrainTest(data,oWnd=300,trainPerc=0.5):
	#print(data.shape[0]	%2)
	if data.shape[0]%oWnd != 0 :
		data = data[0:(int(data.shape[0]/oWnd))*oWnd]
		nSamp = data.shape[0]
	else :
		nSamp=data.shape[0]

	nCols = 1
	nObs=int(nSamp/oWnd)

	data_obs=data.reshape(nObs,oWnd) #??

	order=np.random.permutation(nObs)

	nTrain=int(nObs*trainPerc)

	data_train=data_obs[order[:nTrain],:]
	data_test=data_obs[order[nTrain:],:]

	return(data_train,data_test)

# 3
# Extracts features from trains
def extractFeatures(data,Class=0):
        features=[]
        #print(data.shape)
        nObs=data.shape[0]#sum(1 for _ in data)
        oClass=np.ones((nObs,1))*Class

        for i in range(nObs):
        	print('\nTrain number : {}'.format(i))
	        M1=np.mean(data[i],axis=0)
	        print('Mean',M1)
	        Md1=np.median(data[i],axis=0)
	        print('Median',Md1)
	        Std1=np.std(data[i],axis=0)
	        print('Deviation',Std1)
	        S1=stats.skew(data[i])
	        print('Skew',S1)
	        K1=stats.kurtosis(data[i])
	        print('Kurtosis',K1)
	        p=[75,90,95]
	        Pr1=np.array(np.percentile(data[i],p,axis=0)).T.flatten()
	        print('Percentile(75, 90, 95)',Pr1)
	        
	        faux=np.hstack((M1,Md1,Std1,S1,K1,Pr1))
	        faux=np.hstack((M1,Std1,Pr1))
	        features.append(faux)

        return(np.array(features),oClass)

# 4
# Plots all the features
def plotFeatures(features,oClass,f1index=0,f2index=1):
	nObs,nFea=features.shape

	colors=['b','g','r']
	for i in range(nObs):
		plt.plot(features[i],'o'+colors[int(oClass[i])])

	plt.show()
	waitforEnter()

# 7
# Extracts features from wavelet
def extractFeaturesWavelet(data,scales=[2,4,8,16,32],Class=0):
	features=[]
	nObs,nSamp=data.shape
	oClass=np.ones((nObs,1))*Class
	for i in range(nObs):
		scalo_features=np.array([])
		scalo,fscales=scalogram.scalogramCWT(data[i,:],scales)
		scalo_features=np.append(scalo_features,scalo)
		features.append(scalo_features)
		
	return(np.array(features),oClass)

# 11
def distance(c,p):
	return(np.sqrt(np.sum(np.square(p-c))))


################ Main Code #######################
Classes={0:'Plot1',1:'Plot2'}
#Classes={0:'Plot1',1:'Plot2',2:'Plot3'}

# 1
# Read data from files and plot the download + upload traffic
# We don't need to verify if it is download or upload at least by now
traffic_data = np.loadtxt(sys.argv[1])
traffic_data2 = np.loadtxt(sys.argv[2])
#traffic_data3 = np.loadtxt(sys.argv[2])

plt.figure(1)
plot(traffic_data,'Plot1 Title',traffic_data2,'Plot2 Title')
#plot(traffic_data,'Plot1 Title',traffic_data2,'Plot2 Title',traffic_data3,'Plot3 Title')

# 2
# Divide each stream in observation windows of 5 minutes
# Divide randomly a set for training and one for testing
traffic_data_train, traffic_data_test= breakTrainTest(traffic_data)
traffic_data_train2, traffic_data_test2= breakTrainTest(traffic_data2)
#traffic_data_train3, traffic_data_test3= breakTrainTest(traffic_data3)

plt.figure(2)
plt.subplot(3,1,1)
for i in range(2):
	plt.plot(traffic_data_train[i],'b')
	plt.plot(traffic_data_test[i],'g')
plt.title('Plot1')
plt.ylabel('Bytes/sec')
plt.subplot(3,1,2)
for i in range(2):
	plt.plot(traffic_data_train2[i],'b')
	plt.plot(traffic_data_test2[i],'g')
plt.title('Plot2')
plt.ylabel('Bytes/sec')
plt.show()
'''
plt.subplot(3,1,3)
for i in range(2):
	plt.plot(traffic_data_train3[i],'b')
	plt.plot(traffic_data_test3[i],'g')
plt.title('Plot2')
plt.ylabel('Bytes/sec')
plt.show()
'''
waitforEnter()

# 3
# Extract and print features
features_traffic, oClass_traffic = extractFeatures(traffic_data_train,Class=0)
features_traffic2, oClass_traffic2 = extractFeatures(traffic_data_train2,Class=1)
#features_traffic3, oClass_traffic3 = extractFeatures(traffic_data_train3,Class=2)

features= np.vstack((features_traffic,features_traffic2))
oClass= np.vstack((oClass_traffic,oClass_traffic2))
'''
features= np.vstack((features_traffic,features_traffic2, features_traffic3))
oClass= np.vstack((oClass_traffic,oClass_traffic2, oClass_traffic3))
'''
print('\nTrain Stats Features Size:', features.shape)

# 4
# Plots the features extracted
plt.figure(4)
plotFeatures(features, oClass, 0, 1)

# 7
# Extracts features from wavelet
import scalogram
scales=[2,4,8,16,32,64,128,256]
features_plot1_W,oClass_plot1 = extractFeaturesWavelet(traffic_data_train,scales,Class=0)
features_plot2_W,oClass_plot2 = extractFeaturesWavelet(traffic_data_train2,scales,Class=1)
#features_plot3_W,oClass_plot3 = extractFeaturesWavelet(traffic_data_train3,scales,Class=2)

featuresW=np.vstack((features_plot1_W,features_plot2_W))
oClass=np.vstack((oClass_plot1,oClass_plot2))
'''
featuresW=np.vstack((features_plot1_W,features_plot2_W,features_plot3_W)))
oClass=np.vstack((oClass_plot1,oClass_plot2,oClass_plot3))
'''

print('Train Wavelet Features Size:',featuresW.shape)
plt.figure(7)
plotFeatures(featuresW,oClass,3,10)

# 8
# Reduces features to 3
from sklearn.decomposition import PCA
pca = PCA(n_components=2, svd_solver='full')
pcaFeatures = pca.fit(features).transform(features)
plt.figure(8)
plotFeatures(pcaFeatures,oClass,0,1)

# 9
# Reduces features from Wavelet to 3
pca = PCA(n_components=2, svd_solver='full')
pcaFeatures = pca.fit(featuresW).transform(featuresW)

plt.figure(9)
plotFeatures(pcaFeatures,oClass,0,1)

# 10
# Reduces all features extracted to 3
allFeatures=np.hstack((features,featuresW))
print('Train (All) Features Size:',allFeatures.shape)

pca = PCA(n_components=2, svd_solver='full')
pcaFeatures = pca.fit(allFeatures).transform(allFeatures)

plt.figure(10)
plotFeatures(pcaFeatures,oClass,0,1)

# 11
# Classification based on distance
centroids={}
for c in range(3):
	pClass=(oClass==c).flatten()
	centroids.update({c:np.mean(allFeatures[pClass,:],axis=0)})
print('All Features Centroids:\n',centroids)

traffic_data_test,oClass_traffic = extractFeatures(traffic_data_test,Class=0)
traffic_data_test2,oClass_traffic2 = extractFeatures(traffic_data_test2,Class=1)
#traffic_data_test3,oClass_traffic3 = extractFeatures(traffic_data_test3,Class=2)
testFeatures=np.vstack((traffic_data_test,traffic_data_test2))
#testFeatures=np.vstack((traffic_data_test,traffic_data_test2,traffic_data_test3))

features_plot1_W,oClass_plot1=extractFeaturesWavelet(features_plot1_W,scales,Class=0)
features_plot2_W,oClass_plot2=extractFeaturesWavelet(features_plot2_W,scales,Class=1)
#features_plot3_W,oClass_plot3=extractFeaturesWavelet(features_plot3_W,scales,Class=2)
testFeaturesW=np.vstack((features_plot1_W,features_plot2_W))
#testFeaturesW=np.vstack((features_plot1_W,features_plot2_W,features_plot3_W))

alltestFeatures=np.hstack((testFeatures,testFeaturesW))
print('Test Features Size:', alltestFeatures.shape)

testpcaFeatures=pca.transform(alltestFeatures)
print('\n-- Classification based on Distances --')
nObsTest,nFea=alltestFeatures.shape
for i in range(nObsTest):
	x=alltestFeatures[i]
	dists=[distance(x,centroids[0]),distance(x,centroids[1])]
	#dists=[distance(x,centroids[0]),distance(x,centroids[1]),,distance(x,centroids[2])]
	ndists=dists/np.sum(dists)
	testClass=np.argsort(dists)[0]
	
	print('Obs: {:2}: Normalized Distances to Centroids: [{:.4f},{:.4f}] -> Classification: {} -> {}'.format(i,*ndists,testClass,Classes[testClass]))
	#print('Obs: {:2}: Normalized Distances to Centroids: [{:.4f},{:.4f},{:.4f}] -> Classification: {} -> {}'.format(i,*ndists,testClass,Classes[testClass]))

# 12
# Classification based on highest probability 
from scipy.stats import multivariate_normal
print('\n-- Classification based on Multivariate PDF (PCA Features) --')
means={}
for c in range(2):
	pClass=(oClass==c).flatten()
	means.update({c:np.mean(pcaFeatures[pClass,:],axis=0)})
#print(means)

covs={}
for c in range(2):
	pClass=(oClass==c).flatten()
	covs.update({c:np.cov(pcaFeatures[pClass,:],rowvar=0)})
#print(covs)

testpcaFeatures=pca.transform(alltestFeatures)	#uses pca fitted above, only transforms test data
print(testpcaFeatures)
nObsTest,nFea=testpcaFeatures.shape
for i in range(nObsTest):
	x=testpcaFeatures[i,:]
	probs=np.array([multivariate_normal.pdf(x,means[0],covs[0]),multivariate_normal.pdf(x,means[1],covs[1])])
	#probs=np.array([multivariate_normal.pdf(x,means[0],covs[0]),multivariate_normal.pdf(x,means[1],covs[1]),multivariate_normal.pdf(x,means[2],covs[2])])
	testClass=np.argsort(probs)[-1]
	
	print('Obs: {:2}: Probabilities: [{:.4e},{:.4e}] -> Classification: {} -> {}'.format(i,*probs,testClass,Classes[testClass]))