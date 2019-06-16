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
def plot(data,name,data1,name1,data2,name2):
	plt.subplot(3,1,1)
	plt.plot(data)
	plt.title(name)
	
	plt.subplot(3,1,2)
	plt.plot(data1)
	plt.title(name1)

	plt.subplot(3,1,3)
	plt.plot(data2)
	plt.title(name2)
	
	plt.show()
	waitforEnter()

# 2
# Breaks data into trains
def breakTrainTest(data,oWnd=300,trainPerc=0.5):
	#print(data.shape[0]	%2)
	nSamp,nCols= data.shape
	if data.shape[0]%oWnd != 0 :
		data = data[0:(int(data.shape[0]/oWnd))*oWnd]
		nSamp = data.shape[0]
	else :
		nSamp=data.shape[0]
	
	nObs=int(nSamp/oWnd)

	data_obs=data.reshape((nObs,oWnd,nCols))

	order=np.random.permutation(nObs)

	nTrain=int(nObs*trainPerc)

	data_train=data_obs[order[:nTrain],:,:]
	data_test=data_obs[order[nTrain:],:,:]

	return(data_train,data_test)

# 3
# Extracts features from trains
def extractFeatures(data,Class=0):
        features=[]
        #print(data.shape)
        nObs,nSamp,nCols=data.shape
        oClass=np.ones((nObs,1))*Class

        for i in range(nObs):
        	print('\nTrain number : {}'.format(i))
	        M1=np.mean(data[i,:,:],axis=0)
	        print('Mean',M1)
	        Md1=np.median(data[i,:,:],axis=0)
	        print('Median',Md1)
	        Std1=np.std(data[i,:,:],axis=0)
	        print('Deviation',Std1)
	        S1=stats.skew(data[i,:,:])
	        print('Skew',S1)
	        K1=stats.kurtosis(data[i,:,:])
	        print('Kurtosis',K1)
	        p=[75,90,95]
	        Pr1=np.array(np.percentile(data[i,:,:],p,axis=0)).T.flatten()
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
		plt.plot(features[i,f1index],features[i,f2index],'o'+colors[int(oClass[i])])
	plt.show()
	waitforEnter()

def logplotFeatures(features,oClass,f1index=0,f2index=1):
	nObs,nFea=features.shape
	colors=['b','g','r']
	for i in range(nObs):
		plt.loglog(features[i,f1index],features[i,f2index],'o'+colors[int(oClass[i])])

	plt.show()
	waitforEnter()

## -- 5 -- ##
def extratctSilence(data,threshold=256):
	if(data[0]<=threshold):
		s=[1]
	else:
		s=[]
	for i in range(1,len(data)):
		if(data[i-1]>threshold and data[i]<=threshold):
			s.append(1)
		elif (data[i-1]<=threshold and data[i]<=threshold):
			s[-1]+=1
	
	return(s)
	
def extractFeaturesSilence(data,Class=0):
	features=[]
	nObs,nSamp,nCols=data.shape
	oClass=np.ones((nObs,1))*Class
	for i in range(nObs):
		silence_features=np.array([])
		for c in range(nCols):
			silence=extratctSilence(data[i,:,c],threshold=0)
			if len(silence)>0:
				silence_features=np.append(silence_features,[np.mean(silence),np.var(silence)])
			else:
				silence_features=np.append(silence_features,[0,0])
			
			
		features.append(silence_features)
		
	return(np.array(features),oClass)

# 7
# Extracts features from wavelet
def extractFeaturesWavelet(data,scales=[2,4,8,16,32],Class=0):
	features=[]
	nObs,nSamp,nCols=data.shape
	oClass=np.ones((nObs,1))*Class
	for i in range(nObs):
		scalo_features=np.array([])
		for c in range(nCols):
			scalo,fscales=scalogram.scalogramCWT(data[i,:,c],scales)
			scalo_features=np.append(scalo_features,scalo)
		
		features.append(scalo_features)
		
	return(np.array(features),oClass)

# 11
def distance(c,p):
	return(np.sqrt(np.sum(np.square(p-c))))


################ Main Code #######################

Classes={0:'Kid',1:'Teenager',2:'Adult'}

# 1
# Read data from files and plot the download + upload traffic
# We don't need to verify if it is download or upload at least by now
traffic_data = np.loadtxt(sys.argv[1])
traffic_data2 = np.loadtxt(sys.argv[2])
traffic_data3 = np.loadtxt(sys.argv[3])

plt.figure(1)
plot(traffic_data,'Kid',traffic_data2,'Teenager',traffic_data3,'Adult')

# 2
# Divide each stream in observation windows of 5 minutes
# Divide randomly a set for training and one for testing
traffic_data_train, traffic_data_test= breakTrainTest(traffic_data)
traffic_data_train2, traffic_data_test2= breakTrainTest(traffic_data2)
traffic_data_train3, traffic_data_test3= breakTrainTest(traffic_data3)

plt.figure(2)
plt.subplot(3,1,1)
for i in range(2):
	plt.plot(traffic_data_train[i,:,0],'b')
	plt.plot(traffic_data_test[i,:,1],'g')
plt.title('Adult')
plt.ylabel('Bytes/sec')

plt.subplot(3,1,2)
for i in range(2):
	plt.plot(traffic_data_train2[i,:,0],'b')
	plt.plot(traffic_data_test2[i,:,1],'g')
plt.title('Kid')
plt.ylabel('Bytes/sec')

plt.subplot(3,1,3)
for i in range(2):
	plt.plot(traffic_data_train3[i,:,0],'b')
	plt.plot(traffic_data_test3[i,:,1],'g')
plt.title('Teenager')
plt.ylabel('Bytes/sec')

plt.show()
waitforEnter()

# 3
# Extract and print features
features_traffic, oClass_traffic = extractFeatures(traffic_data_train,Class=0)
features_traffic2, oClass_traffic2 = extractFeatures(traffic_data_train2,Class=1)
features_traffic3, oClass_traffic3 = extractFeatures(traffic_data_train3,Class=2)

features= np.vstack((features_traffic, features_traffic2, features_traffic3))
oClass= np.vstack((oClass_traffic, oClass_traffic2, oClass_traffic3))
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
features_plot3_W,oClass_plot3 = extractFeaturesWavelet(traffic_data_train3,scales,Class=2)


featuresW=np.vstack((features_plot1_W,features_plot2_W,features_plot3_W))
oClass=np.vstack((oClass_plot1,oClass_plot2,oClass_plot3))


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

'''
# 11
# Classification based on distance
centroids={}
for c in range(3):
	pClass=(oClass==c).flatten()
	centroids.update({c:np.mean(allFeatures[pClass,:],axis=0)})
print('All Features Centroids:\n',centroids)

features_traffic,oClass_traffic = extractFeatures(traffic_data_test,Class=0)
features_traffic2,oClass_traffic2 = extractFeatures(traffic_data_test2,Class=1)
features_traffic3,oClass_traffic3 = extractFeatures(traffic_data_test3,Class=2)
testFeatures=np.vstack((features_traffic,features_traffic2,features_traffic3))

features_plot1_W,oClass_plot1=extractFeaturesWavelet(traffic_data_test,scales,Class=0)
features_plot2_W,oClass_plot2=extractFeaturesWavelet(traffic_data_test2,scales,Class=1)
features_plot3_W,oClass_plot3=extractFeaturesWavelet(traffic_data_test3,scales,Class=2)
testFeaturesW=np.vstack((features_plot1_W,features_plot2_W,features_plot3_W))

alltestFeatures=np.hstack((testFeatures,testFeaturesW))
print('Test Features Size:', alltestFeatures.shape)

testpcaFeatures=pca.transform(alltestFeatures)
print('\n-- Classification based on Distances --')
nObsTest,nFea=alltestFeatures.shape
for i in range(nObsTest):
	x=alltestFeatures[i]
	dists=[distance(x,centroids[0]),distance(x,centroids[1]),distance(x,centroids[2])]
	ndists=dists/np.sum(dists)
	testClass=np.argsort(dists)[0]
	
	print('Obs: {:2}: Normalized Distances to Centroids: [{:.4f},{:.4f},{:.4f}] -> Classification: {} -> {}'.format(i,*ndists,testClass,Classes[testClass]))

# 12
# Classification based on highest probability 
from scipy.stats import multivariate_normal
print('\n-- Classification based on Multivariate PDF (PCA Features) --')
means={}
for c in range(3):
	pClass=(oClass==c).flatten()
	means.update({c:np.mean(pcaFeatures[pClass,:],axis=0)})
print(means)

covs={}
for c in range(3):
	pClass=(oClass==c).flatten()
	covs.update({c:np.cov(pcaFeatures[pClass,:],rowvar=0)})
print(covs)

testpcaFeatures=pca.transform(alltestFeatures)	#uses pca fitted above, only transforms test data
print(testpcaFeatures)
nObsTest,nFea=testpcaFeatures.shape

for i in range(nObsTest):
	x=testpcaFeatures[i,:]
	probs=np.array([multivariate_normal.pdf(x,means[0],covs[0]),multivariate_normal.pdf(x,means[1],covs[1]),multivariate_normal.pdf(x,means[2],covs[2])])
	testClass=np.argsort(probs)[-1]
	
	print('Obs: {:2}: Probabilities: [{:.4e},{:.4e}, {:.4e}] -> Classification: {} -> {}'.format(i,*probs,testClass,Classes[testClass]))

## -- 13 -- ##
scaler=StandardScaler()
NormAllFeatures=scaler.fit_transform(allFeatures)

NormAllTestFeatures=scaler.fit_transform(alltestFeatures)

pca = PCA(n_components=3, svd_solver='full')
NormPcaFeatures = pca.fit(NormAllFeatures).transform(NormAllFeatures)

NormTestPcaFeatures = pca.fit(NormAllTestFeatures).transform(NormAllTestFeatures)

##

print('\n-- Classification based on Clustering (Kmeans) --')
from sklearn.cluster import KMeans
#K-means assuming 3 clusters
centroids=np.array([])
for c in range(3):
	pClass=(oClass==c).flatten()
	centroids=np.append(centroids,np.mean(NormPcaFeatures[pClass,:],axis=0))
centroids=centroids.reshape((3,3))
print('PCA (pcaFeatures) Centroids:\n',centroids)

kmeans = KMeans(init=centroids, n_clusters=3)
kmeans.fit(NormPcaFeatures)
labels=kmeans.labels_
print('Labels:',labels)

#Determines and quantifies the presence of each original class observation in each cluster
KMclass=np.zeros((3,3))
for cluster in range(3):
	p=(labels==cluster)
	aux=oClass[p]
	for c in range(3):
		KMclass[cluster,c]=np.sum(aux==c)

probKMclass=KMclass/np.sum(KMclass,axis=1)[:,np.newaxis]
print(probKMclass)
nObsTest,nFea=NormTestPcaFeatures.shape
for i in range(nObsTest):
	x=NormTestPcaFeatures[i,:].reshape((1,nFea))
	label=kmeans.predict(x)
	testClass=100*probKMclass[label,:].flatten()
	print('Obs: {:2}: Probabilities beeing in each class: [{:.2f}%,{:.2f}%,{:.2f}%]'.format(i,*testClass))


## -- 14 -- ##
from sklearn.cluster import DBSCAN
#DBSCAN assuming a neighborhood maximum distance of 1e11
dbscan = DBSCAN(eps=10000)
dbscan.fit(pcaFeatures)
labels=dbscan.labels_
print('Labels:',labels)

## -- 15 -- #
from sklearn import svm
print('\n-- Classification based on Support Vector Machines --')
svc = svm.SVC(kernel='linear').fit(NormAllFeatures, oClass)  
rbf_svc = svm.SVC(kernel='rbf').fit(NormAllFeatures, oClass)  
poly_svc = svm.SVC(kernel='poly',degree=2).fit(NormAllFeatures, oClass)  
lin_svc = svm.LinearSVC().fit(NormAllFeatures, oClass)  

L1=svc.predict(NormAllTestFeatures)
print('class (from test PCA features with SVC):',L1)
L2=rbf_svc.predict(NormAllTestFeatures)
print('class (from test PCA features with Kernel RBF):',L2)
L3=poly_svc.predict(NormAllTestFeatures)
print('class (from test PCA features with Kernel poly):',L3)
L4=lin_svc.predict(NormAllTestFeatures)
print('class (from test PCA features with Linear SVC):',L4)
print('\n')

nObsTest,nFea=NormAllTestFeatures.shape
for i in range(nObsTest):
	print('Obs: {:2}: SVC->{} | Kernel RBF->{} | Kernel Poly->{} | LinearSVC->{}'.format(i,Classes[L1[i]],Classes[L2[i]],Classes[L3[i]],Classes[L4[i]]))

## -- 16 -- #
print('\n-- Classification based on Support Vector Machines  (PCA Features) --')
svc = svm.SVC(kernel='linear').fit(NormPcaFeatures, oClass)  
rbf_svc = svm.SVC(kernel='rbf').fit(NormPcaFeatures, oClass)  
poly_svc = svm.SVC(kernel='poly',degree=2).fit(NormPcaFeatures, oClass)  
lin_svc = svm.LinearSVC().fit(NormPcaFeatures, oClass)  

L1=svc.predict(NormTestPcaFeatures)
print('class (from test PCA features with SVC):',L1)
L2=rbf_svc.predict(NormTestPcaFeatures)
print('class (from test PCA features with Kernel RBF):',L2)
L3=poly_svc.predict(NormTestPcaFeatures)
print('class (from test PCA features with Kernel poly):',L3)
L4=lin_svc.predict(NormTestPcaFeatures)
print('class (from test PCA features with Linear SVC):',L4)
print('\n')

nObsTest,nFea=NormTestPcaFeatures.shape
for i in range(nObsTest):
	print('Obs: {:2}: SVC->{} | Kernel RBF->{} | Kernel Poly->{} | LinearSVC->{}'.format(i,Classes[L1[i]],Classes[L2[i]],Classes[L3[i]],Classes[L4[i]]))

'''
## -- 17 -- ##

## -- 13 -- ##
features_traffic,oClass_traffic = extractFeatures(traffic_data_test,Class=0)
features_traffic2,oClass_traffic2 = extractFeatures(traffic_data_test2,Class=1)
features_traffic3,oClass_traffic3 = extractFeatures(traffic_data_test3,Class=2)
testFeatures=np.vstack((features_traffic,features_traffic2,features_traffic3))

features_plot1_W,oClass_plot1=extractFeaturesWavelet(traffic_data_test,scales,Class=0)
features_plot2_W,oClass_plot2=extractFeaturesWavelet(traffic_data_test2,scales,Class=1)
features_plot3_W,oClass_plot3=extractFeaturesWavelet(traffic_data_test3,scales,Class=2)
testFeaturesW=np.vstack((features_plot1_W,features_plot2_W,features_plot3_W))

alltestFeatures=np.hstack((testFeatures,testFeaturesW))
print('Test Features Size:', alltestFeatures.shape)

scaler=StandardScaler()
NormAllFeatures=scaler.fit_transform(allFeatures)

NormAllTestFeatures=scaler.fit_transform(alltestFeatures)

pca = PCA(n_components=3, svd_solver='full')
NormPcaFeatures = pca.fit(NormAllFeatures).transform(NormAllFeatures)

NormTestPcaFeatures = pca.fit(NormAllTestFeatures).transform(NormAllTestFeatures)

from sklearn.neural_network import MLPClassifier
print('\n-- Classification based on Neural Networks --')

alpha=1
max_iter=100000
clf = MLPClassifier(solver='lbfgs',alpha=alpha,hidden_layer_sizes=(100,),max_iter=max_iter)
clf.fit(NormPcaFeatures, oClass) 
LT=clf.predict(NormTestPcaFeatures) 
print('class (from test PCA):',LT)

nObsTest,nFea=NormTestPcaFeatures.shape
for i in range(nObsTest):
	print('Obs: {:2}: Classification->{}'.format(i,Classes[LT[i]]))

