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
import scalogram
warnings.filterwarnings('ignore')

# Function that waits for a user enter before continuing
def waitforEnter(fstop=True):
        if fstop:
                if sys.version_info[0] == 2:
                        raw_input("Press ENTER to continue.")
                else:
                        input("Press ENTER to continue.")

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

# Breaks data into trains
def breakTrainTest(data,oWnd=300,trainPerc=0.5):
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

# Extracts features from trains
def extractFeatures(data,Class=0):
        features=[]
        nObs,nSamp,nCols=data.shape
        oClass=np.ones((nObs,1))*Class

        for i in range(nObs):	
	        M1=np.mean(data[i,:,:],axis=0)
	        Md1=np.median(data[i,:,:],axis=0)
	        Std1=np.std(data[i,:,:],axis=0)
	        S1=stats.skew(data[i,:,:])
	        K1=stats.kurtosis(data[i,:,:])
	        p=[75,90,95]
	        Pr1=np.array(np.percentile(data[i,:,:],p,axis=0)).T.flatten()
	        
	        faux=np.hstack((M1,Md1,Std1,S1,K1,Pr1))
	        faux=np.hstack((M1,Std1,Pr1))
	        features.append(faux)
	        if( plotEnable == 1) :
	        	print('\nTrain number : {}'.format(i))
	        	print('Mean',M1)
	        	print('Median',Md1)
	        	print('Deviation',Std1)
	        	print('Skew',S1)	
	        	print('Kurtosis',K1)  	
	        	print('Percentile(75, 90, 95)',Pr1)

        return(np.array(features),oClass)

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

# Extracts silence time from data
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
	
# Extracts silence time from features
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

# Basic function for distance calculation
def distance(c,p):
	return(np.sqrt(np.sum(np.square(p-c))))


# Main Code
def main ():
	Classes={0:'Kid',1:'Teenager',2:'Adult'}
	plotEnable = 0;

	# Read data from files and plot the download + upload traffic
	# We don't need to verify if it is download or upload at least by now
	if (sys.argv[1] == "-s") : 
		traffic_data = np.loadtxt(sys.argv[2])
		traffic_data2 = np.loadtxt(sys.argv[3])
		traffic_data3 = np.loadtxt(sys.argv[4])
		plotEnable = 0;

	else: 	
		traffic_data = np.loadtxt(sys.argv[1])
		traffic_data2 = np.loadtxt(sys.argv[2])
		traffic_data3 = np.loadtxt(sys.argv[3])
		plotEnable = 1;

	if (plotEnable == 1) :
		plt.figure(1)
		plot(traffic_data,'Kid',traffic_data2,'Teenager',traffic_data3,'Adult')

	# Divide each stream in observation windows of 5 minutes
	# Divide randomly a set for training and one for testing
	traffic_data_train, traffic_data_test= breakTrainTest(traffic_data)
	traffic_data_train2, traffic_data_test2= breakTrainTest(traffic_data2)
	traffic_data_train3, traffic_data_test3= breakTrainTest(traffic_data3)

	if( plotEnable == 1) :
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

	# Extract and print features
	features_traffic, oClass_traffic = extractFeatures(traffic_data_train,Class=0)
	features_traffic2, oClass_traffic2 = extractFeatures(traffic_data_train2,Class=1)
	features_traffic3, oClass_traffic3 = extractFeatures(traffic_data_train3,Class=2)

	features= np.vstack((features_traffic, features_traffic2, features_traffic3))
	oClass= np.vstack((oClass_traffic, oClass_traffic2, oClass_traffic3))

	if( plotEnable == 1) :
		print('\nTrain Stats Features Size:', features.shape)
		plt.figure(4)
		plotFeatures(features, oClass, 0, 1)

	# Extracts features from wavelet
	scales=[2,4,8,16,32,64,128,256]
	features_plot1_W,oClass_plot1 = extractFeaturesWavelet(traffic_data_train,scales,Class=0)
	features_plot2_W,oClass_plot2 = extractFeaturesWavelet(traffic_data_train2,scales,Class=1)
	features_plot3_W,oClass_plot3 = extractFeaturesWavelet(traffic_data_train3,scales,Class=2)

	featuresW=np.vstack((features_plot1_W,features_plot2_W,features_plot3_W))
	oClass=np.vstack((oClass_plot1,oClass_plot2,oClass_plot3))

	if( plotEnable == 1) :
		print('Train Wavelet Features Size:',featuresW.shape)
		plt.figure(7)
		plotFeatures(featuresW,oClass,3,10)

	# Reduces features to 3
	from sklearn.decomposition import PCA
	pca = PCA(n_components=2, svd_solver='full')
	pcaFeatures = pca.fit(features).transform(features)

	if( plotEnable == 1) :
		plt.figure(8)
		plotFeatures(pcaFeatures,oClass,0,1)

	# Reduces features from Wavelet to 3
	pca = PCA(n_components=2, svd_solver='full')
	pcaFeatures = pca.fit(featuresW).transform(featuresW)

	if( plotEnable == 1) :
		plt.figure(9)
		plotFeatures(pcaFeatures,oClass,0,1)

	# Reduces all features extracted to 3
	allFeatures=np.hstack((features,featuresW))
	pca = PCA(n_components=2, svd_solver='full')
	pcaFeatures = pca.fit(allFeatures).transform(allFeatures)

	if( plotEnable == 1) :
		print('Train (All) Features Size:',allFeatures.shape)
		plt.figure(10)
		plotFeatures(pcaFeatures,oClass,0,1)

	#Extract all features.
	features_traffic,oClass_traffic = extractFeatures(traffic_data_test,Class=0)
	features_traffic2,oClass_traffic2 = extractFeatures(traffic_data_test2,Class=1)
	features_traffic3,oClass_traffic3 = extractFeatures(traffic_data_test3,Class=2)
	testFeatures=np.vstack((features_traffic,features_traffic2,features_traffic3))

	features_plot1_W,oClass_plot1=extractFeaturesWavelet(traffic_data_test,scales,Class=0)
	features_plot2_W,oClass_plot2=extractFeaturesWavelet(traffic_data_test2,scales,Class=1)
	features_plot3_W,oClass_plot3=extractFeaturesWavelet(traffic_data_test3,scales,Class=2)
	testFeaturesW=np.vstack((features_plot1_W,features_plot2_W,features_plot3_W))

	alltestFeatures=np.hstack((testFeatures,testFeaturesW))

	if( plotEnable == 1) :
		print('Test Features Size:', alltestFeatures.shape)

	scaler=StandardScaler()
	NormAllFeatures=scaler.fit_transform(allFeatures)
	NormAllTestFeatures=scaler.fit_transform(alltestFeatures)

	pca = PCA(n_components=3, svd_solver='full')
	NormPcaFeatures = pca.fit(NormAllFeatures).transform(NormAllFeatures)

	NormTestPcaFeatures = pca.fit(NormAllTestFeatures).transform(NormAllTestFeatures)

	from sklearn.neural_network import MLPClassifier
	print('\nClassification based on Neural Networks \n')

	alpha=1
	max_iter=100000
	clf = MLPClassifier(solver='lbfgs',alpha=alpha,hidden_layer_sizes=(100,),max_iter=max_iter)
	clf.fit(NormPcaFeatures, oClass) 
	LT=clf.predict(NormTestPcaFeatures) 
	print('class (from test PCA):',LT)

	nObsTest,nFea=NormTestPcaFeatures.shape
	for i in range(nObsTest):
		print('Obs: {:2}: Classification->{}'.format(i,Classes[LT[i]]))

# Main Code
def main_v2 (data1):
	Classes={0:'Kid',1:'Teenager',2:'Adult', 3:'Live'}
	plotEnable = 0;

	# Read data from files and plot the download + upload traffic
	# We don't need to verify if it is download or upload at least by now

	live_data = np.loadtxt(data1)

	kid_data = np.loadtxt('kid.dat')
	teen_data = np.loadtxt('teenager.dat')
	adult_data = np.loadtxt('adult.dat')

	# Divide each stream in observation windows of 5 minutes
	# Divide randomly a set for training and one for testing
	live_data_train, live_data_test= breakTrainTest(live_data)

	traffic_data_train, traffic_data_test= breakTrainTest(kid_data)
	traffic_data_train2, traffic_data_test2= breakTrainTest(teen_data)
	traffic_data_train3, traffic_data_test3= breakTrainTest(adult_data)

	# Extract and print features
	#live_traffic, oClass_life= extractFeatures(live_data_train,Class=3) #TODO

	features_traffic, oClass_traffic = extractFeatures(traffic_data_train,Class=0)
	features_traffic2, oClass_traffic2 = extractFeatures(traffic_data_train2,Class=1)
	features_traffic3, oClass_traffic3 = extractFeatures(traffic_data_train3,Class=2)

	#features= np.vstack((features_traffic, features_traffic2, features_traffic3, live_traffic))
	#oClass= np.vstack((oClass_traffic, oClass_traffic2, oClass_traffic3, oClass_life))

	features= np.vstack((features_traffic, features_traffic2, features_traffic3))
	oClass= np.vstack((oClass_traffic, oClass_traffic2, oClass_traffic3))

	# Extracts features from wavelet
	scales=[2,4,8,16,32,64,128,256]

	#features_live_W,oClass_live = extractFeaturesWavelet(live_data_train,scales,Class=3) #TODO

	features_plot1_W,oClass_plot1 = extractFeaturesWavelet(traffic_data_train,scales,Class=0)
	features_plot2_W,oClass_plot2 = extractFeaturesWavelet(traffic_data_train2,scales,Class=1)
	features_plot3_W,oClass_plot3 = extractFeaturesWavelet(traffic_data_train3,scales,Class=2)

	#featuresW=np.vstack((features_plot1_W,features_plot2_W,features_plot3_W,features_live_W))
	#oClass=np.vstack((oClass_plot1,oClass_plot2,oClass_plot3,oClass_live))

	featuresW=np.vstack((features_plot1_W,features_plot2_W,features_plot3_W))
	oClass=np.vstack((oClass_plot1,oClass_plot2,oClass_plot3))

	# Reduces features to 3
	from sklearn.decomposition import PCA
	pca = PCA(n_components=2, svd_solver='full')
	pcaFeatures = pca.fit(features).transform(features)

	# Reduces features from Wavelet to 3
	pca = PCA(n_components=2, svd_solver='full')
	pcaFeatures = pca.fit(featuresW).transform(featuresW)

	# Reduces all features extracted to 3
	allFeatures=np.hstack((features,featuresW))
	pca = PCA(n_components=2, svd_solver='full')
	pcaFeatures = pca.fit(allFeatures).transform(allFeatures)

	#Extract all features.
	live_traffic, oClass_life= extractFeatures(live_data_train,Class=3)

	features_traffic,oClass_traffic = extractFeatures(traffic_data_test,Class=0)
	features_traffic2,oClass_traffic2 = extractFeatures(traffic_data_test2,Class=1)
	features_traffic3,oClass_traffic3 = extractFeatures(traffic_data_test3,Class=2)
	#testFeatures=np.vstack((features_traffic,features_traffic2,features_traffic3))
	#testFeatures=np.vstack((features_traffic,features_traffic2,features_traffic3,live_traffic))
	testFeatures= live_traffic

	features_live_W,oClass_live = extractFeaturesWavelet(live_data_train,scales,Class=3) 

	features_plot1_W,oClass_plot1=extractFeaturesWavelet(traffic_data_test,scales,Class=0)
	features_plot2_W,oClass_plot2=extractFeaturesWavelet(traffic_data_test2,scales,Class=1)
	features_plot3_W,oClass_plot3=extractFeaturesWavelet(traffic_data_test3,scales,Class=2)
	#testFeaturesW=np.vstack((features_plot1_W,features_plot2_W,features_plot3_W))
	#testFeaturesW=np.vstack((features_plot1_W,features_plot2_W,features_plot3_W,features_live_W))
	testFeaturesW=features_live_W

	alltestFeatures=np.hstack((testFeatures,testFeaturesW))

	scaler=StandardScaler()
	NormAllFeatures=scaler.fit_transform(allFeatures)
	NormAllTestFeatures=scaler.fit_transform(alltestFeatures)

	pca = PCA(n_components=4, svd_solver='full')
	NormPcaFeatures = pca.fit(NormAllFeatures).transform(NormAllFeatures)
	NormTestPcaFeatures = pca.transform(NormAllTestFeatures)

	from sklearn.neural_network import MLPClassifier
	print('\nClassification based on Neural Networks \n')

	alpha=1
	max_iter=500000
	clf = MLPClassifier(solver='lbfgs',alpha=alpha,hidden_layer_sizes=(100,),max_iter=max_iter)
	clf.fit(NormPcaFeatures, oClass) 
	LT=clf.predict(NormTestPcaFeatures) 
	print('class (from test PCA):',LT)

	nObsTest,nFea=NormTestPcaFeatures.shape
	for i in range(nObsTest):
		print('Obs: {:2}: Classification->{}'.format(i,Classes[LT[i]]))




plotEnable = 0;
#main()
main_v2("test.dat")