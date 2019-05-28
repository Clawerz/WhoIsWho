import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt 
import sys


def waitforEnter(fstop=True):
        if fstop:
                if sys.version_info[0] == 2:
                        raw_input("Press ENTER to continue.")
                else:
                        input("Press ENTER to continue.")
####1
def plot(data,name):
    plt.plot(data)
    plt.title(name)
    plt.show()
    waitforEnter()

####2
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
	order=np.arange(nObs)   #Comment out to random split
        
	nTrain=int(nObs*trainPerc)
	
	data_train=data_obs[order[:nTrain],:]
	data_test=data_obs[order[nTrain:],:]

	#print(data_train)
	return(data_train,data_test)

####3
def extractFeatures(data,Class=0):
        features=[]
        nObs=data.shape[0]#sum(1 for _ in data)
        oClass=np.ones((nObs,1))*Class

        for i in range(nObs):
	        M1=np.mean(data,axis=0)
	        print('Mean',M1)
	        Md1=np.median(data,axis=0)
	        print('Median',Md1)
	        Std1=np.std(data,axis=0)
	        print('Deviation',Std1)
	        S1=stats.skew(data)
	        print('Skew',S1)
	        K1=stats.kurtosis(data)
	        print('Kurtosis',K1)
	        p=[75,90,95]
	        Pr1=np.array(np.percentile(data,p,axis=0)).T.flatten()
	        print('Percentile(75, 90, 95)',Pr1)
	        
	        faux=np.hstack((M1,Md1,Std1,S1,K1,Pr1))
	        faux=np.hstack((M1,Std1,Pr1))
	        features.append(faux)

        return(np.array(features),oClass)

## -- 4 -- ##
def plotFeatures(features,oClass,f1index=0,f2index=1):
	nObs,nFea=features.shape
	#print(features[0][0])
	#print(nObs,nFea)
	colors=['b','g','r']
	for i in range(nFea):
		plt.plot(features[0][i],'o'+colors[int(oClass[i])])

	plt.show()
	waitforEnter()

################ Main Code #######################

# 1
# Read data from files and plot the download + upload traffic
# We don't need to verify if it is download or upload at least by now
traffic_data = np.loadtxt(sys.argv[1])
plt.figure(1)
plot(traffic_data,'Plot Title')

# 2
# Divide each stream in observation windows of 5 minutes
# Divide randomly a set for training and one for testing
traffic_data_train, traffic_data_test= breakTrainTest(traffic_data)
#print(traffic_data_train)
plt.figure(2)
for i in len(traffic_data_train):
	plt.plot(traffic_data_train[i],'b')
	plt.plot(traffic_data_test[i],'g')
plt.title('Adult')
plt.ylabel('Bytes/sec')
plt.show()
waitforEnter()

'''
# 3
# Extract and print features
features_traffic, oClass_traffic = extractFeatures(traffic_data_train[0],Class=0)

features= np.vstack(features_traffic)
oClass= np.vstack(oClass_traffic)

#print(oClass)

print('Train Stats Features Size:', features.shape)


####4
# TODO : fix
plt.figure(4)
plotFeatures(features, oClass, 0, 1)
'''


