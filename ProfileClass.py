import numpy as np 
import matplotlib.pyplot as plt 
import sys


def waitforEnter(fstop=True):
	if fstop:
		if sys.version_info[0] == 2:
			raw_input("Press ENTER to continue.")
		else:
			input("Press ENTER to continue.")
			
def plot(data,name):
    plt.plot(data)
    plt.title(name)
    plt.show()
    waitforEnter()

def breakTrainTest(data,oWnd=300,trainPerc=0.5):
	nSamp,nCols=data.shape
	nObs=int(nSamp/oWnd)
	data_obs=data.reshape((nObs,oWnd,nCols))
	
	order=np.random.permutation(nObs)
	order=np.arange(nObs)	#Comment out to random split
	
	nTrain=int(nObs*trainPerc)
	
	data_train=data_obs[order[:nTrain],:,:]
	data_test=data_obs[order[nTrain:],:,:]
	
	return(data_train,data_test)

def extractFeatures(data,Class=0):
	features=[]
	nObs,nSamp,nCols=data.shape
	oClass=np.ones((nObs,1))*Class
	for i in range(nObs):
		M1=np.mean(data[i,:,:],axis=0)
		#Md1=np.median(data[i,:,:],axis=0)
		Std1=np.std(data[i,:,:],axis=0)
		#S1=stats.skew(data[i,:,:])
		#K1=stats.kurtosis(data[i,:,:])
		p=[75,90,95]
		Pr1=np.array(np.percentile(data[i,:,:],p,axis=0)).T.flatten()
		
		#faux=np.hstack((M1,Md1,Std1,S1,K1,Pr1))
		faux=np.hstack((M1,Std1,Pr1))
		features.append(faux)
		
	return(np.array(features),oClass)

################ Main Code #######################
traffic_data = np.loadtxt(sys.argv[1])
plt.figure(1)
plot(traffic_data,'Plot Title')

#####
features_traffic, oClass_traffic = extractFeatures()