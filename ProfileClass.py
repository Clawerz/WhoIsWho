import numpy as np 
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
        nSamp=data.shape
        nObs=int(nSamp/oWnd)
        data_obs=data.reshape((nObs,oWnd))
        
        order=np.random.permutation(nObs)
        order=np.arange(nObs)   #Comment out to random split
        
        nTrain=int(nObs*trainPerc)
        
        data_train=data_obs[order[:nTrain],:,:]
        data_test=data_obs[order[nTrain:],:,:]
        
        return(data_train,data_test)
####3
def extractFeatures(data,Class=0):
        features=[]
        nObs=sum(1 for _ in data)
        print(nObs)

        oClass=np.ones((nObs,1))*Class
        M1=np.mean(data,axis=0)
        print('Mean',M1)
        Md1=np.median(data,axis=0)
        print('M')
        Std1=np.std(data,axis=0)
        #S1=stats.skew(data)
        #K1=stats.kurtosis(data)
        p=[75,90,95]
        Pr1=np.array(np.percentile(data,p,axis=0)).T.flatten()
        
        #faux=np.hstack((M1,Md1,Std1,S1,K1,Pr1))
        faux=np.hstack((M1,Std1,Pr1))
        features.append(faux)
        
        return(np.array(features),oClass)

################ Main Code #######################

####1
traffic_data = np.loadtxt(sys.argv[1])
plt.figure(1)
plot(traffic_data,'Plot Title')

#####2
#traffic_data_train, traffic_data_test= breakTrainTest(traffic_data)
#plt.figure(2)
#for i in range(10):
#       plt.plot(traffic_data_train[i,:,0],'b')
#       plt.plot(traffic_data_train[i,:,1],'g')
#plt.title('Adult')
#plt.ylabel('Bytes/sec')
#plt.subplot(3,1,2)
#plt.show()
#waitforEnter()
#####3
features_traffic, oClass_traffic = extractFeatures(traffic_data,Class=0)

features= np.vstack(features_traffic)
oClass= np.vstack(oClass_traffic)

print('Train Stats Features Size:', features.shape)
