import time
import sys
from sniffer import sniffer_main
from ProfileClass import main_v2

# Creates a dataset file 
# Two arguments passed command line, first dataset name and second dataset generation timer
def main():

	# Generate a live dataset with 5 min traffic
	elapsed_time = 0
	start_time = time.time()
	byte_set = []

	print('Capturing traffic and defining your profile ...')
	while (elapsed_time < 2*5):
		byte_set.append(sniffer_main(1,1))
		elapsed_time = time.time() - start_time
	
	#print(*byte_set, sep = ", ")  
	createDat('live.dat', byte_set)

	# Get evaluation
	main_v2(normalizeDataset('live.dat'))

# Creates a .dat file with the dataset generated
def createDat(name,data):

	f = open(name, "w")
	for i in data:
		f.write(str(i[0])+' '+str(i[1])+'\n')
	#nRows,nCols = data.shape;
	#print(nRows, nCols)
	#f.write(data)
	f.close()

# Normalize dataset
def normalizeDataset(name):
	
	# Counts number of lines
	with open(name) as f:
   		size=sum(1 for _ in f)
	#print(size) 

	# Replicates lines until the there are exactly 6000 lines
	f1 = open('live_normalize.dat', "w")
	while(size < 6000):
		with open(name, "r") as f:
			for line in f:
				if(size < 6000):
					f1.write(line)
					size = size+1

	f1.close()
	return 'live_normalize.dat'

main()
