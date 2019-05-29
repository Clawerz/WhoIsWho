import time
import sys
from sniffer import sniffer_main

# Creates a dataset file 
# Two arguments passed command line, first dataset name and second dataset generation timer
def mainGen():

	# Elapsed time in seconds
	# To create a profile we will start by evaluating only 1 minute
	# While under 1 minute, capture packets and generate a name.dat file
	# with the nubmer of bytes captured per second, this will indirectly
	# give us the traffic type and allow to distinguish between each profile
	elapsed_time = 0
	start_time = time.time()
	byte_set = []

	# sys.argv[2] second argument passed command line
	while elapsed_time < int(sys.argv[2]):
		byte_set.append(sniffer_main(1));
		elapsed_time = time.time() - start_time;
	
	print(*byte_set, sep = ", ")  
	createDat(sys.argv[1]+'.dat', byte_set)

# Creates a .dat file with the dataset generated
def createDat(name,data):

	f = open(name, "w")
	for i in data:
		f.write(str(i[0])+' '+str(i[1])+'\n')
	#nRows,nCols = data.shape;
	#print(nRows, nCols)
	#f.write(data)
	f.close()

mainGen()
