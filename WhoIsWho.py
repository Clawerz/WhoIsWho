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

	while (elapsed_time < 60*5):
		byte_set.append(sniffer_main(1))
		elapsed_time = time.time() - start_time
	
	print(*byte_set, sep = ", ")  
	createDat('live.dat', byte_set)

	# Get evaluation
	main_v2('test.dat')

# Creates a .dat file with the dataset generated
def createDat(name,data):

	f = open(name, "w")
	for i in data:
		f.write(str(i[0])+' '+str(i[1])+'\n')
	#nRows,nCols = data.shape;
	#print(nRows, nCols)
	#f.write(data)
	f.close()

main()
