#!/usr/bin/env python3

import sys
import numpy as np

filename = sys.argv[1]
file = open(filename, 'r')
firstline = file.readline()
firstline = firstline.replace('\n','',1)
firstfields = firstline.split(",")
accumulated = firstfields;
for i in range(len(firstfields)):
	if firstfields[i].replace('.','',1).isdigit():
		accumulated[i] = float(firstfields[i])
linecount = 1
for line in file:
	linecount += 1
	line = line.replace('\n','',1)
	#print(line)
	fields = line.split(",")
	for i in range(len(fields)):
		if fields[i].replace('.','',1).isdigit():
			accumulated[i] += float(fields[i])

for i in range(len(accumulated)):
	if isinstance(accumulated[i], float):
		accumulated[i] /= linecount


#print(linecount)
#final = ','.join(accumulated)
print(str(accumulated).replace('[','',1).replace(']','',1).replace('\'','').replace(' ',''))
