# This file is used to calculate the mean and standard deviation from test results files

import sys
import statistics

dataType = sys.argv[1]
fileName = str(sys.argv[2])
iterations = int(sys.argv[3])

resultFile = [l[:-1] for l in open(fileName, 'rt').readlines()]
fileList = []
for line in resultFile:
    fileList.append(line)

outputName = "calculations.txt"
outputFile = open(outputName, 'w')  # Use 'w' to reset the file

outputFile.write(dataType)
outputFile.write('\n')

percentList = []
for line in fileList:
    if 'correct out of' in line and '%' in line and 'Test_Percent' in line:
        start = line.index('(') + 1
        end = line.index('%')
        temp = line[start:end]
        real_num = float(temp)
        percentList.append(real_num)


for data in ['faces', 'digits']:
    outputFile.write('\n')
    outputFile.write(f"Data: {data}\n")
    for i in ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']:
        smallList = []
        for j in range(iterations):
            if percentList:
                smallList.append(percentList.pop(0))

        if smallList:
            average = statistics.mean(smallList)
            if len(smallList) > 1:
                std_dev = statistics.stdev(smallList)
            else:
                std_dev = 0
            message = i + ": " + str(smallList) + "  " + "Accuracy: " + \
                str(average) + "  Standard Deviation: " + str(std_dev) + '\n'
            outputFile.write(message)
        else:
            outputFile.write(f"No data for {i}\n")
    outputFile.write('------------------\n')

outputFile.close()
