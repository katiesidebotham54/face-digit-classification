# This file is used to test naiveBayes.py on faces and digits data


import subprocess
import datetime

# clear results.txt file
with open('nb_results.txt', 'w') as file:
    pass

# Values for training data with 10% increase; Use arrays to show improved accuracy with larger datasets
faces = [45, 90, 135, 180, 225, 270, 315, 360, 405, 451]
digits = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
iterations = 5

# Run Naive Bayes
for amount in faces:
    start_faces = datetime.datetime.now()
    for i in range(iterations):
        print(f"Running Naive Bayes Faces: {amount}")
        subprocess.call(
            f"python dataClassifier.py -c naiveBayes -d faces -t {amount} >> nb_results.txt", shell=True)
    end_faces = datetime.datetime.now()
    print(f"{(end_faces - start_faces).total_seconds()} seconds")

for amount in digits:
    start_digits = datetime.datetime.now()
    for i in range(iterations):
        print(f"Running Naive Bayes Digits: {amount}")
        subprocess.call(
            f"python dataClassifier.py -c naiveBayes -d digits -t {amount}  >> nb_results.txt", shell=True)
    end_digits = datetime.datetime.now()
    print(f"{(end_digits - start_digits).total_seconds()} seconds")
