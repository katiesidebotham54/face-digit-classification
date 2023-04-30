import subprocess
import datetime

# clear results.txt file
with open('results.txt', 'w') as file:
    pass

# Append values to the face and digit arrays to see how the algorithms perform with larger percentages of training data.
# Only using 10% of each data set to make it quick.
# You can also choose how many iterations of each to run.
# Note: The values for 10%-100% of training data for Faces are as follows:
# 45, 90, 135, 180, 225, 270, 315, 360, 405, 451
# For digits:
# 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000
face_array = [45, 90, 135, 180, 225, 270, 315, 360, 405, 451]
digit_array = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
iterations = 1

# Choose which algorithms to run
run_naive_bayes = True
run_perceptron = True
run_knn = True

# Used to globally track how much time has passed
start_time = datetime.datetime.now()

# -----------------------------------------------------NAIVE BAYES-----------------------------------------------------------
if run_naive_bayes:
    for amount in face_array:
        start_faces = datetime.datetime.now()
        for i in range(iterations):
            print(f"Running naiveBayes Faces: {amount}")
            subprocess.call(
                f"python dataClassifier.py -c naiveBayes -d faces -t {amount} -s 150 >> results.txt", shell=True)
        end_faces = datetime.datetime.now()
        print(f"{(end_faces - start_faces).total_seconds()} seconds")

    for amount in digit_array:
        start_digits = datetime.datetime.now()
        for i in range(iterations):
            print(f"Running naiveBayes Digits: {amount}")
            subprocess.call(
                f"python dataClassifier.py -c naiveBayes -d digits -t {amount} -s 1000 >> results.txt", shell=True)
        end_digits = datetime.datetime.now()
        print(f"{(end_digits - start_digits).total_seconds()} seconds")

# -----------------------------------------------------PERCEPTRON-----------------------------------------------------------
if run_perceptron:
    for amount in face_array:
        start_faces = datetime.datetime.now()
        for i in range(iterations):
            print(f"Running Perceptron Faces: {amount}")
            subprocess.call(
                f"python dataClassifier.py -c perceptron -d faces -t {amount} -i 2 -s 150 >> results.txt", shell=True)
        end_faces = datetime.datetime.now()
        print(f"{(end_faces - start_faces).total_seconds()} seconds")

    for amount in digit_array:
        start_digits = datetime.datetime.now()
        for i in range(iterations):
            print(f"Running Perceptron Digits: {amount}")
            subprocess.call(
                f"python dataClassifier.py -c perceptron -d digits -t {amount} -s 1000 >> results.txt", shell=True)
        end_digits = datetime.datetime.now()
        print(f"{(end_digits - start_digits).total_seconds()} seconds")

# -------------------------------------------------k NEAREST NEIGHBOR--------------------------------------------------------
if run_knn:

    for amount in face_array:
        start_faces = datetime.datetime.now()
        for i in range(iterations):
            print("Running k Nearest Neighbor Faces:", amount)
            os.system(
                f"python dataClassifier.py -c knn -d faces -t {amount} -s 150 >> results.txt")
        end_faces = datetime.datetime.now()
        print(f"{end_faces - start_faces} seconds")

    for amount in digit_array:
        start_digits = datetime.datetime.now()
        for i in range(iterations):
            print("Running k Nearest Neighbor Digits:", amount)
            subprocess.call(
                f"python dataClassifier.py -c kNN -d digits -t {amount} -s 1000 >> results.txt")
        end_digits = datetime.datetime.now()
        print(f"{end_digits - start_digits} seconds")

end_time = datetime.datetime.now()
print(f"{end_time - start_time} seconds (TOTAL)")
