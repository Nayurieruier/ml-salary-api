import numpy as np


#Question 1
prices = np.array([10, 20, 30, 40])
discounts=np.round(prices*0.08+prices)
#Question 2
scores = np.array([55, 92, 78, 40, 88, 63, 71])
above_60=scores>60
below_60=scores<60
#print(scores[above_60])
#print(scores[below_60])

#Question 3

grades = np.array([
    [85, 90, 78],   # student 1
    [60, 95, 88],   # student 2
    [70, 72, 91]    # student 3
])
student_avg=grades.mean(axis=1)
subject_avg=grades.mean(axis=0)

#print(student_avg)
#print(subject_avg)

#Question 4

data=np.array([1,2,3,4,5,6])
expected=[1,4]
result = []
for x in data:
    if x % 2 == 0:
        result.append(x ** 2)
    else:
        result.append(x ** 3)
result = np.where(data%2==0, data**2,data**3)
#print(result)

#Question 5

X = np.array([[1, 2], [3, 4], [5, 6]])   # 3 samples, 2 features
weights = np.array([0.5, -1.2])           # 2 weights
p=X.dot(weights)
#print(p)
#print(np.matmul(X, weights))


#Question 6

np.random.seed(10)
mat=np.random.randint(low=0, high=99, size=10)
#or index in np.ndindex(mat.shape):
 #   print(index)

#Question 7



# 1. Create dummy data (100 samples, 4 features)
np.random.seed(42)  # For reproducible results
X = np.random.rand(100, 4)
y = np.random.randint(0, 2, size=100)

# 2. Define the split ratio (e.g., 80% train, 20% test)
train_ratio = 0.8
num_samples = len(X)
num_train = int(num_samples * train_ratio)

# 3. Create a shuffled array of indices
indices = np.arange(num_samples)
np.random.shuffle(indices)

# 4. Split the indices into train and test sets
train_indices = indices[:num_train]
test_indices = indices[num_train:]

# 5. Extract the actual data using fancy indexing
X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Verify the shapes
#print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
#print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
#Questio  8
matrix = np.random.randint(0, 100, size=(5, 5))
avg=matrix.mean(axis=0)
replace_value=np.where(matrix.mean(axis=0),matrix-avg,matrix)

#More Questions

matrix = np.array([
    [10, 20, 30],
    [5, 15, 25],
    [100, 200, 300]
])
row_sums = matrix.sum(axis=1, keepdims=True)
normalized = matrix / row_sums
#print(normalized)
#print(normalized.sum(axis=1))  # sanity check — should be [1. 1. 1.]

#Question 2

predictions = np.array([
    [0.1, 0.7, 0.2],
    [0.6, 0.3, 0.1],
    [0.2, 0.2, 0.6]
])
row=predictions.argmax(axis=1,keepdims=False)

#Question 3

feature_importances = np.array([0.05, 0.42, 0.12, 0.31, 0.10])
np.argsort(feature_importances)
# → [0, 4, 2, 3, 1]   (indices, from smallest value to largest)
sorted_desc = np.argsort(feature_importances)[::-1]
top_3 = sorted_desc[:3]
#print(top_3)                          # → [1, 3, 2]
#print(feature_importances[top_3])     # → [0.42, 0.31, 0.12] — descending, correct

# Question 4
feature_a = np.array([1, 2, 3])
feature_b = np.array([4, 5, 6])
feature_c = np.array([7, 8, 9])
np.vstack([feature_a, feature_b, feature_c])
# [[1, 2, 3],
#  [4, 5, 6],
#  [7, 8, 9]]
np.column_stack([feature_a, feature_b, feature_c])
# [[1, 4, 7],
#  [2, 5, 8],
#  [3, 6, 9]]
feature_matrix = np.column_stack([feature_a, feature_b, feature_c])
#print(feature_matrix)
#print(feature_matrix.shape)  # (3, 3)

#Question 5

data = np.array([5, -3, 8, -1, 0, -7, 12])
data=np.where(data>=0,data,0)
print(data)

