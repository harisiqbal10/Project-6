## Project-6: Privacy-Preserving Machine Learning

In this project, our focus is on implementing the Differentially Private ID3 algorithm. This algorithm utilizes a 𝜀-DP (Differentially Private) setting to create a decision tree training algorithm. The primary objective is to train the model on a given dataset, allowing it to make predictions based on learned patterns. This README provides instructions on how to run the Privacy-Preserving Machine Learning code. 

### Prerequisites

Before running the code, make sure you have the following prerequisites installed:

Python 3.x

### A. How to Run the Code

1. **Clone this Repository:**

```bash
git clone https://github.com/harisiqbal10/Project-6.git
cd Project-6
```

2. **Install Dependencies:**

Required Python libraries (You can install them using `pip install <library_name>`):

`pandas`

`numpy`

3. **Run the Code:**

```bash
python privacy_preserving_ml.py
```

4. **Parameters:**

`epsilon`: Privacy parameter. Adjust based on privacy requirements.

`d`: Maximum depth of the decision tree.


### B. Alternative Inputs

To run the code with alternative inputs:

**Dataset**: Replace the dataset URL in the code with the URL of your preferred classification dataset.

**Attributes**: Modify the `attributes` list to include the attributes of your dataset.

**Columns**: Modify the `columns` list to include the columns of your dataset in order. 

**Epsilon**: Set your preffered epsilon value.

**d**: Set your desired max depth of the descison tree. 

All of the above inputs are in the main function. 

In the write-up below, we analyze our program parameters, exploring their implications for privacy, accuracy, and efficiency.

### Program Parameters and their implications for privacy, accuracy, and efficiency

The main parameters that impact the privacy, accuracy, and efficiency of the program are `epsilon`, which controls the privacy level, and `d`, which controls the maximum depth of the decision tree. Increasing `epsilon` generally improves accuracy but reduces privacy, as it allows more noise in the Laplace mechanism. On the other hand, increasing `d` may improve accuracy but could lead to overfitting and increased computation time.

It's essential to maintain a balance between privacy and accuracy based on the specific requirements of your use case. Additionally, consider the dataset's characteristics and adjust parameters accordingly. Monitoring the accuracy-privacy trade-off is crucial for making informed decisions about parameter tuning.

Beyond privacy and accuracy, efficiency is a key consideration. The Laplace noise addition and tree construction process introduce computational overhead. Larger values of `epsilon` and `d` can increase the computational cost, potentially impacting the efficiency of the algorithm. It's essential to maintain a balance between model complexity and computational efficiency, especially in resource-constrained environments.
