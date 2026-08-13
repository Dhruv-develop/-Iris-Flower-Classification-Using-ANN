# 🌸 Iris Flower Classification Using Perceptron & ANN

## 📌 Project Overview

This project implements both a **Perceptron** and an **Artificial Neural Network (ANN)** to classify Iris flower species using the famous **Iris Dataset**.

The Iris dataset contains measurements of flower characteristics:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

Using these features, the models learn patterns and classify flowers into one of three species.

This project demonstrates the practical application of **Machine Learning and Deep Learning** for multi-class classification using **Scikit-Learn and TensorFlow/Keras**.

---

## 🎯 Objective

The primary objective of this project is to compare the performance of a traditional **Perceptron** model and an **Artificial Neural Network (ANN)** on the Iris dataset.

The project covers:

* Data Exploration
* Feature Scaling
* Label Encoding
* Train-Test Split
* Perceptron Implementation
* ANN Implementation
* Model Training
* Model Evaluation
* Species Prediction

---

## 📊 Dataset

The project uses the **Iris Dataset** available through Seaborn.

### Dataset Source

```python
df = sns.load_dataset("iris")
```

### Dataset Statistics

| Item          | Value |
| ------------- | ----: |
| Total Samples |   150 |
| Features      |     4 |
| Classes       |     3 |

### Features

| Feature      |
| ------------ |
| Sepal Length |
| Sepal Width  |
| Petal Length |
| Petal Width  |

### Target Classes

```text
Setosa
Versicolor
Virginica
```

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* Scikit-Learn
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## 📈 Exploratory Data Analysis

The dataset was explored using visualizations and statistical analysis.

### Pair Plot

```python
sns.pairplot(df, hue="species")
```

The pair plot helps visualize relationships between the features and understand how the three Iris species are distributed.

---

## ⚙️ Data Preprocessing

### 1. Train-Test Split

The dataset was divided into training and testing sets to evaluate the models on unseen data.

### 2. Label Encoding

The target variable was converted into numerical format using `LabelEncoder`.

```python
le = LabelEncoder()

y_encoded = le.fit_transform(y)
```

The classes are converted into numerical labels:

```text
Setosa      → 0
Versicolor  → 1
Virginica   → 2
```

### 3. Feature Scaling

`StandardScaler` was used to standardize the feature values.

```python
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
```

Scaling is particularly useful for the Perceptron and ANN because it puts the input features on a comparable scale.

### 4. One-Hot Encoding

For ANN training with categorical crossentropy, the target labels were converted into one-hot encoded vectors.

```python
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)
```

For example:

```text
Class 0 → [1, 0, 0]
Class 1 → [0, 1, 0]
Class 2 → [0, 0, 1]
```

---

## 🧠 Perceptron Model

A Perceptron classifier was implemented using Scikit-Learn.

```python
per = Perceptron(
    max_iter=1000,
    random_state=42
)
```

### Workflow

```text
Input Features
      ↓
   Perceptron
      ↓
Class Prediction
```

The Perceptron is a basic neural-network-inspired linear classifier and provides a useful baseline for comparison with the ANN.

---

## 🧠 Artificial Neural Network (ANN)

The Artificial Neural Network consists of two hidden layers with ReLU activation and dropout regularization.

### Architecture

```text
Input Layer (4 Features)
        ↓
Dense (16 Neurons, ReLU)
        ↓
Dropout (0.2)
        ↓
Dense (8 Neurons, ReLU)
        ↓
Dropout (0.2)
        ↓
Dense (3 Neurons, Softmax)
        ↓
Output Layer
```

### Model Architecture

| Layer              | Purpose                                    |
| ------------------ | ------------------------------------------ |
| Dense (16, ReLU)   | Learn relationships between input features |
| Dropout (0.2)      | Reduce overfitting                         |
| Dense (8, ReLU)    | Learn higher-level feature patterns        |
| Dropout (0.2)      | Additional regularization                  |
| Dense (3, Softmax) | Multi-class classification                 |
| Softmax            | Generate class probabilities               |

---

## 📈 Model Training

The ANN was compiled using the **Adam optimizer** and **Categorical Crossentropy** loss function.

```python
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

### Training Configuration

* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy
* **Metric:** Accuracy
* **Validation:** Validation split
* **Training:** Batch-based training

---

## 🔄 Project Workflow

```text
Iris Dataset
      ↓
Data Exploration
      ↓
Train-Test Split
      ↓
Label Encoding
      ↓
Feature Scaling
      ↓
      ├───────────────┐
      ↓               ↓
 Perceptron          ANN
      ↓               ↓
      └───────┬───────┘
              ↓
          Evaluation
              ↓
       Species Prediction
```

---

## 📊 Model Evaluation

Both models were evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report

For the ANN, training history was also visualized using:

* Training Accuracy
* Validation Accuracy
* Training Loss
* Validation Loss

These metrics help evaluate model performance and identify potential overfitting during training.

---

## 💡 Key Learning Outcomes

Through this project, I gained practical experience with:

* Machine Learning Fundamentals
* Perceptron Algorithm
* Artificial Neural Networks
* Deep Learning Basics
* Multi-Class Classification
* Feature Scaling
* Label Encoding
* One-Hot Encoding
* Softmax Activation
* ReLU Activation
* Dropout Regularization
* TensorFlow/Keras
* Scikit-Learn
* Model Evaluation
* Confusion Matrix
* Classification Report

---

## 📁 Project Structure

```text
Iris-Flower-Classification-Using-ANN/
│
├── Iris_ANN_PROJECT.ipynb
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Iris-Flower-Classification-Using-ANN.git
```

### 2. Install Dependencies

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

### 3. Open the Notebook

```text
Iris_ANN_PROJECT.ipynb
```

### 4. Run the Project

Follow the notebook workflow:

1. Load the Iris dataset
2. Explore the data
3. Perform train-test split
4. Encode target labels
5. Scale the features
6. Train the Perceptron model
7. Build and train the ANN
8. Evaluate both models
9. Visualize model performance
10. Predict Iris flower species

---

## 📌 Conclusion

This project demonstrates the implementation and comparison of a **Perceptron** and an **Artificial Neural Network (ANN)** for Iris flower classification.

The project applies important machine learning and deep learning concepts such as **feature scaling, label encoding, one-hot encoding, ReLU, Softmax, dropout regularization, and multi-class classification**.

Overall, this project provides practical experience in building, training, evaluating, and comparing traditional machine learning and neural network models using **Scikit-Learn and TensorFlow/Keras**.

---

## 👨‍💻 Author

**Dhruv Rapariya**

*M.Sc. CA & IT | Data Science Enthusiast*

### Skills Demonstrated

`Python` `TensorFlow` `Keras` `ANN` `Perceptron` `Deep Learning` `Classification` `Machine Learning` `Softmax` `ReLU` `Scikit-Learn` `Pandas` `NumPy` `Seaborn`
