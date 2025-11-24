# airfoil-ml-pytorch

This project/repository aim is train a model capable of predicting Airfoil Lift/Drag Coefficient using Pytorch.

The dataset utilized in this project is a comprehensive collection of airfoil data, available through: https://github.com/nasa/airfoil-learning.
The power of XFOIL is its ability to generate vast amounts of data. However, manually running and analyzing simulations for numerous airfoils across various conditions is time-consuming.
The dataset consists of thousands of XFOIL simulations, covering a wide range of airfoils under various aerodynamic conditions, including:
*   Angle of Attack (Alpha)
*   Reynolds Number
*   Ncrit
And provides the pressure distribution and aerodynamics coefficients.

The primary goal of this project is to experiment with this rich dataset.

The project's initial workflow involved converting the raw JSON data into a single CSV for easy manipulation. However, this approach quickly proved inefficient due to the dataset's size, leading to high storage demands and slow data loading. To address this, the data was migrated to .npy format, which drastically improved I/O performance.

The first machine learning experiments were conducted using TensorFlow. We implemented a Multilayer Perceptron (MLP) and used this phase to establish a baseline. Testing various techniques like `StandardScaler` and `MinMaxScaler` , Comparing the performance of `ReLU`, `Tanh`, and others. Employing K-Fold cross-validation to get a robust estimate of model performance.
Then project was migrated to PyTorch (why? It was easier to use CUDA with pytorch).
