# Deep-Learning-Challenge

# Overview of Analysis

The goal of this machine learning analysis is to predict whether applicants will be successful if funded by AlphabetSoup, a charity organization. This classification model helps optimize which organizations to support based on historical application data, improving funding allocation efficiency and maximizing social impact.

# 1. Data Processing

  - Target Variable: IS_SUCCESSFUL (This column indicates whether the funded organization was successful (1) or not (0))
  - Features Used (after encoding and cleaning):
    - APPLICATION_TYPE
    - AFFILIATION
    - CLASSIFICATION
    - USE_CASE
    - ORGANIZATION
    - STATUS
    - INCOME_AMT
    - SPECIAL_CONSIDERATION
    - ASK_AMT
   
  - Columns Removed: EIN and NAME

# 2. Compiling, Training, and Evaluating the Model

I developed a neural network model using TensorFlow and Keras, implementing two hidden dense layers with 80 and 30 neurons, respectively. The model was trained over 100 epochs and evaluated using test data to assess loss and accuracy. Upon completion, the finalized model was exported as an HDF5 file for future deployment or further tuning.

I cleaned and preprocessed the data by consolidating low-frequency categorical values into an "Other" category (e.g., APPLICATION_TYPE, CLASSIFICATION) and applied one-hot encoding. For model tuning, I experimented with different architectures by adjusting the number of layers and neurons, testing various activation functions like ReLU and tanh, and incorporating additional or dropout layers as needed. The model was compiled using binary cross-entropy as the loss function and optimized with the Adam optimizer. Throughout training, I monitored performance by reviewing accuracy and loss trends over epochs and validated the model to guard against overfitting.

# 3. Results

The model achieved an accuracy of approximately 73% with a loss exceeding 0.5, falling short of the 75% target benchmark. Despite experimenting with additional layers, adjusting the number of neurons per layer, and tuning the number of training epochs, the model's performance remained below the desired threshold.
