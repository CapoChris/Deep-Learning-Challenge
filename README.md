# AlphabetSoupCharity Neural Network Report
- To Run: Open the notebooks in Google Colab or Jupyter, run all cells in order. Models are saved automatically.

## Overview

The objective of this project was to develop a binary classification neural network to predict whether organizations funded by Alphabet Soup would be successful (1) or unsuccessful (0). Using historical records from charity_data.csv, I:

- Preprocessed the dataset for modeling

- Built an initial deep learning model using TensorFlow/Keras

- Conducted iterative optimization trials to improve accuracy

The final submission includes:

- A base model in the starter code

- An optimized model in AlphabetSoupCharity_Optimization.ipynb

- This written technical summary

## Data Preprocessing

- Target Variable: IS_SUCCESSFUL (1 for success, 0 for failure)

- Dropped Columns:

  - EIN and NAME (identifiers)

  - SPECIAL_CONSIDERATIONS_Y, USE_CASE_Other, and INCOME_AMT_0 (minimal predictive value)

- Techniques Used:

  - One-hot encoding via pd.get_dummies() for categorical features

  - Data split with train_test_split

- Feature normalization using StandardScaler

## Initial Model Development

- Architecture:

  - 3 layers: [80 → 30 → 1 neurons]

- Activations: ReLU (hidden), Sigmoid (output)

- Loss Function: Binary Crossentropy

- Regularization: EarlyStopping to prevent overfitting

- Performance:

  - Accuracy: ~57% on test data

  - Served as a baseline for future improvements

## Model Optimization Attempts

- Attempt 1: Increased Model Complexity

  - Changes:

    - Expanded to 4 layers: [256 → 128 → 64 → 1]
  
    - Mixed activation functions (ReLU + Tanh)
  
    - Increased number of training epochs
  
  - Result: ~44% accuracy; performance declined due to potential overfitting or poor function choices

- Attempt 2: Simplified Input Data

  - Changes:

    - Removed additional low-signal columns

    - Maintained simpler architecture

  - Result: Slight improvement, but still below baseline

- Attempt 3: Dropout Regularization

  - Changes:

    - Reintroduced complex architecture

    - Added 20% dropout after each hidden layer

    - Mixed activations: ReLU + Tanh

  - Result: Accuracy improved to ~52.9%, better than Attempt 1, but still not surpassing the original

## Conclusion

Despite multiple optimization attempts, the initial model remained the best, with an accuracy of approximately 57%. However, the experimentation process was valuable and offered the following insights:

 - Model complexity does not guarantee better results

  - Thoughtful feature selection and regularization are critical

  - Activation functions significantly influence outcomes

