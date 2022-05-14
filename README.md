# Neural_Network_Charity_Analysis

Neural Networks and Deep Learning Models
For this part of the Challenge, we are preparing a report on the performance of the deep learning model we created for AlphabetSoup.

The report contains the following:

## Overview of the analysis: Explain the purpose of this analysis.

  * Compare the differences between the traditional machine learning classification and regression models and the neural network models.
  * Describe the perceptron model and its components.
  * Implement neural network models using TensorFlow.
  * Explain how different neural network structures change algorithm performance.
  * Preprocess and construct datasets for neural network models.
  * Compare the differences between neural network models and deep neural networks.
  * Implement deep neural network models using TensorFlow.
  * Save trained TensorFlow models for later use.
  
## Purpose

Our Client this time is charitable foundation, Alphabet Soup, wants to predict where to invest? The goal is to use machine learning and neural networks to apply features on a provided dataset to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup. The initial file has 34,000 organizations and a number of columns that capture metadata about each organization from past successful fundings.

## Results: Using bulleted lists and images to support your answers, address the following questions.

### Data Preprocessing

  1. What variable(s) are considered the target(s) for your model?
    - Checking to see if the target is marked as successful in the DataFrame, indicating that it has been successfully funded by AlphabetSoup.
    
  2. What variable(s) are considered to be the features for your model?
    - The IS_SUCCESSFUL column is the feature chosen for this dataset.

  3. What variable(s) are neither targets nor features, and should be removed from the input data?
    - The EIN and NAME columns will not increase the accuracy of the model and can be removed to improve code efficiency.

### Compiling, Training, and Evaluating the Model

  4. How many neurons, layers, and activation functions did you select for your neural network model, and why?
    - In the optimized model, layer 1 started with 120 neurons with a relu activation. For layer 2, it dropped to 80 neurons and continued with the relu activation. From there, the sigmoid activation seemed to be the better fit for layers 3 (40 neurons) and layer 4 (20 neurons).
 
 ![image1](https://github.com/nayanbarhate/Neural_Network_Charity_Analysis/blob/main/Images/Image1.png)    
  
  5. Where you able to achieve the target model performance?
    - The target for the model was 75%, but the best the model could produce was 72.7%.
  
  6. What steps did you take to try and increase model performance?
   - Columns were reviewed and the STATUS and SPECIAL_CONSIDERATIONS columns were dropped as well as increasing the number of neurons and layers. Other activations were tried such as tanh, but the range that model produced went from 40% to 68% accuracy. The linear activation produced the worst accuracy, around 28%. The relu activation at the early layers and sigmoid activation at the latter layers gave the best results.
     
 ![image2](https://github.com/nayanbarhate/Neural_Network_Charity_Analysis/blob/main/Images/Image2.png)
 
## Summary

 The relu and sigmoid activations yielded a 72.7% accuracy, which is the best the model could produce using various number of neurons and layers. The next step should be to try the random forest classifier as it is less influenced by outliers. Since our accuracy score was not particularly up to the standard with neural networks, we could have used the Random Forest classifiers. This is because random forest is a robust and accurate model due to their sufficient number of estimators and tree depth. Also the random forest models have a faster performance than neural networks and could have avoided the data from being overfitted. 
     
