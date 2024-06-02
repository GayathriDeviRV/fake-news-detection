# fake-news-detection
The project utilizes machine learning and deep learning techniques to detect fake news in low resource languages such as Tamil and Malayalam languages, employing methods like Bag of Words, TF-IDF, logistic regression, random forest, and a multi-layer perceptron model with dropout regularization.

## Methodologies
### Data Loading and Preprocessing

- Load Data: The dataset is loaded from a CSV file.
- Inspect Columns: Data is cleaned by converting it to lowercase, removing square brackets content, URLs, HTML tags, punctuation, and newline characters.

## Feature Extraction

### Bag of Words (BoW):
- BoW is a simple text representation technique where a text is represented as a collection (or "bag") of its words, disregarding grammar and word order but keeping multiplicity.
- Each unique word in the corpus (the entire set of texts) is treated as a feature. Texts are then converted into fixed-length vectors based on the count (or frequency) of each word.

### Term Frequency-Inverse Document Frequency (TF-IDF):
- TF-IDF is a more advanced text representation technique that aims to reflect the importance of a word in a document relative to the entire corpus. It combines two metrics: Term Frequency (TF) and Inverse Document Frequency (IDF).
- Term Frequency (TF): Measures how frequently a word appears in a document.
- Inverse Document Frequency (IDF): Measures how important a word is. It decreases the weight of words that appear frequently in many documents (common words) and increases the weight of words that appear rarely.
- TF-IDF score is calculated as TF-IDF = TF × IDF

## Machine Learning Models

### Logistic Regression (LR):
A linear model for binary classification.
### Random Forest (RF):
An ensemble learning method that uses multiple decision trees to improve classification performance.

## Deep Learning Model

### Multi-Layer Perceptron (MLP):
- An artificial neural network model defined with an input layer, hidden layer, and output layer.
- Increased Dropout Rate: Dropout regularization is applied with a dropout rate of 0.6 to prevent overfitting.
- Activation Function: ReLU (Rectified Linear Unit) activation function is used.

## Model Training

Training Process for MLP:
- Loss Function: Cross-Entropy Loss is used as the loss function.
- Optimizer: Adam optimizer is used for updating model weights.
- Learning Rate Scheduler: A learning rate scheduler is used to adjust the learning rate during training.
- Early Stopping: Implemented to stop training when the model performance on the validation set stops improving.
- Validation Set: A subset of the training data is used for validation during training.
- Evaluation includes metrics like accuracy, F1 score, precision, recall, and confusion matrix analysis.

Dataset is taken from [DFND : DRAVIDIAN_FAKE NEWS DATA](https://ieee-dataport.org/documents/dfnd-dravidianfake-news-data)

Eduri Raja, Badal Soni, Samir Kumar Borgohain. (2023). "DFND : Dravidian_Fake News Data." Web.
