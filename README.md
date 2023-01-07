# Spatio-Temporal Analysis in high resolution of Tweets

During and around the occurrence of natural hazards, tweets increase significantly.
Although Twitter is often used for real-time alerts, there is still a need to improve the process of extracting reported damage from tweets and accurately mapping their geographical location.
In this study we attempt to examine what was the spatio-temporal distribution of the tweets associated with natural hazard events.
The attached code excerpt includes a tool for classifying tweets as relevant or not relevant to the examined event.

## Running the Code
We now provide the running instructions for the provided code.

### Configuration
The hyperparameters for the code are defined in condig.py.
The class ConfigMain defines hyperparameters used for the metadata (file used for training, train-dev ration, model used, etc.).
Please notice that if toy already trained the model and you only want to test it please define the parameter PRIMARY_ALREADY_TRAINED as True.
To approach tweet relevance task please define the PROBLEM parameter as 'relevance'.
The parameter FEATURE_TEXT defines the name of attribute that contains the tweet contents.

The class CondigPrimary defines the hyperparameters of the RCNN model.


### Execution
To train and test the model, just run the file main.py as follows:
'''
python main.py
'''
