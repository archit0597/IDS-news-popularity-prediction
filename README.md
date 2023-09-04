# Online News Popularity

## Dependencies required for Running locally.
* Python3
* numpy
* pandas
* sklearn
* flask
* tensorflow

## Running instructions
* It is recommended to run the project in a tensorflow environment.
* conda is most easy and preferred tool for tensorflow.
* conda activate <your-tensor-flow-env-name>
* python3 UI.py => To run the end code with UI.
* python3 onp_train.py => To run the data science process without UI.

## File structure

* +ve.txt, -ve.txt => Movie reviews dataset for Sentiment Analysis.
* OnlineNewsPopularity.csv => UCI dataset for training and testing.
* onp_train.py => 
* onp_test.py => 
* eda.py => Exploratory Data Analytics of the data.
* Driver.py, Filters.py => Adapter layer with NLP module to translate the user input perceivable by the models.
* UI.py => UI entry point.
* pickled_algos => Serialized model states post training.

