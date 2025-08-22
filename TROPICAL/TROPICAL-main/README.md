# TROPICAL
This is the Pytorch implementation of "TROPICAL: A transformer-based hypergraph learning framework for detecting camouflaged malicious actors" in fraud graphs.

![TROPICAL-FRAMEWORK](https://github.com/VenusHaghighi/TROPICAL/blob/main/TROPICAL.JPG).

# Dependencies
- Python >= 3.10.9
- Pytorch >= 2.0.1
- DGL >= 1.1.1

# Datasets
-YelpChi: Contains hotel and restaurant reviews filtered (spam) and recommended (legitimate) by Yelp.

-Amazon: Contains product reviews under the Musical Instruments category.

# Usage
- hypergraph_generation includes codes for generating different groups of hyperedges.
- sequential_data includes generated input fesature sequences corresponding to the Yelpchi and Amazon datasets.
- run data_preparation.py to generate different groups of hyperedges for YelpChi and Amazon datasets.
- run hypergraph_to_seq.py to generate sequential input data for YelpChi and Amazon datasets. We saved the generated sequential input data for both Yerlpchi and Amazon in sequential_data folder.

# Training
- for traing the model: run main_train.py
- for setting the hyperparameters please follow the help in parser.add_argument of code. 
 
