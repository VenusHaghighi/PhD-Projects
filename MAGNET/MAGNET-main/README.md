# MAGNET
This is the Pytorch implementation of A Multiple Adaptive Channels Aggregation Graph Neural Network for Camouflaged Fraudsters Detection, named MAGNET for short. The overall view of MAGNET framework comes in the below:

![MA2C-GNN Framework](https://github.com/FraudDetectionModel/MA2C-GNN/assets/136766753/d13de2d9-117a-428d-82a1-983ed51e476b)

# Dependencies
- Python >= 3.10.9

- Pytorch >= 2.0.1

- DGL >= 1.1.1

# Usage
- src: Includes all code scripts, i.e., data preparation, model, and training.
- original-data: Includes original datasets, i.e., YelpChi and Amazon.
     - YelpChi.zip: Contains hotel and restaurant reviews filtered (spam) and recommended (legitimate) by Yelp.
     - Amazon.zip: Contains product reviews under the Musical Instruments category.
- dgl-data: Include dgl graph of original datasets, i.e., YelpChi and Amazon. 
     - yelp.dgl: The dgl graph for original data of YelpChi.
     - amazon.dgl: The dgl graph for original data of Amazon.


# Model Training
To generate domination signals of each neighborhood, we need to run the prepare_data.py and train the graph-agnostic edge labeling module by calling the edgeLabelling_train function from train.py. We have added domination signals corresponding to each neighborhood to dgl graphs in dgl-data directory. Therefore, you need to Unzip the yelp.dgl.zip and amazon.dgl.zip from dgl-data directory and move them to the src directory for the training phase.

- We take YelpChi as an example to illustrate the usage of repository:
    - Run train.py
    - Load the yelp.dgl 
    - Start training and testing phases 
