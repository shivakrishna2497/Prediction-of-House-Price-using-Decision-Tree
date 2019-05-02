# For importing and assessing data
import pandas as pd
# Defining and Fitting Decsion Tree Model(Output is Continuous Value)
from sklearn.tree import DecisionTreeRegressor
# For Model Validation by finding Mean Absolute Error
from sklearn.metrics import mean_absolute_error 
# To split our dataset into training set and validation set 
from sklearn.model_selection import train_test_split 



tree_sizes = [5, 25, 50, 100, 250, 500]
# Path of the file to read
iowa_file_path = '/home/shiva/Projects/Housepricing/train.csv'

# Selecting data for Modeling
home_data = pd.read_csv(iowa_file_path)
print(home_data.columns)

# Selecting the Prediction Target
y = home_data.SalePrice

# Feature Selection (Choosing those features which are relevent in predicting price)
feature_names = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X = home_data[feature_names]

# Reviewing Data
print(X.describe())
print(X.head())

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
iowa_model = DecisionTreeRegressor(random_state =1)
iowa_model.fit(train_X,train_y)

# Making Predictions with Validation Data
val_predictions = iowa_model.predict(val_X)
print(val_predictions)
print(iowa_model)

# Calculating Mean Absolute Error in Validation Data
val_mae = mean_absolute_error(val_predictions, val_y)
print(val_mae)

# We can use a utility function to help compare MAE scores from different values for max_leaf_nodes
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# Comparing the accuracy of models built with different values for max_leaf_nodes.
for max_leaf_nodes in tree_sizes:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

# Comparing Different Tree Sizes and Fitting Model Using All Data
candidate_max_leaf_nodes = tree_sizes
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)

final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)
final_model.fit(X, y)

