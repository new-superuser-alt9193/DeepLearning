import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from joblib import load
load('random_forest_churn.joblib')

#function to get all clients churn probability: 0 means no churn, 1 means churn
def getChurnProbabilities(random_forest, x):
    return random_forest.predict_proba(x)

#function to set each client according to low, mid or high churn chance (given low and high percentages by the user)
def showProbabilities(low,mid,high):
    clients_permanent = []
    clients_low = []
    clients_mid = []
    clients_high = []
    i = 0
    #for each client in the data set
    for client in proba_matrix:
        #get all their data and their churn chance into one list
        client_index = x_test.index[i]
        client_info = x_test.loc[client_index].values
        client_info = np.append(client_info,client[1])
        #store client data into profiles(permanent, low, mid, high) list
        if client[1] < low:
            clients_permanent.append(client_info)
        elif client[1] < mid:
            clients_low.append(client_info)
        elif client[1] < high:
            clients_mid.append(client_info)
        else:
            clients_high.append(client_info)
        i += 1
    return clients_permanent, clients_low, clients_mid, clients_high

#function to get each groups relevance in terms of bill amount by getting the sum of a determined group
def getClassificationSum(churn_group):
    names = x_test.columns.to_list()
    names[0] = "CUSTOMER_ID"
    names.append('CHURN_PERCENTAGE')
    clients_high_pd = pd.DataFrame(churn_group, columns = names)
    churn_bill_value = clients_high_pd['BILL_AMOUNT'].sum()
    return churn_bill_value

def getConfussionMatrix(y_test, y_pred):
    return confusion_matrix(y_test, y_pred)


df = pd.read_csv('../../data/telecom_pca.csv')
x = df.drop(columns=['TARGET'])
y = df['TARGET']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25, random_state= 1)

random_forest = load('random_forest_churn.joblib')
proba_matrix = getChurnProbabilities(random_forest, x_test)

clients_permanent, clients_low, clients_mid, clients_high = showProbabilities(.30,.60,.80)

print(len(clients_permanent), " clients have no churn chances")
print(len(clients_low), " clients have low churn chances")
print(len(clients_mid), " clients have mid churn chances")
print(len(clients_high), " clients have high churn chances")

permanent_value = getClassificationSum(clients_permanent)
low_value = getClassificationSum(clients_low)
mid_value = getClassificationSum(clients_mid)
high_value = getClassificationSum(clients_high)

print("Bill amount for clients with no churn chances: ", permanent_value)
print("Bill amount for clients with low churn chances: ", low_value)
print("Bill amount for clients with mid churn chances: ", mid_value)
print("Bill amount for clients with high churn chances: ", high_value)