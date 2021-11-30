#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4200- Assignment #5
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

#Use the command: "pip install mlxtend" on your terminal to install the mlxtend library

#read the dataset using pandas
df = pd.read_csv('retail_dataset.csv', sep=',')

#find the unique items all over the data an store them in the set below
itemset = set()
for i in range(0, len(df.columns)):
    items = (df[str(i)].unique())
    itemset = itemset.union(set(items))

#remove nan (empty) values by using:
itemset.remove(np.nan)

#To make use of the apriori module given by mlxtend library, we need to convert the dataset accordingly. Apriori module requires a
# dataframe that has either 0 and 1 or True and False as data.
#Example:

#Bread Wine Eggs
#1     0    1
#0     1    1
#1     1    1

#To do that, create a dictionary (labels) for each transaction, store the corresponding values for each item (e.g., {'Bread': 0, 'Milk': 1}) in that transaction,
#and when is completed, append the dictionary to the list encoded_vals below (this is done for each transaction)
#-->add your python code below
encoded_vals = []
valCount = [0,0,0,0,0,0,0,0,0]
for index, row in df.iterrows():
    #key vals[0] = bread; 1 = wine, 2 = eggs, 3 = meat, 4 = cheese
    # 5 = pencil, 6 = diaper, 7 = bagel 8 = milk
    vals = [0,0,0,0,0,0,0,0,0]
    for i in range(0,7):
        if(row[i] == 'Bread'):
            vals[0] = 1
            valCount[0]+=1
        if(row[i] == 'Wine'):
            vals[1] = 1
            valCount[1]+=1
        if(row[i] == 'Eggs'):
            vals[2] = 1
            valCount[2]+=1
        if(row[i] == 'Meat'):
            vals[3] = 1
            valCount[3]+=1
        if(row[i] == 'Cheese'):
            vals[4] = 1
            valCount[4]+=1
        if(row[i] == 'Pencil'):
            vals[5] = 1
            valCount[5]+=1
        if(row[i] == 'Diaper'):
            vals[6] = 1
            valCount[6]+=1
        if(row[i] == 'Bagel'):
            vals[7] = 1
            valCount[7]+=1
        if(row[i] == 'Milk'):
            vals[8] = 1
            valCount[8]+=1

    labels = {'Bread' : vals[0], 'Wine' : vals[1], 'Eggs': vals[2], 
    'Meat' : vals[3], 'Cheese': vals[4],'Pencil': vals[5], 
    'Diaper': vals[6], 'Bagel': vals[7], 'Milk' : vals[8]}

    encoded_vals.append(labels)

#adding the populated list with multiple dictionaries to a data frame
ohe_df = pd.DataFrame(encoded_vals)

#calling the apriori algorithm informing some parameters

freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)

#iterate the rules data frame and print the apriori algorithm results by using the following format:
#Meat, Cheese -> Eggs
#Support: 0.21587301587301588
#Confidence: 0.6666666666666666
#Prior: 0.4380952380952381
#Gain in Confidence: 52.17391304347825
#-->add your python code below
for index,row in rules.iterrows():
    #print(str(row['antecedents']) + " -> " + str(row['consequents']))
    suportCount = 0
    for x in row['antecedents']:
        print(str(x) + " ", end="")
    print("-> ", end= "")
    for x in row['consequents']:
        print(str(x) + " ", end = "")
        if (x == 'Bread'):
            suportCount += valCount[0]
        if (x == 'Wine'):
            suportCount += valCount[1]
        if (x == 'Eggs'):
            suportCount += valCount[2]
        if (x == 'Meat'):
            suportCount += valCount[3]
        if (x == 'Cheese'):
            suportCount += valCount[4]
        if (x == 'Pencil'):
            suportCount += valCount[5]
        if (x == 'Diaper'):
            suportCount += valCount[6]
        if (x == 'Bagel'):
            suportCount += valCount[7]
        if (x == 'Milk'):
            suportCount += valCount[8]
    print()
    print("Support: " + str(row['support'])) # support
    print("Confidence: " + str(row['confidence'])) #confidence
#To calculate the prior and gain in confidence, find in how many transactions the consequent of the rule appears (the supporCount below). Then,
#use the gain formula provided right after.
#-->add your python code below
    prior = suportCount/len(encoded_vals) #-> encoded_vals is the number of transactions
    print("Prior: " + str(prior))
    print("Gain in Confidence: " + str(100*(row['confidence']-prior)/prior))
    print()

#Finally, plot support x confidence

plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()