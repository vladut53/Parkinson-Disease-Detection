import numpy as np
import pandas as pd
from sklearn import neural_network

# Citire date din dataset
file_path = "parkinsons.csv"
date_parkinson = pd.read_csv(file_path)
date_parkinson = date_parkinson.to_numpy()
etichete = date_parkinson[:, 16]
etichte = etichete.astype(int)
date = np.delete(date_parkinson, 16, 1)
date = date.astype(float)

# Definirea functiei care calc. acuratetea
def accuracy_calculator (predictii, etichete_test):
    counter = 0
    for i in range(len(etichete_test)):
        if (predictii[i] == etichete_test[i]):
            counter+=1            
    return round(counter/len(etichete_test),3)

# etichete si date split : 75% train 25% test
size = int(0.75*len(etichete))
date_train = date[:size, :]
date_test = date[size:, :]
etichete_train = etichete[:size]
etichete_test = etichete[size:]

# Learning-rates si nr de neuroni de pe stratul ascuns folosite
learning_rates = np.array([0.1, 0.01])
hidden_layers = [22, 11]

# Learning-rates si nr de neuroni de pe stratul ascuns folosite
learning_rates = np.array([0.1, 0.01])
hidden_layers = [22, 11]

for j in range(len(hidden_layers)):
    print ("Pentru un strat ascuns, cu " + str(hidden_layers[j])+ " de neuroni: ")
    sum = 0
    for i in range(len(learning_rates)):
        clf = neural_network.MLPClassifier(solver="adam", max_iter = 1200, hidden_layer_sizes=hidden_layers[j], learning_rate_init=learning_rates[i])
        clf.fit(date_train, etichete_train)
        predictii = clf.predict(date_test)
        print("Predictia pentru learning rate-ul de " + str(learning_rates[i]) + " este", accuracy_calculator(predictii, etichete_test))
        sum += accuracy_calculator(predictii, etichete_test);
    print("Predictia medie este:", round(sum/2 ,3))
print("\n")


for j in range(len(hidden_layers)):
    print ("Pentru 2 straturi de neuroni, egal cu stratul anterior, adica " + str(hidden_layers[j]) + ":")
    sum = 0
    for i in range(len(learning_rates)):
        clf = neural_network.MLPClassifier(solver="adam", max_iter = 1200, hidden_layer_sizes=(hidden_layers[j], hidden_layers[j]), learning_rate_init=learning_rates[i])
        clf.fit(date_train, etichete_train)
        predictii = clf.predict(date_test)
        print("Predictia pentru learning rate-ul de " + str(learning_rates[i]) + " este", accuracy_calculator(predictii, etichete_test))
        sum += accuracy_calculator(predictii, etichete_test);
    print("Predictia medie este:", round(sum/2 ,3))
print("\n")

for j in range(len(hidden_layers)):
     print ("Pentru 2 straturi de neuroni, jumatate fata de stratul anterior, adica " + str(hidden_layers[j]) + " si " + str(int(0.5*hidden_layers[j])) + ":")
     sum = 0
     for i in range(len(learning_rates)):
         clf = neural_network.MLPClassifier(solver="adam", max_iter = 1200, hidden_layer_sizes=(hidden_layers[j],int(0.5 * hidden_layers[j])), learning_rate_init=learning_rates[i])
         clf.fit(date_train, etichete_train)
         predictii = clf.predict(date_test)
         print("Predictia pentru learning rate-ul de " + str(learning_rates[i]) + " este", accuracy_calculator(predictii, etichete_test))
         sum += accuracy_calculator(predictii, etichete_test);
     print("Predictia medie este:", round(sum/2 ,3))
print("\n")

for j in range(len(hidden_layers)):
     print ("Pentru 2 straturi de neuroni, dublu fata de stratul anterior, adica " + str(hidden_layers[j]) + " si " + str(int(2*hidden_layers[j])) + ":")
     sum = 0
     for i in range(len(learning_rates)):
         clf = neural_network.MLPClassifier(solver="adam", max_iter = 1200, hidden_layer_sizes=(hidden_layers[j],int(2* hidden_layers[j])), learning_rate_init=learning_rates[i])
         clf.fit(date_train, etichete_train)
         predictii = clf.predict(date_test)
         print("Predictia pentru learning rate-ul de " + str(learning_rates[i]) + " este", accuracy_calculator(predictii, etichete_test))
         sum += accuracy_calculator(predictii, etichete_test);
     print("Predictia medie este:", round(sum/2 ,3))


