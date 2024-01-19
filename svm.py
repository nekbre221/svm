""" eng:suport vecteur machine / fr: separateur a vaste marge
modele lineaire qui marche bien avec des observations lineairement separable
il repose sur la marge maximale qui separe les points d'observtions de classse
differante les plus proche"""

#SVM
#data_preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importer le dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

#selection des variables par comptage de modalité
counts = dataset.groupby("Gender")["Purchased"].value_counts()
"""nous remarquons que la var Gender influence très faiblement la variable Purchased!
il peut donc être legitimement éliminé. Le t.test aurrai aussi put nous permettre
d'avoir le même resultat sur la significativité de la variable"""  

X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, -1].values

#gestion des valeurs manquantes
missing_values = dataset.isna().sum()

# Diviser le dataset entre le Training set et le Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
# cette operation ne se deroule pas en regression lin sim,poly et mult
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Construction du modèle
from sklearn.svm import SVC
model_svm = SVC(random_state = 0, kernel="linear")
model_svm.fit(X_train, y_train)

# Faire de nouvelles prédictions
y_pred = model_svm.predict(X_test)

# Matrice de confusion
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)


#impression du rapport
rapport= classification_report(y_test,y_pred,zero_division=True)

#accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy

prediction_df=pd.DataFrame({"y_test":y_test,"y_pred":y_pred})


# Visualiser les résultats
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model_svm.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.4, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Résultats du Training set')
plt.xlabel('Age')
plt.ylabel('Salaire Estimé')
plt.legend()
plt.show()






























