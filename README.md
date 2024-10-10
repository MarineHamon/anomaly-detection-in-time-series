Actuellement, deux approches prédominantes sont en usage dans le cadre de la détection d'anomalies dans les séries temporelles (TSAD). La première approche implique l'application d'une fenêtre glissante sur l'ensemble de la série temporelle, suivie de la concaténation, en lignes, des sous-séquences obtenues sous une forme tabulaire, avant d'appliquer des méthodes spécialisées de détection d'anomalies. La seconde approche, quant à elle, maintient l'analyse dans le domaine temporel en développant des modèles conçus spécifiquement pour traiter ce type de données.

Notre méthode s'apparente à la première approche, mais elle se distingue par le fait que, au lieu d'utiliser directement les sous-séquences, nous procédons au calcul de caractéristiques pour chacune d'elles, lesquelles sont ensuite employées comme variables dans l'analyse.


**Versions des bibliothèques python employées :** \
numpy : 1.23.5 \
pandas : 2.0.3 \
tsfresh : 0.20.1 \
scikit-learn : 1.4.2 \
periodicity_detection : 0.1.2 \
pyod : 1.1.3 \
aeon : 0.11.0 

Nous utilisons également la version 3.11.9 de python.
