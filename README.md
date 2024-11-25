Currently, there are two predominant approaches to time series anomaly detection (TSAD). The first approach involves the application of a sliding window over the entire time series, followed by the concatenation, in rows, of the subsequences obtained in tabular form, before applying specialised anomaly detection methods. The second approach maintains the analysis in the time domain by developing models designed specifically to process this type of data.

Our method is similar to the first approach, but differs in that instead of using the subsequences directly, we calculate features for each of them, which are then used as variables in the analysis.


**Versions of the python libraries used:** \
numpy : 1.23.5 \
pandas : 2.0.3 \
tsfresh : 0.20.1 \
scikit-learn : 1.4.2 \
periodicity_detection : 0.1.2 \
pyod : 1.1.3 \
aeon : 0.11.0 

We also use version 3.11.9 of python.
