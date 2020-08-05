IoT article extension for Elsevier Information Fusion Journal submission

## Set-up

4 base classifiers:
* Gaussian Naive Bayes
* Hoeffding Tree
* k-Nearest Neighbors
* Support Vector Machine

5 combination methods:
* SEA
* KNORA-U1 (bagging)
* KNORA-E1 (bagging)
* KNORA-U2 (base)
* KNORA-E2 (base)

Preprocessing:
* None
* Random Oversampling
* BL2-SMOTE
* Random Undersampling
* CNN

Data:
* stream-learn
  * single concept drift
  * drift types: sudden, incremental, gradual
  * IR: 5%, 10%, 15%, 20%
  * label noise: 1%, 3%, 5%
* MOA
  * same configurations as above
  * Agrawal (sudden and gradual)
  * Hyperplane (incremental)

Test-Then-Train evaluation procedure (200 data chunks, 250 samples each):
* BAC
* G-mean
* F1-measure
* precision
* recall
* specificity

### Goals of the experiments

#### Experiment 1 -- Dynamic selection level

The main purpose of the first experiment is, due to a large number of methods, the pre-selection of further used dynamic ensemble selection approaches. Dynamic selection without the use of preprocessing techniques is evaluated for the potential to classify highly imbalanced data. Based on the results obtained from this shortened experiment in which the results are presented only for the highest tested imbalance ratio and stream-learn generated data streams, a pool of classifiers will be selected, on which DES methods will be used later for a given type of base classifier (i.e., bagging level or the level of all base classifiers present in the pool).

In addition, the behavior of each Dynamic Ensemble Selection method during the gradual and sudden drift occurrence was analyzed. For this purpose, the measures defined by Shaker and Hüllermeier were used, namely the restoration time informing about the speed of model recovery after drift and maximum performance loss associated with the decrease in the method performance in the event of a concept drift.

#### Experiment 2 -- DES with preprocessing techniques (repository -- experiment 2.1 (oversampling) and 2.2 (undersampling))

The second experiment aims to examine how two previously chosen DES methods perform based on the preprocessing technique with which they were paired compared to using solely dynamic selection. We divided the experiment into two parts, i.e., oversampling (2.1) and undersampling (2.2). After analyzing the results obtained, one preprocessing method will be selected from both groups, which then will be used in subsequent experiments. Again, this is a shortened experiment in which we present results only for a 5% of the minority class and stream generated using stream-learn package.

#### Experiment 3 -- Comparison with state-of-art (repository -- experiment 3 (online) and 5 (chunk-based))

In the third experiment, two previously selected dynamic selection methods and two preprocessing techniques are compared with state-of-art online data stream classification approaches based on the notion of offline Bagging (3), as well as with the chunk-based stream classification methods (5). Because online methods require a base classifier capable of incremental learning, a comparison was possible only for Gaussian Naive Bayes and Hoeffding Tree classifiers. Batch methods use 5 bagging classifiers, each of which consists of 10 base models, while online methods maintain ensembles consisting of 20 base classifiers.

The comparative methods were:
* Online Bagging (OB)}, which updates each base classifer in the pool with the appearance of a new instance using the Poisson(lambda= 1) distribution.
* Oversampling-Based Online Bagging (OOB) and Undersampling-Based Online Bagging (UOB), which integrate resampling into Online Bagging algorithm. This was achieved by making the lambda value dependent on the proportion between classes.
* Learn++.NIE (Nonstationary and Imbalanced Environments) and Learn ++.CDS (Concept Drift with SMOTE), which extend the Learn++.NSE (Non Stationary Environments) algorithm.
* Recursive Ensemble Approach (REA), which incorporates part of previous minority class samples into the current data chunk and combines base models in a dynamically weighted manner.
* Over/UnderSampling Ensemble (OUSE), which uses minority class instances from all previously seen data chunks and a subset of majority class present in the most recent chunk to generate new ensemble.
* KMC, an ensemble-based approach, which performs, on each arriving data chunk, undersampling based on k-means clustering algorithm.
