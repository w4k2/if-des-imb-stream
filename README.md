Rozwiniecie pomyslu z ICDM, Springer Machine Learning Journal special issue

3 klasyfikatory proste:
* Gaussian Naive Bayes
* MLP
* Hoeffding Tree

7 metod kombinacji (poziom meta):
* SEA
* KNORA-U (z i bez baggingu)
* KNORA-E (z i bez baggingu)
* OOB
* UOB

Preprocessing (nie dotyczy OOB i UOB):
* Brak
* Random Oversampling
* BL2-SMOTE
* Random Undersampling
* CNN

Dane:
* stream-learn
  * pojedynczy dryft
  * typy dryftu: nagly, inkrementalny, gradualny
  * IR: 5%, 10%, 15%, 20%
  * szum: 1%, 5%, 10%
* rzeczywiste

Metryki Test-Then-Train (chunk 250 instancji):
* BAC
* AUC
* g-mean
* F1-measure
* precision
* recall
* specificity

Prezentancja:
* poziom niezbalansowania
* typ dryftu
* przykladowe srednie przebiegi (z zaznaczeniem momentu dryftu)
* radary
* analiza statystyczna rankingów
* realne osobno
