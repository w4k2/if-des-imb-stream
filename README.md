Rozwiniecie pomyslu z ICDM, Elsevier Knowledge-Based Systems Journal

## Zalozenia

3 klasyfikatory proste:
* Gaussian Naive Bayes
* MLP (?)
* Hoeffding Tree

7 metod kombinacji (poziom meta):
* SEA
* KNORA-U1 (bagging)
* KNORA-E1 (bagging)
* KNORA-U2 (bazowe)
* KNORA-E2 (bazowe)
* OOB
* UOB
* OB (jest ale tak serio to nie ma)

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
  * szum: 1%, 3%, 5%
* rzeczywiste

Metryki Test-Then-Train (chunk 250 instancji):
* BAC
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

### Eskperymenty

liczba metod:
* Kombinacja - 7 (w tym 2 bez preproc), preprocessing - 5, razem - 27

Eksperyment 1 - Poziom selekcji:
* Skrócony eksperyment
* Chcemy sprawdzić, czy lepsza jest selekcja na poziomie baggingu czy na poziomie klasyfikatorów bazowych, Czyste SEA dla porównania
* kombinacja -  SEA, KNORA-U1, KNORA-U2, KNORA-E1, KNORA-E2
* preprocessing - brak
* Porównać metody na dwóch poziomach i odrzucić z dalszych eksperymentów słabsze (prawdopodobnie na podstawie wcześniejszych eksperymentów będą to KNORA-U1 i KNORA-E1, ma to sens bo na poziomie bazowych pula jest większa i bardziej róznorodna). Może pozbyć się też czystego SEA.

liczba metod:
* Kombinacja - 4 (w tym 2 bez preproc), preprocessing - 5, razem - 12

Eksperyment 2 - Preprocessing:
* Skrócony eksperyment (tylko IR)
* Chcemy sprawdzić, które metody preprocessingu osiagąją najlepsze wyniki w połączeniu z metodami kombinacji
* kombinacja - KNORA-U?, KNORA-E?
* preprocessing - Brak, Random Oversampling, BL2-SMOTE, Random Undersampling, CNN
* Podział prezentacji na oversampling i undersampling, zostawiamy po jednej metodzie z obu kategorii. Może pozbywamy się też opcji braku oversamplingu.

liczba metod:
* Kombinacja - 4 (w tym 2 bez preproc), preprocessing - 2, razem - 6

Eksperyment 3 - Porównanie najlepszych ze state of art:
* Pełny eksperyment (IR i typy dryftu)
* Sprawdzamy, czy połączenie najlepszych metod kombinacji oraz preprocessingu sprawuje się lepiej niz OOB i UOB.

Eskperyment 4 - Realne strumienie:
* Ten sam zetaw metod co w eksperymencie 3, ale tym razem na realnych strumieniach od Bartka.
