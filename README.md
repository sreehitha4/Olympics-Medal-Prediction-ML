\# Olympics Medal Prediction using Machine Learning



A Machine Learning mini-project that predicts Olympic medal-winning status and medal counts using historical Olympic performance and socioeconomic indicators.



\## 1. Project Overview



Predicting Olympic medal performance is a challenging machine learning problem because medal outcomes depend on both previous sporting performance and broader country-level characteristics.



This project implements a two-stage machine learning system:



1\. \*\*Classification:\*\* Predict whether a country will win at least one medal.

2\. \*\*Regression:\*\* Predict the number of medals for the country.

3\. \*\*Two-stage integration:\*\* The regression prediction is used only when the classification model predicts that the country will win a medal.



The system is evaluated using the \*\*2016 Summer Olympics as the held-out test edition\*\*.



\---



\## 2. Problem Statement



Develop a machine learning system that uses historical Olympic performance and socioeconomic indicators to predict:



\- Whether a country will win at least one medal.

\- The expected number of medals for a country.



The project uses a chronological train-validation-test setup to avoid randomly mixing Olympic editions across the splits.



\---



\## 3. Dataset



\### Olympic Dataset



The project uses the \*\*120 Years of Olympic History — Athletes and Results\*\* dataset containing historical Olympic athlete and event information.



Main files:



```text

data/raw/

├── athlete\_events.csv

└── noc\_regions.csv

