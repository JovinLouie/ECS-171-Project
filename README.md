# ECS 171 Project: Student Dropout Prevention

By Team 1: Jovin Louie, Cesar Aguirre, Joey Suen, Jessica Zhao, Eli Seligman \
Professor Setareh Rafatirad \
ECS 171: Machine Learning \
Winter Quarter 2026 - UC Davis

---

## Abstract

Student dropout in higher education remains a significant challenge for both institutions and students. This project investigates whether machine learning models can effectively identify students at risk of dropping out early enough to allow meaningful intervention or allow for better usage of institutional resources. Using the Predict Students’ Dropout and Academic Success dataset from the Polytechnic Institute of Portalegre, we trained and evaluated three classification models: Logistic Regression, Random Forest, and XGBoost. The models were built using demographic, financial, and early academic performance features, with recall emphasized as a primary evaluation metric to prioritize identifying at-risk students. Results show that each model exhibits different strengths depending on institutional priorities, such as maximizing recall or reducing false positives. Feature analysis also highlights the importance of both academic progress and financial stability in predicting student outcomes. While the models demonstrate promising predictive performance, the dataset’s origin might limit the ability to generalize for other universities around the globe. Future work should focus on validating similar models across different universities and incorporating additional behavioral or engagement-based features to improve robustness.

---

## References

- Dataset Source:
  https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
