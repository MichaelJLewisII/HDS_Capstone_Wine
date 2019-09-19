# Wine Type Classification
HarvardX  |  PH125.9x  |  Capstone pt. 2 - Choose your own

This analysis constitutes the second project of the Capstone course for Harvard's Data Science Professional Certificate program.  I selected a wine dataset from the University of California at Irvine's Machine Learning Repository.  Wines from three Italian cultivators ("types") were processed for different physical and chemical properties.  

The aim of this project was to develop a machine learning model capable of differentiating these wine types based on thirteen properties associated with each observation (n = 178). The scope of this analysis is constrained, at least by region, but offers insights into what variables are important to classify wine from different cultivators.  Possible future implications of such work include improved quality assurance protocols as well as the detection of counterfeit or diluted wine (projected to be up to 5% of commercial wine sold - Wine Spectator 2007).  In such situations, specificity and sensitivity are vital characteristics to monitor.  However, in the simpler case - classifying wine types - accuracy serves as a reasonably useful measure for model evaluation.

K-Nearest Neighbor, Classification Decision Trees, and Random Forest were employed with a high degree of accuracy on training and test sets (>90%).  These results, informed by robust exploratory data analysis, suggest that a combination of the variables evaluated are effective at differentiating types of wine.  To further demonstrate this point, Principal Component Analysis was performed and reduced the number of variables being modeled by 50%.  Ultimately, to reduce overfitting risk and assist with intrepreability, a tuned-KNN model was chosen predict wine types in a holdout sample.  The results show 94.1% accurate classification and indicate promise for future, larger-scale applications.

NOTE:  This work was performed in 2019 using R 3.5.3.
