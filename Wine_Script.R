#########################################################

# Capstone Project II:  Classifying Italian Wine "Types"
# Course:  HarvardX - PH125.9x
# Student:  Mike Lewis
# Date:  July-August 2019

#########################################################



###################################################

# Begin Script

###################################################



# WINE CLASSIFICATION FROM UC IRVINE MACHINE LEARNING REPOSITORY

wineData <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"), header = FALSE, sep = ",", 
                      col.names =  c("Type", "Alcohol", "MalicAcid", 
                                     "Ash", "Alcalinity", "Magnesium",
                                     "Phenols", "Flavanoids", "Nonflavanoids",
                                     "Proanthocyanins", "ColorIntensity", "Hue", 
                                     "Dilution", "Proline"))   # Col names shortened from info found here https://archive.ics.uci.edu/ml/datasets/wine



# LIBRARIES

  # Install - this process might take up to 10 minutes
    if(!require(caret)) install.packages("caret")
    if(!require(factoextra)) install.packages("factoextra")
    if(!require(GGally)) install.packages("GGally") 
    if(!require(knitr)) install.packages("knitr")
    if(!require(randomForest)) install.packages("randomForest")
    if(!require(rattle)) install.packages("rattle")    
    if(!require(rpart.plot)) install.packages("rpart.plot")
    if(!require(tidyverse)) install.packages("tidyverse")
                                             
                                             
                                         
# DATASET SNAPSHOT
    
  head(wineData)  
  class(wineData)
  sapply(wineData, typeof)
  
  # Update 'Type' to factor
    wineData$Type <- as.factor(wineData$Type)
  
  # Checking summary statistics  
    summary(wineData)
    # The different scales of variable range will dictate variable standardization (after separating train and test sets) in order to not over-represent the importance of variables with larger ranges (e.g. Magnesium) in algos that rely on Euclidean distance measures (e.g. KNN as in KNN3)
    # In practice, when the training and validation sets are TRULY distinct, this general data evaulation would likely take place for each separately.

  # Check for missing values
    sapply(wineData, function(x) sum(is.na(x))) %>% 
      kable(col.names = c("Missing Values"))  # Bless those who provide clean datasets




# EXPLORATORY DATA ANALYSIS
    
  pairsPlot <- ggpairs(wineData, title = "Pairwise variable exploration", legend = 1,
                       aes(colour = Type), lower=list(combo=wrap("facethist", binwidth=0.8)))
  pairsPlot  
  
  # Various insights can be gleaned from this visual
  # First, some variables have non-normal distributions; however, due to differences across class in these parameters, tranformations (e.g. log or squaring) are likely unnecessary
  # Nevertheless for classification algorithms relying on distance measures, i.e. Euclidean, the varying scales of predictors will likely require standardization
  # There appears to be some correlation in the variables, across wine types, signaling that dimension reduction might prove fruitful, esp. given the ratio of observations to predictors (13.7:1) is not particularily robust
    # Examples include:  Flavanoids-Proanthocyanins (0.653), Flavanoids-Phenols (0.865), and Flavanoids-Nonflavanoids (-0.538)
  # Lastly, it is a reasonable expectation that wine some wine types will be classified with high accuracy given the unique range occupied across certain variables
    # Some of the most pronounced examples include:  (1) Proline for Type 1 and (2) Malic Acid and Color Intensity for Type 3
  
  
  
# DATA PARTIONING
  
  # Create Training set
    set.seed(9)
    train_ind <- createDataPartition(wineData$Type, times = 1, p = 0.6, list = FALSE)
    train_wineData <- wineData %>% slice(train_ind)  # n_train = 108
    temp_wineData <- wineData %>% slice(-train_ind)
    
  # Create Test and Validation sets
    set.seed(9)
    testValid_ind <- createDataPartition(temp_wineData$Type, times = 1, p = 0.5, list = FALSE)
    test_wineData <- temp_wineData %>% slice(testValid_ind)  # n_test = 36 
    validation_wineData <- temp_wineData %>% slice(-testValid_ind)  #n_valid = 34
  
  # Remove temporary objects
    rm(train_ind, testValid_ind, temp_wineData)
    
  
  
# STANDARDIZE NON-CATEGORICAL VARIABLES
# Standardization is a distinct step (at least for validation data) because the computation should not mix training and evaluation data.
    
  # Train dataset
    train_wineData$Alcohol <- scale(train_wineData$Alcohol, center = TRUE, scale = TRUE)
    train_wineData$MalicAcid <- scale(train_wineData$MalicAcid, center = TRUE, scale = TRUE)
    train_wineData$Ash <- scale(train_wineData$Ash, center = TRUE, scale = TRUE)
    train_wineData$Alcalinity<- scale(train_wineData$Alcalinity, center = TRUE, scale = TRUE)
    train_wineData$Magnesium<- scale(train_wineData$Magnesium, center = TRUE, scale = TRUE)
    train_wineData$Phenols <- scale(train_wineData$Phenols, center = TRUE, scale = TRUE)
    train_wineData$Flavanoids <- scale(train_wineData$Flavanoids, center = TRUE, scale = TRUE)
    train_wineData$Nonflavanoids <- scale(train_wineData$Nonflavanoids, center = TRUE, scale = TRUE)
    train_wineData$Proanthocyanins <- scale(train_wineData$Proanthocyanins, center = TRUE, scale = TRUE)
    train_wineData$ColorIntensity <- scale(train_wineData$ColorIntensity, center = TRUE, scale = TRUE)
    train_wineData$Hue <- scale(train_wineData$Hue, center = TRUE, scale = TRUE)
    train_wineData$Dilution <- scale(train_wineData$Dilution, center = TRUE, scale = TRUE)
    train_wineData$Proline<- scale(train_wineData$Proline, center = TRUE, scale = TRUE)
  
  # Test dataset
    test_wineData$Alcohol <- scale(test_wineData$Alcohol, center = TRUE, scale = TRUE)
    test_wineData$MalicAcid <- scale(test_wineData$MalicAcid, center = TRUE, scale = TRUE)
    test_wineData$Ash <- scale(test_wineData$Ash, center = TRUE, scale = TRUE)
    test_wineData$Alcalinity<- scale(test_wineData$Alcalinity, center = TRUE, scale = TRUE)
    test_wineData$Magnesium<- scale(test_wineData$Magnesium, center = TRUE, scale = TRUE)
    test_wineData$Phenols <- scale(test_wineData$Phenols, center = TRUE, scale = TRUE)
    test_wineData$Flavanoids <- scale(test_wineData$Flavanoids, center = TRUE, scale = TRUE)
    test_wineData$Nonflavanoids <- scale(test_wineData$Nonflavanoids, center = TRUE, scale = TRUE)
    test_wineData$Proanthocyanins <- scale(test_wineData$Proanthocyanins, center = TRUE, scale = TRUE)
    test_wineData$ColorIntensity <- scale(test_wineData$ColorIntensity, center = TRUE, scale = TRUE)
    test_wineData$Hue <- scale(test_wineData$Hue, center = TRUE, scale = TRUE)
    test_wineData$Dilution <- scale(test_wineData$Dilution, center = TRUE, scale = TRUE)
    test_wineData$Proline<- scale(test_wineData$Proline, center = TRUE, scale = TRUE)
    
  # Validation set
    validation_wineData$Alcohol <- scale(validation_wineData$Alcohol, center = TRUE, scale = TRUE)
    validation_wineData$MalicAcid <- scale(validation_wineData$MalicAcid, center = TRUE, scale = TRUE)
    validation_wineData$Ash <- scale(validation_wineData$Ash, center = TRUE, scale = TRUE)
    validation_wineData$Alcalinity<- scale(validation_wineData$Alcalinity, center = TRUE, scale = TRUE)
    validation_wineData$Magnesium<- scale(validation_wineData$Magnesium, center = TRUE, scale = TRUE)
    validation_wineData$Phenols <- scale(validation_wineData$Phenols, center = TRUE, scale = TRUE)
    validation_wineData$Flavanoids <- scale(validation_wineData$Flavanoids, center = TRUE, scale = TRUE)
    validation_wineData$Nonflavanoids <- scale(validation_wineData$Nonflavanoids, center = TRUE, scale = TRUE)
    validation_wineData$Proanthocyanins <- scale(validation_wineData$Proanthocyanins, center = TRUE, scale = TRUE)
    validation_wineData$ColorIntensity <- scale(validation_wineData$ColorIntensity, center = TRUE, scale = TRUE)
    validation_wineData$Hue <- scale(validation_wineData$Hue, center = TRUE, scale = TRUE)
    validation_wineData$Dilution <- scale(validation_wineData$Dilution, center = TRUE, scale = TRUE)
    validation_wineData$Proline<- scale(validation_wineData$Proline, center = TRUE, scale = TRUE)
  
    
    
# MACHINE LEARNING
   
  # ALGORITHM 1 - Classification Decision Tree (CDT) w/ tuned 'complexity parameter' (cp)
    set.seed(9)
    cdt_fit <- train(Type ~ ., 
                     method = "rpart",
                     tuneGrid = data.frame (cp = seq(0.0, 0.1, len = 20)),
                     data = train_wineData)
  
    CP_plot <- ggplot(cdt_fit, highlight = TRUE) +
               ggtitle("Complexity Parameter Tuned for Accuracy") # Modest affect of complexity parameter on accuracy (~ 1.5 percentage points over range tested).
    CP_plot
    
    cdt_fit$bestTune  # 0.02631579
    
    # Fit the CDT model to test data
    final_cdt_fit <- train(Type ~ ., 
                           method = "rpart",
                           tuneGrid = data.frame (cp = cdt_fit$bestTune),
                           data = train_wineData)
    
    set.seed(9)
    cdt_predict <- predict(final_cdt_fit, test_wineData)
    cdt_ConfusionMatrix <- confusionMatrix(cdt_predict, reference = test_wineData$Type)
    # Produces accuracy of 94.4% - it seems the predictors provide sufficiently meaningful distinctions between wine 'types'.
    
    # Visualize the tree
    rpart.plot(final_cdt_fit$finalModel , type = 3, main = "Wine Type Classification Tree", legend.x = .85, legend.y = 1.05, extra = 2)
    
  
    
  # ALGORITHM 2 - Random Forest
    set.seed(9)
    rf_fit <- randomForest(Type ~ ., data = train_wineData)
    
    # Extracting variable importance
      varImport_df <- as.data.frame(rf_fit$importance, row.names = NULL , optional = FALSE,
                                    make.names = TRUE,
                                    importance = as.vector(importance(rf_fit)),
                                    stringsAsFactors = default.stringsAsFactors())
      varImport_df <- rownames_to_column(varImport_df, var = "Characteristic")
    
      # Variable importance plot
        plot_varImport <- ggplot(varImport_df, aes(x = reorder(Characteristic, -MeanDecreaseGini), y = MeanDecreaseGini)) + 
                          ggtitle("Variable Importance from Random Forests") +
                          geom_text(aes(label = round(MeanDecreaseGini, 1), vjust = -0.5)) +
                          xlab("Characteristic") +
                          geom_col(fill = "#06b5ed")
        
      # Shows the relatively large roles that color intensity, proline, flavanoids, dilution, and alcohol content played in decreasing node impurity
        plot_varImport
    
    
    # Fit the random forest model to test data
    set.seed(9)
    rf_predicts <- predict(rf_fit, test_wineData)
    rf_confusionMatrix <- confusionMatrix(rf_predicts, reference = test_wineData$Type)
    rf_confusionMatrix  # Shows 100% accuracy - a possible indication of overfitting and/or random forest's ability to delineate fairly robust categories - i.e. "types" of wine.
    
    
    
  # ALGORITHM 3 - K-Nearest Neighbor w/ tuned 'k'
    set.seed(9)
    knn_fit <- train(Type ~ ., method = "knn", 
                     data = train_wineData,
                     tuneGrid = data.frame(k = seq(3, 19, 2)))
    
    NumNeighbors_plot <- ggplot(knn_fit, highlight = TRUE) + 
                         ggtitle("Neighbors Tuned for Accuracy")
    NumNeighbors_plot
    
    knn_fit$bestTune  # 11 neighbors
    
    knn_fit$results
    # FINDINGS
      # (1) All k's produce accuracy >92%.  Such accuracy appears to be a result of wine 'types' being meaningful categories.  
      # (2) Based on the training data, k = 11 maximizes accuracy (94.1%)
    
    set.seed(9)
    knn_confusionMatrix <- confusionMatrix(predict(knn_fit, test_wineData, type = "raw"), test_wineData$Type)
    knn_confusionMatrix$overall["Accuracy"]
    # High overall accuracy (97.2%), specificity, and sensivity, with one Class 2 observation being misclassified as Class 3.

    

# EXPLORING DIMENSION REDUCTION WITH PCA
# The dominance of select predictors and intra-predictor correlation indicates that dimensino reduction might (also) be a fruitful approach
    
    set.seed(9)
    pcaLoadings <- prcomp(train_wineData[,c(2:14)], center = TRUE, scale. = TRUE)

  # Visualizations
    # Scree PLot - shows 2 elbows, the first after component 3 (66.3% of cum. variance) and the second after component 6 (86.7% of cum. variance).
      fviz_eig(pcaLoadings, main = "Scree plot - Wine PCA", xlab = "Component Num")
    
    # Cumulative Variance Plot
      cumVariancePlot <- plot(cumsum(pcaLoadings$sdev^2 / sum(pcaLoadings$sdev^2)), type = "b", ylim = 0:1, main = "Cumulative Variance Explained in PCA", xlab = "Component Number", ylab = "Cum. % variance explained")
      summary(pcaLoadings)$importance[, 1:6]  # Cumulative variance explained from second elbow is 86.3%.
    
  # Fitting & Predicting
    # Taking first 6 PCs and append to training wine types
      set.seed(9)
      trainPCA <- as.data.frame(cbind(train_wineData[, 1], pcaLoadings$x[, 1:6]))
      names(trainPCA) <- c("Type", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6")
      trainPCA$Type <- as.factor(trainPCA$Type)
    
    # Use PCs in K-NN model    
      set.seed(9)
      knnPCA_fit <- train(Type ~ ., method = "knn", 
                          data = trainPCA,
                          tuneGrid = data.frame(k = seq(3, 19, 2)))
      
      
      NumNeighborsPCA_plot <- ggplot(knnPCA_fit, highlight = TRUE) + 
                              ggtitle("Neighbors Tuned for Accuracy")
      NumNeighborsPCA_plot
      
      knnPCA_fit$bestTune  # Optimal tune - 17 neighbors
      
    # Taking first 6 PCs and append to training wine types
      set.seed(9)
      wineTestPCA <- predict(pcaLoadings, newdata = test_wineData)
      testPCA <- as.data.frame(cbind(test_wineData[, 1], wineTestPCA[, 1:6]))
      names(testPCA) <- c("Type", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6")
      
      testPCA

      knnPCA_confusionMatrix <- confusionMatrix(predict(knnPCA_fit, testPCA, type = "raw"), test_wineData$Type)
      knnPCA_confusionMatrix  
      # As expected, modeling with uncorrelated, linear combinations of the original 13 parameters (via PCA) produces high accuracy (100%).
      # Again, under broader circumstances such accuracy could represent overfitting.
      # Nevertheless, given the issue being addressed is distinguishing between the three win tyeps, the PCA-KNN approach performed above is effective.
    

# MODEL SELECTION FOR VALDATION DATA
    methods <- c("Decision Tree", "Random Forest", "KNN", "PCA-KNN")
    
    accuracies <- c(round(cdt_ConfusionMatrix$overall["Accuracy"]*100,1), 
                    round(rf_confusionMatrix$overall["Accuracy"]*100,1),
                    round(knn_confusionMatrix$overall["Accuracy"]*100,1),
                    round(knnPCA_confusionMatrix$overall["Accuracy"]*100,1))

    model_Results <- data.frame(Methods = methods, 
                                Accuracy = accuracies)
    
    model_Results
    # Revisting the effectiveness each approach, shows that each approach yields high accuracy.
    # This relatively easy to predict data problem presents a unique, albeit somewhat uncommon challenge:  What should the decision rule be when deciding between comparably high-performing models?
    # In order to maintain straight-forward interpretability and avoid models with the highest likelihood of overfitting, the vanilla K-Nearest Neighbor model will be used on the "new" (validation) data.
    
    validate_knn <- confusionMatrix(predict(knn_fit, validation_wineData, type = "raw"), validation_wineData$Type)
    
    # Final results
    validate_knn$table
    validate_knn$overall["Accuracy"]  # 94.1%
    
