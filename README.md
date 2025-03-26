# Vechicle-Classfication-with-Machine-Learning
The goal of this project is to develop a machine learning model capable of accurately predicting a vehicle’s body type based on various numerical and categorical features. 

Machine Learning Code 

Use Jupyter Notebook through the Anaconda Library to open the .ipynb file to view the code for the program.


Data Collection
The data was collected from carvana.com using import.io, which is a web scraping tool. The tool provided me with 500 rows or queries during the free trial, so there was a limited amount of data. This is a relatively small project, so I didn’t think it was feasible to spend $399 per month, which was the next plan if I wanted to collect more data. During the data collection, the extractor had to extract the URLs of the cars on the listing page and then extract the details of each car from their respective URLs. This process used up a lot of queries because two extractors needed to be built to match the data with the URL of the car. So, while I had 500 queries, only 345 rows of data were collected using the tool. 

Data Description and Analysis
The data collected was only across 4 body types to have enough samples of each class. The body types are hatchback, SUV, sedan, and truck. The dataset contains 345 rows of data with 10 feature columns and one class column to identify each vehicle. It also contains a URL column that holds the URL of the car in each row, but this will be discarded at the beginning of data cleaning because it doesn’t hold any significant correlation with the data. The URL only helps in identifying where the data was collected from. The dataset consists of the following features:
•	fuel_type: The type of fuel the vehicle uses. This includes categories like Electric, Gas, Hybrid, and Flexible Fuel.
•	horsepower: The engine's power output, which is measured in horsepower (HP).
•	MPG (Miles Per Gallon): A measure of the vehicle's fuel efficiency. The data collected uses the city MPG of each vehicle.
•	seats: The number of seats in the vehicle.
•	curb_weight: The weight of the vehicle when it is not carrying passengers or cargo. This is measured in pounds. 
•	length: The overall length of the vehicle, which is measured in inches.
•	width: The width of the vehicle, which is measured in inches.
•	height: The height of the vehicle, which is measured in inches.
•	wheel_base: The distance between the front and rear axles. This feature is also measured in inches.
•	drivetrain: The system that connects the engine to the wheels and determines how power is distributed. The categories include 2WD (Two-Wheel Drive), FWD (Front-Wheel Drive), RWD (Rear-Wheel Drive), AWD (All-Wheel Drive), and 4WD (Four-Wheel Drive).
The final column of the dataset is “body_type”. This is the label that the model will be trained to predict for unknown vehicles. 

Data Preparation and Cleaning
	The dataset needs to be prepared before training the model using different classifiers. The first step will be to discard unwanted columns. There isn’t a substantial number of columns and rows, so the only column that will be discarded is the URL column. The URL is only there to allow for the link of each car to be opened and to double-check the data that is already represented in the other features. There are no null values in the dataset, so there won’t be any data imputing needed. The next step is to change all categorical columns to have numeric representations. The columns that need to be dealt with are the fuel type, drivetrain, and body type. Body type is easily dealt with since it is the class label of each car; it can be identified using numbers 0-3. Each number will correspond to a body type. This is easily done using the LabelEncoder() to transform the body_type column. The next step would be to one-hot encode the fuel type and drivetrain columns so that each value can be represented using 0 and 1 binary values. This will create more columns to represent each value for the features we are encoding. 

![image](https://github.com/user-attachments/assets/8a598eb8-cc60-4eca-ae46-ebdc76087703)
Figure 1: Categorical columns are transformed into numeric values

An example of a column created from one-hot encoding is the fuel_type_Electric column from Figure 1. This identifies whether a car is electric using a 1 or if it doesn’t use electric fuel, it is a 0. This creation of columns means the dataset will now have 18 columns. This raises the question of whether fuel_type and drivetrain are significant enough features to keep since they will add more computation for the model, increasing training time. 

![image](https://github.com/user-attachments/assets/b7ab3ad9-287a-4b9d-9818-91ecec31e062)
Figure 2: Feature Importances Bar Graph

This bar graph represents the importance of each feature in predicting the cars’ body type. This proves that some values of drivetrain and fuel type offer at least some significance for the model. This significance offsets the little complexity that the columns will add to the data. 
Scaling the Data
	Other than the newly created columns, the other columns that were already numerical will need to be scaled. This will put all the data on the same scale, whether scaling all numbers to be between 0 and 1, or to be represented using Z-Scores. The newly created columns are already represented with 0s and 1s, so it would be more efficient to scale the rest of the features between 0 and 1. This will be done using the MinMaxScaler to transform the numerical columns. 

![image](https://github.com/user-attachments/assets/3e5babc3-e0d3-444e-9fc8-6cdc1d926745)
Figure 3: The columns scaled using the MinMaxScaler
	This feature scaling will be more significant when it comes to training specific classifiers using the data. Some classifiers that use the distance of each data point or feature weights to train are heavily affected by the scaling of the data. The final step is to store the data into X and y variables, with X representing all the rows of data with each feature except the class labels. The y variable will be the class labels that the model will try to predict based on the data in X. 
Algorithm Selection
	This is a multidimensional dataset because while there is only one y value for each row, there are multiple x values representing each feature. So classifiers like the perceptron, which don’t work well with multidimensional data, will not be used. The data will be separated into training and testing data. There will be 70% of the rows of data used to train the model, and the other 30% will be used to test how well the model is performing. I chose 6 classifiers to use on the data that will be used to train the model. These will be cut down to 4 classifiers that will be tuned using hyperparameter tuning to get the best accuracy out of the 4 classifiers. The classifier with the highest accuracy on the testing data will be used for the model. The 6 classifiers will be the Random Forest, Decision Tree, Logistic Regression, KNeighbors, Support Vector Machines (SVM), and Bagging (Decision Tree) classifiers. 10-fold Cross Validation will be used to test the initial performance of these classifiers within the training data. 10-fold cross-validation splits the training data into 10 different sections and produces accuracy of how the classifier performed within each section. The data that isn’t within each section is used to train the data, and the section is used like it is the data being tested.



Initial Classifier Accuracy (Same Train/Test Split)
Classifier	Cross-Validation Accuracy
Random Forest	97.5%
Decision Tree	96.3%
Logistic Regression	86.9%
KNeighbors	90.1%
Bagging (Decision Tree)	96.7%
SVM	95%

The most important aspect of the cross-validation testing is that there aren’t any outliers when it comes to each section's accuracy. 

Parameter Tuning
	The 4 classifiers that showed promise when it comes to training the model are the Random Forest, Decision Tree, Bagging (Decision Tree), and the Support Vector Machine (SVM). Utilizing a GridSearch of parameters for each classifier, we will optimize the performance of each classifier and test it using the 10-fold cross-validation. 
 
![image](https://github.com/user-attachments/assets/92c54172-d7cb-46be-aa22-00a61d47d442)
Figure 4: Utilizing GridSearch to optimize the Random Forest Classifier

After all the classifiers have their best parameters selected by GridSearch. Another 10-fold cross-validation will be used to test them before the final test using the actual test data.
Post-Hyperparameter Tuning Cross-Validation
Classifier	Cross-Validation Accuracy
Random Forest	97.5%
Decision Tree	96.7%
SVM	98.3%
Baggin (Decision Tree)	96.7%

Results
	The final step towards having a model is to test the 4 best classifiers using the unknown test data. Each classifier will be across 100 different train/test splits, and then the average accuracy across the 100 runs will be taken. This will be the final test to find the best classifier for this dataset. 
Final Average Accuracy of each classifier
Classifier	Final Average Accuracy
Random Forest	96.3%
Decision Tree	96%
SVM	97.5%
Baggin (Decision Tree)	95.7%

Conclusion
The initial results from cross-validation showed that most classifiers performed well, with accuracies ranging from 86.9% (Logistic Regression) to 97.5% (Random Forest). The best classifier at that point was the Random Forest. But after hyperparameter tuning, SVM took the lead as the best classifier to train the model. The final average accuracy scores showed that SVM outperformed the other classifiers with an accuracy of 97.5%, followed closely by Random Forest (96.3%) and Decision Tree (96%). This suggests that parameter tuning is incredibly impactful in finding the best classifier for a dataset. A mediocre classifier could become the best one with perfect parameters

![image](https://github.com/user-attachments/assets/f79288d4-25d1-4162-bc1e-4ea1c093de59)
Figure 5: Confusion matrix for SVM classifier as the best model

This confusion matrix shows that the model classified all the class labels that are 0 and 3 accurately. These are the numbers for the hatchback and truck body types, respectively. The model accurately identified one sample of each for the sedan and SUV body type. If there was more than one label being inaccurately identified, it could point out that the features don’t provide enough information to identify that class label. Luckily, that is not the case here because 1 out of 30 samples being misidentified is acceptable.
One limitation of this study is the dataset size, which was constrained by the scraping tool's query limit. Expanding the dataset could further improve the model’s generalization. It would also allow for data from other car body types to be implemented. Overall, the results demonstrate that machine learning can effectively classify vehicle body types using structured data, and the SVM algorithm is the best-performing classifier for this dataset. 

