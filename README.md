<!-- wp:heading -->
<h2>High School Math Performance</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Introduction</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>In the world of math education, one of the major issues that universities and educators have is that students do not succeed at mathematics at a satisfactory level and at a rate that is satisfactory. Universities and educators complain of the high failure, drop, and withdrawal rates of their students. This is a problem for students because low performance in math prevents them from pursuing their degrees and careers. It is a problem for universities and educators because it means that the university or educator is not successfully teaching students, not retaining their students, and not satisfying the needs of their students — these problems hurt the profitability and attractiveness of the university and educator.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>If we can gain some insights into what factors most contribute to or hurt student performance in math, we have the potential to solve the above-mentioned problems. If we can produce predictive models that can predict whether a student will pass or fail, that can predict the numerical score of students on math assessments, and that can predict the overall strength and promise of a student, then universities and educators will be able to use these models to better place students at the appropriate level of competence, to better select students for admission, and to better understand the factors that can be improved upon to help students be successful.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>In this paper, we will perform data science and machine learning to a dataset representing the math performance of students from two Portuguese high schools.  The dataset can be found at the link at the end of this article.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>The data file was separated by semicolons rather than commas. I replaced the semicolons by commas. Then, copy and pasted everything into notepad. Then, convert to a csv file using the steps from the following link:</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><a href="https://knowledgebase.constantcontact.com/articles/KnowledgeBase/6269-convert-a-text-file-to-an-excel-file?lang=en_US">https://knowledgebase.constantcontact.com/articles/KnowledgeBase/6269-convert-a-text-file-to-an-excel-file?lang=en_US</a></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Now, I have a nice csv file.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>There are 30 attributes that include things like student age, parent’s education, parent’s job, weekly study time, number of absences, number of past class failures, etc. There are grades for years 1, 2, and 3; these are denoted by G1, G2, and G3. The grades range from 0–20. G1 and G2 can be used as input features, and G3 will be the main target output.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Some of the attributes are ordinal, some are binary yes-no, some are numeric, and some are nominal. We do need to do some data preprocessing. For the binary yes-no attributes, I will encode them using 0’s and 1’s. I did this for schoolsup, famsup, paid, activities, nursery, higher, internet, and romantic. The attributes famrel, freetime, goout, Dalc, Walc, and health are ordinal; the values for these range from 1 to 5. The attributes Medu, Fedu, traveltime, studytime, failures are also ordinal; the values range from 0 to 4 or 1 to 4. The attribute absences is a count attribute; the values range from 0 to 93. The attributes sex, school, address, Pstatus, Mjob, Fjob, guardian, famsize, reason are nominal. For nominal attributes, we can use one-hot encoding. The attributes age, G1, G2, and G3 can be thought of as interval attributes.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>I one-hot encoded each nominal attribute, one at a time. I exported the dataframe as a csv file each time, relabeling the columns as I go. Finally, I reordered the columns.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Here is the python code:</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted"> import numpy as np
 import pandas as pd
 dataset = pd.read_csv(‘C:\Users\ricky\Downloads\studentmath.csv’)
 X = dataset.iloc[:,:-1].values
 Y = dataset.iloc[:,32].values
 from sklearn.preprocessing import LabelEncoder, OneHotEncoder
 labelencoder_X = LabelEncoder()
 #Encoding binary yes-no attributes
 X[:,15] = labelencoder_X.fit_transform(X[:,15])
 X[:,16] = labelencoder_X.fit_transform(X[:,16])
 X[:,17] = labelencoder_X.fit_transform(X[:,17])
 X[:,18] = labelencoder_X.fit_transform(X[:,18])
 X[:,19] = labelencoder_X.fit_transform(X[:,19])
 X[:,20] = labelencoder_X.fit_transform(X[:,20])
 X[:,21] = labelencoder_X.fit_transform(X[:,21])
 X[:,22] = labelencoder_X.fit_transform(X[:,22])
 #Encoding nominal attributes
 X[:,0] = labelencoder_X.fit_transform(X[:,0])
 X[:,1] = labelencoder_X.fit_transform(X[:,1])
 X[:,3] = labelencoder_X.fit_transform(X[:,3])
 X[:,4] = labelencoder_X.fit_transform(X[:,4])
 X[:,5] = labelencoder_X.fit_transform(X[:,5])
 X[:,8] = labelencoder_X.fit_transform(X[:,8])
 X[:,9] = labelencoder_X.fit_transform(X[:,9])
 X[:,10] = labelencoder_X.fit_transform(X[:,10])
 X[:,11] = labelencoder_X.fit_transform(X[:,11])
 onehotencoder = OneHotEncoder(categorical_features = [0])
 X = onehotencoder.fit_transform(X).toarray()
 from pandas import DataFrame
 df = DataFrame(X)
 export_csv = df.to_csv (r’C:\Users\Ricky\Downloads\highschoolmath.csv’, index = None, header=True)</pre>
<!-- /wp:preformatted -->

<!-- wp:paragraph -->
<p>Using seaborn, we can take a look at some visualizations. Here’s a histogram for “absences”.</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://miro.medium.com/max/508/1*G-D8u-goqafC14K1uI6GYA.png" alt=""/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>Note that most students have a very low number of absences and that as the absences increases the number of students with that many absences decreases.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Let’s look at the distributions for grades G1, G2, and G3:</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://miro.medium.com/max/508/1*amsHLtoIEQ6iasf8i0LWUg.png" alt=""/></figure>
<!-- /wp:image -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://miro.medium.com/max/508/1*OrxTK-5M4AEwzKrFDbsdGg.png" alt=""/></figure>
<!-- /wp:image -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://miro.medium.com/max/508/1*26Zid6mP7mCkMQMYuwDuUg.png" alt=""/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>For G1, the first year grade, the scores look normally distributed. For G2 and G3, there are two maxima if you look at the curve; a certain number of students seem to have low scores near 0.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>One might wonder how age affects the grades G1, G2, and G3. Here are some boxplots for age against G1:</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://miro.medium.com/max/515/1*FZvohfbW4P6AySe3-0dcYQ.png" alt=""/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>The median score appears to have a local maximum at age 16, decreasing until it reaches age 19, then increasing sharply at age 20. The same thing can be seen with age versus G2:</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://miro.medium.com/max/528/1*TmvjUAS7m6z6Pfq1FcN1dA.png" alt=""/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>In the third year, the local maximum at age 16 disappears, but the sharp increase at age 20 remains:</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://miro.medium.com/max/528/1*5l3tQ2QwluLlcCtLih7ZEg.png" alt=""/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>Note also that the range in scores gets tighter at ages 19 and 20.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>We might also wonder whether males or females perform better. In all of three years, the median score for males is higher than for females. For instance, here’s the boxplots for Male vs G3:</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://miro.medium.com/max/528/1*hoFY_2RwqIOKHAzvBxp0QA.png" alt=""/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>If we apply linear regression to model G3 as a function of G1, we get the following:</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://miro.medium.com/max/469/1*0OzEPFl64Dkz_UZqr4nPlQ.png" alt=""/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>As G1 score increases, so does G3 score. Similarly, we can see a linear relationship between G3 and G2:</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://miro.medium.com/max/469/1*VaiDiO3gvJ9u1z1r_pnmGg.png" alt=""/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>Here is the python code:</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted"> import numpy as np
 import pandas as pd
 dataset = pd.read_csv(‘C:\Users\ricky\Downloads\studentmathdummified.csv’)
 X = dataset.iloc[:,:-1].values
 Y = dataset.iloc[:,50].values
 import matplotlib.pyplot as plt
 import seaborn as sns
 %matplotlib inline
 plt.rcParams[‘figure.figsize’]=8,4
 import warnings
 warnings.filterwarnings(‘ignore’)
 #Distribution
 vis1 = sns.distplot(dataset[“Fedu”])
 #Boxplots
 vis2 = sns.boxplot(data=dataset, x=”Male”,y=”G3")
 #Linear regression model
 vis3 = sns.lmplot(x=”G2", y=”G3", data=dataset)</pre>
<!-- /wp:preformatted -->

<!-- wp:paragraph -->
<p><strong>Simple Linear Regression</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Now, let’s apply some machine learning to our dataset. It’s a good guess that G3 depends on G1 in a linear fashion. We can see this more clearly by applying simple linear regression with G3 as the dependent variable and G1 as the independent variable.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>First, we let X be the matrix consisting of the first 50 columns of our dataset studentmathdummified.csv. Then, let Y be the last column of the dataset — namely, G3.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>We then split our dataset into a training set and a test set. We apply linear regression to our training set to train our simple linear regression model; we then apply the model to our test set, and we can compare the predicted Y values with the actual Y values of our test set.</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted"> Here is the python code:
 #Simple Linear Regression
 #Importing the libraries
 import numpy as np
 import pandas as pd
 import matplotlib.pyplot as plt
 #Importing the dataset
 dataset = pd.read_csv(“studentmathdummified.csv”)
 X = dataset.iloc[:,:-1].values
 Y = dataset.iloc[:,-1].values
 #Splitting the dataset into the Training set and Test set
 from sklearn.model_selection import train_test_split
 X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)
 #Fitting Simple Linear Regression to the Training set
 from sklearn.linear_model import LinearRegression
 regressor = LinearRegression()
 regressor.fit(X_train[:,48:49],Y_train)
 #Predicting the Test set results
 Y_pred = regressor.predict(X_test[:,48:49])
 X_train[:,48:49]</pre>
<!-- /wp:preformatted -->

<!-- wp:paragraph -->
<p> Let’s take a look at the linear model attained from training it on our training set. Here is what it looks like: </p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://miro.medium.com/max/528/1*xDEH-j6sc3jmwgfnxkkYRw.png" alt=""/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>In red is the scatter plot of our training set G3 values versus the training set G1 values. The blue line is our linear regression model.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Here is the python code used to generate the graph:</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted"> #Visualizing the Training set results
 plt.scatter(X_train[:,48:49], Y_train, color = ‘red’)
 plt.plot(X_train[:,48:49],regressor.predict(X_train[:,48:49]), color = ‘blue’)
 plt.title(‘G3 vs G1 (Training set)’)
 plt.xlabel(‘G1’)
 plt.ylabel(‘G3’)
 plt.show()</pre>
<!-- /wp:preformatted -->

<!-- wp:paragraph -->
<p> Now, let’s see how the linear regression model performs on the test set. Here is the scatter plot of the test set G3 values versus the test set G1 values in red and the linear model in blue. </p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://miro.medium.com/max/528/1*RsSTq6Wx2yZBlGNlXJ5J2Q.png" alt=""/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>As you can see, the linear regression model performs extremely well on the test set.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Here is the python code to generate the graph:</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted"> #Visualizing the Test set results
 plt.scatter(X_test[:,48:49], Y_test, color = ‘red’)
 plt.plot(X_train[:,48:49],regressor.predict(X_train[:,48:49]), color = ‘blue’)
 plt.title(‘G3 vs G1 (Test set)’)
 plt.xlabel(‘G1’)
 plt.ylabel(‘G3’)
 plt.show()</pre>
<!-- /wp:preformatted -->

<!-- wp:paragraph -->
<p> We can see a very similar relationship between G3 and G2. We can apply simple linear regression with G3 as the dependent variable and G2 as the independent variable. Here are the results: </p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://miro.medium.com/max/515/1*GtH4YyzDHIjM9H4nwVpgUg.png" alt=""/></figure>
<!-- /wp:image -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="https://miro.medium.com/max/515/1*WttS-zEdVIQmDcrQ3htxmA.png" alt=""/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p><strong>Multiple Linear Regression</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>So far, we’ve applied linear regression using a single variable, either G1 or G2. Perhaps the other independent variables have an effect on G3. To see, we can apply multiple linear regression where we take into account all of the independent variables.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>First, in order to avoid the dummy variable trap, I deleted the columns for GP, Male, urban,LE3, Apart,mother_at_home, father_at_home, reason_course, guardian_other. I named the new dataset ‘dataset_trap’. Then, I defined X and Y using dataset_trap. I split the dataset into a training set and a test set, trained the multiple linear regression model on the training set, and applied the model to the X_test.</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted"> Here is the python code:
 #Multiple Linear Regression
 #Importing the libraries
 import numpy as np
 import pandas as pd
 import matplotlib.pyplot as plt
 #Importing the dataset
 dataset = pd.read_csv(“studentmathdummified.csv”)
 #Avoiding the dummy variable trap
 #Dropping GP, Male, urban,LE3, Apart,mother_at_home, father_at_home, reason_course, guardian_other
 dataset_trap = dataset.drop(dataset.columns[[0,2,4,6,8,10,15,20,26]],axis=1)
 #Define X and Y using dataset_trap
 X = dataset_trap.iloc[:,:-1].values
 Y = dataset_trap.iloc[:,-1].values
 #Splitting the dataset into the Training set and Test set
 from sklearn.model_selection import train_test_split
 X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)
 #Fitting Multiple Linear Regression to the Training set
 from sklearn.linear_model import LinearRegression
 regressor = LinearRegression()
 regressor.fit(X_train,Y_train)
 #Predicting the Test set results
 Y_pred = regressor.predict(X_test)</pre>
<!-- /wp:preformatted -->

<!-- wp:paragraph -->
<p>Comparing the predicted Y values with the test set Y values, the model did a pretty good job but not an excellent job. Perhaps we can get better performance by only including the attributes that have a significant effect on G3. We can do this by performing backward elimination. If we do this, using a threshold of 0.05 for the p-value, we end up with the attributes Age, famrel, absences, G1, and G2 as our optimal set of attributes. The age of the student, the quality of family relationships, the number of absences, and the grades in the first and in the second years are found to be the most significant attributes.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Here is the python code for performing backward elimination.</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted"> #Performing backward elimination
 import statsmodels.formula.api as sm
 X = np.append(arr = np.ones((395,1)).astype(int), values = X, axis = 1)
 X_opt = X[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41]]
 regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
 regressor_OLS.summary()</pre>
<!-- /wp:preformatted -->

<!-- wp:paragraph -->
<p> We eliminate the column in X_opt that corresponds to the independent variable with highest p-value over 0.05. Then, we perform the following code again with the new X_opt: </p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">X_opt = X[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41]]
 regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
 regressor_OLS.summary()</pre>
<!-- /wp:preformatted -->

<!-- wp:paragraph -->
<p> We repeat this process until all independent variables have p-value below 0.05. We end up with: </p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted"> X_opt = X[:, [19,33,39,40,41]]
 regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
 regressor_OLS.summary()</pre>
<!-- /wp:preformatted -->

<!-- wp:paragraph -->
<p>Columns 19, 33, 39, 40, 41 correspond to the attributes Age, famrel, absences, G1, and G2.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Instead of performing backward elimination manually, we can also use the following code to perform it automatically:</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted"> import statsmodels.formula.api as sm
 def backwardElimination(x, sl):
 numVars = len(x[0])
 for i in range(0, numVars):
 regressor_OLS = sm.OLS(Y, x).fit()
 maxVar = max(regressor_OLS.pvalues).astype(float)
 if maxVar &gt; sl:
 for j in range(0, numVars — i):
 if (regressor_OLS.pvalues[j].astype(float) == maxVar):
 x = np.delete(x, j, 1)
 regressor_OLS.summary()
 return x
 SL = 0.05
 X_opt = X[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41]]
 X_Modeled = backwardElimination(X_opt, SL)
 regressor_OLS = sm.OLS(endog = Y, exog = X_Modeled).fit()
 regressor_OLS.summary()</pre>
<!-- /wp:preformatted -->

<!-- wp:paragraph -->
<p><strong>SVR Regression</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>In this section, we will perform support vector regression using Gaussian kernel. Given our earlier insight that the attributes that have most significance are Age, famrel, absences, G1, and G2, I trained an SVR model using these attributes on a training set. I performed feature scaling on X_train, X_test, and Y_train. I then compared Y_test with Y_pred. The performance was very impressive.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Here is the python code:</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted"> #SVR Regression
 #Importing the libraries
 import numpy as np
 import pandas as pd
 import matplotlib.pyplot as plt
 #Importing the dataset
 dataset = pd.read_csv(“studentmathdummified.csv”)
 #Avoiding the dummy variable trap
 #Dropping GP, Male, urban,LE3, Apart,mother_at_home, father_at_home, reason_course, guardian_other
 dataset_trap = dataset.drop(dataset.columns[[0,2,4,6,8,10,15,20,26]],axis=1)
 #Define X and Y using dataset_trap
 X = dataset_trap.iloc[:,:-1].values
 Y = dataset_trap.iloc[:,-1].values
 #Splitting the dataset into the Training set and Test set
 from sklearn.model_selection import train_test_split
 X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)
 #Feature Scaling
 from sklearn.preprocessing import StandardScaler
 sc_X = StandardScaler()
 X_train = sc_X.fit_transform(X_train[:, [18,32,38,39,40]])
 X_test = sc_X.fit_transform(X_test[:, [18,32,38,39,40]])
 sc_Y = StandardScaler()
 Y_train = sc_Y.fit_transform(Y_train.reshape(-1,1))
 #Fitting SVR Regression to the Training set
 from sklearn.svm import SVR
 regressor = SVR(kernel = ‘rbf’)
 regressor.fit(X_train,Y_train)
 #Predicting the Test set results
 Y_pred = sc_Y.inverse_transform(regressor.predict(X_test))</pre>
<!-- /wp:preformatted -->

<!-- wp:paragraph -->
<p><strong>Decision Tree Regression</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>I performed decision tree regression on the training set, without removing any of the attributes. The performance was very good, even though we didn’t use only the attributes Age, famrel, absences, G1, and G2. Here is the python code:</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted"> #Decision Tree Regression
 #Importing the libraries
 import numpy as np
 import pandas as pd
 import matplotlib.pyplot as plt
 #Importing the dataset
 dataset = pd.read_csv(“studentmathdummified.csv”)
 #Avoiding the dummy variable trap
 #Dropping GP, Male, urban,LE3, Apart,mother_at_home, father_at_home, reason_course, guardian_other
 dataset_trap = dataset.drop(dataset.columns[[0,2,4,6,8,10,15,20,26]],axis=1)
 #Define X and Y using dataset_trap
 X = dataset_trap.iloc[:,:-1].values
 Y = dataset_trap.iloc[:,-1].values
 #Splitting the dataset into the Training set and Test set
 from sklearn.model_selection import train_test_split
 X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)
 #Feature Scaling
 “””from sklearn.preprocessing import StandardScaler
 sc_X = StandardScaler()
 X_train = sc_X.fit_transform(X_train[:, [18,32,38,39,40]])
 X_test = sc_X.fit_transform(X_test[:, [18,32,38,39,40]])
 sc_Y = StandardScaler()
 Y_train = sc_Y.fit_transform(Y_train.reshape(-1,1))”””
 #Fitting Decision Tree Regression to the Training set
 from sklearn.tree import DecisionTreeRegressor
 regressor = DecisionTreeRegressor(random_state = 0)
 regressor.fit(X_train,Y_train)
 #Predicting the Test set results
 Y_pred = regressor.predict(X_test)</pre>
<!-- /wp:preformatted -->

<!-- wp:paragraph -->
<p><strong>Random Forest Regression</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>I also applied random forest regression using 10, 100, and 500 trees. In random forests, a bunch of trees are grown and the average of the predicted values is taken to be the prediction. In the python code, it’s similar to the decision tree regression code except we replace the section on fitting decision tree regression to the training set with the following:</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted"> #Fitting Random Forest Regression to the Training set
 from sklearn.ensemble import RandomForestRegressor
 regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
 regressor.fit(X_train,Y_train)</pre>
<!-- /wp:preformatted -->

<!-- wp:paragraph -->
<p><strong>Assessing Model Performance</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>In order to determine which model is the best, we will perform k-fold cross validation (k=10) for each model and pick the one that has the best accuracy.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>For multiple linear regression using all attributes, I got an accuracy of 80%. For multiple linear regression using only the five attributes Age, famrel, absences, G1, and G2, I got an accuracy of 83%.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>For SVR regression using all attributes, I got an accuracy of 73%. For SVR regression using only the five attributes Age, famrel, absences, G1, and G2, I got an accuracy of 83%.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>For decision tree regression, I got an accuracy of 77%. For decision tree regression using only the five attributes Age, famrel, absences, G1, and G2, the accuracy is 83%.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>For random forest regression, using 10 trees, I got an accuracy of 85%. For 100 trees, I got an accuracy of 86%. Using 500 trees, I got an accuracy of 87%. I’ve tried increasing the number of trees, but the accuracy doesn’t go beyond 87%.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>For random forest regression using only the five attributes Age, famrel, absences, G1, and G2, I got the following: 86% for 10 trees, 88% for 100 trees, 88% for 500 trees. I tried increasing the number of trees, but the accuracy doesn’t go beyond 88%.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>What’s interesting about the above results is that by limiting the attributes to only the five attributes Age, famrel, absences, G1, and G2, the accuracy went up for each model.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>The best performing model appears to be random forest regression using 500 trees. The performance is even better if we limit the attributes to the five attributes Age, famrel, absences, G1, and G2.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Here is the python code:</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted"> #Applying k-fold cross validation
 from sklearn.model_selection import cross_val_score
 accuracies = cross_val_score(estimator=regressor,X=X_train, y=Y_train, cv=10)
 accuracies.mean()
 accuracies.std()</pre>
<!-- /wp:preformatted -->

<!-- wp:paragraph -->
<p><strong>Conclusion</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>In our regression analysis of the dataset, we found that some of the most significant attributes were grades in years 1 and 2, quality of family relationships, age, and the number of absences. The random forest regression with 500 trees turned out to be one of the best performing models with 87–88% accuracy. We also saw a strong linear relationship between the grade in year 3 with the grades in years 1 and 2.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>The dataset can be found here:</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><a href="https://archive.ics.uci.edu/ml/datasets/student+performance">https://archive.ics.uci.edu/ml/datasets/student+performance</a></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance. In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) pp. 5–12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978–9077381–39–7.<br><a rel="noreferrer noopener" href="http://www3.dsi.uminho.pt/pcortez/student.pdf" target="_blank">[Web Link]</a></p>
<!-- /wp:paragraph -->
