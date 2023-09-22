# [Project Overview: Vitual Avatars with Stable Diffusion: View project source code on Github](https://github.com/rajinipreethajohn/VirtualAvatars_StableDiffusion/blob/main/README.md)

<ins>Project Description:</ins> This project was a fun project aimed at creating virtual avatars of me using Stable diffusion XL on Hugging Face. I trained the model with ~7 images of me on  Google Colab and generated my vitual avatars. The model is able to generate a wide variety of virtual avatars of me based on my prompts.

<ins>Technology:</ins> Python

<ins>Tools and techniques used:</ins> **Stable diffusion**

<ins>Project tasks:</ins>
*Download a copy of Autotrain of stable diffusion from Hugging Face for Google Colab.
*Make necessary changes to the code cells to fit your project needs.
*Create a folder for train data
*Import good quality images that you will train the model on (Mine were ~7)
*Run the code cells
*Save the results in a new folder, so it does not wipe out the images of your train data
*Be creative in your prompts to get a wide variety of virtual avatars

<ins>Topics:</ins>  **1. Get a copy of Auto-train from Hugging face 2. Create a folder for train data with images 3. Run the code 4. Save generated images 5.Be creative with prompts**

![graph](/images/IMG_4414.JPG)
![graph](/images/im8.JPEG)
![graph](/images/IMG_4377.JPG)


# [Project Overview: Face Body Recognition in real time: View project source code on Github](https://github.com/rajinipreethajohn/FaceBodyRecognition_ComputerVision)

<ins>Project Description:</ins> This project is an implementation of a computer vision in order to real time detect: face, left and right hands and body poses and map it to expressions of happy, sad or being victorious.

<ins>Technology:</ins> Python

<ins>Tools and techniques used:</ins> **Mediapipe, Open CV python, NumPy, Pandas, SKlearn**

<ins>Project tasks:</ins>
*Import the required libraries: mediapipe, cv2, pickle, np, pd, StandardScaler, and the classifiers (LogisticRegression, RidgeClassifier, RandomForestClassifier, GradientBoostingClassifier) from sklearn.
*Load the pre-trained model using pickle.load() from a .pkl file.
*Initialize the video capture using cv2.VideoCapture().
*Initialize the Holistic model from mp_holistic.Holistic() with the desired parameters.
*Read frames from the video capture in a loop.
*Preprocess the frame by converting it to RGB and making it writeable.
*Process the frame using the Holistic model and obtain the landmarks for face, hands, and body.
*Recolor the frame back to BGR for rendering.
*Draw the landmarks on the frame using mp_drawing.draw_landmarks() for face, right hand, left hand, and pose.
*Extract the coordinates of the landmarks and create a feature row.
*Create a pandas DataFrame X from the feature row.
*Predict the body language class and the probabilities using the pre-trained model.
*Display the body language class and probabilities on the frame using cv2.putText() and cv2.rectangle().
*Show the resulting frame with the annotations using cv2.imshow().
*Exit the loop when the 'q' key is pressed.
*Release the video capture and destroy all OpenCV windows.

<ins>Topics:</ins>  **1. Load pre-trained model using pickle file 2. Use videocaopture to and mediapipe 2a. Draw landmarks on face, hands and poses 2b. Map the expressions and poses back to the pre-trained model in real-time **

![graph](/images/FB.png)


# [Project Overview: Image Converter Streamlit App: View project source code on Github](https://github.com/rajinipreethajohn/Image_Converter_App/blob/main/README.md)

<ins>Project Description:</ins> This project is an implementation of a computer vision which allows you to convert your favorite photo to different effects such as a pencil sketch, a grayscale image, or an image with a blurring effect.

<ins>Technology:</ins> Python

<ins>Tools and techniques used:</ins> **NumPy, Streamlit, CV2, Pillow**

<ins>Project tasks:</ins>
*Upload your photo by clicking the "Browse files" button.
*Select the desired conversion option from the sidebar:
 *Original: Displays the original image without any conversion.
 *Gray Image: Converts the image to grayscale.
 *Black & White: Converts the image to black and white with adjustable intensity.
 *Pencil Sketch: Converts the image to a pencil sketch with adjustable intensity.
 *Blurred Effect: Applies a blurring effect to the image with adjustable intensity.
*Adjust the intensity sliders (if applicable) to control the effect.
*The "After" image will update automatically based on your selection.

<ins>Topics:</ins>  **1. Upload image 2. Select from the different filters 2a. Original 2b. Gray image filter 2c. Black & White filter 2d. Sketch filter 2e.Blurred Effect filter**

![graph](/images/img3.png)

# [Project Overview: Image Classifier using a CNN architecture : View project source code on Github](https://github.com/rajinipreethajohn/ImageClassification_CNN_85Accuracy/blob/main/CIFAR10_CNN.ipynb)

<ins>Project Description:</ins> This project is an implementation of a convolutional neural network (CNN) for image classification using the CIFAR-10 dataset. The goal of this project is to train a model to recognize and classify images into one of ten classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

<ins>Technology:</ins> Python

<ins>Tools and techniques used:</ins> **NumPy, Keras, Tensorflow**

<ins>Project tasks:</ins>
*Load CIFAR 10 dataset
*Preprocessing
 normalizing the pixel values to the range of 0 to 1 and converting the labels to one-hot encoding
*Define the model architecture: CNN architecture using the CNN Sequential model
*Compile the model
 compile the model by specifying the loss function, optimizer, and evaluation metrics
*Train the model
*Evaluate the model
*Conclusion
*Saving the model for reproducing and increasing model accuracy

<ins>Topics:</ins>  **1.Data Loading 2.Visualizing data 4. Preprocessing data 5. Define model architecture 6. Train and evaluate model 7.Make predictions 8.Evaluate predictions**

![graph](/images/CNN_85.png)


# [Project Overview: Spam/Ham Classifier : View project source code on Github](https://github.com/rajinipreethajohn/SPAM_classifier/blob/main/SpamClassifier.ipynb)

<ins>Project Description:</ins> In this Project, I have come up with a classifier that classifes SMS mesages into either Spam or Ham classes

<ins>Technology:</ins> Python

<ins>Tools and techniques used:</ins> **Pandas, NumPy, Matplotlib, Seaborn, Classification, Sklearn, NLP, Nltk**

<ins>Project tasks:</ins>
*Importing libraries
*Loading data
*Data exploration *Feature engineering *Oulier detection
*Data preprocessing *Cleaning text *Tokenization *Removing stopwords *Lemmatization
*Vectorization
*Model building
*Evaluating models
*End

<ins>Topics:</ins>  **1.Data Manipulation 2.Data Visualization 3.Importing & Cleaning Data 4. Preprocessing data 5. Natural Language Processing 6. Classification**

![graph](/images/spam_ham.png)


# [Project Overview: Recommendation engine for movies : View project source code on Github](https://github.com/rajinipreethajohn/movies-recommendation-engine/blob/main/Recommendation%20Engine%20copy.ipynb)

<ins>Project Description:</ins> In this Project, I have come up with a recommendation engine which is based on three different analyses. The first recommendation is based on demographics, the second on content and the last on collaborative analyses.

<ins>Technology:</ins> Python

<ins>Tools and techniques used:</ins> **Pandas, NumPy, Matplotlib, Seaborn, Natural Language Processing(NPL)**

<ins>Project tasks:</ins>
*Importing libraries
*Loading cleaning
*Data cleaning
*Data preprocessing
*Building a model based on demographics
*Building a model based on content
*Building a model based on collaborative analysis
*Evaluating models
*Conclusion

<ins>Topics:</ins>  **1.Data Manipulation 2.Data Visualization 3.Importing & Cleaning Data 4. Preprocessing data 5. Natural Language Processing**

![graph](/images/popular_movies.png)


# [Project Overview: Customer segmentation using unsupervised clustering : View project source code on Github](https://github.com/rajinipreethajohn/Customer-Segmentation/blob/main/Customer%20Segmentation.ipynb)

<ins>Project Description:</ins> In this Project, I have performed unsupervised clustering on customer data based on the different features provided in the dataset such as demographic details and also their income and spending habits.

<ins>Technology:</ins> Python

<ins>Tools and techniques used:</ins> **Pandas, NumPy, Matplotlib, Seaborn, PCA, Kmeans, Agglomerative clustering**

<ins>Project tasks:</ins>
*Importing libraries
*Loading cleaning
*Data cleaning
*Data preprocessing
*Principal component analysis and dimesionality reduction
*Kmeans, agglomerative clustering
*Evaluating models
*Customer profiling
*Conclusion

<ins>Topics:</ins>  **1.Data Manipulation 2.Data Visualization 3.Importing & Cleaning Data 4. Preprocessing data 5. PCA & dimesionality reduction 6. Kmeans and agglomerative clustering 7. Customer profiling**

![graph](/images/cluster.png)

# [Project Overview: Forecasting power generated by a Windturbine using Machine Learning models : View project source code on Github](https://github.com/rajinipreethajohn/Forescasting-Power-generated-by-a-WindTurbine/blob/main/Predict%20Wind%20Power%20Output%20with%20four%20different%20models%20(1).ipynb)

<ins>Project Description:</ins> In this Project, we will be looking at Windturbine data and use the time-series dataset to forecast the power generated for the next 15 days based on the Windspeed.

<ins>Technology:</ins> Python

<ins>Tools and techniques used:</ins> **Pandas, Pandas profiling, NumPy, Matplotlib, Seaborn, SARIMA, LSTM, XGBoost, Random Forest**

<ins>Project tasks:</ins>
*Import the data 
*Exploratory data analysis - using pandas profiling and then some boxplots 
*Missing values 
*Graph power production 
*Graph together 
*What about other correlations? 
*Look for a pattern of power generation versus wind speed 
*Define a function to plot this graph and then use curve_fit from scipy to solve for the optimal parameters and finally graph against the measured values. *Transparency of measured values is set very low - hence the outliers look like they have disappeared 
*SARIMA 
*Accuracy statistics 
*Now to investigate some other models to see if a better prediction can be made- XGBoost/ Random Forest 
*LSTM prediction 
*Results and discussion

<ins>Topics:</ins>  **1.Importing & feature analysis 2. Data visualization & analysis 3.Feature reduction 4. Cleaning & manipulating data 5. Pre-prosessing completion 6. Applying machine learning models**

![graph](/images/ForecastingActivepower.png)

# [Project Overview: Using SQL on Netflix Data: View project source code on Github](https://github.com/rajinipreethajohn/SQL-on-Netflix-data/blob/main/Using%20SQL%20on%20Netflix%20data.ipynb)

<ins>Project Description:</ins> This project is a demonstration of SQL Syntax using the Netflix Movies and TV shows Dataset. We showcase example query on key SQL topics such as SQL Keywords,Data filtering, Joins, Unions, Aggregate functions, temp tables,Window functions etc. This Notebook will serve as an excellent one stopper for reference while writing related SQL projects.

<ins>Technology:</ins> Python, SQL

<ins>Tools and techniques used:</ins> **Pandas, NumPy, Matplotlib, Seaborn, SQL, SQLite **

<ins>Project tasks:</ins> 
*select
*select distinct 
*select where
*select and,or,not
*order by
*limit values
*Min,Max,count,avg,sum
*like
*in
*between
*joins
*unions
*case statements
*sub queries
*coalesce
*convert/cast
*lag/lead
*row number
*dense rank
*with

<ins>Topics:</ins>  **1.Data Cleaning 2.Data Manipulation 2.Data Visualization 3.Querying with SQL**

![graph](/images/SQLnetflix.png)

# [Project Overview: Stroke prediction by deploying seven different Machine Learning Models : View project source code on Github](https://github.com/rajinipreethajohn/Stroke-prediction/blob/main/Stroke%20Dataset-%20ML%20models%20and%20prediction.ipynb)

<ins>Project Description:</ins> Predicting the probability of a person suffering stroke based on their age, gender, bmi, hypertension,	heart_disease,	ever_married,	work_type,	Residence_type,	avg_glucose_level and smoking_status. Used various ML models to predict and provided the accuracy of these ML models.

<ins>Technology:</ins> Python

<ins>Tools and techniques used:</ins> **Pandas, NumPy, Plotly, Matplotlib, Seaborn, sklearn ML models (Linear Regression, K NEarest Neighbor (KNN), Support Vector Machine (SVM),Gaussian Naive Bayes (GNB), Decision Tree, Random Forest, Gradient Boosting), xgboost, metric score evaluation with sklearn**

<ins>Project tasks:</ins> 
*Importing Libraries
*Color Palettes 
*Reading Dataset
*Exploratory data analysis- EDA
*Pandas profiling
*Auto visualization
*Model implementation

<ins>Topics:</ins>  **1.Data Cleaning 2.Data Manipulation 2.Data Visualization 3.Programming 4. Deploying Machine Learning Models**

![graph](/images/Stroke1.png)

# [Project Overview: The trending topics in Machine Learning using Natural Language Processing (NLP) : View project source code on Github](https://github.com/rajinipreethajohn/The-Trending-Topics-in-Machine-Learning/blob/main/The%20Hottest%20Topics%20in%20Machine%20Learning/notebook.ipynb)

<ins>Project Description:</ins> Neural Information Processing Systems (NIPS) is one of the top machine learning conferences in the world where groundbreaking work is published. In this Project, I have analyzed a large collection of NIPS research papers from (1987 to 2017) to discover the latest trends in machine learning.

<ins>Technology:</ins> Python

<ins>Tools and techniques used:</ins> **Pandas, Matplotlib, regular expression library, wordcloud library, CountVectorizer & Natural Language Processing (NPL) from sklearn**

<ins>Project tasks:</ins>
  *Loading the NIPS papers
  *Preparing the data for analysis
  *Plotting how machine learning has evolved over time
  *Preprocessing the text data
  *A word cloud to visualize the preprocessed text data
  *Prepare the text for Latent Dirichlet allocation- LDA analysis
  *Analysing trends with LDA
  *The future of machine learning

<ins>Topics:</ins>  **1.Data Manipulation 2.Data Visualization 3.Machine Learning 4.Probability & Statistics 5.Importing & Cleaning Data**

![graph](/images/NLP.png)



# [Project Overview: Investigating Netflix Movies and Guest Stars in The Office : View project source code on Github](https://github.com/rajinipreethajohn/Netflix/blob/main/Investigating%20Netflix%20movies%20%26%20guest%20stars%20in%20the%20office.ipynb)

<ins>Project Description:</ins> Netflix! Boasting over 200 million subscribers as of January 2021. In this project I have discovered how Netflix’s movies are getting shorter over time and which guest stars appear in the most popular episode of "The Office".

<ins>Technology:</ins> Python

<ins>Tools and techniques used:</ins> **Pandas, Matplotlib**

<ins>Project tasks:</ins> 
*Loading your friend's data into a dictionary
*Creating a DataFrame from a dictionary
*A visual inspection of our data
*Loading the rest of the data from a CSV
*Filtering for movies!
*Creating a scatter plot
*Digging deeper
*Marking non-feature films
*Plotting with color!
*What next?

<ins>Topics:</ins>  **1.Data Manipulation 2.Data Visualization 3.Programming**

![graph](/images/Netflix.png)




# [Project Overview: The Android app market on Google Play : View project source code on Github](https://github.com/rajinipreethajohn/Android-App-Market/blob/main/Android%20app%20market.ipynb)

<ins>Project Description:</ins> Mobile apps are everywhere.In this project, I did a comprehensive analysis of the Android app market by comparing over ten thousand apps in Google Play across different categories. The insights in the data help us devise strategies to drive growth and retention.

<ins>Technology:</ins> Python

<ins>Tools and techniques used:</ins> **Pandas, NumPy, Matplotlib, Seaborn**

<ins>Project tasks:</ins>
*Google Play Store apps and reviews
*Data cleaning
*Correcting data types
*Exploring app categories
*Distribution of app ratings
*Size and price of an app
*Relation between app category and app price
*Filter out "junk" apps
*Popularity of paid apps vs free apps
*Sentiment analysis of user reviews

<ins>Topics:</ins>  **1.Data Manipulation 2.Data Visualization 3.Probability & Statistics 4.Importing & Cleaning Data**

![graph](/images/Android.png)



# [Project Overview: The GitHub History of the Scala Language : View project source code on Github](https://github.com/rajinipreethajohn/Scala-Language/blob/main/The%20Github%20history%20of%20the%20Scala%20language.ipynb)

<ins>Project Description:</ins> Open source projects contain entire development histories, such as who made changes, the changes themselves, and code reviews. In this project, I have read in, cleaned up, and visualized the real-world project repository of Scala that spans data from a version control system (Git) as well as a project hosting site (GitHub). With almost 30,000 commits and a history spanning over ten years, Scala is a mature language. Also have found out who has had the most influence on its development and who are the experts.

<ins>Technology:</ins> Python

<ins>Tools and techniques used:</ins> **Joining data with Pandas, Matplotlib**

<ins>Project tasks:</ins>
*Scala's real-world project repository data
*Preparing and cleaning the data
*Merging the DataFrames
*Is the project still actively maintained?
*Is there camaraderie in the project?
*What files were changed in the last ten pull requests?
*Who made the most pull requests to a given file?
*Who made the last ten pull requests on a given file?
*The pull requests of two special developers
*Visualizing the contributions of each developer

<ins>Topics:</ins>  **1.Data Manipulation 2.Data Visualization 3.Importing & Cleaning Data**

![graph](/images/Scala.png)



# [Project Overview: A Visual History of Nobel Prize Winners : View project source code on Github](https://github.com/rajinipreethajohn/Nobel-Prize-Winners/blob/main/A%20visual%20history%20of%20Nobel%20prize%20winners.ipynb)

<ins>Project Description:</ins> In this Project, I have tried to explore a dataset from Kaggle containing a century's worth of Nobel Laureates. Who won? Who got snubbed? 

<ins>Technology:</ins> Python

<ins>Tools and techniques used:</ins> **Pandas, NumPy, Matplotlib, Seaborn**

<ins>Project tasks:</ins>
*The most Nobel of Prizes
*So, who gets the Nobel Prize?
*USA dominance
*USA dominance, visualized
*What is the gender of a typical Nobel Prize winner?
*The first woman to win the Nobel Prize
*Repeat laureates
*How old are you when you get the prize?
*Age differences between prize categories
*Oldest and youngest winners
*You get a prize!

<ins>Topics:</ins>  **1.Data Manipulation 2.Data Visualization 3.Importing & Cleaning Data**

![graph](/images/Female_Nobel_winners.png)
