# Portfolio

CRITERIA TO BE GRADED (HAS TO BE DONE)
1.     The contents of your personal portfolio reflect your contribution to the project, your abilities and what you have learned (y/n).
2.     Your portfolio consists of materials that you either realized individually, or in case of a group effort, a clear statement of what your contribution is in this group effort (y/n).
3.     The (digital) portfolio is written in a very easily accessible way (y/n).
4.     The main document is a reader's guide (index) that shortly introduces your contributions and links to pages where the contributions are described in detail (y/n).
5.     Every contribution should be accessible from the reader's guide in a single click (y/n).
6.     The portfolio consists of links to the Python Notebooks or other evidence material about your contribution on the project that you have finished yourself. 



Datacamp


Notebooks
•	Transfering data to csv (first approaches)
•	K-nearest neighbors->KNearstNeighboors
•	Helped detecting sprints->PredictSprintsTrial(found detect peaks function rest of the code was almost already done the for was by jake and I added regression to it)
•	Detect rotations->DetectRotations/DetectRotationsV2 
•	SGD->LinearModelsStory
•	2d cnn->GPU all links
•	The sprints cript (I gave some ideas major part of the coding was martijn)
Jupyter Notebooks
Filter them
Presentations
	Internal-> 2 one alone another with jake
	External->Last one with martijn
	Research Paper-> Summary took part in summary, introduction, problem description and research questions 


REFLECTION AND EVALUATION
-Reflection on own contribution to the project(needs improvement)
Situation
In this minor the project I was assigned to was project Wheels, the task assigned to the group was to improve the outcomes of collecting data of wheelchair basketball players from the Dutch national team through sensors in the wheelchair.
Task
This was my first time working in a task with such level of importance and with a group of more than 3 people. Knowing this I couldn’t take a main role as I knew almost anything about what to do or how to do it so I had to listen many times as at first, I couldn’t add much, though throughout the project this was improved. Even though my situation, I expected myself to be helpful and at least have some general knowledge on all the aspects in which the project is involved. 
Action
To make my role in the project more relevant I tried to work on many different aspects of it such as model preparation, research in models and data preprocessing.
Reflection
I think that my contribution in some parts due to lack of knowledge and not a complete domain of the language could have been better, but I think I tried my hardest for getting results good enough and tried to make something relevant for the project. I think in the future I should try to get more involved in the decision making and understand better the line in which the project is going, this means to know hat has to be done and why, this could help me to take a more important role in future projects.
	STARR, reflected contribution and 400+ words(263)
-Reflection on own learning objectives
Situation
For the first time thanks to this minor, I got hands into python, also was my first contact with data science and my first project with this level of seriousness, so the situation was really challenging from the first moment
Task
From the first moment my objectives were to get some basic knowledge of python and learn everything I could of machine learning methods, and I wanted to understand how a real project works too, as they all are really valuable for my future.
Action
In order to achieve my goals, I tried from the first moment to take notes of everything that was new for me, which made me have a slower pace compared to others but would help me to understand everything better. Also, I tried to assist to all the lectures and workshops possible as even if they were optional, I knew they would be helpful too
Reflection
In my opinion my first approach to all these new things was good, but I think I went too slow on the python learning, which delayed me in other tasks. So definitely the approach I wanted to take was that, but I should have distinguished better in what parts I should be more conscientious in order to optimize time and efforts

	STARR, reflected learning obectives and 400+ words(213)
-Evaluation on group project as a whole
Situation
For this project our group had 2 electrical engineers, 2 software engineers, 1 informatic engineer and Daan(ask him what were his studies about. This was a really good starting point as we had great variety of fields of expertise which could provide us with different points of view on the problems that could show up.
Task
As good it was to have a great variety of points of view for a same point, because that would help us have more than one opinion on one problem, our main deal would be also to reach a point of agreement between all of us, because as we all come from different fields it would be very possible that our opinions on some actions would be opposed to each other
Action
To avoid any kind of opinion difference, we just wanted to know each other’s opinions before making any decision, so we could make sure that everybody could understand why we are doing something
Reflection
In my opinion the group was good, everybody tried to do its hardest and to communicate with each other. Of course, there were points of non-agreement, but I think that they were solved easily, just by exposing what each other means with their idea and then see how that would affect the course of the project.

	STARR, reflected group project and 400+ words(213)


Predictive Analytics->Knn, SGD(linear models), 
Divide into:
-Selecting a Model
-Configuring a Model
-Training a model
-Evaluating a model
-Visualizing the outcome of a model 



KNN
As a first approach of the project, we decided to split the group into machine learning and data team. The machine learning team was in charge of making the first approaches to any usable method we could use. So, we decided to try make work a decision tree, knn and gaussian naïve bayes. My task was to make a working model of K-nearest Neighbors. This selection was made since they were the first models we were told about, so it would be a very naïve approach to the project in order to get used to the project, python and how machine learning works. Although some research was conducted before to see some examples of the model, also the datacamp resources were helpful(Supervised Learning with scikit-learn)
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn
https://paperswithcode.com/method/k-nn
https://www.kdnuggets.com/2020/04/introduction-k-nearest-neighbour-algorithm-using-examples.html
Filter links
As this was a first approach to machine learning the model was at its simplest so no tunning was done, just tried to work over to small sets of neighbors and see which one provided us with better accuracy. Then the accuracy of both train and test sets was plotted to see the difference between newighbours and apporpiately choose the optimum one

SGD
At this moment we wanted to try new models and we decided we wanted to see how linear models adapted to our dataset. At first, we researched on basic models(explain what models), also there was a team who made learning lab presentation based on svm which is a really well known linear model. Out of all the methods svm was really suitable as it’s a good method for two-class problems and also can perform well in time series dataframe, but for us there was only one problem which was the data frame’s size as it isn’t a good method for big amounts of data. Then we looked for similar options and we found out that the sgd was the one that could fit better out of the linear models we knew. After conducting some research and getting a working code we decided to tune the model through gridsearch providing a wide range of possible parameters and obtaining the best ones out of it, I also balanced the data(tag balanced=TRUE) which provided with not really good results so we decided to drop the model as the accuracy was not what we expected(have to explain why it didn’t work in the end)the data of the dataframe was not absolute??
Show image of confusion matrix/precision or whatever I have

2d CNN
After having our neural network and being in the middle of the 1d CNN development we decided to start working on the 2d CNN. For this task lot of time was required as I didn’t find that much information about how neural networks worked and how to make our data work into a 2d CNN. The main problem with this method was how to adapt our data to a 2d cnn, in the internet there aren’t that many resources about time series data being used for a 2d cnn so at first one option considered was converting the data into a 3d matrix, this method wouldn’t provide with any success, so we decided to make the graphs into images and then save them all, as we would have images and its information ready; the problem comes when there would be lots of problems when labeling the information, as we would need to make sure that only one plot occurred per image and also if a way of making the machine learning code know wether it comes from a previous sprint or its has just started ( Another idea we had was taking the plots into images which was achieved, we have the data almost ready as we only needed one of the most important parts the labels, in this case wether something was a sprint or not, this was impossible as we would need to make sure that there was not more than one sprint per image but also the sprints length don’t have same length so some would take more than 10 secs and others just 3 secs) the code was taken from some webs and videos of youtube(look for them) and also some of the previous code made for the 1d cnn was used, so the model would have been functional if it wasn’t for our dataset problems .so it was almost impossible for us and also we were told in our internal presentation to drop it.
(explain better)


Domain knowledge


Introduction of the subject field
	Today we live in the information age, where information comes and goes with just a few clicks. However, this information is of utmost importance in the workplace, ["The use of analytics is no longer limited to big companies with deep pockets. It’s now widespread, with 59% of enterprises using analytics in some capacity"]. The Applied Data Science minor has given me a first approach to the ins and outs of information manipulation and how to make predictions of events or classify information using large volumes of data and detecting patterns in it, in short the value of data science.
	"The use of analytics is no longer limited to big companies with deep pockets. It’s now widespread, with 59% of enterprises using analytics in some capacity" 

Literature research
	As this was my first approach to data science and python, a minimum of research was required in order to understand the situation posed by the wheels project. The project consists in the detection of patterns and movement classification from IMU sensors' collected data. But before I started learning random things, I stated we needed an understanding of the problem I was facing, in order to find the points on which the project should focus. After some difficulties in finding information about the topic I found a paper that mentions the importance of activity trackers for wheelchair users [https://www.rehab.research.va.gov/jour/2016/536/pdf/jrrd-2016-01-0006.pdf], which, when supported by another paper found by my colleague Martijn [https://dl.acm.org/doi/abs/10.1145/2700648.2809845], which explain the difficulties of impaired athletes to track their activities and collect useful data of them, gave me a better understanding of the situation we were in.

(Look a bit more this part)
After knowing the situation it was decided that we should all look for information about previous studies on the subject, among them I found [https://www.researchgate.net/publication/353663819_Machine_Learning_to_Improve_Orientation_Estimation_in_Sports_Situations_Challenging_for_Inertial_Sensor_Use#pf4] (uses Gaussian Naive Bayes algorithm, a logistic regression, a decisiontree algorithm, and a random forest algorithm) which is a study of IMU sensors in wheelchairs to detect inclinations in the trunk of athletes, I also found another paper [https://www.sciencedirect.com/science/article/pii/S1877050921011121] that made use of machine learning to detect fractal gait patterns in soldiers. On the other hand, other colleagues found different papers quite useful, among them are these, which make use of different methods from those I had found as convolutional neural networks [https://www.sciencedirect.com/science/article/pii/S2666307421000140] and recurrent [http://www.ijpmbs.com/uploadfile/2017/1227/20171227050020234.pdf].


After investigating the situation of the use of machine learning for motion detection and the use of sensors or data collection devices for wheelchair athletes, there were some interesting ideas in the use of different methods, but I still lacked the knowledge to develop the initiatives, among them and the one I focused on is the development of an image recognition model. First I had to do research about the simplest neural networks, for this I used the following link [link here] because it helped me a lot to get started on the subject, then I could understand the convolutional neural networks, thanks to [link here] explain a little I could understand the difference between CNN 1d, 2d and 3d. In this way I could focus on the 2d cnn, for this the following links [] were quite useful although some differed in their implementation approach but still helped me to find a common structure in the code

Translated with www.DeepL.com/Translator (free version)
When assigned to the task of making a functional 2d CNN to detect sprints with our data, first there was some research needed as I had no previous knowledge about image recognition.(explain what point of the project where we) 2d CNN consists of ->explain whats it and what benefits has compared to 1d cnn

Explain concepts
	Dataset: To make use of macine learning it is essential to have a dataset, this consists of a series of contents, of any type (numbers, letters, etc.), within a table. Within the table, each column represents a particular variable called features, and each row represents a particular member of the dataset we are dealing with.
		Feature:Features are the basic building blocks of datasets, for example age, sex are some possible features you can find in a dataset. The quality of the features in your dataset has a major impact on the quality of the insights you will gain when you use that dataset for machine learning
		IMU: They are an electronic device that measures and reports the speed, orientation and gravitational forces of a device using a combination of accelerometers and gyroscopes. The name IMU comes from Inertial Measurement Unit. This sensors can measure in three axis (X, Y, Z) and they are the source of our raw data
			Accelerometers: Device that can measure linear acceleration, can be also used for specific purposes such as inclination and vibration measurement.
			Gyroscopes: Device that can measure and maintain the orientation and angular velocity of an object.
	Data PreProcessing & preparation: The core of a machine learning project is its dataset, but before you can get hands into models you need to prepare and clean your dataset because data may be incomplete (attributes, values or both missing), noisy (data has errors and outliers) or inconsistent (data contains differences in codes or names)
	
		Data Cleaning: Process consisting in identifying data errors and correct them to create a complete and accurate dataset. In the case of our project we had some missing values due to game stops, players being substitued and also there were sitations where sensors stopped working. To solve this, it was needed to fill those blank spaces with NaN (Not a number) values which is an often used technique for data cleaning.
		Data Structuring: Identifying those input variables that are most relevant to the task, this means the selecon of features that will be more relevant, in our project they were wheel rotational speed and frame rotational speed for example. Sometimes its also needed to decompose features in simpler ones to help in capturing more specific relationships.
		Data Transforms & enrichment: Data often must be transformed to make it consistent and turn it into usable information. For example one technique worth mentioning is data balancing, this one has been essential for our project, as its necessary for classification problems and our aim was to classify movements from data. One of our problems was that our dataset was imbalanced, so there were a lot of points which wereN't sprints and few classified sprints. Because of this we couldnt classify appropiately with our models and we had to balnce the dataset so that the amount of sprints classified were similar to the amount of non-sprints
		Data Validation: Machine learning models are vulnerable to poor data quality, to avoid this its important to check the accuracy and quality of source data before training a new model version. Taht way you ensure that anomalies that are infrequent or manifested in incremental data are not silently ignored
	Machine learning:
		Types
			Regression
				Linear
				Polynomial
				Ridge
				Lasso
			Classification
				Logistic reg
				Knn
				SVM
				Decision tree
			Ranking(not in depth)
			Clustering(not in depth)
		Training
			Train set
			Test set
			Optimizer
		Evaluation
			Evaluation models
			Overfitting
			Underfitting
			Loss
			Confusion mtrix
			Precision
			Recall
			Score
	NN
		Kinds of nn
			NN 
			1d CNN
			2d CNN
		Tensors
		Epochs
		Learning rate
		Perceptron layers
		Linear layers
	
			

Explain what are tensors
Convolutional layers
ReLU layers
Pooling layers
a Fully connected layer

Data preprocessing->Detect rotations

When choosing tasks for next sprints I wanted to change from machine learning tasks to data tasks, so I got into the rotation detection. In this task we already had divided the data into chunk of n, also we were using the max values of the chunks among those values because the chunks wore amongst one second so we chose the best value for it(explain better whats the best value and why) I was required to look at different of our features as wheelRotSpeed, … Also I needed to establish parameters for when the rotation started, mainly the criteria was the frame rotationspeed going above a certain value, but also I tried to work on different calculations that could helpto get more accurate on the detection of rotations.
All the measing values had been replaced by NaN at first instance so first I detected Nan from the datframe they were replaced by 0. Explain possibility of rotations not ending in the same plot
Once I had the conditions for a rotations, I plotted them i order o see if they were actually rotations, after trying out many combinations I found out the one providing better results, which aren’t perfect as we are working with max values we can’t find exactly all the rotations and sometimes finds rotations which aren’t 

Communiction
Presentations
	Internal-> 2nd internal alone, last one with jake 
	External->Last one with martijn
	Research Paper-> Abstract took part in summary, introduction, problem description(with jake) and research questions (with jake), again problem description but with daan and finally introduction joined with problem description (adapted by jake, introduction adapted by martijn and collin, and our part) this time i had to make it fluent, aloso i added comments, as feedback to other people parts
