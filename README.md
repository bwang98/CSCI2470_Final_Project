##  **Title**
Facial Expression Detection and Emoji Conversion
 
 
## **Who**
Elbert Wu (ewu12), Bohao Wang (bwang98), Ying Sun(ysun141)
 
 
## **Introduction**
Under the exploding development of computer vision, facial recognition is now becoming increasingly important. In fact, facial recognition in today’s era is no longer just about whether some specific facials are from an identical person or not in order to achieve the function of facial recognition unlocking. Instead, facial recognition is now evolving towards the direction of capturing changes in facial expressions and making accurate classifications of people's emotions. At the same time, the rise of the idea, meta, has given us some inspirations. If we can successfully capture people’s expressions through facial images with the application of deep learning techniques and convert them into corresponding emoji expressions in the virtual world, this would help us build more realistic connections between the real world and the virtual world. This project is made up of two parts. The first part is a classification problem under the umbrella of supervised learning to do emotional analysis , and the second part is simply converting the resulting classifications into emojis which have corresponding emotions.



## **Related Work**
Mellouk and Handouzi (2020) summarized the facial available databases and related research about facial emotion recognition using deep learning from 2016 to 2019. Three methods of pre-processing are introduced: deep CNN for FER across several databases, combining data augmentation, rotation correction, cropping, downsampling with 32x32 pixels and intensity normalization before CNN is more efficient, and a novel CNN for detecting AUs of the face was proposed. 

**Citation**: [https://www.sciencedirect.com/science/article/pii/S1877050920318019](https://www.sciencedirect.com/science/article/pii/S1877050920318019)
 
 
## **Data**
The dataset we plan to use is FER-2013. It contains approximately 30,000 facial RGB images of 7 different expressions. 35,887 grayscale 48x48-pixel images are contained. We will do significant preprocessing on it and possible methods we can use are data augmentation, rotation correction, cropping, downsampling with 32x32 pixels and intensity normalization before CNN, and other modeling processes.
 
 
## **Methodology**
Convolutional Neural Network(CNN) would be the basic structure of this project because CNN has been proven to be very effective for capturing information in images. However, we might need to stack several CNN layers with different activation functions together. In this way, we would take full advantages of the characteristics of CNN and increase the adaptability of the model through nonlinear functions
Support Vector Machine(SVM) is considered as one of the most powerful machine learning models to do classifications especially for its ability of doing feature extractions. Through this project, We would like to investigate the possibility of combining SVM and CNN together. 
Based on the above points, our model would be very likely to be a combination of several layers of CNN followed by activation functions such as ReLu in addition to max-pooling and the layer of SVM with different kernels. At the same time, we might need to use modified CNNs such as Visual Geometric Group Network(VGG Net) as well. 

We need to divide the training data into the training set and the validation set. After doing the preprocessing and data augmentation, we would train data batches by batches with different models described above. If the combination of the above models does not seem to work well, we might need to consider only including CNN models. In principle, CNN is enough to capture the image information. 


## **Metrics**
We plan on running several iterations of the classification algorithm on the training set and testing set and seeing how accurate the test results are. The notion of accuracy does apply to this project, since our data is classified already, so it is easy to evaluate how well the model does by calculating the accuracy. Our base goal accuracy is 85% within a reasonable training/testing time (30 minutes atm), our target goal is 95% accuracy, and our stretch goal is 99% accuracy. If possible, a further stretch goal is to significantly decrease the amount of time the model takes to train and test.
 
 
## **Ethics**
Higher accuracy is obtained using deep learning than using classical Machine Learning algorithms. Arroyo(2022) tried to apply classical machine learning on image classification problems to explore its limits. The model with the highest accuracy, 91%, is the one with GridSearchCV and best hyper-parameters for the SVM applied. However, it’s much lower than the accuracy obtained in any deep learning related methods summarized by Mellouk and Handouzi (2020), the paper we mentioned in the related work part. 

**Citation**: [https://towardsdatascience.com/facial-expression-recognition-fer-without-artificial-neural-networks-4fa981da9724](https://towardsdatascience.com/facial-expression-recognition-fer-without-artificial-neural-networks-4fa981da9724)
 
FER-2013 is the dataset we use. The bias caused by collection and label is probably a problem we need to concern. To see if FER related datasets suffer from obvious biases caused by different cultures and collection conditions, Li and other researchers (2019) quantitatively verified that the datasets have a strong build-in bias and proposed Emotion-Conditional Adoption Network to address the issue. 

**Citation**: [https://arxiv.org/pdf/1904.11150.pdf](https://arxiv.org/pdf/1904.11150.pdf)
 
 
 
## **Division of labor** 
Preprocessing - Bohao, Ying \\
Train the model - Ying, Elbert\\
Test for the accuracy - Bohao\\
Convert the result into emojis - Elbert\\
