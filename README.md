# Breast_Cancer_Classification
ABSTRACT

Breast cancer is regarded as the nemesis that threatens women’s health worldwide. Early detection plays a crucial role in reducing mortality rates, but traditional diagnostic methods rely on manual interpretation of medical images, which can be time-consuming, prone to human error, and limited by the availability of expert radiologists. To overcome these challenges, this project introduces an AI-powered Diagnostic System for Breast Cancer Classification, utilizing Convolutional Neural Networks (CNNs) to enhance accuracy and efficiency. The system is trained on a Kaggle dataset containing medical images, where preprocessing techniques such as resizing, normalization, and augmentation are applied to improve model performance. The CNN model is developed and trained using Jupyter Notebook and is later integrated with a Tkinter-based graphical user interface (GUI) to provide a user-friendly platform for breast cancer diagnosis. Users can upload medical images, and the system will automatically classify them as benign or malignant or normal, delivering instant and reliable results. This project aims to streamline the diagnostic process, reduce dependency on manual assessments, and assist healthcare professionals in making faster and more accurate clinical decisions, ultimately improving patient outcomes.


INTRODUCTION

Breast cancer is one of the most prevalent and life-threatening diseases affecting women worldwide, with its incidence rising significantly due to genetic, lifestyle, and environmental factors. Early detection is crucial, as timely diagnosis increases survival rates and improves treatment effectiveness. However, traditional diagnostic methods rely heavily on manual examination of mammograms or histopathological images, which can be time-consuming, prone to human error, and dependent on the expertise of radiologists. In many cases, misdiagnosis or delayed detection leads to severe consequences, making it essential to develop automated, AI-driven solutions that can enhance accuracy and efficiency in breast cancer classification. Recent advancements in artificial intelligence (AI) and deep learning have demonstrated remarkable potential in medical diagnostics, particularly in image-based disease detection. By leveraging Convolutional Neural Networks (CNNs), AI-powered systems can analyze medical images with high precision, identifying patterns that may not be easily detectable by human experts.

This project aims to develop a Diagnostic System for Breast Cancer Classification using CNN-based deep learning models to improve the accuracy and efficiency of cancer detection. The system will be trained on a Kaggle dataset, where medical images will undergo preprocessing techniques such as resizing, normalization, and augmentation to enhance model performance. The trained CNN model will be deployed using Jupyter Notebook for training and evaluation, and an intuitive Tkinter-based GUI will be developed to allow users to upload images for classification. The system will automatically analyze the input image and classify it as benign or malignant or normal, providing immediate results to assist medical professionals in decision-making. By integrating AI-driven automation, this project aims to reduce dependency on manual assessments, minimize diagnostic errors, and accelerate the detection process, ultimately contributing to better patient care and improved healthcare outcomes.


PROBLEM STATEMENT

Breast cancer is one of the leading causes of mortality among women worldwide, and early detection plays a crucial role in improving survival rates. However, traditional diagnostic methods rely heavily on manual examination of medical images, which can be time-consuming, prone to human error, and dependent on the expertise of radiologists. These conventional techniques often lack standardized accuracy, leading to misdiagnosis or delayed detection, which can severely impact treatment outcomes. Additionally, access to specialized diagnostic services is limited in remote or underdeveloped areas, further delaying crucial medical intervention. To address these challenges, an AI-driven breast cancer classification system is needed to automate the diagnostic process, enhance accuracy, and speed up detection. By leveraging deep learning techniques, this system can overcome the limitations of traditional methods, providing fast, reliable, and cost-effective screening, ultimately assisting healthcare professionals in making informed clinical decisions.


OBJECTIVE  

1.	To enhance accuracy in medical diagnosis using AI-driven technology.
2.	To automate the classification of medical images for faster analysis.
3.	To reduce human error in disease detection and improve reliability.
4.	To provide a user-friendly interface for easy access to diagnostic results.
5.	To support early disease detection and improve healthcare outcomes.


LITERATURE REVIEW

1.	Zahra abdolali Kazemi, et al. In this study, two approaches for the presentation of mammography (comparison of previous and current mammography images) are evaluated: together (simultaneously) and alternately on the same screen. In this study, MATLAB software is used.In this study, image processing algorithms of support vector machine (SVM), genetic algorithm (GA), convolutional neural networks (CNN), and K-nearest neighbors (KNN) are exploited. In this regard, the performance of these algorithms will be explained in this section. In this method, it is first essential to conduct training. Training means that a number of features related to class one and class two are given to the function, and the algorithm updates its parameters based on the labeling done. Then, the unlabeled data are given to the algorithm for the classification, and it automatically specifies the corresponding class. Segmentation is the simplification or modification of image view for more meaningful and easier analysis. This is the process of labeling each pixel in each image, which results in a set of segments that together cover the whole image. By analyzing the resulting images, the physicians can identify cancer cells and offer their diagnostic results. It is possible to expand the MATLAB environment by adding a toolbox for various purposes. For simulation, training and classification need to be done with the classification method.

2.	Saif Ali et al.  In this paper, several different algorithms have been discussed such as SVM, KNN, DT, etc. for the classification of the different cancers. This paper also presents a comparative analysis of the research done in the past. Cancer can be classified into two main categories: malignant and benign. Early detection of cancer is the key to the successful treatment of cancer. There are various methodologies for the detection of cancer some include manual procedures, Manual identification is time-consuming and unreliable therefore computer-aided detection came into the research. Computer-aided detection involves image processing for feature extraction and classification techniques for the recognition of cancer type and stages.

3.	Zubair, M., S. Wang, et al. This review offers consolidated information on currently available BC diagnosis and treatment options. It further describes advanced biomarkers for the development of state-of-the-art early screening and diagnostic technologies. The author stated that they have seven prognostic multigene signature tests for BC providing a risk profile that can avoid unnecessary treatments in low-risk patients. Many comparative studies on multigene analysis projected the importance of integrating clinicopathological information with genomic-imprint analysis.

4.	Yousif M.Y Abdallah et al.  The image processing methods in this paper used contrast improvement, noise lessening, texture scrutiny and portioning algorithm. The mammography images were kept in high quality to conserve the quality. Those methods aim to augment and hone the image intensity and eliminate noise from the images. The assortment factor of augmentation depends on the backdrop tissues and type of the breast lesions; hence, some lesions gave better improvement than the rest due to their density. The computation speed examined used correspondence and matching ratio. The results were 96.3 ± 8.5 (p>0.05). The results showed that the breast lesions could be improved by using the proposed image improvement and segmentation methods.

5.	Prannoy Giri et al. The author stated that using CAD (Computer Aided Diagnosis) on the mammographic images is the most efficient and easiest way to diagnose breast cancer. Accurate discovery can effectively reduce the mortality rate brought about by using mamma cancer. Masses and microcalcifications clusters are important early symptoms of possible breast cancers. They can help predict breast cancer in its infant state. The image for this work is being used from the DDSM Database (Digital Database for Screening Mammography) which contains approximately 3000 cases and is being used worldwide for cancer research. This paper quantitatively depicts the analysis methods used for texture features for the detection of cancer. These texture features are extracted from the ROI of the mammogram to characterize the microcalcifications as harmless, ordinary, or threatening. These features are further decreased using Principal Component Analysis(PCA) for better identification of Masses. These features are further compared and passed through the Back Propagation algorithm (Neural Network) for a better understanding of the cancer pattern in the mammography image.

6.	Arpita Joshi and Dr. Ashish Mehta compared the classification results obtained from the techniques i.e. KNN, SVM, Random Forest, Decision Tree (Recursive Partitioning and Conditional Inference Tree). The dataset used was Wisconsin Breast Cancer dataset obtained from UCI repository. Simulation results showed that KNN was thebest classifier followed by SVM, Random Forest and Decision Tree.

7.	David A. Omondiagbe, Shanmugam Veeramani, Amandeep S. Sidhu investigated the performance of Support Vector Machine, Artificial Neural Network and Naïve Bayes using the Wisconsin Diagnostic Breast Cancer (WDBC) Dataset by integrating these machine learning techniques with feature selection/feature extraction methods to obtain the most suitable one. The simulation results showed that SVM-LDA was chosen over all the other methods due to their longer computational time.

8.	Kalyani Wadkar, Prashant Pathak and Nikhil Wagh, et al. did a comparative study on ANN and SVM and integrated various classifiers like CNN, KNN and Inception V3 for better processing of the dataset. The experimental results and performance analysis concluded that ANN was a better classifier than SVM as ANN proved to have a higher efficiency rate.

9.	Anji Reddy Vaka, Badal Soni and Sudheer Reddy K., et al. presented a novel method to detect breast cancer by employing techniques of Machine Learning such as Naïve Bayes classifier, SVM classifier, Bi-clustering Ada Boost techniques, RCNN classifier and Bidirectional Recurrent Neural Networks (HA-BiRNN). A comparative analysis was done between the Machine learning techniques and the proposed methodology (Deep Neural Network with Support Value) and the simulated results concluded that the DNN algorithm was advantageous in both performance, efficiency and quality of images are crucial in the latest medical systems whilst the other techniques didn’t perform as expected.

10.	 Monica Tiwari, Rashi Bharuka, Praditi Shah and Reena Lokare, et al. presented a novel method to detect breast cancer by employing techniques of Machine Learning that is Logistic Regression, Random Forest, K-Nearest Neighbor, Decision tree, Support Vector Machine and Naïve Bayes Classifier and techniques of Deep Learning that is Artificial Neural Network, Convolutional Neural Network and Recurrent Neural Network. The comparative analysis between the Machine Learning and Deep learning techniques concluded that the accuracy obtained in the case of CNN model (97.3%) and ANN model (99.3%) was more efficient than the Machine Learning models.


RESEARCH METHODOLOGY

Proposed System

The proposed system aims to develop an efficient diagnostic system for breast cancer classification using deep learning techniques. The system begins by collecting a labeled dataset from Kaggle, which includes medical imaging or histopathological data related to breast cancer. The dataset undergoes preprocessing steps such as resizing, normalization, and augmentation to enhance model performance. A Convolutional Neural Network (CNN) is then trained on Jupyter Notebook, leveraging its powerful computational capabilities for feature extraction and classification. The trained model is integrated into a user-friendly interface developed using Tkinter, allowing users to upload medical images for analysis. Upon image submission, the system processes the input and classifies it as malignant or benign or normal, providing accurate diagnostic results. This project aims to assist medical professionals in early breast cancer detection, improving diagnostic accuracy and patient outcomes.


SYSTEM REQUIREMENT

SOFTWARE REQUIREMENT

➢	Python Software IDE

LIBRARIES USED

➢	Tkinter 
➢	Flask==0.12.3
➢	OpenCV 
➢	NumPy
➢	TensorFlow 


ADVANTAGE

1.	Improves diagnostic accuracy by using deep learning for breast cancer classification.
2.	Enables early detection, leading to timely treatment and better patient outcomes.
3.	Supports remote accessibility, allowing integration with cloud-based healthcare systems.
4.	Facilitates cost-effective screening, reducing the need for expensive manual tests.


CONCLUSION

The Diagnostic System for Breast Cancer Classification provides a reliable and efficient approach to detecting breast cancer using deep learning. The implementation of a CNN model enhances accuracy in distinguishing between malignant and benign cases, reducing the chances of misdiagnosis. By automating the diagnostic process, the system minimizes the dependency on manual evaluations, making early detection more accessible and precise. The use of artificial intelligence in medical imaging helps in identifying patterns that may not be easily detected by the human eye, thereby improving clinical decision-making. Additionally, the integration of a user-friendly Tkinter interface ensures ease of access for medical professionals and researchers, allowing them to analyze images with minimal effort. This system has the potential to assist in early diagnosis, timely intervention, and better treatment planning, ultimately increasing survival rates among breast cancer patients. 


REFERENCES

[1] Zahra abdolali kazemi, “Diagnosis of breast cancer using image processing with SVM and KNN” International Journal of Engineering and Technology, Vol 13 No 1 Mar-Apr 2021. 

[2] Ali, Saif, Aneeqa Tanveer, Azhar Hussain, and Saif Ur Rehman. "Identification of cancer disease using image processing approahes." International Journal of Intelligent Information Systems 9, no. 2 (2020): 6-15.

[3] Zubair, M., S. Wang, and N. Ali. "Advanced approaches to breast cancer classification and diagnosis." Frontiers in Pharmacology 11 (2021): 632079.

[4] Abdallah, Yousif MY, Sami Elgak, Hosam Zain, Mohammed Rafiq, Elabbas A. Ebaid, and Alaeldein A. Elnaema. "Breast cancer detection using image enhancement and segmentation algorithms." Biomedical Research 29, no. 20 (2021): 3732-3736.

[5] Giri, Prannoy, and K. Saravanakumar. "Breast cancer detection using image processing techniques." Oriental journal of computer science and technology 10, no. 2 (2022): 391-399.

[6] Arpita Joshi and Dr. Ashish Mehta “Comparative Analysis of Various Machine Learning Techniques for Diagnosis of Breast Cancer” IRJET, (2020). 

[7] David A. Omondiagbe, Shanmugam Veeramani and Amandeep S. Sidhu “Machine Learning Classification Techniques for Breast Cancer Diagnosis” IRJET, (2021). 

[8] Kalyani Wadkar, Prashant Pathak and Nikhil Wagh “Breast Cancer Detection Using ANN Network and Performance Analysis with SVM” International Journal of Computer Engineering and Technology. (2020). 

[9] Anji Reddy Vaka, Badal Soni and Sudheer Reddy “Breast Cancer Detection by Leveraging Machine Learning” (2020). Ict Express 6, no. 4 (2020): 320-324.

[10] Tiwari, Monika, Rashi Bharuka, Praditi Shah, and Reena Lokare. "Breast cancer prediction using deep learning and machine learning techniques." Available at SSRN 3558786 (2020).

