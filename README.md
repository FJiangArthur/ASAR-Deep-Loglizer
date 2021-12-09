
# System Anomaly Detection via Deep Log Analysis

***Members: Aparna Natarajan (anatara7) , Art Jiang (fjiang7), Rohit Mohanty (rmohant2) , Sima Arfania (sarfania)***

  

## Introduction:

  

In March of 2017, a patient’s intestines were accidentally nicked while the da Vinci robotic system was performing an inguinal hernia repair; this fatal error was not discovered until the patient had been discharged from the hospital. Over the past 14 years, the FDA received over 8,000 reports of robotic-surgery malfunctions which led to higher numbers of injuries and deaths in relatively complex procedures relating to the heart, brain, head and neck. For these reasons, adoption of advanced techniques and mechanisms for adverse event detection and reporting may reduce preventable incidents. Anomalies that can occur during surgical robot procedures include broken instruments falling into patients’ bodies, electric spark burning tissues, and system malfunctions leading to longer surgery times. As the global surgical robots market size and usage is rapidly expanding, it is important to be able to detect and prevent such anomalies. Reporting may reduce these preventable incidents in the future, and in turn, save the lives of patients.

Widely used methods employed to predict and detect system issues using logs are largely rudimentary keyword checks. Developers have to set up these checks manually themselves in logging tools. These manual checks often miss various problems log trails indicate, leaving issues unflagged and uncaught until a production issue arises. System logs are often used to reflect the runtime status of a software system, which can be useful for monitoring, administering, and troubleshooting a system; the quality of logs that systems produce prime this problem for a deep learning approach to analyzing system logs for anomalies.

The issue associated with current widely adapted manual methods of log analysis is scalability, interpretability and adaptability. Visual examination, hardware inspections, manufacturing data and calibration log inspections, and software diagnostics are usually combined with log analysis for failure analysis in most companies. Despite their insightfulness, Failure Analysis Engineers usually treat system logs as their last resort due to the interpretability issue of the logs compared to other inspection methods.

Modern systems contain individual components that are highly sophisticated, and require enormous domain specific knowledge to understand the logics that drive such sub-components. Failure at a sub-system level usually cascades into a series of un-predictable events in other sub-components and eventually results in the failure of the whole system. The closely interconnected nature between different sub-components make it hard for a single engineer to debug a system-level failure.

In addition to these complexities, increase in log volume has served as a catalyst for the development of deep learning algorithms for anomaly detection, but the field remains largely unexplored. In this project, we explore a variety of different deep learning models—including CNNs, LSTMs, GANs, and VAEs—for log anomaly detection.

  

## Methodology:

  

***Open Source Library and Toolkit:***

  

LogParser is a open source toolkit (Zhu, Logpai/logparser: A toolkit for Automated Log Parsing [ICSE'19, TDSC'18, DSN'16]), that provides specialized methods for automated conversion from raw log message into structured events. More specifically, the team parsed the raw message into Template and Structured Log by extracting variable information in each log line and matching each log line with a parsed event template.

  

Information such as "Power Module I2C Communication Error in Node Number F73H4", "At Memory Address 0xD0000H2" is parsed into "Power Module I2C Communication Error in Node Number <*>", "At Memory Address <*>" with parameter of "F73H4", and "0xD0000H2". Figure 1 on the bottom shows a general process of how the log parsing method works.

![](https://lh3.googleusercontent.com/wGWQN0BSAGcLK9O9eln7_6N8690ifnEIFZ5HLYiDr9oHDeV54nxhd3wycdh0_0vbGhn47nFKD7RoM8iAIL8UVXv63KmXmPf49w5o02ZaxJQxE6-CBtmlEvdj29HxpX38Z6-DFjgN)

*Figure 1, Steps for conversion from raw log message into structured log event,Jieming Z. (2017)*

  

Deep-Loglizer (Z. Chen, “Logpai/deep-loglizer,” GitHub) is the open-source benchmark toolkit for deep learning based methods in log analysis. The team utilized the dataset provided and followed the guideline in the Experiment Report [4] when designing the model and implementing from top to bottom.

  
  

## Metrics:

  

Accuracy of each model was evaluated by precision, recall, and F1 scores. Precision measures the percentage of anomalous log windows that are successfully identified as anomalies over all log windows that are predicted as anomalies. Recall calculates the portion of anomalies that are successfully identified over all anomalies. F1 scores are the harmonic mean of precision and recall. Therefore, in the design of our models, our ultimate goal was to maximize the F1 score. Hyperparameters, architectures, and loss functions were adjusted to reflect this purpose.

  

The following formulas mathematically describe the precision, recall, and F1 scores.:

  

![](https://lh4.googleusercontent.com/_JSZ6a47Nvsh68KFWtMgp60gTCR7YrKU6TM_nyoih1WmH_FpvRXPn3eyOOvcp8qfKdxATWjBkg-aSVVGTDg_wqiKIru5Up_crAZ4mlB_Ik6CWvendWn0WrLg5g9v0z4JKbEIknRn)

  

*TP : number of anomalies that are correctly discovered by the model
FP : number of normal log sequences that are wrongly predicted as anomalies by the model
FN : number of anomalies failed to be discovered by the model*

## Models:

**Convolutional Neural Network:**

Typically used to extract features of visual inputs, Convolutional Neural Networks (CNNs) are effective in capturing local semantic information instead of global information, preventing overfitting against training data.

  

CNN network architecture was constructed as follows:

Convolution → Batch Normalization → Convolution → Batch Normalization → Convolution → Batch Normalization → Dense with ‘ReLu’ activation → Dense Layer with ‘Softmax’ Activation

  

**Bidirectional Long Short-Term Memory:**

Bidirectional LSTM Networks (BiLSTMs) gather long term contextual information inputs in both orientations, allowing access to information both prior to and after a particular event. Due to the complexity of large scale distributed systems, errors in behavior may occur long after an error was produced; for these reasons, LSTM models are effective in deriving long-term dependencies between errors and system events.

  

LSTM network architecture was constructed as follows:

Embedding → BiLSTM → Dense with ‘Sigmoid’ activation → Dense → Dense with ‘Softmax’ activation

  

**Variational AutoEncoder:**

Variational Autoencoders (VAEs) transform system logs into vector space and can be used to eliminate issues resulting from small variations between system logs.

  

Encoder architecture was constructed as follows:

Dense with ‘ReLu’ Activation → Dense with ‘ReLu’ Activation → Dense with ‘ReLu’ Activation → Mu → Logvar

Decoder architecture was constructed as follows:

Dense with ‘ReLu’ Activation → Dense with ‘ReLu’ Activation → Dense with ‘ReLu’ Activation → Dense with ‘Sigmoid’ Activation

  

**Generative Adversarial Network:**

Due to the limitations in volume of open-source datasets, Generative Adversarial Networks (GANs) can be used for generation of abnormal log data to be used in training.

  

Discriminator architecture was constructed as follows:

Embedding → Stacked RNN → Dense → Dense

  

Generator architecture was constructed as follows:

Embedding → Stacked RNN → Dense → Dense

  

Note: While the CNNs and BiLSTMs models were adapted from previous implementations/papers, the VAE and GAN models were constructed, trained, and tested from scratch using our own implementations.

## Results:

***HDFS Dataset (Binary Classification)***

|      | F1    | Precision | Recall | Accuracy |
|------|-------|-----------|--------|----------|
| CNN  | 0.690 | 1.000     | 0.526  | 0.983    |
| GAN  | 0.678 | 0.037     | 0.542  | 0.971    |
| VAE  | 0.979 | 0.933     | 0.981  | 0.971    |
| LSTM | 0.438 | 1.000     | 0.281  | 0.971    |

***Intrepid Dataset (Binary Classification)***

|      | F1    | Precision | Recall | Accuracy |
|------|-------|-----------|--------|----------|
| CNN  | 0.579 | 0.971     | 0.413  | 0.827    |
| GAN  | 0.034 | 0.139     | 0.138  | 0.861    |
| VAE  | 0.633 | 0.404     | 0.615  | 0.861    |
| LSTM | 0.428 | 0.330     | 0.607  | 0.861    |

## Challenges

The hardest challenge in the project so far has been data-acquisition and preprocessing. While our initial plans consisted of potentially acquiring a dataset from Intuitive Surgical Robots, it does not appear that the timeline will work according to our schedule, given the amount of legal work needed to be done to give us the rights to use such data.

Although log datasets exist and are available for use, they do not contain significant amounts of data, especially of varying data, which is a challenge because one of our goals was to make a general purpose framework for anomaly detection in many different log types. We sought out a new Intrepid dataset, however the trouble with this is we needed to do additional preprocessing for the tokenization of these logs to be identical to that of the tokenization of the other logs provided by logpai’s log parsing infrastructure.

Deep learning is advantageous specifically when large quantities of data exist for processing. Another challenge is for NLP to be used, language must be vectorized through embedding layers based on an existing vocabulary, and log data contains a lot of unexpected tokens like memory addresses and code snippings. It is almost impossible to find a vocabulary that is entirely encompassing of all the tokens we can encounter in a log trace for any system.

Another challenge we dealt with was exploding gradients with VAEs. We had to toy with the loss function and the activation function of the final layer of the decoder to reach a point where there were no exploding gradients. On the intrepid dataset however, we were not able to tweak the loss function to prevent exploding gradients on multiclass classification, and after a certain number of epochs in binary classification. The results for VAE listed in the table are the stats for our model for the last epoch before our gradient explodes.

### Implementation Challenges:

Since the implementation of the Deep-Loglizer DataLoader is written in PyTorch, the team worked on implementing data preprocessing, and creating data pipelines in TensorFlow.

HDFS and BGL dataset contains only binary labels (Normal/Abnormal) and since the goal of this project is to generalize the framework and be able to process multi-class labels, the team decided to use RAS (Reliability,Availability, and Serviceability) logs from a high end computing system, Intrepid Blue Gene, at Argonne NationalLaboratory, provided by Zheng Et al[3]. The dataset contains 15 unique log entries such as Processor Information, Node Information, Block Number, Physical Location, Error Code, Flags, Component, Message, with a time span of six months. The decompressed RAS log has several encoding issues and the team has to write specific parser scripts to manually convert the log into a processable log.

## Reflection:

We started out with an ambition to develop a universal log analyzer but it turned out to be quite the challenge. As expected, the collection and preprocessing of data turned out to be the biggest challenge for us. In order to work our way around the lack of uniformity in the data, we tried to use a GAN model to generate a more uniform data. But finally, we decided to go with the HDFS and the Intrepid Blue Gene RAS logs. We implemented two models from scratch, the GAN model and the VAE model, and ran them on these datasets along with a couple of other existing models. We were extremely delighted to see that our VAE model was able to achieve the best metrics on these datasets on the binary classification problem.

One of our biggest takeaways from this project is that we learned about the challenges in data collection and preprocessing. We learned about handling various encoding issues in the data and cleaning it so that we could generate a log file that could be easily processed by our model.

If we had more time, we would have tried to extend our model to the multiclass classification problem. We would have tried to gather more uniform data for multiclass classification and worked on improving the metrics of our model on that data so that we could have had a much more generalized version of our model. We would have also tried to work on a dashboard that would have displayed the metrics in a graphical way so that the users would have been able to easily compare the performance of different models.

## References:

[1] Xu Zhang, Yong Xu, Qingwei Lin, Bo Qiao, Hongyu Zhang, Yingnong Dang, Chunyu Xie, Xinsheng Yang, Qian Cheng, Ze Li, et al. 2019. Robust log-based anomaly detection on unstable log data. In Proceedings of the 2019 27th ACM Joint Meeting on European Software

Engineering Conference and Symposium on the Foundations of Software Engineering. 807–817

  

[2] Weibin Meng, Ying Liu, Yichen Zhu, Shenglin Zhang, Dan Pei, Yuqing Liu, Yihao Chen, Ruizhi Zhang, Shimin Tao, Pei Sun, et al. 2019. LogAnomaly: Unsupervised Detection of Sequential and Quantitative Anomalies in Unstructured Logs.. In IJCAI, Vol. 7. 4739–4745.

  

[3] Z. Zheng, L. Yu, W. Tang, Z. Lan, R. Gupta, N. Desai, S. Coghlan, and D. Buettner, ''Co-Analysis of RAS Log and Job Log on Blue Gene/P,'' in Proc. of IEEE International Parallel & Distributed Processing Symposium (IPDPS'11), Anchorage, AK, USA, 2011.

  

[4] Zhuangbin Chen, Jinyang Liu, Wenwei Gu, Yuxin Su, Michael R. Lyu: “Experience Report: Deep Learning-based System Log Analysis for Anomaly Detection”, 2021; arXiv:2107.05908.

  

[5] Jieming Zhu, Shilin He, Jinyang Liu, Pinjia He, Qi Xie, Zibin Zheng, Michael R. Lyu. Tools and Benchmarks for Automated Log Parsing. International Conference on Software Engineering (ICSE), 2019.

  

[6] Pinjia He, Jieming Zhu, Shilin He, Jian Li, Michael R. Lyu. An Evaluation Study on Log Parsing and Its Use in Log Mining. IEEE/IFIP International Conference on Dependable Systems and Networks (DSN), 2016.
