# ASAR-Deep-loglizer
#### Aparna Natarajan, Art Jiang,  Rohit Mohanty, Sima Arfania 

#### Introduction:

In March of 2017, a patient’s intestines were accidentally nicked while the da Vinci robotic system was performing an inguinal hernia repair; this fatal error was not discovered until the patient had been discharged from the hospital. Over the past 14 years, the FDA received over 8,000 reports of robotic-surgery malfunctions which led to higher numbers of injuries and deaths in relatively complex procedures relating to the heart, brain, head and neck. For these reasons, adoption of advanced techniques and mechanisms for adverse event detection and reporting may reduce preventable incidents. Anomalies that can occur during surgical robot procedures include broken instruments falling into patients’ bodies, electric spark burning tissues, and system malfunctions leading to longer surgery times. As the global surgical robots market size and usage is rapidly expanding, it is important to be able to detect and prevent such anomalies. Reporting may reduce these preventable incidents in the future, and in turn, save the lives of patients.
Widely used methods employed to predict and detect system issues using logs are largely rudimentary keyword checks. 

Developers have to set up these checks manually themselves in logging tools. These manual checks often miss various problems log trails indicate, leaving issues unflagged and uncaught until a production issue arises. System logs are often used to reflect the runtime status of a software system, which can be useful for monitoring, administering, and troubleshooting a system; the quality of logs that systems produce prime this problem for a deep learning approach to analyzing system logs for anomalies.
The issue associated with current widely adapted manual methods of log analysis is scalability, interpretability and adaptability. Visual examination, hardware inspections, manufacturing data and calibration log inspections, and software diagnostics are usually combined with log analysis for failure analysis in most companies. Despite their insightfulness, Failure Analysis Engineers usually treat system logs as their last resort due to the interpretability issue of the logs compared to other inspection methods. 
Modern systems contain individual components that are highly sophisticated, and require enormous domain specific knowledge to understand the logics that drive such sub-components. 


Failure at a sub-system level usually cascades into a series of un-predictable events in other sub-components and eventually results in the failure of the whole system. The closely interconnected nature between different sub-components make it hard for a single engineer to debug a system-level failure. 
In addition to these complexities, increase in log volume has served as a catalyst for the development of deep learning algorithms for anomaly detection, but the field remains largely unexplored. In this project, we explore a variety of different deep learning models—including CNNs, LSTMs, GANs, and VAEs—for log anomaly detection.

#### Methodology:

##### Open Source Library and Toolkit: 

LogParser is a open source toolkit (Zhu, Logpai/logparser: A toolkit for Automated Log Parsing [ICSE'19, TDSC'18, DSN'16]), that provides specialized methods for automated conversion from raw log message into structured events. More specifically, the team parsed the raw message into Template and Structured Log by extracting variable information in each log line and matching each log line with a parsed event template.  

Information such as  "Power Module I2C Communication Error in Node Number F73H4", "At Memory Address 0xD0000H2" is parsed into "Power Module I2C Communication Error in Node Number <*>", "At Memory Address <*>" with parameter of "F73H4", and "0xD0000H2".  

##### Models 

###### Convolutional Neural Network: 
Typically used to extract features of visual inputs, Convolutional Neural Networks (CNNs) are effective in capturing local semantic information instead of global information, preventing overfitting against training data. 

CNN network architecture was constructed as follows: 
Convolution → Batch Normalization → Convolution → Batch Normalization → Convolution → Batch Normalization → Dense with ‘ReLu’ activation → Dense Layer with ‘Softmax’ Activation

###### Bidirectional Long Short-Term Memory: 
Bidirectional LSTM Networks (BiLSTMs)  gather long term contextual information inputs in both orientations, allowing access to information both prior to and after a particular event. Due to the complexity of large scale distributed systems, errors in behavior may occur long after an error was produced; for these reasons, LSTM models are effective in deriving long-term dependencies between errors and system events. 

LSTM network architecture was constructed as follows: 
Embedding → BiLSTM  → Dense with ‘Sigmoid’ activation → Dense → Dense with ‘Softmax’ activation

###### Variational AutoEncoder: 
     Variational Autoencoders (VAEs) transform system logs into vector space and can be used to eliminate issues resulting from small variations between system logs. 

Encoder architecture was constructed as follows: 
Dense with ‘ReLu’ Activation → Dense with ‘ReLu’ Activation → Dense with ‘ReLu’ Activation → Mu → Logvar
Decoder architecture was constructed as follows: 
Dense with ‘ReLu’ Activation → Dense with ‘ReLu’ Activation → Dense with ‘ReLu’ Activation → Dense with ‘Sigmoid’ Activation

###### Generative Adversarial Network:
Due to the limitations in volume of open-source datasets, Generative Adversarial Networks (GANs) can be used for generation of abnormal log data to be used in training. 

Discriminator architecture was constructed as follows: 
Embedding → Stacked RNN → Dense → Dense

Generator architecture was constructed as follows:
Embedding → Stacked RNN → Dense → Dense


### References

- [Zhuangbin Chen](http://www.cse.cuhk.edu.hk/~zbchen), The Chinese University of Hong Kong
- [Jinyang Liu](http://www.cse.cuhk.edu.hk/~jyliu), The Chinese University of Hong Kong
- Wenwei Gu, The Chinese University of Hong Kong

### Log Preprocess

The best results are achieved by saving Message only.

#### For Blue Gene Intrepid RAS Dataset :

```python:
log_foramt = <LineID>,<BLOCK>,<COMPONENT>,<ERRCODE>,<EVENT_TIME>,<LOCATION>,<MESSAGE>,<MSG_ID>,<PROCESSOR>,<RECID>,<SEVERITY>,<SUBCOMPONENT>'
```
#### For Trinity Log Dataset: 

```python:
log_foramt = <user_ID>,<group_ID>,<submit_time>,<start_time>,<dispatch_time>,<queue_time>,<end_time>,<wallclock_limit>,<job_status>,<node_count>,<tasks_requested>
```
***
## Acknowledgement
***
####1. RAS processing 
This part of the code, `DatasetSpecific/ras_parse.py` is developed by Github User [JarvisXX](https://github.com/JarvisXX/). 

And ASAR Team uses this code to clean up the Intrepid RAS Log Dataset.

#####Title:  Parser for Intrepid RAS log Dataset
    
#####Author:Xingyi Wang, [Email](arvis_wxy@sjtu.edu.cn)
#####Date: 2018
#####Availability: [Github](https://github.com/JarvisXX/Parser-N-Analyzer-for-Intrepid-RAS-log-Dataset)
***

####2.logparser 
logparser is a opensource package developed by [LOGPAI](https://github.com/logpai)

#####Title:  LogParser
    
#####Author:LogPai
#####Date: 2018
#####Availability: [Github](https://github.com/logpai/logparser)

