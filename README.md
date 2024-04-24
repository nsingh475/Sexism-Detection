# Sexism Detection in English and Spanish Tweets

Sexism is defined as discrimination of a person or group based on sex. In many cases, these gender stereotypes assume a difference in social standing between men and women. However, this discrimination can be expressed explicitly (discrimination that is stated plainly) or implicitly (discrimination that is implied or obfuscated). The examples below from the EXIST 2023 dataset show the distinction between explicit and implicit sexism:

Explicit: Call me sexist all you want but no Nation ever succeeds with a woman as the Head. It’s just the way it is. They final nail is already in the coffin.

Implicit: Wife material, wake up and cook for your husband

## Research Objective:
Our goal was to examine:
- The effects of data pre-processing techniques on model performance.
- The implications of utilizing word-level versus character-level models.
- The use of soft labels versus hard labels during the training process.

### Optimal Hyperparameters used for our experiments
![image](https://github.com/nsingh475/Sexism-Detection/assets/87938938/4c7502cf-abd0-4c02-b223-d4c3f67ebac0)

### Model Evaluation results
![image](https://github.com/nsingh475/Sexism-Detection/assets/87938938/85fa8284-0f27-4bb9-bab4-92e659cb8b56)

### Conclusion
Our experiment established the following results:
- Model behavior and performance is dependent on the choice of evaluation metric.
- Word-level Neural models (RNN, CNN) works better than character-level Neural models when the dataset is small/ medium in size.
- Upsampling techniques on the minority class can enhance the performance of SVM models when dealing with an imbalanced datasets.

### Publication: 
https://ceur-ws.org/Vol-3497/paper-073.pdf






