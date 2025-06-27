# 🔎 Attentiveness
This github repo has some of the data and scripts used for testing the Attentiveness Metric for my project in 'Advanced Topics for Computer Science'.

## What is the Attentiveness Metric?
The Attentiveness metric is a custom metric that identifies whether a model can summarise a text without missing a key detail that is mentioned once. The metric utilises the [G-Eval framework](https://deepeval.com/docs/metrics-llm-evals) using the DeepEval library.

## How does it work?
The metric uses the following steps.
1. The model is asked to summarise a text which has a key detail mentioned once.
2. The metric then evaluates with the criteria: “Does the ‘actual output’ provide the small detail within ‘expected output’?”.
3. A score is outputted by the metric between 0-1.

## Scoring Criteria
|Score|Criteria|
|---|--|
|0.0| The key detail is not mentioned.|
|0.1-0.5| Some of the key detail is mentioned.|
|0.6-0.9| Most of the key detail is mentioned.|
|1.0| The key detail is completely mentioned.|

## How to Run 
For this investigation, the models run locally. A sandbox such as Amazon Sagemaker Instance would be necessary to run the following code.
### Set up a Python Environment 
Go to your root dictionary and set up a python environment. In your CLI run:
```
python3 -m venv venv
source venv/bin/activate
```


### Dataset - data.csv
The dataset was fully generated using ChatGPT-4o and contains three fields: the first is  an id, the second is a passage containing a key detail, and the third is a sentence that describes or summarizes that detail.

### Attentiveness.py
A script that runs the Attentiveness metric and saves the data to a csv file.  
Run by:
```
python3 attentiveness.py
```

### Attentiveness_mean_sd.py
Provides a summary of the mean and standard deviation from attentiveness_mean_sd.py.  
Run by:
```
python3 attentiveness_mean_sd.py
```

### Attentiveness_boxplot.py
Provides a boxplot of the score ranges from attentiveness_boxplot.py.  
Run by:
```
python3 attentiveness_boxplot.py
```


