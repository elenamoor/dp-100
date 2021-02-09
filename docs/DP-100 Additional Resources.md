# Module 01 - Getting Started with Azure Machine Learning

1. 


# Module 02 - No-Code Machine Learning
## Articles mentioned

## Course - related documentation

## Related Learning Paths:

# Module 03 - Running Experiments and Training Models

## Articles mentioned

## Course - related documentation

## Related Learning Paths:


# Module 04 - Working with Data

## Articles mentioned

## Course - related documentation

## Related Learning Paths:

# Module 05 - Working with Compute

## Articles mentioned

## Course - related documentation

## Related Learning Paths:

# Module 06 - Orchestrating Machine Learning Workflows
# Module 07 - Deploying and Consuming Models
# Module 08 - Training Optimal Models
# Module 09 - Responsible Machine Learning
# Module 10 - Monitoring Models



## Experiment Run Context
*import class **Experiment** from the **Core** azureml package*

```python
from azureml.core import Experiment 
```

*create and define an **experiment** variable. It will create a new object of the class  **Experiment** that was imported earlier. This object will be created within the workspace **"ws"** and it will be called **"my_experiment"**. Both workspace and experiment name are passed as parameters*

```python
experiment = Experiment(workspace = ws, name = "my-experiment")
```

*start the defined **experiment**, using **start_logging** method of the experiment object we have created earlier*
 
```python
run = experiment.start_logging()
```

*the whole **experiment** steps description should be between **run = experiment.start_logging()** and **run.complete()***

*end the experiment*

```python
run.complete()
```

### Experiment Run Context - Related URLs:
1. [Core Package](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core?view=azure-ml-py)

2. [Experiment class with the list of all its methods](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.experiment(class)?view=azure-ml-py)