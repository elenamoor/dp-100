# Module 03

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

### Experiment Run Context: Related URLs:
1. [Core Package](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core?view=azure-ml-py)

2. [Experiment class with the list of all its methods](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.experiment(class)?view=azure-ml-py)


## Logging metrics and creating outputs

*#Import Experiment and panda library, giving this library a nickname for shortness.*  *#Define experiment variable and started logging similarly how it was done in Experiment Run Context part of this document; .*

```python
from azureml.core import Experiment 
import panda as pd

experiment = Experiment(workspace = ws, name = 'my-experiment')
run = experiment.start_logging()
```

*read 'data.csv' to the **data** variable.*

*results of the len() function applied to the data variable are written to the **row_count** variable*

*the sample of the data (1st 100 rows) will be written to the sample.csv file. Filename, indexing and header are parameters*

*upload_file method will be uploaded to the outputs folder. Method parameters are filename and the file path (path_or_stream)*

```python
data = pd.read_csv('data.csv')
row_count = (len(data))
run.log('observations', row_count)
data.sample(100).to_csv('sample.csv', index=False, header=True)
run.upload_file(name = 'outputs/sample.csv', path_or_stream = './sample.csv')
run.complete()
```

## Get the run context in the experiment script:
*To see the details of the experiment we will need to import **RunDetails** class and use its method **show()***

```python
from azureml.widgets import RunDetails

RunDetails(run).show()
```

### Related URLs
1. [RunDetails class](https://docs.microsoft.com/en-us/python/api/azureml-widgets/azureml.widgets.rundetails?view=azure-ml-py)


## Running a Script as an Experiment: Get the run context in the experiment script:
*to see the run context of the experiment we will need to import **Run** class and log the required metrics*
```python
from azureml.core import Run

run = Run.get_context()
run.log(…)
run.complete()
```

### Related URLs
1. [Run Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run.run?view=azure-ml-py)

## Running a Script as an Experiment: Define the script run configuration

```python
from azureml.core import Environment, Experiment, ScriptRunConfig

src = ScriptRunConfig(source_directory='experiment_files',
                      script='experiment.py')

experiment = Experiment(workspace = ws, name = experiment_name)
run = experiment.submit(config=src)
```

### Related URLs:
1. [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)


## Registering a Model

```python
run.register_model( model_name='classification_model',
                    model_path='outputs/model.pkl',
                    description='A classification model',
                    tags={'dept': 'sales'},
                    model_framework=Model.Framework.SCIKITLEARN,
                    model_framework_version='0.19.1'))
```

## Model import:

*To import a model, the class **Model** should be imported to the environment. To get a list of all objects of the class **Model** in the workspace **ws** that were created, use method **list**. To see the list, it is necessary to print out required information for each model. List all object attributes that are desired to be printed.*

```python
from azureml.core import Model

for model in Model.list(ws):
    print(model.name, 'version:', model.version)
```

### Related URLs:

1. [Model Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.model.model?view=azure-ml-py)



# Module 05

*This module's logic is same, as Mod.3, please use this explanation as a guide for the related URLs to the class documentation*

## Create environment from specification file

```python
env = Environment.from_conda_specification(name='training_environment',
                                          file_path='./conda.yml')
```

## Create environment from existing environment

```python
env = Environment.from_existing_conda_environment(name='training_environment',    
						    conda_environment_name='py36')
```

## Create environment with specified packages

```python
env = Environment('training_environment')
deps = CondaDependencies.create(conda_packages=['scikit-learn','pandas'],              
                                pip_packages=['azureml-sdk']
env.python.conda_dependencies = deps
```

# Register an environment in the workspace

```python
env.register(workspace=ws)
```

## View Registered Environments

```python
env_names = Environment.list(workspace=ws)
for env_name in env_names:
    print('Name:',env_name)
```

## Retrieve and use an environment

```python
training_env = Environment.get(workspace=ws, name='training_environment’)
script_config = ScriptRunConfig(source_directory='experiment_folder',
			entry_script='training_script.py',
			compute_target='local',
			environment_definition=training_env)
```

## Creating Compute Targets (Using SDK)

```python
from azureml.core.compute import ComputeTarget, AmlCompute

compute_name = 'aml-cluster'
compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS12_V2', 
                                                       max_nodes=4)
aml_compute = ComputeTarget.create(ws, compute_name, compute_config)
aml_compute.wait_for_completion(show_output=True)
```

## Using Compute Target (SDK)

```python
Script_config = ScriptRunConfig(source_directory='experiment_folder',
                      entry_script='training_script.py',
                      environment_definition=training_env,
                      compute_target='aml-cluster')
```

## Related URLs
1. [Environment class documentation](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.environment.environment?view=azure-ml-py)
2. [ScriptRunConfig](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
3. [ComputeTarget class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.computetarget?view=azure-ml-py)
