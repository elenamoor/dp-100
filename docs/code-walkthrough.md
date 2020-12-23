# Module 03 - Experiments, Training and Registering Models

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

# Module 4 - Datastores, Datasets
## Datastores
*Datastores enable you to connect to your data on Azure storage services*

### To use a Datastore object, you must first register it
*Import the Datastore class, and use one of the many register methods*
```python
from azureml.core import Workspace, Datastore
ws = Workspace.from_config()
# Register a new datastore
blob_ds = Datastore.register_azure_blob_container(workspace=ws, datastore_name='blob_data', container_name='data_container',                                                 account_name='az_store_acct', account_key='123456abcde789…')
```
### Work with a datastore directly to upload and download data
```python
blob_ds.upload(src_dir='/files', target_path='/data/files', overwrite=True, show_progress=True)
blob_ds.download(target_path='downloads', prefix='/data', show_progress=True)
```
### Get datastores from your workspace
```python
# Get a named datastore from the current workspace
datastore = Datastore.get(ws, datastore_name='your datastore name')
```
### Related URLs:
1. [Connect to data in storage services on Azure](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-connect-data-ui)
2. [Datastore class documentation](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.datastore.datastore?view=azure-ml-py)

## Datasets
*Use datasets to access data for your local or remote experiments. Creating a dataset creates a reference to the data source location, along with a copy of its metadata*

### Create and register a tabular Dataset object
*Be sure to import the Dataset class first.*
```python
from azureml.core import Dataset
csv_paths = [(blob_ds, 'data/files/current_data.csv'),(blob_ds, 'data/files/archive/*.csv')]
tab_ds = Dataset.Tabular.from_delimited_files(path=csv_paths)
tab_ds = tab_ds.register(workspace=ws, name='csv_table')
```
### Retrieve registered dataset
```python
from azureml.core import Dataset
csv_ds = ws.datasets['csv_table'] # Using workspace datasets attribute
```
### Create and register a file dataset
```python
from azureml.core import Dataset
file_ds = Dataset.File.from_files(path=(blob_ds, 'data/files/images/*.jpg'))
file_ds = file_ds.register(workspace=ws, name='img_files')
img_ds = Dataset.get_by_name(ws, 'img_files') # using Dataset get_by_name method
```
### Retrieve registered dataset
```python
from azureml.core import Dataset
img_ds = Dataset.get_by_name(ws, 'img_files') # using Dataset get_by_name method
```
### Read data from a tabular dataset
```python
df = tab_ds.to_pandas_dataframe()
for file_path in file_ds.to_path():
    print(file_path)
``` 
### Create a new version of an existing dataset
```python
# add .png files to dataset definition
img_paths = [(blob_ds, 'data/files/images/*.jpg'),(blob_ds, 'data/files/images/*.png')]
file_ds = Dataset.File.from_files(path=img_paths)
file_ds = file_ds.register(workspace=ws, name='img_files', create_new_version=True)
``` 
# Specify a version to retrieve
```python
ds = Dataset.get_by_name(workspace=ws, name='img_files', version=2)
```
### Related URLs:
1. [Create Azure Machine Learning datasets to access data](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-register-datasets)
2. [Dataset class documentation](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.dataset(class)?view=azure-ml-py)
3. [Practice using datasets with Azure ML Notebooks](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/work-with-data/)

# Module 05 - Environments, Compute Targets

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

# Module 6 - Pipelines, Publishing and Running Pipelines
## Pipelines
*Build, optimize, and manage Azure ML workflows with Pipelines. Pipelines connect listed steps together.*

### Pipelines are built from pre-configured pipeline steps that cover many common scenarios
```python
from azureml.pipeline.core import Pipeline
step1 = PythonScriptStep(name='prepare data', ...) 
step2 = DatabricksStep(name='add notebook', ...) 
```
### Create a Pipeline object, constructor requires a workspace and a set of steps as arguments
```python
from azureml.pipeline.core import Pipeline
training_pipeline = Pipeline(workspace=ws, steps=[step1,step2]
```
### Send data between pipeline steps by using a Pipeline Data object
*Use the output of one step as the input for the next*
```python
#as_dataset is called here and is passed to both the output and input of the next step. 
from azureml.pipeline.core import PipelineData
pipeline_data = PipelineData('output').as_dataset() 
step1 = PythonScriptStep(..., outputs=[pipeline_data]) 
step2 = PythonScriptStep(..., inputs=[pipeline_data]) 
```
### Pipeline Step Reuse
*Save time by reusing output without re-running the step*
```python
step1 = PythonScriptStep(name='prepare data', arguments = ['--folder', prepped], outputs=[prepped], allow_reuse=True, ...) 
```
### Force all steps to re-run to ensure that you are always using the freshest data
```python
pipeline_run = experiment.submit(pipeline_experiment, regenerate_outputs=True) 
```
### Related URLs
1. [Pipeline class documentation](https://docs.microsoft.com/en-us/python/api/azureml-pipeline-core/azureml.pipeline.core.pipeline.pipeline)
2. [Pipeline steps class documentation](https://docs.microsoft.com/en-us/python/api/azureml-pipeline-steps/azureml.pipeline.steps?view=azure-ml-py)
3. [Pipeline data class documentation](https://docs.microsoft.com/en-us/python/api/azureml-pipeline-core/azureml.pipeline.core.pipelinedata?view=azure-ml-py)
4. [Azure Machine Learning Pipelines](https://docs.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines)
5. [Azure ML Notebooks on GitHub](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/machine-learning-pipelines) 

## Pipeline Endpoints
*A PipelineEndpoint object defines a pipeline workflow that can be triggered from a unique endpoint URL*

### Publish a pipeline to create a REST endpoint 
```python
published_pipeline = pipeline_run.publish(name='training_pipeline', description='Model training pipeline',version='1.0') 
```
*Post a JSON request to initiate a pipeline requires an authorization header and returns a run ID*
```python
import requests 

response = requests.post(rest_endpoint, headers=auth_header, json={"ExperimentName": "run training pipeline"}) 

run_id = response.json()["Id"] 
```
## Pipeline Parameters 
*PipelineParameter class object defines a parameter in a pipeline execution* 

## Parameterize a pipeline before publishing 
```python
reg_param = PipelineParameter(name='reg_rate', default_value=0.01) 
... 
step2 = EstimatorStep(name='train model', estimator_entry_script_arguments=['—reg', reg_param], ...) 
... 
published_pipeline = pipeline_run.publish(name='model training pipeline', description='trains a model with reg parameter', version='2.0') 
```
## Pass parameters in the JSON request 

```python
response = requests.post(r
est_endpoint, headers=auth_header, json={"ExperimentName": "run training pipeline", 

"ParameterAssignments": {"reg_rate": 0.1}}) 
```
## Scheduling Pipelines 
*Schedule class – Defines a schedule on which to submit a pipeline:* [Schedule class](https://docs.microsoft.com/en-us/python/api/azureml-pipeline-core/azureml.pipeline.core.schedule(class)?view=azure-ml-py)
### Schedule pipeline runs based on time 
```python
daily = ScheduleRecurrence(frequency='Day', interval=1) 
pipeline_schedule = Schedule.create(ws, name='Daily Training', description='trains model every day',                                 pipeline_id=published_pipeline_id, experiment_name='Training_Pipeline', recurrence=daily) 
```
### Trigger pipeline runs when data changes 
```python
training_datastore = Datastore(workspace=ws, name='blob_data') 
pipeline_schedule = Schedule.create(ws, name='Reactive Training', description='trains model on data change', pipeline_id=published_pipeline_id,experiment_name='Training_Pipeline', datastore=training_datastore, path_on_datastore='data/training') 
```
### Related URLs
1. [What are Azure Machine Learning Pipelines](https://docs.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines#:~:text=An%20Azure%20Machine%20Learning%20pipeline%20is%20an%20independently,Python%20script%2C%20so%20may%20do%20just%20about%20anything.)
2. [PipelineEndpoint class documentation](https://docs.microsoft.com/en-us/python/api/azureml-pipeline-core/azureml.pipeline.core.pipeline_endpoint.pipelineendpoint?view=azure-ml-py)
3. [PipelineParameter class documentation](https://docs.microsoft.com/en-us/python/api/azureml-pipeline-core/azureml.pipeline.core.graph.pipelineparameter?view=azure-ml-py)
