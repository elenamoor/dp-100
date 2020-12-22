Module 4
Datastores – Connect to your data to Azure storage services on Azure
Use the Datastore object in the SDK – To work with a datastore, you must first register it
Please see Microsoft Docs reference to Connect to data in storage services on Azure - Azure Machine Learning:  https://docs.microsoft.com/en-us/azure/machine-learning/how-to-connect-data-ui
Also, see Azure ML reference on Datastore class:
https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.datastore.datastore?view=azure-ml-py
from azureml.core import Workspace, Datastore
ws = Workspace.from_config()
# Register a new datastore
blob_ds = Datastore.register_azure_blob_container(workspace=ws, datastore_name='blob_data', container_name='data_container',                                                 account_name='az_store_acct',                                                 account_key='123456abcde789…')
Work with a datastore directly to upload and download data:
blob_ds.upload(src_dir='/files', target_path='/data/files', overwrite=True, show_progress=True)
blob_ds.download(target_path='downloads', prefix='/data', show_progress=True)
Get datastores from your workspace
# Get a named datastore from the current workspace
datastore = Datastore.get(ws, datastore_name='your datastore name')
Datasets -  Use to access data for your local or remote experiments; creates a reference to the data source location, along with a copy of its metadata
Please see Microsoft Docs reference to Create Azure Machine Learning datasets to access data - Azure Machine Learning:  https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-register-datasets
Also, please see Azure ML reference on Dataset class:
https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.dataset(class)?view=azure-ml-py
Use the Dataset object in the SDK
from azureml.core import Dataset
# Create and register a tabular dataset
csv_paths = [(blob_ds, 'data/files/current_data.csv'),(blob_ds, 'data/files/archive/*.csv')]
tab_ds = Dataset.Tabular.from_delimited_files(path=csv_paths)
tab_ds = tab_ds.register(workspace=ws, name='csv_table')
# Retrieve registered dataset
csv_ds = ws.datasets['csv_table'] # Using workspace datasets attribute
from azureml.core import Dataset
# Create and register a file dataset
file_ds = Dataset.File.from_files(path=(blob_ds, 'data/files/images/*.jpg'))
file_ds = file_ds.register(workspace=ws, name='img_files')
# Retrieve registered dataset
img_ds = Dataset.get_by_name(ws, 'img_files') # using Dataset get_by_name method
Reading data from a dataset
df = tab_ds.to_pandas_dataframe()
for file_path in file_ds.to_path():
    print(file_path)
Get practice using datasets with Azure ML Notebooks:  MachineLearningNotebooks/how-to-use-azureml/work-with-data at master · Azure/MachineLearningNotebooks · GitHub
Create a new version of an existing dataset
# add .png files to dataset definition
img_paths = [(blob_ds, 'data/files/images/*.jpg'),(blob_ds, 'data/files/images/*.png')]
file_ds = Dataset.File.from_files(path=img_paths)
file_ds = file_ds.register(workspace=ws, name='img_files', create_new_version=True)
Specify a version to retrieve
ds = Dataset.get_by_name(workspace=ws, name='img_files', version=2)

# Module 05

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

Module 6
Pipelines – Build, optimize, and manage Azure ML workflows
Please see Microsoft Docs reference to What are Azure Machine Learning Pipelines - Azure Machine Learning:  https://docs.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines
Azure ML Notebooks on GitHub: MachineLearningNotebooks/how-to-use-azureml/machine-learning-pipelines at master · Azure/MachineLearningNotebooks · GitHub
Pipeline steps class – These are pre-built steps for common scenarios in ML workflows:
azureml.pipeline.steps package - Azure Machine Learning Python | Microsoft Docs
step1 = PythonScriptStep(name='prepare data', ...)
step2 = DatabricksStep(name='add notebook', ...)
Pipeline class – Pipelines connect the listed steps together:
https://docs.microsoft.com/en-us/python/api/azureml-pipeline-core/azureml.pipeline.core.pipeline.pipeline?view=azure-ml-py
training_pipeline = Pipeline(workspace=ws, steps=[step1,step2])	
Pipeline data class – Send data along the pipeline using the output of one step as the input for the next:
azureml.pipeline.core.module module - Azure Machine Learning Python | Microsoft Docs
 # as_dataset is called here and is passed to both the output and input of the next step.
pipeline_data = PipelineData('output').as_dataset()
step1 = PythonScriptStep(..., outputs=[pipeline_data])
step2 = PythonScriptStep(..., inputs=[pipeline_data])
Pipeline Step Reuse - Reuse output without re-running the step
step1 = PythonScriptStep(name='prepare data', arguments = ['--folder', prepped], outputs=[prepped], allow_reuse=True, ...)
Force all steps to re-run
pipeline_run = experiment.submit(pipeline_experiment, regenerate_outputs=True)

Pipeline Endpoints
PipelineEndpoint class – Represents a Pipeline workflow that can be triggered from a unique endpoint URL:
https://docs.microsoft.com/en-us/python/api/azureml-pipeline-core/azureml.pipeline.core.pipeline_endpoint.pipelineendpoint?view=azure-ml-py
•	Publish a pipeline to create a REST endpoint
published_pipeline = pipeline_run.publish(name='training_pipeline', description='Model training pipeline',version='1.0')
•	Post a JSON request to initiate a pipeline
o	Requires an authorization header
o	Returns a run ID
import requests
response = requests.post(rest_endpoint, headers=auth_header, json={"ExperimentName": "run training pipeline"})
run_id = response.json()["Id"]

Pipeline Parameters
PipelineParameter class – Defines a parameter in a pipeline execution:
https://docs.microsoft.com/en-us/python/api/azureml-pipeline-core/azureml.pipeline.core.graph.pipelineparameter?view=azure-ml-py
Parameterize a pipeline before publishing
reg_param = PipelineParameter(name='reg_rate', default_value=0.01)
...
step2 = EstimatorStep(name='train model', estimator_entry_script_arguments=['—reg', reg_param], ...)
...
published_pipeline = pipeline_run.publish(name='model training pipeline', description='trains a model with reg parameter', version='2.0')
Pass parameters in the JSON request
response = requests.post(rest_endpoint, headers=auth_header, json={"ExperimentName": "run training pipeline",
"ParameterAssignments": {"reg_rate": 0.1}})
Scheduling Pipelines
Schedule class – Defines a schedule on which to submit a pipeline.:
https://docs.microsoft.com/en-us/python/api/azureml-pipeline-core/azureml.pipeline.core.schedule(class)?view=azure-ml-py
•	Schedule pipeline runs based on time
daily = ScheduleRecurrence(frequency='Day', interval=1)
pipeline_schedule = Schedule.create(ws, name='Daily Training', description='trains model every day',                                 pipeline_id=published_pipeline_id,  experiment_name='Training_Pipeline', recurrence=daily)
•	Trigger pipeline runs when data changes
training_datastore = Datastore(workspace=ws, name='blob_data')
pipeline_schedule = Schedule.create(ws, name='Reactive Training', description='trains model on data change', pipeline_id=published_pipeline_id,    experiment_name='Training_Pipeline',   datastore=training_datastore,          path_on_datastore='data/training')
Module 7
Real-time Inferencing
Batch Inferencing
Deploying a Real-Time Inferencing Service
See resources available on Microsoft Learn – 
Deploy real-time machine learning services with Azure Machine Learning:
https://docs.microsoft.com/en-us/learn/modules/register-and-deploy-model-with-amls/ 
Microsoft Docs – 
Tutorial: Deploy ML models with the designer - Azure Machine Learning | Microsoft Docs:
https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-designer-automobile-price-deploy 
Model.deploy documentation-
https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.model(class)?view=azure-ml-py#deploy-workspace--name--models--inference-config-none--deployment-config-none--deployment-target-none--overwrite-false-
from azureml.core.model import Model
service = Model.deploy(workspace=ws,
                       name = 'classifier-service',
                       models = [classification_model],
                       inference_config = classifier_inference_config,
                       deployment_config = classifier_deploy_config,
                       deployment_target = production_cluster)
service.wait_for_deployment(show_output = True)

Consuming a Real-time Inferencing Service
Use the SDK
import json
x_new = [[0.1,2.3,4.1,2.0],[0.2,1.8,3.9,2.1]] # Array of feature vectors
json_data = json.dumps({"data": x_new})
response = service.run(input_data = json_data)
predictions = json.loads(response)
Use the REST Endpoint
import json
import requests
x_new = [[0.1,2.3,4.1,2.0],[0.2,1.8,3.9,2.1]] # Array of feature vectors
json_data = json.dumps({"data": x_new})
request_headers = { 'Content-Type':'application/json' }
response = requests.post(url=endpoint, data=json_data, headers=request_headers)
predictions = json.loads(response.json())
Batch Inferencing Pipeline
