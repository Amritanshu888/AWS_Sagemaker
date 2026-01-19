#!/usr/bin/env python
# coding: utf-8

# #### What We will Learn
# 
# 1. S3 Buckets --> Boto3 library
# 2. IAM Roles and Users
# 3. Complete Infrastructure of AWS Sagemaker ---> Bcoz here my entire training of the model will happen in AWS Sagemaker , along with that we will also try to create the Endpoint(and that too in the Sagemaker). U need not do the deployment in the EC2 instance and all(Bcoz AWS sagemaker provides that complete infrastructure).

# In[ ]:


import sagemaker
from sklearn.model_selection import train_test_split
import boto3
import pandas as pd

## First we will make boto3 client so that we will be able to communicate with the S3 bucket.
sm_boto3 = boto3.client("sagemaker")
sess = sagemaker.Session()
region = sess.boto_session.region_name ## Region name : us-east-1 will be coming over here.
bucket = "mobbucketsagemaker" ## This is my S3 bucket which i really want to create.
print("Using bucket" + bucket)
## Similarily u can display ur region and everything.


# In[ ]:


print(region) ## This will give u us-east-1


# ## Another Functionality
# - SageMaker Runtime: Whenever u want to work with most of the services for deploying the model, training the model and all u basically use this boto3.client("sagemaker"). Sagemaker Runtime is basically used for inferencing.

# In[12]:


df = pd.read_csv("mob_price_classification_train.csv")
df.head()


# - Here the entire training will not happen in my local system it will happen in the cloud itself.

# In[13]:


df.shape


# In[14]:


df.isnull().sum()  ## Here we are checking that there are null values or not.


# In[15]:


## Since we are going to solve a classification problem we are going to check whether this dataset is imbalanced or not.
df['price_range'].value_counts()


# - Since its not an imbalanced dataset we can directly use this. Otherwise i would have to make it balanced.

# In[16]:


df.columns


# In[17]:


features = list(df.columns)
features


# In[18]:


## Storing the last feature in the label --> which is my Price_Range
label = features.pop(-1)
label


# In[ ]:


features ## Last feature price_range will be removed.


# In[21]:


## Converting our features to X and y dataset.
X = df[features] ## Popping the last feature(price_range) as it belongs to my dependent_feature
y = df[label]


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.15, random_state=0)


# In[23]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


## Converting X_train and X_test into a DataFrame
trainX = pd.DataFrame(X_train)
trainX[label] = y_train  ## So that i can save this files/data in my local or in my S3 bucket.

testX = pd.DataFrame(X_test)
testX[label] = y_test    ## So that i can save this files/data in my local or in my S3 bucket.


# In[ ]:


trainX   ## This training data will also have my last feature which is price_range


# - The reason why we are doing this is that i want to save this all in my S3 bucket and also want to save this in my csv file(locally).

# In[26]:


## Saving in my CSV file
trainX.to_csv("train-V-1.csv",index=False)
testX.to_csv("test-V-1.csv",index=False)


# In[ ]:


bucket ## See what is my bucket name : My bucket name is "mobbucketsagemaker"


# - Whatever train and test data we have created we need to probably save this in my bucket.
# - First we will see whether our bucket is created or not:
# - In AWS search for S3 --> Click create bucket --> select bucket type as "General Purpose" enter the bucket name as "mobbucketsagemaker".(The bucket name should not be existing previously) Remember to block all the public access(go below it there)--> we need to just uncheck it. Then go below and click on "Create Bucket".
# - Inside this bucket i will be uploading each and everything.

# In[ ]:


## send the data to S3. and Sagemaker will take the data for training from S3
sk_prefix = "sagemaker/mobile_price_classification/sklearncontainer" ## Creating a prefix like inside what folder i need to go and upload that particular dataset
trainpath = sess.upload_data(path='train-V-1.csv',bucket=bucket, key_prefix=sk_prefix) ## Here we are giving the path of the local file where its present.

testpath = sess.upload_data(path='test-V-1.csv',bucket=bucket, key_prefix=sk_prefix)

print(trainpath)
print(testpath) ## Where ur entire csv files are getting uploaded. Note: upload_data() is a function that is available in boto3 it is basically going to upload the data to the particular location.


# ## We will be writing a script which will be used by AWS Sagemaker To Train Models
# - Bcoz i m not going to run my model on the local machine.

# In[ ]:


get_ipython().run_cell_magic('writefile', 'script.py', '## Name of the script is script.py\n## As soon as i execute this entire cell this will basically get converted to my script.py file.\n\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_score\nimport sklearn\nimport joblib  ## To save our model in this format inside our s3 bucket\nimport boto3\nimport pathlib  ## So that we will be able to setup our path\nfrom io import StringIO\nimport argparse  ## So that we provide the arguments in the runtime\nimport os\nimport numpy as np\nimport pandas as pd\n## Once i execute this automatically my script.py file is created.\n\ndef model_fn(model_dir): ## This model directory is the same directory which will have my trained model path(Whatever path my model is saved after training that path i need to provide over here)\n    clf = joblib.load(os.path.join(model_dir,"model.joblib")) ## First i have to load that particular model\n    ## Here im joinging the path of model_dir and model.joblib so that i will be able to load it.\n\n## Starting the execution of the script over here\nif __name__ == "__main__":\n    print("[Info] Extracting Arguments")\n    ## Bcoz whenever we execute this in the AWS Sagemaker it definitely requires some kinds of Arguments\n    ## Inorder to get the arguments i will write\n    parser = argparse.ArgumentParser()\n\n    ## Hyperparameter\n    parser.add_argument("--n_estimators",type=int,default=100) ## This is my first argument\n    parser.add_argument("--random_state",type=int,default=0)  ## So basically i m using some parameters and what is the default values\n    ## If im passing in the argument that value will be set over here.\n    ## This will be required by my AWS Sagemaker\n\n    ## Data,model,and output directories\n    ## This is required as sagemaker will be expecting these values like where my data is uploaded, where my model is, where my output directories is\n    ## Why default sagemaker also creates some kind of environment variables\n    ## Lets say once my training basically happens, so by default Sagemaker does one thing it creates a path and saves that particular model in that particular path.\n\n    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR")) ## After the training is done by the sagemaker, automatically it will go and set up an environment variable which is called as "SM_MODEL_DIR" which will have the path where ur model is basically saved and it is also inside s3 only we will be seeing that particular path\n    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))  ## Next argument : After the model is basially trained in the AWS Sagemaker, this particular environment variable is also created over here\n    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))  ## This environment variable will also be created over there and it will have a value ---> This is also decided by AWS Sagemaker\n    parser.add_argument("--train-file", type=str, default="train-V-1.csv")\n    parser.add_argument("--test-file", type=str, default="test-V-1.csv")  ## These are my default train and test file name\n\n    ## This 3 pass are basically getting created in AWS-Sagemaker : "--model-dir", "--train", "--test"\n\n    args,_ = parser.parse_known_args() ## This is the function where i will be able to get my known arguments itself.\n\n    print("Sklearn Version: ", sklearn.__version__)\n    print("Joblib Version: ", joblib.__version__)\n\n    print("[INFO] Reading data")\n    print()\n    ## I will also be reading my train and test csv\n    train_df = pd.read_csv(os.path.join(args.train, args.train_file))\n    test_df = pd.read_csv(os.path.join(args.test, args.test_file))\n\n    features = list(train_df.columns)\n    label = features.pop(-1) ## Removing the last column\n\n    ## Building my train and test data\n    print("Building training and testing datasets")\n    print()\n    X_train = train_df[features]\n    X_test = test_df[features]\n    y_train = train_df[label]\n    y_test = test_df[label]\n\n    print(\'Column order:\')\n    print(features)\n    print()\n\n    print("Label column is: ",label)\n    print()\n\n    print(\'Data Shape: \')\n    print()\n    print("---- SHAPE OF TRAINING DATA (85%) ----")\n    print(X_train.shape)\n    print(y_train.shape)\n    print()\n\n    print("Training RandomForest Model ....") ## These all files will be running in my AWS Sagemaker\n    print()\n    model = RandomForestClassifier(n_estimator = args.n_estimator, random_state = args.random_state, verbose=2, n_jobs=1)\n    model.fit(X_train,y_train)\n\n    print()\n    ## Now i will setup my model path so that i save that particular model path over here\n    model_path = os.path.join(args.model_dir,"model.joblib")\n    joblib.dump(model, model_path)\n\n    print("Model saved at" + model_path) ## This model_path is basically available in our S3 bucket\n\n    y_pred_test = model.predict(X_test)\n    test_acc = accuracy_score(y_test,y_pred_test)\n    test_rep = classification_report(y_test,y_pred_test)\n\n    print()\n    print("---- METRICS RESULTS FOR TESTING DATA ----")\n    print()\n    print("Total Rows are: ",X_test.shape[0])\n    print("[TESTING] Model Accuracy is: ",test_acc)\n    print("[TESTING] Testing Report: ")\n    print(test_rep)\n\n')


# ### AWS Sagemaker Entry Point to Execute the Training script
# - Because AWS needs to probably execute this and how we are going to provide script.py that we are going to see over here.

# - Note : In the below code in the role part , search for IAM in the AWS account , in IAM go to roles(its a option just like user) --> Here we need to specify a role which has the access to the AWS Sagemaker execution(bcoz we are doing the execution from my coding environments).
# - Click on "Create Role" --> Select trusted entity type: "AWS Service" , below in services select the service as sagemaker , then it will ask me whether we need to select : SageMaker Execution or SageMaker - HyperPod Clusters , we really need to select this first option(Sagemaker Execution) bcoz this allows to access S3, ECR, and cloudwatch(all these access u will be able to get it). Then click next. Again next.
# - The next page --> Name, Review, and create , in role details it will ask me to provide Role Details , where we have to enter role details like Role Name , lets provide the name sagemakeraccess , then finally click create role.
# - So here u will be able to see that sagemaker access role is created , from this u will be able to access everything like AWS S3 bucket, click that created role , in the summary u will have ARN url , this url will specifically be my role. We will copy this ARN and it will be assigned to my role in below code.
# - In AWS Sagemaker also u will get different different instance access. Instance basically means what kind of machine.

# In[ ]:


## Setup the entry point for the Sagemaker : Like how its going to pickup the script and all.
from sagemaker.sklearn.estimator import SKLearn

FRAMEWORK_VERSION="0.23-1"

sklearn_estimator = SKLearn(
    entry_point="script.py", ## This script.py will be responsible in executing my entire code
    role="",
    instance_count = 1,
    instance_type = "ml.m5.large", ## There are many instances like this(This instance will be used for training my entire script).
    framework_version = FRAMEWORK_VERSION,
    base_job_name = "RF-custom-sklearn",
    hyperparameters = {
        "n_estimators":100,
        "random_state":0
    },
    use_spot_instance = True, ## It is going to create the instance, its going to train the model and then its going to delete the instance
    max_run = 3600
)
## This is the entire configuration that we require in order to start my instance in the AWS Sagemaker


# - Now i will launch my Training Job and this will happen Asynchronously(means we are just going to execute this in the backend and automatically this will get started)

# In[ ]:


# launch training job, with asynchronous call
sklearn_estimator.fit({"train":trainpath, "test":testpath}, wait=True)
## As soon as i execute this my training will automatically get started in my AWS sagemaker
## First it will create an instance in my AWS Sagemaker, it will be creating a training-job with name: Will be shown , this training job will
## be available in my AWS Sagemaker


# - Search for AWS Sagemaker in AWS account , there in sidebar u have something called as Training in training go to Training Job automatically u will see that training job has been created , when u click that training job u will be able to see the entire details.
# - In the details , under Output Data Configuration u will see S3 output path this is where ur model will be saved.
# - Search S3 bucket , inside S3 go and click on buckets there u will be able to see the folder that is basically created , click on the folder u will be able to see the model that is basically created. There when u click on the folder look for RF-custom-sklearn which u have given above. Click on it --> click source and there ur directory location will be there.

# ## Note:
# - Whenever sagemaker does training its going to create the folder inside S3 buckets where ur model details will be there.

# ## Below code written is basically to get the model from S3

# In[ ]:


sklearn_estimator.latest_training_job.wait(logs="None") ## From here i will get the latest training job
## Then i will probably tell my boto3 to describe this
artifact = sm_boto3.describe_training_job(
    TrainingJobName = sklearn_estimator.latest_training_job.name
)["ModelArtifacts"]["S3ModelArtifacts"]
## Whatever was my latest training job name i will take up the ModelArtifacts and S3ModelArtifacts(these are the keys that is basically present inside that).


# - If i execute the above code i should be able to get my artifacts(artifact will probably provide u the entire information).

# In[ ]:


## Printing my Artifact
artifact  


# ## Best thing about AWS Sagemaker
# - Here on instance we are getting the instance, we are training the model, and closing it so that much more charges should not happen that is the best thing abt AWS Sagemaker.

# ## Next we will create the endpoint(in AWS Sagemaker) and from there we will do the inferencing.

# ## Deploy the Model for Endpoint
# - Here we will discuss how we can probably deploy this model and create a endpoint.
# - This should also happen in our AWS Sagemaker.

# In[ ]:


from sagemaker.sklearn.model import SKLearnModel
from time import gmtime,strftime  ## I want to create my model with recent time itself.

model_name = "Custom-sklearn-model-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
model = SKLearnModel(
    name=model_name, ## Entire model will be created with this name into our endpoint
    model_data=artifact,
    role="Same what u have pasted above. That ARN URL",
    entry_point = "script.py",
    framework_version=FRAMEWORK_VERSION
)


# In[ ]:


model   ## It will give output that : It is a type of sklearn model at this particular location


# In[ ]:


## Endpoint Deployment
endpoint_name = "Custom-sklearn-model-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print("EndpointName={}".format(endpoint_name))

predictor = model.deploy(
    initial_instance_count = 1, ## Bcoz for deployment also i require a seperate instance
    instance_type = "ml.m4.xlarge", ## This is the instance type that i m going to take for the deployment purpose and again it depends on the model size
    endpoint_name = endpoint_name
)
## Once we execute this model.deploy here our model will be deploying


# - Now if u go to AWS Sagemaker and there u go to Inference , so there in inference u have something called as Endpoints , click it and here u will be able to see my endpoint is basically getting created. Along with endpoint we also have endpoint coniguration, u will be able to also see the information over here.
# - I can use the above predictor variable and i can do any kind of prediction that i want. So like i can take records and directly do the prediction itself.
# - Once the endpoint is generated click the endpoint and u will be able to see more info.

# In[ ]:


testX[features]


# In[ ]:


testX[features][0:2]  ## Here u will be able to see the first two records


# In[ ]:


print(predictor.predict(testX[features][0:2].values.tolist())) ## Here i have to get two outputs saying that what category of price it basically follows
## Output : [3,0] --> here we did prediction for 1st two records i.e. for 1st record the output is 3 and for second record output is 0. i.e. 3rd category and 0th category


# In[ ]:


sm_boto3.delete_endpoint(EndpointName=endpoint_name)
## Final thing is deleting the endpoint
## This is important bcoz : Otherwise ur endpoint will be created permanently i.e. it will keep on adding billing to ur AWS account
## This is one way of doing it from the code
## Other way : Search for Sagemaker , inside the sagemaker from the endpoint also i can close it up(Go to inferences --> There endpoints , select the endpoint , go to actions and delete the endpoint).
## Either u can do it from the code or u can do it from here.
## In sagemaker if u go to training --> training jobs this is basically opening the machine , training it and closing it from this u don't need to delete it.

