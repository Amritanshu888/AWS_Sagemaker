## Name of the script is script.py
## As soon as i execute this entire cell this will basically get converted to my script.py file.

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_score
import sklearn
import joblib  ## To save our model in this format inside our s3 bucket
import boto3
import pathlib  ## So that we will be able to setup our path
from io import StringIO
import argparse  ## So that we provide the arguments in the runtime
import os
import numpy as np
import pandas as pd
## Once i execute this automatically my script.py file is created.

def model_fn(model_dir): ## This model directory is the same directory which will have my trained model path(Whatever path my model is saved after training that path i need to provide over here)
    clf = joblib.load(os.path.join(model_dir,"model.joblib")) ## First i have to load that particular model
    ## Here im joinging the path of model_dir and model.joblib so that i will be able to load it.

## Starting the execution of the script over here
if __name__ == "__main__":
    print("[Info] Extracting Arguments")
    ## Bcoz whenever we execute this in the AWS Sagemaker it definitely requires some kinds of Arguments
    ## Inorder to get the arguments i will write
    parser = argparse.ArgumentParser()

    ## Hyperparameter
    parser.add_argument("--n_estimators",type=int,default=100) ## This is my first argument
    parser.add_argument("--random_state",type=int,default=0)  ## So basically i m using some parameters and what is the default values
    ## If im passing in the argument that value will be set over here.
    ## This will be required by my AWS Sagemaker

    ## Data,model,and output directories
    ## This is required as sagemaker will be expecting these values like where my data is uploaded, where my model is, where my output directories is
    ## Why default sagemaker also creates some kind of environment variables
    ## Lets say once my training basically happens, so by default Sagemaker does one thing it creates a path and saves that particular model in that particular path.

    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR")) ## After the training is done by the sagemaker, automatically it will go and set up an environment variable which is called as "SM_MODEL_DIR" which will have the path where ur model is basically saved and it is also inside s3 only we will be seeing that particular path
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))  ## Next argument : After the model is basially trained in the AWS Sagemaker, this particular environment variable is also created over here
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))  ## This environment variable will also be created over there and it will have a value ---> This is also decided by AWS Sagemaker
    parser.add_argument("--train-file", type=str, default="train-V-1.csv")
    parser.add_argument("--test-file", type=str, default="test-V-1.csv")  ## These are my default train and test file name

    ## This 3 pass are basically getting created in AWS-Sagemaker : "--model-dir", "--train", "--test"

    args,_ = parser.parse_known_args() ## This is the function where i will be able to get my known arguments itself.

    print("Sklearn Version: ", sklearn.__version__)
    print("Joblib Version: ", joblib.__version__)

    print("[INFO] Reading data")
    print()
    ## I will also be reading my train and test csv
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    features = list(train_df.columns)
    label = features.pop(-1) ## Removing the last column

    ## Building my train and test data
    print("Building training and testing datasets")
    print()
    X_train = train_df[features]
    X_test = test_df[features]
    y_train = train_df[label]
    y_test = test_df[label]

    print('Column order:')
    print(features)
    print()

    print("Label column is: ",label)
    print()

    print('Data Shape: ')
    print()
    print("---- SHAPE OF TRAINING DATA (85%) ----")
    print(X_train.shape)
    print(y_train.shape)
    print()

    print("Training RandomForest Model ....") ## These all files will be running in my AWS Sagemaker
    print()
    model = RandomForestClassifier(n_estimator = args.n_estimator, random_state = args.random_state, verbose=2, n_jobs=1)
    model.fit(X_train,y_train)

    print()
    ## Now i will setup my model path so that i save that particular model path over here
    model_path = os.path.join(args.model_dir,"model.joblib")
    joblib.dump(model, model_path)

    print("Model saved at" + model_path) ## This model_path is basically available in our S3 bucket

    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test,y_pred_test)
    test_rep = classification_report(y_test,y_pred_test)

    print()
    print("---- METRICS RESULTS FOR TESTING DATA ----")
    print()
    print("Total Rows are: ",X_test.shape[0])
    print("[TESTING] Model Accuracy is: ",test_acc)
    print("[TESTING] Testing Report: ")
    print(test_rep)

