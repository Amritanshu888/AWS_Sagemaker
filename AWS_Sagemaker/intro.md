- In AWS SageMaker u will be able to build, train, and deploy machine learning models for any use case with fully managed infrastructure, tools, and workflows.
- Till Now for all projects we used AWS EC2 instance. Did entire coding in our local machine, after that once we build our model, once we build our entire pipeline, then we convert that into a Docker Image, then we deploy that docker image into AWS instance then , from that docker image we try to deploy it in EC2 instance.
- This was the entire process.
- AWS Sagemaker: It provides a fully managed infrastructure(of building pipelines, tools and workflows) it will definitely be providing.

- The dataset which we have taken is mob_price_classification_dataset ---> Output feature(to be predicted) is Price Range.

- We deploy each and every thing in the Sagemaker. From dataset we will also be using lot of AWS services like S3 bucket, we will be creating roles, We will be creating IAM users.

- Used by many companies : Bcoz here we are going to automate the entire process of creating the model, training the model, creating the endpoints, and then we will also try to delete the endpoints(as there are some charges in AWS Sagemaker).

## AWS
- Go to your AWS console search for IAM , in IAM click on users , there click on Create User , on next page enter the user details(User name = sagemakerdemo) then click on next.
- Under Permission options : select Attach policies directly(whenever u work in a specific company they will provide u access for only servies that u will be using). In permissions policy there u will have multiple permissions below (there u can click the Checkbox -> tick mark) to get the access. Right now we will only select AdministratorAccess --> as its for my personal use. Then click on next ,then click Create user.
- Then in IAM sagemakerdemo will be there, click on that there u go to Security credentials and under Access Keys click "Create access key". --> Then in usecase click on : Command Line Interface(bcoz i require an access key inorder to access it). Click the confirmation given below in that same page. Click on next. On next page it will ask to to set description tag(which is optional), on that same page uwill click on Create access key
- U will reach the page Retrieve access keys, there u will have two things : Access key, Secret access key.
- U can also download it as a CSV file(u will be able to remember this Access Key and Secret Access Key).

- Once this is done next thing u need to install is AWS cli. Search for "AWS Cli Windows Download". For windows u can download the .msi file. We have commands on website from which u can unzip the file and install it in ur local.

- To check whether AWS Cli is downloaded or not we can configure it : Open the terminal enter "aws configure" it will only work if the AWS cli is already installed.
- This will ask u for ur AWS Access Key ID: Copy the AWS access key id and paste it here.
- Then this will ask for AWS Secret Access Key ID: Copy and Paste the Secret Access Key ID.
- Then it will ask for Default Region Name: which is us-east-1 , press enter
- Then the default output format which will be json : press enter.
- This is how we configure our AWS cli.
