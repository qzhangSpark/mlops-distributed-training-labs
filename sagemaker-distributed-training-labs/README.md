## Distributed training labs
Welcome to the art and science of optimizing neural networks at scale! You'll get hands-on experience working with our high performance [SageMaker distributed training libraries](https://aws.amazon.com/sagemaker/distributed-training/) to achieve the best performance on AWS.

You'll walk through three hands-on labs. Please note that the labs are copied from https://github.com/aws-samples/sagemaker-distributed-training-workshop with minor modifications.
* Lab 1: data parallelism 
* Lab 2: model parallelism
* Lab 3: inference with model parallelism

# Prerequisites
* Open the AWS Console and Amazon SageMaker. 
* Open the SageMaker Studio. Click "File" -> "New" -> "Terminal".
* In the terminal, git clone https://github.com/qzhangSpark/mlops-distributed-training-labs.git
* In the SageMaker Studio, click the File Browser on the left menu and navigate to the **sagemaker-distributed-training-labs** folder
* For all labs, select "Data Science" image and "Python 3" Kernel for Notebooks.

The labs are self-contained. All of the content you need is produced by the notebooks themselves or included in the directory. However, if you are in an AWS-led workshop you will most likely use the Event Engine to manage your AWS account. If not, please make sure you have an AWS account with a SageMaker Studio domain created. In this account, please [request a service limit increase](https://us-east-1.console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas) for the `ml.g4dn.12xlarge`, `ml.p3.16xlarge`, `ml.g5.48xlarge` instance types within SageMaker training. 

# Helpful links 
1. [SageMaker Data Parallel, aka Herring](https://www.amazon.science/publications/herring-rethinking-the-parameter-server-at-scale-for-the-cloud). In this paper, we introduce a custom high performance computing configuration for distributed gradient descent on AWS, available within Amazon SageMaker Training.
2. [SageMaker Model Parallel](https://arxiv.org/abs/2111.05972). In this paper, we propose a model parallelism framework available within Amazon SageMaker Training to reduce memory errors and enable training GPT-3 sized models and more! See our case study achieving 32 samples / second with 175B parameters on SageMaker over 140 p4d nodes.
3. [Preparing data for distributed training](https://aws.amazon.com/blogs/machine-learning/choose-the-best-data-source-for-your-amazon-sagemaker-training-job/). This blog post introduces different modes of working with data on SageMaker training.
4. [Hosting distributed models with DeepSpeed on SageMaker](https://github.com/dhawalkp/MLR402-reMARS-workshop/tree/master/3_deploy_gptj_with_deepspeed). In this example notebook, we demonstrate using SageMaker hosting to deploy a GPT-J model using DeepSpeed.
