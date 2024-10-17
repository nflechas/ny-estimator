
## Challenge 1: Refactor DEV code

### Training, testing and inference

A class for training, testing and performing inference can be found in app/models/random_forest.py

For the sake of simplicity, the methods defined in the class, specially `training`, do no take hyperparameters as arguments, although it would be desirable to do further tuning experiments. It would also be advisable to perform cross-validation to get a better picture of the performance of the classifier, and further nested cross-validation to tune the hyperparameters (As it is, only a simple partitioning of test and train is performed). Experiment tracking with tools such as Weights&Biases or MLflow would be useful for keeping track of the experiments and storing artifcats such as weighs (this could then be loaded by the API upon initialization of the model). 

I include a very simple main function. This could be improved by adding command-line arguments to further control its behavior. However, it is enough as an integration test, allowing to test for proper inputs and outputs.

The provided pickle file (do not try to load pickle files from random people on the Internet) uses a very old sklearn version (1.0.2), you can check it with `grep -a _sklearn_version original_simple_classifier.pkl`.

Lastly, good code practices have been ensured using:

```
black .
isort --profile black .
flake8
```

### Note on EDA

Exploratory data analysis (EDA) is contained on one of the provided notebooks. I have decided against refactoring this code because I deem it too experimental to include it in a production environment. 

EDA usually requires interaction from a human expert, which would not be the case in our setup.

## Challenge 2: build an API

For this point, FastAPI will be the chosen framework.
The REST API will use the provided JSONs are input and ouput models.

Regarding the requirements, I have listed the bare minimum needed for
deployment:

```
fastapi==0.115.2
numpy==1.26.2
pandas==2.2.3
scikit-learn==1.5.2
uvicorn==0.32.0
```

This has been tested successfuly against a local browser together with the docker setup in the next challenge.

## Challenge 3: Dockerize your solution

In order to acomplish this, a docker-compose.yaml has been set up. The dockerfile can be found in docker/nyestimator.dockerfile.

Some observations: 

- Because the model is small (53M) and it doesn't need a GPU for serving, I have decided not to build from an image that includes CUDA/pytorch, but FROM python:3.11-slim.
- Again, taking into consideration the size of the model, I am including the model weights in the Docker image. If the weights were bigger, other solutions could be thought of (e.g. storing the in a S3 bucket).

To use one should run the following command:

```
docker-compose build nyestimator
docker-compose up -p 8080:8080 nyestimator
```