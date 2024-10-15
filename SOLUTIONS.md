
## Challenge 3 - Dockerize your solution

In order to acomplish this, a docker-compose.yaml has been set up. The dockerfile can be found in docker/nyestimator.dockerfile.

Some observations: because the model is small and doesn't need a GPU for serving, I have decided not to build from an image that includes CUDA/pytorch, but FROM python:3.11-slim

To use one should:

```
docker-compose run -p 8080:8080 nyestimator bash
```

### Challenge 2: build an API
