# PyTorch Scholarship Challenge 

| [HOME PAGE](https://sites.google.com/udacity.com/pytorch-scholarship-facebook/home) | [PROGRAM OVERVIEW](https://sites.google.com/udacity.com/pytorch-scholarship-facebook/program-overview) | [CALENDAR](https://sites.google.com/udacity.com/pytorch-scholarship-facebook/calendar) | [LIBRARY](https://docs.google.com/spreadsheets/d/1HnlcuI3I-d3Cli__RxOgMrxmE3aiZ8Vw5ar14WoPVRo/edit#gid=1462963974) | [FAQs](https://sites.google.com/udacity.com/pytorch-scholarship-facebook/faqs) | [RESOURCES](https://sites.google.com/udacity.com/pytorch-scholarship-facebook/resources) |
|-------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|

![logo](./docs/udacity-pytorch-256.png)

## Syllabus

1. [Introduction to Neural Networks](/intro-nn/)
1. Talking PyTorch with Soumith Chintala
1. [Introduction to PyTorch](/intro-pytorch/)
1. [Convolutional Neural Networks](/cnn/)
1. [Style Transfer](/style-transfer/)
1. [Recurrent Neural Networks](/rnn/)
1. [Sentiment Predictions with RNNs](/sentiment-rnn/)
1. Deploying PyTorch Models
1. [Challenge Project](/challenge-project/)

---

### Docker

[**GPU Version**](Dockerfile.gpu):

```sh
docker container run --rm -ti \
    -p 8888:8888 \
    -v ${pytorch-challenge-dir}:/src/ \
    --runtime=nvidia \
    hsneto/pytorch-challenge:cuda9.0-cudnn7-devel
```

[**CPU Version**](Dockerfile.cpu):

```sh
docker container run --rm -ti \
    -p 8888:8888 \
    -v ${pytorch-challenge-dir}:/src/ \
    hsneto/pytorch-challenge:cpu
```
