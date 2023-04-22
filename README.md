# llmtopk

This is a deployment of this model.

### Docker Image

Build this image:

```shell
docker build -t llmptopk .
```


### Run


```shell
docker run -p 9999:9999 llmptopk
```

### API

The API is a simple REST API that takes a question as input and returns a list of answers.

```shell
curl -X POST 
```
