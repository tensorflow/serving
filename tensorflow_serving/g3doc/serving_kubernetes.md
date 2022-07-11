# Use TensorFlow Serving with Kubernetes

This tutorial shows how to use TensorFlow Serving components running in Docker
containers to serve the TensorFlow ResNet model and how to deploy the serving
cluster with Kubernetes.

To learn more about TensorFlow Serving, we recommend
[TensorFlow Serving basic tutorial](serving_basic.md) and
[TensorFlow Serving advanced tutorial](serving_advanced.md).

To learn more about TensorFlow ResNet model, we recommend reading
[ResNet in TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet).

-   [Part 1](#part_1_setup) gets your environment setup
-   [Part 2](#part_2_running_in_docker) shows how to run the local Docker
    serving image
-   [Part 3](#part_3_deploy_in_kubernetes) shows how to deploy in Kubernetes.

## Part 1: Setup

Before getting started, first [install Docker](docker.md#installing-docker).

### Download the ResNet SavedModel

Let's clear our local models directory in case we already have one:

```shell
rm -rf /tmp/resnet
```

Deep residual networks, or ResNets for short, provided the breakthrough idea of
identity mappings in order to enable training of very deep convolutional neural
networks. For our example, we will download a TensorFlow SavedModel of ResNet
for the ImageNet dataset.

```shell
# Download Resnet model from TF Hub
wget https://tfhub.dev/tensorflow/resnet_50/classification/1?tf-hub-format=compressed -o resnet.tar.gz

# Extract SavedModel into a versioned subfolder ‘123’
mkdir -p /tmp/resnet/123
tar xvfz resnet.tar.gz -C /tmp/resnet/123/
```

We can verify we have the SavedModel:

```console
$ ls /tmp/resnet/*
saved_model.pb  variables
```

## Part 2: Running in Docker

### Commit image for deployment

Now we want to take a serving image and
[commit](https://docs.docker.com/engine/reference/commandline/commit/) all
changes to a new image `$USER/resnet_serving` for Kubernetes deployment.

First we run a serving image as a daemon:

```shell
docker run -d --name serving_base tensorflow/serving
```

Next, we copy the ResNet model data to the container's model folder:

```shell
docker cp /tmp/resnet serving_base:/models/resnet
```

Finally we commit the container to serving the ResNet model:

```shell
docker commit --change "ENV MODEL_NAME resnet" serving_base \
  $USER/resnet_serving
```

Now let's stop the serving base container

```shell
docker kill serving_base
docker rm serving_base
```

### Start the server

Now let's start the container with the ResNet model so it's ready for serving,
exposing the gRPC port 8500:

```shell
docker run -p 8500:8500 -t $USER/resnet_serving &
```

### Query the server

For the client, we will need to clone the TensorFlow Serving GitHub repo:

```shell
git clone https://github.com/tensorflow/serving
cd serving
```

Query the server with
[resnet_client_grpc.py](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example/resnet_client_grpc.py).
The client downloads an image and sends it over gRPC for classification into
[ImageNet](http://www.image-net.org/) categories.

```shell
tools/run_in_docker.sh python tensorflow_serving/example/resnet_client_grpc.py
```

This should result in output like:

```console
outputs {
  key: "classes"
  value {
    dtype: DT_INT64
    tensor_shape {
      dim {
        size: 1
      }
    }
    int64_val: 286
  }
}
outputs {
  key: "probabilities"
  value {
    dtype: DT_FLOAT
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 1001
      }
    }
    float_val: 2.41628322328e-06
    float_val: 1.90121829746e-06
    float_val: 2.72477100225e-05
    float_val: 4.42638565801e-07
    float_val: 8.98362372936e-07
    float_val: 6.84421956976e-06
    float_val: 1.66555237229e-05
...
    float_val: 1.59407863976e-06
    float_val: 1.2315689446e-06
    float_val: 1.17812135159e-06
    float_val: 1.46365800902e-05
    float_val: 5.81210713335e-07
    float_val: 6.59980651108e-05
    float_val: 0.00129527016543
  }
}
model_spec {
  name: "resnet"
  version {
    value: 123
  }
  signature_name: "serving_default"
}
```

It works! The server successfully classifies a cat image!

## Part 3: Deploy in Kubernetes

In this section we use the container image built in Part 0 to deploy a serving
cluster with [Kubernetes](http://kubernetes.io) in the
[Google Cloud Platform](http://cloud.google.com).


### GCloud project login

Here we assume you have created and logged in a
[gcloud](https://cloud.google.com/sdk/gcloud/) project named
`tensorflow-serving`.

```shell
gcloud auth login --project tensorflow-serving
```

### Create a container cluster

First we create a
[Google Kubernetes Engine](https://cloud.google.com/container-engine/) cluster
for service deployment.

```shell
$ gcloud container clusters create resnet-serving-cluster --num-nodes 5
```

Which should output something like:

```console
Creating cluster resnet-serving-cluster...done.
Created [https://container.googleapis.com/v1/projects/tensorflow-serving/zones/us-central1-f/clusters/resnet-serving-cluster].
kubeconfig entry generated for resnet-serving-cluster.
NAME                       ZONE           MASTER_VERSION  MASTER_IP        MACHINE_TYPE   NODE_VERSION  NUM_NODES  STATUS
resnet-serving-cluster  us-central1-f  1.1.8           104.197.163.119  n1-standard-1  1.1.8         5          RUNNING
```

Set the default cluster for gcloud container command and pass cluster
credentials to [kubectl](http://kubernetes.io/docs/user-guide/kubectl-overview/).

```shell
gcloud config set container/cluster resnet-serving-cluster
gcloud container clusters get-credentials resnet-serving-cluster
```

which should result in:

```console
Fetching cluster endpoint and auth data.
kubeconfig entry generated for resnet-serving-cluster.
```

### Upload the Docker image

Let's now push our image to the
[Google Container Registry](https://cloud.google.com/container-registry/docs/)
so that we can run it on Google Cloud Platform.

First we tag the `$USER/resnet_serving` image using the Container Registry
format and our project name,

```shell
docker tag $USER/resnet_serving gcr.io/tensorflow-serving/resnet
```

Next, we configure Docker to use gcloud as a credential helper:

```shell
gcloud auth configure-docker
```

Next we push the image to the Registry,

```shell
docker push gcr.io/tensorflow-serving/resnet
```

### Create Kubernetes Deployment and Service

The deployment consists of 3 replicas of `resnet_inference` server controlled by
a [Kubernetes Deployment](http://kubernetes.io/docs/user-guide/deployments/).
The replicas are exposed externally by a
[Kubernetes Service](http://kubernetes.io/docs/user-guide/services/) along with
an
[External Load Balancer](http://kubernetes.io/docs/user-guide/load-balancer/).

We create them using the example Kubernetes config
[resnet_k8s.yaml](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example/resnet_k8s.yaml).

```shell
kubectl create -f tensorflow_serving/example/resnet_k8s.yaml
```

With output:

```console
deployment "resnet-deployment" created
service "resnet-service" created
```

To view status of the deployment and pods:

```console
$ kubectl get deployments
NAME                    DESIRED   CURRENT   UP-TO-DATE   AVAILABLE   AGE
resnet-deployment    3         3         3            3           5s
```

```console
$ kubectl get pods
NAME                         READY     STATUS    RESTARTS   AGE
resnet-deployment-bbcbc   1/1       Running   0          10s
resnet-deployment-cj6l2   1/1       Running   0          10s
resnet-deployment-t1uep   1/1       Running   0          10s
```

To view status of the service:

```console
$ kubectl get services
NAME                    CLUSTER-IP       EXTERNAL-IP       PORT(S)     AGE
resnet-service       10.239.240.227   104.155.184.157   8500/TCP    1m
```

It can take a while for everything to be up and running.

```console
$ kubectl describe service resnet-service
Name:           resnet-service
Namespace:      default
Labels:         run=resnet-service
Selector:       run=resnet-service
Type:           LoadBalancer
IP:         10.239.240.227
LoadBalancer Ingress:   104.155.184.157
Port:           <unset> 8500/TCP
NodePort:       <unset> 30334/TCP
Endpoints:      <none>
Session Affinity:   None
Events:
  FirstSeen LastSeen    Count   From            SubobjectPath   Type        Reason      Message
  --------- --------    -----   ----            -------------   --------    ------      -------
  1m        1m      1   {service-controller }           Normal      CreatingLoadBalancer    Creating load balancer
  1m        1m      1   {service-controller }           Normal      CreatedLoadBalancer Created load balancer
```

The service external IP address is listed next to LoadBalancer Ingress.

### Query the model

We can now query the service at its external address from our local host.

```console
$ tools/run_in_docker.sh python \
  tensorflow_serving/example/resnet_client_grpc.py \
  --server=104.155.184.157:8500
outputs {
  key: "classes"
  value {
    dtype: DT_INT64
    tensor_shape {
      dim {
        size: 1
      }
    }
    int64_val: 286
  }
}
outputs {
  key: "probabilities"
  value {
    dtype: DT_FLOAT
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 1001
      }
    }
    float_val: 2.41628322328e-06
    float_val: 1.90121829746e-06
    float_val: 2.72477100225e-05
    float_val: 4.42638565801e-07
    float_val: 8.98362372936e-07
    float_val: 6.84421956976e-06
    float_val: 1.66555237229e-05
...
    float_val: 1.59407863976e-06
    float_val: 1.2315689446e-06
    float_val: 1.17812135159e-06
    float_val: 1.46365800902e-05
    float_val: 5.81210713335e-07
    float_val: 6.59980651108e-05
    float_val: 0.00129527016543
  }
}
model_spec {
  name: "resnet"
  version {
    value: 1538687457
  }
  signature_name: "serving_default"
}
```

You have successfully deployed the ResNet model serving as a service in
Kubernetes!
