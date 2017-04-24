# Serving Inception Model with TensorFlow Serving and Kubernetes

This tutorial shows how to use TensorFlow Serving components running in Docker
containers to serve the TensorFlow Inception model and how to deploy the
serving cluster with Kubernetes.

To learn more about TensorFlow Serving, we recommend
[TensorFlow Serving basic tutorial](serving_basic.md) and
[TensorFlow Serving advanced tutorial](serving_advanced.md).

To learn more about TensorFlow Inception model, we recommend
[Inception in TensorFlow](https://github.com/tensorflow/models/tree/master/inception).

-   [Part 0](#part_0_create_a_docker_image) shows how to create a TensorFlow
    Serving Docker image for deployment
-   [Part 1](#part_1_run_in_local_docker_container) shows how to run the image
    in local containers.
-   [Part 2](#part_2_deploy_in_kubernetes) shows how to deploy in Kubernetes.

## Part 0: Create a Docker image

Please refer to [Using TensorFlow Serving via Docker](docker.md) for details
about building a TensorFlow Serving Docker image.

### Run container

We build a based image `$USER/tensorflow-serving-devel` using
[Dockerfile.devel](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/tools/docker/Dockerfile.devel).
And then start a container locally using the built image.

```shell
$ docker build --pull -t $USER/tensorflow-serving-devel -f tensorflow_serving/tools/docker/Dockerfile.devel .
$ docker run --name=inception_container -it $USER/tensorflow-serving-devel
```

### Clone, configure, and build TensorFlow Serving in a container

In the running container, we clone, configure and build TensorFlow Serving.
Then test run [tensorflow_model_server](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/model_servers/main.cc).

```shell
root@c97d8e820ced:/# git clone --recurse-submodules https://github.com/tensorflow/serving
root@c97d8e820ced:/# cd serving/tensorflow
root@c97d8e820ced:/serving/tensorflow# ./configure
root@c97d8e820ced:/serving# cd ..
root@c97d8e820ced:/serving# bazel build -c opt tensorflow_serving/...
root@c97d8e820ced:/serving# ls
AUTHORS          LICENSE    RELEASE.md  bazel-bin       bazel-out      bazel-testlogs  tensorflow          zlib.BUILD
CONTRIBUTING.md  README.md  WORKSPACE   bazel-genfiles  bazel-serving  grpc            tensorflow_serving
root@c97d8e820ced:/serving# bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
Usage: model_server [--port=8500] [--enable_batching] [--model_name=my_name] --model_base_path=/path/to/export
```

### Export Inception model in container

In the running container, we run
[inception_saved_model.py](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example/inception_saved_model.py)
to export the inception model using the released
[Inception model training checkpoint](http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz).
Instead of training from scratch, we use the readily available checkpoints
of well trained variables to restore the inference graph and export it
directly.

```shell
root@c97d8e820ced:/serving# curl -O http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz
root@c97d8e820ced:/serving# tar xzf inception-v3-2016-03-01.tar.gz
root@c97d8e820ced:/serving# ls inception-v3
README.txt  checkpoint  model.ckpt-157585
root@c97d8e820ced:/serving# bazel-bin/tensorflow_serving/example/inception_saved_model --checkpoint_dir=inception-v3 --output_dir=inception-export
Successfully loaded model from inception-v3/model.ckpt-157585 at step=157585.
Successfully exported model to inception-export
root@c97d8e820ced:/serving# ls inception-export
1
root@c97d8e820ced:/serving# [Ctrl-p] + [Ctrl-q]
```

### Commit image for deployment

Note that we detach from the container at the end of above instructions
instead of terminating it, as we want to [commit](https://docs.docker.com/engine/reference/commandline/commit/)
all changes to a new image `$USER/inception_serving` for Kubernetes deployment.

```shell
$ docker commit inception_container $USER/inception_serving
$ docker stop inception_container
```

## Part 1: Run in local Docker container

Let's test the serving workflow locally using the built image.

```shell
$ docker run -it $USER/inception_serving
```

### Start the server

Run the [gRPC]( http://www.grpc.io/) `tensorflow_model_server` in the container.

```shell
root@f07eec53fd95:/# cd serving
root@f07eec53fd95:/serving# bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=inception --model_base_path=inception-export &> inception_log &
[1] 45
```

### Query the server

Query the server with [inception_client.py](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example/inception_client.py).
The client sends an image specified by the command line parameter to the server
over gRPC for classification into human readable descriptions of the
[ImageNet](http://www.image-net.org/) categories.

```shell
root@f07eec53fd95:/serving# bazel-bin/tensorflow_serving/example/inception_client --server=localhost:9000 --image=/path/to/my_cat_image.jpg
outputs {
  key: "classes"
  value {
    dtype: DT_STRING
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 5
      }
    }
    string_val: "tiger cat"
    string_val: "Egyptian cat"
    string_val: "tabby, tabby cat"
    string_val: "lynx, catamount"
    string_val: "Cardigan, Cardigan Welsh corgi"
  }
}
outputs {
  key: "scores"
  value {
    dtype: DT_FLOAT
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 5
      }
    }
    float_val: 9.5486907959
    float_val: 8.52025032043
    float_val: 8.05995368958
    float_val: 4.30645561218
    float_val: 3.93207240105
  }
}

root@f07eec53fd95:/serving# exit
```

It works! The server successfully classifies your cat image!

## Part 2: Deploy in Kubernetes

In this section we use the container image built in Part 0 to deploy a serving
cluster with [Kubernetes](http://kubernetes.io) in the
[Google Cloud Platform](http://cloud.google.com).


### GCloud project login

Here we assume you have created and logged in a
[gcloud](https://cloud.google.com/sdk/gcloud/) project named
`tensorflow-serving`.

```shell
$ gcloud auth login --project tensorflow-serving
```

### Create a container cluster

First we create a [Google Container Engine](https://cloud.google.com/container-engine/)
cluster for service deployment.

```shell
$ gcloud container clusters create inception-serving-cluster --num-nodes 5
Creating cluster inception-serving-cluster...done.
Created [https://container.googleapis.com/v1/projects/tensorflow-serving/zones/us-central1-f/clusters/inception-serving-cluster].
kubeconfig entry generated for inception-serving-cluster.
NAME                       ZONE           MASTER_VERSION  MASTER_IP        MACHINE_TYPE   NODE_VERSION  NUM_NODES  STATUS
inception-serving-cluster  us-central1-f  1.1.8           104.197.163.119  n1-standard-1  1.1.8         5          RUNNING
```

Set the default cluster for gcloud container command and pass cluster
credentials to [kubectl](http://kubernetes.io/docs/user-guide/kubectl-overview/).

```shell
$ gcloud config set container/cluster inception-serving-cluster
$ gcloud container clusters get-credentials inception-serving-cluster
Fetching cluster endpoint and auth data.
kubeconfig entry generated for inception-serving-cluster.
```

### Upload the Docker image

Let's now push our image to the
[Google Container Registry](https://cloud.google.com/container-registry/docs/)
so that we can run it on Google Cloud Platform.

First we tag the `$USER/inception_serving` image using the Container Registry
format and our project name,

```shell
$ docker tag $USER/inception_serving gcr.io/tensorflow-serving/inception
```

Next we push the image to the Registry,

```shell
$ gcloud docker -- push gcr.io/tensorflow-serving/inception
```

### Create Kubernetes Deployment and Service

The deployment consists of 3 replicas of `inception_inference` server
controlled by a
[Kubernetes Deployment](http://kubernetes.io/docs/user-guide/deployments/).
The replicas are exposed externally by a
[Kubernetes Service](http://kubernetes.io/docs/user-guide/services/)
along with an
[External Load Balancer](http://kubernetes.io/docs/user-guide/load-balancer/).

We create them using the example Kubernetes config
[inception_k8s.yaml](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example/inception_k8s.yaml).

```shell
$ kubectl create -f tensorflow_serving/example/inception_k8s.yaml
deployment "inception-deployment" created
service "inception-service" created
```

To view status of the deployment and pods:

```shell
$ kubectl get deployments
NAME                    DESIRED   CURRENT   UP-TO-DATE   AVAILABLE   AGE
inception-deployment    3         3         3            3           5s
```

```shell
$ kubectl get pods
NAME                         READY     STATUS    RESTARTS   AGE
inception-deployment-bbcbc   1/1       Running   0          10s
inception-deployment-cj6l2   1/1       Running   0          10s
inception-deployment-t1uep   1/1       Running   0          10s
```

To view status of the service:

```shell
$ kubectl get services
NAME                    CLUSTER-IP       EXTERNAL-IP       PORT(S)     AGE
inception-service       10.239.240.227   104.155.184.157   9000/TCP    1m
```

It can take a while for everything to be up and running.

```shell
$ kubectl describe service inception-service
Name:			inception-service
Namespace:		default
Labels:			run=inception-service
Selector:		run=inception-service
Type:			LoadBalancer
IP:			10.239.240.227
LoadBalancer Ingress:	104.155.184.157
Port:			<unset>	9000/TCP
NodePort:		<unset>	30334/TCP
Endpoints:		<none>
Session Affinity:	None
Events:
  FirstSeen	LastSeen	Count	From			SubobjectPath	Type		Reason		Message
  ---------	--------	-----	----			-------------	--------	------		-------
  1m		1m		1	{service-controller }			Normal		CreatingLoadBalancer	Creating load balancer
  1m		1m		1	{service-controller }			Normal		CreatedLoadBalancer	Created load balancer
```

The service external IP address is listed next to LoadBalancer Ingress.

### Query the model

We can now query the service at its external address from our local host.

```shell
$ bazel-bin/tensorflow_serving/example/inception_client --server=104.155.184.157:9000 --image=/path/to/my_cat_image.jpg
outputs {
  key: "classes"
  value {
    dtype: DT_STRING
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 5
      }
    }
    string_val: "tiger cat"
    string_val: "Egyptian cat"
    string_val: "tabby, tabby cat"
    string_val: "lynx, catamount"
    string_val: "Cardigan, Cardigan Welsh corgi"
  }
}
outputs {
  key: "scores"
  value {
    dtype: DT_FLOAT
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 5
      }
    }
    float_val: 9.5486907959
    float_val: 8.52025032043
    float_val: 8.05995368958
    float_val: 4.30645561218
    float_val: 3.93207240105
  }
}
```

You have successfully deployed Inception model serving as a service in
Kubernetes!
