# Serving Inception Model with TensorFlow Serving and Kubernetes

This tutorial shows how to use TensorFlow Serving components running in Docker
containers to serve the TensorFlow Inception model and how to deploy the
serving cluster with Kubernetes.

To learn more about TensorFlow Serving, we recommend
[TensorFlow Serving basic tutorial](serving_basic.md) and
[TensorFlow Serving advanced tutorial](serving_advanced.md).

To learn more about TensorFlow Inception model, we recommend
[Inception in TensorFlow](https://github.com/tensorflow/models/tree/master/research/inception).

-   [Part 1](#part_1_export_the_inception_model) shows how to export the
    Inception model
-   [Part 2](#part_2_run_in_local_docker_container) shows how to run the local
    Docker serving image
-   [Part 3](#part_3_deploy_in_kubernetes) shows how to deploy in Kubernetes.

## Part 1: Export the Inception model

Before getting started, first [install Docker](docker.md#installing-docker)

### Clone Tensorflow Serving

First we want to clone the TensorFlow Serving source to our local machine:

```shell
git clone https://github.com/tensorflow/serving
cd serving
```

Clear our local models directory in case we already have one

```shell
rm -rf ./models/inception
```

### Build TensorFlow Serving Inception model exporter

Next, we will build the Inception model exporter using a docker container.

Note: All `bazel build` commands below use the standard `-c opt` flag. To
further optimize the build, refer to the
[instructions here](setup.md#optimized-build).

```shell
tools/bazel_in_docker.sh bazel build -c opt \
  tensorflow_serving/example:inception_saved_model
```

### Export Inception model

With the inception model built, we run
[inception_saved_model.py](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example/inception_saved_model.py)
to export the Inception model using the released
[Inception model training checkpoint](http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz).
Instead of training from scratch, we use the readily available checkpoints of
well trained variables to restore the inference graph and export it directly.

```shell
curl -O http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz
tar xzf inception-v3-2016-03-01.tar.gz
```

We can verify we have the checkpoint:

```console
$ ls inception-v3
README.txt  checkpoint  model.ckpt-157585
```

Now let's export the Inception model:

```shell
tools/bazel_in_docker.sh \
  bazel-bin/tensorflow_serving/example/inception_saved_model \
  --checkpoint_dir=inception-v3 --output_dir=models/inception
```

This should result in output like:

```console
Successfully loaded model from inception-v3/model.ckpt-157585 at step=157585.
Exporting trained moedl to models/inception/1
Successfully exported model to models/inception
$ ls models/inception
1
```

## Part 2: Run in local Docker container

### Commit image for deployment

Now we want to take a serving image and
[commit](https://docs.docker.com/engine/reference/commandline/commit/) all
changes to a new image `$USER/inception_serving` for Kubernetes deployment.

First we run a serving image as a daemon:

```shell
docker run -d --name serving_base tensorflow/serving
```

Next, we copy the Inception model data to the container's model folder:

```shell
docker cp models/inception serving_base:/models/inception
```

Finally we commit the container to serving the Inception model:

```shell
docker commit --change "ENV MODEL_NAME inception" serving_base \
  $USER/inception_serving
```

Now let's stop the serving base container

```shell
docker kill serving_base
```

### Start the server

Now let's start the container with the Inception model so it's ready for
serving, exposing the gRPC port 8500:

```shell
docker run -p 8500:8500 -t $USER/inception_serving &
```

### Query the server

Query the server with [inception_client.py](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example/inception_client.py).
The client sends an image specified by the command line parameter to the server
over gRPC for classification into human readable descriptions of the
[ImageNet](http://www.image-net.org/) categories.

Note: We leave it as an exercise to the reader to find an image of a cat on the
Internet.

```shell
tools/bazel_in_docker.sh bazel build -c opt \
  tensorflow_serving/example:inception_client
tools/bazel_in_docker.sh bazel-bin/tensorflow_serving/example/inception_client \
  --server=127.0.0.1:8500 --image=local/path/to/my_cat_image.jpg
```

This should result in output like:

```console
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

It works! The server successfully classifies your cat image!

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
$ gcloud container clusters create inception-serving-cluster --num-nodes 5
```

Which should output something like:

```console
Creating cluster inception-serving-cluster...done.
Created [https://container.googleapis.com/v1/projects/tensorflow-serving/zones/us-central1-f/clusters/inception-serving-cluster].
kubeconfig entry generated for inception-serving-cluster.
NAME                       ZONE           MASTER_VERSION  MASTER_IP        MACHINE_TYPE   NODE_VERSION  NUM_NODES  STATUS
inception-serving-cluster  us-central1-f  1.1.8           104.197.163.119  n1-standard-1  1.1.8         5          RUNNING
```

Set the default cluster for gcloud container command and pass cluster
credentials to [kubectl](http://kubernetes.io/docs/user-guide/kubectl-overview/).

```shell
gcloud config set container/cluster inception-serving-cluster
gcloud container clusters get-credentials inception-serving-cluster
```

which should result in:

```console
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
docker tag $USER/inception_serving gcr.io/tensorflow-serving/inception
```

Next we push the image to the Registry,

```shell
gcloud docker -- push gcr.io/tensorflow-serving/inception
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
kubectl create -f tensorflow_serving/example/inception_k8s.yaml
```

With output:

```console
deployment "inception-deployment" created
service "inception-service" created
```

To view status of the deployment and pods:

```console
$ kubectl get deployments
NAME                    DESIRED   CURRENT   UP-TO-DATE   AVAILABLE   AGE
inception-deployment    3         3         3            3           5s
```

```console
$ kubectl get pods
NAME                         READY     STATUS    RESTARTS   AGE
inception-deployment-bbcbc   1/1       Running   0          10s
inception-deployment-cj6l2   1/1       Running   0          10s
inception-deployment-t1uep   1/1       Running   0          10s
```

To view status of the service:

```console
$ kubectl get services
NAME                    CLUSTER-IP       EXTERNAL-IP       PORT(S)     AGE
inception-service       10.239.240.227   104.155.184.157   8500/TCP    1m
```

It can take a while for everything to be up and running.

```console
$ kubectl describe service inception-service
Name:           inception-service
Namespace:      default
Labels:         run=inception-service
Selector:       run=inception-service
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
$ tools/bazel_in_docker.sh \
  bazel-bin/tensorflow_serving/example/inception_client \
  --server=104.155.184.157:8500 --image=local/path/to/my_cat_image.jpg
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
