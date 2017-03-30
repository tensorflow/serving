---
---
<style>hr{display:none;}</style>

# TensorFlow Serving Batching Guide

[TOC]

## Introduction

While serving a TensorFlow model, batching individual model inference requests
together can be important for performance. In particular, batching is necessary
to unlock the high throughput promised by hardware accelerators such as GPUs.
This is a library for batching requests and scheduling the batches. The library
is not tied to GPUs, per se, and can be used for any situation that benefits
from processing groups of small tasks in tandem (but this document assumes GPUs
to simplify exposition). It offers a specific TensorFlow Session API, as well as
lower-level APIs that can be used to batch at other granularities.

The library is currently split across two locations:
(1) tensorflow/contrib/batching (core API and implementation), and
(2) tensorflow_serving/batching (higher-level and experimental code).

The library offers several alternative classes to choose from. The reason for
the choices is that there are many reasonable ways to perform batching. No
single "best" approach dominates because different use-cases have different
requirements, e.g.:

* API preferences: Tensor API vs. general API; synchronous vs. asynchronous.

* Does the model have significant CPU components, in addition to GPU?

* Does the server need to interleave requests to multiple models (or versions)?

* Is this for online serving or bulk processing (e.g. map-reduce jobs)?

Furthermore, whereas some deployments need advanced capabilities to squeeze out
maximal performance, others just want something simple that performs reasonably
well.

This document gives a tour of the batching library, including when to use which
class, and some best practices.

## Simple Batching

If you are new to the batching library and/or you have only basic requirements,
you can focus just on `BatchingSession` and/or `BasicBatchScheduler`.

### `BatchingSession`

`BatchingSession` adds batching to a standard `tensorflow::Session`, and lets
you call `Session::Run()` with individual (non-batched) tensors while getting
the benefits of batching "under the covers". This abstraction works well if your
application uses TensorFlow (naturally), and can accommodate `Session::Run()`'s
synchronous API -- request threads make `Session::Run()` calls that block while
awaiting other calls to group into the same batch. To achieve good throughput
with this synchronous API, it is recommended to set the number of client threads
to roughly twice the maximum batch size.

`BatchingSession` can be used with any of the library's batch schedulers
including `BasicBatchScheduler`, which offers a way to bound how long each
`Session::Run()` call blocks. The simplest way to use `BatchingSession` is via
`CreateRetryingBasicBatchingSession()`, which gives you a `tensorflow::Session`
object that uses a `BasicBatchScheduler` underneath, and also handles retrying
requests that overflow the scheduler's queue. You will supply some key
parameters governing the scheduling and execution of batched requests that are
passed to the underlying `BasicBatchScheduler`; see below for details.
`BasicBatchScheduler` has a bounded-size queue; you can set parameters that
govern whether `Session::Run()` should fail upon finding a full queue, or retry
some number of times with a delay; again, see below.

A final configuration parameter is `allowed_batch_sizes`. This parameter is
optional. If unset, then batch sizes can vary freely between 1 and the maximum
allowed size, say 1024. Depending on your environment, having a large numbrer
of possible batch sizes may cause problems. The `allowed_batch_sizes` parameter
lets you limit the batch sizes to a fixed set, say 128, 256, 512, 1024.
`BatchingSession` adheres to this restriction by padding invalid-size batches
with dummy data to round up to the next valid size.

### `BasicBatchScheduler`

`BasicBatchScheduler` is a lower-level abstraction than `BatchingSession`. It
is not tied to tensors/TensorFlow per se, making it quite flexible. It is
suitable for servers that handle homogeneous requests (see
`basic_batch_scheduler.h` for a precise characterization of that restriction).

`BasicBatchScheduler` offers an asynchronous API that it shares with its less
basic cousins (discussed below), called `BatchScheduler`. The API is
templetized by a `BatchTask` class that encapsulates a unit of work to be
batched. A non-blocking `Schedule()` method is used to enqueue a task for
processing. Once a batch of tasks is ready to be processed, a callback is
invoked on a separate thread to process the batch. A good illustration of how
to use this API is found in the implementation of `BatchingSession` in
`batching_session.cc`.

## Batch Scheduling Parameters and Tuning

The parameters that govern batch scheduling (e.g. in
`BasicBatchScheduler::Options`) are:

* `max_batch_size`: The maximum size of any batch. This parameter governs the
throughput/latency tradeoff, and also avoids having batches that are so large
they exceed some resource constraint (e.g. GPU memory to hold a batch's data).

* `batch_timeout_micros`: The maximum amount of time to wait before executing a
batch (even if it hasn't reached `max_batch_size`). Used to rein in tail
latency. (See `basic_batch_scheduler.h` for the exact latency contract.)

* `num_batch_threads`: The degree of parallelism, i.e. the maximum number of
batches processed concurrently.

* `max_enqueued_batches`: The number of batches worth of tasks that can be
enqueued to the scheduler. Used to bound queueing delay, by turning away
requests that would take a long time to get to, rather than building up a large
backlog.

### Performance Tuning

The best values to use for the batch scheduling parameters depend on your model,
system and environment, as well as your throughput and latency goals. Choosing
good values is best done via experiments. Here are some guidelines that may be
helpful in selecting values to experiment with.

#### Overall Guidelines

First of all, while experimenting you should temporarily set
`max_enqueued_batches` to infinity. Later, for your production setup, set it as
follows: If you are performing online serving, depending on the policy used to
(re-)route requests to server instances, consider setting `max_enqueued_batches`
equal to `num_batch_threads` to minimize queueing delay at a given server while
keeping it busy. For bulk processing jobs, set `max_enqueued_batches` to a
generous value, but low enough to avoid out-of-memory crashes.

Second, if for system architecture reasons you need to constrain the set of
possible batch sizes (e.g. just 100, 200 or 400, rather than any value between 1
and 400): If you are using `BatchingSession` you can set the
`allowed_batch_sizes` parameter. Otherwise, you can arrange for your callback
code to pad the batches with dummy elements.

#### CPU-only: One Approach

If your system is CPU-only (no GPU), then consider starting with the following
values: `num_batch_threads` equal to the number of CPU cores; `max_batch_size`
to infinity; `batch_timeout_micros` to 0. Then experiment with
`batch_timeout_micros` values in the 1-10 millisecond (1000-10000 microsecond)
range, while keeping in mind that 0 may be the optimal value.

#### GPU: One Approach

If your model uses a GPU device for part or all of your its inference work,
consider the following approach:

1. Set `num_batch_threads` to the number of CPU cores.

2. Temporarily set `batch_timeout_micros` to infinity while you tune
`max_batch_size` to achieve the desired balance between throughput and average
latency. Consider values in the hundreds or thousands.

3. For online serving, tune `batch_timeout_micros` to rein in tail latency. The
idea is that batches normally get filled to `max_batch_size`, but occasionally
when there is a lapse in incoming requests, to avoid introducing a latency spike
it makes sense to process whatever's in the queue even if it represents an
underfull batch. The best value for `batch_timeout_micros` is typically a few
milliseconds, and depends on your context and goals. Zero is a value to
consider; it works well for some workloads. (For bulk processing jobs, choose a
large value, perhaps a few seconds, to ensure good throughput but not wait too
long for the final (and likely underfull) batch.)

## Servers with Multiple Models, Model Versions or Subtasks

Some server instances service multiple request types (e.g. multiple models, or
multiple versions of a model offered concurrently). In another scenario, a
single request gets broken down into sub-requests involving multiple distinct
servables (e.g. a recommender system might have a triggering model that decides
whether to formulate a recommendation, followed by a model that selects the
actual recommendation).

Generally speaking, using a separate batch scheduler for each kind of request
or sub-task does not work well if they share a common underlying compute
resource -- each scheduler would run its own threads that compete with the
others' threads to access the resource. It is better to have a single scheduler
with a single thread pool, that is aware of multiple distinct types of tasks
and is able to interleave batches of one kind of task with batches of another.

That is what `SharedBatchScheduler` does. It presents an abstraction of queues,
accepts requests to schedule a particular kind of task. Each batch contains
tasks of just one type, i.e. from one queue. The scheduler ensures fairness by
interleaving the different types of batches.

The queues implement the `BatchScheduler` API, so they can be used anywhere a
simpler (non-shared) scheduler can be used, including with `BatchingSession`.
Queues can be added and removed over time, which is useful e.g. for
transitioning to new model versions in environments in which clients specify a
specific version: while clients learn about the new version, the server will
have to process requests for both versions, and `SharedBatchScheduler` takes
care of interleaving batches of both kinds of requests.

## Mixed CPU/GPU/IO Workloads

Some models perform nontrivial CPU work, in addition to their main GPU work.
While the core matrix operations may run well on a GPU, peripheral operations
may take place on a CPU, e.g. embedding lookup, vocabulary lookup,
quantization/dequantization. Depending on how the GPU is managed, batching the
entire sequence of CPU and GPU steps as a unit can underutilize the GPU.

Non-GPU pre- and post-processing can be performed in the request threads, with
the batch scheduler used only for the GPU portion of the work.

Alternatively, the non-GPU work can be done in the batch threads, in the
callback the batch scheduler calls. To allow the callback to perform non-
batched work on tasks before a batch is fully formed, you can use
`StreamingBatchScheduler`. It is designed for servers that control latency very
precisely, and need fine control over each stage of the pipeline.

`StreamingBatchScheduler` will reject a task if the scheduler currently has
no capacity to process it. If you want to automatically retry tasks that are
rejected for that reason you can layer a `BatchSchedulerRetrier` on top of the
batch scheduler. There is a convenience function for creating a streaming
scheduler coupled with a retrier: `CreateRetryingStreamingBatchScheduler()'.

When splitting model inference logic into multiple distinct phases to optimize
latency or utilization, keep in mind that for a given request, every phase
should use the same version of the model. A good way to ensure this property is
to coordinate which ServableHandle object(s) get used in each phase, across the
threads.

Lastly, I/O-intensive phases of inference, e.g. lookups to disk or remote
servers, may benefit from batching to hide their latency. You can use two batch
scheduler instances: one to batch these lookups, and a separate one to batch
the GPU work.
