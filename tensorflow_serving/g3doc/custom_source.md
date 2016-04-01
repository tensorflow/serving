# Creating a module that discovers new servable paths

This document explains how to extend TensorFlow Serving to monitor different
storage systems to discover new (versions of) models or data to serve. In
particular, it covers how to create and use a module that monitors a storage
system path for the appearance of new sub-paths, where each sub-path represents
a new servable version to load. That kind of module is called a
`Source<StoragePath>`, because it emits objects of type `StoragePath` (typedefed
to `string`). It can be composed with a `SourceAdapter` that creates a servable
`Loader` from a given path that the source discovers.

## First, a note about generality

Using paths as handles to servable data is not required; it merely
illustrates one way to ingest servables into the system. Even if your
environment does not encapsulate servable data in paths, this document will
familiarize you with the key abstractions. You have the option to create
`Source<T>` and `SourceAdapter<T1, T2>` modules for types that suit your
environment (e.g. RPC or pub/sub messages, database records), or to simply
create a monolithic `Source<std::unique_ptr<Loader>>` that emits servable
loaders directly.

Of course, whatever kind of data your source emits (whether it is POSIX paths,
Google Cloud Storage paths, or RPC handles), there needs to be accompanying
module(s) that are able to load servables based on that. Such modules are called
`SourceAdapters`. Creating a custom one is described in the `custom_servable`
document. TensorFlow Serving comes with one for instantiating TensorFlow
sessions based on paths in file systems that TensorFlow supports. One can add
support for additional file systems to TensorFlow by extending the
`RandomAccessFile` abstraction (`tensorflow/core/public/env.h`).

This document focuses on creating a source that emits paths in a
TensorFlow-supported file system. It ends with a walk-through of how to use your
source in conjunction with pre-existing modules to serve TensorFlow models.

## Creating your Source

We have a reference implementation of a `Source<StoragePath>`, called
`FileSystemStoragePathSource` (at
`sources/storage_path/file_system_storage_path_source*`).
`FileSystemStoragePathSource` monitors a particular file system path, watches
for numerical sub-directories, and reports the latest of these as the version
it aspires to load. This document walks through the salient aspects of
`FileSystemStoragePathSource`. You may find it convenient to make a copy of
`FileSystemStoragePathSource` and then modify it to suit your needs.

First, `FileSystemStoragePathSource` implements the `Source<StoragePath>` API,
which is a specialization of the `Source<T>` API with `T` bound to
`StoragePath`. The API consists of a single method
`SetAspiredVersionsCallback()`, which supplies a closure the source can invoke
to communicate that it wants a particular set of servable versions to be
loaded.

`FileSystemStoragePathSource` uses the aspired-versions callback in a very
simple way: it periodically inspects the file system (doing an `ls`,
essentially), and if it finds one or more paths that look like servable
versions it determines which one is the latest version and invokes the callback
with a list of size one containing just that version. So, at any given time
`FileSystemStoragePathSource` requests at most one servable to be loaded, and
its implementation takes advantage of the idempotence of the callback to keep
itself stateless (there is no harm in invoking the callback repeatedly with the
same arguments).

`FileSystemStoragePathSource` has a static initialization factory (the
`Create()` method), which takes a configuration protocol message. The
configuration message includes details such as the base path to monitor and the
monitoring interval. It also includes the name of the servable stream to emit.
(Alternative approaches might extract the servable stream name from the base
path, to emit multiple servable streams based on observing a deeper directory
hierarchy; those variants are beyond the scope of the reference
implementation.)

The bulk of the implementation consists of a thread that periodically examines
the file system, along with some logic for identifying and sorting any
numerical sub-paths it discovers. The thread is launched inside
`SetAspiredVersionsCallback()` (not in `Create()`) because that is the point at
which the source should "start" and knows where to send aspired-version
requests.

## Using your Source to load TensorFlow sessions

You will likely want to use your new source module in conjunction with
`SessionBundleSourceAdapter`
(`servables/tensorflow/session_bundle_source_adapter*`), which will interpret
each path your source emits as a TensorFlow export, and convert each path to a
loader for a TensorFlow `SessionBundle` servable. You will likely plug the
`SessionBundle` adapter into a `DynamicManager`, which takes care of actually
loading and serving the servables. A good illustration of chaining these three
kinds of modules together to get a working server library is found in
`servables/tensorflow/simple_servers.cc`. Here is a walk-through of the main
code flow (with bad error handling; real code should be more careful):

First, create a manager:

~~~c++
std::unique_ptr<DynamicManager> manager = ...;
~~~

Then, create a `SessionBundle` source adapter and plug it into the manager:

~~~c++
std::unique_ptr<SessionBundleSourceAdapter> bundle_adapter;
SessionBundleSourceAdapterConfig config;
// ... populate 'config' with TensorFlow options.
TF_CHECK_OK(SessionBundleSourceAdapter::Create(config, &bundle_adapter));
ConnectSourceToTarget(bundle_adapter.get(), manager.get());
~~~

Lastly, create your path source and plug it into the `SessionBundle` adapter:

~~~c++
auto your_source = new YourPathSource(...);
ConnectSourceToTarget(your_source, bundle_adapter.get());
~~~

The `ConnectSourceToTarget()` function (defined in `core/target.h`) merely
invokes `SetAspiredVersionsCallback()` to connect a `Source<T>` to a
`Target<T>` (a `Target` is a module that catches aspired-version requests, i.e.
an adapter or manager).
