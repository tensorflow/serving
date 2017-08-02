# Creating a new kind of servable

This document explains how to extend TensorFlow Serving with a new kind of
servable. The most prominent servable type is `SavedModelBundle`, but it can be
useful to define other kinds of servables, to serve data that goes along with
your model. Examples include: a vocabulary lookup table, feature transformation
logic. Any C++ class can be a servable, e.g. `int`, `std::map<string, int>` or
any class defined in your binary -- let us call it `YourServable`.

## Defining a `Loader` and `SourceAdapter` for `YourServable`

To enable TensorFlow Serving to manage and serve `YourServable`, you need to
define two things:

  1. A `Loader` class that loads, provides access to, and unloads an instance
  of `YourServable`.

  2. A `SourceAdapter` that instantiates loaders from some underlying data
  format e.g. file-system paths. As an alternative to a `SourceAdapter`, you
  could write a complete `Source`. However, since the `SourceAdapter`
  approach is more common and more modular, we focus on it here.

The `Loader` abstraction is defined in `core/loader.h`. It requires you to
define methods for loading, accessing and unloading your type of servable. The
data from which the servable is loaded can come from anywhere, but it is common
for it to come from a storage-system path. Let us assume that is the case for
`YourServable`. Let us further assume you already have a `Source<StoragePath>`
that you are happy with (if not, see the [Custom Source](custom_source.md)
document).

In addition to your `Loader`, you will need to define a `SourceAdapter` that
instantiates a `Loader` from a given storage path. Most simple use-cases can
specify the two objects concisely via the `SimpleLoaderSourceAdapter` class
(in `core/simple_loader.h`). Advanced use-cases may opt to specify `Loader` and
`SourceAdapter` classes separately using the lower-level APIs, e.g. if the
`SourceAdapter` needs to retain some state, and/or if state needs to be shared
among `Loader` instances.

There is a reference implementation of a simple hashmap servable that uses
`SimpleLoaderSourceAdapter` in `servables/hashmap/hashmap_source_adapter.cc`.
You may find it convenient to make a copy of `HashmapSourceAdapter` and then
modify it to suit your needs.

The implementation of `HashmapSourceAdapter` has two parts:

  1. The logic to load a hashmap from a file, in `LoadHashmapFromFile()`.

  2. The use of `SimpleLoaderSourceAdapter` to define a `SourceAdapter` that
  emits hashmap loaders based on `LoadHashmapFromFile()`. The new
  `SourceAdapter` can be instantiated from a configuration protocol message of
  type `HashmapSourceAdapterConfig`. Currently, the configuration message
  contains just the file format, and for the purpose of the reference
  implementation just a single simple format is supported.

  Note the call to `Detach()` in the destructor. This call is required to avoid
  races between tearing down state and any ongoing invocations of the Creator
  lambda in other threads. (Even though this simple source adapter doesn't have
  any state, the base class nevertheless enforces that Detach() gets called.)

## Arranging for `YourServable` objects to be loaded in a manager

Here is how to hook your new `SourceAdapter` for `YourServable` loaders to a
basic source of storage paths, and a manager (with bad error handling; real code
should be more careful):

First, create a manager:

~~~c++
std::unique_ptr<AspiredVersionsManager> manager = ...;
~~~

Then, create a `YourServable` source adapter and plug it into the manager:

~~~c++
auto your_adapter = new YourServableSourceAdapter(...);
ConnectSourceToTarget(your_adapter, manager.get());
~~~

Lastly, create a simple path source and plug it into your adapter:

~~~c++
std::unique_ptr<FileSystemStoragePathSource> path_source;
// Here are some FileSystemStoragePathSource config settings that ought to get
// it working, but for details please see its documentation.
FileSystemStoragePathSourceConfig config;
// We just have a single servable stream. Call it "default".
config.set_servable_name("default");
config.set_base_path(FLAGS::base_path /* base path for our servable files */);
config.set_file_system_poll_wait_seconds(1);
TF_CHECK_OK(FileSystemStoragePathSource::Create(config, &path_source));
ConnectSourceToTarget(path_source.get(), your_adapter.get());
~~~

## Accessing loaded `YourServable` objects

Here is how to get a handle to a loaded `YourServable`, and use it:

~~~c++
auto handle_request = serving::ServableRequest::Latest("default");
ServableHandle<YourServable*> servable;
Status status = manager->GetServableHandle(handle_request, &servable);
if (!status.ok()) {
  LOG(INFO) << "Zero versions of 'default' servable have been loaded so far";
  return;
}
// Use the servable.
(*servable)->SomeYourServableMethod();
~~~

## Advanced: Arranging for multiple servable instances to share state

SourceAdapters can house state that is shared among multiple emitted servables.
For example:

  * A shared thread pool or other resource that multiple servables use.

  * A shared read-only data structure that multiple servables use, to avoid the
  time and space overhead of replicating the data structure in each servable
  instance.

Shared state whose initialization time and size is negligible (e.g. thread
pools) can be created eagerly by the SourceAdapter, which then embeds a pointer
to it in each emitted servable loader. Creation of expensive or large shared
state should be deferred to the first applicable Loader::Load() call, i.e.
governed by the manager. Symmetrically, the Loader::Unload() call to the final
servable using the expensive/large shared state should tear it down.
