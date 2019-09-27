""" TensorFlow Http Archive

Modified http_archive that allows us to override the TensorFlow commit that is
downloaded by setting an environment variable. This override is to be used for
testing purposes.

Add the following to your Bazel build command in order to override the
TensorFlow revision.

build: --action_env TF_REVISION="<git commit hash>"

  * `TF_REVISION`: tensorflow revision override (git commit hash)
"""

_TF_REVISION = "TF_REVISION"

def _tensorflow_http_archive(ctx):
    git_commit = ctx.attr.git_commit
    sha256 = ctx.attr.sha256
    patch_file = ctx.attr.patch_file

    override_git_commit = ctx.os.environ.get(_TF_REVISION)
    if override_git_commit:
        sha256 = ""
        git_commit = override_git_commit

    strip_prefix = "tensorflow-%s" % git_commit
    urls = [
        "https://mirror.bazel.build/github.com/tensorflow/tensorflow/archive/%s.tar.gz" % git_commit,
        "https://github.com/tensorflow/tensorflow/archive/%s.tar.gz" % git_commit,
    ]
    ctx.download_and_extract(
        urls,
        "",
        sha256,
        "",
        strip_prefix,
    )
    if patch_file != None:
      cmd = ["patch", "-p1", "-d", ctx.path("."), "-i", ctx.path(patch_file)]
      result = ctx.execute(cmd, timeout = 60)
      if result.return_code != 0:
        fail(("Non-zero return code({1}) when executing '{0}':\n" +
              "Stdout: {2}\n" +
              "Stderr: {3}").format(
                  " ".join(cmd), result.return_code, result.stdout,
                  result.stderr))

tensorflow_http_archive = repository_rule(
    implementation = _tensorflow_http_archive,
    attrs = {
        "git_commit": attr.string(mandatory = True),
        "sha256": attr.string(mandatory = True),
        "patch_file": attr.label(),
    },
)
