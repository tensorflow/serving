from shutil import which
from subprocess import run
from pathlib import Path
from mkdocs.structure.files import File


doxygen_path = Path(which("doxygen"))
if not doxygen_path:
    raise FileNotFoundError("`doxygen` not found")

doxyfile_path = Path.cwd() / "Doxyfile"
print(doxyfile_path)
if not doxyfile_path.is_file():
    raise FileNotFoundError("`Doxyfile` not found")

src_dir = Path.cwd() / "tmp-cpp-api-docs/html"
src_dir.mkdir(exist_ok=True, parents=True)

dest_dir = Path("api/cpp/")


def on_pre_build(config):
    run(
        [doxygen_path, doxyfile_path],
        check=True,
    )


def on_files(files, config):
    for subdir in ("", "search"):
        for doc_file in (src_dir / subdir).iterdir():
            if not doc_file.is_file():
                continue
            f = File(
                path=doc_file.name,
                src_dir=src_dir / subdir,
                dest_dir=Path(config.site_dir) / dest_dir / subdir,
                use_directory_urls=config.use_directory_urls,
            )
            files.append(f)
    return files
