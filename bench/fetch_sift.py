"""Download SIFT1M (Jegou et al., INRIA TEXMEX corpus, 161 MB) into data/sift/."""
import pathlib
import tarfile
import urllib.request

data = pathlib.Path(__file__).resolve().parent.parent / "data"
data.mkdir(exist_ok=True)
if not (data / "sift" / "sift_base.fvecs").exists():
    tar = data / "sift.tar.gz"
    if not tar.exists():
        print("downloading sift.tar.gz ...")
        urllib.request.urlretrieve("ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz", tar)
    with tarfile.open(tar) as t:
        t.extractall(data, filter="data")
print("SIFT1M ready in", data / "sift")
