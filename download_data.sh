# Download datasets
mkdir tmp_; cd tmp_
curl -OL https://github.com/kevinid/molecule_generator/releases/download/1.0/datasets.tar.gz
tar -xvzf datasets.tar.gz
mv -v datasets/* ../datasets/
cd ../
rm -rf tmp_

# Download models (Rexgen): only three dirs
# TODO