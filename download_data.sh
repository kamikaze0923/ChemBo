# Download datasets
# mkdir tmp_; cd tmp_
# curl -OL https://github.com/kevinid/molecule_generator/releases/download/1.0/datasets.tar.gz
# tar -xvzf datasets.tar.gz
# mv -v datasets/* ../datasets/
# cd ../
# rm -rf tmp_

# Download ZINC250k
curl https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv \
	--output ./datasets/zinc250k.csv
# Download models (Rexgen): only three dirs
# TODO
