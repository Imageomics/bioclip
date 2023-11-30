!#/usr/bin/env bash

echo "Getting eol data"
mkdir -p data/eol
pushd data/eol

echo "Getting provider_ids.csv"
wget https://eol.org/data/provider_ids.csv.gz
gunzip provider_ids.csv.gz

echo "Getting dh21/taxon.tab"
wget https://opendata.eol.org/dataset/0a023d9a-f8c3-4c80-a8d1-1702475cda18/resource/00adb47b-57ed-4f6b-8f66-83bfdb5120e8/download/dh21.zip
unzip -d dh21 dh21.zip
rm dh21.zip

echo "Getting vernacularnames.csv"
wget https://opendata.eol.org/dataset/d40073b7-1da7-428a-a404-9233f5153147/resource/bf63dc02-a7ff-416e-b252-e77e23c29d5c/download/vernacularnames.csv

echo "Getting trait_bank"
wget https://editors.eol.org/other_files/SDR/traits_all.zip
unzip traits_all.zip

popd

echo "Getting inaturalist data"
mkdir -p data/inat
pushd data/inat

echo "Getting .dwca files"
wget https://www.inaturalist.org/taxa/inaturalist-taxonomy.dwca.zip
unzip -d dwca inaturalist-taxonomy.dwca.zip
rm inaturalist-taxonomy.dwca.zip

popd