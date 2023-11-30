---
License: cc0-1.0
language:
- en
- la
pretty_name: TreeOfLife-10M
task_categories:
- image-classification
- zero-shot-classification
tags:
- biology
- images
- animals
- evolutionary biology
- CV
- multimodal
- clip
- biology
- species
- taxonomy
- knowledge-guided
- imbalanced
size_categories: 10M<n<100M
---

# Dataset Card for TreeOfLife-10M

## Dataset Description

<!-- - **Homepage:** -->
- **Repository:** [Imageomics/bioclip](https://github.com/Imageomics/bioclip)
- **Paper:** BioCLIP: A Vision Foundation Model for the Tree of Life ([arXiv](https://doi.org/10.48550/arXiv.2311.18803))
<!-- - **Leaderboard:** -->

### Dataset Summary

With over 10 million images covering 454 thousand taxa in the tree of life, TreeOfLife-10M is the largest-to-date ML-ready dataset of images of biological organisms paired with their associated taxonomic labels. It expands on the foundation established by existing high-quality datasets, such as iNat21 and BIOSCAN-1M, by further incorporating newly curated images from the Encyclopedia of Life (eol.org), which supplies most of TreeOfLife-10M’s data diversity. Every image in TreeOfLife-10M is labeled to the most specific taxonomic level possible, as well as higher taxonomic ranks in the tree of life (see [Text Types](#text-types) for examples of taxonomic ranks and labels). TreeOfLife-10M was generated for the purpose of training [BioCLIP](https://huggingface.co/imageomics/bioclip) and future biology foundation models.  

<!--This dataset card aims to be a base template for new datasets. It has been generated using [this raw template](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/datasetcard_template.md?plain=1). And further altered to suit Imageomics Institute needs. -->

|![treemap from phyla down to family](https://huggingface.co/datasets/imageomics/treeoflife-10m/resolve/main/visuals/phyla_ToL_tree.png)|
|:--|
|**Figure 1.** Treemap from phyla down to family for TreeOfLife-10M. Interactive version available in [`visuals`](https://huggingface.co/datasets/imageomics/TreeOfLife-10M/tree/main/visuals) folder.|


### Supported Tasks and Leaderboards

Image Classification, Zero-shot and few-shot Classification.


### Languages
English, Latin

## Dataset Contents

```
/dataset/
    EOL/
        image_set_01.tar.gz
        image_set_02.tar.gz
        ...
        image_set_63.tar.gz
    metadata/
        catalog.csv
        species_level_taxonomy_chains.csv
        taxon.tab
        licenses.csv
    visuals/
        kingodm_ToL_tree.html
        kingdom_ToL_tree.pdf
        phyla_ToL_tree.html
        phyla_ToL_tree.pdf
        phyla_ToL_tree.png
```

Each `image_set` is approximately 30GB and contains 100 thousand images, each named `<treeoflife_id>.jpg`. 
We cannot reproduce the `iNat21` data, but after downloading it and BIOSCAN-1M, one can follow the directions from step 6 of [docs/imageomics/treeoflife10m.md](https://github.com/Imageomics/bioclip/blob/main/docs/imageomics/treeoflife10m.md) in the BioCLIP GitHub repo to combine them with the EOL data into the proper webdataset structure. This will produce a collection of files named `shard-######.tar` in a `train`, `val`, and `train_small` folder with which to work.

Inside each shard is a collection of images (named `<treeoflife_id>.jpg`), for which each has the following files:
```
<treeoflife_id>.com.txt
<treeoflife_id>.common_name.txt
<treeoflife_id>.jpg
<treeoflife_id>.sci.txt
<treeoflife_id>.sci_com.txt
<treeoflife_id>.scientific_name.txt
<treeoflife_id>.taxon.txt
<treeoflife_id>.taxonTag.txt
<treeoflife_id>.taxonTag_com.txt
<treeoflife_id>.taxon_com.txt
<treeoflife_id>.taxonomic_name.txt

```


### Data Instances


This dataset is a collection of images with associated text. The text matched to images contains both [Linnaean taxonomy](https://www.britannica.com/science/taxonomy/The-objectives-of-biological-classification) (kingdom through species) for the particular subject of the image and its common (or vernacular) name where available. There are 8,455,243 images with full taxonomic labels.


### Data Fields

#### Metadata Files

`catalog.csv`: contains the following metadata associated with each image in the dataset
  - `split`: indicates which data split the image belongs to (`train`, `val`, or `train_small`), `train_small` is a duplicated subset of `train` and thus should not be included when analyzing overall stats of the dataset.
  - `treeoflife_id`: unique identifier for the image in the dataset.
  - `eol_content_id`: unique identifier within EOL database for images sourced from [EOL](https://eol.org). Note that EOL content IDs are not stable.
  - `eol_page_id`: identifier of page from which images from EOL are sourced. Note that an image's association to a particular page ID may change with updates to the EOL (or image provider's) hierarchy. However, EOL taxon page IDs are stable.
  - `bioscan_part`: indicates to which of the 113 data chunks of [BIOSCAN-1M](https://github.com/zahrag/BIOSCAN-1M#-iv-rgb-images) each image belongs. Note that there are 10K images per chunk and 8,313 in chunk #113.
  - `bioscan_filename`: unique identifier within BIOSCAN-1M dataset for images sourced from [BIOSCAN-1M](https://github.com/zahrag/BIOSCAN-1M).
  - `inat21_filename`: unique identifier within iNat21 dataset for images sourced from [iNat21](https://github.com/visipedia/inat_comp/blob/master/2021/README.md). 
<!-- (`file_name` given in `images` of the [`train.json`](https://github.com/visipedia/inat_comp/tree/master/2021#annotation-format) `file_name` = "train/#####_Kingdom_Phylum_..._Genus_species/STRING(uuid?).jpg"). `inat21_filename` is the end of the `file_name` string. The taxa are the `cls_name`, and the number is the `cls_num` (leading 0 may be lost here).-->
  - `inat21_cls_name`: `<Kingdom>_<Phylum>_<Class>_<Order>_<Family>_<Genus>_<species>` as labeled by iNaturalist.
  - `inat21_cls_num`: Number assigned by iNat21 to the given species (unique identifier for that species within iNat21 dataset).
  The remaining terms describe the _Linnaean taxonomy_ of the subject of the image; they are sourced as described in [Annotation Process, below](#annotation-process).
  - `kingdom`: kingdom to which the subject of the image belongs (`Animalia`, `Plantae`, `Fungi`, `Chromista`, `Protozoa`, `Bacteria`, `Viridiplantae`, `Protista`, `Orthornavirae`, `Bamfordvirae`, `Archaea`, or `Shotokuvirae`). Note: this large number of kingdoms are considered in recognition of the fact that there is not agreement on merging them.
  - `phylum`: phylum to which the subject of the image belongs.
  - `class`: class to which the subject of the image belongs.
  - `order`: order to which the subject of the image belongs.
  - `family`: family to which the subject of the image belongs.
  - `genus`: genus to which the subject of the image belongs.
  - `species`: species to which the subject of the image belongs.
  - `common`: common name associated with the subject of the image where available. Otherwise, this is the scientific name (`genus-species`), else whatever subset of the taxonomic hierarchy is available (eg., `kingdom-phylum-class-order` or `kingdom-phylum-class-order-family`). All images have a non-null entry for this column.

Note that the `species` column occasionally has entries such as "sp. ___(get ex)" with some string following. This seems to be used to indicate the species is unknown, but various specimens/images are known to be the same species. Additionally, for `species` values containing an `x` between names, this is indicative of a hybrid that is a cross of the two species listed on either side of the `x`.

##### Text Types

| Text Type | Example |
| ---- | -------- |
| Common | black-billed magpie |
| Scientific | _Pica hudsonia_ |
| Taxonomic | _Animalia Chordata Aves Passeriformes Corvidae Pica hudsonia_ |


`species_level_taxonomy_chains.csv`: CSV with the ITIS taxonomic hierarchy, indicated as follows:
  - `hierarchy_string_tsn`: string of Taxonomic Serial Numbers (TSN)* for the names of the ranks provided from highest to lowest, connected by dashes (eg., `202422-846491-660046-846497-846508-846553-954935-5549-5550`).
  - `hierarchy_string_names`: string of the names of the ranks provided from highest to lowest, connected by arrows (eg., `Plantae->Biliphyta->Rhodophyta->Cyanidiophytina->Cyanidiophyceae->Cyanidiales->Cyanidiaceae->Cyanidium->Cyanidium caldarium`).
  - `terminal_tsn`: Taxonomic Serial Number (TSN)* of designated species (eg., `5550`).
  - `terminal_scientific_name`: scientific name (`<Genus> <species>`) of subject.
  - `terminal_vernacular`: vernacular or common name(s) of the subject, multiple names are separated by commas (eg., `rockskipper`, `Highland Small Rice Rat, Páramo Colilargo`).
  - `terminal_vernacular_lang`: language(s) of the vernacular name(s) provided; when there are multiple names, language is listed for each, separated by commas (eg., `English`, `English, English`, respectively for the vernacular name examples above).
  - `hierarchy_string_ranks`: string of ranks provided from highest to lowest, connected by arrows (eg., `Kingdom->Subkingdom->Phylum->Subphylum->Class->Order->Family->Genus->Species`).
  The remaining columns consist of the hierarchy string ranks describing the Linnaean taxonomy of the subject (as defined above), with `<Genus> <species>` filled in the `Species` column.

*ITIS assigns a Taxonomic Serial Number (TSN) to each taxonomic rank; this is a stable and unique ID.


`taxon.tab`: Tab-delimited file with taxonomic information for EOL images based on EOL page IDs. 
  - `taxonID`: unique identifier for the file.
  - `source`: often `<source>:<id>` where the source corresponds to the domain of the `furtherInformationURL`. The ID likely corresponds to an ID at the source.
  - `furtherInformationURL`: URL with more information on the indicated taxon.
  - `acceptedNameUsageID`: `taxonID` for the name accepted to represent this entry. Less than a third of these are non-null
  - `parentNameUsageID`: `taxonID` of taxonomic rank above the indicated `taxonRank` in the hierarchy (eg., the `taxonID` of the genus `Atadinus` for the `Atadinus fallax (Boiss.) Hauenschild` entry).
  - `scientificName`: scientific name associated with the EOL page (`<canonicalName> <authority>`, authority as available).
  - `taxonRank`: lowest rank of the taxonomic tree indicated (eg., `genus` or `species`), occasionally not indicated, even for accepted names.
  - `taxonomicStatus`: whether the name is accepted by EOL or not (`accepted` or `not accepted`, correspond to existence of non-null `eolID` or `acceptedNameUsageID` entry, respectively).
  - `datasetID`: generally corresponds to the source identified in `source` column.
  - `canonicalName`: the name(s) associate with the `taxonRank` (eg., `<Genus> <species>` for species).
  - `authority`: usually name of person who assigned the name, with the year as available.
  - `eolID`: the EOL page ID (only non-null when `taxonomicStatus` is accepted by EOL).
  - `Landmark`: numeric values, meaning unknown, mostly null.
  - `higherClassification`: labeling in the EOL Dynamic Hierarchy above the `taxonRank` (eg., `Life|Cellular Organisms|Eukaryota|Opisthokonta|Metazoa|Bilateria|Protostomia|Ecdysozoa|Arthropoda|Pancrustacea|Hexapoda|Insecta|Pterygota|Neoptera|Endopterygota|Coleoptera|Adephaga|Carabidae|Paussus`).



`licenses.csv`: File with license, source, and copyright holder associated to each image from EOL listed in `catalog.csv`; `treeoflife_id` is the shared unique identifier to link the two files. Columns are
  - `treeoflife_id`, `eol_content_id`, and `eol_page_id` are as defined above.
  - `md5`: MD5 hash of the image.
  - `medium_source_url`: URL pointing to source of image.
  - `eol_full_size_copy_url`: URL to access the full-sized image; this is the URL from which the image was downloaded for this dataset (see [Initial Data Collection and Normalization](#initial-data-collection-and-normalization) for more information on this process).
  - `license_name`: name of license attached to the image (eg., `cc-by`).
  - `copyright_owner`: copyright holder for the image, filled with `not provided` if no copyright owner was provided. 
  - `license_link`: URL to the listed license, left null in the case that `License Name` is `No known copyright restrictions`.
  - `title`: title provided for the image, filled with `not provided` if no title was provided.


### Data Splits

As noted above, the `split` column of `catalog.csv` indicates to which split each image belongs. Note that `train_small` is a 1M-image, uniformly sampled, subset of `train` used for fine-tuned ablation training and all entries with this label are also listed with the `train` label. The `val` label is applied to images used for validation. 

10 biologically-relevant datasets were used for various tests of [BioCLIP](https://huggingface.co/imageomics/bioclip) (which was trained on this dataset), they are described (briefly) and linked to below.

#### Test Sets

- [Meta-Album](https://paperswithcode.com/dataset/meta-album): Specifically, we used the Plankton, Insects, Insects 2, PlantNet, Fungi, PlantVillage, Medicinal Leaf, and PlantDoc datasets from Set-0 through Set-2 (Set-3 was still not released as of our publication/evaluation (Nov. 2023).
 - [Birds 525](https://www.kaggle.com/datasets/gpiosenka/100-bird-species): We evaluated on the 2,625 test images provided with the dataset.
 - [Rare Species](https://huggingface.co/datasets/imageomics/rare-species): A new dataset we curated for the purpose of testing this model and to contribute to the ML for Conservation community. It consists of 400 species labeled Near Threatened through Extinct in the Wild by the [IUCN Red List](https://www.iucnredlist.org/), with 30 images per species. For more information, see our dataset, [Rare Species](https://huggingface.co/datasets/imageomics/rare-species).

For more information about the contents of these datasets, see Table 2 and associated sections of [our paper](https://doi.org/10.48550/arXiv.2311.18803).

## Dataset Creation

### Curation Rationale

Previously, the largest ML-ready biology image dataset was [iNat21](https://github.com/visipedia/inat_comp/tree/master/2021), which consists of 2.7M images of 10K species. This is significant breadth when comparing to popular general-domain datasets, such as [ImageNet-1K](https://huggingface.co/datasets/imagenet-1k); 10K species are rather limited when considering the vast scope of biology. For context, in 2022, [The International Union for Conservation of Nature (IUCN)](https://www.iucnredlist.org/) reported over 2M total described species, with over 10K distinct species of birds and reptiles alone. Thus, the lesser species diversity of iNat21 limits its potential for pre-training a foundation model for the entire tree of life. 

With this focus on species diversity and the need for high-quality images of biological organisms, we looked to the [Encyclopedia of Life Project (EOL)](https://eol.org/). EOL is an image aggregator that collaborates with a variety of institutions to source and label millions of images. After downloading 6.6M images from EOL, we were able to expand our dataset to cover an additional 440K taxa. 

Insects (of the class Insecta with 1M+ species), birds (of the class Aves with 10K+ species) and reptiles (of the class Reptilia with 10K+ species) are examples of highly diverse subtrees with many more species than other taxonomic classes. This imbalance among subtrees in the tree of life present challenges in training a foundation model that can recognize extremely fine-grained visual representations of these classes. To help address this challenge for insects, we incorporated [BIOSCAN-1M](https://zenodo.org/doi/10.5281/zenodo.8030064), a recent dataset of 1M expert-labeled lab images of insects, covering 494 different families. The added variety of lab images, rather than in situ images (as in iNat21), further diversifies the _image_ distribution of TreeOfLife-10M.

Overall, this dataset contains approximately 454K unique taxonomic labels of the more than 2M recorded by [IUCN](iucnredlist.org) in 2022. To the best of our knowledge, this is still the most diverse and largest such ML-ready dataset available, hence our curation. 


### Source Data

[iNat21 data](https://github.com/visipedia/inat_comp/tree/master/2021#data) was downloaded, unzipped, and our compilation scripts pointed to the training split. As per their [terms of use](https://github.com/visipedia/inat_comp/tree/master/2021#terms-of-use), the data is catalogued, but not reproduced, here.

[BIOSCAN-1M](https://zenodo.org/doi/10.5281/zenodo.8030064): Collection of insect images hand-labeled by experts.

[EOL](https://eol.org/): Biological image aggregator.

#### Initial Data Collection and Normalization

[iNat21 training data](https://github.com/visipedia/inat_comp/tree/master/2021#data) and [BIOSCAN-1M data](https://zenodo.org/doi/10.5281/zenodo.8030064) were downloaded and assigned `treeoflife_id`s for unique identification within the TreeOfLife-10M dataset. The iNat21 training data is formatted into a webdataset format prior to `treeoflife_id` assignments, since this is also used for a comparison to [BioCLIP](https://huggingface.co/imageomics/bioclip) as trained on the full TreeOfLife-10M dataset. For more detailed information on this process, please see [How to Create TreeOfLife-10M](https://github.com/Imageomics/bioclip/tree/main/docs/imageomics/treeoflife10m.md#how-to-create-treeoflife-10m) in the BioCLIP GitHub repo. 

First, media manifest data was sourced from EOL using [this script](https://github.com/Imageomics/bioclip/blob/main/scripts/get_media_manifest.py). The media manifest includes EOL content and page IDs from which to connect the taxonomic information, along with source URLs and licensing information. The `EOL Full-Size Copy URL` was then used to download all the images, naming each `<eol_content_id>_<eol_page_id>_eol_full-size-copy.jpg` for reference back to the media manifest. [Scripts](https://github.com/Imageomics/bioclip/tree/main/scripts/evobio10m) to perform these downloads and [instructions](https://github.com/Imageomics/bioclip/blob/main/docs/imageomics/treeoflife10m.md) can be found in the [BioCLIP GitHub repository](https://github.com/Imageomics/bioclip).

See [below](#Annotation-Process) for details of annotation following data collection.

Species selected for the Rare Species dataset were removed from this dataset (see [Initial Data Collection and Normalization of Rare Species](https://huggingface.co/datasets/imageomics/rare-species#initial-data-collection-and-normalization)).


### Annotations

#### Annotation Process

Annotations were primarily sourced from image source providers. 

For iNat21 and BIOSCAN-1M images, the labels provided by those sources were used.
- iNat21: iNaturalist English vernacular names and taxa were used.
- BIOSCAN-1M: Linnaean taxonomic rankings were applied as labeled in the [BIOSCAN-1M dataset](https://zenodo.org/doi/10.5281/zenodo.8030064), which is all hand-labeled by experts. Note that the dataset provides other ranks (not considered in the 7-rank Linnaean taxonomy), such as tribe, which were not included in this dataset.

For images from EOL, the scientific name (`genus-species`) was used to look up the higher-order taxa from the following sources as listed: BIOSCAN-1M metadata, EOL aggregate datasets (described below), then match this against the ITIS hierarchy for the higher-order taxa standardization. A small number of these are [homonyms](https://en.wikipedia.org/wiki/Homonym_(biology)), for which a list was generated to ensure proper matching of higher-order taxa (manual homonym resolution is in class `NameUpgrader` in the [naming script](https://github.com/Imageomics/bioclip/blob/main/src/imageomics/naming.py)). After these resources were exhausted, any remaining unresolved taxa were fed through the [Global Names Resolver (GNR) API](https://resolver.globalnames.org/api). Despite our efforts, we discovered after training that some hemihomonyms were mislabeled at higher-level taxa (family up to kingdom). This impacts approximately 0.1-0.2% of our data. We are in the process of developing a more robust solution to taxonomic labeling which will also account for re-naming (as is currently in process for many bird species). We intend to release a patch alongside the solution.

This process allowed us to reach full taxa labels for 84% of images. To put this in perspective, 10% of images in TreeOfLife-10M are only labeled to the `family` level (no `genus-species` designations) as part of BIOSCAN-1M, so this places a cap on the taxa coverage. Taxonomic ranking also is not entirely standardized and agreed-upon throughout the biology community, so most gaps are more indicative of lack of consensus on label than missing information.


#### Who are the annotators?

Samuel Stevens, Jiaman Wu, Matthew J. Thompson, and Elizabeth G. Campolongo

### Personal and Sensitive Information
N/A

## Considerations for Using the Data

### Social Impact of Dataset

The hope is that this dataset could be helpful in conservation efforts or biodiversity research.

### Discussion of Biases and Other Known Limitations

This dataset is imbalanced in its representation of various species with the greatest representation available for those in the phyla _Arthropoda_, _Tracheophyta_, and _Chordata_ (see our [interactive treemap from phylum to family](https://huggingface.co/imageomics/treeoflife-10m/raw/main/phyla_ToL_tree.html) for further details of this distribution). This class imbalance is both a result of availability of images and actual variance in class diversity. Additionally, as noted above, there are 2M+ estimated species according to [IUCN](iucnredlist.org), so overall taxonomic coverage is still limited (though it far surpasses the species diversity of other well-known animal datasets).

Not all data is labeled to the species level, and some entries are more or less precise. For instance, the `species` column occasionally has entries such as "sp. ___(get ex)" with some string following. This seems to be used to indicate the species is unknown, but various specimens/images are known to be the same species. Additionally, for `species` values containing an `x` between names, this is indicative of a hybrid that is a cross of the two species listed on either side of the `x`. Due to the additional information provided about the higher order taxa, these labeling anomalies still present valuable information providing links between these classes.


As stated above, 84% of images have full taxa labels. However, due to the incomplete standardization and agreement on the taxonomic hierarchy throughout the biology community, most gaps are more indicative of lack of consensus on label than missing information. 

Note that BIOSCAN-1M’s label granularity may still be limited for insects, as 98.6% of BIOSCAN-1M’s images are labeled to the family level but only 22.5% and 7.5% of the images have genus or species indicated, respectively. Lack of label granularity is an inherent challenge.


## Additional Information

### Dataset Curators

Samuel Stevens, Jiaman Wu, Matthew J. Thompson, and Elizabeth G. Campolongo

### Licensing Information

The data (images and text) contain a variety of licensing restrictions mostly within the CC family. Each image and text in this dataset is provided under the least restrictive terms allowed by its licensing requirements as provided to us (i.e, we impose no additional restrictions past those specified by licenses in the license file).

Please see the [iNat21 terms of use](https://github.com/visipedia/inat_comp/tree/master/2021#terms-of-use) for full information on use of their images.

All BIOSCAN-1M images are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

EOL images contain a variety of licenses ranging from [CC0](https://creativecommons.org/publicdomain/zero/1.0/) to [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/).
For license and citation information by image, see our [license file](https://huggingface.co/datasets/imageomics/treeoflife-10m/blob/main/metadata/licenses.csv).

**Note**: Due to licensing restrictions discovered after training, approximately 30K of the images used to train BioCLIP (about 0.3%) cannot be republished here and links to original content are no longer available. Overall, 14 families that were included in training BioCLIP are not republished in this dataset, a loss of 0.38% of the taxa diversity.

This dataset (the compilation) has been marked as dedicated to the public domain by applying the [CC0 Public Domain Waiver](https://creativecommons.org/publicdomain/zero/1.0/). However, images may be licensed under different terms (as noted above).


### Citation Information

```
@dataset{treeoflife_10m,
  author = {Samuel Stevens and Jiaman Wu and Matthew J Thompson and Elizabeth G Campolongo and Chan Hee Song and David Edward Carlyn and Li Dong and Wasila M Dahdul and Charles Stewart and Tanya Berger-Wolf and Wei-Lun Chao and Yu Su},
  title = {TreeOfLife-10M},
  year = {2023},
  url = {https://huggingface.co/datasets/imageomics/TreeOfLife-10M},
  doi = {<doi once generated>},
  publisher = {Hugging Face}
}
```
Please also cite our paper: 
```
@article{stevens2023bioclip,
  title = {BIOCLIP: A Vision Foundation Model for the Tree of Life}, 
  author = {Samuel Stevens and Jiaman Wu and Matthew J Thompson and Elizabeth G Campolongo and Chan Hee Song and David Edward Carlyn and Li Dong and Wasila M Dahdul and Charles Stewart and Tanya Berger-Wolf and Wei-Lun Chao and Yu Su},
  year = {2023},
  eprint = {2311.18803},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV}}
```


Please be sure to also cite the original data sources and all constituent parts as appropriate.


- iNat21:
```
@misc{inat2021,
  author={Van Horn, Grant and Mac Aodha, Oisin},
  title={iNat Challenge 2021 - FGVC8},
  publisher={Kaggle},
  year={2021},
  url={https://kaggle.com/competitions/inaturalist-2021}
}
```

- BIOSCAN-1M:
```
@inproceedings{gharaee2023step,
    title={A Step Towards Worldwide Biodiversity Assessment: The {BIOSCAN-1M} Insect Dataset},
    booktitle = {Advances in Neural Information Processing Systems ({NeurIPS}) Datasets \& Benchmarks Track},
    author={Gharaee, Z. and Gong, Z. and Pellegrino, N. and Zarubiieva, I. and Haurum, J. B. and Lowe, S. C. and McKeown, J. T. A. and Ho, C. Y. and McLeod, J. and Wei, Y. C. and Agda, J. and Ratnasingham, S. and Steinke, D. and Chang, A. X. and Taylor, G. W. and Fieguth, P.},
    year={2023},
}
```
- EOL: Encyclopedia of Life. Available from http://eol.org. Accessed 29 July 2023.

For license and citation information by image, see our [license file](https://huggingface.co/datasets/imageomics/treeoflife-10m/blob/main/metadata/licenses.csv).


- ITIS: Retrieved July, 20 2023, from the Integrated Taxonomic Information System (ITIS) on-line database, www.itis.gov, CC0
https://doi.org/10.5066/F7KH0KBK


### Contributions

The [Imageomics Institute](https://imageomics.org) is funded by the US National Science Foundation's Harnessing the Data Revolution (HDR) program under [Award #2118240](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2118240) (Imageomics: A New Frontier of Biological Information Powered by Knowledge-Guided Machine Learning). Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.
