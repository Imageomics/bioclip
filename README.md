# BioCLIP [![DOI](https://zenodo.org/badge/725653485.svg)](https://zenodo.org/doi/10.5281/zenodo.10895870)


This is the repository for the [BioCLIP model](https://huggingface.co/imageomics/bioclip) and the [TreeOfLife-10M dataset](https://huggingface.co/datasets/imageomics/TreeOfLife-10M). It contains the code used for training and the evaluation of BioCLIP (testing and visualizing embeddings). Additionally, we include a collection of scripts for forming, evaluating, and visualizing the data used for TreeOfLife-10M and the [Rare Species benchmark](https://huggingface.co/datasets/imageomics/rare-species) we created alongside it. The BioCLIP website is hosted from the `gh-pages` branch of this repository.

[Paper](https://arxiv.org/abs/2311.18803) | [Model](https://huggingface.co/imageomics/bioclip) | [Data](https://huggingface.co/datasets/imageomics/TreeOfLife-10M) | [Demo](https://huggingface.co/spaces/imageomics/bioclip-demo)
---

BioCLIP is a CLIP model trained on a new 10M-image dataset of biological organisms with fine-grained taxonomic labels.
BioCLIP outperforms general domain baselines on a wide spread of biology-related tasks, including zero-shot and few-shot classification.

## Table of Contents

1. [Model](#model)
2. [Data](#data)
3. [Paper, website, and docs](#paper)
4. [Citation](#citation)

## Model

The BioCLIP model is a ViT-B/16 pre-trained with the CLIP objective.
Both the ViT and the (small) autoregressive text encoder are available to download on [Hugging Face](https://huggingface.co/imageomics/bioclip).

The only dependency is the [`open_clip`](https://github.com/mlfoundations/open_clip) package.

See the [`examples/`](https://huggingface.co/imageomics/bioclip/tree/main/examples) directory on the [Hugging Face model repo](https://huggingface.co/imageomics/bioclip) for an example implementation.
You can also use the [pybioclip](https://github.com/Imageomics/pybioclip) package or the [BioCLIP demo](https://huggingface.co/spaces/imageomics/bioclip-demo) on Hugging Face.

## Data

BioCLIP was trained on TreeOfLife-10M (ToL-10M).
The data is a combination of [iNat21](https://github.com/visipedia/inat_comp/tree/master/2021), [BIOSCAN-1M](https://github.com/zahrag/BIOSCAN-1M), and data we collected and cleaned from [Encyclopedia of Life (EOL)](https://eol.org). It contains images for more than 450K distinct taxa, as measured by 7-rank [Linnaean taxonomy](https://www.britannica.com/science/taxonomy/The-objectives-of-biological-classification) (kingdom through species); this taxonomic string is associated to each image along with its common (or vernacular name) where available.

We cannot re-release the iNat21 or the BIOSCAN-1M datasets; however, we have uploaded our cleaned EOL data to [TreeOfLife-10M](https://huggingface.co/datasets/imageomics/TreeOfLife-10M) on Hugging Face.
After downloading iNat21 and BIOSCAN-1M, the three datasets can be combined into TreeOfLife-10M in the [webdataset format](https://github.com/webdataset/webdataset) for model training by following the directions in [`treeoflife10m.md`](/docs/imageomics/treeoflife10m.md).


10 biologically-relevant datasets were used for various tests of [BioCLIP](https://huggingface.co/imageomics/bioclip), they are described (briefly) and linked to below. For more information about the contents of these datasets, see Table 2 and associated sections of [our paper](https://doi.org/10.48550/arXiv.2311.18803). Annotations used alongside the datasets for evaluation are provided in subfolders of the `data/` directory named for the associated dataset.

#### Test Sets

- [Meta-Album](https://paperswithcode.com/dataset/meta-album): Specifically, we used the Plankton, Insects, Insects 2, PlantNet, Fungi, PlantVillage, Medicinal Leaf, and PlantDoc datasets from Set-0 through Set-2 (Set-3 had not yet been released).
 - [Birds 525](https://www.kaggle.com/datasets/gpiosenka/100-bird-species): We evaluated on the 2,625 test images provided with the dataset.
 - [Rare Species](https://huggingface.co/datasets/imageomics/rare-species): A new dataset we curated for the purpose of testing this model and to contribute to the ML for Conservation community. It consists of nearly 12K images representing 400 species labeled Near Threatened through Extinct in the Wild by the [IUCN Red List](https://www.iucnredlist.org/). For more information, see our [Rare Species dataset](https://huggingface.co/datasets/imageomics/rare-species).



<h2 id="paper">Paper, Website, and Docs</h2>

We have a preprint on [arXiv](https://arxiv.org/abs/2311.18803) and a [project website](https://imageomics.github.io/bioclip/).
We also will link to the upcoming CVPR 2024 version when it is publicly available.


The `docs/` directory is divided into two subfolders: [`imageomics/`](/docs/imageomics) and `open_clip/`. The former is documentation relating to the creation of BioCLIP, TreeOfLife-10M, and the Rare Species dataset, while the latter is documentation from the [`open_clip`](https://github.com/mlfoundations/open_clip) package (this has not been altered).
We plan on adding more docs on how to use BioCLIP in a variety of settings.
For now, if it is unclear how to integrate BioCLIP into your project, please open an issue with your questions.

## Citation

Our paper:

```
@inproceedings{stevens2024bioclip,
  title = {{B}io{CLIP}: A Vision Foundation Model for the Tree of Life}, 
  author = {Samuel Stevens and Jiaman Wu and Matthew J Thompson and Elizabeth G Campolongo and Chan Hee Song and David Edward Carlyn and Li Dong and Wasila M Dahdul and Charles Stewart and Tanya Berger-Wolf and Wei-Lun Chao and Yu Su},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2024},
  pages = {19412-19424}
}
```

Our code (this repository):
```
@software{bioclip2023code,
  author = {Samuel Stevens and Jiaman Wu and Matthew J. Thompson and Elizabeth G. Campolongo and Chan Hee Song and David Edward Carlyn},
  doi = {10.5281/zenodo.10895871},
  title = {BioCLIP},
  version = {v1.0.0},
  year = {2024}
}
```


Also consider citing OpenCLIP, iNat21 and BIOSCAN-1M:

```
@software{ilharco_gabriel_2021_5143773,
  author={Ilharco, Gabriel and Wortsman, Mitchell and Wightman, Ross and Gordon, Cade and Carlini, Nicholas and Taori, Rohan and Dave, Achal and Shankar, Vaishaal and Namkoong, Hongseok and Miller, John and Hajishirzi, Hannaneh and Farhadi, Ali and Schmidt, Ludwig},
  title={OpenCLIP},
  year={2021},
  doi={10.5281/zenodo.5143773},
}
```

```
@misc{inat2021,
  author={Van Horn, Grant and Mac Aodha, Oisin},
  title={iNat Challenge 2021 - FGVC8},
  publisher={Kaggle},
  year={2021},
  url={https://kaggle.com/competitions/inaturalist-2021}
}
```

```
@inproceedings{gharaee2023step,
  author={Gharaee, Z. and Gong, Z. and Pellegrino, N. and Zarubiieva, I. and Haurum, J. B. and Lowe, S. C. and McKeown, J. T. A. and Ho, C. Y. and McLeod, J. and Wei, Y. C. and Agda, J. and Ratnasingham, S. and Steinke, D. and Chang, A. X. and Taylor, G. W. and Fieguth, P.},
  title={A Step Towards Worldwide Biodiversity Assessment: The {BIOSCAN-1M} Insect Dataset},
  booktitle={Advances in Neural Information Processing Systems ({NeurIPS}) Datasets \& Benchmarks Track},
  year={2023},
}
```
