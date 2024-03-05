# BioCLIP

This is the repository for the BioCLIP model and the TreeOfLife-10M dataset.

[Paper](https://arxiv.org/abs/2311.18803) | [Model](https://huggingface.co/imageomics/bioclip) | Data (coming soon) | [Demo](https://huggingface.co/spaces/imageomics/bioclip-demo)
---

BioCLIP is a CLIP model trained on new 10M-image dataset of biological organisms with fine-grained taxonomic labels.
BioCLIP outperforms general domain baselines on a wide spread of biology-related tasks, including zero-shot and few-shot classification.

## Table of Contents

1. [Model](#model)
2. [Data](#data)
3. [Paper, website and docs](#paper)
4. [Citation](#citation)

## Model

The BioCLIP model is a ViT-B/16 pretrained with the CLIP objective.
Both the ViT and the (small) autoregressive text encoder are available to download on [Huggingface](https://huggingface.co/imageomics/bioclip).

The only depedency is the [`open_clip`](https://github.com/mlfoundations/open_clip) package.
See the `examples/` directory for examples on how to use it.

## Data

BioCLIP was trained on TreeOfLife-10M (ToL-10M).
The data is a combination of [iNat21](https://github.com/visipedia/inat_comp/tree/master/2021), [BIOSCAN-1M](https://github.com/zahrag/BIOSCAN-1M) and data we collected and cleaned from [EOL](https://eol.org).
We cannot re-release the iNat21 or the BIOSCAN-1M datasets; however, we've uploaded our cleaned EOL data to Huggingface (TODO: add link).
Then you can use our scripts to download iNat21 and BIOSCAN-10M and then combine the three datsaets into TreeOfLife-10M into the [webdataset format](https://github.com/webdataset/webdataset) for model training.

<h2 id="paper">Paper, Website and Docs</h2>

We have a preprint on [arXiv](https://arxiv.org/abs/2311.18803) and a [project website](https://imageomics.github.io/bioclip/).
We also will link to the upcoming CVPR 2024 version when it is publicly available.

We plan on adding more docs on how to use BioCLIP in a variety of settings.
For now, if it is unclear how to integrate BioCLIP into your project, open an issue or email [Sam](mailto:stevens.994@buckeyemail.osu.edu) with your questions.

## Citation

Our paper:

```
@article{stevens2023bioclip,
  title = {BIOCLIP: A Vision Foundation Model for the Tree of Life}, 
  author = {Samuel Stevens and Jiaman Wu and Matthew J Thompson and Elizabeth G Campolongo and Chan Hee Song and David Edward Carlyn and Li Dong and Wasila M Dahdul and Charles Stewart and Tanya Berger-Wolf and Wei-Lun Chao and Yu Su},
  year = {2023},
  eprint = {2311.18803},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV}
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

