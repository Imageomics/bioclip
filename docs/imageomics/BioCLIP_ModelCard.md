---
license: 
- mit
language:
- en
library_name: open_clip
tags:
- zero-shot-image-classification
- clip
- biology
- CV
- images
- animals
- species
- taxonomy
- rare species
- endangered species
- evolutionary biology
- multimodal
- knowledge-guided
datasets:
- TreeOfLife-10M
- iNat21
- BIOSCAN-1M
- EOL
---


# Model Card for BioCLIP

<!-- 
This modelcard has been generated using [this raw template](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md?plain=1). And further altered to suit Imageomics Institute needs -->

BioCLIP is a foundation model for the tree of life, built using CLIP architecture as a vision model for general organismal biology. 
It is trained on [TreeOfLife-10M](https://huggingface.co/datasets/imageomics/TreeOfLife-10M), our specially-created dataset covering over 450K taxa--the most biologically diverse ML-ready dataset available to date. 
Through rigorous benchmarking on a diverse set of fine-grained biological classification tasks, BioCLIP consistently outperformed existing baselines by 17% to 20% absolute. 
Through intrinsic evaluation, we found that BioCLIP learned a hierarchical representation aligned to the tree of life, which demonstrates its potential for robust generalizability.

**See the `examples/` directory for examples of how to use BioCLIP in zero-shot and few-shot settings.**

## Model Details

### Model Description

BioCLIP is based on OpenAI's [CLIP](https://openai.com/research/clip). 
We trained the model on [TreeOfLife-10M](https://huggingface.co/datasets/imageomics/TreeOfLife-10M) from OpenAI's ViT-B/16 checkpoint, using [OpenCLIP's](https://github.com/mlfoundations/open_clip) code.
BioCLIP is trained with the standard CLIP objective to imbue the model with an understanding, not just of different species, but of the hierarchical structure that relates species across the tree of life. 
In this way, BioCLIP offers potential to aid biologists in discovery of new and related creatures, since it does not see the 454K different taxa as distinct classes, but as part of an interconnected hierarchy. 


- **Developed by:** Samuel Stevens, Jiaman Wu, Matthew J. Thompson, Elizabeth G. Campolongo, Chan Hee Song, David Edward Carlyn, Li Dong, Wasila M. Dahdul, Charles Stewart, Tanya Berger-Wolf, Wei-Lun Chao, and Yu Su
- **Model type:** Vision Transformer (ViT-B/16)
- **License:** MIT
- **Fine-tuned from model:** OpenAI CLIP, ViT-B/16

This model was developed for the benefit of the community as an open-source product, thus we request that any derivative products are also open-source.

### Model Sources

- **Repository:** [BioCLIP](https://github.com/Imageomics/BioCLIP)
- **Paper:** BioCLIP: A Vision Foundation Model for the Tree of Life ([arXiv](https://doi.org/10.48550/arXiv.2311.18803))
- **Demo:** [BioCLIP Demo](https://huggingface.co/spaces/imageomics/bioclip-demo)

## Uses

BioCLIP has been extensively evaluated on species classification tasks across many different subtrees of the tree of life.
The ViT-B/16 vision encoder is recommended as a base model for any computer vision task for biology; we expect it to outperform general domain models with the same architecture on biology-specific tasks.


### Direct Use

See the demo [here](https://huggingface.co/spaces/imageomics/bioclip-demo) for examples of zero-shot classification.
It can also be used in a few-shot setting with a KNN; please see [our paper](https://doi.org/10.48550/arXiv.2311.18803) for details for both few-shot and zero-shot settings without fine-tuning.


## Bias, Risks, and Limitations

This model was developed from the original CLIP model, thus many of the concerns discussed in ([Radford et al. 2021](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf)) apply. 
We encourage the concerned/curious user to read their extensive ethics statement, while we focus our attention on the biological perspective which is unique to BioCLIP. 
 - No specific geographic information (eg., GPS coordinates) are included in training, so the species classification does not pose a direct threat to animals through aiding poachers, as it cannot inform them of their location.
 - BioCLIP is designed to aid in scientific discovery through an association of images to the hierarchical taxonomy structure. As with many--if not all--models currently in production, it is important to retain the context that it is meant to assist biologists in their work, not replace them. As such, we caution against over-reliance on model predictions.

### Recommendations

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. 
More information needed for further recommendations.

## How to Get Started with the Model

BioCLIP can be used with the `open_clip` library:

```py
import open_clip

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
```

## Training Details

### Compute Infrastructure

Training was performed on 8 NVIDIA A100-80GB GPUs distributed over 2 nodes on [OSC's](https://www.osc.edu/) Ascend HPC Cluster with global batch size 32,768 for 4 days.

Based on [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://doi.org/10.48550/arXiv.1910.09700), that's 132.71 kg of CO<sub>2</sub> eq., or 536km driven by an average ICE car.

### Training Data

This model was trained on [TreeOfLife-10M](https://huggingface.co/datasets/imageomics/TreeOfLife-10M), which is a compilation of images matched to [Linnaean taxonomic rank](https://www.britannica.com/science/taxonomy/The-objectives-of-biological-classification) from kingdom through species. They are also matched with common (vernacular) name of the subject of the image where available. For more information, please see our dataset, [TreeOfLife-10M](https://huggingface.co/datasets/imageomics/TreeOfLife-10M).

### Training Hyperparameters

- **Training regime:** fp16 mixed precision.

We resize images to 224 x 224 pixels.
We use a maximum learning rate of 1e4 with 1000 linear warm-up steps, then use cosine decay to 0 over 100 epochs.
We also use a weight decay of 0.2 and a batch size of 32K.

## Evaluation

### Testing Data

We tested BioCLIP on the following collection of 10 biologically-relevant tasks.
 - [Meta-Album](https://paperswithcode.com/dataset/meta-album): Specifically, we used the Plankton, Insects, Insects 2, PlantNet, Fungi, PlantVillage, Medicinal Leaf, and PlantDoc datasets from Set-0 through Set-2 (Set-3 was still not released as of our publication/evaluation (Nov. 2023).
 - [Birds 525](https://www.kaggle.com/datasets/gpiosenka/100-bird-species): We evaluated on the 2,625 test images provided with the dataset.
 - [Rare Species](https://huggingface.co/datasets/imageomics/rare-species): A new dataset we curated for the purpose of testing this model and to contribute to the ML for Conservation community. It consists of 400 species labeled Near Threatened through Extinct in the Wild by the [IUCN Red List](https://www.iucnredlist.org/), with 30 images per species. For more information, see our dataset, [Rare Species](https://huggingface.co/datasets/imageomics/rare-species).

For more information about the contents of these datasets, see Table 2 and associated sections of [our paper](https://doi.org/10.48550/arXiv.2311.18803).

### Metrics

We use top-1 and top-5 accuracy to evaluate models, and validation loss to choose the best performing checkpoints from training.

### Results

We compare BioCLIP to OpenAI's CLIP and OpenCLIP's LAION-2B checkpoint.
Here are the zero-shot classification results on our benchmark tasks. 
Please see [our paper](https://doi.org/10.48550/arXiv.2311.18803) for few-shot results.

<table cellpadding="0" cellspacing="0">
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="4">Animals</th>
      <th colspan="5">Plants & Fungi</th>
      <th rowspan="2">Rare Species</th>
      <th rowspan="2">Mean</th>
    </tr>
    <tr>
      <th>Birds 525</th>
      <th>Plankton</th>
      <th>Insects</th>
      <th>Insects 2</th>
      <th>PlantNet</th>
      <th>Fungi</th>
      <th>PlantVillage</th>
      <th>Med. Leaf</th>
      <th>PlantDoc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>CLIP</td>
      <td>49.9</td>
      <td>3.2</td>
      <td>9.1</td>
      <td>9.8</td>
      <td>58.5</td>
      <td>10.2</td>
      <td>5.4</td>
      <td>15.9</td>
      <td>26.1</td>
      <td>26.6</td>
      <td>21.4</td>
    </tr>
    <tr>
      <td>OpenCLIP</td>
      <td>54.7</td>
      <td>2.2</td>
      <td>6.5</td>
      <td>9.6</td>
      <td>50.2</td>
      <td>5.7</td>
      <td>8.0</td>
      <td>12.4</td>
      <td>25.8</td>
      <td>31.0</td>
      <td>20.6</td>
    </tr>
    <tr>
      <td>BioCLIP</td>
      <td><b>74.7</b></td>
      <td><b>5.4</b></td>
      <td><b>32.7</b></td>
      <td><b>21.2</b></td>
      <td><b>91.0</b></td>
      <td><b>51.8</b></td>
      <td><b>24.0</b></td>
      <td><b>48.1</b></td>
      <td><b>27.5</b></td>
      <td><b>39.2</b></td>
      <td><b>41.5</b></td>
    </tr>
    <tr>
      <td>iNat21 Only</td>
      <td>55.7</td>
      <td>2.7</td>
      <td>29.9</td>
      <td>12.0</td>
      <td>89.3</td>
      <td>42.7</td>
      <td>16.4</td>
      <td>22.2</td>
      <td>18.8</td>
      <td>19.4</td>
      <td>30.9</td>
    </tr>
  </tbody>
</table>


### Summary

BioCLIP outperforms general-domain baselines by 18% on average.

### Model Examination

We encourage readers to see Section 4.6 of [our paper](https://doi.org/10.48550/arXiv.2311.18803).
In short, BioCLIP forms representations that more closely align to the taxonomic hierarchy compared to general-domain baselines like CLIP or OpenCLIP.


## Citation 

**BibTeX:**

```
@software{bioclip2023,
  author = {Samuel Stevens and Jiaman Wu and Matthew J. Thompson and Elizabeth G. Campolongo and Chan Hee Song and David Edward Carlyn and Li Dong and Wasila M. Dahdul and Charles Stewart and Tanya Berger-Wolf and Wei-Lun Chao and Yu Su},
  doi = {10.57967/hf/1511},
  month = nov,
  title = {BioCLIP},
  version = {v0.1},
  year = {2023}
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
  primaryClass = {cs.CV}
}

```


Please also consider citing OpenCLIP, iNat21 and BIOSCAN-1M:
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

## Acknowledgements

The authors would like to thank Josef Uyeda, Jim Balhoff, Dan Rubenstein, Hank Bart, Hilmar Lapp, Sara Beery, and colleagues from the Imageomics Institute and the OSU NLP group for their valuable feedback. We also thank the BIOSCAN-1M team and the iNaturalist team for making their data available and easy to use, and Jennifer Hammack at EOL for her invaluable help in accessing EOL’s images.

The [Imageomics Institute](https://imageomics.org) is funded by the US National Science Foundation's Harnessing the Data Revolution (HDR) program under [Award #2118240](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2118240) (Imageomics: A New Frontier of Biological Information Powered by Knowledge-Guided Machine Learning). Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.


## Model Card Authors

Elizabeth G. Campolongo, Samuel Stevens, and Jiaman Wu

## Model Card Contact

[stevens.994@osu.edu](mailto:stevens.994@osu.edu)
