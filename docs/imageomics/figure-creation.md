# Figure Creation

This document describes how to create the t-SNE plot visualizations of the iNat validation set from BioCLIP's and OpenAI's model embeddings.

## Extract Data

OpenAI embeddings
```
python -m src.evaluation.extract_features --pretrained openai --exp_type openai --feature_output features --val_root [path_to_iNat21_val]
```

BioCLIP embeddings

```
python -m src.evaluation.extract_features --model hf-hub:imageomics/bioclip  --exp_type bioclip --feature_output features --val_root [path_to_iNat21_val]
```


## Create t-SNE Figures

OpenAI figures
```
python -m src.evaluation.hierarchy_tree_image --exp_type openai --rerun --remove_outliers --features_output features --results results --val_root [path_to_iNat21_val]

```

BioCLIP figures
```
python -m src.evaluation.hierarchy_tree_image --exp_type bioclip --rerun --remove_outliers --features_output features --results results --val_root [path_to_iNat21_val]
```

## Create Final Figure

```
python -m src.evaluation.create_final_figure --openai_tsne_data openai_tsne_rerun_top_6_remove_outliers --bioclip_tsne_data bioclip_tsne_rerun_top_6_remove_outliers --results_output results --val_root [path_to_iNat21_val]
```
