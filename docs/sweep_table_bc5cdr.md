| run | base model | lr | batch | epochs | seed | all subwords | types | val P | val R | val F1 | test F1 | val F1 Chemical | val F1 Disease | test F1 Chemical | test F1 Disease |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lr3e-5_bs16_ep3_chemical | biobert-base-cased-v1.2 | 3e-05 | 16 | 3 | 42 | False | Chemical | 0.9455 | 0.9416 | 0.9436 | 0.9284 | 0.9436 | - | 0.9284 | - |
| pubmedbert_lr3e-5_bs16_ep3 | BiomedNLP-PubMedBERT-base-uncased-abstract | 3e-05 | 16 | 3 | 42 | False | - | 0.9041 | 0.9146 | 0.9093 | 0.9014 | 0.9512 | 0.8561 | 0.9397 | 0.8550 |
| lr3e-5_bs16_ep3 | biobert-base-cased-v1.2 | 3e-05 | 16 | 3 | 42 | False | - | 0.8947 | 0.8983 | 0.8965 | 0.8886 | 0.9400 | 0.8403 | 0.9312 | 0.8361 |
| lr3e-5_bs16_ep3_seed1337 | biobert-base-cased-v1.2 | 3e-05 | 16 | 3 | 1337 | False | - | 0.8921 | 0.8988 | 0.8954 | 0.8886 | 0.9408 | 0.8378 | 0.9300 | 0.8385 |
| lr3e-5_bs16_ep3_disease | biobert-base-cased-v1.2 | 3e-05 | 16 | 3 | 42 | False | Disease | 0.8357 | 0.8399 | 0.8378 | 0.8395 | - | 0.8378 | - | 0.8395 |
