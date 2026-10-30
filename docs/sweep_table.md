| run | base model | lr | batch | epochs | seed | all subwords | val P | val R | val F1 | test F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| pubmedbert_lr3e-5_bs16_ep3 | BiomedNLP-PubMedBERT-base-uncased-abstract | 3e-05 | 16 | 3 | 42 | False | 0.8730 | 0.8907 | 0.8818 | 0.8718 |
| lr3e-5_bs16_ep3_seed2024 | biobert-base-cased-v1.2 | 3e-05 | 16 | 3 | 2024 | False | 0.8534 | 0.8729 | 0.8631 | 0.8809 |
| lr3e-5_bs16_ep5 | biobert-base-cased-v1.2 | 3e-05 | 16 | 5 | 42 | False | 0.8548 | 0.8679 | 0.8613 | 0.8702 |
| lr3e-5_bs16_ep3_seed1337 | biobert-base-cased-v1.2 | 3e-05 | 16 | 3 | 1337 | False | 0.8516 | 0.8679 | 0.8597 | 0.8838 |
| lr3e-5_bs16_ep3_all-subwords | biobert-base-cased-v1.2 | 3e-05 | 16 | 3 | 42 | True | 0.8483 | 0.8666 | 0.8573 | 0.8767 |
| lr3e-5_bs16_ep3 | biobert-base-cased-v1.2 | 3e-05 | 16 | 3 | 42 | False | 0.8500 | 0.8640 | 0.8570 | 0.8761 |
| lr3e-5_bs32_ep5 | biobert-base-cased-v1.2 | 3e-05 | 32 | 5 | 42 | False | 0.8518 | 0.8615 | 0.8566 | 0.8809 |
| lr5e-5_bs16_ep3 | biobert-base-cased-v1.2 | 5e-05 | 16 | 3 | 42 | False | 0.8405 | 0.8640 | 0.8521 | 0.8746 |
| lr2e-5_bs16_ep3 | biobert-base-cased-v1.2 | 2e-05 | 16 | 3 | 42 | False | 0.8309 | 0.8742 | 0.8520 | 0.8804 |
