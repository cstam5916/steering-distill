Note: If `pip install -r requirements.txt` gives an error, try running ```pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu129.html``` before installing from requirements.

Code for knowledge distillation based on steering vectors, for the graduate course CMPSC 292F taught by Ambuj Singh WI26 at UCSB.

Currently planning to use LLama 3.2 8B as the teacher model, and LLama 3.1 1B as the student model
Zero-shot performance on test set
 - 8B: 73.69%
 - 1B: 42.53%

As of 2/26, code for fine-tuning the student model should be usable
Remaining tasks
1. Incorporate logit-based KD loss
2. Extract steering vectors from the teacher model
3. Incorporate steering-based loss
