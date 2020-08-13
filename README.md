# A Unified Dual-view Model for Review Summarization and Sentiment Classification with Inconsistency Loss

This repository contains the source code for our SIGIR 2020 paper "[A Unified Dual-view Model for Review Summarization and Sentiment Classification with Inconsistency Loss](https://arxiv.org/abs/2006.01592)". 

Some of our source code are adapted from https://github.com/ChenRocks/fast_abs_rl. 

If you use this code, please cite our paper:
```
@inproceedings{DBLP:conf/sigir/ChanCK20,
  author    = {Hou Pong Chan and
               Wang Chen and
               Irwin King},
  title     = {A Unified Dual-view Model for Review Summarization and Sentiment Classification
               with Inconsistency Loss},
  booktitle = {Proceedings of {SIGIR} 2020, Virtual
               Event, China, July 25-30, 2020},
  pages     = {1191--1200},
  year      = {2020},
  url       = {https://doi.org/10.1145/3397271.3401039},
  doi       = {10.1145/3397271.3401039},
  biburl    = {https://dblp.org/rec/conf/sigir/ChanCK20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

### Model Architecture
![](figs/dual_view_model_architecture.png)

## Dependencies
- Pytorch 1.1
- NLTK
- pyrouge

Please refer to the requirements.txt for the full dependencies. 


## Datasets
We use the Sports and Outdoors, Toys and Games, Home and Kitchen, Movies and TV datasets from from 5-core subsets of the Amazon review corpus. If you use their data, please cite their papers as well. 

## Data Preprocessing

- Download Stanford CoreNLP English from https://stanfordnlp.github.io/CoreNLP/history.html

- First, you need to run a corenlp server on the same server. cd to the directory of the standford corenlp. Then execute `java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit -status_port 9000 -port 9000 -timeout 60000`.

- Open another terminal and execute the following command. It will create a file called `reviews_Sports_and_Outdoors_5_tokenized.json`. 

`python3 tokenize_raw_review.py -raw_data_file reviews_Sports_and_Outdoors_5.json`


- Split the data into train, valid, test. Remove too long and too short review and summary

`python3 preprocess_raw_review.py -raw_data_file reviews_Sports_and_Outdoors_5_tokenized.json -out_dir datasets/processed_reviews_Sports_and_Outdoors_5 -num_valid 9000 -num_test 9000 -is_shuffle`
`python3 preprocess_raw_review.py -raw_data_file reviews_Toys_and_Games_5_tokenized.json -out_dir datasets/processed_reviews_Toys_and_Games_5 -num_valid 8000 -num_test 8000 -is_shuffle`
`python3 preprocess_raw_review.py -raw_data_file reviews_Home_and_Kitchen_5_tokenized.json -out_dir datasets/processed_reviews_ome_and_Kitchen_5 -num_valid 10000 -num_test 10000 -is_shuffle`
`python3 preprocess_raw_review.py -raw_data_file reviews_Movies_and_TV_5_tokenized.json -out_dir datasets/processed_reviews_Movies_and_TV_5 -num_valid 20000 -num_test 20000 -is_shuffle`

- Compute and export the class distribution on the training set

`python3 compute_rating_stat.py -data_dir datasets/processed_reviews_Sports_and_Outdoors_5 -split train`

## Build Rating and Vocabulary
It will create five vocab pickle files in the `-data_dir` for each rating score: `rating_1_vocab_counter.pkl`, `rating_2_vocab_counter.pkl`, ...

`python3 find_rating_vocab.py -data_dir datasets/processed_reviews_Sports_and_Outdoors_5 -split train`

## Training

- Script for training a word2vec embedding on the training set: 

`python3 train_word2vec.py -data datasets/processed_reviews_Sports_and_Outdoors_5 -path word_embeddings/sport_and_outdoors -dim 128`

- Script for training a hss model on the full sports_and_outdoors dataset

`python3 train_ml.py -data /research/king3/ik_grp/wchen/amazon_review_data/processed_reviews_Sports_and_Outdoors_5 -exp_path exp/%s.%s -model_path /research/king3/ik_grp/wchen/senti_summ_models/saved_model/%s.%s -exp train_ml_baseline_sport_dataset_hidden_size_512_seed_9527 -epochs 50 -copy_attention -batch_size 32 -seed 9527 -w2v /research/king3/ik_grp/wchen/senti_summ_models/word_embeddings/sport_and_outdoors/word2vec.128d.39k.bin -v_size 50000 -word_vec_size 128 -encoder_size 256 -decoder_size 512`

- Scripts for training our dual-view model with inconsistency loss

```
python3 train_ml.py \
-data=datasets/processed_reviews_Sports_and_Outdoors_5 \
-exp_path=exp/%s.%s \
-exp=train_movie_dual_view_inc_seed_250 \
-epochs=50 \
-checkpoint_interval=1000 \
-copy_attention \
-batch_size=32 \
-seed=250 \
-w2v=word_embeddings/sport_and_outdoors \
-v_size=50000 \
-word_vec_size=128 \
-encoder_size=256 \
-decoder_size=512 \
-enc_layers=2 \
-residual \
-model_type=multi_view_multi_task_basic \
-dropout=0.0 \
-dec_classify_input_type=dec_state \
-classifier_type=word_multi_hop_attn \
-dec_classifier_type=word_multi_hop_attn \
-gen_loss_weight=0.8 \
-class_loss_weight=0.1 \
-inconsistency_loss_type=KL_div \
-inconsistency_loss_weight=0.1 \
-early_stop_loss=joint \
-batch_workers 0
```

## Predict

- Download pyrouge, and save it to `path/to/pyrouge`. 

`git clone https://github.com/andersjo/pyrouge.git`

- Export ROUGE score enviornment variable

`export ROUGE=[path/to/pyrouge/tools/ROUGE-1.5.5]`

- Make evaluation reference for a dataset (Only need to do it for once for each dataset)

`python make_eval_reference.py -data datasets/processed_reviews_Sports_and_Outdoors_5 -split all`

- Run predict, specify the path to the best checkpoint (lowest validation loss) in the `-pretrained_model` argument. 

```
/research/king3/hpchan/anaconda3/envs/summ_sentiment/bin/python3 -u predict.py \
-data datasets/processed_reviews_Sports_and_Outdoors_5 \
-pred_path /research/king3/ik_grp/wchen/senti_summ_outputs/pred/%s.%s \
-exp predict_dual_view_inc_seed_250 \
-pretrained_model saved_model/train_movie_dual_view_inc_seed_250.ml.copy.bi-directional.20191212-154843/ckpt/train_movie_dual_view_inc_seed_250.ml.copy.bi-directional-epoch-2-total_batch-75000-joint-2.640 \
-seed 9527 \
-batch_size 16 \
-replace_unk \
-src_max_len -1
```

- Run evaluate prediction to compute ROGUE scores, macro F1, and balanced accuracy. The reported macro F1 and balanced accuracy are results from our source-view sentiment classifier. 

`python evaluate_prediction.py -rouge -decode_dir pred/predict_dual_view_inc_seed_250.20190901-160022 -data datasets/processed_reviews_Sports_and_Outdoors_5`
