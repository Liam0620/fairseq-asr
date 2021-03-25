model_path=examples/wav2vec/pretrained_models/MTL_large_vad_v2.pt
lm_path=wav2vec2_big_finetune_CN17000_char/model/general0.5_lbk0.1_medicine0.4__v1.arpa.5gram
lexicon_path=wav2vec2_big_finetune_CN17000_char/model/lexicon.txt

python http/websocket_server_develop.py None \
        --task audio_mtl --datasets_dict "asr_none,vad_none" \
        --nbest 1 --path $model_path \
        --w2l-decoder kenlm \
        --lm-model $lm_path --lm-weight 0.46 --word-score 2.22 \
        --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
        --post-process letter --lexicon $lexicon_path --beam 20 --use-cuda True \
        --min-speech 0.2 --min-silence 0.4 --speech-onset 0.5 \
        --chunk-size 1 --asr-chunk-size 2 \
        --address 172.18.30.90 --port 4008 --devices 2 --max-users-per-device 1 \




