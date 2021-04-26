
#lm_path=/data/syzhou/kd_model/general0.6_GVHD0.2_medicine0.2_v1.arpa.5gram
#/data/syzhou/kd_model/general0.4_medicine0.1_shenwai0.5_v1.arpa.5gram

model_path=/data3/mli2/mli/fairseq-master/examples/wav2vec/exp/TS_VAD_only_noise_KD_WPL/checkpoint_mixsim_last.pt
lm_path=/data/syzhou/kd_model/general0.4_medicine0.1_shenwai0.5_v1.arpa.5gram
#wav2vec2_big_finetune_CN17000_char/model/general0.5_lbk0.1_medicine0.4__v1.arpa.5gram
lexicon_path=kd_model/lexicon.txt

:<<BLOCK
# screen -L -t $port
for assign in "4005|0" "4006|1" "4007|2" "4008|3" "4009|4" "4010|5" "4011|0" "4012|1" "4013|2" "4014|3" "4015|4" "4016|5"
    do
        array=(${assign//|/ })
        port=${array[0]}
        device=${array[1]}
        echo "port:"$port "devices:"$device
        screen -L -t $port python http/websocket_server_develop.py None \
          --task audio_mtl --datasets_dict "asr_KD,vad_none" \
          --nbest 1 --path $model_path \
          --w2l-decoder kenlm \
          --lm-model $lm_path --lm-weight 0.46 --word-score 2.22 \
          --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
          --post-process letter --lexicon $lexicon_path --beam 20 --use-cuda True \
          --min-speech 0.2 --min-silence 0.5 --speech-onset 0.42 \
          --chunk-size 0.5 --asr-chunk-size 3 --step 1 \
          --address 172.18.30.90 --port $port --devices $device --max-users-per-device 1
        echo "done" $assign
    done

echo "FINISH"
BLOCK

#:<<BLOCK # screen -L -t 5005
python http_TS_vad/websocket_server_develop.py None \
        --task audio_mtl --datasets_dict "asr_KD,vad_none_TS" \
        --nbest 1 --path $model_path \
        --w2l-decoder kenlm \
        --lm-model $lm_path --lm-weight 0.46 --word-score 0 \
        --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
        --post-process letter --lexicon $lexicon_path --beam 20 --use-cuda True \
        --min-speech 0.3 --min-silence 0.1 --speech-onset 0.6 \
        --chunk-size 2 --asr-chunk-size 10 --step 0.5 \
        --address 172.18.30.90 --port 5009 --devices 2 --max-users-per-device 2
#BLOCK



