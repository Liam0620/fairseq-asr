valid_subset=valid
train_path=examples/wav2vec/manifest/MTL
ts_path="/data5/syzhou/work/MTL_data/Large_ASV_pack/Large_ASV_embeddings"
noise_path=/data5/syzhou/work/MTL_data/background_noise/ #--noise-path $noise_path
W2V_PATH=kd_model/w2v_base/checkpoint_last.pt
finetune_model=kd_model/checkpoint_last.pt.nokd
SAVE_DIR=examples/wav2vec/exp/TS_VAD_only_noise_KD_WPLOSS_MixSim
log_file=./run_MTL_TS_VAD_KD_WPLOSS.log #_ce
PORT=-1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python train.py --distributed-world-size 8 $train_path \
--tensorboard-logdir $SAVE_DIR --datasets "asr_KD_none,vad_large_asv_TSP,vad_noise_TSP" \
--save-dir $SAVE_DIR --update-freq 2 \
--max-sample-size 32320 --min-sample-size 32320 --ts-path $ts_path \
--best-checkpoint-metric loss \
--valid-subset $valid_subset --no-epoch-checkpoints --num-workers 8 \
--max-update 80000 --sentence-avg --task audio_mtl_ts --arch wav2vec_class_TS_vad --w2v-path $W2V_PATH --w2v-path2 $finetune_model \
--labels cls --apply-mask --mask-selection static --mask-other 0 --mask-length 10 --mask-prob 0.5 --layerdrop 0.1 \
--mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.5 \
--feature-grad-mult 0.0 --freeze-finetune-updates 5000 --validate-after-updates 10000 --optimizer adam \
--adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --lr 2e-05 --lr-scheduler tri_stage --warmup-steps 2000 --hold-steps 38000 \
--decay-steps 40000 --final-lr-scale 0.05 --final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 --criterion mtl_ts_vad_only \
--attention-dropout 0.0 --max-tokens 4800000 --max-tokens-valid 2400000 --seed 2345 --log-format json --log-interval 100 --ddp-backend=no_c10d  \
> $log_file &


#vad_hkust_TSP,
