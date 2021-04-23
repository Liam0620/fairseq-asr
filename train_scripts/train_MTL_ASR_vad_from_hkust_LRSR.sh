valid_subset=valid
train_path=examples/wav2vec/manifest/MTL
#base_model=examples/wav2vec/pretrained_models/wav2vec_small_aishell.pt
#finetune_model=examples/wav2vec/exp/hkust/char/checkpoint_last.pt
#finetune_model=examples/wav2vec/exp/MTL_ASR_vad_from_hkust/checkpoint_last.pt
finetune_model=examples/wav2vec/exp/MTL_ASR_vad_from_hkust_conv_layer_seg_cnn_grad/checkpoint_best.pt

SAVE_DIR=examples/wav2vec/exp/MTL_ASR_vad_from_hkust_conv_layer_seg_cnn_grad_integrate_new
log_file=./run_hukst_small_MTL_ASR_vad_from_hkust_conv_layer_seg_cnn_grad_integrate_new.log
PORT=-1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python train.py --distributed-world-size 8 $train_path \
--tensorboard-logdir $SAVE_DIR --datasets "asr_hkust,vad_combine" \
--save-dir $SAVE_DIR --update-freq 2 \
--max-sample-size 100000 \
--post-process letter  \
--valid-subset $valid_subset --no-epoch-checkpoints --best-checkpoint-metric uer --num-workers 12 \
--max-update 90000 --sentence-avg --task audio_mtl --arch wav2vec_class_vad_v3 --w2v-path2 $finetune_model \
--labels cls --apply-mask --mask-selection static --mask-other 0 --mask-length 10 --mask-prob 0.5 --layerdrop 0.1 \
--mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.5  --zero-infinity \
--feature-grad-mult 0.0 --freeze-finetune-updates 15000 --validate-after-updates 10000 --optimizer adam \
--adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --lr 2e-05 --lr-scheduler tri_stage --warmup-steps 8000 --hold-steps 42000 \
--decay-steps 40000 --final-lr-scale 0.05 --final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 --criterion mtl_asr_vad_wl \
--attention-dropout 0.0 --max-tokens 2400000 --seed 2337 --log-format json --log-interval 100 --ddp-backend=no_c10d  \
> $log_file &
#--wer-args '("examples/wav2vec/pretrained_models/hkust.bin","examples/wav2vec/manifest/hkust/lexicon/char2pin.lexicon.txt",2,-1)' \
#--reset-optimizer --reset-lr-scheduler --max-sample-size 80000 --min-sample-size 0 \
#--ddp-backend no_c10d --max-tokens 1280000 # ,train_asr_libri_10h ,train_hkust_1w --max-sample-size 80000 \ --use-bmuf
# "train_AP18,train_vox1,train_hkust_vad,train_hkust" --w2v-path2 $finetune_model
