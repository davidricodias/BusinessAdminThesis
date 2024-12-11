seq_len=64
pred_len=4
dataset_dimension=62  # Number of features
model_name=PatchMixer
model_id_name=lob
data_name="custom"
root_path_name=./dataset/lob
random_seed=42

source .venv/bin/activate  # Activate virtual environment

for ticker in GRLS SAN AENA
do
    file="$root_path_name"/"$ticker"_lob.csv
    if [ ! -f "$file" ]; then
        python -u process_book.py --ric_ticker $ticker
    fi

    python -u run.py \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $ticker"_lob.csv" \
        --model_id $ticker \
        --model $model_name \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in $dataset_dimension \
        --dec_in $dataset_dimension \
        --dropout 0.2\
        --des 'LOB_TRAIN' \
        --train_epochs 10 \
        --itr 1 \
        --batch_size 16

done
