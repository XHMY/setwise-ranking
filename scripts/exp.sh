
dataset=20
model=meta-llama/Meta-Llama-3-8B-Instruct

# Pointwise
python3 run.py run --model_name_or_path $model \
  --tokenizer_name_or_path $model \
  --run_path "data/run.msmarco-v1-passage.bm25-default.dl${dataset}.txt" \
  --save_path outputs/run.pointwise.yes_no.txt \
  --ir_dataset_name "msmarco-passage/trec-dl-20${dataset}" \
  --hits 100 --query_length 32 --passage_length 128 \
  --device cuda pointwise --method yes_no --batch_size 32

# Listwise
python3 run.py run --model_name_or_path $model \
  --tokenizer_name_or_path $model \
  --run_path "data/run.msmarco-v1-passage.bm25-default.dl${dataset}.txt" \
  --save_path outputs/run.liswise.generation.txt \
  --ir_dataset_name "msmarco-passage/trec-dl-20${dataset}" \
  --hits 100 --query_length 32 --passage_length 100 \
  --scoring generation --device cuda \
  listwise --window_size 4 --step_size 2 --num_repeat 5

# Setwise
python3 run.py run --model_name_or_path $model \
  --tokenizer_name_or_path $model \
  --run_path "data/run.msmarco-v1-passage.bm25-default.dl${dataset}.txt" \
  --save_path outputs/run.setwise.heapsort.txt \
  --ir_dataset_name "msmarco-passage/trec-dl-20${dataset}" \
  --hits 100 --query_length 32 --passage_length 128 \
  --scoring generation --device cuda \
  setwise --num_child 2 --method heapsort --k 10