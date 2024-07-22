# VLDB 2024: Proto-X

Source code for the Proto-X framework in "The Holon Approach for Simultaneously Tuning Multiple
Components in a Self-Driving Database Management System with Machine Learning via Synthesized
Proto-Actions", appearing in VLDB 2024.

### Environment Setup

We assume Ubuntu 22.04 when executing the following commands:

```
$ conda create -n protox python=3.9
$ conda activate protox
$ conda env config vars set PYTHONNOUSERSITE=1
$ conda deactivate && conda activate protox
$ pip install -r requirements.txt
```

### Preparing a Postgres Data Archive

1. Create/Load the initial database contents and the initial starting configuration.
2. Clean the database by optionally running `VACUUM FULL` followed by `VACUUM` and `ANALYZE`.
3. Shutdown Postgres and navigate to the directory containing the database data (i.e., pgdata)
4. `tar cf pgdata.tgz <database data>`

### Generating Embeddings

1. Modify the following values within `embeddings/config.json` to suit specific purposes:
    - `subspaces/latent_dim`: Dimension of latent dimension
    - `subspaces/hidden_sizes`: Hidden Layers
    - `metric_loss_md/bias_separation`: The size (coverage) of each discrete embedding layer.
    - `metric_loss_md/addtl_bias_separation`: Additional padding to insert between each embedding layer.
    - `num_epochs`: Number of epochs to train embeddings
    - `results/latent_spaces/spaces/*/*/config` provides example configuration parameters.
2. Invoke the following command (with postgres already running and `HypoPG` available) to generate training data:
```
$ conda activate protox
$ rm -rf <data_output_directory>
$ python3 index_embedding.py generate \
    --config configs/config.yaml \
    --benchmark-config configs/benchmark/<benchmark>.yaml \
    --generate-costs \
    --connection <postgres connection> \
    --num-processes <number processes> \
    --output-dir <data_output_directory> \
    --per-table \
    --table <comma separated table names, REQUIRED if --per-table not specified> \
    --batch-limit <number or comma separated number (corresponding to --table) of samples> \
    --per-leading-col <comma separated table names to sample (table, attr) indexes on, OPTIONAL> \
```
3. Next, invoke the following to generate embeddings:
```
$ OMP_NUM_THREADS=<# threads for each training job> ray start --head --num-cpus=<# parallel jobs>
$ rm -rf <embed_output_directory> && mkdir <embed_output_directory>
$ python3 index_embedding.py train \
    --output-dir <absolute path to embed_output_directory> \
    --input-data <absolute path to data_output_directory> \
    --config embeddings/config.json \
    --num-trials <number embeddings to create> \
    --benchmark-config <absolute path to configs/benchmark/<benchmark>.yaml> \
    --train-size <size of training data, i.e., 0.9> \
    --iterations-per-epoch <number of batches in an epoch, i.e., 1000> \
    --max-concurrent <# parallel jobs> \
    --mythril-dir <root of proto-x folder> \
    --num-threads <# threads for each training job> \
    --inflate-ratio <OPTIONAL, inflate input data by certain ratio, i.e., 80> \
    --rebias <OPTIONAL, force separates each "index class" by rebias distance. Useful when "benefits" are very similar>
$ ray stop -f
```
4. To curate/select the best embeddings:
```
$ python3 scripts/redist.py --input <absolute path to embed_output_directory> --num-parts 1
$ python3 index_embedding.py eval \
    --models <absolute path to embed_output_directory> \
    --benchmark-config <absolute path to configs/benchmark/<benchmark>.yaml> \
    --start-epoch 0 \
    --num-batches 100 \
    --batch-size <size of each batch to test, defaults to 8192> \
    --dataset <absolute path to the out.parquet within embed_output_directory>
$ python3 scripts/embed_analyze.py \
    --base <embed_output_directory>/part0 \
    --num-points <number of points to sample to approximate distribution> \
    --benchmark-config configs/benchmark/<benchmark>.yaml \
    --start-epoch 0 \
    --top <top-k index distribution to track, i.e., 5> \
    --max-segments <number of embedding layers to analyze, i.e., 15>
$ python3 scripts/embed_filter.py \
    --idx-limit <--max-segments from embed_analyze.py> \
    --curate \
    --num-curate <number of embeddings to select> \
    --flatten --flatten-idx 0 \
    --input <embed_output_directory> \
    --out <embed_output_directory>/out.csv \
    --data <embed_output_directory>/out.parquet
```
5. Best embeddings will be placed at `<embed_output_directory>/curated`
6. ray stop -f

### Tuning the DBMS

Assuming that there's a valid configuration generated from a prior tuning job (i.e., from `results`), you can
directly utilize `hpo.py` to run the targeted configuration. **Note** that while some arguments in `hpo.py`
are repeated in the `--initial-configs` file, the argument values in `--initial-configs` take *precedence*
over the command-line arguments.

```
$ OMP_NUM_THREADS=<# threads for each parallel tuning> ray start --head --num-cpus=<# parallel jobs>
$ python3 hpo.py
    --config <absolute path of root>/configs/config.yaml \
    --agent wolp \
    --model-config <absolute path of root>/configs/wolp_params.yaml \
    --benchmark-config <absolute path of root>/configs/benchmark/<benchmark>.yaml \
    --mythril-dir <absolute path of root> \
    --num-trials <number of runs> \
    --max-concurrent <# parallel tuning jobs> \
    --max-iterations 1000 \
    --horizon 5 \
    --duration 30.0 \
    --target latency \
    --data-snapshot-path <absolute path to data archive> \
    --workload-timeout 600 \
    --timeout 30 \
    --benchbase-config-path <benchbase config path or a dummy valid path> \
    --initial-configs <list of configurations to run> \
    --initial-repeats 1
```
