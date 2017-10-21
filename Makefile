.PHONY: clean data lint requirements product_info big_sample big_sample_vgg16_vecs big_sample_resnet50_vecs \
	test_vgg16_vecs train_vgg16_vecs test_resnet50_vecs category_indexes top_2000_sample top_3000_sample \
	vgg16_head_top_2000_v1 vgg16_head_top_2000_v2 vgg16_head_top_2000_v3 vgg16_head_top_2000_v4 vgg16_head_top_2000_v5 \
	vgg16_head_top_2000_v6 vgg16_head_top_2000_v7 vgg16_head_top_2000_v8 vgg16_head_top_2000_v9 vgg16_head_top_2000_v10 \
	vgg16_head_top_2000_v11 vgg16_head_top_3000_v1 vgg16_head_top_3000_v2 vgg16_head_full_v1 vgg16_head_full_v2

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = kaggle-cdiscount-image-classification
PYTHON_INTERPRETER = python3
include .env

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	pipenv install

## Delete all compiled Python files
clean:
	find . -name "*.pyc" -exec rm {} \;

## Lint using flake8
lint:
	flake8 --exclude=lib/,bin/,docs/conf.py .

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Run through dataset and compile csv with products information
product_info: ${DATA_INTERIM}/train_product_info.csv ${DATA_INTERIM}/test_product_info.csv

${DATA_INTERIM}/train_product_info.csv:
	pipenv run $(PYTHON_INTERPRETER) src/data/product_info.py --bson ${TRAIN_BSON} \
		--output_file ${DATA_INTERIM}/train_product_info.csv

${DATA_INTERIM}/test_product_info.csv:
	pipenv run $(PYTHON_INTERPRETER) src/data/product_info.py --bson ${TEST_BSON} \
		--without_categories --output_file ${DATA_INTERIM}/test_product_info.csv

## Create stratified sample with 200000 products
big_sample: ${DATA_INTERIM}/big_sample_product_info.csv

${DATA_INTERIM}/big_sample_product_info.csv: ${DATA_INTERIM}/train_product_info.csv
	pipenv run $(PYTHON_INTERPRETER) src/data/big_sample.py --prod_info_csv ${DATA_INTERIM}/train_product_info.csv \
		--output_file ${DATA_INTERIM}/big_sample_product_info.csv

## Precompute VGG16 vectors for big sample
big_sample_vgg16_vecs: ${DATA_INTERIM}/big_sample_product_info.csv
	pipenv run $(PYTHON_INTERPRETER) -m src.model.vgg16_vecs --bson ${TRAIN_BSON} \
		--prod_info_csv ${DATA_INTERIM}/big_sample_product_info.csv \
		--output_dir ${DATA_INTERIM}/big_sample_vgg16_vecs \
		--save_step 100000 \
		--only_first_image

## Precompute ResNet50 vectors for big sample
big_sample_resnet50_vecs: ${DATA_INTERIM}/big_sample_product_info.csv
	pipenv run $(PYTHON_INTERPRETER) -m src.model.resnet50_vecs --bson ${TRAIN_BSON} \
		--prod_info_csv ${DATA_INTERIM}/big_sample_product_info.csv \
		--output_dir ${DATA_INTERIM}/big_sample_resnet50_vecs \
		--save_step 100000 \
		--only_first_image

## Precompute VGG16 vectors for test dataset
test_vgg16_vecs: ${DATA_INTERIM}/test_product_info.csv
	pipenv run $(PYTHON_INTERPRETER) -m src.model.vgg16_vecs --bson ${TEST_BSON} \
		--prod_info_csv ${DATA_INTERIM}/test_product_info.csv \
		--output_dir ${TEST_VGG16_VECS_PATH} \
		--save_step 100000

## Precompute VGG16 vectors for train dataset
train_vgg16_vecs: ${DATA_INTERIM}/train_product_info.csv
	pipenv run $(PYTHON_INTERPRETER) -m src.model.vgg16_vecs --bson ${TRAIN_BSON} \
		--prod_info_csv ${DATA_INTERIM}/train_product_info.csv \
		--output_dir ${TRAIN_VGG16_VECS_PATH} \
		--save_step 100000 \
		--shuffle 123

## Precompute ResNet50 vectors for test dataset
test_resnet50_vecs: ${DATA_INTERIM}/test_product_info.csv
	pipenv run $(PYTHON_INTERPRETER) -m src.model.resnet50_vecs --bson ${TEST_BSON} \
		--prod_info_csv ${DATA_INTERIM}/test_product_info.csv \
		--output_dir ${TEST_RESNET50_VECS_PATH} \
		--save_step 100000

## Create category indexes
category_indexes: ${DATA_INTERIM}/category_idx.csv

${DATA_INTERIM}/category_idx.csv: ${DATA_INTERIM}/train_product_info.csv
	pipenv run $(PYTHON_INTERPRETER) -m src.data.category_idx --prod_info_csv ${DATA_INTERIM}/train_product_info.csv \
		--output_file ${DATA_INTERIM}/category_idx.csv

## Create top 2000 categories sample
top_2000_sample: ${DATA_INTERIM}/top_2000_sample_product_info.csv

${DATA_INTERIM}/top_2000_sample_product_info.csv: ${DATA_INTERIM}/train_product_info.csv
	pipenv run $(PYTHON_INTERPRETER) -m src.data.top_categories_sample \
		--prod_info_csv ${DATA_INTERIM}/train_product_info.csv \
		--output_file ${DATA_INTERIM}/top_2000_sample_product_info.csv \
		--num_categories 2000

## Create top 3000 categories sample
top_3000_sample: ${DATA_INTERIM}/top_3000_sample_product_info.csv

${DATA_INTERIM}/top_3000_sample_product_info.csv: ${DATA_INTERIM}/train_product_info.csv
	pipenv run $(PYTHON_INTERPRETER) -m src.data.top_categories_sample \
		--prod_info_csv ${DATA_INTERIM}/train_product_info.csv \
		--output_file ${DATA_INTERIM}/top_3000_sample_product_info.csv \
		--num_categories 3000

${DATA_INTERIM}/train_split.csv: ${DATA_INTERIM}/train_product_info.csv
	pipenv run $(PYTHON_INTERPRETER) -m src.data.train_split \
		--prod_info_csv ${DATA_INTERIM}/train_product_info.csv \
		--output_file ${DATA_INTERIM}/train_split.csv

## Train head dense layer of VGG16 on top 2000 categories V1
vgg16_head_top_2000_v1: ${DATA_INTERIM}/top_2000_sample_product_info.csv ${DATA_INTERIM}/category_idx.csv \
${DATA_INTERIM}/train_split.csv
	pipenv run $(PYTHON_INTERPRETER) -m src.model.tune_vgg16_vecs --fit \
		--bcolz_root ${TRAIN_VGG16_VECS_PATH} \
		--bcolz_prod_info_csv ${DATA_INTERIM}/train_product_info.csv \
		--sample_prod_info_csv ${DATA_INTERIM}/top_2000_sample_product_info.csv \
		--category_idx_csv ${DATA_INTERIM}/category_idx.csv \
		--train_split_csv ${DATA_INTERIM}/train_split.csv \
        --models_dir models/vgg16_head_top_2000_v1 \
		--batch_size 250 \
		--lr 0.001 \
		--epochs 3 \
		--shuffle 123 \
		--mode 0

## Train head dense layer of VGG16 on top 2000 categories V2
vgg16_head_top_2000_v2: ${DATA_INTERIM}/top_2000_sample_product_info.csv ${DATA_INTERIM}/category_idx.csv \
${DATA_INTERIM}/train_split.csv
	pipenv run $(PYTHON_INTERPRETER) -m src.model.tune_vgg16_vecs --fit \
		--bcolz_root ${TRAIN_VGG16_VECS_PATH} \
		--bcolz_prod_info_csv ${DATA_INTERIM}/train_product_info.csv \
		--sample_prod_info_csv ${DATA_INTERIM}/top_2000_sample_product_info.csv \
		--category_idx_csv ${DATA_INTERIM}/category_idx.csv \
		--train_split_csv ${DATA_INTERIM}/train_split.csv \
        --models_dir models/vgg16_head_top_2000_v2 \
		--batch_size 250 \
		--lr 0.0001 \
		--epochs 3 \
		--shuffle 123 \
		--mode 0

## Train head dense layer of VGG16 on top 2000 categories V3
vgg16_head_top_2000_v3: ${DATA_INTERIM}/top_2000_sample_product_info.csv ${DATA_INTERIM}/category_idx.csv \
${DATA_INTERIM}/train_split.csv
	pipenv run $(PYTHON_INTERPRETER) -m src.model.tune_vgg16_vecs --fit \
		--bcolz_root ${TRAIN_VGG16_VECS_PATH} \
		--bcolz_prod_info_csv ${DATA_INTERIM}/train_product_info.csv \
		--sample_prod_info_csv ${DATA_INTERIM}/top_2000_sample_product_info.csv \
		--category_idx_csv ${DATA_INTERIM}/category_idx.csv \
		--train_split_csv ${DATA_INTERIM}/train_split.csv \
        --models_dir models/vgg16_head_top_2000_v3 \
		--batch_size 250 \
		--lr 0.001 \
		--epochs 3 \
		--shuffle 123 \
		--mode 1

## Train head dense layer of VGG16 on top 2000 categories V4
vgg16_head_top_2000_v4: ${DATA_INTERIM}/top_2000_sample_product_info.csv ${DATA_INTERIM}/category_idx.csv \
${DATA_INTERIM}/train_split.csv
	pipenv run $(PYTHON_INTERPRETER) -m src.model.tune_vgg16_vecs --fit \
		--bcolz_root ${TRAIN_VGG16_VECS_PATH} \
		--bcolz_prod_info_csv ${DATA_INTERIM}/train_product_info.csv \
		--sample_prod_info_csv ${DATA_INTERIM}/top_2000_sample_product_info.csv \
		--category_idx_csv ${DATA_INTERIM}/category_idx.csv \
		--train_split_csv ${DATA_INTERIM}/train_split.csv \
        --models_dir models/vgg16_head_top_2000_v4 \
		--batch_size 250 \
		--lr 0.001 \
		--epochs 3 \
		--shuffle 123 \
		--mode 2

## Train head dense layer of VGG16 on top 2000 categories V5
vgg16_head_top_2000_v5: ${DATA_INTERIM}/top_2000_sample_product_info.csv ${DATA_INTERIM}/category_idx.csv \
${DATA_INTERIM}/train_split.csv
	pipenv run $(PYTHON_INTERPRETER) -m src.model.tune_vgg16_vecs --fit \
		--bcolz_root ${TRAIN_VGG16_VECS_PATH} \
		--bcolz_prod_info_csv ${DATA_INTERIM}/train_product_info.csv \
		--sample_prod_info_csv ${DATA_INTERIM}/top_2000_sample_product_info.csv \
		--category_idx_csv ${DATA_INTERIM}/category_idx.csv \
		--train_split_csv ${DATA_INTERIM}/train_split.csv \
        --models_dir models/vgg16_head_top_2000_v5 \
		--batch_size 250 \
		--lr 0.001 \
		--epochs 3 \
		--shuffle 123 \
		--mode 3

## Train head dense layer of VGG16 on top 2000 categories V6
vgg16_head_top_2000_v6: ${DATA_INTERIM}/top_2000_sample_product_info.csv ${DATA_INTERIM}/category_idx.csv \
${DATA_INTERIM}/train_split.csv
	pipenv run $(PYTHON_INTERPRETER) -m src.model.tune_vgg16_vecs --fit \
		--bcolz_root ${TRAIN_VGG16_VECS_PATH} \
		--bcolz_prod_info_csv ${DATA_INTERIM}/train_product_info.csv \
		--sample_prod_info_csv ${DATA_INTERIM}/top_2000_sample_product_info.csv \
		--category_idx_csv ${DATA_INTERIM}/category_idx.csv \
		--train_split_csv ${DATA_INTERIM}/train_split.csv \
        --models_dir models/vgg16_head_top_2000_v6 \
		--batch_size 250 \
		--lr 0.001 \
		--epochs 3 \
		--shuffle 123 \
		--mode 4

## Train head dense layer of VGG16 on top 2000 categories V7
vgg16_head_top_2000_v7: ${DATA_INTERIM}/top_2000_sample_product_info.csv ${DATA_INTERIM}/category_idx.csv \
${DATA_INTERIM}/train_split.csv
	pipenv run $(PYTHON_INTERPRETER) -m src.model.tune_vgg16_vecs --fit \
		--bcolz_root ${TRAIN_VGG16_VECS_PATH} \
		--bcolz_prod_info_csv ${DATA_INTERIM}/train_product_info.csv \
		--sample_prod_info_csv ${DATA_INTERIM}/top_2000_sample_product_info.csv \
		--category_idx_csv ${DATA_INTERIM}/category_idx.csv \
		--train_split_csv ${DATA_INTERIM}/train_split.csv \
        --models_dir models/vgg16_head_top_2000_v7 \
		--batch_size 250 \
		--lr 0.001 \
		--epochs 3 \
		--shuffle 123 \
		--mode 5

## Train head dense layer of VGG16 on top 2000 categories V8
vgg16_head_top_2000_v8: ${DATA_INTERIM}/top_2000_sample_product_info.csv ${DATA_INTERIM}/category_idx.csv \
${DATA_INTERIM}/train_split.csv
	pipenv run $(PYTHON_INTERPRETER) -m src.model.tune_vgg16_vecs --fit \
		--bcolz_root ${TRAIN_VGG16_VECS_PATH} \
		--bcolz_prod_info_csv ${DATA_INTERIM}/train_product_info.csv \
		--sample_prod_info_csv ${DATA_INTERIM}/top_2000_sample_product_info.csv \
		--category_idx_csv ${DATA_INTERIM}/category_idx.csv \
		--train_split_csv ${DATA_INTERIM}/train_split.csv \
        --models_dir models/vgg16_head_top_2000_v8 \
		--batch_size 250 \
		--lr 0.01 \
		--epochs 3 \
		--shuffle 123 \
		--mode 6

## Train head dense layer of VGG16 on top 2000 categories V9
vgg16_head_top_2000_v9: ${DATA_INTERIM}/top_2000_sample_product_info.csv ${DATA_INTERIM}/category_idx.csv \
${DATA_INTERIM}/train_split.csv
	pipenv run $(PYTHON_INTERPRETER) -m src.model.tune_vgg16_vecs --fit \
		--bcolz_root ${TRAIN_VGG16_VECS_PATH} \
		--bcolz_prod_info_csv ${DATA_INTERIM}/train_product_info.csv \
		--sample_prod_info_csv ${DATA_INTERIM}/top_2000_sample_product_info.csv \
		--category_idx_csv ${DATA_INTERIM}/category_idx.csv \
		--train_split_csv ${DATA_INTERIM}/train_split.csv \
        --models_dir models/vgg16_head_top_2000_v9 \
		--batch_size 250 \
		--lr 0.001 \
		--epochs 3 \
		--shuffle 123 \
		--mode 7

## Train head dense layer of VGG16 on top 2000 categories V10
vgg16_head_top_2000_v10: ${DATA_INTERIM}/top_2000_sample_product_info.csv ${DATA_INTERIM}/category_idx.csv \
${DATA_INTERIM}/train_split.csv
	pipenv run $(PYTHON_INTERPRETER) -m src.model.tune_vgg16_vecs --fit \
		--bcolz_root ${TRAIN_VGG16_VECS_PATH} \
		--bcolz_prod_info_csv ${DATA_INTERIM}/train_product_info.csv \
		--sample_prod_info_csv ${DATA_INTERIM}/top_2000_sample_product_info.csv \
		--category_idx_csv ${DATA_INTERIM}/category_idx.csv \
		--train_split_csv ${DATA_INTERIM}/train_split.csv \
        --models_dir models/vgg16_head_top_2000_v10 \
		--batch_size 250 \
		--lr 0.001 \
		--epochs 3 \
		--shuffle 123 \
		--mode 8 \
		--batch_seed 518

## Train head dense layer of VGG16 on top 2000 categories V11
vgg16_head_top_2000_v11: ${DATA_INTERIM}/top_2000_sample_product_info.csv ${DATA_INTERIM}/category_idx.csv \
${DATA_INTERIM}/train_split.csv
	pipenv run $(PYTHON_INTERPRETER) -m src.model.tune_vgg16_vecs --fit \
		--bcolz_root ${TRAIN_VGG16_VECS_PATH} \
		--bcolz_prod_info_csv ${DATA_INTERIM}/train_product_info.csv \
		--sample_prod_info_csv ${DATA_INTERIM}/top_2000_sample_product_info.csv \
		--category_idx_csv ${DATA_INTERIM}/category_idx.csv \
		--train_split_csv ${DATA_INTERIM}/train_split.csv \
        --models_dir models/vgg16_head_top_2000_v11 \
		--batch_size 64 \
		--lr 0.001 \
		--epochs 3 \
		--shuffle 123 \
		--mode 2 \
		--batch_seed 438

## Train head dense layer of VGG16 on top 3000 categories V1
vgg16_head_top_3000_v1: ${DATA_INTERIM}/top_3000_sample_product_info.csv ${DATA_INTERIM}/category_idx.csv \
${DATA_INTERIM}/train_split.csv
	pipenv run $(PYTHON_INTERPRETER) -m src.model.tune_vgg16_vecs --fit \
		--bcolz_root ${TRAIN_VGG16_VECS_PATH} \
		--bcolz_prod_info_csv ${DATA_INTERIM}/train_product_info.csv \
		--sample_prod_info_csv ${DATA_INTERIM}/top_3000_sample_product_info.csv \
		--category_idx_csv ${DATA_INTERIM}/category_idx.csv \
		--train_split_csv ${DATA_INTERIM}/train_split.csv \
        --models_dir models/vgg16_head_top_3000_v1 \
		--batch_size 250 \
		--lr 0.001 \
		--epochs 3 \
		--shuffle 123 \
		--mode 2 \
		--batch_seed 812

## Train head dense layer of VGG16 on top 3000 categories V2
vgg16_head_top_3000_v2: ${DATA_INTERIM}/top_3000_sample_product_info.csv ${DATA_INTERIM}/category_idx.csv \
${DATA_INTERIM}/train_split.csv
	pipenv run $(PYTHON_INTERPRETER) -m src.model.tune_vgg16_vecs --fit \
		--bcolz_root ${TRAIN_VGG16_VECS_PATH} \
		--bcolz_prod_info_csv ${DATA_INTERIM}/train_product_info.csv \
		--sample_prod_info_csv ${DATA_INTERIM}/top_3000_sample_product_info.csv \
		--category_idx_csv ${DATA_INTERIM}/category_idx.csv \
		--train_split_csv ${DATA_INTERIM}/train_split.csv \
        --models_dir models/vgg16_head_top_3000_v2 \
		--batch_size 250 \
		--lr 0.001 \
		--epochs 4 \
		--shuffle 123 \
		--mode 3 \
		--batch_seed 813

## Train head dense layer of VGG16 on all categories V1
vgg16_head_full_v1: ${DATA_INTERIM}/train_product_info.csv ${DATA_INTERIM}/category_idx.csv \
${DATA_INTERIM}/train_split.csv
	pipenv run $(PYTHON_INTERPRETER) -m src.model.tune_vgg16_vecs --fit \
		--bcolz_root ${TRAIN_VGG16_VECS_PATH} \
		--bcolz_prod_info_csv ${DATA_INTERIM}/train_product_info.csv \
		--sample_prod_info_csv ${DATA_INTERIM}/train_product_info.csv \
		--category_idx_csv ${DATA_INTERIM}/category_idx.csv \
		--train_split_csv ${DATA_INTERIM}/train_split.csv \
        --models_dir models/vgg16_head_full_v1 \
		--batch_size 250 \
		--lr 0.001 \
		--epochs 3 \
		--shuffle 123 \
		--mode 2 \
		--batch_seed 814

## Train head dense layer of VGG16 on all categories V2
vgg16_head_full_v2: ${DATA_INTERIM}/train_product_info.csv ${DATA_INTERIM}/category_idx.csv \
${DATA_INTERIM}/train_split.csv
	pipenv run $(PYTHON_INTERPRETER) -m src.model.tune_vgg16_vecs --fit \
		--bcolz_root ${TRAIN_VGG16_VECS_PATH} \
		--bcolz_prod_info_csv ${DATA_INTERIM}/train_product_info.csv \
		--sample_prod_info_csv ${DATA_INTERIM}/train_product_info.csv \
		--category_idx_csv ${DATA_INTERIM}/category_idx.csv \
		--train_split_csv ${DATA_INTERIM}/train_split.csv \
        --models_dir models/vgg16_head_full_v2 \
		--batch_size 250 \
		--lr 0.001 \
		--epochs 3 \
		--shuffle 123 \
		--mode 3 \
		--batch_seed 815

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := show-help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: show-help
show-help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
