.PHONY: clean data lint requirements product_info big_sample big_sample_vgg16_vecs big_sample_resnet50_vecs

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
