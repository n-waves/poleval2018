#!/usr/bin/env bash
set -e

TASK3_DATA_DIR="../data/task3/train"
SENTENCE_FILE="${TASK3_DATA_DIR}/task3_train_segmented.txt"
TEMP_DIR="../work/just_low/tmp"
OUTPUT_DIR="../work/just_low/tmp"
SENTENCEPIECE_MODEL_NAME="${OUTPUT_DIR}/sp-100k"
DICTIONARY_FILE="${TEMP_DIR}/word_freq.pkl"

# sort sentences and remove duplicates
mkdir -p "${TEMP_DIR}"
UNIQ_SENTENCE_FILE="${TEMP_DIR}/task3_train_uniq.txt"
if [ ! -f "${UNIQ_SENTENCE_FILE}" ] ; then
    sort "${SENTENCE_FILE}" | uniq > "${UNIQ_SENTENCE_FILE}"
fi


LOWERCASE_SENTENCE_FILE="${TEMP_DIR}/task3_train_lowercase.txt"

# creates ${LOWERCASE_SENTECE_FILE}
[ -f "${LOWERCASE_SENTENCE_FILE}" ] || python ./escape-caps.py --sentence_file "${UNIQ_SENTENCE_FILE}" --output "${LOWERCASE_SENTENCE_FILE}"

# creates "${DICTIONARY_FILE}"
[ -f "${DICTIONARY_FILE}" ] || python ./extract-dict.py --sentence-file "${SENTENCE_FILE}" --output "${DICTIONARY_FILE}"

# creates "${TEMP_DIR}/escaped.txt"
[ -f "${TEMP_DIR}/escaped.txt" ] || python ./cap-to-dict.py --sentence-file "${LOWERCASE_SENTENCE_FILE}" --lower-case=True --dictionary-file "${DICTIONARY_FILE}" --output "${TEMP_DIR}/escaped.txt"

# train sentencepiece model
if [ ! -f "${SENTENCEPIECE_MODEL_NAME}.model" ]; then
  spm_train --input="${LOWERCASE_SENTENCE_FILE}" --model_prefix="${SENTENCEPIECE_MODEL_NAME}"\
            --vocab_size=100000\
            --unk_id=0 --pad_id=1 --bos_id=2 --eos_id=3\
            --input_sentence_size=30000000\
            --model_type=unigram\
            --normalization_rule_name=nmt_nfkc_cf\
            --num_threads=10\
            --user_defined_symbols='<up>'
else
  echo "Setencepiece model '${SENTENCEPIECE_MODEL_NAME}.model' already exists."
fi

# running spm_encode in parallel
SPM_PROCESSES=8
PARTS_DIR="${TEMP_DIR}/parts"
IDS_FILE="${TEMP_DIR}/ids.txt"
mkdir -p "${PARTS_DIR}"
rm -fr "${PARTS_DIR}"/sentence_part-*
split -n "l/${SPM_PROCESSES}" "${TEMP_DIR}/escaped.txt" "${PARTS_DIR}/sentence_part-"

ls "${PARTS_DIR}"/sentence_part-* | xargs '-I{}' -P "${SPM_PROCESSES}" -n 1 spm_encode --model="${SENTENCEPIECE_MODEL_NAME}.model" --extra_options=bos:eos --output_format=id '--output={}.ids' '{}'
cat "${PARTS_DIR}"/sentence_part-*.ids > "${IDS_FILE}"

# creates "${OUTPUT_DIR}/val_ids.npy" and "${OUTPUT_DIR}/trn_ids.npy"
python ./split-datasets.py --ids-file "${IDS_FILE}" --output-path "${OUTPUT_DIR}"
