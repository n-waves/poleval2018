#!/usr/bin/env bash
set -e

TASK3_DATA_DIR="../data/task3/train"
SENTENCE_FILE="${TASK3_DATA_DIR}/task3_train_segmented.txt"
TEMP_DIR="../work/most_low30k/tmp"
OUTPUT_DIR="../work/most_low30k/tmp"
SENTENCEPIECE_MODEL_NAME="${OUTPUT_DIR}/sp-30k"
DICTIONARY_FILE="${TEMP_DIR}/word_freq.pkl"

# sort sentences and remove duplicates
mkdir -p "${TEMP_DIR}"
UNIQ_SENTENCE_FILE="${TEMP_DIR}/task3_train_uniq.txt"
if [ ! -f "${UNIQ_SENTENCE_FILE}" ] ; then
    sort "${SENTENCE_FILE}" | uniq > "${UNIQ_SENTENCE_FILE}"
fi


LOWERCASE_SENTENCE_FILE="${TEMP_DIR}/task3_train_lowercase.txt"
ESCAPED_SENTENCE_FILE="${TEMP_DIR}/task3_train_escaped.txt"

# creates "${DICTIONARY_FILE}"
[ -f "${DICTIONARY_FILE}" ] || python ./extract-dict.py --sentence-file "${SENTENCE_FILE}" --output "${DICTIONARY_FILE}"

# creates "${TEMP_DIR}/escaped.txt"
[ -f "${ESCAPED_SENTENCE_FILE}" ] || python ./cap-to-dict.py --sentence-file "${UNIQ_SENTENCE_FILE}" --lower-case=False --dictionary-file "${DICTIONARY_FILE}" --output "${ESCAPED_SENTENCE_FILE}"

# creates ${LOWERCASE_SENTECE_FILE}
[ -f "${LOWERCASE_SENTENCE_FILE}" ] || python ./escape-caps.py --most-low True --sentence_file "${ESCAPED_SENTENCE_FILE}" --output "${LOWERCASE_SENTENCE_FILE}"


# train sentencepiece model
if [ ! -f "${SENTENCEPIECE_MODEL_NAME}.model" ]; then
  spm_train --input="${LOWERCASE_SENTENCE_FILE}" --model_prefix="${SENTENCEPIECE_MODEL_NAME}"\
            --vocab_size=30000\
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

if [ ! -f ${IDS_FILE} ]; then
    mkdir -p "${PARTS_DIR}"
    rm -fr "${PARTS_DIR}"/sentence_part-*
    split -n "l/${SPM_PROCESSES}" "${LOWERCASE_SENTENCE_FILE}" "${PARTS_DIR}/sentence_part-"
    ls "${PARTS_DIR}"/sentence_part-* | xargs '-I{}' -P "${SPM_PROCESSES}" -n 1 spm_encode --model="${SENTENCEPIECE_MODEL_NAME}.model" --extra_options=bos:eos --output_format=id '--output={}.ids' '{}'
    cat "${PARTS_DIR}"/sentence_part-*.ids > "${IDS_FILE}"
fi

if [ ! -f "${OUTPUT_DIR}/val_ids.npy" ] || [ ! -f "${OUTPUT_DIR}/trn_ids.npy" ]; then
    # creates "${OUTPUT_DIR}/val_ids.npy" and "${OUTPUT_DIR}/trn_ids.npy"
    python ./split-datasets.py --ids-file "${IDS_FILE}" --output-path "${OUTPUT_DIR}"
fi

TEST_FILE="../data/task3/test/task3_test_segmented.txt"
ESCAPED_TEST_FILE="${TEMP_DIR}/task3_test_escaped.txt"
LOWERCASE_TEST_FILE="${TEMP_DIR}/task3_test_lowercase.txt"
TEST_IDS_FILE="${TEMP_DIR}/test_ids.txt"

# creates "${TEMP_DIR}/escaped_test.txt"
[ -f "${ESCAPED_TEST_FILE}" ] || python ./cap-to-dict.py --sentence-file "${TEST_FILE}" --lower-case=False --dictionary-file "${DICTIONARY_FILE}" --output "${ESCAPED_TEST_FILE}"

# creates ${LOWERCASE_TEST_FILE}
[ -f "${LOWERCASE_TEST_FILE}" ] || python ./escape-caps.py --sentence_file "${ESCAPED_TEST_FILE}" --output "${LOWERCASE_TEST_FILE}"

if [ ! -f ${TEST_IDS_FILE} ]; then
    mkdir -p "${PARTS_DIR}"
    rm -fr "${PARTS_DIR}"/sentence_part-*
    split -n "l/${SPM_PROCESSES}" "${LOWERCASE_TEST_FILE}" "${PARTS_DIR}/sentence_part-"
    ls "${PARTS_DIR}"/sentence_part-* | xargs '-I{}' -P "${SPM_PROCESSES}" -n 1 spm_encode --model="${SENTENCEPIECE_MODEL_NAME}.model" --extra_options=bos:eos --output_format=id '--output={}.ids' '{}'
    cat "${PARTS_DIR}"/sentence_part-*.ids > "${TEST_IDS_FILE}"
fi

if [ ! -f "${OUTPUT_DIR}/test_ids.npy" ]; then
    python ./split-datasets.py --ids-file "${TEST_IDS_FILE}" --output-path "${OUTPUT_DIR}" --test-set=True
fi