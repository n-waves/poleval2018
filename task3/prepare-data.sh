#!/usr/bin/env bash
set -e

die() { echo "$*" 1>&2 ; exit 1; }

force_bool() { [ "$2" != "True" -a "$2" != "False"] && die "$1 must be 'True' of 'False', but '$2' was given" }

while [[ $# -gt 0 ]]
do
    case "$1" in
        --work-dir)
            WORK_DIR="$2"
            shift
            shift
            ;;
        --vocab-size)
            VOCAB_SIZE="$2"
            shift
            shift
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift
            shift
            ;;
        --user-defined-symbols)
            USER_DEF_SYMBOLS="$2"
            shift
            shift
            ;;
        --lower-case)
            LOWER_CASE="$2"
            shift
            shift
            force_bool '--lower-case' "${LOWER_CASE}"
            ;;
        --most-low)
            MOST_LOW="$2"
            shift
            shift
            force_bool '--most-low' "${MOST_LOW}"
            ;;
        *)
            die "Unknown argument: '$1'"
            ;;
    esac
done

[ -z ${WORK_DIR+x} -o -z ${VOCAB_SIZE+x} -i -z ${MODEL_NAME+X} ] && die "--work-dir or --vocab-size is missing"

TASK3_DATA_DIR="../data/task3/train"
SENTENCE_FILE="${TASK3_DATA_DIR}/task3_train_segmented.txt"
TEMP_DIR="${WORK_DIR}/tmp"
OUTPUT_DIR="${WORK_DIR}/tmp"
SENTENCEPIECE_MODEL_NAME="${OUTPUT_DIR}/${MODEL_NAME}"
DICTIONARY_FILE="${TEMP_DIR}/word_freq.pkl"

# sort sentences and remove duplicates
mkdir -p "${TEMP_DIR}"
UNIQ_SENTENCE_FILE="${TEMP_DIR}/task3_train_uniq.txt"
if [ ! -f "${UNIQ_SENTENCE_FILE}" ] ; then
    sort "${SENTENCE_FILE}" | uniq > "${UNIQ_SENTENCE_FILE}"
fi
# creates "${DICTIONARY_FILE}"
[ -f "${DICTIONARY_FILE}" ] || python ./extract-dict.py --sentence-file "${SENTENCE_FILE}" --output "${DICTIONARY_FILE}"

if [ "${LOWER_CASE}" = "True" ]; then
    LOWERCASE_SENTENCE_FILE="${TEMP_DIR}/task3_train_lowercase.txt"
    [ -f "${LOWERCASE_SENTENCE_FILE}" ] || python ./escape-caps.py --sentence_file "${UNIQ_SENTENCE_FILE}" --output "${LOWERCASE_SENTENCE_FILE}" --most-low "${MOST_LOW}"
    INPUT_SENTENCE_FILE="${LOWERCASE_SENTENCE_FILE}"
else
    INPUT_SENTENCE_FILE="${UNIQ_SENTENCE_FILE}"
fi

# creates "${TEMP_DIR}/escaped.txt"
[ -f "${TEMP_DIR}/escaped.txt" ] || python ./cap-to-dict.py --sentence-file "${INPUT_SENTENCE_FILE}" --dictionary-file "${DICTIONARY_FILE}" --output "${TEMP_DIR}/escaped.txt" --lower-case "${LOWER_CASE}"

# train sentencepiece model
if [ ! -f "${SENTENCEPIECE_MODEL_NAME}.model" ]; then
  read -r -p "Train a sentencepiece model (y/n)? " choice
  case "$choice" in
    y|Y ) echo "Training...";;
    n|N ) echo "Exiting";exit 1;;
    * ) echo "Invalid answer";exit 1;;
  esac
  eval spm_train '--input="${INPUT_SENTENCE_FILE}"'\
            --character_coverage 1.0\
            '--model_prefix="${SENTENCEPIECE_MODEL_NAME}"'\
            '--vocab_size="${VOCAB_SIZE}"'\
            --unk_id=0 --pad_id=1 --bos_id=2 --eos_id=3\
            --input_sentence_size=30000000\
            --model_type=unigram\
            --normalization_rule_name=nmt_nfkc_cf\
            --num_threads=10\
            "${USER_DEF_SYMBOLS:+--user-defined-symbols=\''$USER_DEF_SYMBOLS'\'}"
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
    split -n "l/${SPM_PROCESSES}" "${TEMP_DIR}/escaped.txt" "${PARTS_DIR}/sentence_part-"

    ls "${PARTS_DIR}"/sentence_part-* | xargs '-I{}' -P "${SPM_PROCESSES}" -n 1 spm_encode  --model="${SENTENCEPIECE_MODEL_NAME}.model" --extra_options=bos:eos --output_format=id '--output={}.ids' '{}'
    cat "${PARTS_DIR}"/sentence_part-*.ids > "${IDS_FILE}"
fi
if [ ! -f "${OUTPUT_DIR}/val_ids.npy" ] || [ ! -f "${OUTPUT_DIR}/trn_ids.npy" ]; then
    # creates "${OUTPUT_DIR}/val_ids.npy" and "${OUTPUT_DIR}/trn_ids.npy"
    python ./split-datasets.py --ids-file "${IDS_FILE}" --output-path "${OUTPUT_DIR}"
fi


TEST_FILE="../data/task3/test/task3_test_segmented.txt"
TEST_IDS_FILE="${TEMP_DIR}/test_ids.txt"
# creates ${LOWERCASE_TEST_FILE}

if [ "${LOWER_CASE}" = "True" ]; then
    LOWERCASE_TEST_FILE="${TEMP_DIR}/task3_test_lowercase.txt"
    [ -f "${LOWERCASE_TEST_FILE}" ] || python ./escape-caps.py --sentence_file "${TEST_FILE}" --output "${LOWERCASE_TEST_FILE}" --most-low "${MOST_LOW}"
    TEST_FILE="${LOWERCASE_TEST_FILE}"
fi
# creates "${TEMP_DIR}/escaped_test.txt"
[ -f "${TEMP_DIR}/escaped_test.txt" ] || python ./cap-to-dict.py --sentence-file "${TEST_FILE}" --lower-case "${LOWER_CASE}" --dictionary-file "${DICTIONARY_FILE}" --output "${TEMP_DIR}/escaped_test.txt"

if [ ! -f ${TEST_IDS_FILE} ]; then
    mkdir -p "${PARTS_DIR}"
    rm -fr "${PARTS_DIR}"/sentence_part-*
    split -n "l/${SPM_PROCESSES}" "${TEMP_DIR}/escaped_test.txt" "${PARTS_DIR}/sentence_part-"
    ls "${PARTS_DIR}"/sentence_part-* | xargs '-I{}' -P "${SPM_PROCESSES}" -n 1 spm_encode --model="${SENTENCEPIECE_MODEL_NAME}.model" --extra_options=bos:eos --output_format=id '--output={}.ids' '{}'
    cat "${PARTS_DIR}"/sentence_part-*.ids > "${TEST_IDS_FILE}"
fi

if [ ! -f "${OUTPUT_DIR}/test_ids.npy" ]; then
    python ./split-datasets.py --ids-file "${TEST_IDS_FILE}" --output-path "${OUTPUT_DIR}" --test-set=True
fi
