TASK3_DATA_DIR="data/task3/train"
SENTENCE_FILE="${TASK3_DATA_DIR}/task3_train_segmented.txt"
TEMP_DIR="tmp"
OUTPUT_DIR="tmp"
SENTENCEPIECE_MODEL_NAME="sp-100k"

# sort sentences and remove duplicates
mkdir -p "${TEMP_DIR}"
UNIQ_SENTENCE_FILE="${TEMP_DIR}/task3_train_uniq.txt"
sort "${SENTENCE_FILE}" | uniq > "${UNIQ_SENTENCE_FILE}"

# train sentencepiece model
if [ ! -f "${SENTENCEPIECE_MODEL_NAME}.model" ]; then
  read -r -p "Train a sentencepiece model (y/n)? " choice
  case "$choice" in
    y|Y ) echo "Training...";;
    n|N ) echo "Exiting";exit 1;;
    * ) echo "Invalid answer";exit 1;;
  esac
  spm_train --input="${UNIQ_SENTENCE_FILE}" --model_prefix="${SENTENCEPIECE_MODEL_NAME}" --vocab_size=100000 --unk_id=0 --pad_id=1 --bos_id=2 --eos_id=3 --input_sentence_size=30000000 --model_type=unigram
else
  echo "Setencepiece model '${SENTENCEPIECE_MODEL_NAME}.model' already exists."
fi

# creates "${TEMP_DIR}/escaped.txt" and "${TEMP_DIR}/vacabulary.txt"
python ./limit-dict.py --sentence-file "${UNIQ_SENTENCE_FILE}" --output-path "${TEMP_DIR}" --threads 1

# running spm_encode in parallel
SPM_PROCESSES=8
PARTS_DIR="${TEMP_DIR}/parts"
IDS_FILE="${TEMP_DIR}/ids.txt"
mkdir "${PARTS_DIR}"
rm -r "${PARTS_DIR}"/sentence_part-*
split -n "l/${SPM_PROCESSES}" "${TEMP_DIR}/escaped.txt" "${PARTS_DIR}/sentence_part-"
ls "${PARTS_DIR}"/sentence_part-*" | xargs '-I{}' -P "${SPM_PROCESSES}" -n 1 spm_encode --model="${SENTENCEPIECE_MODEL_NAME}.model" --extra_options=bos:eos --output_format=id '--output={}.ids' '{}'
cat "${PARTS_DIR}"/sentence_part-*.ids > "${IDS_FILE}"

# creates "${OUTPUT_DIR}/val_ids.npy" and "${OUTPUT_DIR}/trn_ids.npy"
python ./split-datasets.py --ids-file "${IDS_FILE}" --output-path "${OUTPUT_DIR}"
