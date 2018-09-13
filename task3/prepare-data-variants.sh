#!/usr/bin/env bash

WORK_DIR="../work"
CACHE_DIR="${WORK_DIR}/shared"
for DICT_SIZE in 25 50 100;
do
    ./prepare-data.sh --work-dir "${WORK_DIR}/up_low${DICT_SIZE}k" --cache-dir "${CACHE_DIR}" --vocab-size "${DICT_SIZE}000" --model-name "sp" --most-low "False" --lower-case "False"
    ./prepare-data.sh --work-dir "${WORK_DIR}/most_low${DICT_SIZE}k" --cache-dir "${CACHE_DIR}" --vocab-size "${DICT_SIZE}000" --model-name "sp" --most-low "True" --user-defined-symbols '<up>' --lower-case "True"
done
