#!/usr/bin/env bash
set -e

source utils.sh

MOST_LOW="False"

while [[ $# -gt 0 ]]
do
    case "$1" in
        --sentence-file)
            SENTENCE_FILE="$2"
            shift
            shift
            ;;
        --output)
            OUTPUT="$2"
            shift
            shift
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

if [ -z "${SENTENCE_FILE+x}" -o -z "${OUTPUT+x}" ]; then
    die "--sentence-file or --output is missing"
fi

if [ "${MOST_LOW}" = "True" ]; then
    # reverse transform: sed -e 's/<up> \(.\)/\U\1/g'
    perl -CSD -pe 's/(^| )([[:upper:]][^ [:upper:]]*)(?=$| )/\1<up> \l\2/g' "${SENTENCE_FILE}" > "${OUTPUT}"
else
    echo "just_low is not supported"
    exit 1
fi
