#!/usr/bin/env bash 
set -e
set -o pipefail

TESTCASE="$1"
LOG_OUTPUT="$2"
# "${@:2}"
TMPDIR="$(mktemp -d)"

function remove_temp {
  [[ -d "$TMPDIR" ]] && rm -rf "$TMPDIR"
}

trap remove_temp EXIT
trap 'trap - EXIT; remove_temp; exit -1' INT PIPE TERM

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

CURRENT_DIR=$(pwd)

BUILD_DIR="${SCRIPT_DIR}/judge"



NUM_CPU=$(grep ^"core id" /proc/cpuinfo | sort -u | wc -l)



FIFO="${TMPDIR}/fifo"
mkfifo "$FIFO"

EXEC_NAME="${BUILD_DIR}/judge"


stdbuf -i0 -o0 -e0 ${EXEC_NAME} $TESTCASE $LOG_OUTPUT < $FIFO |tee debug_input.txt|stdbuf -i0 -o0 -e0 "${@:3}"|tee debug_output.txt > $FIFO
