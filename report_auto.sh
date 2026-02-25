source venv_py310/bin/activate

SRC_ROOT="/home/s1k2/04_BadRubber/datasets/raw"
DST_ROOT="/home/s1k2/04_BadRubber/datasets/processed"
NAS_ROOT="/home/s1k2/04_BadRubber/src/DefectDetection/NAS/sub/inspection/defect/classify/datasets/processed2"

LINE="BR-A"
GRADE="unknown"

BATCH_SIZE=9
RESUME_THRESHOLD=80

echo "======================================"
echo "[${LINE}] Processing started"
echo "Start Time: $(date)"
echo "======================================"


upload_to_nas() {

    local_path=$1
    nas_path=$2

    if [ ! -d "$local_path" ]; then
        return 1
    fi

    mkdir -p "$nas_path"

    rsync -avz --partial --progress "$local_path/" "$nas_path/"

    return $?
}


cleanup_after_upload() {

    date=$1
    done_flag=$2
    raw_path=$3
    dst_path=$4
    processing_json=$5
    delete_json=$6

    if [ "$delete_json" = "true" ]; then
        if [ -f "$processing_json" ]; then
            rm -f "$processing_json"
        fi
    fi

    touch "$done_flag"

    if [ -d "$raw_path" ]; then
        rm -rf "$raw_path"
        echo "RAW Delete Complete"
    fi

    if [ -d "$dst_path" ]; then
        rm -rf "$dst_path"
        echo "DST Folder Delete Complete"
    fi

    echo "$date Complete"
}


run_python_process() {

    date=$1

    python report_auto.py \
        --src-root "$SRC_ROOT" \
        --dst-root "$DST_ROOT" \
        --line "$LINE" \
        --grade "$GRADE" \
        --dates "$date" \
        --batch-size "$BATCH_SIZE"

    return $?
}


while true; do

    line_path="$SRC_ROOT/$LINE"

    date_folders=$(find "$line_path" -maxdepth 1 -type d -name "20*-*-*" | sort)

    if [ -z "$date_folders" ]; then
        sleep 60
        continue
    fi

    for folder in $date_folders; do

        date=$(basename "$folder")

        done_flag="$DST_ROOT/$LINE/$GRADE/$date.done"
        processing_json="$DST_ROOT/$LINE/$GRADE/$date.processing.json"
        local_processed_path="$DST_ROOT/$LINE/$GRADE/$date"
        nas_target_path="$DST_ROOT/$LINE/$GRADE/$date"
        raw_folder_path="$line_path/$date"

        if [ -f "$done_flag" ]; then
            continue
        fi

        # Interrupted recovery logic
        if [ -f "$processing_json" ]; then

            progress=$(cat "$processing_json" | jq -r '.progress_percent')

            echo "Processing: ${progress}%"

            if (( $(echo "$progress > $RESUME_THRESHOLD" | bc -l) )); then

                echo "Skip reprocessing and NAS upload"

                if upload_to_nas "$local_processed_path" "$nas_target_path"; then

                    cleanup_after_upload \
                        "$date" \
                        "$done_flag" \
                        "$raw_folder_path" \
                        "$local_processed_path" \
                        "$processing_json" \
                        "false"
                else
                    echo "NAS Upload Failed"
                fi

                continue

            else
                echo "reprocessing start"
            fi
        fi

        echo "---------------------------------------"
        echo "$date Processing started"
        echo "Start Time: $(date)"

        if run_python_process "$date"; then

            if upload_to_nas "$local_processed_path" "$nas_target_path"; then

                cleanup_after_upload \
                    "$date" \
                    "$done_flag" \
                    "$raw_folder_path" \
                    "$local_processed_path" \
                    "$processing_json" \
                    "true"
            else
                echo "NAS Upload Failed"
            fi
        fi

        echo "---------------------------------------"

    done

    sleep 60

done