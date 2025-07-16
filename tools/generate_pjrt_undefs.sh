#!/bin/bash
# Usage: ./generate_pjrt_undefs.sh path/to/pjrt_c_api.h

HEADER="$1"
OUTPUT="pjrt_undefs.h"

if [[ ! -f "$HEADER" ]]; then
  echo "Usage: $0 path/to/pjrt_c_api.h"
  exit 1
fi

echo "#ifdef __cplusplus" > "$OUTPUT"
grep -oE 'typedef PJRT_Error\* [A-Za-z0-9_]+\(' "$HEADER" \
  | sed -E 's/typedef PJRT_Error\* ([A-Za-z0-9_]+)\(.*/#undef \1/' \
  | sort | uniq >> "$OUTPUT"
echo "#endif  // __cplusplus" >> "$OUTPUT"
echo "Generated $OUTPUT from $HEADER"
