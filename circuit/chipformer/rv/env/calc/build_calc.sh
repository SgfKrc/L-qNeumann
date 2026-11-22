#!/bin/bash
# Build the calc firmware (run in WSL):  build_calc.sh OUT
# OUT/calc.elf: the ROM / DROM / RAM map, the expression buffer calc_in at RAM 0x100.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ENV=$HERE/..
OUT=$(realpath "$1"); NAME=${2:-calc}
PICO=${PICO:-/usr/lib/picolibc/riscv64-unknown-elf}
GCC=${GCC:-riscv64-unknown-elf-gcc}
MARCH=${MARCH:-rv32im_zba}
SIZE=${SIZE:-riscv64-unknown-elf-size}
NM=${NM:-riscv64-unknown-elf-nm}
# picolibc 1.8 keeps the multilibs under lib/release/, 1.7 under lib/ : take whichever is there
if [ -d "$PICO/lib/release/rv32im/ilp32" ]; then
  PICO_LIB=$PICO/lib/release/rv32im/ilp32
else
  PICO_LIB=$PICO/lib/rv32im/ilp32
fi
LIBGCC=$($GCC -march=rv32im -mabi=ilp32 -print-libgcc-file-name)
B=$(mktemp -d)
python3 "$HERE/gen_sled.py" "$B/sled.S"
# calc_in first in RAM (0x100)
sed 's/  .data : {/  .calcin : { *(.calcin) } > RAM\n  .data : {/' "$ENV/rom.ld" > "$B/calc.ld"
$GCC -march=$MARCH -mabi=ilp32 ${CFLAGS:--O2} -isystem "$PICO/include" -I"$ENV" -I"$HERE" \
  -nostdlib -nostartfiles -static -T"$B/calc.ld" -o "$OUT/$NAME.elf" \
  "$ENV/crt0.S" "$HERE/calc.c" "$ENV/sys.c" "$B/sled.S" \
  -Wl,--start-group "$PICO_LIB/libc.a" "$LIBGCC" -Wl,--end-group
$SIZE -A "$OUT/$NAME.elf" | grep -E '^\.(text|rodata|calcin|data|bss)'
$NM "$OUT/$NAME.elf" | grep -E ' calc_in$'
rm -rf "$B"
