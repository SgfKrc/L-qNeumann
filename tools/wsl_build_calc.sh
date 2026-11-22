#!/bin/bash
# Build calc with the dpkg -x toolchain (no root).  Run inside WSL.
#   bash wsl_build_calc.sh OUTDIR NAME
# Mirrors chipformer/rv/env/calc/build_calc.sh, adapted for gcc 10.2 / picolibc 1.7.4:
#   - -march=rv32im (gcc 10.2 has no zba)
#   - picolibc libs live in lib/rv32im/ilp32, not lib/release/rv32im/ilp32
set -euo pipefail

export PATH=$HOME/rvbin:/usr/bin:$PATH
GCC=$HOME/rvroot/usr/bin/riscv64-unknown-elf-gcc
PICO=$HOME/rvroot/usr/lib/picolibc/riscv64-unknown-elf
PICO_LIB=$PICO/lib/rv32im/ilp32
MARCH=rv32im

HERE=/mnt/h/sgfd9/QwenPU/chipformer/rv/env
OUT=$(realpath "$1"); NAME=${2:-calc}
mkdir -p "$OUT"
B=$(mktemp -d)
trap 'rm -rf "$B"' EXIT

python3 "$HERE/calc/gen_sled.py" "$B/sled.S"
sed 's/  .data : {/  .calcin : { *(.calcin) } > RAM\n  .data : {/' "$HERE/rom.ld" > "$B/calc.ld"

LIBGCC=$($GCC -march=$MARCH -mabi=ilp32 -print-libgcc-file-name)

$GCC -march=$MARCH -mabi=ilp32 ${CFLAGS:--O2} -B"$HOME/rvbin/" \
  -isystem "$PICO/include" -I"$HERE" -I"$HERE/calc" \
  -nostdlib -nostartfiles -static -T"$B/calc.ld" -o "$OUT/$NAME.elf" \
  "$HERE/crt0.S" "$HERE/calc/calc.c" "$HERE/sys.c" "$B/sled.S" \
  -Wl,--start-group "$PICO_LIB/libc.a" "$LIBGCC" -Wl,--end-group

$HOME/rvroot/usr/bin/riscv64-unknown-elf-size -A "$OUT/$NAME.elf" | grep -E '^\.(text|rodata|calcin|data|bss)'
$HOME/rvroot/usr/bin/riscv64-unknown-elf-nm "$OUT/$NAME.elf" | grep -E ' calc_in$'
