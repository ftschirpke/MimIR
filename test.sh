#!/usr/bin/env bash

mim=/home/friedrich/studium/master/mimir/build/bin/mim

name=kernel_detection

if [[ $# -gt 1 ]]; then
    name=$1
fi

error () {
    echo "$1"
    exit 1
}

test_path=/home/friedrich/studium/master/mimir/lit/gpu/$name.mim
host_ll_path=/home/friedrich/studium/master/mimir/build/test/gpu/Output/$name.mim.tmp.ll 
dev_ll_path=/home/friedrich/studium/master/mimir/build/test/gpu/Output/$name.mim.tmp.dev.ll 
hostobj_path=host.obj
hostbin_path=host
ptx_path=mimir.ptx
cubin_path=mimir.cubin
fatbin=mimir.fatbin

echo "test_path: $test_path"
echo "host_ll_path: $host_ll_path"
echo "dev_ll_path: $dev_ll_path"

rm $host_ll_path
rm $dev_ll_path

$mim $test_path \
    --output-ll $host_ll_path \
    --output-device-ll $dev_ll_path \
    --device-target nvptx \
    -o - \
    ${@:2} \
    || error "mim step"

echo
echo "-----  HOST LLVM  -----"
[[ -f $host_ll_path ]] && cat $host_ll_path
echo "----- DEVICE LLVM -----"
[[ -f $dev_ll_path ]] && cat $dev_ll_path

echo "----- COMPILATION -----"
llc -filetype=obj -relocation-model=pic $host_ll_path -o $hostobj_path || error "host object"

clang $hostobj_path -o $hostbin_path -lcuda || error "host binary"

llc -march=nvptx64 $dev_ll_path -o $ptx_path || error "llc step"

ptxas $ptx_path -o $cubin_path || error "ptxas step"

nvcc -fatbin $cubin_path -o $fatbin || error "fatbin step"

echo "----- EXECUTE BIN -----"
args="2 3 4 5 6"
echo "executing './$hostbin_path $args'"
echo ">>>"
./$hostbin_path $args
echo "<<< rv = $(echo $?)"
echo "-----------------------"
