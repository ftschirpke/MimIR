#!/usr/bin/env bash

mim=/home/friedrich/studium/master/mimir/build/bin/mim

name=kernel_detection

if [[ $# -gt 0 ]]; then
    name=$1
fi

error () {
    echo "$1"
    exit 1
}

test_dir_path=/home/friedrich/studium/master/mimir/lit/gpu
test_path=$test_dir_path/$name.mim
output_dir_path=$test_dir_path/Output
host_ll_path=$output_dir_path/$name.mim.tmp.ll 
dev_ll_path=$output_dir_path/$name.mim.tmp.dev.ll 
hostbin_path=$output_dir_path/$name.mim.tmp.out

mkdir -p $output_dir_path

gpu_arch=sm_89

if [[ -f $test_path ]]; then
    echo "test_path: $test_path"
    echo "host_ll_path: $host_ll_path"
    echo "dev_ll_path: $dev_ll_path"
    echo "hostbin_path: $hostbin_path"
else
    error "test does not exist: $test_path"
fi

rm $host_ll_path $dev_ll_path $hostbin_path

$mim $test_path \
    --output-ll $host_ll_path \
    --output-device-ll $dev_ll_path \
    --device-target nvptx \
    --embed-device-binary \
    -o - \
    ${@:2} \
    || error "mim step"

echo
echo "-----  HOST LLVM  -----"

if [[ -f $host_ll_path ]]; then
    if [[ ! -s $host_ll_path ]]; then
        error "device llvm file is empty"
    fi
    cat $host_ll_path
else
    error "host llvm was not created"
fi

echo "----- DEVICE LLVM -----"

if [[ -f $dev_ll_path ]]; then
    if [[ ! -s $dev_ll_path ]]; then
        error "device llvm file is empty"
    fi
    cat $dev_ll_path
else
    error "device llvm was not created"
fi

echo "----- COMPILATION -----"

clang $host_ll_path -o $hostbin_path -lcuda -Wno-override-module || error "host binary"

echo "----- EXECUTE BIN -----"

args="2 3 4 5 6"
echo "executing '$hostbin_path $args'"
echo ">>>"
$hostbin_path $args
echo "<<< rv = $(echo $?)"

echo "-----------------------"
