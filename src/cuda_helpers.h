#pragma once

const size_t BLOCK_SIZE = 64;
const size_t MM_BLOCK_SIZE = 8;

inline size_t num_blocks(size_t num_items, size_t block_size) {
    return (num_items + block_size - 1) / block_size;
}

inline size_t num_blocks(size_t num_items) {
    return num_blocks(num_items, BLOCK_SIZE);
}
