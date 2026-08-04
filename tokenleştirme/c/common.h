#ifndef BPE_COMMON_H
#define BPE_COMMON_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#define SPECIAL_TOKEN_COUNT 7

static const char *SPECIAL_TOKENS[SPECIAL_TOKEN_COUNT] = {
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "[SOS]",
    "[EOS]"
};

typedef struct {
    char *key;
    uint64_t value;
    struct StrU64MapEntry *next;
} StrU64MapEntry;

typedef struct {
    StrU64MapEntry **buckets;
    size_t bucket_count;
    size_t size;
} StrU64Map;

typedef struct {
    uint64_t key;
    uint64_t value;
    struct U64U64MapEntry *next;
} U64U64MapEntry;

typedef struct {
    U64U64MapEntry **buckets;
    size_t bucket_count;
    size_t size;
} U64U64Map;

typedef struct {
    char **items;
    uint32_t count;
    uint32_t capacity;
} CharPtrArray;

typedef struct {
    uint32_t *data;
    uint32_t len;
    uint32_t cap;
} U32Array;

typedef struct {
    char *word;
    uint64_t freq;
    U32Array symbols;
} WordEntry;

typedef struct {
    WordEntry *entries;
    size_t count;
    size_t capacity;
} WordTable;

typedef struct {
    char **symbols;
    uint32_t count;
    uint32_t capacity;
} SymbolTable;

typedef struct {
    uint32_t left;
    uint32_t right;
} MergePair;

typedef struct {
    MergePair *pairs;
    uint32_t count;
    uint32_t capacity;
} MergeList;

typedef struct {
    const char *input_dir;
    const char *output_dir;
    const char *log_path;
    const char *checkpoint_path;
    uint32_t vocab_size;
    uint32_t min_frequency;
    int max_files;
    int max_lines;
    uint64_t max_bytes;
    int lowercase;
} Config;

static inline uint64_t fnv1a64(const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static inline uint64_t pair_key(uint32_t left, uint32_t right) {
    return ((uint64_t)left << 32) | (uint64_t)right;
}

static inline void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) {
        fprintf(stderr, "malloc failed (%zu bytes)\n", n);
        exit(1);
    }
    return p;
}

static inline char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *d = (char *)xmalloc(n);
    memcpy(d, s, n);
    return d;
}

static inline void str_u64_map_init(StrU64Map *m, size_t bucket_count) {
    m->bucket_count = bucket_count;
    m->size = 0;
    m->buckets = (StrU64MapEntry **)calloc(bucket_count, sizeof(StrU64MapEntry *));
    if (!m->buckets) exit(1);
}

static inline void str_u64_map_free(StrU64Map *m) {
    for (size_t i = 0; i < m->bucket_count; i++) {
        StrU64MapEntry *e = m->buckets[i];
        while (e) {
            StrU64MapEntry *next = e->next;
            free(e->key);
            free(e);
            e = next;
        }
    }
    free(m->buckets);
    m->buckets = NULL;
    m->bucket_count = 0;
    m->size = 0;
}

static inline uint64_t str_u64_map_get(StrU64Map *m, const char *key, uint64_t default_value) {
    size_t idx = fnv1a64(key, strlen(key)) % m->bucket_count;
    StrU64MapEntry *e = m->buckets[idx];
    while (e) {
        if (strcmp(e->key, key) == 0) return e->value;
        e = e->next;
    }
    return default_value;
}

static inline void str_u64_map_set(StrU64Map *m, const char *key, uint64_t value) {
    size_t idx = fnv1a64(key, strlen(key)) % m->bucket_count;
    StrU64MapEntry *e = m->buckets[idx];
    while (e) {
        if (strcmp(e->key, key) == 0) {
            e->value = value;
            return;
        }
        e = e->next;
    }
    StrU64MapEntry *ne = (StrU64MapEntry *)xmalloc(sizeof(StrU64MapEntry));
    ne->key = xstrdup(key);
    ne->value = value;
    ne->next = m->buckets[idx];
    m->buckets[idx] = ne;
    m->size++;
}

static inline void str_u64_map_add(StrU64Map *m, const char *key, uint64_t delta) {
    uint64_t cur = str_u64_map_get(m, key, 0);
    str_u64_map_set(m, key, cur + delta);
}

static inline void u64_u64_map_init(U64U64Map *m, size_t bucket_count) {
    m->bucket_count = bucket_count;
    m->size = 0;
    m->buckets = (U64U64MapEntry **)calloc(bucket_count, sizeof(U64U64MapEntry *));
    if (!m->buckets) exit(1);
}

static inline void u64_u64_map_free(U64U64Map *m) {
    for (size_t i = 0; i < m->bucket_count; i++) {
        U64U64MapEntry *e = m->buckets[i];
        while (e) {
            U64U64MapEntry *next = e->next;
            free(e);
            e = next;
        }
    }
    free(m->buckets);
    m->buckets = NULL;
    m->bucket_count = 0;
    m->size = 0;
}

static inline uint64_t u64_u64_map_get(U64U64Map *m, uint64_t key, uint64_t default_value) {
    size_t idx = (size_t)(key % m->bucket_count);
    U64U64MapEntry *e = m->buckets[idx];
    while (e) {
        if (e->key == key) return e->value;
        e = e->next;
    }
    return default_value;
}

static inline void u64_u64_map_set(U64U64Map *m, uint64_t key, uint64_t value) {
    size_t idx = (size_t)(key % m->bucket_count);
    U64U64MapEntry *e = m->buckets[idx];
    while (e) {
        if (e->key == key) {
            e->value = value;
            return;
        }
        e = e->next;
    }
    U64U64MapEntry *ne = (U64U64MapEntry *)xmalloc(sizeof(U64U64MapEntry));
    ne->key = key;
    ne->value = value;
    ne->next = m->buckets[idx];
    m->buckets[idx] = ne;
    m->size++;
}

static inline void u32_array_init(U32Array *a) {
    a->data = NULL;
    a->len = 0;
    a->cap = 0;
}

static inline void u32_array_free(U32Array *a) {
    free(a->data);
    a->data = NULL;
    a->len = 0;
    a->cap = 0;
}

static inline void u32_array_push(U32Array *a, uint32_t v) {
    if (a->len >= a->cap) {
        uint32_t new_cap = a->cap ? a->cap * 2 : 8;
        a->data = (uint32_t *)realloc(a->data, new_cap * sizeof(uint32_t));
        if (!a->data) exit(1);
        a->cap = new_cap;
    }
    a->data[a->len++] = v;
}

static inline void char_ptr_array_init(CharPtrArray *a) {
    a->items = NULL;
    a->count = 0;
    a->capacity = 0;
}

static inline void char_ptr_array_free(CharPtrArray *a) {
    for (uint32_t i = 0; i < a->count; i++) free(a->items[i]);
    free(a->items);
    a->items = NULL;
    a->count = 0;
    a->capacity = 0;
}

static inline void char_ptr_array_push(CharPtrArray *a, char *s) {
    if (a->count >= a->capacity) {
        uint32_t new_cap = a->capacity ? a->capacity * 2 : 16;
        a->items = (char **)realloc(a->items, new_cap * sizeof(char *));
        if (!a->items) exit(1);
        a->capacity = new_cap;
    }
    a->items[a->count++] = s;
}

static inline int char_ptr_array_contains(CharPtrArray *a, const char *s) {
    for (uint32_t i = 0; i < a->count; i++) {
        if (strcmp(a->items[i], s) == 0) return 1;
    }
    return 0;
}

#endif
