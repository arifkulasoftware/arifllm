#include "bpe.h"
#include <time.h>

void symbol_table_init(SymbolTable *st) {
    st->symbols = NULL;
    st->count = 0;
    st->capacity = 0;
    for (int i = 0; i < SPECIAL_TOKEN_COUNT; i++) {
        symbol_table_add(st, SPECIAL_TOKENS[i]);
    }
}

void symbol_table_free(SymbolTable *st) {
    for (uint32_t i = 0; i < st->count; i++) free(st->symbols[i]);
    free(st->symbols);
    st->symbols = NULL;
    st->count = 0;
    st->capacity = 0;
}

uint32_t symbol_table_add(SymbolTable *st, const char *s) {
    if (st->count >= st->capacity) {
        uint32_t new_cap = st->capacity ? st->capacity * 2 : 16;
        st->symbols = (char **)realloc(st->symbols, new_cap * sizeof(char *));
        if (!st->symbols) exit(1);
        st->capacity = new_cap;
    }
    st->symbols[st->count] = xstrdup(s);
    return st->count++;
}

const char *symbol_table_get(SymbolTable *st, uint32_t id) {
    if (id >= st->count) return "";
    return st->symbols[id];
}

int symbol_id_for_char(SymbolTable *st, const char *ch_utf8) {
    for (uint32_t i = SPECIAL_TOKEN_COUNT; i < st->count; i++) {
        if (strcmp(st->symbols[i], ch_utf8) == 0) return (int)i;
    }
    return -1;
}

static int strcmp_ptr(const void *a, const void *b) {
    return strcmp(*(const char **)a, *(const char **)b);
}

void collect_and_add_base_chars(StrU64Map *word_freqs, SymbolTable *symbols, CharPtrArray *sorted_chars) {
    CharPtrArray all_chars;
    char_ptr_array_init(&all_chars);

    for (size_t b = 0; b < word_freqs->bucket_count; b++) {
        StrU64MapEntry *e = word_freqs->buckets[b];
        while (e) {
            utf_collect_chars(e->key, &all_chars);
            e = e->next;
        }
    }

    qsort(all_chars.items, all_chars.count, sizeof(char *), strcmp_ptr);

    char_ptr_array_init(sorted_chars);
    for (uint32_t i = 0; i < all_chars.count; i++) {
        char_ptr_array_push(sorted_chars, xstrdup(all_chars.items[i]));
        symbol_table_add(symbols, all_chars.items[i]);
    }

    char_ptr_array_free(&all_chars);
}

WordTable *build_word_table_from_freqs(StrU64Map *word_freqs, SymbolTable *symbols) {
    WordTable *wt = (WordTable *)xmalloc(sizeof(WordTable));
    wt->entries = NULL;
    wt->count = 0;
    wt->capacity = 0;

    for (size_t b = 0; b < word_freqs->bucket_count; b++) {
        StrU64MapEntry *e = word_freqs->buckets[b];
        while (e) {
            if (wt->count >= wt->capacity) {
                size_t new_cap = wt->capacity ? wt->capacity * 2 : 1024;
                wt->entries = (WordEntry *)realloc(wt->entries, new_cap * sizeof(WordEntry));
                if (!wt->entries) exit(1);
                wt->capacity = new_cap;
            }

            WordEntry *we = &wt->entries[wt->count++];
            we->word = xstrdup(e->key);
            we->freq = e->value;
            u32_array_init(&we->symbols);

            const char *p = e->key;
            const char *end = e->key + strlen(e->key);
            while (p < end) {
                const char *start = p;
                uint32_t cp;
                if (utf8_decode(&p, end, &cp) != 1) {
                    p++;
                    continue;
                }
                char *ch = utf_buffer_dup(start, (size_t)(p - start));
                int sid = symbol_id_for_char(symbols, ch);
                free(ch);
                if (sid < 0) {
                    fprintf(stderr, "Missing symbol for char in word: %s\n", e->key);
                    continue;
                }
                u32_array_push(&we->symbols, (uint32_t)sid);
            }
            e = e->next;
        }
    }
    return wt;
}

void word_table_free(WordTable *wt) {
    if (!wt) return;
    for (size_t i = 0; i < wt->count; i++) {
        free(wt->entries[i].word);
        u32_array_free(&wt->entries[i].symbols);
    }
    free(wt->entries);
    free(wt);
}

static void count_pairs(WordTable *words, U64U64Map *pair_counts) {
    for (size_t i = 0; i < words->count; i++) {
        WordEntry *we = &words->entries[i];
        U32Array *syms = &we->symbols;
        for (uint32_t j = 0; j + 1 < syms->len; j++) {
            uint64_t key = pair_key(syms->data[j], syms->data[j + 1]);
            uint64_t cur = u64_u64_map_get(pair_counts, key, 0);
            u64_u64_map_set(pair_counts, key, cur + we->freq);
        }
    }
}

static int find_best_pair(U64U64Map *pair_counts, uint32_t min_frequency,
                          uint32_t *left, uint32_t *right, uint64_t *best_freq) {
    uint64_t best = 0;
    int found = 0;
    for (size_t b = 0; b < pair_counts->bucket_count; b++) {
        U64U64MapEntry *e = pair_counts->buckets[b];
        while (e) {
            if (e->value >= min_frequency && e->value > best) {
                best = e->value;
                *left = (uint32_t)(e->key >> 32);
                *right = (uint32_t)(e->key & 0xFFFFFFFFu);
                found = 1;
            }
            e = e->next;
        }
    }
    if (found) *best_freq = best;
    return found;
}

static void merge_pair_in_word(WordEntry *we, uint32_t left, uint32_t right, uint32_t new_id) {
    U32Array *syms = &we->symbols;
    uint32_t write = 0;
    uint32_t read = 0;
    while (read < syms->len) {
        if (read + 1 < syms->len && syms->data[read] == left && syms->data[read + 1] == right) {
            syms->data[write++] = new_id;
            read += 2;
        } else {
            syms->data[write++] = syms->data[read++];
        }
    }
    syms->len = write;
}

static char *concat_symbols(SymbolTable *symbols, uint32_t left, uint32_t right) {
    const char *a = symbol_table_get(symbols, left);
    const char *b = symbol_table_get(symbols, right);
    size_t la = strlen(a), lb = strlen(b);
    char *out = (char *)xmalloc(la + lb + 1);
    memcpy(out, a, la);
    memcpy(out + la, b, lb);
    out[la + lb] = '\0';
    return out;
}

int bpe_train(WordTable *words, SymbolTable *symbols, MergeList *merges,
              uint32_t vocab_size, uint32_t min_frequency, FILE *log_fp) {
    merges->pairs = NULL;
    merges->count = 0;
    merges->capacity = 0;

    U64U64Map pair_counts;
    u64_u64_map_init(&pair_counts, 65536);

    while (symbols->count < vocab_size) {
        u64_u64_map_free(&pair_counts);
        u64_u64_map_init(&pair_counts, 65536);
        count_pairs(words, &pair_counts);

        uint32_t left = 0, right = 0;
        uint64_t best_freq = 0;
        if (!find_best_pair(&pair_counts, min_frequency, &left, &right, &best_freq)) {
            if (log_fp) {
                fprintf(log_fp, "No more pairs meeting min_frequency=%u\n", min_frequency);
            }
            break;
        }

        char *merged = concat_symbols(symbols, left, right);
        uint32_t new_id = symbol_table_add(symbols, merged);

        if (merges->count >= merges->capacity) {
            uint32_t new_cap = merges->capacity ? merges->capacity * 2 : 1024;
            merges->pairs = (MergePair *)realloc(merges->pairs, new_cap * sizeof(MergePair));
            if (!merges->pairs) exit(1);
            merges->capacity = new_cap;
        }
        merges->pairs[merges->count].left = left;
        merges->pairs[merges->count].right = right;
        merges->count++;

        for (size_t i = 0; i < words->count; i++) {
            merge_pair_in_word(&words->entries[i], left, right, new_id);
        }

        if (log_fp) {
            fprintf(log_fp, "Merge %u: (%u,%u) freq=%llu -> id=%u token=%s vocab=%u\n",
                    merges->count, left, right, (unsigned long long)best_freq, new_id, merged, symbols->count);
        }

        free(merged);

        if (symbols->count >= vocab_size) break;
    }

    u64_u64_map_free(&pair_counts);
    return 0;
}

static void json_escape(const char *s, FILE *fp) {
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        unsigned char c = *p;
        switch (c) {
            case '"': fputs("\\\"", fp); break;
            case '\\': fputs("\\\\", fp); break;
            case '\b': fputs("\\b", fp); break;
            case '\f': fputs("\\f", fp); break;
            case '\n': fputs("\\n", fp); break;
            case '\r': fputs("\\r", fp); break;
            case '\t': fputs("\\t", fp); break;
            default:
                if (c < 0x20) fprintf(fp, "\\u%04x", c);
                else fputc(c, fp);
                break;
        }
    }
}

int write_tokenizer_json(const char *path, SymbolTable *symbols, MergeList *merges) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "Cannot open output file: %s (%s)\n", path, strerror(errno));
        return -1;
    }

    fputs("{\n", fp);
    fputs("  \"version\": \"1.0\",\n", fp);
    fputs("  \"truncation\": null,\n", fp);
    fputs("  \"padding\": null,\n", fp);
    fputs("  \"added_tokens\": [\n", fp);

    for (int i = 0; i < SPECIAL_TOKEN_COUNT; i++) {
        fprintf(fp, "    {\n");
        fprintf(fp, "      \"id\": %d,\n", i);
        fprintf(fp, "      \"content\": \"");
        json_escape(SPECIAL_TOKENS[i], fp);
        fprintf(fp, "\",\n");
        fputs("      \"single_word\": false,\n", fp);
        fputs("      \"lstrip\": false,\n", fp);
        fputs("      \"rstrip\": false,\n", fp);
        fputs("      \"normalized\": false,\n", fp);
        fputs("      \"special\": true\n", fp);
        fprintf(fp, "    }%s\n", i + 1 < SPECIAL_TOKEN_COUNT ? "," : "");
    }

    fputs("  ],\n", fp);
    fputs("  \"normalizer\": {\n", fp);
    fputs("    \"type\": \"Sequence\",\n", fp);
    fputs("    \"normalizers\": [\n", fp);
    fputs("      {\n", fp);
    fputs("        \"type\": \"NFKC\"\n", fp);
    fputs("      }\n", fp);
    fputs("    ]\n", fp);
    fputs("  },\n", fp);
    fputs("  \"pre_tokenizer\": {\n", fp);
    fputs("    \"type\": \"Whitespace\"\n", fp);
    fputs("  },\n", fp);
    fputs("  \"post_processor\": null,\n", fp);
    fputs("  \"decoder\": null,\n", fp);
    fputs("  \"model\": {\n", fp);
    fputs("    \"type\": \"BPE\",\n", fp);
    fputs("    \"dropout\": null,\n", fp);
    fputs("    \"unk_token\": \"[UNK]\",\n", fp);
    fputs("    \"continuing_subword_prefix\": null,\n", fp);
    fputs("    \"end_of_word_suffix\": null,\n", fp);
    fputs("    \"fuse_unk\": false,\n", fp);
    fputs("    \"byte_fallback\": false,\n", fp);
    fputs("    \"ignore_merges\": false,\n", fp);
    fputs("    \"vocab\": {\n", fp);

    for (uint32_t i = 0; i < symbols->count; i++) {
        fprintf(fp, "      \"");
        json_escape(symbols->symbols[i], fp);
        fprintf(fp, "\": %u%s\n", i, i + 1 < symbols->count ? "," : "");
    }

    fputs("    },\n", fp);
    fputs("    \"merges\": [\n", fp);

    for (uint32_t i = 0; i < merges->count; i++) {
        MergePair *mp = &merges->pairs[i];
        fprintf(fp, "      [\n");
        fprintf(fp, "        \"");
        json_escape(symbol_table_get(symbols, mp->left), fp);
        fprintf(fp, "\",\n");
        fprintf(fp, "        \"");
        json_escape(symbol_table_get(symbols, mp->right), fp);
        fprintf(fp, "\"\n");
        fprintf(fp, "      ]%s\n", i + 1 < merges->count ? "," : "");
    }

    fputs("    ]\n", fp);
    fputs("  }\n", fp);
    fputs("}\n", fp);

    fclose(fp);
    return 0;
}
