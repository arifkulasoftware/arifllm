#ifndef BPE_TRAIN_H
#define BPE_TRAIN_H

#include "common.h"
#include "utf.h"

void symbol_table_init(SymbolTable *st);
void symbol_table_free(SymbolTable *st);
uint32_t symbol_table_add(SymbolTable *st, const char *s);
const char *symbol_table_get(SymbolTable *st, uint32_t id);

int symbol_id_for_char(SymbolTable *st, const char *ch_utf8);

WordTable *build_word_table_from_freqs(StrU64Map *word_freqs, SymbolTable *symbols);
void word_table_free(WordTable *wt);

void collect_and_add_base_chars(StrU64Map *word_freqs, SymbolTable *symbols, CharPtrArray *sorted_chars);

int bpe_train(WordTable *words, SymbolTable *symbols, MergeList *merges,
              uint32_t vocab_size, uint32_t min_frequency, FILE *log_fp);

int write_tokenizer_json(const char *path, SymbolTable *symbols, MergeList *merges);

#endif
