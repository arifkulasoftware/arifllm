#ifndef BPE_UTF_H
#define BPE_UTF_H

#include "common.h"

/* UTF-8 decode: returns codepoint and advances *p. Returns -1 on error. */
int utf8_decode(const char **p, const char *end, uint32_t *out);

/* Append UTF-32 codepoint as UTF-8 to growable buffer. */
void utf8_append_codepoint(char **buf, size_t *len, size_t *cap, uint32_t cp);

/* Reallocate string buffer helpers. */
char *utf_buffer_dup(const char *src, size_t len);
void utf_buffer_append(char **buf, size_t *len, size_t *cap, const char *s);

/*
 * NFKC normalization (Windows: NormalizeString).
 * Returns newly allocated UTF-8 string or NULL on failure.
 */
char *utf_normalize_nfkc(const char *input);

/*
 * Unicode casefold approximation for training (Turkish-aware where possible).
 * Returns newly allocated UTF-8 string.
 */
char *utf_casefold(const char *input);

/* Split on Unicode whitespace; append tokens to word frequency map. */
void utf_split_whitespace_to_map(const char *line, StrU64Map *word_freqs);

/* Collect unique UTF-8 characters from a UTF-8 string into sorted array. */
void utf_collect_chars(const char *s, CharPtrArray *chars);

#endif
