#include "utf.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#pragma comment(lib, "Normaliz.lib")
#endif

int utf8_decode(const char **p, const char *end, uint32_t *out) {
    const unsigned char *s = (const unsigned char *)*p;
    if (s >= (const unsigned char *)end) return 0;

    unsigned char c0 = s[0];
    if (c0 < 0x80) {
        *out = c0;
        *p = (const char *)(s + 1);
        return 1;
    }
    if ((c0 & 0xE0) == 0xC0 && s + 1 < (const unsigned char *)end) {
        *out = ((uint32_t)(c0 & 0x1F) << 6) | (s[1] & 0x3F);
        *p = (const char *)(s + 2);
        return 1;
    }
    if ((c0 & 0xF0) == 0xE0 && s + 2 < (const unsigned char *)end) {
        *out = ((uint32_t)(c0 & 0x0F) << 12) | ((uint32_t)(s[1] & 0x3F) << 6) | (s[2] & 0x3F);
        *p = (const char *)(s + 3);
        return 1;
    }
    if ((c0 & 0xF8) == 0xF0 && s + 3 < (const unsigned char *)end) {
        *out = ((uint32_t)(c0 & 0x07) << 18) | ((uint32_t)(s[1] & 0x3F) << 12) |
               ((uint32_t)(s[2] & 0x3F) << 6) | (s[3] & 0x3F);
        *p = (const char *)(s + 4);
        return 1;
    }
    *p = (const char *)(s + 1);
    return -1;
}

void utf8_append_codepoint(char **buf, size_t *len, size_t *cap, uint32_t cp) {
    char tmp[4];
    int n = 0;
    if (cp <= 0x7F) {
        tmp[0] = (char)cp;
        n = 1;
    } else if (cp <= 0x7FF) {
        tmp[0] = (char)(0xC0 | (cp >> 6));
        tmp[1] = (char)(0x80 | (cp & 0x3F));
        n = 2;
    } else if (cp <= 0xFFFF) {
        tmp[0] = (char)(0xE0 | (cp >> 12));
        tmp[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        tmp[2] = (char)(0x80 | (cp & 0x3F));
        n = 3;
    } else {
        tmp[0] = (char)(0xF0 | (cp >> 18));
        tmp[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        tmp[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        tmp[3] = (char)(0x80 | (cp & 0x3F));
        n = 4;
    }
    if (*len + n > *cap) {
        size_t new_cap = *cap ? *cap * 2 : 64;
        while (new_cap < *len + n) new_cap *= 2;
        *buf = (char *)realloc(*buf, new_cap);
        if (!*buf) exit(1);
        *cap = new_cap;
    }
    memcpy(*buf + *len, tmp, n);
    *len += n;
}

char *utf_buffer_dup(const char *src, size_t len) {
    char *d = (char *)xmalloc(len + 1);
    memcpy(d, src, len);
    d[len] = '\0';
    return d;
}

void utf_buffer_append(char **buf, size_t *len, size_t *cap, const char *s) {
    size_t slen = strlen(s);
    if (*len + slen + 1 > *cap) {
        size_t new_cap = *cap ? *cap * 2 : 64;
        while (new_cap < *len + slen + 1) new_cap *= 2;
        *buf = (char *)realloc(*buf, new_cap);
        if (!*buf) exit(1);
        *cap = new_cap;
    }
    memcpy(*buf + *len, s, slen);
    *len += slen;
    (*buf)[*len] = '\0';
}

#ifdef _WIN32
static wchar_t *utf8_to_wide(const char *input, int *wide_len) {
    int needed = MultiByteToWideChar(CP_UTF8, 0, input, -1, NULL, 0);
    if (needed <= 0) return NULL;
    wchar_t *w = (wchar_t *)xmalloc((size_t)needed * sizeof(wchar_t));
    if (MultiByteToWideChar(CP_UTF8, 0, input, -1, w, needed) == 0) {
        free(w);
        return NULL;
    }
    if (wide_len) *wide_len = needed - 1;
    return w;
}

static char *wide_to_utf8(const wchar_t *w) {
    int needed = WideCharToMultiByte(CP_UTF8, 0, w, -1, NULL, 0, NULL, NULL);
    if (needed <= 0) return NULL;
    char *out = (char *)xmalloc((size_t)needed);
    if (WideCharToMultiByte(CP_UTF8, 0, w, -1, out, needed, NULL, NULL) == 0) {
        free(out);
        return NULL;
    }
    return out;
}
#endif

char *utf_normalize_nfkc(const char *input) {
#ifdef _WIN32
    wchar_t *w = utf8_to_wide(input, NULL);
    if (!w) return xstrdup(input);

    int norm_len = NormalizeString(NormalizationC, w, -1, NULL, 0);
    if (norm_len <= 0) {
        free(w);
        return xstrdup(input);
    }
    wchar_t *norm = (wchar_t *)xmalloc((size_t)norm_len * sizeof(wchar_t));
    if (NormalizeString(NormalizationC, w, -1, norm, norm_len) == 0) {
        free(norm);
        free(w);
        return xstrdup(input);
    }
    free(w);
    char *out = wide_to_utf8(norm);
    free(norm);
    return out ? out : xstrdup(input);
#else
    /* Linux/macOS: link with libicu for full NFKC; fallback is identity. */
    return xstrdup(input);
#endif
}

static uint32_t casefold_codepoint(uint32_t cp) {
    switch (cp) {
        case 0x0049: return 0x0069; /* LATIN CAPITAL LETTER I -> i (Turkish dotless handling via context not done) */
        case 0x0130: return 0x0069; /* LATIN CAPITAL LETTER I WITH DOT ABOVE -> i */
        case 0x004C: return 0x006C;
        case 0x00DF: return 0x0073; /* eszett fold */
        default:
            if (cp >= 'A' && cp <= 'Z') return cp + 32;
            return cp;
    }
}

char *utf_casefold(const char *input) {
#ifdef _WIN32
    wchar_t *w = utf8_to_wide(input, NULL);
    if (!w) return xstrdup(input);

    int norm_len = NormalizeString(NormalizationC, w, -1, NULL, 0);
    if (norm_len <= 0) {
        free(w);
        return xstrdup(input);
    }
    wchar_t *norm = (wchar_t *)xmalloc((size_t)norm_len * sizeof(wchar_t));
    if (NormalizeString(NormalizationC, w, -1, norm, norm_len) == 0) {
        free(norm);
        free(w);
        return xstrdup(input);
    }
    free(w);

    for (int i = 0; norm[i]; i++) {
        wchar_t ch = norm[i];
        if (ch == 0x0130) norm[i] = 0x0069;
        else if (ch == 0x0049) norm[i] = 0x0069;
        else if (ch >= 'A' && ch <= 'Z') norm[i] = ch + 32;
    }

    char *out = wide_to_utf8(norm);
    free(norm);
    return out ? out : xstrdup(input);
#else
    const char *p = input;
    const char *end = input + strlen(input);
    char *buf = NULL;
    size_t len = 0, cap = 0;
    uint32_t cp;
    while (utf8_decode(&p, end, &cp) == 1) {
        utf8_append_codepoint(&buf, &len, &cap, casefold_codepoint(cp));
    }
    if (!buf) return xstrdup("");
    return buf;
#endif
}

static int is_unicode_space(uint32_t cp) {
    return cp == 0x20 || cp == 0x09 || cp == 0x0A || cp == 0x0B || cp == 0x0C || cp == 0x0D ||
           cp == 0x85 || cp == 0xA0 || cp == 0x1680 || cp == 0x2000 || cp == 0x2001 || cp == 0x2002 ||
           cp == 0x2003 || cp == 0x2004 || cp == 0x2005 || cp == 0x2006 || cp == 0x2007 || cp == 0x2008 ||
           cp == 0x2009 || cp == 0x200A || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
           cp == 0x3000;
}

void utf_split_whitespace_to_map(const char *line, StrU64Map *word_freqs) {
    const char *p = line;
    const char *end = line + strlen(line);

    while (p < end) {
        uint32_t cp;
        if (utf8_decode(&p, end, &cp) != 1) continue;
        if (is_unicode_space(cp)) continue;

        const char *start = p - 1;
        while (start > line && (*start & 0xC0) == 0x80) start--;

        while (p < end) {
            const char *peek = p;
            if (utf8_decode(&peek, end, &cp) != 1) break;
            if (is_unicode_space(cp)) break;
            p = peek;
        }

        char *word = utf_buffer_dup(start, (size_t)(p - start));
        str_u64_map_add(word_freqs, word, 1);
        free(word);
    }
}

void utf_collect_chars(const char *s, CharPtrArray *chars) {
    const char *p = s;
    const char *end = s + strlen(s);
    uint32_t cp;

    while (p < end) {
        const char *start = p;
        if (utf8_decode(&p, end, &cp) != 1) {
            p++;
            continue;
        }
        char *tmp = utf_buffer_dup(start, (size_t)(p - start));
        if (!char_ptr_array_contains(chars, tmp)) {
            char_ptr_array_push(chars, tmp);
        } else {
            free(tmp);
        }
    }
}
