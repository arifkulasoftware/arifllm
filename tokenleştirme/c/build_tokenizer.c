/*
 * build_tokenizer.c
 *
 * Pure C BPE tokenizer trainer — equivalent to build_tokenizer.py without
 * the Hugging Face tokenizers library.
 *
 * Build (MSVC):
 *   cl /O2 /W4 build_tokenizer.c utf.c bpe.c /Fe:build_tokenizer.exe Normaliz.lib
 *
 * Build (GCC/Clang):
 *   gcc -O2 -Wall -Wextra -o build_tokenizer build_tokenizer.c utf.c bpe.c -lnormaliz
 */

#include "common.h"
#include "utf.h"
#include "bpe.h"

#include <ctype.h>
#include <sys/stat.h>
#include <time.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <direct.h>
#define PATH_SEP '\\'
#else
#include <dirent.h>
#include <unistd.h>
#define PATH_SEP '/'
#endif

static FILE *log_fp = NULL;

static void log_line(const char *msg) {
    time_t now = time(NULL);
    struct tm tm_now;
#ifdef _WIN32
    localtime_s(&tm_now, &now);
#else
    localtime_r(&now, &tm_now);
#endif
    char stamp[32];
    snprintf(stamp, sizeof(stamp), "%04d-%02d-%02d %02d:%02d:%02d",
             tm_now.tm_year + 1900, tm_now.tm_mon + 1, tm_now.tm_mday,
             tm_now.tm_hour, tm_now.tm_min, tm_now.tm_sec);
    fprintf(stdout, "[%s] %s\n", stamp, msg);
    if (log_fp) {
        fprintf(log_fp, "[%s] %s\n", stamp, msg);
        fflush(log_fp);
    }
}

static int ends_with_txt(const char *name) {
    size_t n = strlen(name);
    if (n < 4) return 0;
    const char *ext = name + n - 4;
    return ext[0] == '.' &&
           (tolower((unsigned char)ext[1]) == 't') &&
           (tolower((unsigned char)ext[2]) == 'x') &&
           (tolower((unsigned char)ext[3]) == 't');
}

static void join_path(char *out, size_t out_cap, const char *dir, const char *name) {
    size_t dlen = strlen(dir);
    if (dlen + 1 + strlen(name) + 1 > out_cap) {
        fprintf(stderr, "Path too long: %s/%s\n", dir, name);
        exit(1);
    }
    memcpy(out, dir, dlen);
    out[dlen] = PATH_SEP;
    memcpy(out + dlen + 1, name, strlen(name) + 1);
}

typedef struct {
    char **paths;
    size_t count;
    size_t capacity;
} PathList;

static void path_list_init(PathList *pl) {
    pl->paths = NULL;
    pl->count = 0;
    pl->capacity = 0;
}

static void path_list_push(PathList *pl, const char *path) {
    if (pl->count >= pl->capacity) {
        size_t new_cap = pl->capacity ? pl->capacity * 2 : 256;
        pl->paths = (char **)realloc(pl->paths, new_cap * sizeof(char *));
        if (!pl->paths) exit(1);
        pl->capacity = new_cap;
    }
    pl->paths[pl->count++] = xstrdup(path);
}

static void path_list_free(PathList *pl) {
    for (size_t i = 0; i < pl->count; i++) free(pl->paths[i]);
    free(pl->paths);
    pl->paths = NULL;
    pl->count = 0;
    pl->capacity = 0;
}

#ifdef _WIN32
static void find_txt_files_win(const char *dir, PathList *pl) {
    char pattern[MAX_PATH];
    snprintf(pattern, sizeof(pattern), "%s\\*", dir);

    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) return;

    do {
        if (strcmp(fd.cFileName, ".") == 0 || strcmp(fd.cFileName, "..") == 0) continue;
        char full[MAX_PATH];
        join_path(full, sizeof(full), dir, fd.cFileName);

        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            find_txt_files_win(full, pl);
        } else if (ends_with_txt(fd.cFileName)) {
            path_list_push(pl, full);
        }
    } while (FindNextFileA(h, &fd));

    FindClose(h);
}
#else
static void find_txt_files_posix(const char *dir, PathList *pl) {
    DIR *d = opendir(dir);
    if (!d) return;

    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;
        char full[4096];
        join_path(full, sizeof(full), dir, ent->d_name);

        struct stat st;
        if (stat(full, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            find_txt_files_posix(full, pl);
        } else if (ends_with_txt(ent->d_name)) {
            path_list_push(pl, full);
        }
    }
    closedir(d);
}
#endif

static void find_txt_files(const char *input_dir, PathList *pl) {
#ifdef _WIN32
    find_txt_files_win(input_dir, pl);
#else
    find_txt_files_posix(input_dir, pl);
#endif
}

static void write_checkpoint(const char *path, int processed_files, int processed_lines,
                             uint64_t processed_bytes, const char *last_file) {
    FILE *fp = fopen(path, "w");
    if (!fp) return;
    fprintf(fp, "processed_files=%d\n", processed_files);
    fprintf(fp, "processed_lines=%d\n", processed_lines);
    fprintf(fp, "processed_bytes=%llu\n", (unsigned long long)processed_bytes);
    fprintf(fp, "last_file=%s\n", last_file);
    fclose(fp);
}

static int read_corpus(PathList *files, const Config *cfg, StrU64Map *word_freqs,
                       int *processed_lines, uint64_t *processed_bytes) {
    int processed_files = 0;
    *processed_lines = 0;
    *processed_bytes = 0;

    for (size_t i = 0; i < files->count; i++) {
        if (cfg->max_files > 0 && processed_files >= cfg->max_files) break;

        const char *path = files->paths[i];
        processed_files++;

        char msg[1024];
        snprintf(msg, sizeof(msg), "Processing file %d/%zu: %s", processed_files, files->count, path);
        log_line(msg);

        FILE *fp = fopen(path, "rb");
        if (!fp) {
            snprintf(msg, sizeof(msg), "Cannot open file: %s", path);
            log_line(msg);
            continue;
        }

        char line[65536];
        while (fgets(line, sizeof(line), fp)) {
            if (cfg->max_lines > 0 && *processed_lines >= cfg->max_lines) {
                log_line("Reached max_lines limit");
                fclose(fp);
                return processed_files;
            }

            size_t len = strlen(line);
            while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
                line[--len] = '\0';
            }
            if (len == 0) continue;

            char *norm = utf_normalize_nfkc(line);
            char *text = cfg->lowercase ? utf_casefold(norm) : norm;
            if (cfg->lowercase && norm != text) free(norm);

            if (cfg->max_bytes > 0) {
                uint64_t nbytes = strlen(text);
                if (*processed_bytes + nbytes > cfg->max_bytes) {
                    log_line("Reached max_bytes limit");
                    free(text);
                    fclose(fp);
                    return processed_files;
                }
                *processed_bytes += nbytes;
            }

            utf_split_whitespace_to_map(text, word_freqs);
            (*processed_lines)++;
            free(text);
        }
        fclose(fp);

        if (cfg->checkpoint_path) {
            write_checkpoint(cfg->checkpoint_path, processed_files, *processed_lines,
                             *processed_bytes, path);
        }
    }

    return processed_files;
}

static void ensure_dir(const char *path) {
#ifdef _WIN32
    _mkdir(path);
#else
    mkdir(path, 0755);
#endif
}

static void usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s --input-dir DIR [options]\n\n"
            "Options:\n"
            "  --output-dir DIR         Output directory (default: .)\n"
            "  --vocab-size N           Target vocabulary size (default: 32000)\n"
            "  --min-frequency N        Minimum pair frequency (default: 1000)\n"
            "  --max-files N            Limit number of input files\n"
            "  --max-lines N            Limit number of training lines\n"
            "  --max-bytes N            Limit total training bytes\n"
            "  --lowercase              Apply casefold before training\n"
            "  --log-file PATH          Log file path\n"
            "  --checkpoint-file PATH   Checkpoint file path\n",
            prog);
}

static int parse_args(int argc, char **argv, Config *cfg) {
    cfg->input_dir = NULL;
    cfg->output_dir = ".";
    cfg->log_path = "tokenizer_training.log";
    cfg->checkpoint_path = NULL;
    cfg->vocab_size = 32000;
    cfg->min_frequency = 1000;
    cfg->max_files = -1;
    cfg->max_lines = -1;
    cfg->max_bytes = 0;
    cfg->lowercase = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input-dir") == 0 && i + 1 < argc) {
            cfg->input_dir = argv[++i];
        } else if (strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) {
            cfg->output_dir = argv[++i];
        } else if (strcmp(argv[i], "--vocab-size") == 0 && i + 1 < argc) {
            cfg->vocab_size = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--min-frequency") == 0 && i + 1 < argc) {
            cfg->min_frequency = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-files") == 0 && i + 1 < argc) {
            cfg->max_files = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-lines") == 0 && i + 1 < argc) {
            cfg->max_lines = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-bytes") == 0 && i + 1 < argc) {
            cfg->max_bytes = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--lowercase") == 0) {
            cfg->lowercase = 1;
        } else if (strcmp(argv[i], "--log-file") == 0 && i + 1 < argc) {
            cfg->log_path = argv[++i];
        } else if (strcmp(argv[i], "--checkpoint-file") == 0 && i + 1 < argc) {
            cfg->checkpoint_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            return -1;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return -1;
        }
    }

    if (!cfg->input_dir) {
        fprintf(stderr, "--input-dir is required\n");
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    Config cfg;
    if (parse_args(argc, argv, &cfg) != 0) {
        usage(argv[0]);
        return 1;
    }

    ensure_dir(cfg.output_dir);

    char log_full[4096];
    snprintf(log_full, sizeof(log_full), "%s%c%s", cfg.output_dir, PATH_SEP, cfg.log_path);
    log_fp = fopen(log_full, "a");
    if (!log_fp) {
        fprintf(stderr, "Warning: cannot open log file %s\n", log_full);
    }

    char checkpoint_full[4096];
    if (cfg.checkpoint_path) {
        snprintf(checkpoint_full, sizeof(checkpoint_full), "%s%c%s",
                 cfg.output_dir, PATH_SEP, cfg.checkpoint_path);
        cfg.checkpoint_path = checkpoint_full;
    }

    log_line("Starting tokenizer training (pure C BPE)");

    PathList files;
    path_list_init(&files);
    find_txt_files(cfg.input_dir, &files);

    if (files.count == 0) {
        log_line("No .txt files found");
        if (log_fp) fclose(log_fp);
        return 1;
    }

    char msg[256];
    snprintf(msg, sizeof(msg), "Found %zu .txt files", files.count);
    log_line(msg);

    StrU64Map word_freqs;
    str_u64_map_init(&word_freqs, 1048576);

    int processed_lines = 0;
    uint64_t processed_bytes = 0;
    int processed_files = read_corpus(&files, &cfg, &word_freqs, &processed_lines, &processed_bytes);

    snprintf(msg, sizeof(msg), "Corpus read: %d files, %d lines, %llu bytes, %zu unique words",
             processed_files, processed_lines, (unsigned long long)processed_bytes, word_freqs.size);
    log_line(msg);

    SymbolTable symbols;
    symbol_table_init(&symbols);

    CharPtrArray sorted_chars;
    collect_and_add_base_chars(&word_freqs, &symbols, &sorted_chars);
    char_ptr_array_free(&sorted_chars);

    snprintf(msg, sizeof(msg), "Base vocabulary: %u symbols (including specials)", symbols.count);
    log_line(msg);

    WordTable *words = build_word_table_from_freqs(&word_freqs, &symbols);
    snprintf(msg, sizeof(msg), "Word table: %zu entries", words->count);
    log_line(msg);

    MergeList merges;
    log_line("Beginning BPE merges...");
    bpe_train(words, &symbols, &merges, cfg.vocab_size, cfg.min_frequency, log_fp);

    snprintf(msg, sizeof(msg), "Training complete: vocab_size=%u merges=%u", symbols.count, merges.count);
    log_line(msg);

    char out_path[4096];
    snprintf(out_path, sizeof(out_path), "%s%ckendi_tokenizerim.json", cfg.output_dir, PATH_SEP);

    if (write_tokenizer_json(out_path, &symbols, &merges) != 0) {
        log_line("Failed to write tokenizer JSON");
        word_table_free(words);
        symbol_table_free(&symbols);
        free(merges.pairs);
        str_u64_map_free(&word_freqs);
        path_list_free(&files);
        if (log_fp) fclose(log_fp);
        return 1;
    }

    snprintf(msg, sizeof(msg), "Tokenizer saved to: %s", out_path);
    log_line(msg);

    word_table_free(words);
    symbol_table_free(&symbols);
    free(merges.pairs);
    str_u64_map_free(&word_freqs);
    path_list_free(&files);

    if (log_fp) fclose(log_fp);
    return 0;
}
