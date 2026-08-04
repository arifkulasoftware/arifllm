# Pure C BPE tokenizer trainer (no Hugging Face tokenizers library)

CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
LDFLAGS ?=

SRC = build_tokenizer.c utf.c bpe.c
OBJ = $(SRC:.c=.o)
TARGET = build_tokenizer

ifeq ($(OS),Windows_NT)
    LDFLAGS += -lnormaliz
    TARGET := build_tokenizer.exe
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        LDFLAGS += -lnormaliz
    endif
endif

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET) build_tokenizer.exe
