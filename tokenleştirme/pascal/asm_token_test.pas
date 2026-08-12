{$mode objfpc}{$H+}
{$OPTIMIZATION ON}
program asm_token_test;

type
  PWordBuf = ^Word;

var
  gOutBuf: PWordBuf;
  gOutPos: Int64;
  gTokenCtx: Pointer;

procedure Token_Ekle(var pToken: Pointer; pStart, pSize: Int64);
begin
  Writeln('Token @', pStart, ' len=', pSize, ' ctx=', PtrUInt(pToken));
end;

{ Token_Ekle cagrisi icin kucuk Pascal koprusu (asm -> bl) }
procedure EmitToken; assembler; nostackframe;
var
  S, N: Int64;
  T: Pointer;
asm
  adrp  x9, S
  add   x9, x9, :lo12:S
  ldr   x1, [x9]                 // pStart

  adrp  x9, N
  add   x9, x9, :lo12:N
  ldr   x2, [x9]                 // pSize

  adrp  x9, T
  add   x9, x9, :lo12:T
  ldr   x0, [x9]                 // @pToken (Pointer degiskeninin adresi)

  bl    P$ASM_TOKEN_TEST_$$_TOKEN_EKLE$TPTRINT64$INT64$INT64
end;

{
  UTF-8 Turkce metni UTF-16LE'ye cevirir, token sinirlarini bulur.
  pBuf : UTF-8 kaynak
  pSize: byte sayisi (max ~256MB)

  Kurallar:
    a-z (97-122)       -> ayni
    A-Z (65-90), I haric -> +32
    I (73)             -> $0131
    0-9 (48-57)        -> ayni
    UTF-8 2-byte       -> decode, Turkce harf map, diger -> $0020
    diger ASCII        -> $0020 (space)
}
procedure Buf_Isle(pBuf: Pointer; pSize: Int64); assembler; nostackframe;
asm
  stp   x29, x30, [sp, #-112]!
  mov   x29, sp
  stp   x19, x20, [sp, #16]
  stp   x21, x22, [sp, #32]
  stp   x23, x24, [sp, #48]
  stp   x25, x26, [sp, #64]
  stp   x27, x28, [sp, #80]

  mov   x19, x0                  // inPtr
  add   x20, x0, x1              // inEnd

  adrp  x9, gOutBuf
  ldr   x21, [x9, :lo12:gOutBuf] // outPtr (PWord)
  mov   x22, x21                 // outBase

  adrp  x9, gTokenCtx
  ldr   x28, [x9, :lo12:gTokenCtx] // @Token degiskeni

  mov   x26, #-1                 // tokenStart (wide index)
  mov   x27, #0                  // inToken flag

  adrp  x9, gOutPos
  str   xzr, [x9, :lo12:gOutPos]

.Lmain_loop:
  cmp   x19, x20
  b.ge  .Lflush_token

  ldrb  w2, [x19], #1            // oku, inPtr++

  // --- UTF-8 2-byte yolu (lead > 127) ---
  cmp   w2, #127
  b.gt  .Lutf8_2byte

  // --- ASCII yolu ---
  mov   w3, w2                   // w3 = cikis WideChar

  // a-z
  cmp   w2, #97
  b.lt  .Lascii_not_lower
  cmp   w2, #122
  b.le  .Lwrite_char
.Lascii_not_lower:

  // A-Z (I=73 haric -> +32)
  cmp   w2, #65
  b.lt  .Lascii_not_upper
  cmp   w2, #90
  b.gt  .Lascii_not_upper
  cmp   w2, #73
  b.eq  .Lascii_I
  add   w3, w2, #32
  b     .Lwrite_char
.Lascii_I:
  mov   w3, #$31
  movk  w3, #$01, lsl #16        // $0131
  b     .Lwrite_char

.Lascii_not_upper:
  // 0-9
  cmp   w2, #48
  b.lt  .Lascii_space
  cmp   w2, #57
  b.le  .Lwrite_char
.Lascii_space:
  mov   w3, #32
  b     .Lwrite_char

  // --- UTF-8 2-byte decode ---
.Lutf8_2byte:
  cmp   x19, x20
  b.ge  .Lflush_token
  ldrb  w4, [x19], #1            // trail byte

  and   w5, w2, #$1F
  and   w6, w4, #$3F
  lsl   w5, w5, #6
  orr   w3, w5, w6               // w3 = Unicode code point

  // kucuk harfler: aynen
  cmp   w3, #$0131
  b.eq  .Lwrite_char
  cmp   w3, #$011F
  b.eq  .Lwrite_char
  cmp   w3, #$015F
  b.eq  .Lwrite_char
  cmp   w3, #$00E7
  b.eq  .Lwrite_char
  cmp   w3, #$00F6
  b.eq  .Lwrite_char
  cmp   w3, #$00FC
  b.eq  .Lwrite_char

  // buyuk -> kucuk map
  cmp   w3, #$0130
  b.ne  .Lmap_011E
  mov   w3, #$0131               // kullanici spec: $0130 -> $0131
  b     .Lwrite_char
.Lmap_011E:
  cmp   w3, #$011E
  b.ne  .Lmap_015E
  mov   w3, #$011F
  b     .Lwrite_char
.Lmap_015E:
  cmp   w3, #$015E
  b.ne  .Lmap_00C7
  mov   w3, #$015F
  b     .Lwrite_char
.Lmap_00C7:
  cmp   w3, #$00C7
  b.ne  .Lmap_00D6
  mov   w3, #$00E7
  b     .Lwrite_char
.Lmap_00D6:
  cmp   w3, #$00D6
  b.ne  .Lmap_00DC
  mov   w3, #$00F6
  b     .Lwrite_char
.Lmap_00DC:
  cmp   w3, #$00DC
  b.ne  .Lutf8_space
  mov   w3, #$00FC
  b     .Lwrite_char
.Lutf8_space:
  mov   w3, #32

  // --- yaz + token mantigi ---
.Lwrite_char:
  cmp   w3, #32
  b.eq  .Lhandle_space

  // token baslat
  cmp   x27, #0
  b.ne  .Ltok_started
  adrp  x9, gOutPos
  ldr   x26, [x9, :lo12:gOutPos]
  mov   x27, #1
.Ltok_started:
  strh  w3, [x21], #2            // UTF-16LE yaz
  adrp  x9, gOutPos
  ldr   x25, [x9, :lo12:gOutPos]
  add   x25, x25, #1
  str   x25, [x9, :lo12:gOutPos]
  b     .Lmain_loop

.Lhandle_space:
  cmp   x27, #0
  b.eq  .Lmain_loop
  // Token_Ekle(tokenStart, outPos - tokenStart)
  adrp  x9, gOutPos
  ldr   x25, [x9, :lo12:gOutPos]
  sub   x25, x25, x26             // size (wide)
  adrp  x9, EmitToken$S
  str   x26, [x9, :lo12:EmitToken$S]
  adrp  x9, EmitToken$N
  str   x25, [x9, :lo12:EmitToken$N]
  adrp  x9, EmitToken$T
  str   x28, [x9, :lo12:EmitToken$T]
  bl    P$ASM_TOKEN_TEST_$$_EMITTOKEN
  mov   x27, #0
  mov   x26, #-1
  b     .Lmain_loop

.Lflush_token:
  cmp   x27, #0
  b.eq  .Ldone
  adrp  x9, gOutPos
  ldr   x25, [x9, :lo12:gOutPos]
  sub   x25, x25, x26
  adrp  x9, EmitToken$S
  str   x26, [x9, :lo12:EmitToken$S]
  adrp  x9, EmitToken$N
  str   x25, [x9, :lo12:EmitToken$N]
  adrp  x9, EmitToken$T
  str   x28, [x9, :lo12:EmitToken$T]
  bl    P$ASM_TOKEN_TEST_$$_EMITTOKEN

.Ldone:
  ldp   x27, x28, [sp, #80]
  ldp   x25, x26, [sp, #64]
  ldp   x23, x24, [sp, #48]
  ldp   x21, x22, [sp, #32]
  ldp   x19, x20, [sp, #16]
  ldp   x29, x30, [sp], #112
  ret
.end

var
  TestToken: Pointer;
  InBytes: array[0..31] of Byte;
  OutWords: array[0..63] of Word;

begin
  TestToken := Pointer(1);
  gTokenCtx := @TestToken;
  gOutBuf := @OutWords[0];
  gOutPos := 0;

  { "Merhaba İstanbul" UTF-8 }
  InBytes[0] := Ord('M'); InBytes[1] := Ord('e'); InBytes[2] := Ord('r');
  InBytes[3] := Ord('h'); InBytes[4] := Ord('a'); InBytes[5] := Ord('b');
  InBytes[6] := Ord('a'); InBytes[7] := Ord(' ');
  InBytes[8] := $C4; InBytes[9] := $B0;  // İ
  InBytes[10] := Ord('s'); InBytes[11] := Ord('t');
  InBytes[12] := Ord('a'); InBytes[13] := Ord('n'); InBytes[14] := Ord('b');
  InBytes[15] := Ord('u'); InBytes[16] := Ord('l');

  Buf_Isle(@InBytes[0], 17);
  Halt(0);
end.
