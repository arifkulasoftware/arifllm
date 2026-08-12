{$mode objfpc}{$H+}
{$asmmode gas}

type
  TTokenProc = procedure(UserData: Pointer; Start, Size: Int64); cdecl;

procedure Buf_Isle(pBuf: Pointer; pSize: Int64; pOut: PWord;
  OnToken: TTokenProc; UserData: Pointer); assembler; nostackframe;
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
  mov   x21, x2                  // outPtr
  mov   x22, x2                  // outBase
  mov   x23, x3                  // OnToken
  mov   x24, x4                  // UserData
  mov   x26, #-1                 // tokenStart
  mov   x27, #0                  // inToken
  mov   x28, #0                  // outPos

.Lloop:
  cmp   x19, x20
  b.ge  .Lflush

  ldrb  w2, [x19], #1

  cmp   w2, #127
  b.gt  .Lutf8

  mov   w3, w2

  cmp   w2, #97
  b.lt  .Lnot_lower
  cmp   w2, #122
  b.le  .Lemit

.Lnot_lower:
  cmp   w2, #65
  b.lt  .Lnot_upper
  cmp   w2, #90
  b.gt  .Lnot_upper
  cmp   w2, #73
  b.eq  .Lto_131
  add   w3, w2, #32
  b     .Lemit

.Lto_131:
  mov   w3, #305                 // $0131
  b     .Lemit

.Lnot_upper:
  cmp   w2, #48
  b.lt  .Lspace
  cmp   w2, #57
  b.le  .Lemit
  b     .Lspace

.Lutf8:
  cmp   x19, x20
  b.ge  .Lflush
  ldrb  w4, [x19], #1
  and   w5, w2, #31
  and   w6, w4, #63
  lsl   w5, w5, #6
  orr   w3, w5, w6

  cmp   w3, #305
  b.eq  .Lemit
  cmp   w3, #287
  b.eq  .Lemit
  cmp   w3, #351
  b.eq  .Lemit
  cmp   w3, #231
  b.eq  .Lemit
  cmp   w3, #246
  b.eq  .Lemit
  cmp   w3, #252
  b.eq  .Lemit

  cmp   w3, #304
  b.ne  .Lm286
  mov   w3, #305
  b     .Lemit
.Lm286:
  cmp   w3, #286
  b.ne  .Lm350
  mov   w3, #287
  b     .Lemit
.Lm350:
  cmp   w3, #350
  b.ne  .Lm199
  mov   w3, #351
  b     .Lemit
.Lm199:
  cmp   w3, #199
  b.ne  .Lm214
  mov   w3, #231
  b     .Lemit
.Lm214:
  cmp   w3, #214
  b.ne  .Lm220
  mov   w3, #246
  b     .Lemit
.Lm220:
  cmp   w3, #220
  b.ne  .Lspace
  mov   w3, #252
  b     .Lemit

.Lspace:
  mov   w3, #32

.Lemit:
  cmp   w3, #32
  b.eq  .Ltok_end

  cmp   x27, #0
  b.ne  .Ltok_on
  mov   x26, x28
  mov   x27, #1
.Ltok_on:
  strh  w3, [x21], #2
  add   x28, x28, #1
  b     .Lloop

.Ltok_end:
  cmp   x27, #0
  b.eq  .Lloop
  sub   x2, x28, x26              // size
  mov   x0, x24
  mov   x1, x26
  blr   x23
  mov   x27, #0
  mov   x26, #-1
  b     .Lloop

.Lflush:
  cmp   x27, #0
  b.eq  .Ldone
  sub   x2, x28, x26
  mov   x0, x24
  mov   x1, x26
  blr   x23

.Ldone:
  ldp   x27, x28, [sp, #80]
  ldp   x25, x26, [sp, #64]
  ldp   x23, x24, [sp, #48]
  ldp   x21, x22, [sp, #32]
  ldp   x19, x20, [sp, #16]
  ldp   x29, x30, [sp], #112
  ret
end;

procedure test_cb(UserData: Pointer; Start, Size: Int64); cdecl;
begin
  Writeln('token @', Start, ' len=', Size);
end;

var
  InB: array[0..15] of Byte;
  OutW: array[0..31] of Word;

begin
  InB[0]:=Ord('M'); InB[1]:=Ord('e'); InB[2]:=Ord('r');
  InB[3]:=Ord('h'); InB[4]:=Ord('a'); InB[5]:=Ord('b');
  InB[6]:=Ord('a'); InB[7]:=32;
  InB[8]:=$C4; InB[9]:=$B0;  // İ UTF-8
  InB[10]:=Ord('s'); InB[11]:=Ord('t');
  InB[12]:=Ord('a'); InB[13]:=Ord('n');
  InB[14]:=Ord('b'); InB[15]:=Ord('u');
  Buf_Isle(@InB[0], 16, @OutW[0], @test_cb, nil);
end.
