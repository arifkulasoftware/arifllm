{$mode objfpc}{$H+}
{$IFDEF DEBUG}
  {$OPTIMIZATION OFF}
  {$ASSERTIONS ON}
{$ELSE}
  {$OPTIMIZATION ON}
  {$ASSERTIONS OFF}
{$ENDIF}
program build_tokenizer;

uses
{$ifdef unix}
  cmem,
  cthreads,
{$endif}
  SysUtils,
  Classes;

const
  FNV_OFFSET = QWord($cbf29ce484222325);
  FNV_PRIME  = QWord($100000001b3);
  INITIAL_BUCKETS = 65536;
  INITIAL_ARENA   = 4 * 1024 * 1024;
  LOAD_FACTOR_NUM = 7;
  LOAD_FACTOR_DEN = 10;
  SPECIAL_TOKEN_COUNT = 7;
  DEFAULT_MAKSIMUM_TOKEN = 8192;
  JSON_WRITE_BUF = 1024 * 1024;

type
  TTopEntry = record
    KeyOff: Int32;
    KeyLen: Int32;
    Count: Int32;
  end;

  TTopEntryArray = array of TTopEntry;

  TTokenSlot = record
    Hash: QWord;
    KeyOff: Int32;
    KeyLen: Int32;
    Count: Int32;
  end;

  TTokenFreqTable = object
  private
    FBuckets: array of TTokenSlot;
    FBucketMask: Integer;
    FBucketCount: Integer;
    FUsedSlots: Integer;
    FArena: PByte;
    FArenaCap: Int64;
    FArenaPos: Int64;
    FUniqueCount: QWord;
    FTotalTokens: QWord;
    function HashBytes(const p: PByte; Len: Int64): QWord;
    function CollectAllEntries: TTopEntryArray;
    procedure EnsureArena(Need: Int64);
    procedure GrowBuckets;
    procedure MergeInsert(const pToken: PByte; pLen: Int64; Cnt: Int32);
  public
    procedure Init(InitialBuckets: Integer = INITIAL_BUCKETS);
    procedure Done;
    procedure IncOrInsert(const pToken: PByte; pLen: Int64);
    procedure MergeFrom(const Src: TTokenFreqTable);
    procedure PrintTop(N: Integer);
    function SaveTokenizerJson(const Path: string; MaxVocab: Integer): Integer;
    property TotalTokens: QWord read FTotalTokens;
    property UniqueCount: QWord read FUniqueCount;
  end;

  TMyThread = class(TThread)
  private
    FFileName: string;
    FLocalFreq: TTokenFreqTable;
  protected
    procedure Execute; override;
  public
    constructor Create(const pFileName: string);
    destructor Destroy; override;
  end;

var
  DataDir: string;
  FilePattern: string;
  OutputDir: string;
  MaxThreads: Integer;
  MaxPrint: Integer;
  MaksimumToken: Integer;
  ThreadCount: Integer = 0;
  GlobalFreq: TTokenFreqTable;
  FreqLock: TRTLCriticalSection;
  GThreadFreq: ^TTokenFreqTable = nil;

function SpecialTokenText(Index: Integer): string;
begin
  case Index of
    0: Result := '[PAD]';
    1: Result := '[UNK]';
    2: Result := '[CLS]';
    3: Result := '[SEP]';
    4: Result := '[MASK]';
    5: Result := '[SOS]';
    6: Result := '[EOS]';
  else
    Result := '';
  end;
end;

procedure JsonEscapeAndWrite(const p: PByte; Len: Int64; out F: Text);
var
  I: Int64;
  C: Byte;
begin
  for I := 0 to Len - 1 do
  begin
    C := PByte(PtrUInt(p) + PtrUInt(I))^;
    case C of
      Ord('"'): Write(F, '\"');
      Ord('\'): Write(F, '\\');
      Ord(#8): Write(F, '\b');
      Ord(#12): Write(F, '\f');
      Ord(#10): Write(F, '\n');
      Ord(#13): Write(F, '\r');
      Ord(#9): Write(F, '\t');
    else
      if C < 32 then
        Write(F, '\u00', HexStr(C shr 4, 1), HexStr(C and $F, 1))
      else
        Write(F, Char(C));
    end;
  end;
end;

procedure HeapSiftDown(var Heap: array of TTopEntry; Root, HeapSize: Integer);
var
  Cur, Smallest, Left, Right: Integer;
  Tmp: TTopEntry;
begin
  Cur := Root;
  while True do
  begin
    Smallest := Cur;
    Left := (Cur shl 1) + 1;
    Right := Left + 1;
    if (Left < HeapSize) and (Heap[Left].Count < Heap[Smallest].Count) then
      Smallest := Left;
    if (Right < HeapSize) and (Heap[Right].Count < Heap[Smallest].Count) then
      Smallest := Right;
    if Smallest = Cur then
      Break;
    Tmp := Heap[Cur];
    Heap[Cur] := Heap[Smallest];
    Heap[Smallest] := Tmp;
    Cur := Smallest;
  end;
end;

procedure TopKSelect(const Items: array of TTopEntry; ItemCount, K: Integer;
  out Top: array of TTopEntry; out TopCount: Integer);
var
  Heap: array of TTopEntry;
  HeapSize, I, J: Integer;
  Tmp: TTopEntry;
begin
  TopCount := 0;
  if (K <= 0) or (ItemCount <= 0) then
    Exit;
  if K > ItemCount then
    K := ItemCount;

  SetLength(Heap, K);
  HeapSize := 0;
  for I := 0 to ItemCount - 1 do
  begin
    if HeapSize < K then
    begin
      Heap[HeapSize] := Items[I];
      Inc(HeapSize);
      if HeapSize = K then
        for J := (K div 2) - 1 downto 0 do
          HeapSiftDown(Heap, J, HeapSize);
    end
    else if Items[I].Count > Heap[0].Count then
    begin
      Heap[0] := Items[I];
      HeapSiftDown(Heap, 0, HeapSize);
    end;
  end;

  TopCount := HeapSize;
  for I := 0 to HeapSize - 1 do
    Top[I] := Heap[I];

  for I := 0 to TopCount - 2 do
    for J := I + 1 to TopCount - 1 do
      if Top[J].Count > Top[I].Count then
      begin
        Tmp := Top[I];
        Top[I] := Top[J];
        Top[J] := Tmp;
      end;
end;

function TTokenFreqTable.CollectAllEntries: TTopEntryArray;
var
  I, N: Integer;
begin
  SetLength(Result, FUsedSlots);
  N := 0;
  for I := 0 to FBucketCount - 1 do
  begin
    if FBuckets[I].KeyLen = 0 then
      Continue;
    Result[N].KeyOff := FBuckets[I].KeyOff;
    Result[N].KeyLen := FBuckets[I].KeyLen;
    Result[N].Count := FBuckets[I].Count;
    Inc(N);
  end;
  SetLength(Result, N);
end;

function TTokenFreqTable.HashBytes(const p: PByte; Len: Int64): QWord;
var
  I: Int64;
  B: Byte;
begin
  Result := FNV_OFFSET;
  for I := 0 to Len - 1 do
  begin
    B := PByte(PtrUInt(p) + PtrUInt(I))^;
    Result := Result xor B;
    Result := Result * FNV_PRIME;
  end;
end;

procedure TTokenFreqTable.Init(InitialBuckets: Integer);
var
  Pow2: Integer;
begin
  Pow2 := 16;
  while Pow2 < InitialBuckets do
    Pow2 := Pow2 shl 1;
  FBucketCount := Pow2;
  FBucketMask := FBucketCount - 1;
  SetLength(FBuckets, FBucketCount);
  FillChar(FBuckets[0], FBucketCount * SizeOf(TTokenSlot), 0);
  FUsedSlots := 0;
  FArenaCap := INITIAL_ARENA;
  GetMem(FArena, FArenaCap);
  FArenaPos := 0;
  FUniqueCount := 0;
  FTotalTokens := 0;
end;

procedure TTokenFreqTable.Done;
begin
  if FArena <> nil then
  begin
    FreeMem(FArena);
    FArena := nil;
  end;
  SetLength(FBuckets, 0);
  FBucketCount := 0;
  FBucketMask := 0;
  FUsedSlots := 0;
  FArenaCap := 0;
  FArenaPos := 0;
end;

procedure TTokenFreqTable.EnsureArena(Need: Int64);
var
  NewCap: Int64;
begin
  if FArenaPos + Need <= FArenaCap then
    Exit;
  NewCap := FArenaCap;
  while NewCap < FArenaPos + Need do
    NewCap := NewCap shl 1;
  ReAllocMem(FArena, NewCap);
  FArenaCap := NewCap;
end;

procedure TTokenFreqTable.GrowBuckets;
var
  OldBuckets: array of TTokenSlot;
  OldCount, I: Integer;
  SavedTotal: QWord;
begin
  OldBuckets := FBuckets;
  OldCount := FBucketCount;
  SavedTotal := FTotalTokens;

  FBucketCount := FBucketCount shl 1;
  FBucketMask := FBucketCount - 1;
  SetLength(FBuckets, FBucketCount);
  FillChar(FBuckets[0], FBucketCount * SizeOf(TTokenSlot), 0);
  FUsedSlots := 0;
  FUniqueCount := 0;

  for I := 0 to OldCount - 1 do
  begin
    if OldBuckets[I].KeyLen = 0 then
      Continue;
    MergeInsert(FArena + OldBuckets[I].KeyOff,
                OldBuckets[I].KeyLen,
                OldBuckets[I].Count);
  end;

  FTotalTokens := SavedTotal;
end;

procedure TTokenFreqTable.MergeInsert(const pToken: PByte; pLen: Int64; Cnt: Int32);
var
  H: QWord;
  Idx, Start: Integer;
  KeyPtr: PByte;
begin
  if pLen <= 0 then
    Exit;
  if (FUsedSlots + 1) * LOAD_FACTOR_DEN >= FBucketCount * LOAD_FACTOR_NUM then
    GrowBuckets;

  H := HashBytes(pToken, pLen);
  Start := Integer(H and QWord(FBucketMask));
  Idx := Start;
  repeat
    if FBuckets[Idx].KeyLen = 0 then
    begin
      EnsureArena(pLen);
      KeyPtr := FArena + FArenaPos;
      Move(pToken^, KeyPtr^, pLen);
      FBuckets[Idx].Hash := H;
      FBuckets[Idx].KeyOff := FArenaPos;
      FBuckets[Idx].KeyLen := pLen;
      FBuckets[Idx].Count := Cnt;
      Inc(FArenaPos, pLen);
      Inc(FUsedSlots);
      Inc(FUniqueCount);
      Exit;
    end;
    if (FBuckets[Idx].Hash = H) and (FBuckets[Idx].KeyLen = pLen) and
       CompareMem(FArena + FBuckets[Idx].KeyOff, pToken, pLen) then
    begin
      Inc(FBuckets[Idx].Count, Cnt);
      Exit;
    end;
    Idx := (Idx + 1) and FBucketMask;
  until Idx = Start;

  GrowBuckets;
  MergeInsert(pToken, pLen, Cnt);
end;

procedure TTokenFreqTable.IncOrInsert(const pToken: PByte; pLen: Int64);
begin
  if pLen <= 0 then
    Exit;
  Inc(FTotalTokens);
  MergeInsert(pToken, pLen, 1);
end;

procedure TTokenFreqTable.MergeFrom(const Src: TTokenFreqTable);
var
  I: Integer;
begin
  for I := 0 to Src.FBucketCount - 1 do
  begin
    if Src.FBuckets[I].KeyLen = 0 then
      Continue;
    MergeInsert(Src.FArena + Src.FBuckets[I].KeyOff,
                  Src.FBuckets[I].KeyLen,
                  Src.FBuckets[I].Count);
  end;
  Inc(FTotalTokens, Src.FTotalTokens);
end;

procedure TTokenFreqTable.PrintTop(N: Integer);
var
  AllItems: array of TTopEntry;
  TopItems: array of TTopEntry;
  TopCount, I: Integer;
  S: string;
begin
  if N <= 0 then
    Exit;

  AllItems := CollectAllEntries;
  SetLength(TopItems, N);
  TopKSelect(AllItems, Length(AllItems), N, TopItems, TopCount);

  Writeln('--- En sik ', TopCount, ' token ---');
  for I := 0 to TopCount - 1 do
  begin
    SetLength(S, TopItems[I].KeyLen);
    if TopItems[I].KeyLen > 0 then
      Move((FArena + TopItems[I].KeyOff)^, S[1], TopItems[I].KeyLen);
    Writeln(I + 1, #9, S, #9, TopItems[I].Count);
  end;
end;

function TTokenFreqTable.SaveTokenizerJson(const Path: string; MaxVocab: Integer): Integer;
var
  AllItems: array of TTopEntry;
  TopItems: array of TTopEntry;
  TopCount, WordSlots, VocabSize, I, TokenId: Integer;
  F: Text;
  Buf: array[0..JSON_WRITE_BUF - 1] of Char;
begin
  Result := 0;
  if MaxVocab <= SPECIAL_TOKEN_COUNT then
    Exit;

  WordSlots := MaxVocab - SPECIAL_TOKEN_COUNT;
  AllItems := CollectAllEntries;
  SetLength(TopItems, WordSlots);
  TopKSelect(AllItems, Length(AllItems), WordSlots, TopItems, TopCount);
  VocabSize := SPECIAL_TOKEN_COUNT + TopCount;

  Assign(F, Path);
  SetTextBuf(F, Buf, SizeOf(Buf));
  Rewrite(F);
  try
    WriteLn(F, '{');
    WriteLn(F, '  "version": "1.0",');
    WriteLn(F, '  "truncation": null,');
    WriteLn(F, '  "padding": null,');
    WriteLn(F, '  "added_tokens": [');

    for I := 0 to SPECIAL_TOKEN_COUNT - 1 do
    begin
      WriteLn(F, '    {');
      WriteLn(F, '      "id": ', I, ',');
      Write(F, '      "content": "');
      Write(F, SpecialTokenText(I));
      WriteLn(F, '",');
      WriteLn(F, '      "single_word": false,');
      WriteLn(F, '      "lstrip": false,');
      WriteLn(F, '      "rstrip": false,');
      WriteLn(F, '      "normalized": false,');
      Write(F, '      "special": true');
      if I + 1 < SPECIAL_TOKEN_COUNT then
        WriteLn(F, '    },')
      else
        WriteLn(F, '    }');
    end;

    WriteLn(F, '  ],');
    WriteLn(F, '  "normalizer": {');
    WriteLn(F, '    "type": "Sequence",');
    WriteLn(F, '    "normalizers": [');
    WriteLn(F, '      {');
    WriteLn(F, '        "type": "NFKC"');
    WriteLn(F, '      }');
    WriteLn(F, '    ]');
    WriteLn(F, '  },');
    WriteLn(F, '  "pre_tokenizer": {');
    WriteLn(F, '    "type": "Whitespace"');
    WriteLn(F, '  },');
    WriteLn(F, '  "post_processor": null,');
    WriteLn(F, '  "decoder": null,');
    WriteLn(F, '  "model": {');
    WriteLn(F, '    "type": "BPE",');
    WriteLn(F, '    "dropout": null,');
    WriteLn(F, '    "unk_token": "[UNK]",');
    WriteLn(F, '    "continuing_subword_prefix": null,');
    WriteLn(F, '    "end_of_word_suffix": null,');
    WriteLn(F, '    "fuse_unk": false,');
    WriteLn(F, '    "byte_fallback": false,');
    WriteLn(F, '    "ignore_merges": false,');
    WriteLn(F, '    "vocab": {');

    for I := 0 to SPECIAL_TOKEN_COUNT - 1 do
    begin
      Write(F, '      "');
      Write(F, SpecialTokenText(I));
      Write(F, '": ', I);
      WriteLn(F, ',');
    end;

    for I := 0 to TopCount - 1 do
    begin
      TokenId := SPECIAL_TOKEN_COUNT + I;
      Write(F, '      "');
      JsonEscapeAndWrite(FArena + TopItems[I].KeyOff, TopItems[I].KeyLen, F);
      Write(F, '": ', TokenId);
      if I + 1 < TopCount then
        WriteLn(F, ',')
      else
        WriteLn(F);
    end;

    WriteLn(F, '    },');
    WriteLn(F, '    "merges": [');
    WriteLn(F, '    ]');
    WriteLn(F, '  }');
    WriteLn(F, '}');
    Result := VocabSize;
  finally
    CloseFile(F);
  end;
end;

procedure kullanim_talimatlari;
begin
  Writeln('Kullanim: build_tokenizer [secenekler]');
  Writeln('  --input-dir <dizin>     Varsayilan: ./all_txt/');
  Writeln('  --pattern <glob>        Varsayilan: *.txt');
  Writeln('  --max-threads <N>       Varsayilan: CPU cekirdek sayisi (', GetCPUCount, ')');
  Writeln('  --max-print <N>         En sik N token (0=kapali)');
  Writeln('  --maksimum-token <N>    JSON vocab boyutu (varsayilan: ', DEFAULT_MAKSIMUM_TOKEN, ')');
  Writeln('  --output-dir <dizin>    kendi_tokenizerim.json cikti dizini (varsayilan: .)');
  Writeln('  --help                  Bu mesaj');
end;

procedure Arguman_Oku;
var
  I: Integer;
begin
  DataDir := './all_txt/';
  FilePattern := '*.txt';
  OutputDir := './';
  MaxThreads := GetCPUCount;
  MaxPrint := 0;
  MaksimumToken := DEFAULT_MAKSIMUM_TOKEN;

  I := 1;
  while I <= ParamCount do
  begin
    if ParamStr(I) = '--help' then
    begin
      kullanim_talimatlari;
      Halt(0);
    end
    else if ParamStr(I) = '--input-dir' then
    begin
      Inc(I);
      if I > ParamCount then
        raise Exception.Create('--input-dir degeri gerekli');
      DataDir := IncludeTrailingPathDelimiter(ParamStr(I));
    end
    else if ParamStr(I) = '--pattern' then
    begin
      Inc(I);
      if I > ParamCount then
        raise Exception.Create('--pattern degeri gerekli');
      FilePattern := ParamStr(I);
    end
    else if ParamStr(I) = '--max-threads' then
    begin
      Inc(I);
      if I > ParamCount then
        raise Exception.Create('--max-threads degeri gerekli');
      MaxThreads := StrToIntDef(ParamStr(I), 1);
    end
    else if ParamStr(I) = '--max-print' then
    begin
      Inc(I);
      if I > ParamCount then
        raise Exception.Create('--max-print degeri gerekli');
      MaxPrint := StrToIntDef(ParamStr(I), 0);
    end
    else if ParamStr(I) = '--maksimum-token' then
    begin
      Inc(I);
      if I > ParamCount then
        raise Exception.Create('--maksimum-token degeri gerekli');
      MaksimumToken := StrToIntDef(ParamStr(I), DEFAULT_MAKSIMUM_TOKEN);
      if MaksimumToken <= SPECIAL_TOKEN_COUNT then
        raise Exception.Create('--maksimum-token ozel tokenlardan buyuk olmali');
    end
    else if ParamStr(I) = '--output-dir' then
    begin
      Inc(I);
      if I > ParamCount then
        raise Exception.Create('--output-dir degeri gerekli');
      OutputDir := IncludeTrailingPathDelimiter(ParamStr(I));
    end
    else
      raise Exception.Create('Bilinmeyen arguman: ' + ParamStr(I));
    Inc(I);
  end;

  if MaxThreads < 1 then
    MaxThreads := 1;
end;

procedure Token_Ekle(const pToken: PByte; pLen: Int64); inline;
begin
  if (pLen <= 0) or (GThreadFreq = nil) then
    Exit;
  GThreadFreq^.IncOrInsert(pToken, pLen);
end;

procedure Bufer_isle(pBuf: PByte; pSize: Int64);
var
  pEnd: PByte;
  pWordStart: PByte;
  B, B2: Byte;
  pW: PByte;
begin
  pEnd := pBuf + pSize;
  pWordStart := pBuf;
  pW := pBuf;

  while pBuf < pEnd do
  begin
    B := pBuf^;

    if B <= 127 then
    begin
      if (B >= 97) and (B <= 122) then
      begin
        pW^ := B;
        Inc(pW);
      end
      else if (B >= 65) and (B <= 90) then
      begin
        if B = 73 then
        begin
          pW^ := $C4; Inc(pW);
          pW^ := $B1; Inc(pW);
        end
        else
        begin
          pW^ := B + 32;
          Inc(pW);
        end;
      end
      else if (B >= 48) and (B <= 57) then
      begin
        pW^ := B;
        Inc(pW);
      end
      else
      begin
        if pW > pWordStart then
          Token_Ekle(pWordStart, pW - pWordStart);
        Inc(pBuf);
        pWordStart := pW;
        Continue;
      end;
      Inc(pBuf);
    end
    else
    begin
      if pBuf + 1 >= pEnd then
      begin
        Inc(pBuf);
        Continue;
      end;
      B2 := (pBuf + 1)^;
      case B of
        $C3: case B2 of
               $87: begin pW^ := $C3; Inc(pW); pW^ := $A7; Inc(pW); end;
               $96: begin pW^ := $C3; Inc(pW); pW^ := $B6; Inc(pW); end;
               $9C: begin pW^ := $C3; Inc(pW); pW^ := $BC; Inc(pW); end;
               $A7: begin pW^ := $C3; Inc(pW); pW^ := $A7; Inc(pW); end;
               $B6: begin pW^ := $C3; Inc(pW); pW^ := $B6; Inc(pW); end;
               $BC: begin pW^ := $C3; Inc(pW); pW^ := $BC; Inc(pW); end;
             else
               Inc(pBuf, 2);
               Continue;
             end;
        $C4: case B2 of
               $B0: begin pW^ := $69; Inc(pW); end;
               $B1: begin pW^ := $C4; Inc(pW); pW^ := $B1; Inc(pW); end;
               $9E: begin pW^ := $C4; Inc(pW); pW^ := $9F; Inc(pW); end;
               $9F: begin pW^ := $C4; Inc(pW); pW^ := $9F; Inc(pW); end;
             else
               Inc(pBuf, 2);
               Continue;
             end;
        $C5: case B2 of
               $9E: begin pW^ := $C5; Inc(pW); pW^ := $9F; Inc(pW); end;
               $9F: begin pW^ := $C5; Inc(pW); pW^ := $9F; Inc(pW); end;
             else
               Inc(pBuf, 2);
               Continue;
             end;
      else
        if B >= $E0 then
          Inc(pBuf, 3)
        else
          Inc(pBuf, 2);
        Continue;
      end;
      Inc(pBuf, 2);
    end;
  end;

  if pW > pWordStart then
    Token_Ekle(pWordStart, pW - pWordStart);
end;

procedure Dosya_Oku(const pFileName: string);
var
  lFl: File of Byte;
  lFSize: Int64;
  lBuf: PByte;
begin
  AssignFile(lFl, pFileName);
  Reset(lFl);
  lFSize := FileSize(lFl);
  GetMem(lBuf, lFSize + 1);
  try
    BlockRead(lFl, lBuf^, lFSize);
    Bufer_isle(lBuf, lFSize);
  finally
    FreeMem(lBuf);
  end;
  CloseFile(lFl);
end;

constructor TMyThread.Create(const pFileName: string);
begin
  Inc(ThreadCount);
  inherited Create(False);
  FreeOnTerminate := True;
  FFileName := pFileName;
end;

destructor TMyThread.Destroy;
begin
  Dec(ThreadCount);
  inherited Destroy;
end;

procedure TMyThread.Execute;
begin
  GThreadFreq := @FLocalFreq;
  FLocalFreq.Init;
  try
    Dosya_Oku(FFileName);
    EnterCriticalSection(FreqLock);
    try
      GlobalFreq.MergeFrom(FLocalFreq);
    finally
      LeaveCriticalSection(FreqLock);
    end;
  finally
    FLocalFreq.Done;
    GThreadFreq := nil;
  end;
end;

var
  Sr: TSearchRec;
  Er: Integer;
  T0, T1: QWord;
  OutPath: string;
  SavedVocab: Integer;
begin
  InitCriticalSection(FreqLock);
  GlobalFreq.Init;
  try
    Arguman_Oku;

    Writeln('tokenizer başladı...');
    Writeln('  input-dir      : ', DataDir);
    Writeln('  pattern        : ', FilePattern);
    Writeln('  max-threads    : ', MaxThreads);
    Writeln('  maksimum-token : ', MaksimumToken);
    Writeln('  output-dir     : ', OutputDir);

    T0 := GetTickCount64;
    Er := FindFirst(DataDir + FilePattern, faAnyFile, Sr);
    while Er = 0 do
    begin
      while ThreadCount >= MaxThreads do
        Sleep(1);
      Writeln('Processing: ', Sr.Name);
      TMyThread.Create(DataDir + Sr.Name);
      Er := FindNext(Sr);
    end;
    FindClose(Sr);
    while ThreadCount > 0 do
      Sleep(1);
    T1 := GetTickCount64;

    OutPath := OutputDir + 'kendi_tokenizerim.json';
    SavedVocab := GlobalFreq.SaveTokenizerJson(OutPath, MaksimumToken);

    Writeln;
    Writeln('=== Ozet ===');
    Writeln('  Token sayisi     : ', GlobalFreq.TotalTokens);
    Writeln('  Benzersiz token  : ', GlobalFreq.UniqueCount);
    Writeln('  JSON vocab       : ', SavedVocab);
    Writeln('  JSON dosya       : ', OutPath);
    Writeln('  Sure (sn)        : ', (T1 - T0) / 1000.0:0:2);
    if MaxPrint > 0 then
      GlobalFreq.PrintTop(MaxPrint);

    Writeln('build tokenizer finished.');
  finally
    GlobalFreq.Done;
    DoneCriticalSection(FreqLock);
  end;
end.
