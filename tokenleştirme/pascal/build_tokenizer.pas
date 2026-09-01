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

type
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
    procedure EnsureArena(Need: Int64);
    procedure GrowBuckets;
    procedure MergeInsert(const pToken: PByte; pLen: Int64; Cnt: Int32);
  public
    procedure Init(InitialBuckets: Integer = INITIAL_BUCKETS);
    procedure Done;
    procedure IncOrInsert(const pToken: PByte; pLen: Int64);
    procedure MergeFrom(const Src: TTokenFreqTable);
    procedure PrintTop(N: Integer);
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
  MaxThreads: Integer;
  MaxPrint: Integer;
  ThreadCount: Integer = 0;
  GlobalFreq: TTokenFreqTable;
  FreqLock: TRTLCriticalSection;
  GThreadFreq: ^TTokenFreqTable = nil;

function TTokenFreqTable.HashBytes(const p: PByte; Len: Int64): QWord;
var
  I: Int64;
  B: Byte;
begin
  Result := FNV_OFFSET;
  for I := 0 to Len - 1 do
  begin
    B := PByte(Int64(p) + I)^;
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
type
  TTopEntry = record
    KeyOff: Int32;
    KeyLen: Int32;
    Count: Int32;
  end;
var
  Items: array of TTopEntry;
  ItemCount, I, J: Integer;
  S: string;
  Tmp: TTopEntry;
begin
  if N <= 0 then
    Exit;

  SetLength(Items, FUsedSlots);
  ItemCount := 0;
  for I := 0 to FBucketCount - 1 do
  begin
    if FBuckets[I].KeyLen = 0 then
      Continue;
    Items[ItemCount].KeyOff := FBuckets[I].KeyOff;
    Items[ItemCount].KeyLen := FBuckets[I].KeyLen;
    Items[ItemCount].Count := FBuckets[I].Count;
    Inc(ItemCount);
  end;

  for I := 0 to ItemCount - 2 do
    for J := I + 1 to ItemCount - 1 do
      if Items[J].Count > Items[I].Count then
      begin
        Tmp := Items[I];
        Items[I] := Items[J];
        Items[J] := Tmp;
      end;

  if N > ItemCount then
    N := ItemCount;

  Writeln('--- En sik ', N, ' token ---');
  for I := 0 to N - 1 do
  begin
    SetLength(S, Items[I].KeyLen);
    if Items[I].KeyLen > 0 then
      Move((FArena + Items[I].KeyOff)^, S[1], Items[I].KeyLen);
    Writeln(I + 1, #9, S, #9, Items[I].Count);
  end;
end;

procedure kullanim_talimatlari;
begin
  Writeln('Kullanim: build_tokenizer [secenekler]');
  Writeln('  --input-dir <dizin>     Varsayilan: ./all_txt/');
  Writeln('  --pattern <glob>        Varsayilan: *.txt');
  Writeln('  --max-threads <N>       Varsayilan: CPU cekirdek sayisi (', GetCPUCount, ')');
  Writeln('  --max-print <N>         En sik N token (0=kapali)');
  Writeln('  --help                  Bu mesaj');
end;

procedure Arguman_Oku;
var
  I: Integer;
begin
  DataDir := './all_txt/';
  FilePattern := '*.txt';
  MaxThreads := GetCPUCount;
  MaxPrint := 0;

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
begin
  InitCriticalSection(FreqLock);
  GlobalFreq.Init;
  try
    Arguman_Oku;

    Writeln('tokenizer başladı...');
    Writeln('  input-dir   : ', DataDir);
    Writeln('  pattern     : ', FilePattern);
    Writeln('  max-threads : ', MaxThreads);

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

    Writeln;
    Writeln('=== Ozet ===');
    Writeln('  Token sayisi     : ', GlobalFreq.TotalTokens);
    Writeln('  Benzersiz token  : ', GlobalFreq.UniqueCount);
    Writeln('  Sure (sn)        : ', (T1 - T0) / 1000.0:0:2);
    if MaxPrint > 0 then
      GlobalFreq.PrintTop(MaxPrint);

    Writeln('build tokenizer finished.');
  finally
    GlobalFreq.Done;
    DoneCriticalSection(FreqLock);
  end;
end.
