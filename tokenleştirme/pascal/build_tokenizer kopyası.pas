{$mode objfpc}{$H+}
{$OPTIMIZATION ON}
{$ASSERTIONS OFF}
program build_tokenizer;

uses
  SysUtils,
  Classes,
  Generics.Collections,
  textnorm;

type
  TWordFreqMap = specialize TDictionary<string, Integer>;

  TStats = record
    FilesProcessed: QWord;
    LinesRead: QWord;
    WordsProcessed: QWord;
    BytesRead: QWord;
    FileBytesRead: QWord;
  end;

const
  FILE_READ_CHUNK = 4 * 1024 * 1024;

var
  FreqMap: TWordFreqMap;
  Stats: TStats;
  DataDir: string;
  FilePattern: string;
  MaxBytes: QWord;
  MaxPrint: Integer;
  UseLowerCase: Boolean;
  UseNormalize: Boolean;
  LineRemainder: string;

procedure OnSliceFound(const Buf: UnicodeString; StartPos, EndPos: Integer);
var
  W: string;
  Len, Count: Integer;
begin
  Len := EndPos - StartPos + 1;
  if Len <= 0 then
    Exit;

  SetLength(W, Len);
  Move(Buf[StartPos], W[1], Len * SizeOf(Char));

  if FreqMap.TryGetValue(W, Count) then
    FreqMap.AddOrSetValue(W, Count + 1)
  else
    FreqMap.Add(W, 1);
  Inc(Stats.WordsProcessed);
end;

procedure ProcessSlice(const Text: UnicodeString; StartPos, EndPos: Integer);
begin
  Inc(Stats.BytesRead, EndPos - StartPos + 1);
  TokenizeSlice(Text, StartPos, EndPos, UseLowerCase, UseNormalize, @OnSliceFound);
end;

procedure ProcessBufferedLines(const Text: string; FinalFlush: Boolean);
var
  I, Start: Integer;
  Ch: Char;
begin
  if Text = '' then
    Exit;

  Start := 1;
  I := 1;
  while I <= Length(Text) do
  begin
    Ch := Text[I];
    if Ch in [#10, #13] then
    begin
      if I > Start then
        ProcessSlice(Text, Start, I - 1);
      Inc(Stats.LinesRead);
      if (Ch = #13) and (I < Length(Text)) and (Text[I + 1] = #10) then
        Inc(I);
      Start := I + 1;
    end;
    Inc(I);
  end;

  if FinalFlush then
  begin
    if Start <= Length(Text) then
    begin
      ProcessSlice(Text, Start, Length(Text));
      Inc(Stats.LinesRead);
    end;
    LineRemainder := '';
  end
  else if Start <= Length(Text) then
    LineRemainder := Copy(Text, Start, Length(Text) - Start + 1)
  else
    LineRemainder := '';
end;

function ProcessFile(const FileName: string): Boolean;
var
  FS: TFileStream;
  Buffer: TBytes;
  BytesRead, ToRead: Integer;
  ChunkText: string;
begin
  Result := False;
  LineRemainder := '';
  FS := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  try
    SetLength(Buffer, FILE_READ_CHUNK);
    repeat
      if (MaxBytes > 0) and (Stats.FileBytesRead >= MaxBytes) then
        Break;

      ToRead := Length(Buffer);
      if (MaxBytes > 0) and (Stats.FileBytesRead + ToRead > MaxBytes) then
        ToRead := Integer(MaxBytes - Stats.FileBytesRead);

      BytesRead := FS.Read(Buffer[0], ToRead);
      if BytesRead <= 0 then
        Break;

      Inc(Stats.FileBytesRead, BytesRead);
      SetLength(ChunkText, BytesRead);
      Move(Buffer[0], ChunkText[1], BytesRead);
      if LineRemainder <> '' then
        ChunkText := LineRemainder + ChunkText;

      ProcessBufferedLines(ChunkText, False);
    until BytesRead < ToRead;

    if (MaxBytes = 0) or (Stats.FileBytesRead < MaxBytes) then
      ProcessBufferedLines(LineRemainder, True);

    Result := True;
  finally
    FS.Free;
  end;
end;

function FreqCompare(List: TStringList; Index1, Index2: Integer): Integer;
var
  V1, V2: PtrInt;
begin
  V1 := PtrInt(List.Objects[Index1]);
  V2 := PtrInt(List.Objects[Index2]);
  if V1 = V2 then
    Result := CompareStr(List[Index1], List[Index2])
  else if V1 > V2 then
    Result := -1
  else
    Result := 1;
end;

procedure PrintTopWords(Count: Integer);
var
  Items: TStringList;
  Key: string;
  Value: Integer;
  I, N: Integer;
begin
  if Count <= 0 then
    Exit;

  Items := TStringList.Create;
  try
    for Key in FreqMap.Keys do
    begin
      Value := FreqMap[Key];
      Items.AddObject(Key, TObject(PtrInt(Value)));
    end;
    Items.CustomSort(@FreqCompare);

    N := Items.Count;
    if Count < N then
      N := Count;

    Writeln('--- En sik ', N, ' kelime ---');
    for I := 0 to N - 1 do
      Writeln(I + 1, #9, Items[I], #9, PtrInt(Items.Objects[I]));
  finally
    Items.Free;
  end;
end;

procedure PrintUsage;
begin
  Writeln('Kullanim: build_tokenizer [secenekler]');
  Writeln('  --input-dir <dizin>     Varsayilan: ./all_txt/');
  Writeln('  --pattern <glob>        Varsayilan: c4*.txt');
  Writeln('  --max-bytes <N>         Corpus okuma limiti');
  Writeln('  --max-print <N>         En sik N kelime (0=kapali)');
  Writeln('  --lowercase             Unicode casefold');
  Writeln('  --no-normalize          NFKC uyumluluk adimini atla (daha hizli)');
  Writeln('  --help                  Bu mesaj');
end;

procedure ParseArgs;
var
  I: Integer;
begin
  DataDir := './all_txt/';
  FilePattern := 'c4*.txt';
  MaxBytes := 0;
  MaxPrint := 0;
  UseLowerCase := False;
  UseNormalize := True;

  I := 1;
  while I <= ParamCount do
  begin
    if ParamStr(I) = '--help' then
    begin
      PrintUsage;
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
    else if ParamStr(I) = '--max-bytes' then
    begin
      Inc(I);
      if I > ParamCount then
        raise Exception.Create('--max-bytes degeri gerekli');
      MaxBytes := StrToQWordDef(ParamStr(I), 0);
    end
    else if ParamStr(I) = '--max-print' then
    begin
      Inc(I);
      if I > ParamCount then
        raise Exception.Create('--max-print degeri gerekli');
      MaxPrint := StrToIntDef(ParamStr(I), 0);
    end
    else if ParamStr(I) = '--lowercase' then
      UseLowerCase := True
    else if ParamStr(I) = '--no-normalize' then
      UseNormalize := False
    else
      raise Exception.Create('Bilinmeyen arguman: ' + ParamStr(I));
    Inc(I);
  end;
end;

var
  Sr: TSearchRec;
  Er: Integer;
  T0, T1: QWord;
begin
  ParseArgs;
  FillChar(Stats, SizeOf(Stats), 0);

  Writeln('build tokenizer started...');
  Writeln('  input-dir : ', DataDir);
  Writeln('  pattern   : ', FilePattern);
  Writeln('  lowercase : ', UseLowerCase);
  Writeln('  normalize : ', UseNormalize);
  Writeln('  io mode   : TFileStream ', FILE_READ_CHUNK div (1024 * 1024), 'MB blok');
  if MaxBytes > 0 then
    Writeln('  max-bytes : ', MaxBytes)
  else
    Writeln('  max-bytes : (sinirsiz)');

  FreqMap := TWordFreqMap.Create;
  try
    T0 := GetTickCount64;
    Er := FindFirst(DataDir + FilePattern, faAnyFile, Sr);
    while Er = 0 do
    begin
      if (MaxBytes > 0) and (Stats.FileBytesRead >= MaxBytes) then
        Break;

      Writeln('Processing: ', Sr.Name);
      ProcessFile(DataDir + Sr.Name);
      Inc(Stats.FilesProcessed);
      Er := FindNext(Sr);
    end;
    FindClose(Sr);
    T1 := GetTickCount64;

    Writeln;
    Writeln('=== Ozet ===');
    Writeln('  Dosya sayisi     : ', Stats.FilesProcessed);
    Writeln('  Satir sayisi     : ', Stats.LinesRead);
    Writeln('  Kelime sayisi    : ', Stats.WordsProcessed);
    Writeln('  Benzersiz kelime : ', FreqMap.Count);
    Writeln('  Dosya byte       : ', Stats.FileBytesRead);
    Writeln('  Metin byte       : ', Stats.BytesRead);
    Writeln('  Sure (sn)        : ', (T1 - T0) / 1000.0:0:2);
    if Stats.FileBytesRead > 0 then
      Writeln('  Hiz (MB/s)       : ', (Stats.FileBytesRead / (1024 * 1024)) / ((T1 - T0) / 1000.0):0:2);

    if MaxPrint > 0 then
      PrintTopWords(MaxPrint);
  finally
    FreqMap.Free;
  end;

  Writeln('build tokenizer finished.');
end.
