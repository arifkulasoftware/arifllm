{$mode objfpc}{$H+}
unit textnorm;

{ Turkce metin normalizasyonu — tek geciste tokenizasyon (hizli yol) }

interface

uses
  Classes,
  Character;

const
  UTF8_CODEPAGE = 65001;

type
  TWordCallback = procedure(const Word: string);
  TSliceCallback = procedure(const Buf: UnicodeString; StartPos, EndPos: Integer);

function NormalizeNFKC(const S: string): string;
function UnicodeCaseFold(const S: string): string;
function NormalizeLine(const S: string; DoCaseFold: Boolean): string;
function IsTokenizerSeparator(C: Char): Boolean; inline;

procedure TokenizeSlice(
  const Text: UnicodeString;
  StartPos, EndPos: Integer;
  DoCaseFold: Boolean;
  DoNormalize: Boolean;
  const OnSlice: TSliceCallback
);

procedure TokenizeLine(
  const Line: string;
  DoCaseFold: Boolean;
  DoNormalize: Boolean;
  const OnSlice: TSliceCallback
); inline;

implementation

type
  TCompatPair = record
    Ch: UnicodeChar;
    Replacement: UnicodeString;
  end;

const
  COMPAT_PAIRS: array[0..23] of TCompatPair = (
    (Ch: #$00A0; Replacement: ' '),
    (Ch: #$2007; Replacement: ' '),
    (Ch: #$202F; Replacement: ' '),
    (Ch: #$2010; Replacement: '-'),
    (Ch: #$2011; Replacement: '-'),
    (Ch: #$2012; Replacement: '-'),
    (Ch: #$2013; Replacement: '-'),
    (Ch: #$2014; Replacement: '-'),
    (Ch: #$2015; Replacement: '-'),
    (Ch: #$2212; Replacement: '-'),
    (Ch: #$2018; Replacement: ''''),
    (Ch: #$2019; Replacement: ''''),
    (Ch: #$201C; Replacement: '"'),
    (Ch: #$201D; Replacement: '"'),
    (Ch: #$00AB; Replacement: '"'),
    (Ch: #$00BB; Replacement: '"'),
    (Ch: #$FB01; Replacement: 'fi'),
    (Ch: #$FB02; Replacement: 'fl'),
    (Ch: #$FB00; Replacement: 'ff'),
    (Ch: #$FB03; Replacement: 'ffi'),
    (Ch: #$FB04; Replacement: 'ffl'),
    (Ch: #$FF01; Replacement: '!'),
    (Ch: #$FF0E; Replacement: '.'),
    (Ch: #$FF1A; Replacement: ':')
  );

var
  CompatLookup: array of UnicodeString;
  CompatReady: Boolean = False;

procedure EnsureCompatLookup;
var
  I, Code: Integer;
begin
  if CompatReady then
    Exit;
  SetLength(CompatLookup, 65536);
  for I := 0 to 65535 do
    CompatLookup[I] := '';
  for I := Low(COMPAT_PAIRS) to High(COMPAT_PAIRS) do
  begin
    Code := Ord(COMPAT_PAIRS[I].Ch);
    if Code < 65536 then
      CompatLookup[Code] := COMPAT_PAIRS[I].Replacement;
  end;
  CompatReady := True;
end;

function IsCombiningMark(C: Char): Boolean; inline;
var
  Cat: TUnicodeCategory;
begin
  Cat := TCharacter.GetUnicodeCategory(C);
  Result := (Ord(Cat) = 5) or (Ord(Cat) = 6);
end;

function IsTokenizerSeparator(C: Char): Boolean; inline;
begin
  if Ord(C) <= $20 then
    Exit(True);
  if C = #$A0 then
    Exit(True);
  if Ord(C) < $80 then
    Exit(False);
  Result := TCharacter.IsWhiteSpace(C);
end;

function MapCompatChar(C: UnicodeChar): UnicodeString; inline;
var
  Code: Integer;
begin
  Code := Ord(C);
  if (Code < 65536) and (CompatLookup[Code] <> '') then
    Exit(CompatLookup[Code]);
  Result := C;
end;

function UnicodeCaseFoldChar(C: UnicodeChar): UnicodeChar; inline;
var
  O: Cardinal;
begin
  O := Ord(C);
  if (O >= Ord('A')) and (O <= Ord('Z')) then
    Exit(UnicodeChar(O + 32));
  case O of
    $0049, $0130: Result := UnicodeChar($0069);
    $0131: Result := UnicodeChar($0131);
  else
    if O < $80 then
      Result := C
    else
      Result := TCharacter.ToLower(C);
  end;
end;

procedure AppendChar(var Buf: UnicodeString; var Len: Integer; C: UnicodeChar);
begin
  Inc(Len);
  if Len > Length(Buf) then
    SetLength(Buf, Len + 256);
  Buf[Len] := C;
end;

procedure BuildWorkBuffer(
  const Text: UnicodeString;
  StartPos, EndPos: Integer;
  DoCaseFold: Boolean;
  DoNormalize: Boolean;
  var Buf: UnicodeString;
  var BufLen: Integer
);
var
  I, J: Integer;
  C: UnicodeChar;
  Mapped: UnicodeString;
begin
  BufLen := 0;
  if EndPos < StartPos then
    Exit;

  SetLength(Buf, (EndPos - StartPos + 1) + 16);

  if DoNormalize then
    EnsureCompatLookup;

  for I := StartPos to EndPos do
  begin
    C := Text[I];
    if DoNormalize then
    begin
      if IsCombiningMark(C) then
        Continue;
      Mapped := MapCompatChar(C);
      for J := 1 to Length(Mapped) do
      begin
        C := Mapped[J];
        if DoCaseFold then
          C := UnicodeCaseFoldChar(C);
        AppendChar(Buf, BufLen, C);
      end;
    end
    else
    begin
      if DoCaseFold then
        C := UnicodeCaseFoldChar(C);
      AppendChar(Buf, BufLen, C);
    end;
  end;
end;

procedure ScanBufferForWords(
  const Buf: UnicodeString;
  BufLen: Integer;
  const OnSlice: TSliceCallback
);
var
  I, StartPos: Integer;
begin
  if BufLen <= 0 then
    Exit;

  StartPos := 1;
  for I := 1 to BufLen do
  begin
    if IsTokenizerSeparator(Buf[I]) then
    begin
      if I > StartPos then
        OnSlice(Buf, StartPos, I - 1);
      StartPos := I + 1;
    end;
  end;
  if StartPos <= BufLen then
    OnSlice(Buf, StartPos, BufLen);
end;

procedure TokenizeSlice(
  const Text: UnicodeString;
  StartPos, EndPos: Integer;
  DoCaseFold: Boolean;
  DoNormalize: Boolean;
  const OnSlice: TSliceCallback
);
var
  Buf: UnicodeString;
  BufLen, I, WordStart: Integer;
begin
  if EndPos < StartPos then
    Exit;

  if (not DoCaseFold) and (not DoNormalize) then
  begin
    WordStart := StartPos;
    for I := StartPos to EndPos do
    begin
      if IsTokenizerSeparator(Text[I]) then
      begin
        if I > WordStart then
          OnSlice(Text, WordStart, I - 1);
        WordStart := I + 1;
      end;
    end;
    if WordStart <= EndPos then
      OnSlice(Text, WordStart, EndPos);
    Exit;
  end;

  BuildWorkBuffer(Text, StartPos, EndPos, DoCaseFold, DoNormalize, Buf, BufLen);
  ScanBufferForWords(Buf, BufLen, OnSlice);
end;

procedure TokenizeLine(
  const Line: string;
  DoCaseFold: Boolean;
  DoNormalize: Boolean;
  const OnSlice: TSliceCallback
);
begin
  if Line = '' then
    Exit;
  TokenizeSlice(Line, 1, Length(Line), DoCaseFold, DoNormalize, OnSlice);
end;

var
  JoinParts: TStringList;

procedure CollectSlice(const Buf: UnicodeString; StartPos, EndPos: Integer);
var
  W: string;
  Len: Integer;
begin
  Len := EndPos - StartPos + 1;
  SetLength(W, Len);
  if Len > 0 then
    Move(Buf[StartPos], W[1], Len * SizeOf(Char));
  JoinParts.Add(W);
end;

function JoinWords(const Line: string; DoCaseFold, DoNormalize: Boolean): string;
var
  I: Integer;
begin
  JoinParts := TStringList.Create;
  try
    TokenizeLine(Line, DoCaseFold, DoNormalize, @CollectSlice);
    Result := '';
    for I := 0 to JoinParts.Count - 1 do
    begin
      if Result <> '' then
        Result := Result + ' ';
      Result := Result + JoinParts[I];
    end;
  finally
    JoinParts.Free;
    JoinParts := nil;
  end;
end;

function NormalizeNFKC(const S: string): string;
begin
  Result := JoinWords(S, False, True);
end;

function UnicodeCaseFold(const S: string): string;
begin
  Result := JoinWords(S, True, False);
end;

function NormalizeLine(const S: string; DoCaseFold: Boolean): string;
begin
  Result := JoinWords(S, DoCaseFold, True);
end;

end.
