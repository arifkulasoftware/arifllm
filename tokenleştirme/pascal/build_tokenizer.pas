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
  cmem, // the c memory manager is on some systems much faster for multi-threading
  cthreads,
{$endif}
  SysUtils,
  Classes;

var
  DataDir : string;
  FilePattern : string;
  MaxThreads : Integer;
  ThreadCount : Integer = 0;

procedure kullanim_talimatlari;
begin
  Writeln('Kullanim: build_tokenizer [secenekler]');
  Writeln('  --input-dir <dizin>     Varsayilan: ./all_txt/');
  Writeln('  --pattern <glob>        Varsayilan: c4*.txt');
  Writeln('  --max-threads <N>       Varsayilan: CPU cekirdek sayisi (', GetCPUCount, ')');
  Writeln('  --help                  Bu mesaj');
end;

procedure Arguman_Oku;
var
  I: Integer;
begin
  DataDir := './all_txt/';
  FilePattern := '*.txt';
  MaxThreads := GetCPUCount;
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
      if MaxThreads < 1 then
        raise Exception.Create('--max-threads en az 1 olmali');
    end
    else
      raise Exception.Create('Bilinmeyen arguman: ' + ParamStr(I));
    Inc(I);
  end;
end;

procedure Token_Ekle(const pToken: PByte; pLen: Int64); inline;
 var S: string;
begin
  if pLen <= 0 then
    Exit;
  SetLength(S, pLen);
  Move(pToken^, S[1], pLen);
  // Writeln(S);
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
      if (B >= 97) and (B <= 122) then          // a-z
      begin
        pW^ := B;
        Inc(pW);
      end
      else if (B >= 65) and (B <= 90) then       // A-Z
      begin
        if B = 73 then                            // I -> ı (C4 B1)
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
      else if (B >= 48) and (B <= 57) then       // 0-9
      begin
        pW^ := B;
        Inc(pW);
      end
      else                                        // bosluk/noktalama -> token siniri
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
      { UTF-8 2-byte: 110xxxxx 10xxxxxx }
      if pBuf + 1 >= pEnd then
      begin
        Inc(pBuf);
        Continue;
      end;
      B2 := (pBuf + 1)^;
      case B of
        $C3: case B2 of
               $87: begin pW^ := $C3; Inc(pW); pW^ := $A7; Inc(pW); end;  // Ç -> ç
               $96: begin pW^ := $C3; Inc(pW); pW^ := $B6; Inc(pW); end;  // Ö -> ö
               $9C: begin pW^ := $C3; Inc(pW); pW^ := $BC; Inc(pW); end;  // Ü -> ü
               $A7: begin pW^ := $C3; Inc(pW); pW^ := $A7; Inc(pW); end;  // ç
               $B6: begin pW^ := $C3; Inc(pW); pW^ := $B6; Inc(pW); end;  // ö
               $BC: begin pW^ := $C3; Inc(pW); pW^ := $BC; Inc(pW); end;  // ü
             else
               Inc(pBuf, 2);
               Continue;
             end;
        $C4: case B2 of
               $B0: begin pW^ := $69; Inc(pW); end;                       // İ -> i
               $B1: begin pW^ := $C4; Inc(pW); pW^ := $B1; Inc(pW); end;  // ı
               $9E: begin pW^ := $C4; Inc(pW); pW^ := $9F; Inc(pW); end;  // Ğ -> ğ
               $9F: begin pW^ := $C4; Inc(pW); pW^ := $9F; Inc(pW); end;  // ğ
             else
               Inc(pBuf, 2);
               Continue;
             end;
        $C5: case B2 of
               $9E: begin pW^ := $C5; Inc(pW); pW^ := $9F; Inc(pW); end;  // Ş -> ş
               $9F: begin pW^ := $C5; Inc(pW); pW^ := $9F; Inc(pW); end;  // ş
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

  Type
    TMyThread = class(TThread)
    private
      FFileName : String ;
    protected
      procedure Execute; override;
    public
      Constructor Create(pFileName : String );
      Destructor Destroy; override;
    end;

  constructor TMyThread.Create(pFileName : String );
  begin
    Inc(ThreadCount);
    inherited Create(false);
    FreeOnTerminate := True;
    FFileName:=pFileName;
  end;

  destructor TMyThread.Destroy;
  begin
    Dec(ThreadCount);
    inherited Destroy;
  end;

procedure TMyThread.Execute;

 begin
  Dosya_Oku(FFileName);
 End;

var
  Sr: TSearchRec;
  Er: Integer;
  T0, T1: QWord;
begin
  Arguman_Oku;

  Writeln('tokenizer başladı...');
  Writeln('  input-dir : ', DataDir);
  Writeln('  pattern   : ', FilePattern);

  T0 := GetTickCount64;
  Er := FindFirst(DataDir + FilePattern, faAnyFile, Sr);
  while Er = 0 do
  begin
    While ThreadCount >= MaxThreads do
      Sleep(1);
    Writeln('ThreadCount: ', ThreadCount);
    Writeln('Processing: ', Sr.Name);    
    TMyThread.Create(DataDir + Sr.Name);
    Er := FindNext(Sr);
  end;
  FindClose(Sr);
  While ThreadCount > 0 do
    Sleep(1);
  T1 := GetTickCount64;

  Writeln;
  Writeln('=== Ozet ===');
  Writeln('  Sure (sn)        : ', (T1 - T0) / 1000.0:0:2);

  Writeln('build tokenizer finished.');
end.
