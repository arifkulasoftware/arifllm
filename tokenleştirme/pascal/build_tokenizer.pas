{$mode objfpc}{$H+}
{$OPTIMIZATION ON}
{$ASSERTIONS OFF}
program build_tokenizer;

uses
  SysUtils,
  Classes;


var
  DataDir: string;
  FilePattern: string;  
  

procedure kullanim_talimatlari;
begin
  Writeln('Kullanim: build_tokenizer [secenekler]');
  Writeln('  --input-dir <dizin>     Varsayilan: ./all_txt/');
  Writeln('  --pattern <glob>        Varsayilan: c4*.txt');
  Writeln('  --help                  Bu mesaj');
end;

procedure Arguman_Oku;
var
  I: Integer;
begin
  DataDir := './all_txt/';
  FilePattern := 'c4*.txt';

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
    else
      raise Exception.Create('Bilinmeyen arguman: ' + ParamStr(I));
    Inc(I);
  end;
end;

(*
pC in ['.' , '!' , '?' , '-' , '_' , '/' , '(' , ')' , '[' , ']' , '{' , '}' , '<' , '>' , '|' 
            ,'\' , '@' , '#' , '$' , '%' , '^' , '&' , '*' , '+' , '=' , '~' , '`' , '''' , ',' ,
            ';' , ':' , '"']
*)

Function turkcele(var pBuf: PByte; var pPs: Int64):WideChar;
var
  O : Byte ;
  pW : PWord ;
  W : Word ;

begin
  O:=pBuf[pPs];
  Inc(pPs);
  if (O >= 97) and (O <= 122) then // a-z
    Begin Result:=Char(O); Exit; End; 
  if (O >= 65) and (O <= 90) and (O <> 73) then // A-Z, I hariç
    Begin Result:=Char(O+32); Exit; End; 
  if (O >= 48) and (O <= 57) then  
    Begin Result:=Char(O); Exit; End; 
  if O=73 then  
    Begin Result:=WideChar($0131); Exit; End; 
  if O<=127 then  
    Begin Result:=' '; Exit; End; 

  Inc(pPs);  
  pW:=PWord(pBuf);  
  W:=pW[pPs-2];
  Case W of
    $0131, // ı
    $011F, // ğ
    $015F, // ş
    $00E7, // ç
    $00F6, // ö
    $00FC : Begin Result:=WideChar(W); Exit; End;  // return pC
  end;

  case W of
    $0130: Result := 'i';  // İ -> i
    $011E: Result := WideChar($011F);  // Ğ -> ğ
    $015E: Result := WideChar($015F);  // Ş -> ş
    $00C7: Result := WideChar($00E7);  // Ç -> ç
    $00D6: Result := WideChar($00F6);  // Ö -> ö
    $00DC: Result := WideChar($00FC);  // Ü -> ü
  end;

  //While pBuf[pPs]>127 do inc(pPs);

  Result:=' ';
end;

procedure Token_Ekle(pToken: WideString);
begin
  Writeln(pToken);
end;

procedure Bufer_isle(var pBuf: PByte ; pSize: Int64);

var
  lPs : Int64 ;
  lW : WideChar;
  lWord : WideString;

begin
  lWord:='';
  lPs:=0;
  while lPs < pSize do
  begin
    lW:=turkcele(pBuf,lPs);
    if lW<>' ' then
      lWord:=lWord+lW
       else begin      
        Token_Ekle(lWord);
        lWord:='';
      end;
  end;  
end;

procedure Dosya_Oku(const pFileName: string); 
 var
  lFl : File of Byte;
  lFSize,lReadCount : Int64;
  lBuf : PByte ;

begin
  AssignFile(lFl, pFileName);
  Reset(lFl);   
  lFSize:=FileSize(lFl);
  GetMem(lBuf, lFSize);
  try
    BlockRead(lFl, lBuf^, lFSize, lReadCount);
    Bufer_isle(lBuf , lReadCount);  
  finally
    FreeMem(lBuf);
  end;  
  CloseFile(lFl);
end;

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
      Writeln('Processing: ', Sr.Name);
      Dosya_Oku(DataDir + Sr.Name);
      Er := FindNext(Sr);
    end;
    FindClose(Sr);
    T1 := GetTickCount64;

    Writeln;
    Writeln('=== Ozet ===');
    Writeln('  Sure (sn)        : ', (T1 - T0) / 1000.0:0:2);


  Writeln('build tokenizer finished.');
end.
