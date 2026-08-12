{$mode objfpc}{$H+}
program set_in_demo;

type
  TDigit = 1..9;
  TSmallSet = set of TDigit;

function IsInSet(X: TDigit; S: TSmallSet): Boolean;
begin
  Result := X in S;
end;

var
  S: TSmallSet;
  B: Boolean;
begin
  S := [1, 3, 5, 7, 9];
  B := IsInSet(5, S);
  if B then
    Halt(0)
  else
    Halt(1);
end.
