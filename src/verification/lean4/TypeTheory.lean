inductive TypeTheoryError : Type where
  | TypeMismatch : TypeTheoryError
  | UnificationFailure : TypeTheoryError
  | LinearityViolation : TypeTheoryError
  | InvalidTypeConstruction : TypeTheoryError
  | VariableNotInContext : TypeTheoryError
  | InvalidApplication : TypeTheoryError
  | InvalidProjection : TypeTheoryError
  | CategoryLawViolation : TypeTheoryError
  | OutOfMemory : TypeTheoryError
  | InvalidIdentityElimination : TypeTheoryError
  deriving Repr, BEq, DecidableEq

inductive TypeKind : Type where
  | UNIT : TypeKind
  | BOOL : TypeKind
  | NAT : TypeKind
  | INT : TypeKind
  | REAL : TypeKind
  | COMPLEX : TypeKind
  | STRING : TypeKind
  | ARRAY : TypeKind
  | TUPLE : TypeKind
  | RECORD : TypeKind
  | SUM : TypeKind
  | FUNCTION : TypeKind
  | DEPENDENT_FUNCTION : TypeKind
  | DEPENDENT_PAIR : TypeKind
  | UNIVERSE : TypeKind
  | IDENTITY : TypeKind
  | QUANTUM_TYPE : TypeKind
  | BOTTOM : TypeKind
  | TOP : TypeKind
  | VARIABLE : TypeKind
  | APPLICATION : TypeKind
  deriving Repr, BEq, DecidableEq

def TypeKind.toString (k : TypeKind) : String :=
  match k with
  | UNIT => "Unit"
  | BOOL => "Bool"
  | NAT => "Nat"
  | INT => "Int"
  | REAL => "Real"
  | COMPLEX => "Complex"
  | STRING => "String"
  | ARRAY => "Array"
  | TUPLE => "Tuple"
  | RECORD => "Record"
  | SUM => "Sum"
  | FUNCTION => "Function"
  | DEPENDENT_FUNCTION => "Pi"
  | DEPENDENT_PAIR => "Sigma"
  | UNIVERSE => "Type"
  | IDENTITY => "Id"
  | QUANTUM_TYPE => "Quantum"
  | BOTTOM => "Bottom"
  | TOP => "Top"
  | VARIABLE => "Var"
  | APPLICATION => "App"

def TypeKind.fromString (s : String) : Option TypeKind :=
  if s == "Unit" then some UNIT
  else if s == "Bool" then some BOOL
  else if s == "Nat" then some NAT
  else if s == "Int" then some INT
  else if s == "Real" then some REAL
  else if s == "Complex" then some COMPLEX
  else if s == "String" then some STRING
  else if s == "Array" then some ARRAY
  else if s == "Tuple" then some TUPLE
  else if s == "Record" then some RECORD
  else if s == "Sum" then some SUM
  else if s == "Function" then some FUNCTION
  else if s == "Pi" then some DEPENDENT_FUNCTION
  else if s == "Sigma" then some DEPENDENT_PAIR
  else if s == "Type" then some UNIVERSE
  else if s == "Id" then some IDENTITY
  else if s == "Quantum" then some QUANTUM_TYPE
  else if s == "Bottom" then some BOTTOM
  else if s == "Top" then some TOP
  else if s == "Var" then some VARIABLE
  else if s == "App" then some APPLICATION
  else none

def TypeKind.isBaseType (k : TypeKind) : Bool :=
  match k with
  | UNIT => true
  | BOOL => true
  | NAT => true
  | INT => true
  | REAL => true
  | COMPLEX => true
  | STRING => true
  | BOTTOM => true
  | TOP => true
  | _ => false

def TypeKind.isComposite (k : TypeKind) : Bool :=
  match k with
  | ARRAY => true
  | TUPLE => true
  | RECORD => true
  | SUM => true
  | FUNCTION => true
  | DEPENDENT_FUNCTION => true
  | DEPENDENT_PAIR => true
  | _ => false

def TypeKind.isDependent (k : TypeKind) : Bool :=
  match k with
  | DEPENDENT_FUNCTION => true
  | DEPENDENT_PAIR => true
  | IDENTITY => true
  | _ => false

theorem TypeKind.toString_UNIT : TypeKind.toString TypeKind.UNIT = "Unit" := Eq.refl "Unit"

theorem TypeKind.toString_BOOL : TypeKind.toString TypeKind.BOOL = "Bool" := Eq.refl "Bool"

theorem TypeKind.toString_NAT : TypeKind.toString TypeKind.NAT = "Nat" := Eq.refl "Nat"

theorem TypeKind.toString_INT : TypeKind.toString TypeKind.INT = "Int" := Eq.refl "Int"

theorem TypeKind.toString_REAL : TypeKind.toString TypeKind.REAL = "Real" := Eq.refl "Real"

theorem TypeKind.toString_COMPLEX : TypeKind.toString TypeKind.COMPLEX = "Complex" := Eq.refl "Complex"

theorem TypeKind.toString_STRING : TypeKind.toString TypeKind.STRING = "String" := Eq.refl "String"

theorem TypeKind.toString_ARRAY : TypeKind.toString TypeKind.ARRAY = "Array" := Eq.refl "Array"

theorem TypeKind.toString_TUPLE : TypeKind.toString TypeKind.TUPLE = "Tuple" := Eq.refl "Tuple"

theorem TypeKind.toString_RECORD : TypeKind.toString TypeKind.RECORD = "Record" := Eq.refl "Record"

theorem TypeKind.toString_SUM : TypeKind.toString TypeKind.SUM = "Sum" := Eq.refl "Sum"

theorem TypeKind.toString_FUNCTION : TypeKind.toString TypeKind.FUNCTION = "Function" := Eq.refl "Function"

theorem TypeKind.toString_DEPENDENT_FUNCTION : TypeKind.toString TypeKind.DEPENDENT_FUNCTION = "Pi" := Eq.refl "Pi"

theorem TypeKind.toString_DEPENDENT_PAIR : TypeKind.toString TypeKind.DEPENDENT_PAIR = "Sigma" := Eq.refl "Sigma"

theorem TypeKind.toString_UNIVERSE : TypeKind.toString TypeKind.UNIVERSE = "Type" := Eq.refl "Type"

theorem TypeKind.toString_IDENTITY : TypeKind.toString TypeKind.IDENTITY = "Id" := Eq.refl "Id"

theorem TypeKind.toString_QUANTUM_TYPE : TypeKind.toString TypeKind.QUANTUM_TYPE = "Quantum" := Eq.refl "Quantum"

theorem TypeKind.toString_BOTTOM : TypeKind.toString TypeKind.BOTTOM = "Bottom" := Eq.refl "Bottom"

theorem TypeKind.toString_TOP : TypeKind.toString TypeKind.TOP = "Top" := Eq.refl "Top"

theorem TypeKind.toString_VARIABLE : TypeKind.toString TypeKind.VARIABLE = "Var" := Eq.refl "Var"

theorem TypeKind.toString_APPLICATION : TypeKind.toString TypeKind.APPLICATION = "App" := Eq.refl "App"

theorem TypeKind.fromString_Unit : TypeKind.fromString "Unit" = some TypeKind.UNIT := Eq.refl (some TypeKind.UNIT)

theorem TypeKind.fromString_Bool : TypeKind.fromString "Bool" = some TypeKind.BOOL := Eq.refl (some TypeKind.BOOL)

theorem TypeKind.fromString_Nat : TypeKind.fromString "Nat" = some TypeKind.NAT := Eq.refl (some TypeKind.NAT)

theorem TypeKind.fromString_Int : TypeKind.fromString "Int" = some TypeKind.INT := Eq.refl (some TypeKind.INT)

theorem TypeKind.fromString_Real : TypeKind.fromString "Real" = some TypeKind.REAL := Eq.refl (some TypeKind.REAL)

theorem TypeKind.fromString_Complex : TypeKind.fromString "Complex" = some TypeKind.COMPLEX := Eq.refl (some TypeKind.COMPLEX)

theorem TypeKind.fromString_String : TypeKind.fromString "String" = some TypeKind.STRING := Eq.refl (some TypeKind.STRING)

theorem TypeKind.fromString_Array : TypeKind.fromString "Array" = some TypeKind.ARRAY := Eq.refl (some TypeKind.ARRAY)

theorem TypeKind.fromString_Tuple : TypeKind.fromString "Tuple" = some TypeKind.TUPLE := Eq.refl (some TypeKind.TUPLE)

theorem TypeKind.fromString_Record : TypeKind.fromString "Record" = some TypeKind.RECORD := Eq.refl (some TypeKind.RECORD)

theorem TypeKind.fromString_Sum : TypeKind.fromString "Sum" = some TypeKind.SUM := Eq.refl (some TypeKind.SUM)

theorem TypeKind.fromString_Function : TypeKind.fromString "Function" = some TypeKind.FUNCTION := Eq.refl (some TypeKind.FUNCTION)

theorem TypeKind.fromString_Pi : TypeKind.fromString "Pi" = some TypeKind.DEPENDENT_FUNCTION := Eq.refl (some TypeKind.DEPENDENT_FUNCTION)

theorem TypeKind.fromString_Sigma : TypeKind.fromString "Sigma" = some TypeKind.DEPENDENT_PAIR := Eq.refl (some TypeKind.DEPENDENT_PAIR)

theorem TypeKind.fromString_Type : TypeKind.fromString "Type" = some TypeKind.UNIVERSE := Eq.refl (some TypeKind.UNIVERSE)

theorem TypeKind.fromString_Id : TypeKind.fromString "Id" = some TypeKind.IDENTITY := Eq.refl (some TypeKind.IDENTITY)

theorem TypeKind.fromString_Quantum : TypeKind.fromString "Quantum" = some TypeKind.QUANTUM_TYPE := Eq.refl (some TypeKind.QUANTUM_TYPE)

theorem TypeKind.fromString_Bottom : TypeKind.fromString "Bottom" = some TypeKind.BOTTOM := Eq.refl (some TypeKind.BOTTOM)

theorem TypeKind.fromString_Top : TypeKind.fromString "Top" = some TypeKind.TOP := Eq.refl (some TypeKind.TOP)

theorem TypeKind.fromString_Var : TypeKind.fromString "Var" = some TypeKind.VARIABLE := Eq.refl (some TypeKind.VARIABLE)

theorem TypeKind.fromString_App : TypeKind.fromString "App" = some TypeKind.APPLICATION := Eq.refl (some TypeKind.APPLICATION)

theorem TypeKind.fromString_invalid : TypeKind.fromString "Invalid" = none := Eq.refl none

theorem TypeKind.isBaseType_UNIT : TypeKind.isBaseType TypeKind.UNIT = true := Eq.refl true

theorem TypeKind.isBaseType_BOOL : TypeKind.isBaseType TypeKind.BOOL = true := Eq.refl true

theorem TypeKind.isBaseType_NAT : TypeKind.isBaseType TypeKind.NAT = true := Eq.refl true

theorem TypeKind.isBaseType_INT : TypeKind.isBaseType TypeKind.INT = true := Eq.refl true

theorem TypeKind.isBaseType_REAL : TypeKind.isBaseType TypeKind.REAL = true := Eq.refl true

theorem TypeKind.isBaseType_COMPLEX : TypeKind.isBaseType TypeKind.COMPLEX = true := Eq.refl true

theorem TypeKind.isBaseType_STRING : TypeKind.isBaseType TypeKind.STRING = true := Eq.refl true

theorem TypeKind.isBaseType_BOTTOM : TypeKind.isBaseType TypeKind.BOTTOM = true := Eq.refl true

theorem TypeKind.isBaseType_TOP : TypeKind.isBaseType TypeKind.TOP = true := Eq.refl true

theorem TypeKind.isBaseType_ARRAY : TypeKind.isBaseType TypeKind.ARRAY = false := Eq.refl false

theorem TypeKind.isBaseType_TUPLE : TypeKind.isBaseType TypeKind.TUPLE = false := Eq.refl false

theorem TypeKind.isBaseType_RECORD : TypeKind.isBaseType TypeKind.RECORD = false := Eq.refl false

theorem TypeKind.isBaseType_SUM : TypeKind.isBaseType TypeKind.SUM = false := Eq.refl false

theorem TypeKind.isBaseType_FUNCTION : TypeKind.isBaseType TypeKind.FUNCTION = false := Eq.refl false

theorem TypeKind.isBaseType_DEPENDENT_FUNCTION : TypeKind.isBaseType TypeKind.DEPENDENT_FUNCTION = false := Eq.refl false

theorem TypeKind.isBaseType_DEPENDENT_PAIR : TypeKind.isBaseType TypeKind.DEPENDENT_PAIR = false := Eq.refl false

theorem TypeKind.isBaseType_UNIVERSE : TypeKind.isBaseType TypeKind.UNIVERSE = false := Eq.refl false

theorem TypeKind.isBaseType_IDENTITY : TypeKind.isBaseType TypeKind.IDENTITY = false := Eq.refl false

theorem TypeKind.isBaseType_QUANTUM_TYPE : TypeKind.isBaseType TypeKind.QUANTUM_TYPE = false := Eq.refl false

theorem TypeKind.isBaseType_VARIABLE : TypeKind.isBaseType TypeKind.VARIABLE = false := Eq.refl false

theorem TypeKind.isBaseType_APPLICATION : TypeKind.isBaseType TypeKind.APPLICATION = false := Eq.refl false

theorem TypeKind.isComposite_ARRAY : TypeKind.isComposite TypeKind.ARRAY = true := Eq.refl true

theorem TypeKind.isComposite_TUPLE : TypeKind.isComposite TypeKind.TUPLE = true := Eq.refl true

theorem TypeKind.isComposite_RECORD : TypeKind.isComposite TypeKind.RECORD = true := Eq.refl true

theorem TypeKind.isComposite_SUM : TypeKind.isComposite TypeKind.SUM = true := Eq.refl true

theorem TypeKind.isComposite_FUNCTION : TypeKind.isComposite TypeKind.FUNCTION = true := Eq.refl true

theorem TypeKind.isComposite_DEPENDENT_FUNCTION : TypeKind.isComposite TypeKind.DEPENDENT_FUNCTION = true := Eq.refl true

theorem TypeKind.isComposite_DEPENDENT_PAIR : TypeKind.isComposite TypeKind.DEPENDENT_PAIR = true := Eq.refl true

theorem TypeKind.isComposite_UNIT : TypeKind.isComposite TypeKind.UNIT = false := Eq.refl false

theorem TypeKind.isComposite_BOOL : TypeKind.isComposite TypeKind.BOOL = false := Eq.refl false

theorem TypeKind.isComposite_NAT : TypeKind.isComposite TypeKind.NAT = false := Eq.refl false

theorem TypeKind.isComposite_INT : TypeKind.isComposite TypeKind.INT = false := Eq.refl false

theorem TypeKind.isComposite_REAL : TypeKind.isComposite TypeKind.REAL = false := Eq.refl false

theorem TypeKind.isComposite_COMPLEX : TypeKind.isComposite TypeKind.COMPLEX = false := Eq.refl false

theorem TypeKind.isComposite_STRING : TypeKind.isComposite TypeKind.STRING = false := Eq.refl false

theorem TypeKind.isComposite_UNIVERSE : TypeKind.isComposite TypeKind.UNIVERSE = false := Eq.refl false

theorem TypeKind.isComposite_IDENTITY : TypeKind.isComposite TypeKind.IDENTITY = false := Eq.refl false

theorem TypeKind.isComposite_QUANTUM_TYPE : TypeKind.isComposite TypeKind.QUANTUM_TYPE = false := Eq.refl false

theorem TypeKind.isComposite_BOTTOM : TypeKind.isComposite TypeKind.BOTTOM = false := Eq.refl false

theorem TypeKind.isComposite_TOP : TypeKind.isComposite TypeKind.TOP = false := Eq.refl false

theorem TypeKind.isComposite_VARIABLE : TypeKind.isComposite TypeKind.VARIABLE = false := Eq.refl false

theorem TypeKind.isComposite_APPLICATION : TypeKind.isComposite TypeKind.APPLICATION = false := Eq.refl false

theorem TypeKind.isDependent_DEPENDENT_FUNCTION : TypeKind.isDependent TypeKind.DEPENDENT_FUNCTION = true := Eq.refl true

theorem TypeKind.isDependent_DEPENDENT_PAIR : TypeKind.isDependent TypeKind.DEPENDENT_PAIR = true := Eq.refl true

theorem TypeKind.isDependent_IDENTITY : TypeKind.isDependent TypeKind.IDENTITY = true := Eq.refl true

theorem TypeKind.isDependent_UNIT : TypeKind.isDependent TypeKind.UNIT = false := Eq.refl false

theorem TypeKind.isDependent_BOOL : TypeKind.isDependent TypeKind.BOOL = false := Eq.refl false

theorem TypeKind.isDependent_NAT : TypeKind.isDependent TypeKind.NAT = false := Eq.refl false

theorem TypeKind.isDependent_INT : TypeKind.isDependent TypeKind.INT = false := Eq.refl false

theorem TypeKind.isDependent_REAL : TypeKind.isDependent TypeKind.REAL = false := Eq.refl false

theorem TypeKind.isDependent_COMPLEX : TypeKind.isDependent TypeKind.COMPLEX = false := Eq.refl false

theorem TypeKind.isDependent_STRING : TypeKind.isDependent TypeKind.STRING = false := Eq.refl false

theorem TypeKind.isDependent_ARRAY : TypeKind.isDependent TypeKind.ARRAY = false := Eq.refl false

theorem TypeKind.isDependent_TUPLE : TypeKind.isDependent TypeKind.TUPLE = false := Eq.refl false

theorem TypeKind.isDependent_RECORD : TypeKind.isDependent TypeKind.RECORD = false := Eq.refl false

theorem TypeKind.isDependent_SUM : TypeKind.isDependent TypeKind.SUM = false := Eq.refl false

theorem TypeKind.isDependent_FUNCTION : TypeKind.isDependent TypeKind.FUNCTION = false := Eq.refl false

theorem TypeKind.isDependent_UNIVERSE : TypeKind.isDependent TypeKind.UNIVERSE = false := Eq.refl false

theorem TypeKind.isDependent_QUANTUM_TYPE : TypeKind.isDependent TypeKind.QUANTUM_TYPE = false := Eq.refl false

theorem TypeKind.isDependent_BOTTOM : TypeKind.isDependent TypeKind.BOTTOM = false := Eq.refl false

theorem TypeKind.isDependent_TOP : TypeKind.isDependent TypeKind.TOP = false := Eq.refl false

theorem TypeKind.isDependent_VARIABLE : TypeKind.isDependent TypeKind.VARIABLE = false := Eq.refl false

theorem TypeKind.isDependent_APPLICATION : TypeKind.isDependent TypeKind.APPLICATION = false := Eq.refl false

mutual
inductive RecordField : Type where
  | mk : String → Ty → RecordField
with
inductive Ty : Type where
  | mk : TypeKind → String → List Ty → List RecordField → Nat → Option String → Option Ty → Option Ty → Option Ty → Option Nat → Ty
end

def RecordField.name : RecordField → String
  | mk n _ => n

def RecordField.fieldType : RecordField → Ty
  | mk _ t => t

def Ty.kind : Ty → TypeKind
  | mk k _ _ _ _ _ _ _ _ _ => k

def Ty.name : Ty → String
  | mk _ n _ _ _ _ _ _ _ _ => n

def Ty.parameters : Ty → List Ty
  | mk _ _ ps _ _ _ _ _ _ _ => ps

def Ty.fields : Ty → List RecordField
  | mk _ _ _ fs _ _ _ _ _ _ => fs

def Ty.universeLevel : Ty → Nat
  | mk _ _ _ _ l _ _ _ _ _ => l

def Ty.boundVariable : Ty → Option String
  | mk _ _ _ _ _ bv _ _ _ _ => bv

def Ty.leftType : Ty → Option Ty
  | mk _ _ _ _ _ _ lt _ _ _ => lt

def Ty.rightType : Ty → Option Ty
  | mk _ _ _ _ _ _ _ rt _ _ => rt

def Ty.bodyType : Ty → Option Ty
  | mk _ _ _ _ _ _ _ _ bt _ => bt

def Ty.quantumDimension : Ty → Option Nat
  | mk _ _ _ _ _ _ _ _ _ qd => qd

def Ty.init (k : TypeKind) : Ty := Ty.mk k "" [] [] 0 none none none none none

def Ty.initUnit : Ty := Ty.init TypeKind.UNIT

def Ty.initBool : Ty := Ty.init TypeKind.BOOL

def Ty.initNat : Ty := Ty.init TypeKind.NAT

def Ty.initInt : Ty := Ty.init TypeKind.INT

def Ty.initReal : Ty := Ty.init TypeKind.REAL

def Ty.initComplex : Ty := Ty.init TypeKind.COMPLEX

def Ty.initString : Ty := Ty.init TypeKind.STRING

def Ty.initBottom : Ty := Ty.init TypeKind.BOTTOM

def Ty.initTop : Ty := Ty.init TypeKind.TOP

def Ty.initVariable (n : String) : Ty := Ty.mk TypeKind.VARIABLE n [] [] 0 none none none none none

def Ty.initArray (elem : Ty) : Ty := Ty.mk TypeKind.ARRAY "" [elem] [] 0 none none none none none

def Ty.initTuple (types : List Ty) : Ty := Ty.mk TypeKind.TUPLE "" types [] 0 none none none none none

def Ty.initSum (left : Ty) (right : Ty) : Ty := Ty.mk TypeKind.SUM "" [] [] 0 none none (some left) (some right) none

def Ty.initFunction (dom : Ty) (cod : Ty) : Ty := Ty.mk TypeKind.FUNCTION "" [] [] 0 none none (some dom) (some cod) none

def Ty.initUniverse (lvl : Nat) : Ty := Ty.mk TypeKind.UNIVERSE "" [] [] lvl none none none none none

def Ty.initQuantum (base : Ty) (dim : Nat) : Ty := Ty.mk TypeKind.QUANTUM_TYPE "" [base] [] 0 none none none none (some dim)

def Ty.initApplication (func : Ty) (arg : Ty) : Ty := Ty.mk TypeKind.APPLICATION "" [] [] 0 none none (some func) (some arg) none

theorem Ty.kind_initUnit : Ty.initUnit.kind = TypeKind.UNIT := Eq.refl TypeKind.UNIT

theorem Ty.kind_initBool : Ty.initBool.kind = TypeKind.BOOL := Eq.refl TypeKind.BOOL

theorem Ty.kind_initNat : Ty.initNat.kind = TypeKind.NAT := Eq.refl TypeKind.NAT

theorem Ty.kind_initInt : Ty.initInt.kind = TypeKind.INT := Eq.refl TypeKind.INT

theorem Ty.kind_initReal : Ty.initReal.kind = TypeKind.REAL := Eq.refl TypeKind.REAL

theorem Ty.kind_initComplex : Ty.initComplex.kind = TypeKind.COMPLEX := Eq.refl TypeKind.COMPLEX

theorem Ty.kind_initString : Ty.initString.kind = TypeKind.STRING := Eq.refl TypeKind.STRING

theorem Ty.kind_initBottom : Ty.initBottom.kind = TypeKind.BOTTOM := Eq.refl TypeKind.BOTTOM

theorem Ty.kind_initTop : Ty.initTop.kind = TypeKind.TOP := Eq.refl TypeKind.TOP

theorem Ty.kind_initVariable (n : String) : (Ty.initVariable n).kind = TypeKind.VARIABLE := Eq.refl TypeKind.VARIABLE

theorem Ty.kind_initArray (e : Ty) : (Ty.initArray e).kind = TypeKind.ARRAY := Eq.refl TypeKind.ARRAY

theorem Ty.kind_initTuple (ts : List Ty) : (Ty.initTuple ts).kind = TypeKind.TUPLE := Eq.refl TypeKind.TUPLE

theorem Ty.kind_initSum (l r : Ty) : (Ty.initSum l r).kind = TypeKind.SUM := Eq.refl TypeKind.SUM

theorem Ty.kind_initFunction (d c : Ty) : (Ty.initFunction d c).kind = TypeKind.FUNCTION := Eq.refl TypeKind.FUNCTION

theorem Ty.kind_initUniverse (lvl : Nat) : (Ty.initUniverse lvl).kind = TypeKind.UNIVERSE := Eq.refl TypeKind.UNIVERSE

theorem Ty.kind_initQuantum (b : Ty) (d : Nat) : (Ty.initQuantum b d).kind = TypeKind.QUANTUM_TYPE := Eq.refl TypeKind.QUANTUM_TYPE

theorem Ty.kind_initApplication (f a : Ty) : (Ty.initApplication f a).kind = TypeKind.APPLICATION := Eq.refl TypeKind.APPLICATION

mutual
def RecordField.equals : RecordField → RecordField → Bool
  | mk n1 t1, mk n2 t2 => n1 == n2 && Ty.equals t1 t2

def Ty.equalsList : List Ty → List Ty → Bool
  | [], [] => true
  | [], _ :: _ => false
  | _ :: _, [] => false
  | t1 :: ts1, t2 :: ts2 => Ty.equals t1 t2 && Ty.equalsList ts1 ts2

def RecordField.equalsList : List RecordField → List RecordField → Bool
  | [], [] => true
  | [], _ :: _ => false
  | _ :: _, [] => false
  | f1 :: fs1, f2 :: fs2 => RecordField.equals f1 f2 && RecordField.equalsList fs1 fs2

def Ty.optEquals : Option Ty → Option Ty → Bool
  | none, none => true
  | some t1, some t2 => Ty.equals t1 t2
  | _, _ => false

def Ty.equals : Ty → Ty → Bool
  | mk k1 n1 ps1 fs1 l1 bv1 lt1 rt1 bt1 qd1, mk k2 n2 ps2 fs2 l2 bv2 lt2 rt2 bt2 qd2 =>
    k1 == k2 &&
    l1 == l2 &&
    n1 == n2 &&
    Ty.equalsList ps1 ps2 &&
    RecordField.equalsList fs1 fs2 &&
    (match bv1, bv2 with
     | none, none => true
     | some s1, some s2 => s1 == s2
     | _, _ => false) &&
    Ty.optEquals lt1 lt2 &&
    Ty.optEquals rt1 rt2 &&
    Ty.optEquals bt1 bt2 &&
    (match qd1, qd2 with
     | none, none => true
     | some d1, some d2 => d1 == d2
     | _, _ => false)
end

theorem Ty.equals_refl (t : Ty) : Ty.equals t t = true :=
  Ty.recOn t (fun k n ps fs l bv lt rt bt qd =>
    let ps_eq : Ty.equalsList ps ps = true :=
      List.recOn ps (Eq.refl true) (fun p psTail ih =>
        And.intro (Ty.equals_refl p) ih |> fun h =>
          Eq.trans (Eq.refl (Ty.equals p p && Ty.equalsList psTail psTail)) (and_self_iff.mpr h).symm |> Eq.trans (and_self _)
      )
    let fs_eq : RecordField.equalsList fs fs = true :=
      List.recOn fs (Eq.refl true) (fun f fsTail ih =>
        let f_eq := match f with | RecordField.mk fn ft =>
          And.intro (Eq.refl (fn == fn)) (Ty.equals_refl ft) |> fun h =>
            Eq.trans (Eq.refl (fn == fn && Ty.equals ft ft)) (and_self_iff.mpr h).symm |> Eq.trans (and_self _)
        And.intro f_eq ih |> fun h =>
          Eq.trans (Eq.refl (RecordField.equals f f && RecordField.equalsList fsTail fsTail)) (and_self_iff.mpr h).symm |> Eq.trans (and_self _)
      )
    let opt_eq := Ty.optEquals
    let lt_eq := match lt with
      | none => Eq.refl true
      | some t' => Ty.equals_refl t'
    let rt_eq := match rt with
      | none => Eq.refl true
      | some t' => Ty.equals_refl t'
    let bt_eq := match bt with
      | none => Eq.refl true
      | some t' => Ty.equals_refl t'
    let bv_eq := match bv with
      | none => Eq.refl true
      | some s => Eq.refl (s == s) |> Eq.trans (beq_self_eq_true s)
    let qd_eq := match qd with
      | none => Eq.refl true
      | some d => Eq.refl (d == d) |> Eq.trans (beq_self_eq_true d)
    let kind_eq := Eq.refl (k == k) |> Eq.trans (beq_self_eq_true k)
    let level_eq := Eq.refl (l == l) |> Eq.trans (beq_self_eq_true l)
    let name_eq := Eq.refl (n == n) |> Eq.trans (beq_self_eq_true n)
    let all_eq := And.intro kind_eq (And.intro level_eq (And.intro name_eq (And.intro ps_eq (And.intro fs_eq (And.intro bv_eq (And.intro lt_eq (And.intro rt_eq (And.intro bt_eq qd_eq))))))))
    Eq.trans (Eq.refl (k == k && l == l && n == n && Ty.equalsList ps ps && RecordField.equalsList fs fs && (match bv, bv with | none, none => true | some s1, some s2 => s1 == s2 | _, _ => false) && Ty.optEquals lt lt && Ty.optEquals rt rt && Ty.optEquals bt bt && (match qd, qd with | none, none => true | some d1, some d2 => d1 == d2 | _, _ => false))) (and_self_iff.mpr all_eq).symm |> Eq.trans (and_self _)
  )

theorem Ty.equals_initUnit : Ty.equals Ty.initUnit Ty.initUnit = true := Ty.equals_refl Ty.initUnit

theorem Ty.equals_initBool : Ty.equals Ty.initBool Ty.initBool = true := Ty.equals_refl Ty.initBool

theorem Ty.equals_initNat : Ty.equals Ty.initNat Ty.initNat = true := Ty.equals_refl Ty.initNat

theorem Ty.equals_initInt : Ty.equals Ty.initInt Ty.initInt = true := Ty.equals_refl Ty.initInt

theorem Ty.equals_initReal : Ty.equals Ty.initReal Ty.initReal = true := Ty.equals_refl Ty.initReal

theorem Ty.equals_initComplex : Ty.equals Ty.initComplex Ty.initComplex = true := Ty.equals_refl Ty.initComplex

theorem Ty.equals_initString : Ty.equals Ty.initString Ty.initString = true := Ty.equals_refl Ty.initString

theorem Ty.equals_initBottom : Ty.equals Ty.initBottom Ty.initBottom = true := Ty.equals_refl Ty.initBottom

theorem Ty.equals_initTop : Ty.equals Ty.initTop Ty.initTop = true := Ty.equals_refl Ty.initTop

theorem Ty.equals_initUniverse_zero : Ty.equals (Ty.initUniverse 0) (Ty.initUniverse 0) = true := Ty.equals_refl (Ty.initUniverse 0)

theorem Ty.equals_initUniverse_one : Ty.equals (Ty.initUniverse 1) (Ty.initUniverse 1) = true := Ty.equals_refl (Ty.initUniverse 1)

theorem Ty.not_equals_Nat_Bool : Ty.equals Ty.initNat Ty.initBool = false := Eq.refl false

theorem Ty.not_equals_Nat_Int : Ty.equals Ty.initNat Ty.initInt = false := Eq.refl false

theorem Ty.not_equals_Nat_String : Ty.equals Ty.initNat Ty.initString = false := Eq.refl false

theorem Ty.not_equals_Bool_Nat : Ty.equals Ty.initBool Ty.initNat = false := Eq.refl false

theorem Ty.not_equals_Unit_Bottom : Ty.equals Ty.initUnit Ty.initBottom = false := Eq.refl false

theorem Ty.not_equals_Top_Bottom : Ty.equals Ty.initTop Ty.initBottom = false := Eq.refl false

mutual
def RecordField.clone : RecordField → RecordField
  | mk n t => mk n (Ty.clone t)

def Ty.cloneList : List Ty → List Ty
  | [] => []
  | t :: ts => Ty.clone t :: Ty.cloneList ts

def RecordField.cloneList : List RecordField → List RecordField
  | [] => []
  | f :: fs => RecordField.clone f :: RecordField.cloneList fs

def Ty.optClone : Option Ty → Option Ty
  | none => none
  | some t => some (Ty.clone t)

def Ty.clone : Ty → Ty
  | mk k n ps fs l bv lt rt bt qd =>
    mk k n (Ty.cloneList ps) (RecordField.cloneList fs) l bv (Ty.optClone lt) (Ty.optClone rt) (Ty.optClone bt) qd
end

theorem Ty.clone_kind (t : Ty) : (Ty.clone t).kind = t.kind :=
  Ty.recOn t (fun k n ps fs l bv lt rt bt qd => Eq.refl k)

theorem Ty.clone_initNat : (Ty.clone Ty.initNat).kind = TypeKind.NAT := Ty.clone_kind Ty.initNat

theorem Ty.clone_initBool : (Ty.clone Ty.initBool).kind = TypeKind.BOOL := Ty.clone_kind Ty.initBool

theorem Ty.clone_initUnit : (Ty.clone Ty.initUnit).kind = TypeKind.UNIT := Ty.clone_kind Ty.initUnit

def Ty.getDomain (t : Ty) : Option Ty :=
  match t.kind with
  | TypeKind.FUNCTION => t.leftType
  | TypeKind.DEPENDENT_FUNCTION => t.leftType
  | _ => none

def Ty.getCodomain (t : Ty) : Option Ty :=
  match t.kind with
  | TypeKind.FUNCTION => t.rightType
  | TypeKind.DEPENDENT_FUNCTION => t.bodyType
  | _ => none

def Ty.getElementType (t : Ty) : Option Ty :=
  match t.kind with
  | TypeKind.ARRAY =>
    match t.parameters with
    | [] => none
    | e :: _ => some e
  | _ => none

theorem Ty.getDomain_initFunction (d c : Ty) : Ty.getDomain (Ty.initFunction d c) = some d := Eq.refl (some d)

theorem Ty.getCodomain_initFunction (d c : Ty) : Ty.getCodomain (Ty.initFunction d c) = some c := Eq.refl (some c)

theorem Ty.getDomain_initNat : Ty.getDomain Ty.initNat = none := Eq.refl none

theorem Ty.getCodomain_initNat : Ty.getCodomain Ty.initNat = none := Eq.refl none

theorem Ty.getElementType_initArray (e : Ty) : Ty.getElementType (Ty.initArray e) = some e := Eq.refl (some e)

theorem Ty.getElementType_initNat : Ty.getElementType Ty.initNat = none := Eq.refl none

def Ty.getUniverseLevel (t : Ty) : Nat :=
  match t.kind with
  | TypeKind.UNIVERSE => t.universeLevel
  | TypeKind.DEPENDENT_FUNCTION =>
    let left := match t.leftType with
      | none => 0
      | some l => l.getUniverseLevel
    let body := match t.bodyType with
      | none => 0
      | some b => b.getUniverseLevel
    max left body
  | TypeKind.DEPENDENT_PAIR =>
    let left := match t.leftType with
      | none => 0
      | some l => l.getUniverseLevel
    let body := match t.bodyType with
      | none => 0
      | some b => b.getUniverseLevel
    max left body
  | _ => 0

theorem Ty.getUniverseLevel_initUniverse (l : Nat) : Ty.getUniverseLevel (Ty.initUniverse l) = l := Eq.refl l

theorem Ty.getUniverseLevel_initNat : Ty.getUniverseLevel Ty.initNat = 0 := Eq.refl 0

theorem Ty.getUniverseLevel_initBool : Ty.getUniverseLevel Ty.initBool = 0 := Eq.refl 0

theorem Ty.getUniverseLevel_initUnit : Ty.getUniverseLevel Ty.initUnit = 0 := Eq.refl 0

theorem Ty.getUniverseLevel_initInt : Ty.getUniverseLevel Ty.initInt = 0 := Eq.refl 0

theorem Ty.getUniverseLevel_initReal : Ty.getUniverseLevel Ty.initReal = 0 := Eq.refl 0

theorem Ty.getUniverseLevel_initString : Ty.getUniverseLevel Ty.initString = 0 := Eq.refl 0

theorem Ty.getUniverseLevel_initBottom : Ty.getUniverseLevel Ty.initBottom = 0 := Eq.refl 0

theorem Ty.getUniverseLevel_initTop : Ty.getUniverseLevel Ty.initTop = 0 := Eq.refl 0

mutual
def RecordField.substitute : RecordField → String → Ty → RecordField
  | mk n t, vn, r => mk n (Ty.substitute t vn r)

def Ty.substituteList : List Ty → String → Ty → List Ty
  | [], _, _ => []
  | t :: ts, vn, r => Ty.substitute t vn r :: Ty.substituteList ts vn r

def RecordField.substituteList : List RecordField → String → Ty → List RecordField
  | [], _, _ => []
  | f :: fs, vn, r => RecordField.substitute f vn r :: RecordField.substituteList fs vn r

def Ty.optSubstitute : Option Ty → String → Ty → Option Ty
  | none, _, _ => none
  | some t, vn, r => some (Ty.substitute t vn r)

def Ty.substitute : Ty → String → Ty → Ty
  | mk k n ps fs l bv lt rt bt qd, vn, r =>
    if k == TypeKind.VARIABLE && n == vn then r
    else mk k n (Ty.substituteList ps vn r) (RecordField.substituteList fs vn r) l bv
      (Ty.optSubstitute lt vn r)
      (Ty.optSubstitute rt vn r)
      (match bv with
       | none => Ty.optSubstitute bt vn r
       | some bv' => if bv' == vn then bt else Ty.optSubstitute bt vn r)
      qd
end

mutual
def RecordField.containsFreeVariable : RecordField → String → Bool
  | mk _ t, vn => Ty.containsFreeVariable t vn

def Ty.containsFreeVariableList : List Ty → String → Bool
  | [], _ => false
  | t :: ts, vn => Ty.containsFreeVariable t vn || Ty.containsFreeVariableList ts vn

def RecordField.containsFreeVariableList : List RecordField → String → Bool
  | [], _ => false
  | f :: fs, vn => RecordField.containsFreeVariable f vn || RecordField.containsFreeVariableList fs vn

def Ty.optContainsFreeVariable : Option Ty → String → Bool
  | none, _ => false
  | some t, vn => Ty.containsFreeVariable t vn

def Ty.containsFreeVariable : Ty → String → Bool
  | mk k n ps fs _ bv lt rt bt _, vn =>
    if k == TypeKind.VARIABLE && n == vn then true
    else Ty.containsFreeVariableList ps vn ||
         RecordField.containsFreeVariableList fs vn ||
         Ty.optContainsFreeVariable lt vn ||
         Ty.optContainsFreeVariable rt vn ||
         match bv with
         | none => Ty.optContainsFreeVariable bt vn
         | some bv' => bv' != vn && Ty.optContainsFreeVariable bt vn
end

theorem Ty.containsFreeVariable_initNat : Ty.containsFreeVariable Ty.initNat "x" = false := Eq.refl false

theorem Ty.containsFreeVariable_initBool : Ty.containsFreeVariable Ty.initBool "x" = false := Eq.refl false

theorem Ty.containsFreeVariable_initUnit : Ty.containsFreeVariable Ty.initUnit "x" = false := Eq.refl false

theorem Ty.containsFreeVariable_initVariable_self (n : String) : Ty.containsFreeVariable (Ty.initVariable n) n = true :=
  Eq.refl true

theorem Ty.containsFreeVariable_initVariable_other (n : String) (m : String) (h : n != m) : Ty.containsFreeVariable (Ty.initVariable n) m = false :=
  if hnm : n == m then absurd (Eq.refl true) (Bool.false_ne_true)
  else Eq.refl false

def Ty.computeHash : Ty → Nat
  | mk k n ps fs l bv lt rt bt qd =>
    let h := k.toIndex
    let h := h * 31 + n.hash
    let h := h * 31 + l
    let h := List.foldl (fun acc p => acc * 31 + Ty.computeHash p) h ps
    let h := List.foldl (fun acc f => acc * 31 + (RecordField.name f).hash + Ty.computeHash (RecordField.fieldType f)) h fs
    let h := match bv with
      | none => h
      | some s => h * 31 + s.hash
    let h := match lt with
      | none => h
      | some t => h * 31 + Ty.computeHash t
    let h := match rt with
      | none => h
      | some t => h * 31 + Ty.computeHash t
    let h := match bt with
      | none => h
      | some t => h * 31 + Ty.computeHash t
    let h := match qd with
      | none => h
      | some d => h * 31 + d
    h

inductive TypeBinding : Type where
  | mk : String → Ty → TypeBinding

def TypeBinding.name : TypeBinding → String
  | mk n _ => n

def TypeBinding.boundType : TypeBinding → Ty
  | mk _ t => t

inductive TypeContext : Type where
  | empty : TypeContext
  | extend : TypeContext → String → Ty → TypeContext

def TypeContext.extend' (ctx : TypeContext) (name : String) (t : Ty) : TypeContext :=
  TypeContext.extend ctx name t

def TypeContext.lookup : TypeContext → String → Option Ty
  | empty, _ => none
  | extend ctx name t, n =>
    if name == n then some t else TypeContext.lookup ctx n

def TypeContext.contains (ctx : TypeContext) (name : String) : Bool :=
  match TypeContext.lookup ctx name with
  | some _ => true
  | none => false

def TypeContext.size : TypeContext → Nat
  | empty => 0
  | extend ctx _ _ => 1 + TypeContext.size ctx

theorem TypeContext.lookup_extend_self (ctx : TypeContext) (n : String) (t : Ty) :
  TypeContext.lookup (TypeContext.extend ctx n t) n = some t :=
  if h : n == n then Eq.refl (some t)
  else absurd (beq_self_eq_true n) (Bool.eq_false_iff.mpr h)

theorem TypeContext.lookup_extend_other (ctx : TypeContext) (n1 : String) (n2 : String) (t : Ty) (h : n1 != n2) :
  TypeContext.lookup (TypeContext.extend ctx n1 t) n2 = TypeContext.lookup ctx n2 :=
  if hne : n1 == n2 then absurd (beq_iff_eq.mpr hne) h
  else Eq.refl (TypeContext.lookup ctx n2)

theorem TypeContext.contains_extend_self (ctx : TypeContext) (n : String) (t : Ty) :
  TypeContext.contains (TypeContext.extend ctx n t) n = true :=
  if h : n == n then Eq.refl true
  else absurd (beq_self_eq_true n) (Bool.eq_false_iff.mpr h)

theorem TypeContext.size_extend (ctx : TypeContext) (n : String) (t : Ty) :
  TypeContext.size (TypeContext.extend ctx n t) = 1 + TypeContext.size ctx := Eq.refl _

theorem TypeContext.size_empty : TypeContext.size TypeContext.empty = 0 := Eq.refl 0

def TypeContext.clone : TypeContext → TypeContext
  | empty => empty
  | extend ctx n t => extend (TypeContext.clone ctx) n (Ty.clone t)

theorem TypeContext.clone_empty : TypeContext.clone TypeContext.empty = TypeContext.empty := Eq.refl TypeContext.empty

inductive TermKind : Type where
  | VARIABLE : TermKind
  | LITERAL : TermKind
  | LAMBDA : TermKind
  | APPLICATION : TermKind
  | PAIR : TermKind
  | FIRST : TermKind
  | SECOND : TermKind
  | INL : TermKind
  | INR : TermKind
  | CASE : TermKind
  | UNIT : TermKind
  | REFL : TermKind
  | J_ELIMINATOR : TermKind
  | ZERO : TermKind
  | SUCC : TermKind
  | NAT_REC : TermKind
  | LET : TermKind
  | ANNOTATION : TermKind
  deriving Repr, BEq, DecidableEq

def TermKind.toString : TermKind → String
  | VARIABLE => "var"
  | LITERAL => "lit"
  | LAMBDA => "lam"
  | APPLICATION => "app"
  | PAIR => "pair"
  | FIRST => "fst"
  | SECOND => "snd"
  | INL => "inl"
  | INR => "inr"
  | CASE => "case"
  | UNIT => "unit"
  | REFL => "refl"
  | J_ELIMINATOR => "J"
  | ZERO => "zero"
  | SUCC => "succ"
  | NAT_REC => "natrec"
  | LET => "let"
  | ANNOTATION => "ann"

theorem TermKind.toString_VARIABLE : TermKind.toString TermKind.VARIABLE = "var" := Eq.refl "var"
theorem TermKind.toString_LITERAL : TermKind.toString TermKind.LITERAL = "lit" := Eq.refl "lit"
theorem TermKind.toString_LAMBDA : TermKind.toString TermKind.LAMBDA = "lam" := Eq.refl "lam"
theorem TermKind.toString_APPLICATION : TermKind.toString TermKind.APPLICATION = "app" := Eq.refl "app"
theorem TermKind.toString_PAIR : TermKind.toString TermKind.PAIR = "pair" := Eq.refl "pair"
theorem TermKind.toString_FIRST : TermKind.toString TermKind.FIRST = "fst" := Eq.refl "fst"
theorem TermKind.toString_SECOND : TermKind.toString TermKind.SECOND = "snd" := Eq.refl "snd"
theorem TermKind.toString_INL : TermKind.toString TermKind.INL = "inl" := Eq.refl "inl"
theorem TermKind.toString_INR : TermKind.toString TermKind.INR = "inr" := Eq.refl "inr"
theorem TermKind.toString_CASE : TermKind.toString TermKind.CASE = "case" := Eq.refl "case"
theorem TermKind.toString_UNIT : TermKind.toString TermKind.UNIT = "unit" := Eq.refl "unit"
theorem TermKind.toString_REFL : TermKind.toString TermKind.REFL = "refl" := Eq.refl "refl"
theorem TermKind.toString_J_ELIMINATOR : TermKind.toString TermKind.J_ELIMINATOR = "J" := Eq.refl "J"
theorem TermKind.toString_ZERO : TermKind.toString TermKind.ZERO = "zero" := Eq.refl "zero"
theorem TermKind.toString_SUCC : TermKind.toString TermKind.SUCC = "succ" := Eq.refl "succ"
theorem TermKind.toString_NAT_REC : TermKind.toString TermKind.NAT_REC = "natrec" := Eq.refl "natrec"
theorem TermKind.toString_LET : TermKind.toString TermKind.LET = "let" := Eq.refl "let"
theorem TermKind.toString_ANNOTATION : TermKind.toString TermKind.ANNOTATION = "ann" := Eq.refl "ann"

inductive LiteralValue : Type where
  | boolVal : Bool → LiteralValue
  | natVal : Nat → LiteralValue
  | intVal : Int → LiteralValue
  | realVal : Float → LiteralValue
  | stringVal : String → LiteralValue

inductive Trm : Type where
  | mk : TermKind → String → List Trm → Option String → Option Ty → Option LiteralValue → Trm

def Trm.kind : Trm → TermKind
  | mk k _ _ _ _ _ => k

def Trm.name : Trm → String
  | mk _ n _ _ _ _ => n

def Trm.subTerms : Trm → List Trm
  | mk _ _ sts _ _ _ => sts

def Trm.boundVariable : Trm → Option String
  | mk _ _ _ bv _ _ => bv

def Trm.annotationType : Trm → Option Ty
  | mk _ _ _ _ at _ => at

def Trm.literalValue : Trm → Option LiteralValue
  | mk _ _ _ _ _ lv => lv

def Trm.init (k : TermKind) : Trm := Trm.mk k "" [] none none none

def Trm.initVariable (n : String) : Trm := Trm.mk TermKind.VARIABLE n [] none none none

def Trm.initLambda (param : String) (body : Trm) : Trm := Trm.mk TermKind.LAMBDA "" [body] (some param) none none

def Trm.initApplication (func : Trm) (arg : Trm) : Trm := Trm.mk TermKind.APPLICATION "" [func, arg] none none none

def Trm.initPair (first : Trm) (second : Trm) : Trm := Trm.mk TermKind.PAIR "" [first, second] none none none

def Trm.initFirst (pair : Trm) : Trm := Trm.mk TermKind.FIRST "" [pair] none none none

def Trm.initSecond (pair : Trm) : Trm := Trm.mk TermKind.SECOND "" [pair] none none none

def Trm.initInl (value : Trm) : Trm := Trm.mk TermKind.INL "" [value] none none none

def Trm.initInr (value : Trm) : Trm := Trm.mk TermKind.INR "" [value] none none none

def Trm.initUnit : Trm := Trm.mk TermKind.UNIT "" [] none none none

def Trm.initRefl (witness : Trm) : Trm := Trm.mk TermKind.REFL "" [witness] none none none

def Trm.initZero : Trm := Trm.mk TermKind.ZERO "" [] none none none

def Trm.initSucc (n : Trm) : Trm := Trm.mk TermKind.SUCC "" [n] none none none

def Trm.initLiteralNat (v : Nat) : Trm := Trm.mk TermKind.LITERAL "" [] none none (some (LiteralValue.natVal v))

def Trm.initLiteralBool (v : Bool) : Trm := Trm.mk TermKind.LITERAL "" [] none none (some (LiteralValue.boolVal v))

def Trm.initAnnotation (term : Trm) (ann : Ty) : Trm := Trm.mk TermKind.ANNOTATION "" [term] none (some ann) none

theorem Trm.kind_initVariable (n : String) : (Trm.initVariable n).kind = TermKind.VARIABLE := Eq.refl TermKind.VARIABLE
theorem Trm.kind_initLambda (p : String) (b : Trm) : (Trm.initLambda p b).kind = TermKind.LAMBDA := Eq.refl TermKind.LAMBDA
theorem Trm.kind_initApplication (f a : Trm) : (Trm.initApplication f a).kind = TermKind.APPLICATION := Eq.refl TermKind.APPLICATION
theorem Trm.kind_initPair (f s : Trm) : (Trm.initPair f s).kind = TermKind.PAIR := Eq.refl TermKind.PAIR
theorem Trm.kind_initFirst (p : Trm) : (Trm.initFirst p).kind = TermKind.FIRST := Eq.refl TermKind.FIRST
theorem Trm.kind_initSecond (p : Trm) : (Trm.initSecond p).kind = TermKind.SECOND := Eq.refl TermKind.SECOND
theorem Trm.kind_initInl (v : Trm) : (Trm.initInl v).kind = TermKind.INL := Eq.refl TermKind.INL
theorem Trm.kind_initInr (v : Trm) : (Trm.initInr v).kind = TermKind.INR := Eq.refl TermKind.INR
theorem Trm.kind_initUnit : Trm.initUnit.kind = TermKind.UNIT := Eq.refl TermKind.UNIT
theorem Trm.kind_initRefl (w : Trm) : (Trm.initRefl w).kind = TermKind.REFL := Eq.refl TermKind.REFL
theorem Trm.kind_initZero : Trm.initZero.kind = TermKind.ZERO := Eq.refl TermKind.ZERO
theorem Trm.kind_initSucc (n : Trm) : (Trm.initSucc n).kind = TermKind.SUCC := Eq.refl TermKind.SUCC
theorem Trm.kind_initLiteralNat (v : Nat) : (Trm.initLiteralNat v).kind = TermKind.LITERAL := Eq.refl TermKind.LITERAL
theorem Trm.kind_initLiteralBool (v : Bool) : (Trm.initLiteralBool v).kind = TermKind.LITERAL := Eq.refl TermKind.LITERAL
theorem Trm.kind_initAnnotation (t : Trm) (a : Ty) : (Trm.initAnnotation t a).kind = TermKind.ANNOTATION := Eq.refl TermKind.ANNOTATION

def LiteralValue.equals : LiteralValue → LiteralValue → Bool
  | boolVal b1, boolVal b2 => b1 == b2
  | natVal n1, natVal n2 => n1 == n2
  | intVal i1, intVal i2 => i1 == i2
  | realVal r1, realVal r2 => r1 == r2
  | stringVal s1, stringVal s2 => s1 == s2
  | _, _ => false

def Trm.equalsList : List Trm → List Trm → Bool
  | [], [] => true
  | [], _ => false
  | _, [] => false
  | t1 :: ts1, t2 :: ts2 => Trm.equals t1 t2 && Trm.equalsList ts1 ts2

def Trm.optEquals : Option Ty → Option Ty → Bool
  | none, none => true
  | some t1, some t2 => Ty.equals t1 t2
  | _, _ => false

def Trm.optLiteralEquals : Option LiteralValue → Option LiteralValue → Bool
  | none, none => true
  | some l1, some l2 => LiteralValue.equals l1 l2
  | _, _ => false

def Trm.optStringEquals : Option String → Option String → Bool
  | none, none => true
  | some s1, some s2 => s1 == s2
  | _, _ => false

def Trm.equals : Trm → Trm → Bool
  | mk k1 n1 sts1 bv1 at1 lv1, mk k2 n2 sts2 bv2 at2 lv2 =>
    k1 == k2 &&
    n1 == n2 &&
    Trm.equalsList sts1 sts2 &&
    Trm.optStringEquals bv1 bv2 &&
    Trm.optEquals at1 at2 &&
    Trm.optLiteralEquals lv1 lv2

theorem Trm.equals_initZero : Trm.equals Trm.initZero Trm.initZero = true :=
  Eq.refl true

theorem Trm.equals_initUnit : Trm.equals Trm.initUnit Trm.initUnit = true :=
  Eq.refl true

theorem Trm.equals_initVariable_self (n : String) : Trm.equals (Trm.initVariable n) (Trm.initVariable n) = true :=
  Eq.refl true

def Trm.cloneList : List Trm → List Trm
  | [] => []
  | t :: ts => Trm.clone t :: Trm.cloneList ts

def Trm.optClone : Option Ty → Option Ty
  | none => none
  | some t => some (Ty.clone t)

def Trm.optLiteralClone : Option LiteralValue → Option LiteralValue
  | none => none
  | some l => some l

def Trm.optStringClone : Option String → Option String
  | none => none
  | some s => some s

def Trm.clone : Trm → Trm
  | mk k n sts bv at lv =>
    mk k n (Trm.cloneList sts) (Trm.optStringClone bv) (Trm.optClone at) (Trm.optLiteralClone lv)

theorem Trm.clone_kind (t : Trm) : (Trm.clone t).kind = t.kind :=
  Trm.recOn t (fun k n sts bv at lv => Eq.refl k)

structure TypeJudgment where
  context : TypeContext
  term : Trm
  inferredType : Ty
  isValid : Bool
  derivationDepth : Nat

def TypeJudgment.init (ctx : TypeContext) (term : Trm) (inferredType : Ty) : TypeJudgment :=
  { context := ctx, term := term, inferredType := inferredType, isValid := false, derivationDepth := 0 }

def TypeJudgment.checkLiteralType (t : Trm) (inferredType : Ty) : Bool :=
  match t.literalValue with
  | none => false
  | some lv =>
    match lv with
    | LiteralValue.boolVal _ => inferredType.kind == TypeKind.BOOL
    | LiteralValue.natVal _ => inferredType.kind == TypeKind.NAT
    | LiteralValue.intVal _ => inferredType.kind == TypeKind.INT
    | LiteralValue.realVal _ => inferredType.kind == TypeKind.REAL
    | LiteralValue.stringVal _ => inferredType.kind == TypeKind.STRING

def TypeJudgment.checkWellFormedness (j : TypeJudgment) : Bool :=
  match j.term.kind with
  | TermKind.VARIABLE => TypeContext.contains j.context j.term.name
  | TermKind.UNIT => j.inferredType.kind == TypeKind.UNIT
  | TermKind.ZERO => j.inferredType.kind == TypeKind.NAT
  | TermKind.LITERAL => TypeJudgment.checkLiteralType j.term j.inferredType
  | TermKind.LAMBDA => j.inferredType.kind == TypeKind.FUNCTION || j.inferredType.kind == TypeKind.DEPENDENT_FUNCTION
  | TermKind.APPLICATION => true
  | TermKind.PAIR => j.inferredType.kind == TypeKind.TUPLE || j.inferredType.kind == TypeKind.DEPENDENT_PAIR
  | TermKind.FIRST => true
  | TermKind.SECOND => true
  | TermKind.INL => true
  | TermKind.INR => true
  | TermKind.SUCC => j.inferredType.kind == TypeKind.NAT
  | TermKind.REFL => j.inferredType.kind == TypeKind.IDENTITY
  | TermKind.ANNOTATION => true
  | _ => true

def TypeJudgment.validate (j : TypeJudgment) : Bool :=
  j.checkWellFormedness

structure DependentPi where
  paramName : String
  paramType : Ty
  returnType : Ty
  universeLevel : Nat

def DependentPi.init (paramName : String) (paramType : Ty) (returnType : Ty) : DependentPi :=
  { paramName := paramName, paramType := paramType, returnType := returnType, universeLevel := max paramType.getUniverseLevel returnType.getUniverseLevel }

def DependentPi.toType (pi : DependentPi) : Ty :=
  Ty.mk TypeKind.DEPENDENT_FUNCTION "" [] [] pi.universeLevel (some pi.paramName) (some pi.paramType) none (some pi.returnType) none

def DependentPi.apply (pi : DependentPi) (arg : Ty) : Ty :=
  Ty.substitute pi.returnType pi.paramName arg

def DependentPi.clone (pi : DependentPi) : DependentPi :=
  { pi with paramType := Ty.clone pi.paramType, returnType := Ty.clone pi.returnType }

def DependentPi.equals (pi1 : DependentPi) (pi2 : DependentPi) : Bool :=
  pi1.paramName == pi2.paramName &&
  Ty.equals pi1.paramType pi2.paramType &&
  Ty.equals pi1.returnType pi2.returnType

theorem DependentPi.equals_self (pi : DependentPi) : DependentPi.equals pi pi = true :=
  Eq.refl (pi.paramName == pi.paramName && Ty.equals pi.paramType pi.paramType && Ty.equals pi.returnType pi.returnType) |> Eq.trans (and_self _)

structure DependentSigma where
  fstName : String
  fstType : Ty
  sndType : Ty
  universeLevel : Nat

def DependentSigma.init (fstName : String) (fstType : Ty) (sndType : Ty) : DependentSigma :=
  { fstName := fstName, fstType := fstType, sndType := sndType, universeLevel := max fstType.getUniverseLevel sndType.getUniverseLevel }

def DependentSigma.toType (sigma : DependentSigma) : Ty :=
  Ty.mk TypeKind.DEPENDENT_PAIR "" [] [] sigma.universeLevel (some sigma.fstName) (some sigma.fstType) none (some sigma.sndType) none

def DependentSigma.getSecondType (sigma : DependentSigma) (firstValue : Ty) : Ty :=
  Ty.substitute sigma.sndType sigma.fstName firstValue

def DependentSigma.clone (sigma : DependentSigma) : DependentSigma :=
  { sigma with fstType := Ty.clone sigma.fstType, sndType := Ty.clone sigma.sndType }

def DependentSigma.equals (s1 : DependentSigma) (s2 : DependentSigma) : Bool :=
  s1.fstName == s2.fstName &&
  Ty.equals s1.fstType s2.fstType &&
  Ty.equals s1.sndType s2.sndType

theorem DependentSigma.equals_self (sigma : DependentSigma) : DependentSigma.equals sigma sigma = true :=
  Eq.refl (sigma.fstName == sigma.fstName && Ty.equals sigma.fstType sigma.fstType && Ty.equals sigma.sndType sigma.sndType) |> Eq.trans (and_self _)

structure IdentityType where
  baseType : Ty
  leftTerm : Trm
  rightTerm : Trm

def IdentityType.init (baseType : Ty) (left : Trm) (right : Trm) : IdentityType :=
  { baseType := baseType, leftTerm := left, rightTerm := right }

def IdentityType.toType (id : IdentityType) : Ty :=
  Ty.mk TypeKind.IDENTITY "" [id.baseType] [] 0 none none none none none

def IdentityType.refl (baseType : Ty) (term : Trm) : IdentityType :=
  IdentityType.init baseType term (Trm.clone term)

def IdentityType.symmetry (id : IdentityType) : IdentityType :=
  IdentityType.init (Ty.clone id.baseType) (Trm.clone id.rightTerm) (Trm.clone id.leftTerm)

def IdentityType.transitivity (id1 : IdentityType) (id2 : IdentityType) : Option IdentityType :=
  if Trm.equals id1.rightTerm id2.leftTerm && Ty.equals id1.baseType id2.baseType then
    some (IdentityType.init (Ty.clone id1.baseType) (Trm.clone id1.leftTerm) (Trm.clone id2.rightTerm))
  else none

def IdentityType.clone (id : IdentityType) : IdentityType :=
  { baseType := Ty.clone id.baseType, leftTerm := Trm.clone id.leftTerm, rightTerm := Trm.clone id.rightTerm }

def IdentityType.isReflexive (id : IdentityType) : Bool :=
  Trm.equals id.leftTerm id.rightTerm

theorem IdentityType.isReflexive_refl (base : Ty) (term : Trm) :
  IdentityType.isReflexive (IdentityType.refl base term) = true :=
  let cloned := Trm.clone term
  Eq.refl (Trm.equals term cloned) |> congrArg (fun x => Trm.equals term x) (Trm.clone_kind term) |> fun h =>
    Trm.recOn term (fun k n sts bv at lv =>
      Eq.refl (Trm.equals (Trm.mk k n sts bv at lv) (Trm.mk k n (Trm.cloneList sts) (Trm.optStringClone bv) (Trm.optClone at) (Trm.optLiteralClone lv))) |> Eq.trans (and_self _)
    )

structure UniverseType where
  level : Nat
  cumulative : Bool

def UniverseType.init (level : Nat) : UniverseType :=
  { level := level, cumulative := true }

def UniverseType.toType (u : UniverseType) : Ty :=
  Ty.initUniverse u.level

def UniverseType.typeOf (u : UniverseType) : UniverseType :=
  { level := u.level + 1, cumulative := u.cumulative }

def UniverseType.contains (u : UniverseType) (other : UniverseType) : Bool :=
  if u.cumulative then other.level < u.level
  else if u.level > 0 then other.level == u.level - 1
  else false

def UniverseType.lub (u1 : UniverseType) (u2 : UniverseType) : UniverseType :=
  { level := max u1.level u2.level, cumulative := u1.cumulative && u2.cumulative }

def UniverseType.clone (u : UniverseType) : UniverseType :=
  { u with }

theorem UniverseType.typeOf_level (u : UniverseType) : (UniverseType.typeOf u).level = u.level + 1 := Eq.refl _

theorem UniverseType.contains_self_cumulative (u : UniverseType) (h : u.cumulative) :
  UniverseType.contains (UniverseType.typeOf u) u = true :=
  if hc : u.cumulative then Eq.refl (u.level < u.level + 1) |> Eq.trans (Nat.lt_succ_self u.level)
  else absurd h hc

theorem UniverseType.lub_level (u1 : UniverseType) (u2 : UniverseType) :
  (UniverseType.lub u1 u2).level = max u1.level u2.level := Eq.refl _

structure Constructor where
  name : String
  argTypes : List Ty
  resultType : Ty

def Constructor.init (name : String) (resultType : Ty) : Constructor :=
  { name := name, argTypes := [], resultType := resultType }

def Constructor.addArgType (c : Constructor) (argType : Ty) : Constructor :=
  { c with argTypes := c.argTypes ++ [argType] }

def Constructor.clone (c : Constructor) : Constructor :=
  { name := c.name, argTypes := c.argTypes.map Ty.clone, resultType := Ty.clone c.resultType }

structure InductiveTy where
  name : String
  constructors : List Constructor
  parameters : List Ty
  indices : List Ty
  universeLevel : Nat

def InductiveTy.init (name : String) : InductiveTy :=
  { name := name, constructors := [], parameters := [], indices := [], universeLevel := 0 }

def InductiveTy.addConstructor (ind : InductiveTy) (c : Constructor) : InductiveTy :=
  { ind with constructors := ind.constructors ++ [c] }

def InductiveTy.initNat : InductiveTy :=
  let nat := InductiveTy.init "Nat"
  let zero := Constructor.init "zero" Ty.initNat
  let succ := Constructor.init "succ" Ty.initNat |> fun c => Constructor.addArgType c Ty.initNat
  nat |> InductiveTy.addConstructor zero |> InductiveTy.addConstructor succ

def InductiveTy.initBool : InductiveTy :=
  let bool := InductiveTy.init "Bool"
  let trueC := Constructor.init "true" Ty.initBool
  let falseC := Constructor.init "false" Ty.initBool
  bool |> InductiveTy.addConstructor trueC |> InductiveTy.addConstructor falseC

def InductiveTy.toType (ind : InductiveTy) : Ty :=
  Ty.initVariable ind.name

theorem InductiveTy.initNat_name : InductiveTy.initNat.name = "Nat" := Eq.refl "Nat"

theorem InductiveTy.initNat_constructors_len : InductiveTy.initNat.constructors.length = 2 := Eq.refl 2

theorem InductiveTy.initBool_name : InductiveTy.initBool.name = "Bool" := Eq.refl "Bool"

theorem InductiveTy.initBool_constructors_len : InductiveTy.initBool.constructors.length = 2 := Eq.refl 2

structure TypeChecker where
  inferenceCount : Nat
  checkCount : Nat
  unificationCount : Nat

def TypeChecker.init : TypeChecker :=
  { inferenceCount := 0, checkCount := 0, unificationCount := 0 }

def TypeChecker.withInference (tc : TypeChecker) : TypeChecker :=
  { tc with inferenceCount := tc.inferenceCount + 1 }

def TypeChecker.withCheck (tc : TypeChecker) : TypeChecker :=
  { tc with checkCount := tc.checkCount + 1 }

def TypeChecker.withUnification (tc : TypeChecker) : TypeChecker :=
  { tc with unificationCount := tc.unificationCount + 1 }

mutual
def TypeChecker.subtypeList : TypeChecker → List Ty → List Ty → Bool
  | tc, [], [] => true
  | tc, [], _ :: _ => false
  | tc, _ :: _, [] => false
  | tc, s :: ss, p :: ps => TypeChecker.subtype tc s p && TypeChecker.subtypeList tc ss ps

def TypeChecker.subtype : TypeChecker → Ty → Ty → Bool
  | tc, sub, sup =>
    if Ty.equals sub sup then true
    else if sup.kind == TypeKind.TOP then true
    else if sub.kind == TypeKind.BOTTOM then true
    else if sub.kind == TypeKind.NAT && sup.kind == TypeKind.INT then true
    else if sub.kind == TypeKind.INT && sup.kind == TypeKind.REAL then true
    else if sub.kind == TypeKind.REAL && sup.kind == TypeKind.COMPLEX then true
    else match sub.kind, sup.kind with
         | TypeKind.FUNCTION, TypeKind.FUNCTION =>
           match sub.leftType, sup.leftType, sub.rightType, sup.rightType with
           | some subDom, some supDom, some subCod, some supCod =>
             TypeChecker.subtype tc supDom subDom && TypeChecker.subtype tc subCod supCod
           | _, _, _, _ => false
         | TypeKind.TUPLE, TypeKind.TUPLE =>
           if sub.parameters.length == sup.parameters.length then
             TypeChecker.subtypeList tc sub.parameters sup.parameters
           else false
         | TypeKind.UNIVERSE, TypeKind.UNIVERSE =>
           sub.universeLevel <= sup.universeLevel
         | _, _ => false
end

theorem TypeChecker.subtype_refl (tc : TypeChecker) (t : Ty) : TypeChecker.subtype tc t t = true :=
  Eq.refl true |> congrArg (fun x => if x then true else _) (Ty.equals_refl t) |> congrArg (fun x => if x then x else _) (if_pos (Eq.refl true))

theorem TypeChecker.subtype_top (tc : TypeChecker) (t : Ty) : TypeChecker.subtype tc t Ty.initTop = true :=
  if het : Ty.equals t Ty.initTop then Eq.refl true |> congrArg (fun x => if x then true else _) het
  else if htop : Ty.initTop.kind == TypeKind.TOP then Eq.refl true |> congrArg (fun x => if x then true else _) (if_pos htop)
  else Eq.refl false |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.TOP))) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.TOP))) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.BOTTOM))) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.NAT))) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.INT))) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.REAL)))

theorem TypeChecker.subtype_bottom (tc : TypeChecker) (t : Ty) : TypeChecker.subtype tc Ty.initBottom t = true :=
  if het : Ty.equals Ty.initBottom t then Eq.refl true |> congrArg (fun x => if x then true else _) het
  else if hbot : Ty.initBottom.kind == TypeKind.BOTTOM then Eq.refl true |> congrArg (fun x => if x then x else _) (if_neg het) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.TOP))) |> congrArg (fun x => if x then x else _) (if_pos hbot)
  else Eq.refl false

theorem TypeChecker.subtype_nat_int (tc : TypeChecker) : TypeChecker.subtype tc Ty.initNat Ty.initInt = true :=
  if het : Ty.equals Ty.initNat Ty.initInt then absurd (Eq.refl true) (Bool.false_ne_true)
  else if hnat : Ty.initNat.kind == TypeKind.NAT then
    if hint : Ty.initInt.kind == TypeKind.INT then
      Eq.refl true |> congrArg (fun x => if x then true else _) (if_neg het) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.TOP))) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.BOTTOM))) |> congrArg (fun x => if x then x else _) (if_pos hnat) |> congrArg (fun x => if x then x else _) (if_pos hint)
    else Eq.refl false
  else Eq.refl false

theorem TypeChecker.subtype_int_real (tc : TypeChecker) : TypeChecker.subtype tc Ty.initInt Ty.initReal = true :=
  if het : Ty.equals Ty.initInt Ty.initReal then absurd (Eq.refl true) (Bool.false_ne_true)
  else if hint : Ty.initInt.kind == TypeKind.INT then
    if hreal : Ty.initReal.kind == TypeKind.REAL then
      Eq.refl true |> congrArg (fun x => if x then true else _) (if_neg het) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.TOP))) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.BOTTOM))) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.NAT))) |> congrArg (fun x => if x then x else _) (if_pos hint) |> congrArg (fun x => if x then x else _) (if_pos hreal)
    else Eq.refl false
  else Eq.refl false

theorem TypeChecker.not_subtype_int_nat (tc : TypeChecker) : TypeChecker.subtype tc Ty.initInt Ty.initNat = false :=
  if het : Ty.equals Ty.initInt Ty.initNat then absurd (Eq.refl true) (Bool.false_ne_true)
  else Eq.refl false |> congrArg (fun x => if x then true else _) (if_neg het) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.TOP))) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.BOTTOM))) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.NAT))) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.INT))) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.REAL)))

theorem TypeChecker.subtype_universe_zero_one (tc : TypeChecker) :
  TypeChecker.subtype tc (Ty.initUniverse 0) (Ty.initUniverse 1) = true :=
  if het : Ty.equals (Ty.initUniverse 0) (Ty.initUniverse 1) then absurd (Eq.refl true) (Bool.false_ne_true)
  else Eq.refl true |> congrArg (fun x => if x then true else _) (if_neg het) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.TOP))) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.BOTTOM))) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.NAT))) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.INT))) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.REAL))) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.FUNCTION))) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.TUPLE))) |> congrArg (fun x => if x then x else _) (if_pos (beq_self_eq_true TypeKind.UNIVERSE)) |> congrArg (fun x => if x then x else _) (if_pos (beq_self_eq_true TypeKind.UNIVERSE)) |> congrArg (fun x => if x then x else _) (if_pos (Nat.le_step (Nat.le_refl 0)))

theorem TypeChecker.not_subtype_universe_one_zero (tc : TypeChecker) :
  TypeChecker.subtype tc (Ty.initUniverse 1) (Ty.initUniverse 0) = false :=
  if het : Ty.equals (Ty.initUniverse 1) (Ty.initUniverse 0) then absurd (Eq.refl true) (Bool.false_ne_true)
  else Eq.refl false |> congrArg (fun x => if x then true else _) (if_neg het) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.TOP))) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.BOTTOM))) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.NAT))) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.INT))) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.REAL))) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.FUNCTION))) |> congrArg (fun x => if x then x else _) (if_neg (fun h => h (beq_self_eq_true TypeKind.TUPLE))) |> congrArg (fun x => if x then x else _) (if_pos (beq_self_eq_true TypeKind.UNIVERSE)) |> congrArg (fun x => if x then x else _) (if_pos (beq_self_eq_true TypeKind.UNIVERSE)) |> congrArg (fun x => if x then x else _) (if_neg (Nat.not_lt_of_le (Nat.le_step (Nat.le_refl 0))))

def TypeChecker.inferType (tc : TypeChecker) (ctx : TypeContext) (term : Trm) : Except TypeTheoryError Ty :=
  match term.kind with
  | TermKind.VARIABLE =>
    match TypeContext.lookup ctx term.name with
    | some t => Except.ok (Ty.clone t)
    | none => Except.error TypeTheoryError.VariableNotInContext
  | TermKind.LITERAL =>
    match term.literalValue with
    | some (LiteralValue.boolVal _) => Except.ok Ty.initBool
    | some (LiteralValue.natVal _) => Except.ok Ty.initNat
    | some (LiteralValue.intVal _) => Except.ok Ty.initInt
    | some (LiteralValue.realVal _) => Except.ok Ty.initReal
    | some (LiteralValue.stringVal _) => Except.ok Ty.initString
    | none => Except.error TypeTheoryError.InvalidTypeConstruction
  | TermKind.UNIT => Except.ok Ty.initUnit
  | TermKind.ZERO => Except.ok Ty.initNat
  | TermKind.SUCC =>
    match term.subTerms with
    | [n] =>
      match TypeChecker.inferType tc ctx n with
      | Except.ok nType =>
        if nType.kind == TypeKind.NAT then Except.ok Ty.initNat
        else Except.error TypeTheoryError.TypeMismatch
      | Except.error e => Except.error e
    | _ => Except.error TypeTheoryError.InvalidTypeConstruction
  | TermKind.LAMBDA =>
    match term.boundVariable, term.subTerms with
    | some param, [body] =>
      let paramType := match term.annotationType with
        | some ann => ann
        | none => Ty.initTop
      let extendedCtx := TypeContext.extend ctx param paramType
      match TypeChecker.inferType tc extendedCtx body with
      | Except.ok bodyType => Except.ok (Ty.initFunction paramType bodyType)
      | Except.error e => Except.error e
    | _, _ => Except.error TypeTheoryError.InvalidTypeConstruction
  | TermKind.APPLICATION =>
    match term.subTerms with
    | [func, arg] =>
      match TypeChecker.inferType tc ctx func, TypeChecker.inferType tc ctx arg with
      | Except.ok funcType, Except.ok argType =>
        match funcType.kind with
        | TypeKind.FUNCTION =>
          match funcType.leftType, funcType.rightType with
          | some domain, some codomain =>
            if TypeChecker.subtype tc argType domain then Except.ok (Ty.clone codomain)
            else Except.error TypeTheoryError.TypeMismatch
          | _, _ => Except.error TypeTheoryError.InvalidTypeConstruction
        | TypeKind.DEPENDENT_FUNCTION =>
          match funcType.bodyType with
          | some body =>
            let result := Ty.substitute body (match funcType.boundVariable with | some bv => bv | none => "") argType
            Except.ok result
          | none => Except.error TypeTheoryError.InvalidTypeConstruction
        | _ => Except.error TypeTheoryError.InvalidApplication
      | Except.error e, _ => Except.error e
      | _, Except.error e => Except.error e
    | _ => Except.error TypeTheoryError.InvalidApplication
  | TermKind.PAIR =>
    match term.subTerms with
    | [fst, snd] =>
      match TypeChecker.inferType tc ctx fst, TypeChecker.inferType tc ctx snd with
      | Except.ok fstType, Except.ok sndType => Except.ok (Ty.initTuple [fstType, sndType])
      | Except.error e, _ => Except.error e
      | _, Except.error e => Except.error e
    | _ => Except.error TypeTheoryError.InvalidTypeConstruction
  | TermKind.FIRST =>
    match term.subTerms with
    | [pair] =>
      match TypeChecker.inferType tc ctx pair with
      | Except.ok pairType =>
        match pairType.kind with
        | TypeKind.TUPLE =>
          match pairType.parameters with
          | fstType :: _ => Except.ok (Ty.clone fstType)
          | [] => Except.error TypeTheoryError.InvalidProjection
        | TypeKind.DEPENDENT_PAIR =>
          match pairType.leftType with
          | some left => Except.ok (Ty.clone left)
          | none => Except.error TypeTheoryError.InvalidProjection
        | _ => Except.error TypeTheoryError.InvalidProjection
      | Except.error e => Except.error e
    | _ => Except.error TypeTheoryError.InvalidProjection
  | TermKind.SECOND =>
    match term.subTerms with
    | [pair] =>
      match TypeChecker.inferType tc ctx pair with
      | Except.ok pairType =>
        match pairType.kind with
        | TypeKind.TUPLE =>
          match pairType.parameters with
          | _ :: sndType :: _ => Except.ok (Ty.clone sndType)
          | _ => Except.error TypeTheoryError.InvalidProjection
        | TypeKind.DEPENDENT_PAIR =>
          match pairType.bodyType with
          | some body => Except.ok (Ty.clone body)
          | none => Except.error TypeTheoryError.InvalidProjection
        | _ => Except.error TypeTheoryError.InvalidProjection
      | Except.error e => Except.error e
    | _ => Except.error TypeTheoryError.InvalidProjection
  | TermKind.INL =>
    match term.subTerms with
    | [value] =>
      match TypeChecker.inferType tc ctx value with
      | Except.ok innerType => Except.ok (Ty.initSum innerType Ty.initBottom)
      | Except.error e => Except.error e
    | _ => Except.error TypeTheoryError.InvalidTypeConstruction
  | TermKind.INR =>
    match term.subTerms with
    | [value] =>
      match TypeChecker.inferType tc ctx value with
      | Except.ok innerType => Except.ok (Ty.initSum Ty.initBottom innerType)
      | Except.error e => Except.error e
    | _ => Except.error TypeTheoryError.InvalidTypeConstruction
  | TermKind.REFL =>
    match term.subTerms with
    | [witness] =>
      match TypeChecker.inferType tc ctx witness with
      | Except.ok witnessType =>
        Except.ok (Ty.mk TypeKind.IDENTITY "" [witnessType] [] 0 none none none none none)
      | Except.error e => Except.error e
    | _ => Except.error TypeTheoryError.InvalidTypeConstruction
  | TermKind.ANNOTATION =>
    match term.annotationType with
    | some ann => Except.ok (Ty.clone ann)
    | none => Except.error TypeTheoryError.InvalidTypeConstruction
  | _ => Except.error TypeTheoryError.InvalidTypeConstruction

theorem TypeChecker.inferType_zero (tc : TypeChecker) (ctx : TypeContext) :
  TypeChecker.inferType tc ctx Trm.initZero = Except.ok Ty.initNat := Eq.refl (Except.ok Ty.initNat)

theorem TypeChecker.inferType_unit (tc : TypeChecker) (ctx : TypeContext) :
  TypeChecker.inferType tc ctx Trm.initUnit = Except.ok Ty.initUnit := Eq.refl (Except.ok Ty.initUnit)

theorem TypeChecker.inferType_literalNat (tc : TypeChecker) (ctx : TypeContext) (v : Nat) :
  TypeChecker.inferType tc ctx (Trm.initLiteralNat v) = Except.ok Ty.initNat := Eq.refl (Except.ok Ty.initNat)

theorem TypeChecker.inferType_literalBool (tc : TypeChecker) (ctx : TypeContext) (v : Bool) :
  TypeChecker.inferType tc ctx (Trm.initLiteralBool v) = Except.ok Ty.initBool := Eq.refl (Except.ok Ty.initBool)

def TypeChecker.checkType (tc : TypeChecker) (ctx : TypeContext) (term : Trm) (expected : Ty) : Except TypeTheoryError Bool :=
  match TypeChecker.inferType tc ctx term with
  | Except.ok inferred => Except.ok (TypeChecker.subtype tc inferred expected)
  | Except.error e => Except.error e

def TypeChecker.unifyTypesList : TypeChecker → List Ty → List Ty → Except TypeTheoryError (List Ty)
  | tc, [], [] => Except.ok []
  | tc, [], _ :: _ => Except.error TypeTheoryError.UnificationFailure
  | tc, _ :: _, [] => Except.error TypeTheoryError.UnificationFailure
  | tc, t1 :: ts1, t2 :: ts2 =>
    match TypeChecker.unifyTypes tc t1 t2 with
    | Except.ok u =>
      match TypeChecker.unifyTypesList tc ts1 ts2 with
      | Except.ok us => Except.ok (u :: us)
      | Except.error e => Except.error e
    | Except.error e => Except.error e

mutual
def TypeChecker.unifyTypes : TypeChecker → Ty → Ty → Except TypeTheoryError Ty
  | tc, t1, t2 =>
    if Ty.equals t1 t2 then Except.ok (Ty.clone t1)
    else if t1.kind == TypeKind.VARIABLE then Except.ok (Ty.clone t2)
    else if t2.kind == TypeKind.VARIABLE then Except.ok (Ty.clone t1)
    else if t1.kind == TypeKind.TOP then Except.ok (Ty.clone t2)
    else if t2.kind == TypeKind.TOP then Except.ok (Ty.clone t1)
    else if t1.kind == TypeKind.BOTTOM then Except.ok (Ty.clone t2)
    else if t2.kind == TypeKind.BOTTOM then Except.ok (Ty.clone t1)
    else match t1.kind, t2.kind with
         | TypeKind.FUNCTION, TypeKind.FUNCTION =>
           match t1.leftType, t2.leftType, t1.rightType, t2.rightType with
           | some d1, some d2, some c1, some c2 =>
             match TypeChecker.unifyTypes tc d1 d2, TypeChecker.unifyTypes tc c1 c2 with
             | Except.ok ud, Except.ok uc => Except.ok (Ty.initFunction ud uc)
             | Except.error e, _ => Except.error e
             | _, Except.error e => Except.error e
           | _, _, _, _ => Except.error TypeTheoryError.UnificationFailure
         | TypeKind.TUPLE, TypeKind.TUPLE =>
           if t1.parameters.length == t2.parameters.length then
             match TypeChecker.unifyTypesList tc t1.parameters t2.parameters with
             | Except.ok us => Except.ok (Ty.initTuple us)
             | Except.error e => Except.error e
           else Except.error TypeTheoryError.UnificationFailure
         | TypeKind.ARRAY, TypeKind.ARRAY =>
           match t1.parameters, t2.parameters with
           | e1 :: _, e2 :: _ =>
             match TypeChecker.unifyTypes tc e1 e2 with
             | Except.ok ue => Except.ok (Ty.initArray ue)
             | Except.error e => Except.error e
           | _, _ => Except.error TypeTheoryError.UnificationFailure
         | TypeKind.UNIVERSE, TypeKind.UNIVERSE =>
           Except.ok (Ty.initUniverse (max t1.universeLevel t2.universeLevel))
         | _, _ =>
           if TypeChecker.subtype tc t1 t2 then Except.ok (Ty.clone t2)
           else if TypeChecker.subtype tc t2 t1 then Except.ok (Ty.clone t1)
           else Except.error TypeTheoryError.UnificationFailure
end

theorem TypeChecker.unifyTypes_same_Nat (tc : TypeChecker) :
  TypeChecker.unifyTypes tc Ty.initNat Ty.initNat = Except.ok Ty.initNat :=
  Eq.refl (Except.ok Ty.initNat)

theorem TypeChecker.unifyTypes_same_Bool (tc : TypeChecker) :
  TypeChecker.unifyTypes tc Ty.initBool Ty.initBool = Except.ok Ty.initBool :=
  Eq.refl (Except.ok Ty.initBool)

structure TypeCheckerStatistics where
  inferenceCount : Nat
  checkCount : Nat
  unificationCount : Nat

def TypeChecker.getStatistics (tc : TypeChecker) : TypeCheckerStatistics :=
  { inferenceCount := tc.inferenceCount, checkCount := tc.unificationCount, unificationCount := tc.unificationCount }

inductive LogicalConnective : Type where
  | CONJUNCTION : LogicalConnective
  | DISJUNCTION : LogicalConnective
  | IMPLICATION : LogicalConnective
  | NEGATION : LogicalConnective
  | UNIVERSAL : LogicalConnective
  | EXISTENTIAL : LogicalConnective
  | TRUE : LogicalConnective
  | FALSE : LogicalConnective
  | BICONDITIONAL : LogicalConnective
  deriving Repr, BEq, DecidableEq

inductive PropAsTy : Type where
  | mk : LogicalConnective → List PropAsTy → Option String → Option Ty → Option Ty → PropAsTy

def PropAsTy.connective : PropAsTy → LogicalConnective
  | mk c _ _ _ _ => c

def PropAsTy.subPropositions : PropAsTy → List PropAsTy
  | mk _ sps _ _ _ => sps

def PropAsTy.boundVariable : PropAsTy → Option String
  | mk _ _ bv _ _ => bv

def PropAsTy.predicateType : PropAsTy → Option Ty
  | mk _ _ _ pt _ => pt

def PropAsTy.correspondingType : PropAsTy → Option Ty
  | mk _ _ _ _ ct => ct

def PropAsTy.init (c : LogicalConnective) : PropAsTy := PropAsTy.mk c [] none none none

def PropAsTy.initTrue : PropAsTy := PropAsTy.mk LogicalConnective.TRUE [] none none (some Ty.initUnit)

def PropAsTy.initFalse : PropAsTy := PropAsTy.mk LogicalConnective.FALSE [] none none (some Ty.initBottom)

def PropAsTy.initConjunction (left : PropAsTy) (right : PropAsTy) : PropAsTy :=
  let ct :=
    match left.correspondingType, right.correspondingType with
    | some lt, some rt => some (Ty.initTuple [lt, rt])
    | _, _ => none
  PropAsTy.mk LogicalConnective.CONJUNCTION [left, right] none none ct

def PropAsTy.initDisjunction (left : PropAsTy) (right : PropAsTy) : PropAsTy :=
  let ct :=
    match left.correspondingType, right.correspondingType with
    | some lt, some rt => some (Ty.initSum lt rt)
    | _, _ => none
  PropAsTy.mk LogicalConnective.DISJUNCTION [left, right] none none ct

def PropAsTy.initImplication (ant : PropAsTy) (cons : PropAsTy) : PropAsTy :=
  let ct :=
    match ant.correspondingType, cons.correspondingType with
    | some at, some ct => some (Ty.initFunction at ct)
    | _, _ => none
  PropAsTy.mk LogicalConnective.IMPLICATION [ant, cons] none none ct

def PropAsTy.initNegation (inner : PropAsTy) : PropAsTy :=
  let ct :=
    match inner.correspondingType with
    | some it => some (Ty.initFunction it Ty.initBottom)
    | none => none
  PropAsTy.mk LogicalConnective.NEGATION [inner] none none ct

def PropAsTy.initUniversal (var : String) (dom : Ty) (body : PropAsTy) : PropAsTy :=
  let ct :=
    match body.correspondingType with
    | some bt =>
      some (Ty.mk TypeKind.DEPENDENT_FUNCTION "" [] [] 0 (some var) (some dom) none (some bt) none)
    | none => none
  PropAsTy.mk LogicalConnective.UNIVERSAL [body] (some var) (some dom) ct

def PropAsTy.initExistential (var : String) (dom : Ty) (body : PropAsTy) : PropAsTy :=
  let ct :=
    match body.correspondingType with
    | some bt =>
      some (Ty.mk TypeKind.DEPENDENT_PAIR "" [] [] 0 (some var) (some dom) none (some bt) none)
    | none => none
  PropAsTy.mk LogicalConnective.EXISTENTIAL [body] (some var) (some dom) ct

def PropAsTy.toType (p : PropAsTy) : Ty :=
  match p.correspondingType with
  | some t => Ty.clone t
  | none => Ty.initUnit

theorem PropAsTy.initTrue_connective : PropAsTy.initTrue.connective = LogicalConnective.TRUE := Eq.refl LogicalConnective.TRUE
theorem PropAsTy.initFalse_connective : PropAsTy.initFalse.connective = LogicalConnective.FALSE := Eq.refl LogicalConnective.FALSE

inductive ProofKind : Type where
  | ASSUMPTION : ProofKind
  | INTRO_CONJUNCTION : ProofKind
  | ELIM_CONJUNCTION_LEFT : ProofKind
  | ELIM_CONJUNCTION_RIGHT : ProofKind
  | INTRO_DISJUNCTION_LEFT : ProofKind
  | INTRO_DISJUNCTION_RIGHT : ProofKind
  | ELIM_DISJUNCTION : ProofKind
  | INTRO_IMPLICATION : ProofKind
  | ELIM_IMPLICATION : ProofKind
  | INTRO_UNIVERSAL : ProofKind
  | ELIM_UNIVERSAL : ProofKind
  | INTRO_EXISTENTIAL : ProofKind
  | ELIM_EXISTENTIAL : ProofKind
  | INTRO_NEGATION : ProofKind
  | ELIM_NEGATION : ProofKind
  | REFLEXIVITY : ProofKind
  | SYMMETRY : ProofKind
  | TRANSITIVITY : ProofKind
  deriving Repr, BEq, DecidableEq

inductive ProofTerm : Type where
  | mk : ProofKind → PropAsTy → List ProofTerm → Option Trm → Bool → ProofTerm

def ProofTerm.kind : ProofTerm → ProofKind
  | mk k _ _ _ _ => k

def ProofTerm.proposition : ProofTerm → PropAsTy
  | mk _ p _ _ _ => p

def ProofTerm.subProofs : ProofTerm → List ProofTerm
  | mk _ _ sps _ _ => sps

def ProofTerm.witnessTerm : ProofTerm → Option Trm
  | mk _ _ _ w _ => w

def ProofTerm.isValid : ProofTerm → Bool
  | mk _ _ _ _ v => v

def ProofTerm.init (k : ProofKind) (prop : PropAsTy) : ProofTerm :=
  ProofTerm.mk k prop [] none false

def ProofTerm.validateConjunctionIntro (pt : ProofTerm) : Bool :=
  match pt.subProofs with
  | p1 :: p2 :: _ => p1.isValid && p2.isValid
  | _ => false

def ProofTerm.validateConjunctionElim (pt : ProofTerm) : Bool :=
  match pt.subProofs with
  | premise :: _ => premise.isValid && premise.proposition.connective == LogicalConnective.CONJUNCTION
  | _ => false

def ProofTerm.validateImplicationIntro (pt : ProofTerm) : Bool :=
  match pt.subProofs with
  | conclusion :: _ => conclusion.isValid
  | _ => false

def ProofTerm.validateImplicationElim (pt : ProofTerm) : Bool :=
  match pt.subProofs with
  | implProof :: _ :: _ => implProof.isValid && implProof.proposition.connective == LogicalConnective.IMPLICATION
  | _ => false

def ProofTerm.validateUniversalIntro (pt : ProofTerm) : Bool :=
  match pt.subProofs with
  | bodyProof :: _ => bodyProof.isValid
  | _ => false

def ProofTerm.validateUniversalElim (pt : ProofTerm) : Bool :=
  match pt.subProofs with
  | univProof :: _ => univProof.isValid && univProof.proposition.connective == LogicalConnective.UNIVERSAL
  | _ => false

def ProofTerm.validate (pt : ProofTerm) : Bool :=
  match pt.kind with
  | ProofKind.ASSUMPTION => true
  | ProofKind.INTRO_CONJUNCTION => pt.validateConjunctionIntro
  | ProofKind.ELIM_CONJUNCTION_LEFT => pt.validateConjunctionElim
  | ProofKind.ELIM_CONJUNCTION_RIGHT => pt.validateConjunctionElim
  | ProofKind.INTRO_IMPLICATION => pt.validateImplicationIntro
  | ProofKind.ELIM_IMPLICATION => pt.validateImplicationElim
  | ProofKind.INTRO_UNIVERSAL => pt.validateUniversalIntro
  | ProofKind.ELIM_UNIVERSAL => pt.validateUniversalElim
  | ProofKind.REFLEXIVITY => true
  | _ => pt.subProofs.length > 0

structure CategoryObject where
  id : Nat
  name : String

def CategoryObject.equals (o1 : CategoryObject) (o2 : CategoryObject) : Bool :=
  o1.id == o2.id && o1.name == o2.name

theorem CategoryObject.equals_self (o : CategoryObject) : CategoryObject.equals o o = true :=
  Eq.refl (o.id == o.id && o.name == o.name) |> Eq.trans (and_self _)

structure Morphism where
  id : Nat
  name : String
  source : CategoryObject
  target : CategoryObject
  isIdentity : Bool

def Morphism.equals (m1 : Morphism) (m2 : Morphism) : Bool :=
  m1.id == m2.id && m1.name == m2.name

def Morphism.canCompose (m1 : Morphism) (m2 : Morphism) : Bool :=
  CategoryObject.equals m1.target m2.source

theorem Morphism.equals_self (m : Morphism) : Morphism.equals m m = true :=
  Eq.refl (m.id == m.id && m.name == m.name) |> Eq.trans (and_self _)

structure Composition where
  fId : Nat
  gId : Nat
  result : Morphism

structure Category where
  name : String
  objects : List CategoryObject
  morphisms : List Morphism
  compositions : List Composition
  nextObjectId : Nat
  nextMorphismId : Nat

def Category.init (name : String) : Category :=
  { name := name, objects := [], morphisms := [], compositions := [], nextObjectId := 1, nextMorphismId := 1 }

def Category.objectCount (c : Category) : Nat := c.objects.length

def Category.morphismCount (c : Category) : Nat := c.morphisms.length

theorem Category.objectCount_init (name : String) : (Category.init name).objectCount = 0 := Eq.refl 0

theorem Category.morphismCount_init (name : String) : (Category.init name).morphismCount = 0 := Eq.refl 0

def Category.addObject (c : Category) (name : String) : Category × CategoryObject :=
  let obj := { id := c.nextObjectId, name := name }
  let identity := { id := c.nextMorphismId, name := "id", source := obj, target := obj, isIdentity := true }
  ({ name := c.name, objects := c.objects ++ [obj], morphisms := c.morphisms ++ [identity], compositions := c.compositions, nextObjectId := c.nextObjectId + 1, nextMorphismId := c.nextMorphismId + 1 }, obj)

def Category.addMorphism (c : Category) (name : String) (source : CategoryObject) (target : CategoryObject) : Category × Morphism :=
  let m := { id := c.nextMorphismId, name := name, source := source, target := target, isIdentity := false }
  ({ name := c.name, objects := c.objects, morphisms := c.morphisms ++ [m], compositions := c.compositions, nextObjectId := c.nextObjectId, nextMorphismId := c.nextMorphismId + 1 }, m)

def Category.findComposition (c : Category) (fId : Nat) (gId : Nat) : Option Morphism :=
  match c.compositions.find? (fun comp => comp.fId == fId && comp.gId == gId) with
  | some comp => some comp.result
  | none => none

def Category.compose (c : Category) (f : Morphism) (g : Morphism) : Except TypeTheoryError (Category × Morphism) :=
  if !Morphism.canCompose f g then Except.error TypeTheoryError.CategoryLawViolation
  else match Category.findComposition c f.id g.id with
       | some cached => Except.ok (c, cached)
       | none =>
         let composedName := g.name ++ "∘" ++ f.name
         let composed := { id := c.nextMorphismId, name := composedName, source := f.source, target := g.target, isIdentity := false }
         let newComp := { fId := f.id, gId := g.id, result := composed }
         let updated := { name := c.name, objects := c.objects, morphisms := c.morphisms ++ [composed], compositions := c.compositions ++ [newComp], nextObjectId := c.nextObjectId, nextMorphismId := c.nextMorphismId + 1 }
         Except.ok (updated, composed)

def Category.getIdentity (c : Category) (obj : CategoryObject) : Option Morphism :=
  c.morphisms.find? (fun m => m.isIdentity && CategoryObject.equals m.source obj)

def Category.verifyIdentityLaw (c : Category) (f : Morphism) : Bool :=
  match Category.getIdentity c f.source, Category.getIdentity c f.target with
  | some _, some _ => true
  | _, _ => false

theorem Category.addObject_count (c : Category) (name : String) :
  (Category.addObject c name).1.objectCount = c.objectCount + 1 :=
  Eq.refl (c.objects.length + 1)

theorem Category.addMorphism_count (c : Category) (name : String) (s t : CategoryObject) :
  (Category.addMorphism c name s t).1.morphismCount = c.morphismCount + 1 :=
  Eq.refl (c.morphisms.length + 1)

structure ObjMapping where
  sourceId : Nat
  target : CategoryObject

structure MorphMapping where
  sourceId : Nat
  target : Morphism

structure Functor where
  name : String
  sourceCategory : Category
  targetCategory : Category
  objectMapping : List ObjMapping
  morphismMapping : List MorphMapping

def Functor.init (name : String) (source : Category) (target : Category) : Functor :=
  { name := name, sourceCategory := source, targetCategory := target, objectMapping := [], morphismMapping := [] }

def Functor.mapObject (f : Functor) (src : CategoryObject) (tgt : CategoryObject) : Functor :=
  { f with objectMapping := f.objectMapping ++ [{ sourceId := src.id, target := tgt }] }

def Functor.mapMorphism (f : Functor) (src : Morphism) (tgt : Morphism) : Functor :=
  { f with morphismMapping := f.morphismMapping ++ [{ sourceId := src.id, target := tgt }] }

def Functor.applyToObject (f : Functor) (obj : CategoryObject) : Option CategoryObject :=
  match f.objectMapping.find? (fun m => m.sourceId == obj.id) with
  | some m => some m.target
  | none => none

def Functor.applyToMorphism (f : Functor) (m : Morphism) : Option Morphism :=
  match f.morphismMapping.find? (fun mm => mm.sourceId == m.id) with
  | some mm => some mm.target
  | none => none

theorem Functor.applyToObject_self (f : Functor) (obj : CategoryObject) (tgt : CategoryObject) :
  Functor.applyToObject (Functor.mapObject f obj tgt) obj = some tgt :=
  Eq.refl (some tgt)

structure NaturalTransformation where
  name : String
  sourceFunctor : Functor
  targetFunctor : Functor
  components : List (Nat × Morphism)

def NaturalTransformation.init (name : String) (source : Functor) (target : Functor) : NaturalTransformation :=
  { name := name, sourceFunctor := source, targetFunctor := target, components := [] }

def NaturalTransformation.setComponent (nt : NaturalTransformation) (obj : CategoryObject) (comp : Morphism) : NaturalTransformation :=
  { nt with components := nt.components ++ [(obj.id, comp)] }

def NaturalTransformation.getComponent (nt : NaturalTransformation) (obj : CategoryObject) : Option Morphism :=
  match nt.components.find? (fun (id, _) => id == obj.id) with
  | some (_, m) => some m
  | none => none

structure Monad where
  name : String
  endofunctor : Functor
  unit : NaturalTransformation
  multiplication : NaturalTransformation

def Monad.init (name : String) (t : Functor) (eta : NaturalTransformation) (mu : NaturalTransformation) : Monad :=
  { name := name, endofunctor := t, unit := eta, multiplication := mu }

structure CartesianClosedCategory where
  baseCategory : Category
  terminalObject : Option CategoryObject
  hasProducts : Bool
  hasExponentials : Bool

def CartesianClosedCategory.init (base : Category) : CartesianClosedCategory :=
  { baseCategory := base, terminalObject := none, hasProducts := false, hasExponentials := false }

def CartesianClosedCategory.isCartesianClosed (ccc : CartesianClosedCategory) : Bool :=
  ccc.terminalObject.isSome && ccc.hasProducts && ccc.hasExponentials

theorem CartesianClosedCategory.isCartesianClosed_init (c : Category) :
  (CartesianClosedCategory.init c).isCartesianClosed = false := Eq.refl false

inductive LinearityMode : Type where
  | LINEAR : LinearityMode
  | AFFINE : LinearityMode
  | RELEVANT : LinearityMode
  | UNRESTRICTED : LinearityMode
  deriving Repr, BEq, DecidableEq

def LinearityMode.toString : LinearityMode → String
  | LINEAR => "linear"
  | AFFINE => "affine"
  | RELEVANT => "relevant"
  | UNRESTRICTED => "unrestricted"

theorem LinearityMode.toString_LINEAR : LinearityMode.toString LinearityMode.LINEAR = "linear" := Eq.refl "linear"
theorem LinearityMode.toString_AFFINE : LinearityMode.toString LinearityMode.AFFINE = "affine" := Eq.refl "affine"
theorem LinearityMode.toString_RELEVANT : LinearityMode.toString LinearityMode.RELEVANT = "relevant" := Eq.refl "relevant"
theorem LinearityMode.toString_UNRESTRICTED : LinearityMode.toString LinearityMode.UNRESTRICTED = "unrestricted" := Eq.refl "unrestricted"

def LinearityMode.canWeakenTo : LinearityMode → LinearityMode → Bool
  | UNRESTRICTED, _ => true
  | AFFINE, AFFINE => true
  | AFFINE, UNRESTRICTED => true
  | RELEVANT, RELEVANT => true
  | RELEVANT, UNRESTRICTED => true
  | LINEAR, LINEAR => true
  | _, _ => false

def LinearityMode.join : LinearityMode → LinearityMode → LinearityMode
  | LINEAR, _ => LINEAR
  | _, LINEAR => LINEAR
  | AFFINE, RELEVANT => LINEAR
  | RELEVANT, AFFINE => LINEAR
  | AFFINE, AFFINE => AFFINE
  | AFFINE, UNRESTRICTED => AFFINE
  | UNRESTRICTED, AFFINE => AFFINE
  | RELEVANT, RELEVANT => RELEVANT
  | RELEVANT, UNRESTRICTED => RELEVANT
  | UNRESTRICTED, RELEVANT => RELEVANT
  | UNRESTRICTED, UNRESTRICTED => UNRESTRICTED

theorem LinearityMode.join_linear_left (m : LinearityMode) : LinearityMode.join LinearityMode.LINEAR m = LinearityMode.LINEAR := Eq.refl LinearityMode.LINEAR
theorem LinearityMode.join_linear_right (m : LinearityMode) : LinearityMode.join m LinearityMode.LINEAR = LinearityMode.LINEAR :=
  match m with
  | LINEAR => Eq.refl LinearityMode.LINEAR
  | AFFINE => Eq.refl LinearityMode.LINEAR
  | RELEVANT => Eq.refl LinearityMode.LINEAR
  | UNRESTRICTED => Eq.refl LinearityMode.LINEAR

structure LinearTy where
  baseType : Ty
  linearity : LinearityMode

def LinearTy.init (base : Ty) (lin : LinearityMode) : LinearTy :=
  { baseType := base, linearity := lin }

def LinearTy.initLinear (base : Ty) : LinearTy := LinearTy.init base LinearityMode.LINEAR
def LinearTy.initAffine (base : Ty) : LinearTy := LinearTy.init base LinearityMode.AFFINE
def LinearTy.initRelevant (base : Ty) : LinearTy := LinearTy.init base LinearityMode.RELEVANT
def LinearTy.initUnrestricted (base : Ty) : LinearTy := LinearTy.init base LinearityMode.UNRESTRICTED

def LinearTy.mustUseExactlyOnce (lt : LinearTy) : Bool := lt.linearity == LinearityMode.LINEAR
def LinearTy.canDrop (lt : LinearTy) : Bool := lt.linearity == LinearityMode.AFFINE || lt.linearity == LinearityMode.UNRESTRICTED
def LinearTy.canDuplicate (lt : LinearTy) : Bool := lt.linearity == LinearityMode.RELEVANT || lt.linearity == LinearityMode.UNRESTRICTED

theorem LinearTy.mustUseExactlyOnce_linear (base : Ty) : (LinearTy.initLinear base).mustUseExactlyOnce = true := Eq.refl true
theorem LinearTy.mustUseExactlyOnce_affine (base : Ty) : (LinearTy.initAffine base).mustUseExactlyOnce = false := Eq.refl false
theorem LinearTy.canDrop_affine (base : Ty) : (LinearTy.initAffine base).canDrop = true := Eq.refl true
theorem LinearTy.canDrop_linear (base : Ty) : (LinearTy.initLinear base).canDrop = false := Eq.refl false
theorem LinearTy.canDuplicate_unrestricted (base : Ty) : (LinearTy.initUnrestricted base).canDuplicate = true := Eq.refl true
theorem LinearTy.canDuplicate_linear (base : Ty) : (LinearTy.initLinear base).canDuplicate = false := Eq.refl false

structure ResourceUsage where
  variableName : String
  usageCount : Nat
  linearType : LinearTy

def ResourceUsage.init (name : String) (lt : LinearTy) : ResourceUsage :=
  { variableName := name, usageCount := 0, linearType := lt }

def ResourceUsage.use (ru : ResourceUsage) : ResourceUsage :=
  { ru with usageCount := ru.usageCount + 1 }

def ResourceUsage.isValid (ru : ResourceUsage) : Bool :=
  match ru.linearType.linearity with
  | LinearityMode.LINEAR => ru.usageCount == 1
  | LinearityMode.AFFINE => ru.usageCount <= 1
  | LinearityMode.RELEVANT => ru.usageCount >= 1
  | LinearityMode.UNRESTRICTED => true

theorem ResourceUsage.isValid_linear_one (name : String) (lt : LinearTy) (h : lt.linearity = LinearityMode.LINEAR) :
  ResourceUsage.isValid { variableName := name, usageCount := 1, linearType := lt } = true :=
  Eq.refl (1 == 1) |> congrArg (fun x => x == 1) (Eq.refl 1) |> Eq.trans (beq_self_eq_true 1)

theorem ResourceUsage.isValid_linear_zero (name : String) (lt : LinearTy) (h : lt.linearity = LinearityMode.LINEAR) :
  ResourceUsage.isValid { variableName := name, usageCount := 0, linearType := lt } = false :=
  Eq.refl (0 == 1)

theorem ResourceUsage.isValid_affine_zero (name : String) (lt : LinearTy) (h : lt.linearity = LinearityMode.AFFINE) :
  ResourceUsage.isValid { variableName := name, usageCount := 0, linearType := lt } = true :=
  Eq.refl (0 <= 1)

theorem ResourceUsage.isValid_affine_one (name : String) (lt : LinearTy) (h : lt.linearity = LinearityMode.AFFINE) :
  ResourceUsage.isValid { variableName := name, usageCount := 1, linearType := lt } = true :=
  Eq.refl (1 <= 1)

theorem ResourceUsage.isValid_affine_two (name : String) (lt : LinearTy) (h : lt.linearity = LinearityMode.AFFINE) :
  ResourceUsage.isValid { variableName := name, usageCount := 2, linearType := lt } = false :=
  Eq.refl (2 <= 1)

theorem ResourceUsage.isValid_unrestricted (name : String) (lt : LinearTy) (h : lt.linearity = LinearityMode.UNRESTRICTED) (n : Nat) :
  ResourceUsage.isValid { variableName := name, usageCount := n, linearType := lt } = true :=
  Eq.refl true

inductive LinearityViolationType : Type where
  | UNUSED : LinearityViolationType
  | OVERUSED : LinearityViolationType
  | DROPPED : LinearityViolationType
  | DUPLICATED : LinearityViolationType
  deriving Repr, BEq

structure LinearityViolation where
  variableName : String
  expectedUsage : LinearityMode
  actualCount : Nat
  violationType : LinearityViolationType

structure LinearTypeChecker where
  resources : List (String × ResourceUsage)
  violationLog : List LinearityViolation
  checkCount : Nat
  violationCount : Nat

def LinearTypeChecker.init : LinearTypeChecker :=
  { resources := [], violationLog := [], checkCount := 0, violationCount := 0 }

def LinearTypeChecker.introduce (ltc : LinearTypeChecker) (name : String) (lt : LinearTy) : LinearTypeChecker :=
  { ltc with resources := ltc.resources ++ [(name, ResourceUsage.init name lt)] }

def LinearTypeChecker.use (ltc : LinearTypeChecker) (name : String) : LinearTypeChecker :=
  { ltc with resources := ltc.resources.map (fun (n, ru) => if n == name then (n, ResourceUsage.use ru) else (n, ru)) }

def LinearTypeChecker.validateAll (ltc : LinearTypeChecker) : LinearTypeChecker × Bool :=
  let violations := ltc.resources.filterMap (fun (n, ru) =>
    if !ResourceUsage.isValid ru then
      some { variableName := n, expectedUsage := ru.linearType.linearity, actualCount := ru.usageCount,
             violationType := if ru.usageCount == 0 then LinearityViolationType.UNUSED
                              else if ru.usageCount > 1 then LinearityViolationType.OVERUSED
                              else LinearityViolationType.DROPPED }
    else none)
  let newViolCount := ltc.violationCount + violations.length
  ({ ltc with violationLog := ltc.violationLog ++ violations, violationCount := newViolCount }, violations.isEmpty)

def LinearTypeChecker.checkTerm (ltc : LinearTypeChecker) (term : Trm) : LinearTypeChecker × Bool :=
  let ltc1 := { ltc with checkCount := ltc.checkCount + 1 }
  match term.kind with
  | TermKind.VARIABLE => LinearTypeChecker.validateAll (LinearTypeChecker.use ltc1 term.name)
  | TermKind.LAMBDA =>
    match term.subTerms with
    | body :: _ => LinearTypeChecker.checkTerm ltc1 body
    | [] => LinearTypeChecker.validateAll ltc1
  | TermKind.APPLICATION =>
    let ltc2 := List.foldl (fun acc sub => (LinearTypeChecker.checkTerm acc sub).1) ltc1 term.subTerms
    LinearTypeChecker.validateAll ltc2
  | TermKind.PAIR =>
    let ltc2 := List.foldl (fun acc sub => (LinearTypeChecker.checkTerm acc sub).1) ltc1 term.subTerms
    LinearTypeChecker.validateAll ltc2
  | _ => LinearTypeChecker.validateAll ltc1

theorem LinearTypeChecker.introduce_count (ltc : LinearTypeChecker) (name : String) (lt : LinearTy) :
  (LinearTypeChecker.introduce ltc name lt).resources.length = ltc.resources.length + 1 :=
  Eq.refl (ltc.resources.length + 1)

structure LinearCheckerStatistics where
  checkCount : Nat
  violationCount : Nat
  activeResources : Nat

def LinearTypeChecker.getStatistics (ltc : LinearTypeChecker) : LinearCheckerStatistics :=
  { checkCount := ltc.checkCount, violationCount := ltc.violationCount, activeResources := ltc.resources.length }

inductive TypeProofKind : Type where
  | TYPE_JUDGMENT : TypeProofKind
  | SUBTYPING : TypeProofKind
  | EQUALITY : TypeProofKind
  | LINEAR_USAGE : TypeProofKind
  | FUNCTOR_LAW : TypeProofKind
  | MONAD_LAW : TypeProofKind
  | NATURALITY : TypeProofKind
  | UNIVERSE_MEMBERSHIP : TypeProofKind
  deriving Repr, BEq

def TypeProofKind.toString : TypeProofKind → String
  | TYPE_JUDGMENT => "type_judgment"
  | SUBTYPING => "subtyping"
  | EQUALITY => "equality"
  | LINEAR_USAGE => "linear_usage"
  | FUNCTOR_LAW => "functor_law"
  | MONAD_LAW => "monad_law"
  | NATURALITY => "naturality"
  | UNIVERSE_MEMBERSHIP => "universe_membership"

theorem TypeProofKind.toString_TYPE_JUDGMENT : TypeProofKind.toString TypeProofKind.TYPE_JUDGMENT = "type_judgment" := Eq.refl "type_judgment"
theorem TypeProofKind.toString_SUBTYPING : TypeProofKind.toString TypeProofKind.SUBTYPING = "subtyping" := Eq.refl "subtyping"
theorem TypeProofKind.toString_EQUALITY : TypeProofKind.toString TypeProofKind.EQUALITY = "equality" := Eq.refl "equality"
theorem TypeProofKind.toString_LINEAR_USAGE : TypeProofKind.toString TypeProofKind.LINEAR_USAGE = "linear_usage" := Eq.refl "linear_usage"
theorem TypeProofKind.toString_FUNCTOR_LAW : TypeProofKind.toString TypeProofKind.FUNCTOR_LAW = "functor_law" := Eq.refl "functor_law"
theorem TypeProofKind.toString_MONAD_LAW : TypeProofKind.toString TypeProofKind.MONAD_LAW = "monad_law" := Eq.refl "monad_law"
theorem TypeProofKind.toString_NATURALITY : TypeProofKind.toString TypeProofKind.NATURALITY = "naturality" := Eq.refl "naturality"
theorem TypeProofKind.toString_UNIVERSE_MEMBERSHIP : TypeProofKind.toString TypeProofKind.UNIVERSE_MEMBERSHIP = "universe_membership" := Eq.refl "universe_membership"

structure TypeProof where
  proofType : TypeProofKind
  judgment : Option TypeJudgment
  subType : Option Ty
  superType : Option Ty
  proofTerm : Option Trm
  isValid : Bool
  derivationSteps : List String

def TypeProof.init (pt : TypeProofKind) : TypeProof :=
  { proofType := pt, judgment := none, subType := none, superType := none, proofTerm := none, isValid := false, derivationSteps := [] }

def TypeProof.addStep (p : TypeProof) (step : String) : TypeProof :=
  { p with derivationSteps := p.derivationSteps ++ [step] }

def TypeProof.validate (p : TypeProof) : Bool :=
  match p.proofType with
  | TypeProofKind.TYPE_JUDGMENT =>
    match p.judgment with
    | some j => TypeJudgment.validate j
    | none => false
  | TypeProofKind.SUBTYPING => p.subType.isSome && p.superType.isSome
  | TypeProofKind.EQUALITY =>
    match p.subType, p.superType with
    | some t1, some t2 => Ty.equals t1 t2
    | _, _ => false
  | _ => p.derivationSteps.length > 0

theorem TypeProof.validate_subtyping_some (t1 t2 : Ty) :
  TypeProof.validate { proofType := TypeProofKind.SUBTYPING, judgment := none, subType := some t1, superType := some t2, proofTerm := none, isValid := false, derivationSteps := [] } = true :=
  Eq.refl true

theorem TypeProof.validate_subtyping_none :
  TypeProof.validate { proofType := TypeProofKind.SUBTYPING, judgment := none, subType := none, superType := none, proofTerm := none, isValid := false, derivationSteps := [] } = false :=
  Eq.refl false

structure ProofResult where
  success : Bool
  proof : Option TypeProof
  errorMessage : Option String
  ownsProof : Bool

def ProofResult.initSuccess (p : TypeProof) : ProofResult :=
  { success := true, proof := some p, errorMessage := none, ownsProof := false }

def ProofResult.initFailure (msg : String) : ProofResult :=
  { success := false, proof := none, errorMessage := some msg, ownsProof := false }

def ProofResult.initWithOwnedProof (p : TypeProof) : ProofResult :=
  { success := true, proof := some p, errorMessage := none, ownsProof := true }

structure TypeTheoryEngine where
  typeChecker : TypeChecker
  linearChecker : LinearTypeChecker
  categories : List Category
  functors : List Functor
  monads : List Monad
  proofs : List TypeProof
  proofCount : Nat

def TypeTheoryEngine.init : TypeTheoryEngine :=
  { typeChecker := TypeChecker.init, linearChecker := LinearTypeChecker.init, categories := [], functors := [], monads := [], proofs := [], proofCount := 0 }

def TypeTheoryEngine.createCategory (engine : TypeTheoryEngine) (name : String) : TypeTheoryEngine × Category :=
  let cat := Category.init name
  ({ engine with categories := engine.categories ++ [cat] }, cat)

def TypeTheoryEngine.createFunctor (engine : TypeTheoryEngine) (name : String) (source : Category) (target : Category) : TypeTheoryEngine × Functor :=
  let f := Functor.init name source target
  ({ engine with functors := engine.functors ++ [f] }, f)

def TypeTheoryEngine.createMonad (engine : TypeTheoryEngine) (name : String) (t : Functor) (eta : NaturalTransformation) (mu : NaturalTransformation) : TypeTheoryEngine × Monad :=
  let m := Monad.init name t eta mu
  ({ engine with monads := engine.monads ++ [m] }, m)

def TypeTheoryEngine.proveTypeJudgment (engine : TypeTheoryEngine) (ctx : TypeContext) (term : Trm) (expectedType : Ty) : TypeTheoryEngine × ProofResult :=
  let engine1 := { engine with proofCount := engine.proofCount + 1 }
  let proof := TypeProof.init TypeProofKind.TYPE_JUDGMENT
  let proof1 := TypeProof.addStep proof "Begin type judgment proof"
  match TypeChecker.inferType engine.typeChecker ctx term with
  | Except.ok inferred =>
    let proof2 := TypeProof.addStep proof1 "Inferred type from term"
    if TypeChecker.subtype engine.typeChecker inferred expectedType then
      let proof3 := TypeProof.addStep proof2 "Subtyping check passed"
      let judgment := TypeJudgment.init ctx term expectedType
      let finalProof := { proof3 with judgment := some judgment, isValid := true }
      ({ engine1 with proofs := engine1.proofs ++ [finalProof] }, ProofResult.initSuccess finalProof)
    else
      let proof3 := TypeProof.addStep proof2 "Subtyping check failed"
      (engine1, ProofResult.initFailure "Type mismatch")
  | Except.error e =>
    let proof2 := TypeProof.addStep proof1 "Type inference failed"
    (engine1, ProofResult.initFailure (match e with | TypeTheoryError.TypeMismatch => "TypeMismatch" | TypeTheoryError.UnificationFailure => "UnificationFailure" | TypeTheoryError.LinearityViolation => "LinearityViolation" | TypeTheoryError.InvalidTypeConstruction => "InvalidTypeConstruction" | TypeTheoryError.VariableNotInContext => "VariableNotInContext" | TypeTheoryError.InvalidApplication => "InvalidApplication" | TypeTheoryError.InvalidProjection => "InvalidProjection" | TypeTheoryError.CategoryLawViolation => "CategoryLawViolation" | TypeTheoryError.OutOfMemory => "OutOfMemory" | TypeTheoryError.InvalidIdentityElimination => "InvalidIdentityElimination"))

def TypeTheoryEngine.proveSubtyping (engine : TypeTheoryEngine) (sub : Ty) (sup : Ty) : TypeTheoryEngine × ProofResult :=
  let engine1 := { engine with proofCount := engine.proofCount + 1 }
  let proof := TypeProof.init TypeProofKind.SUBTYPING
  let proof1 := TypeProof.addStep proof "Begin subtyping proof"
  let proof2 := { proof1 with subType := some (Ty.clone sub), superType := some (Ty.clone sup) }
  if TypeChecker.subtype engine.typeChecker sub sup then
    let proof3 := TypeProof.addStep proof2 "Subtyping relation verified"
    let finalProof := { proof3 with isValid := true }
    ({ engine1 with proofs := engine1.proofs ++ [finalProof] }, ProofResult.initSuccess finalProof)
  else
    let proof3 := TypeProof.addStep proof2 "Subtyping relation failed"
    (engine1, ProofResult.initFailure "No subtyping relation exists")

def TypeTheoryEngine.proveEquality (engine : TypeTheoryEngine) (t1 : Ty) (t2 : Ty) : TypeTheoryEngine × ProofResult :=
  let engine1 := { engine with proofCount := engine.proofCount + 1 }
  let proof := TypeProof.init TypeProofKind.EQUALITY
  let proof1 := TypeProof.addStep proof "Begin equality proof"
  let proof2 := { proof1 with subType := some (Ty.clone t1), superType := some (Ty.clone t2) }
  if Ty.equals t1 t2 then
    let proof3 := TypeProof.addStep proof2 "Types are definitionally equal"
    let finalProof := { proof3 with isValid := true }
    ({ engine1 with proofs := engine1.proofs ++ [finalProof] }, ProofResult.initSuccess finalProof)
  else
    match TypeChecker.unifyTypes engine.typeChecker t1 t2 with
    | Except.ok _ =>
      let proof3 := TypeProof.addStep proof2 "Types unified via type unification"
      let finalProof := { proof3 with isValid := true }
      ({ engine1 with proofs := engine1.proofs ++ [finalProof] }, ProofResult.initSuccess finalProof)
    | Except.error _ =>
      let proof3 := TypeProof.addStep proof2 "Unification failed"
      (engine1, ProofResult.initFailure "Types are not equal")

def TypeTheoryEngine.checkLinearUsage (engine : TypeTheoryEngine) (term : Trm) : TypeTheoryEngine × ProofResult :=
  let engine1 := { engine with proofCount := engine.proofCount + 1 }
  let proof := TypeProof.init TypeProofKind.LINEAR_USAGE
  let proof1 := TypeProof.addStep proof "Begin linear usage check"
  let (checker, valid) := LinearTypeChecker.checkTerm engine.linearChecker term
  let engine2 := { engine1 with linearChecker := checker }
  if valid then
    let proof2 := TypeProof.addStep proof1 "All linear resources used correctly"
    let finalProof := { proof2 with isValid := true }
    ({ engine2 with proofs := engine2.proofs ++ [finalProof] }, ProofResult.initSuccess finalProof)
  else
    let proof2 := TypeProof.addStep proof1 "Linear usage violation detected"
    (engine2, ProofResult.initFailure "Linear usage violation")

structure TypeTheoryStatistics where
  typeCheckerStats : TypeCheckerStatistics
  linearCheckerStats : LinearCheckerStatistics
  proofCount : Nat
  categoryCount : Nat
  functorCount : Nat
  monadCount : Nat

def TypeTheoryEngine.getStatistics (engine : TypeTheoryEngine) : TypeTheoryStatistics :=
  { typeCheckerStats := TypeChecker.getStatistics engine.typeChecker,
    linearCheckerStats := LinearTypeChecker.getStatistics engine.linearChecker,
    proofCount := engine.proofCount,
    categoryCount := engine.categories.length,
    functorCount := engine.functors.length,
    monadCount := engine.monads.length }

theorem TypeTheoryEngine.init_proofCount : TypeTheoryEngine.init.proofCount = 0 := Eq.refl 0
theorem TypeTheoryEngine.init_categoryCount : TypeTheoryEngine.init.categories.length = 0 := Eq.refl 0
theorem TypeTheoryEngine.init_functorCount : TypeTheoryEngine.init.functors.length = 0 := Eq.refl 0
theorem TypeTheoryEngine.init_monadCount : TypeTheoryEngine.init.monads.length = 0 := Eq.refl 0

theorem TypeTheoryEngine.createCategory_count (e : TypeTheoryEngine) (name : String) :
  (TypeTheoryEngine.createCategory e name).1.categories.length = e.categories.length + 1 :=
  Eq.refl (e.categories.length + 1)

theorem TypeTheoryEngine.proveSubtyping_nat_int_success (e : TypeTheoryEngine) :
  (TypeTheoryEngine.proveSubtyping e Ty.initNat Ty.initInt).2.success = true :=
  Eq.refl true

theorem TypeTheoryEngine.proveEquality_nat_success (e : TypeTheoryEngine) :
  (TypeTheoryEngine.proveEquality e Ty.initNat Ty.initNat).2.success = true :=
  Eq.refl true

theorem TypeTheoryEngine.proveEquality_nat_bool_failure (e : TypeTheoryEngine) :
  (TypeTheoryEngine.proveEquality e Ty.initNat Ty.initBool).2.success = false :=
  if h : Ty.equals Ty.initNat Ty.initBool then absurd (Eq.refl true) (Bool.false_ne_true)
  else Eq.refl false |> congrArg (fun x => if x then true else false) (if_neg h) |> congrArg (fun x => if x then x else false) (if_neg (fun h => h (beq_self_eq_true TypeKind.TOP))) |> congrArg (fun x => if x then x else false) (if_neg (fun h => h (beq_self_eq_true TypeKind.BOTTOM))) |> congrArg (fun x => if x then x else false) (if_neg (fun h => h (beq_self_eq_true TypeKind.VARIABLE))) |> congrArg (fun x => if x then x else false) (if_neg (fun h => h (beq_self_eq_true TypeKind.VARIABLE))) |> congrArg (fun x => if x then x else false) (if_neg (fun h => h (beq_self_eq_true TypeKind.TOP))) |> congrArg (fun x => if x then x else false) (if_neg (fun h => h (beq_self_eq_true TypeKind.TOP))) |> congrArg (fun x => if x then x else false) (if_neg (fun h => h (beq_self_eq_true TypeKind.BOTTOM))) |> congrArg (fun x => if x then x else false) (if_neg (fun h => h (beq_self_eq_true TypeKind.BOTTOM))) |> congrArg (fun x => if x then x else false) (if_neg (fun h => h (beq_self_eq_true TypeKind.FUNCTION))) |> congrArg (fun x => if x then x else false) (if_neg (fun h => h (beq_self_eq_true TypeKind.FUNCTION))) |> congrArg (fun x => if x then x else false) (if_neg (fun h => h (beq_self_eq_true TypeKind.TUPLE))) |> congrArg (fun x => if x then x else false) (if_neg (fun h => h (beq_self_eq_true TypeKind.TUPLE))) |> congrArg (fun x => if x then x else false) (if_neg (fun h => h (beq_self_eq_true TypeKind.ARRAY))) |> congrArg (fun x => if x then x else false) (if_neg (fun h => h (beq_self_eq_true TypeKind.ARRAY))) |> congrArg (fun x => if x then x else false) (if_neg (fun h => h (beq_self_eq_true TypeKind.UNIVERSE))) |> congrArg (fun x => if x then x else false) (if_neg (fun h => h (beq_self_eq_true TypeKind.UNIVERSE))) |> congrArg (fun x => if x then x else false) (if_neg TypeChecker.not_subtype_int_nat) |> congrArg (fun x => if x then x else false) (if_neg (fun h => h (beq_self_eq_true TypeKind.NAT)))
