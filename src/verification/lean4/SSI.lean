import Std

namespace SSIFormal

abbrev U32 := UInt32
abbrev U64 := UInt64

structure F32Word where
  bits : U32
deriving Repr, DecidableEq

abbrev Score := F32Word

structure Similarity where
  numerator : Nat
  denominator : Nat
deriving Repr, DecidableEq

structure Segment where
  tokens : List U32
  position : U64
  score : Score
  anchorHash : U64
deriving Repr, DecidableEq

inductive CollisionNode where
| nil
| link : Segment → CollisionNode → CollisionNode
deriving Repr, DecidableEq

inductive Node where
| leaf : U64 → Option Segment → CollisionNode → Node
| branch : U64 → List (Option Node) → Nat → Node
deriving Repr, DecidableEq

structure SSI where
  root : Option Node
  height : Nat
  size : Nat
  maxHeight : Nat
deriving Repr, DecidableEq

structure InsertResult where
  node : Node
  inserted : Bool
deriving Repr, DecidableEq

structure ChildInsertResult where
  root : Node
  inserted : Bool
deriving Repr, DecidableEq

structure UpdateResult where
  node : Node
  changed : Bool
deriving Repr, DecidableEq

structure ChildUpdateResult where
  root : Node
  changed : Bool
deriving Repr, DecidableEq


def c1 : U64 := 11400714819323198485

def c2 : U64 := 5871781006564002453


def mixHash (state value : U64) : U64 :=
  state * c1 + value + c2


def hashFold : U64 → List U64 → U64
| state, [] => state
| state, x :: xs => hashFold (mixHash state x) xs


def wordOfU32 (x : U32) : U64 :=
  UInt64.ofNat x.toNat


def wordOfNat (x : Nat) : U64 :=
  UInt64.ofNat x


def scoreWord (s : Score) : U64 :=
  UInt64.ofNat s.bits.toNat


def Segment.init (tokens : List U32) (position : U64) (score : Score) (anchorHash : U64) : Segment :=
  { tokens := tokens, position := position, score := score, anchorHash := anchorHash }


def Segment.tokenWords (s : Segment) : List U64 :=
  List.map wordOfU32 s.tokens


def Segment.tokenHash (s : Segment) : U64 :=
  hashFold (wordOfNat s.tokens.length) s.tokenWords


def Segment.fullHash (s : Segment) : U64 :=
  hashFold 0 (s.position :: scoreWord s.score :: s.anchorHash :: wordOfNat s.tokens.length :: s.tokenWords)


def Segment.setScore (s : Segment) (score : Score) : Segment :=
  { s with score := score }


def Segment.setAnchor (s : Segment) (anchorHash : U64) : Segment :=
  { s with anchorHash := anchorHash }


theorem Segment.init_tokens : ∀ tokens position score anchorHash, (Segment.init tokens position score anchorHash).tokens = tokens
| tokens, position, score, anchorHash => Eq.refl


theorem Segment.init_position : ∀ tokens position score anchorHash, (Segment.init tokens position score anchorHash).position = position
| tokens, position, score, anchorHash => Eq.refl


theorem Segment.init_score : ∀ tokens position score anchorHash, (Segment.init tokens position score anchorHash).score = score
| tokens, position, score, anchorHash => Eq.refl


theorem Segment.init_anchor : ∀ tokens position score anchorHash, (Segment.init tokens position score anchorHash).anchorHash = anchorHash
| tokens, position, score, anchorHash => Eq.refl


theorem Segment.tokenHash_det : ∀ s : Segment, s.tokenHash = s.tokenHash
| s => Eq.refl


theorem Segment.fullHash_det : ∀ s : Segment, s.fullHash = s.fullHash
| s => Eq.refl


theorem Segment.setScore_tokens : ∀ s score, (s.setScore score).tokens = s.tokens
| s, score => Eq.refl


theorem Segment.setScore_position : ∀ s score, (s.setScore score).position = s.position
| s, score => Eq.refl


theorem Segment.setScore_anchor : ∀ s score, (s.setScore score).anchorHash = s.anchorHash
| s, score => Eq.refl


theorem Segment.setScore_score : ∀ s score, (s.setScore score).score = score
| s, score => Eq.refl


theorem Segment.setAnchor_tokens : ∀ s anchorHash, (s.setAnchor anchorHash).tokens = s.tokens
| s, anchorHash => Eq.refl


theorem Segment.setAnchor_position : ∀ s anchorHash, (s.setAnchor anchorHash).position = s.position
| s, anchorHash => Eq.refl


theorem Segment.setAnchor_score : ∀ s anchorHash, (s.setAnchor anchorHash).score = s.score
| s, anchorHash => Eq.refl


theorem Segment.setAnchor_anchor : ∀ s anchorHash, (s.setAnchor anchorHash).anchorHash = anchorHash
| s, anchorHash => Eq.refl


def CollisionNode.length : CollisionNode → Nat
| CollisionNode.nil => 0
| CollisionNode.link _ next => Nat.succ (CollisionNode.length next)


def CollisionNode.positions : CollisionNode → List U64
| CollisionNode.nil => []
| CollisionNode.link seg next => seg.position :: CollisionNode.positions next


def CollisionNode.fullHash : CollisionNode → U64
| CollisionNode.nil => 0
| CollisionNode.link seg next => seg.fullHash + CollisionNode.fullHash next


def CollisionNode.find : CollisionNode → U64 → Option Segment
| CollisionNode.nil, position => none
| CollisionNode.link seg next, position => if seg.position = position then some seg else CollisionNode.find next position


def CollisionNode.update : CollisionNode → U64 → Score → CollisionNode
| CollisionNode.nil, position, score => CollisionNode.nil
| CollisionNode.link seg next, position, score =>
  if seg.position = position then
    CollisionNode.link (seg.setScore score) next
  else
    CollisionNode.link seg (CollisionNode.update next position score)


def CollisionNode.insertOrReplace : CollisionNode → Segment → CollisionNode × Bool
| CollisionNode.nil, seg => (CollisionNode.link seg CollisionNode.nil, true)
| CollisionNode.link head next, seg =>
  if head.position = seg.position then
    (CollisionNode.link seg next, false)
  else
    let rest := CollisionNode.insertOrReplace next seg
    (CollisionNode.link head rest.fst, rest.snd)


theorem CollisionNode.length_det : ∀ c : CollisionNode, c.length = c.length
| c => Eq.refl


theorem CollisionNode.hash_det : ∀ c : CollisionNode, c.fullHash = c.fullHash
| c => Eq.refl


theorem CollisionNode.positions_det : ∀ c : CollisionNode, c.positions = c.positions
| c => Eq.refl


theorem CollisionNode.length_update : ∀ c position score, (CollisionNode.update c position score).length = c.length
| CollisionNode.nil, position, score => Eq.refl
| CollisionNode.link seg next, position, score =>
  if h : seg.position = position then
    Eq.refl
  else
    congrArg Nat.succ (CollisionNode.length_update next position score)


theorem CollisionNode.positions_update : ∀ c position score, (CollisionNode.update c position score).positions = c.positions
| CollisionNode.nil, position, score => Eq.refl
| CollisionNode.link seg next, position, score =>
  if h : seg.position = position then
    Eq.refl
  else
    congrArg (List.cons seg.position) (CollisionNode.positions_update next position score)


theorem CollisionNode.find_update_head : ∀ seg next score, CollisionNode.find (CollisionNode.update (CollisionNode.link seg next) seg.position score) seg.position = some (seg.setScore score)
| seg, next, score =>
  if h : seg.position = seg.position then
    Eq.refl
  else
    False.elim (h Eq.refl)


def lowBit (n : Nat) : Nat :=
  n % 2


def shiftBit (n : Nat) : Nat :=
  n / 2


def bitDiff (a b : Nat) : Nat :=
  if lowBit a = lowBit b then 0 else 1


def bitDistanceAux : Nat → Nat → Nat → Nat
| 0, a, b => 0
| Nat.succ k, a, b => bitDiff a b + bitDistanceAux k (shiftBit a) (shiftBit b)


def bitDistance (a b : U64) : Nat :=
  bitDistanceAux 64 a.toNat b.toNat


def similarityBits (a b : U64) : Nat :=
  64 - bitDistance a b


def computeSimilarity (a b : U64) : Similarity :=
  { numerator := similarityBits a b, denominator := 64 }


theorem bitDiff_self : ∀ a, bitDiff a a = 0
| a =>
  if h : lowBit a = lowBit a then
    Eq.refl
  else
    False.elim (h Eq.refl)


theorem bitDistanceAux_self : ∀ n a, bitDistanceAux n a a = 0
| 0, a => Eq.refl
| Nat.succ k, a =>
  Eq.trans
    (congrArg (fun t => bitDiff a a + t) (bitDistanceAux_self k (shiftBit a)))
    (Eq.trans
      (congrArg (fun t => t + 0) (bitDiff_self a))
      Eq.refl)


theorem similarityBits_self : ∀ a, similarityBits a a = 64
| a =>
  Eq.trans
    (congrArg (fun t => 64 - t) (bitDistanceAux_self 64 a.toNat))
    Eq.refl


theorem computeSimilarity_det : ∀ a b, computeSimilarity a b = computeSimilarity a b
| a, b => Eq.refl


theorem computeSimilarity_self : ∀ a, computeSimilarity a a = { numerator := 64, denominator := 64 }
| a =>
  Eq.trans
    (congrArg (fun t => { numerator := t, denominator := 64 }) (similarityBits_self a))
    Eq.refl


def bucketCount : Nat := 64


def bucketIndex (position : U64) : Nat :=
  position.toNat % bucketCount


def getAt {α : Type} : List α → Nat → Option α
| [], idx => none
| x :: xs, 0 => some x
| x :: xs, Nat.succ idx => getAt xs idx


def replaceAt {α : Type} : List α → Nat → α → List α
| [], idx, value => []
| x :: xs, 0, value => value :: xs
| x :: xs, Nat.succ idx, value => x :: replaceAt xs idx value


theorem replaceAt_length {α : Type} : ∀ xs idx value, (replaceAt xs idx value).length = xs.length
| [], idx, value => Eq.refl
| x :: xs, 0, value => Eq.refl
| x :: xs, Nat.succ idx, value => congrArg Nat.succ (replaceAt_length xs idx value)


theorem getAt_replaceAt_hit {α : Type} : ∀ xs idx value old, getAt xs idx = some old → getAt (replaceAt xs idx value) idx = some value
| [], idx, value, old, h => nomatch h
| x :: xs, 0, value, old, h => Eq.refl
| x :: xs, Nat.succ idx, value, old, h => getAt_replaceAt_hit xs idx value old h


theorem replicate_length {α : Type} : ∀ n value, (List.replicate n value).length = n
| 0, value => Eq.refl
| Nat.succ n, value => congrArg Nat.succ (replicate_length n value)


theorem bucketCount_pos : 0 < bucketCount :=
  Nat.succ_le_succ
    (Nat.succ_le_succ
      (Nat.succ_le_succ
        (Nat.succ_le_succ
          (Nat.succ_le_succ
            (Nat.succ_le_succ
              (Nat.succ_le_succ
                (Nat.succ_le_succ
                  (Nat.succ_le_succ
                    (Nat.succ_le_succ
                      (Nat.succ_le_succ
                        (Nat.succ_le_succ
                          (Nat.succ_le_succ
                            (Nat.succ_le_succ
                              (Nat.succ_le_succ
                                (Nat.succ_le_succ
                                  (Nat.succ_le_succ
                                    (Nat.succ_le_succ
                                      (Nat.succ_le_succ
                                        (Nat.succ_le_succ
                                          (Nat.succ_le_succ
                                            (Nat.succ_le_succ
                                              (Nat.succ_le_succ
                                                (Nat.succ_le_succ
                                                  (Nat.succ_le_succ
                                                    (Nat.succ_le_succ
                                                      (Nat.succ_le_succ
                                                        (Nat.succ_le_succ
                                                          (Nat.succ_le_succ
                                                            (Nat.succ_le_succ
                                                              (Nat.succ_le_succ
                                                                (Nat.succ_le_succ
                                                                  (Nat.succ_le_succ
                                                                    (Nat.succ_le_succ
                                                                      (Nat.succ_le_succ
                                                                        (Nat.succ_le_succ
                                                                          (Nat.succ_le_succ
                                                                            (Nat.succ_le_succ
                                                                              (Nat.succ_le_succ
                                                                                (Nat.succ_le_succ
                                                                                  (Nat.succ_le_succ
                                                                                    (Nat.succ_le_succ
                                                                                      (Nat.succ_le_succ
                                                                                        (Nat.succ_le_succ
                                                                                          (Nat.succ_le_succ
                                                                                            (Nat.succ_le_succ
                                                                                              (Nat.succ_le_succ
                                                                                                (Nat.succ_le_succ
                                                                                                  (Nat.succ_le_succ
                                                                                                    (Nat.succ_le_succ
                                                                                                      (Nat.succ_le_succ
                                                                                                        (Nat.succ_le_succ
                                                                                                          (Nat.succ_le_succ
                                                                                                            (Nat.succ_le_succ
                                                                                                              (Nat.succ_le_succ
                                                                                                                (Nat.succ_le_succ
                                                                                                                  (Nat.succ_le_succ
                                                                                                                    (Nat.succ_le_succ
                                                                                                                      (Nat.succ_le_succ
                                                                                                                        (Nat.succ_le_succ
                                                                                                                          (Nat.zero_le 0))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))


theorem bucketIndex_lt : ∀ position, bucketIndex position < bucketCount
| position => Nat.mod_lt position.toNat bucketCount_pos


theorem getAt_replicate_none : ∀ n idx, idx < n → getAt (List.replicate n (none : Option Node)) idx = some none
| 0, idx, h => False.elim (Nat.not_lt_zero idx h)
| Nat.succ n, 0, h => Eq.refl
| Nat.succ n, Nat.succ idx, h => getAt_replicate_none n idx (Nat.lt_of_succ_lt_succ h)


def Node.hash : Node → U64
| Node.leaf h head chain => h
| Node.branch h children height => h


def Node.height : Node → Nat
| Node.leaf h head chain => 0
| Node.branch h children height => height


def Node.isLeaf : Node → Bool
| Node.leaf h head chain => true
| Node.branch h children height => false


def leafAggregateHash (head : Option Segment) (chain : CollisionNode) : U64 :=
  match head with
  | none => chain.fullHash
  | some seg => seg.fullHash + chain.fullHash


def branchAggregateHash (children : List (Option Node)) : U64 :=
  List.foldl (fun acc child =>
    match child with
    | none => acc
    | some node => acc + node.hash) 0 children


def Node.rehash : Node → Node
| Node.leaf h head chain => Node.leaf (leafAggregateHash head chain) head chain
| Node.branch h children height => Node.branch (branchAggregateHash children) children height


def Node.Structural : Node → Prop
| Node.leaf h head chain => True
| Node.branch h children height => height = 6 ∧ children.length = bucketCount


def Node.HashOk : Node → Prop
| Node.leaf h head chain => h = leafAggregateHash head chain
| Node.branch h children height => h = branchAggregateHash children


def Node.Valid (n : Node) : Prop :=
  n.Structural ∧ n.HashOk


def emptyChildren : List (Option Node) :=
  List.replicate bucketCount none


def Node.emptyRoot : Node :=
  Node.branch 0 emptyChildren 6


def Node.findInLeaf : Node → U64 → Option Segment
| Node.leaf h none chain, position => CollisionNode.find chain position
| Node.leaf h (some head) chain, position => if head.position = position then some head else CollisionNode.find chain position
| Node.branch h children height, position => none


def Node.insertIntoLeaf : Node → Segment → InsertResult
| Node.leaf h none chain, seg => { node := Node.rehash (Node.leaf 0 (some seg) chain), inserted := true }
| Node.leaf h (some head) chain, seg =>
  if head.position = seg.position then
    { node := Node.rehash (Node.leaf 0 (some seg) chain), inserted := false }
  else
    let rest := CollisionNode.insertOrReplace chain seg
    { node := Node.rehash (Node.leaf 0 (some head) rest.fst), inserted := rest.snd }
| Node.branch h children height, seg => { node := Node.branch h children height, inserted := false }


def Node.updateLeaf : Node → U64 → Score → UpdateResult
| Node.leaf h none chain, position, score => { node := Node.rehash (Node.leaf 0 none (CollisionNode.update chain position score)), changed := CollisionNode.find chain position != none }
| Node.leaf h (some head) chain, position, score =>
  if head.position = position then
    { node := Node.rehash (Node.leaf 0 (some (head.setScore score)) chain), changed := true }
  else
    { node := Node.rehash (Node.leaf 0 (some head) (CollisionNode.update chain position score)), changed := CollisionNode.find chain position != none }
| Node.branch h children height, position, score => { node := Node.branch h children height, changed := false }


def Node.setChild : Node → Nat → Node → Node
| Node.branch h children height, idx, child => Node.rehash (Node.branch 0 (replaceAt children idx (some child)) height)
| node, idx, child => node


theorem Node.hash_det : ∀ n : Node, n.hash = n.hash
| n => Eq.refl


theorem Node.height_det : ∀ n : Node, n.height = n.height
| n => Eq.refl


theorem Node.rehash_hashOk : ∀ n, (Node.rehash n).HashOk
| Node.leaf h head chain => Eq.refl
| Node.branch h children height => Eq.refl


theorem Node.rehash_structural : ∀ n, n.Structural → (Node.rehash n).Structural
| Node.leaf h head chain, hs => hs
| Node.branch h children height, hs => hs


theorem Node.rehash_valid : ∀ n, n.Structural → (Node.rehash n).Valid
| n, hs => And.intro (Node.rehash_structural n hs) (Node.rehash_hashOk n)


theorem Node.emptyRoot_structural : Node.emptyRoot.Structural :=
  And.intro Eq.refl (replicate_length bucketCount none)


theorem Node.emptyRoot_valid : Node.emptyRoot.Valid :=
  And.intro Node.emptyRoot_structural Eq.refl


theorem Node.insertIntoLeaf_isLeaf : ∀ n seg, (Node.insertIntoLeaf n seg).node.isLeaf = n.isLeaf
| Node.leaf h none chain, seg => Eq.refl
| Node.leaf h (some head) chain, seg =>
  if hEq : head.position = seg.position then
    Eq.refl
  else
    Eq.refl
| Node.branch h children height, seg => Eq.refl


theorem Node.insertIntoLeaf_height : ∀ n seg, (Node.insertIntoLeaf n seg).node.height = n.height
| Node.leaf h none chain, seg => Eq.refl
| Node.leaf h (some head) chain, seg =>
  if hEq : head.position = seg.position then
    Eq.refl
  else
    Eq.refl
| Node.branch h children height, seg => Eq.refl


theorem Node.insertIntoLeaf_valid_empty : ∀ seg chain, ((Node.insertIntoLeaf (Node.leaf 0 none chain) seg).node).Valid
| seg, chain => Node.rehash_valid (Node.leaf 0 (some seg) chain) True.intro


theorem Node.findInLeaf_insert_hit_empty : ∀ seg chain, Node.findInLeaf (Node.insertIntoLeaf (Node.leaf 0 none chain) seg).node seg.position = some seg
| seg, chain =>
  if h : seg.position = seg.position then
    Eq.refl
  else
    False.elim (h Eq.refl)


theorem Node.updateLeaf_head_hit : ∀ head chain score, Node.findInLeaf (Node.updateLeaf (Node.leaf 0 (some head) chain) head.position score).node head.position = some (head.setScore score)
| head, chain, score =>
  if h : head.position = head.position then
    Eq.refl
  else
    False.elim (h Eq.refl)


theorem Node.setChild_structural : ∀ root idx child, root.Structural → child.Valid → (Node.setChild root idx child).Structural
| Node.leaf h head chain, idx, child, hs, hc => hs
| Node.branch h children height, idx, child, hs, hc =>
  match hs with
  | And.intro h1 h2 =>
    And.intro h1 (Eq.trans (replaceAt_length children idx (some child)) h2)


def SSI.init : SSI :=
  { root := none, height := 0, size := 0, maxHeight := 6 }


def SSI.ensureRoot (s : SSI) : SSI :=
  match s.root with
  | none => { s with root := some Node.emptyRoot, height := 6 }
  | some root => s


def SSI.Structural (s : SSI) : Prop :=
  s.maxHeight = 6 ∧
  match s.root with
  | none => s.height = 0
  | some root => root.Structural ∧ s.height = 6


def SSI.HashOk (s : SSI) : Prop :=
  match s.root with
  | none => True
  | some root => root.HashOk


def SSI.Valid (s : SSI) : Prop :=
  s.Structural ∧ s.HashOk


def SSI.findChild (root : Node) (position : U64) : Option Node :=
  match root with
  | Node.branch h children height =>
    match getAt children (bucketIndex position) with
    | some (some child) => some child
    | some none => none
    | none => none
  | Node.leaf h head chain => none


def SSI.getSegment (s : SSI) (position : U64) : Option Segment :=
  match s.root with
  | none => none
  | some root =>
    match SSI.findChild root position with
    | none => none
    | some leaf => Node.findInLeaf leaf position


def SSI.insertRoot (root : Node) (seg : Segment) : ChildInsertResult :=
  match root with
  | Node.branch h children height =>
    let idx := bucketIndex seg.position
    let current :=
      match getAt children idx with
      | some (some child) => child
      | some none => Node.leaf 0 none CollisionNode.nil
      | none => Node.leaf 0 none CollisionNode.nil
    let inserted := Node.insertIntoLeaf current seg
    { root := Node.setChild (Node.branch h children height) idx inserted.node, inserted := inserted.inserted }
  | Node.leaf h head chain => { root := root, inserted := false }


def SSI.addSequence (s : SSI) (tokens : List U32) (position : U64) (score : Score) (anchorHash : U64) : SSI :=
  let s1 := SSI.ensureRoot s
  let seg := Segment.init tokens position score anchorHash
  match s1.root with
  | none => s1
  | some root =>
    let inserted := SSI.insertRoot root seg
    { s1 with root := some inserted.root, size := s1.size + if inserted.inserted then 1 else 0 }


def SSI.updateScore (s : SSI) (position : U64) (score : Score) : SSI :=
  match s.root with
  | none => s
  | some root =>
    match root with
    | Node.branch h children height =>
      let idx := bucketIndex position
      match getAt children idx with
      | some (some child) =>
        let updated := Node.updateLeaf child position score
        { s with root := some (Node.setChild root idx updated.node) }
      | some none => s
      | none => s
    | Node.leaf h head chain => s


theorem SSI.init_root : SSI.init.root = none := Eq.refl


theorem SSI.init_height : SSI.init.height = 0 := Eq.refl


theorem SSI.init_size : SSI.init.size = 0 := Eq.refl


theorem SSI.init_maxHeight : SSI.init.maxHeight = 6 := Eq.refl


theorem SSI.init_structural : SSI.init.Structural :=
  And.intro Eq.refl Eq.refl


theorem SSI.init_valid : SSI.init.Valid :=
  And.intro SSI.init_structural True.intro


theorem SSI.ensureRoot_init_root : (SSI.ensureRoot SSI.init).root = some Node.emptyRoot := Eq.refl


theorem SSI.ensureRoot_init_height : (SSI.ensureRoot SSI.init).height = 6 := Eq.refl


theorem SSI.ensureRoot_init_valid : (SSI.ensureRoot SSI.init).Valid :=
  And.intro
    (And.intro Eq.refl (And.intro Node.emptyRoot_structural Eq.refl))
    Node.emptyRoot_valid.snd


theorem SSI.getSegment_init_none : ∀ position, SSI.getSegment SSI.init position = none
| position => Eq.refl


theorem SSI.getSegment_add_empty_hit : ∀ tokens position score anchorHash,
  SSI.getSegment (SSI.addSequence SSI.init tokens position score anchorHash) position = some (Segment.init tokens position score anchorHash)
| tokens, position, score, anchorHash =>
  let seg : Segment := Segment.init tokens position score anchorHash
  let idx : Nat := bucketIndex position
  let leaf : Node := (Node.insertIntoLeaf (Node.leaf 0 none CollisionNode.nil) seg).node
  let root1 : Node := Node.setChild Node.emptyRoot idx leaf
  let h1 : getAt emptyChildren idx = some none :=
    getAt_replicate_none bucketCount idx (bucketIndex_lt position)
  let h2 : getAt (replaceAt emptyChildren idx (some leaf)) idx = some (some leaf) :=
    getAt_replaceAt_hit emptyChildren idx (some leaf) none h1
  let h3 : SSI.findChild root1 position = some leaf :=
    h2
  let h4 : match SSI.findChild root1 position with | none => none | some child => Node.findInLeaf child position = Node.findInLeaf leaf position :=
    congrArg (fun x => match x with | none => none | some child => Node.findInLeaf child position) h3
  let h5 : Node.findInLeaf leaf position = some seg :=
    Node.findInLeaf_insert_hit_empty seg CollisionNode.nil
  show match SSI.findChild root1 position with | none => none | some child => Node.findInLeaf child position = some seg from
    Eq.trans h4 h5


theorem SSI.updateScore_empty_id : ∀ position score, SSI.updateScore SSI.init position score = SSI.init
| position, score => Eq.refl


theorem SSI.updateScore_add_empty_hit : ∀ tokens position score1 anchorHash score2,
  SSI.getSegment (SSI.updateScore (SSI.addSequence SSI.init tokens position score1 anchorHash) position score2) position =
    some ((Segment.init tokens position score1 anchorHash).setScore score2)
| tokens, position, score1, anchorHash, score2 =>
  let seg1 : Segment := Segment.init tokens position score1 anchorHash
  let seg2 : Segment := seg1.setScore score2
  let idx : Nat := bucketIndex position
  let leaf1 : Node := (Node.insertIntoLeaf (Node.leaf 0 none CollisionNode.nil) seg1).node
  let leaf2 : Node := (Node.updateLeaf leaf1 position score2).node
  let root1 : Node := Node.setChild Node.emptyRoot idx leaf2
  let h1 : getAt emptyChildren idx = some none :=
    getAt_replicate_none bucketCount idx (bucketIndex_lt position)
  let h2 : getAt (replaceAt emptyChildren idx (some leaf2)) idx = some (some leaf2) :=
    getAt_replaceAt_hit emptyChildren idx (some leaf2) none h1
  let h3 : SSI.findChild root1 position = some leaf2 :=
    h2
  let h4 : match SSI.findChild root1 position with | none => none | some child => Node.findInLeaf child position = Node.findInLeaf leaf2 position :=
    congrArg (fun x => match x with | none => none | some child => Node.findInLeaf child position) h3
  let h5 : Node.findInLeaf leaf2 position = some seg2 :=
    Node.updateLeaf_head_hit seg1 CollisionNode.nil score2
  show match SSI.findChild root1 position with | none => none | some child => Node.findInLeaf child position = some seg2 from
    Eq.trans h4 h5

end SSIFormal
