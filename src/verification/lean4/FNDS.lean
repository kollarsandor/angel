def Key := List UInt8
def Value := List UInt8

def wyhash : List UInt8 → Nat
  | [] => 0
  | x :: xs => x.toNat + 31 * wyhash xs

theorem wyhash_nil : wyhash [] = 0 := Eq.refl 0
theorem wyhash_cons (x : UInt8) (xs : List UInt8) : wyhash (x :: xs) = x.toNat + 31 * wyhash xs := Eq.refl _

def list_get {α} : List α → Nat → Option α
  | [], _ => none
  | x :: _, 0 => some x
  | _ :: xs, Nat.succ n => list_get xs n

def list_set {α} : List α → Nat → α → List α
  | [], _, _ => []
  | _ :: xs, 0, v => v :: xs
  | x :: xs, Nat.succ n, v => x :: list_set xs n v

theorem list_get_nil {α} (n : Nat) : list_get ([] : List α) n = none :=
  match n with
  | 0 => Eq.refl none
  | Nat.succ _ => Eq.refl none

theorem list_set_nil {α} (n : Nat) (v : α) : list_set ([] : List α) n v = [] :=
  match n with
  | 0 => Eq.refl []
  | Nat.succ _ => Eq.refl []

theorem list_get_zero {α} (x : α) (xs : List α) : list_get (x :: xs) 0 = some x := Eq.refl _
theorem list_set_zero {α} (x : α) (xs : List α) (v : α) : list_set (x :: xs) 0 v = v :: xs := Eq.refl _

theorem list_get_succ {α} (x : α) (xs : List α) (n : Nat) : list_get (x :: xs) (Nat.succ n) = list_get xs n := Eq.refl _
theorem list_set_succ {α} (x : α) (xs : List α) (n : Nat) (v : α) : list_set (x :: xs) (Nat.succ n) v = x :: list_set xs n v := Eq.refl _

theorem list_length_set {α} : ∀ (l : List α) (i : Nat) (v : α), (list_set l i v).length = l.length
  | [], i, v => match i with | 0 => Eq.refl 0 | Nat.succ _ => Eq.refl 0
  | x :: xs, 0, v => Eq.refl _
  | x :: xs, Nat.succ n, v => congrArg Nat.succ (list_length_set xs n v)

theorem list_get_set_eq {α} : ∀ (l : List α) (i : Nat) (v : α), i < l.length → list_get (list_set l i v) i = some v
  | [], i, v, h => False.elim (Nat.not_lt_zero i h)
  | x :: xs, 0, v, h => Eq.refl (some v)
  | x :: xs, Nat.succ n, v, h => list_get_set_eq xs n v (Nat.lt_of_succ_lt_succ h)

theorem list_get_some_of_lt {α} : ∀ (l : List α) (i : Nat), i < l.length → ∃ x, list_get l i = some x
  | [], i, h => False.elim (Nat.not_lt_zero i h)
  | x :: xs, 0, h => Exists.intro x (Eq.refl _)
  | x :: xs, Nat.succ i, h => list_get_some_of_lt xs i (Nat.lt_of_succ_lt_succ h)

theorem list_length_append {α} : ∀ (xs ys : List α), (xs ++ ys).length = xs.length + ys.length
  | [], ys => Eq.refl _
  | x :: xs, ys => congrArg Nat.succ (list_length_append xs ys)

def list_decEq (l1 l2 : List UInt8) : Decidable (l1 = l2) :=
  match l1, l2 with
  | [], [] => isTrue (Eq.refl [])
  | [], y :: ys => isFalse (fun h => List.noConfusion h)
  | x :: xs, [] => isFalse (fun h => List.noConfusion h)
  | x :: xs, y :: ys =>
    match decEq x y with
    | isTrue hxy =>
      match list_decEq xs ys with
      | isTrue hxsys => isTrue (Eq.subst (motive := fun a => x :: xs = a :: ys) hxy (Eq.subst (motive := fun b => x :: xs = x :: b) hxsys (Eq.refl _)))
      | isFalse hneq => isFalse (fun h => List.noConfusion h (fun _ htail => hneq htail))
    | isFalse hneq => isFalse (fun h => List.noConfusion h (fun hhead _ => hneq hhead))

def remove_key : List (Key × Value) → Key → List (Key × Value)
  | [], _ => []
  | (k', v') :: xs, k =>
    match list_decEq k k' with
    | isTrue _ => remove_key xs k
    | isFalse _ => (k', v') :: remove_key xs k

theorem remove_key_nil (k : Key) : remove_key [] k = [] := Eq.refl []

theorem remove_key_length_le : ∀ (l : List (Key × Value)) (k : Key), (remove_key l k).length ≤ l.length
  | [], k => Nat.le_refl 0
  | (k', v') :: xs, k =>
    match list_decEq k k' with
    | isTrue _ => Nat.le_trans (remove_key_length_le xs k) (Nat.le_step (Nat.le_refl _))
    | isFalse _ => Nat.succ_le_succ (remove_key_length_le xs k)

def splitLast {α} : List α → Option (List α × α)
  | [] => none
  | x :: xs =>
    match splitLast xs with
    | none => some ([], x)
    | some (front, last) => some (x :: front, last)

theorem splitLast_nil {α} : splitLast ([] : List α) = none := Eq.refl none

theorem splitLast_singleton {α} (x : α) : splitLast [x] = some ([], x) := Eq.refl _

theorem splitLast_eq_some {α} (front : List α) (last : α) : splitLast (front ++ [last]) = some (front, last) :=
  List.recOn front
    (Eq.refl (some ([], last)))
    (fun h t ih =>
      let motive := fun (res : Option (List α × α)) =>
        (match res with
         | none => some ([], h)
         | some (f, l) => some (h :: f, l)) = some (h :: t, last)
      have h_base : motive (some (t, last)) := Eq.refl _
      Eq.subst (motive := motive) (Eq.symm ih) h_base)

structure Entry where
  key : Key
  value : Value
  next_index : Option Nat

structure CoalescedHashMap where
  capacity : Nat
  cellar_start : Nat
  cellar_next : Nat
  size : Nat
  buckets : List (Option Entry)

def init_buckets (n : Nat) : List (Option Entry) :=
  match n with
  | 0 => []
  | Nat.succ n' => none :: init_buckets n'

theorem init_buckets_length (n : Nat) : (init_buckets n).length = n :=
  Nat.recOn n (Eq.refl 0) (fun n' ih => congrArg Nat.succ ih)

def CoalescedHashMap.init (capacity : Nat) : CoalescedHashMap :=
  let cellar_size := capacity / 7
  let cellar_start := capacity - cellar_size
  { capacity := capacity,
    cellar_start := cellar_start,
    cellar_next := cellar_start,
    size := 0,
    buckets := init_buckets capacity }

def get_chain (fuel : Nat) (buckets : List (Option Entry)) (idx : Nat) (k : Key) : Option Value :=
  match fuel with
  | 0 => none
  | Nat.succ f =>
    match list_get buckets idx with
    | some (some entry) =>
      match list_decEq k entry.key with
      | isTrue _ => some entry.value
      | isFalse _ =>
        match entry.next_index with
        | none => none
        | some nxt => get_chain f buckets nxt k
    | _ => none

def CoalescedHashMap.get (m : CoalescedHashMap) (k : Key) : Option Value :=
  let idx := if m.cellar_start = 0 then 0 else wyhash k % m.cellar_start
  get_chain m.capacity m.buckets idx k

def find_empty_slot (fuel : Nat) (curr : Nat) (buckets : List (Option Entry)) : Option Nat :=
  match fuel with
  | 0 => none
  | Nat.succ f =>
    match list_get buckets curr with
    | some none => some curr
    | _ =>
      match curr with
      | 0 => none
      | Nat.succ c => find_empty_slot f c buckets

def put_helper_empty (m : CoalescedHashMap) (idx : Nat) (k : Key) (v : Value) : CoalescedHashMap :=
  let new_node := { key := k, value := v, next_index := none : Entry }
  { m with buckets := list_set m.buckets idx (some new_node), size := m.size + 1 }

def put_helper_collision (m : CoalescedHashMap) (idx : Nat) (k : Key) (v : Value) (old_entry : Entry) : CoalescedHashMap :=
  match find_empty_slot m.capacity m.cellar_next m.buckets with
  | none =>
    let new_entry := { key := k, value := v, next_index := old_entry.next_index : Entry }
    { m with buckets := list_set m.buckets idx (some new_entry), size := m.size }
  | some empty_slot =>
    let buckets1 := list_set m.buckets empty_slot (some old_entry)
    let new_entry := { key := k, value := v, next_index := some empty_slot : Entry }
    let buckets2 := list_set buckets1 idx (some new_entry)
    { m with buckets := buckets2, size := m.size + 1, cellar_next := empty_slot + 1 }

def CoalescedHashMap.put (m : CoalescedHashMap) (k : Key) (v : Value) : CoalescedHashMap :=
  let idx := if m.cellar_start = 0 then 0 else wyhash k % m.cellar_start
  match list_get m.buckets idx with
  | some none => put_helper_empty m idx k v
  | some (some old_entry) => put_helper_collision m idx k v old_entry
  | none => put_helper_empty m idx k v

theorem get_chain_head_match (f : Nat) (buckets : List (Option Entry)) (idx : Nat) (k : Key) (v : Value) (nxt : Option Nat)
  (h_get : list_get buckets idx = some (some { key := k, value := v, next_index := nxt })) :
  get_chain (Nat.succ f) buckets idx k = some v :=
  have h_match1 : (match list_get buckets idx with
                   | some (some entry) => match list_decEq k entry.key with
                                          | isTrue _ => some entry.value
                                          | isFalse _ => match entry.next_index with | none => none | some n => get_chain f buckets n k
                   | _ => none) =
                  (match some (some { key := k, value := v, next_index := nxt } : Option Entry) with
                   | some (some entry) => match list_decEq k entry.key with
                                          | isTrue _ => some entry.value
                                          | isFalse _ => match entry.next_index with | none => none | some n => get_chain f buckets n k
                   | _ => none) := congrArg (fun x => match x with | some (some entry) => match list_decEq k entry.key with | isTrue _ => some entry.value | isFalse _ => match entry.next_index with | none => none | some n => get_chain f buckets n k | _ => none) h_get
  have h_eval1 : (match some (some { key := k, value := v, next_index := nxt } : Option Entry) with
                   | some (some entry) => match list_decEq k entry.key with
                                          | isTrue _ => some entry.value
                                          | isFalse _ => match entry.next_index with | none => none | some n => get_chain f buckets n k
                   | _ => none) =
                 (match list_decEq k k with
                  | isTrue _ => some v
                  | isFalse _ => match nxt with | none => none | some n => get_chain f buckets n k) := Eq.refl _
  have h_eval2 : (match list_decEq k k with
                  | isTrue _ => some v
                  | isFalse _ => match nxt with | none => none | some n => get_chain f buckets n k) = some v :=
    match h_dec : list_decEq k k with
    | isTrue _ => Eq.refl _
    | isFalse h_neq => False.elim (h_neq (Eq.refl k))
  Eq.trans h_match1 (Eq.trans h_eval1 h_eval2)

theorem CoalescedHashMap.get_put_eq (m : CoalescedHashMap) (k : Key) (v : Value)
  (h_idx : (if m.cellar_start = 0 then 0 else wyhash k % m.cellar_start) < m.buckets.length)
  (h_cap : m.capacity > 0) :
  (m.put k v).get k = some v :=
  let idx := if m.cellar_start = 0 then 0 else wyhash k % m.cellar_start
  have h_get_chain : ∀ buckets nxt, list_get buckets idx = some (some { key := k, value := v, next_index := nxt : Entry }) → get_chain m.capacity buckets idx k = some v :=
    fun buckets nxt h_get =>
      match m.capacity, h_cap with
      | 0, h => False.elim (Nat.lt_irrefl 0 h)
      | Nat.succ f, _ => get_chain_head_match f buckets idx k v nxt h_get

  match h_get_bucket : list_get m.buckets idx with
  | some none =>
    have h_set : list_get (list_set m.buckets idx (some (some { key := k, value := v, next_index := none : Entry }))) idx = some (some { key := k, value := v, next_index := none : Entry }) := list_get_set_eq m.buckets idx _ h_idx
    h_get_chain _ none h_set
  | some (some old_entry) =>
    match h_empty : find_empty_slot m.capacity m.cellar_next m.buckets with
    | none =>
      have h_set : list_get (list_set m.buckets idx (some (some { key := k, value := v, next_index := old_entry.next_index : Entry }))) idx = some (some { key := k, value := v, next_index := old_entry.next_index : Entry }) := list_get_set_eq m.buckets idx _ h_idx
      h_get_chain _ old_entry.next_index h_set
    | some empty_slot =>
      let buckets1 := list_set m.buckets empty_slot (some old_entry)
      have h_len1 : buckets1.length = m.buckets.length := list_length_set m.buckets empty_slot _
      have h_idx1 : idx < buckets1.length := Eq.subst (motive := fun len => idx < len) (Eq.symm h_len1) h_idx
      let buckets2 := list_set buckets1 idx (some (some { key := k, value := v, next_index := some empty_slot : Entry }))
      have h_set : list_get buckets2 idx = some (some { key := k, value := v, next_index := some empty_slot : Entry }) := list_get_set_eq buckets1 idx _ h_idx1
      h_get_chain _ (some empty_slot) h_set
  | none =>
    have h_contra : list_get m.buckets idx = none → False :=
      fun h_none =>
        let ⟨val, h_some⟩ := list_get_some_of_lt m.buckets idx h_idx
        have h_eq : some val = none := Eq.trans (Eq.symm h_some) h_none
        Option.noConfusion h_eq
    False.elim (h_contra h_get_bucket)

structure LRUCache where
  capacity : Nat
  max_memory : Nat
  queue : List Key
  map : List (Key × Value)
  current_size : Nat
  current_memory : Nat
  hits : Nat
  misses : Nat
  evictions : Nat

def IsLeastRecentlyUsed (k : Key) (c : LRUCache) : Prop :=
  ∃ front, c.queue = front ++ [k]

def LRUCache.init (capacity max_memory : Nat) : LRUCache :=
  { capacity := capacity,
    max_memory := max_memory,
    queue := [],
    map := [],
    current_size := 0,
    current_memory := 0,
    hits := 0,
    misses := 0,
    evictions := 0 }

def evict_helper (c : LRUCache) (split_res : Option (List Key × Key)) : LRUCache :=
  match split_res with
  | none => c
  | some (front, last_k) =>
    { c with queue := front,
             map := remove_key c.map last_k,
             evictions := c.evictions + 1,
             current_size := c.current_size - 1 }

def LRUCache.evict (c : LRUCache) : LRUCache :=
  evict_helper c (splitLast c.queue)

theorem evict_removes_lru (c : LRUCache) (front : List Key) (k : Key) (h : c.queue = front ++ [k]) : (c.evict).queue = front :=
  have h_split : splitLast (front ++ [k]) = some (front, k) := splitLast_eq_some front k
  have h_queue : splitLast c.queue = some (front, k) := Eq.subst (motive := fun q => splitLast q = some (front, k)) (Eq.symm h) h_split
  have h_subst : evict_helper c (splitLast c.queue) = evict_helper c (some (front, k)) := congrArg (evict_helper c) h_queue
  congrArg LRUCache.queue h_subst

def remove_from_queue (k : Key) : List Key → List Key
  | [] => []
  | x :: xs =>
    match list_decEq k x with
    | isTrue _ => remove_from_queue k xs
    | isFalse _ => x :: remove_from_queue k xs

def LRUCache.put (c : LRUCache) (k : Key) (v : Value) : LRUCache :=
  let new_queue := k :: remove_from_queue k c.queue
  let new_map := (k, v) :: remove_key c.map k
  let new_size := if c.current_size < c.capacity then c.current_size + 1 else c.capacity
  let c_updated := { c with queue := new_queue, map := new_map, current_size := new_size }
  if c_updated.current_size > c.capacity then c_updated.evict else c_updated

structure FractalNodeData where
  id : Key
  data : Value
  weight : Nat
  scale : Nat
  children_count : Nat

inductive FractalLevel where
  | mk : Nat → Nat → List FractalNodeData → List FractalLevel → FractalLevel

def FractalLevel.level : FractalLevel → Nat
  | mk l _ _ _ => l

def FractalLevel.nodes : FractalLevel → List FractalNodeData
  | mk _ _ ns _ => ns

def FractalLevel.children : FractalLevel → List FractalLevel
  | mk _ _ _ cs => cs

mutual
def getTotalNodeCount : FractalLevel → Nat
  | FractalLevel.mk _ _ nodes children => nodes.length + sumTotalNodeCount children

def sumTotalNodeCount : List FractalLevel → Nat
  | [] => 0
  | c :: cs => getTotalNodeCount c + sumTotalNodeCount cs
end

mutual
def traversePreOrder : FractalLevel → List FractalNodeData
  | FractalLevel.mk _ _ nodes children => nodes ++ concatPreOrder children

def concatPreOrder : List FractalLevel → List FractalNodeData
  | [] => []
  | c :: cs => traversePreOrder c ++ concatPreOrder cs
end

mutual
def traversePostOrder : FractalLevel → List FractalNodeData
  | FractalLevel.mk _ _ nodes children => concatPostOrder children ++ nodes

def concatPostOrder : List FractalLevel → List FractalNodeData
  | [] => []
  | c :: cs => traversePostOrder c ++ concatPostOrder cs
end

mutual
def length_traversePreOrder (lvl : FractalLevel) : (traversePreOrder lvl).length = getTotalNodeCount lvl :=
  match lvl with
  | FractalLevel.mk l sf nodes children =>
    have h1 : (nodes ++ concatPreOrder children).length = nodes.length + (concatPreOrder children).length := list_length_append nodes (concatPreOrder children)
    have h2 : (concatPreOrder children).length = sumTotalNodeCount children := length_concatPreOrder children
    Eq.trans h1 (congrArg (Nat.add nodes.length) h2)

def length_concatPreOrder (cs : List FractalLevel) : (concatPreOrder cs).length = sumTotalNodeCount cs :=
  match cs with
  | [] => Eq.refl 0
  | c :: cs' =>
    have h1 : (traversePreOrder c ++ concatPreOrder cs').length = (traversePreOrder c).length + (concatPreOrder cs').length := list_length_append (traversePreOrder c) (concatPreOrder cs')
    have h2 : (traversePreOrder c).length = getTotalNodeCount c := length_traversePreOrder c
    have h3 : (concatPreOrder cs').length = sumTotalNodeCount cs' := length_concatPreOrder cs'
    Eq.trans h1 (Eq.trans (congrArg (fun x => x + (concatPreOrder cs').length) h2) (congrArg (Nat.add (getTotalNodeCount c)) h3))
end

def insertNode (node : FractalNodeData) : Nat → FractalLevel → FractalLevel
  | 0, FractalLevel.mk l sf nodes children => FractalLevel.mk l sf (node :: nodes) children
  | Nat.succ d, FractalLevel.mk l sf nodes children =>
    match children with
    | [] => FractalLevel.mk l sf (node :: nodes) children
    | c :: cs => FractalLevel.mk l sf nodes (insertNode node d c :: cs)

theorem succ_add (a b : Nat) : Nat.succ a + b = Nat.succ (a + b) :=
  Nat.recOn b (Eq.refl (Nat.succ a)) (fun b' ih => congrArg Nat.succ ih)

theorem insertNode_count (node : FractalNodeData) : ∀ (d : Nat) (lvl : FractalLevel),
  getTotalNodeCount (insertNode node d lvl) = getTotalNodeCount lvl + 1
  | 0, FractalLevel.mk l sf nodes children =>
    have h_eval : getTotalNodeCount (FractalLevel.mk l sf (node :: nodes) children) = Nat.succ nodes.length + sumTotalNodeCount children := Eq.refl _
    have h_add : Nat.succ nodes.length + sumTotalNodeCount children = Nat.succ (nodes.length + sumTotalNodeCount children) := succ_add nodes.length (sumTotalNodeCount children)
    Eq.trans h_eval h_add
  | Nat.succ d, FractalLevel.mk l sf nodes children =>
    match children with
    | [] =>
      have h_eval : getTotalNodeCount (FractalLevel.mk l sf (node :: nodes) []) = Nat.succ nodes.length + sumTotalNodeCount [] := Eq.refl _
      have h_add : Nat.succ nodes.length + sumTotalNodeCount [] = Nat.succ (nodes.length + sumTotalNodeCount []) := succ_add nodes.length (sumTotalNodeCount [])
      Eq.trans h_eval h_add
    | c :: cs =>
      have h_ih : getTotalNodeCount (insertNode node d c) = Nat.succ (getTotalNodeCount c) := insertNode_count node d c
      have h_eval : getTotalNodeCount (FractalLevel.mk l sf nodes (insertNode node d c :: cs)) = nodes.length + (getTotalNodeCount (insertNode node d c) + sumTotalNodeCount cs) := Eq.refl _
      have h_subst : nodes.length + (getTotalNodeCount (insertNode node d c) + sumTotalNodeCount cs) = nodes.length + (Nat.succ (getTotalNodeCount c) + sumTotalNodeCount cs) := congrArg (fun x => nodes.length + (x + sumTotalNodeCount cs)) h_ih
      have h_succ_add : Nat.succ (getTotalNodeCount c) + sumTotalNodeCount cs = Nat.succ (getTotalNodeCount c + sumTotalNodeCount cs) := succ_add (getTotalNodeCount c) (sumTotalNodeCount cs)
      have h_subst2 : nodes.length + (Nat.succ (getTotalNodeCount c) + sumTotalNodeCount cs) = nodes.length + Nat.succ (getTotalNodeCount c + sumTotalNodeCount cs) := congrArg (Nat.add nodes.length) h_succ_add
      have h_def : nodes.length + Nat.succ (getTotalNodeCount c + sumTotalNodeCount cs) = Nat.succ (nodes.length + (getTotalNodeCount c + sumTotalNodeCount cs)) := Eq.refl _
      Eq.trans h_eval (Eq.trans h_subst (Eq.trans h_subst2 h_def))

def deleteNodeFromList (k : Key) : List FractalNodeData → Option (List FractalNodeData)
  | [] => none
  | n :: ns =>
    match list_decEq k n.id with
    | isTrue _ => some ns
    | isFalse _ =>
      match deleteNodeFromList k ns with
      | some new_ns => some (n :: new_ns)
      | none => none

theorem deleteNodeFromList_length (k : Key) : ∀ (ns new_ns : List FractalNodeData),
  deleteNodeFromList k ns = some new_ns → ns.length = new_ns.length + 1
  | [], new_ns, h => Option.noConfusion h
  | n :: ns, new_ns, h =>
    match h_dec : list_decEq k n.id with
    | isTrue _ =>
      have h_match : (match list_decEq k n.id with | isTrue _ => some ns | isFalse _ => match deleteNodeFromList k ns with | some new_ns => some (n :: new_ns) | none => none) = some ns :=
        Eq.subst (motive := fun x => (match x with | isTrue _ => some ns | isFalse _ => match deleteNodeFromList k ns with | some new_ns => some (n :: new_ns) | none => none) = some ns) (Eq.symm h_dec) (Eq.refl _)
      have h_eq : some ns = some new_ns := Eq.trans (Eq.symm h_match) h
      Option.noConfusion h_eq (fun h_ns_eq =>
        Eq.subst (motive := fun x => (n :: ns).length = x.length + 1) (Eq.symm h_ns_eq) (Eq.refl _)
      )
    | isFalse _ =>
      match h_rec : deleteNodeFromList k ns with
      | none =>
        have h_match : (match list_decEq k n.id with | isTrue _ => some ns | isFalse _ => match deleteNodeFromList k ns with | some new_ns => some (n :: new_ns) | none => none) = none :=
          Eq.subst (motive := fun x => (match x with | isTrue _ => some ns | isFalse _ => match deleteNodeFromList k ns with | some new_ns => some (n :: new_ns) | none => none) = none) (Eq.symm h_dec) (
            Eq.subst (motive := fun y => (match y with | some new_ns => some (n :: new_ns) | none => none) = none) (Eq.symm h_rec) (Eq.refl _)
          )
        have h_contra : none = some new_ns := Eq.trans (Eq.symm h_match) h
        Option.noConfusion h_contra
      | some new_ns' =>
        have h_match : (match list_decEq k n.id with | isTrue _ => some ns | isFalse _ => match deleteNodeFromList k ns with | some new_ns => some (n :: new_ns) | none => none) = some (n :: new_ns') :=
          Eq.subst (motive := fun x => (match x with | isTrue _ => some ns | isFalse _ => match deleteNodeFromList k ns with | some new_ns => some (n :: new_ns) | none => none) = some (n :: new_ns')) (Eq.symm h_dec) (
            Eq.subst (motive := fun y => (match y with | some new_ns => some (n :: new_ns) | none => none) = some (n :: new_ns')) (Eq.symm h_rec) (Eq.refl _)
          )
        have h_eq : some (n :: new_ns') = some new_ns := Eq.trans (Eq.symm h_match) h
        Option.noConfusion h_eq (fun h_ns_eq =>
          have h_ih : ns.length = new_ns'.length + 1 := deleteNodeFromList_length k ns new_ns' h_rec
          have h_len : (n :: ns).length = new_ns'.length + 1 + 1 := congrArg Nat.succ h_ih
          Eq.subst (motive := fun x => (n :: ns).length = x.length + 1) h_ns_eq h_len
        )

mutual
def deleteNode (k : Key) : FractalLevel → Option FractalLevel
  | FractalLevel.mk l sf nodes children =>
    match deleteNodeFromList k nodes with
    | some new_nodes => some (FractalLevel.mk l sf new_nodes children)
    | none =>
      match deleteNodeFromChildren k children with
      | some new_children => some (FractalLevel.mk l sf nodes new_children)
      | none => none

def deleteNodeFromChildren (k : Key) : List FractalLevel → Option (List FractalLevel)
  | [] => none
  | c :: cs =>
    match deleteNode k c with
    | some new_c => some (new_c :: cs)
    | none =>
      match deleteNodeFromChildren k cs with
      | some new_cs => some (c :: new_cs)
      | none => none
end

mutual
theorem deleteNode_count_eq (k : Key) : ∀ (lvl new_lvl : FractalLevel),
  deleteNode k lvl = some new_lvl → getTotalNodeCount lvl = getTotalNodeCount new_lvl + 1
  | FractalLevel.mk l sf nodes children, new_lvl, h =>
    match h_list : deleteNodeFromList k nodes with
    | some new_nodes =>
      have h_match : (match deleteNodeFromList k nodes with | some new_nodes => some (FractalLevel.mk l sf new_nodes children) | none => match deleteNodeFromChildren k children with | some new_children => some (FractalLevel.mk l sf nodes new_children) | none => none) = some (FractalLevel.mk l sf new_nodes children) :=
        Eq.subst (motive := fun x => (match x with | some new_nodes => some (FractalLevel.mk l sf new_nodes children) | none => match deleteNodeFromChildren k children with | some new_children => some (FractalLevel.mk l sf nodes new_children) | none => none) = some (FractalLevel.mk l sf new_nodes children)) (Eq.symm h_list) (Eq.refl _)
      have h_eq : some (FractalLevel.mk l sf new_nodes children) = some new_lvl := Eq.trans (Eq.symm h_match) h
      Option.noConfusion h_eq (fun h_lvl_eq =>
        have h_len : nodes.length = new_nodes.length + 1 := deleteNodeFromList_length k nodes new_nodes h_list
        have h_total : nodes.length + sumTotalNodeCount children = new_nodes.length + 1 + sumTotalNodeCount children := congrArg (fun x => x + sumTotalNodeCount children) h_len
        have h_assoc : new_nodes.length + 1 + sumTotalNodeCount children = new_nodes.length + sumTotalNodeCount children + 1 := succ_add new_nodes.length (sumTotalNodeCount children)
        have h_final : nodes.length + sumTotalNodeCount children = new_nodes.length + sumTotalNodeCount children + 1 := Eq.trans h_total h_assoc
        Eq.subst (motive := fun x => nodes.length + sumTotalNodeCount children = getTotalNodeCount x + 1) h_lvl_eq h_final
      )
    | none =>
      match h_child : deleteNodeFromChildren k children with
      | some new_children =>
        have h_match : (match deleteNodeFromList k nodes with | some new_nodes => some (FractalLevel.mk l sf new_nodes children) | none => match deleteNodeFromChildren k children with | some new_children => some (FractalLevel.mk l sf nodes new_children) | none => none) = some (FractalLevel.mk l sf nodes new_children) :=
          Eq.subst (motive := fun x => (match x with | some new_nodes => some (FractalLevel.mk l sf new_nodes children) | none => match deleteNodeFromChildren k children with | some new_children => some (FractalLevel.mk l sf nodes new_children) | none => none) = some (FractalLevel.mk l sf nodes new_children)) (Eq.symm h_list) (
            Eq.subst (motive := fun y => (match y with | some new_children => some (FractalLevel.mk l sf nodes new_children) | none => none) = some (FractalLevel.mk l sf nodes new_children)) (Eq.symm h_child) (Eq.refl _)
          )
        have h_eq : some (FractalLevel.mk l sf nodes new_children) = some new_lvl := Eq.trans (Eq.symm h_match) h
        Option.noConfusion h_eq (fun h_lvl_eq =>
          have h_ih : sumTotalNodeCount children = sumTotalNodeCount new_children + 1 := deleteNodeFromChildren_count_eq k children new_children h_child
          have h_total : nodes.length + sumTotalNodeCount children = nodes.length + (sumTotalNodeCount new_children + 1) := congrArg (Nat.add nodes.length) h_ih
          have h_assoc : nodes.length + (sumTotalNodeCount new_children + 1) = nodes.length + sumTotalNodeCount new_children + 1 := Eq.refl _
          have h_final : nodes.length + sumTotalNodeCount children = nodes.length + sumTotalNodeCount new_children + 1 := Eq.trans h_total h_assoc
          Eq.subst (motive := fun x => nodes.length + sumTotalNodeCount children = getTotalNodeCount x + 1) h_lvl_eq h_final
        )
      | none =>
        have h_match : (match deleteNodeFromList k nodes with | some new_nodes => some (FractalLevel.mk l sf new_nodes children) | none => match deleteNodeFromChildren k children with | some new_children => some (FractalLevel.mk l sf nodes new_children) | none => none) = none :=
          Eq.subst (motive := fun x => (match x with | some new_nodes => some (FractalLevel.mk l sf new_nodes children) | none => match deleteNodeFromChildren k children with | some new_children => some (FractalLevel.mk l sf nodes new_children) | none => none) = none) (Eq.symm h_list) (
            Eq.subst (motive := fun y => (match y with | some new_children => some (FractalLevel.mk l sf nodes new_children) | none => none) = none) (Eq.symm h_child) (Eq.refl _)
          )
        have h_contra : none = some new_lvl := Eq.trans (Eq.symm h_match) h
        Option.noConfusion h_contra

theorem deleteNodeFromChildren_count_eq (k : Key) : ∀ (cs new_cs : List FractalLevel),
  deleteNodeFromChildren k cs = some new_cs → sumTotalNodeCount cs = sumTotalNodeCount new_cs + 1
  | [], new_cs, h => Option.noConfusion h
  | c :: cs, new_cs, h =>
    match h_c : deleteNode k c with
    | some new_c =>
      have h_match : (match deleteNode k c with | some new_c => some (new_c :: cs) | none => match deleteNodeFromChildren k cs with | some new_cs => some (c :: new_cs) | none => none) = some (new_c :: cs) :=
        Eq.subst (motive := fun x => (match x with | some new_c => some (new_c :: cs) | none => match deleteNodeFromChildren k cs with | some new_cs => some (c :: new_cs) | none => none) = some (new_c :: cs)) (Eq.symm h_c) (Eq.refl _)
      have h_eq : some (new_c :: cs) = some new_cs := Eq.trans (Eq.symm h_match) h
      Option.noConfusion h_eq (fun h_cs_eq =>
        have h_ih : getTotalNodeCount c = getTotalNodeCount new_c + 1 := deleteNode_count_eq k c new_c h_c
        have h_total : getTotalNodeCount c + sumTotalNodeCount cs = getTotalNodeCount new_c + 1 + sumTotalNodeCount cs := congrArg (fun x => x + sumTotalNodeCount cs) h_ih
        have h_assoc : getTotalNodeCount new_c + 1 + sumTotalNodeCount cs = getTotalNodeCount new_c + sumTotalNodeCount cs + 1 := succ_add (getTotalNodeCount new_c) (sumTotalNodeCount cs)
        have h_final : getTotalNodeCount c + sumTotalNodeCount cs = getTotalNodeCount new_c + sumTotalNodeCount cs + 1 := Eq.trans h_total h_assoc
        Eq.subst (motive := fun x => getTotalNodeCount c + sumTotalNodeCount cs = sumTotalNodeCount x + 1) h_cs_eq h_final
      )
    | none =>
      match h_cs : deleteNodeFromChildren k cs with
      | some new_cs' =>
        have h_match : (match deleteNode k c with | some new_c => some (new_c :: cs) | none => match deleteNodeFromChildren k cs with | some new_cs => some (c :: new_cs) | none => none) = some (c :: new_cs') :=
          Eq.subst (motive := fun x => (match x with | some new_c => some (new_c :: cs) | none => match deleteNodeFromChildren k cs with | some new_cs => some (c :: new_cs) | none => none) = some (c :: new_cs')) (Eq.symm h_c) (
            Eq.subst (motive := fun y => (match y with | some new_cs => some (c :: new_cs) | none => none) = some (c :: new_cs')) (Eq.symm h_cs) (Eq.refl _)
          )
        have h_eq : some (c :: new_cs') = some new_cs := Eq.trans (Eq.symm h_match) h
        Option.noConfusion h_eq (fun h_cs_eq =>
          have h_ih : sumTotalNodeCount cs = sumTotalNodeCount new_cs' + 1 := deleteNodeFromChildren_count_eq k cs new_cs' h_cs
          have h_total : getTotalNodeCount c + sumTotalNodeCount cs = getTotalNodeCount c + (sumTotalNodeCount new_cs' + 1) := congrArg (Nat.add (getTotalNodeCount c)) h_ih
          have h_assoc : getTotalNodeCount c + (sumTotalNodeCount new_cs' + 1) = getTotalNodeCount c + sumTotalNodeCount new_cs' + 1 := Eq.refl _
          have h_final : getTotalNodeCount c + sumTotalNodeCount cs = getTotalNodeCount c + sumTotalNodeCount new_cs' + 1 := Eq.trans h_total h_assoc
          Eq.subst (motive := fun x => getTotalNodeCount c + sumTotalNodeCount cs = sumTotalNodeCount x + 1) h_cs_eq h_final
        )
      | none =>
        have h_match : (match deleteNode k c with | some new_c => some (new_c :: cs) | none => match deleteNodeFromChildren k cs with | some new_cs => some (c :: new_cs) | none => none) = none :=
          Eq.subst (motive := fun x => (match x with | some new_c => some (new_c :: cs) | none => match deleteNodeFromChildren k cs with | some new_cs => some (c :: new_cs) | none => none) = none) (Eq.symm h_c) (
            Eq.subst (motive := fun y => (match y with | some new_cs => some (c :: new_cs) | none => none) = none) (Eq.symm h_cs) (Eq.refl _)
          )
        have h_contra : none = some new_cs := Eq.trans (Eq.symm h_match) h
        Option.noConfusion h_contra
end

def log2_bounded : Nat → Nat → Nat
  | 0, _ => 0
  | Nat.succ _, 0 => 0
  | Nat.succ _, 1 => 0
  | Nat.succ fuel, n => 1 + log2_bounded fuel (n / 2)

def log2 (n : Nat) : Nat := log2_bounded n n

def optimalDepth (total_nodes branching_factor : Nat) : Nat :=
  if branching_factor < 2 then total_nodes else log2 total_nodes / log2 branching_factor

structure FractalTree where
  root : FractalLevel
  max_depth : Nat
  branching_factor : Nat
  total_nodes : Nat
  is_balanced : Bool

def FractalTree.insert (t : FractalTree) (target_depth : Nat) (node : FractalNodeData) : FractalTree :=
  { t with root := insertNode node target_depth t.root,
           total_nodes := t.total_nodes + 1,
           is_balanced := false }

theorem FractalTree.insert_total_nodes_eq (t : FractalTree) (d : Nat) (n : FractalNodeData)
  (h_sync : getTotalNodeCount t.root = t.total_nodes) :
  getTotalNodeCount (t.insert d n).root = (t.insert d n).total_nodes :=
  have h1 : getTotalNodeCount (t.insert d n).root = getTotalNodeCount t.root + 1 := insertNode_count n d t.root
  have h2 : getTotalNodeCount t.root + 1 = t.total_nodes + 1 := congrArg (fun x => x + 1) h_sync
  have h3 : t.total_nodes + 1 = (t.insert d n).total_nodes := Eq.refl _
  Eq.trans h1 (Eq.trans h2 h3)

def FractalTree.delete (t : FractalTree) (k : Key) : FractalTree :=
  match deleteNode k t.root with
  | some new_root =>
    { t with root := new_root,
             total_nodes := t.total_nodes - 1,
             is_balanced := false }
  | none => t

theorem FractalTree.delete_total_nodes_eq (t : FractalTree) (k : Key)
  (h_sync : getTotalNodeCount t.root = t.total_nodes)
  (new_root : FractalLevel)
  (h_del : deleteNode k t.root = some new_root) :
  getTotalNodeCount (t.delete k).root = (t.delete k).total_nodes :=
  have h_match : (match deleteNode k t.root with | some r => { t with root := r, total_nodes := t.total_nodes - 1, is_balanced := false } | none => t) = { t with root := new_root, total_nodes := t.total_nodes - 1, is_balanced := false } :=
    Eq.subst (motive := fun x => (match x with | some r => { t with root := r, total_nodes := t.total_nodes - 1, is_balanced := false } | none => t) = { t with root := new_root, total_nodes := t.total_nodes - 1, is_balanced := false }) (Eq.symm h_del) (Eq.refl _)
  have h_eval : (t.delete k) = { t with root := new_root, total_nodes := t.total_nodes - 1, is_balanced := false } := h_match
  have h_root : (t.delete k).root = new_root := congrArg FractalTree.root h_eval
  have h_nodes : (t.delete k).total_nodes = t.total_nodes - 1 := congrArg FractalTree.total_nodes h_eval
  have h_count : getTotalNodeCount t.root = getTotalNodeCount new_root + 1 := deleteNode_count_eq k t.root new_root h_del
  have h_subst1 : getTotalNodeCount new_root + 1 = t.total_nodes := Eq.trans (Eq.symm h_count) h_sync
  have h_subst2 : getTotalNodeCount new_root = t.total_nodes - 1 :=
    have h_add_sub : getTotalNodeCount new_root + 1 - 1 = t.total_nodes - 1 := congrArg (fun x => x - 1) h_subst1
    have h_cancel : getTotalNodeCount new_root + 1 - 1 = getTotalNodeCount new_root := Eq.refl _
    Eq.trans (Eq.symm h_cancel) h_add_sub
  have h_final1 : getTotalNodeCount (t.delete k).root = getTotalNodeCount new_root := congrArg getTotalNodeCount h_root
  have h_final2 : getTotalNodeCount new_root = (t.delete k).total_nodes := Eq.trans h_subst2 (Eq.symm h_nodes)
  Eq.trans h_final1 h_final2

def sum_x : List (Nat × Nat) → Nat
  | [] => 0
  | (x, _) :: xs => x + sum_x xs

def sum_y : List (Nat × Nat) → Nat
  | [] => 0
  | (_, y) :: xs => y + sum_y xs

def sum_xy : List (Nat × Nat) → Nat
  | [] => 0
  | (x, y) :: xs => x * y + sum_xy xs

def sum_x2 : List (Nat × Nat) → Nat
  | [] => 0
  | (x, _) :: xs => x * x + sum_x2 xs

def computeFractalDimension (pts : List (Nat × Nat)) : Nat :=
  let den := pts.length * sum_x2 pts - sum_x pts * sum_x pts
  match Nat.decEq den 0 with
  | isTrue _ => 1000
  | isFalse _ =>
    let num := pts.length * sum_xy pts - sum_x pts * sum_y pts
    num / den

structure PatternLocation where
  tree_id : Key
  level : Nat
  node_id : Key
  offset : Nat
  length : Nat
  confidence : Nat

structure SelfSimilarIndex where
  patterns : List (Key × List PatternLocation)
  dimension_estimate : Nat
  pattern_count : Nat
  total_locations : Nat

def update_patterns (k : Key) (loc : PatternLocation) : List (Key × List PatternLocation) → List (Key × List PatternLocation)
  | [] => [(k, [loc])]
  | (k', locs) :: xs =>
    match list_decEq k k' with
    | isTrue _ => (k, loc :: locs) :: xs
    | isFalse _ => (k', locs) :: update_patterns k loc xs

def SelfSimilarIndex.addPattern (idx : SelfSimilarIndex) (pattern : Key) (location : PatternLocation) : SelfSimilarIndex :=
  { idx with patterns := update_patterns pattern location idx.patterns, total_locations := idx.total_locations + 1 }

def insertLengthCount (len : Nat) : List (Nat × Nat) → List (Nat × Nat)
  | [] => [(len, 1)]
  | (l, c) :: xs =>
    match Nat.decEq len l with
    | isTrue _ => (l, c + 1) :: xs
    | isFalse _ => (l, c) :: insertLengthCount len xs

def aggregatePatternLengths : List (Key × List PatternLocation) → List (Nat × Nat)
  | [] => []
  | (k, _) :: xs => insertLengthCount k.length (aggregatePatternLengths xs)

def SelfSimilarIndex.computeFractalDimension (idx : SelfSimilarIndex) : Nat :=
  computeFractalDimension (aggregatePatternLengths idx.patterns)

structure FNDSStatistics where
  total_trees : Nat
  total_indices : Nat
  cache_hits : Nat
  cache_misses : Nat

def FNDSStatistics.init : FNDSStatistics :=
  { total_trees := 0, total_indices := 0, cache_hits := 0, cache_misses := 0 }

structure FNDSManager where
  fractal_trees : List (Key × FractalTree)
  indices : List (Key × SelfSimilarIndex)
  cache : LRUCache
  statistics : FNDSStatistics

def FNDSManager.init (cache_capacity cache_memory : Nat) : FNDSManager :=
  { fractal_trees := [],
    indices := [],
    cache := LRUCache.init cache_capacity cache_memory,
    statistics := FNDSStatistics.init }

def sumTreeDimensions : List (Key × FractalTree) → Nat
  | [] => 0
  | (_, t) :: xs =>
    let pts := [(t.total_nodes, t.max_depth)]
    computeFractalDimension pts + sumTreeDimensions xs

def sumIndexDimensions : List (Key × SelfSimilarIndex) → Nat
  | [] => 0
  | (_, idx) :: xs => idx.computeFractalDimension + sumIndexDimensions xs

def FNDSManager.computeGlobalFractalDimension (m : FNDSManager) : Nat :=
  let total_dim := sumTreeDimensions m.fractal_trees + sumIndexDimensions m.indices
  let count := m.fractal_trees.length + m.indices.length
  match Nat.decEq count 0 with
  | isTrue _ => 0
  | isFalse _ => total_dim / count

theorem global_dimension_ge_zero (m : FNDSManager) : 0 ≤ m.computeGlobalFractalDimension :=
  Nat.zero_le_theorem insertLengthCount_not_empty (len : Nat) : ∀ (l : List (Nat × Nat)), insertLengthCount len l ≠ []
  | [] => fun h => List.noConfusion h
  | (l', c) :: xs =>
    match Nat.decEq len l' with
    | isTrue _ => fun h => List.noConfusion h
    | isFalse _ => fun h => List.noConfusion h

theorem insertLengthCount_length_le (len : Nat) : ∀ (l : List (Nat × Nat)), (insertLengthCount len l).length ≤ l.length + 1
  | [] => Nat.le_refl 1
  | (l', c) :: xs =>
    match Nat.decEq len l' with
    | isTrue _ => Nat.le_step (Nat.le_refl _)
    | isFalse _ => Nat.succ_le_succ (insertLengthCount_length_le len xs)

theorem insertLengthCount_length_ge (len : Nat) : ∀ (l : List (Nat × Nat)), l.length ≤ (insertLengthCount len l).length
  | [] => Nat.zero_le _
  | (l', c) :: xs =>
    match Nat.decEq len l' with
    | isTrue _ => Nat.le_refl _
    | isFalse _ => Nat.succ_le_succ (insertLengthCount_length_ge len xs)

theorem aggregatePatternLengths_nil : aggregatePatternLengths [] = [] := Eq.refl _

theorem aggregatePatternLengths_length_le : ∀ (l : List (Key × List PatternLocation)), (aggregatePatternLengths l).length ≤ l.length
  | [] => Nat.le_refl 0
  | (k, locs) :: xs =>
    have h_ih : (aggregatePatternLengths xs).length ≤ xs.length := aggregatePatternLengths_length_le xs
    have h_insert : (insertLengthCount k.length (aggregatePatternLengths xs)).length ≤ (aggregatePatternLengths xs).length + 1 := insertLengthCount_length_le k.length (aggregatePatternLengths xs)
    have h_trans : (insertLengthCount k.length (aggregatePatternLengths xs)).length ≤ xs.length + 1 := Nat.le_trans h_insert (Nat.add_le_add_right h_ih 1)
    h_trans

theorem sumTreeDimensions_nil : sumTreeDimensions [] = 0 := Eq.refl 0

theorem sumTreeDimensions_cons (k : Key) (t : FractalTree) (xs : List (Key × FractalTree)) :
  sumTreeDimensions ((k, t) :: xs) = computeFractalDimension [(t.total_nodes, t.max_depth)] + sumTreeDimensions xs := Eq.refl _

theorem sumIndexDimensions_nil : sumIndexDimensions [] = 0 := Eq.refl 0

theorem sumIndexDimensions_cons (k : Key) (idx : SelfSimilarIndex) (xs : List (Key × SelfSimilarIndex)) :
  sumIndexDimensions ((k, idx) :: xs) = idx.computeFractalDimension + sumIndexDimensions xs := Eq.refl _

theorem computeGlobalFractalDimension_empty (c m : Nat) : (FNDSManager.init c m).computeGlobalFractalDimension = 0 := Eq.refl 0

def FNDSManager.addTree (m : FNDSManager) (id : Key) (t : FractalTree) : FNDSManager :=
  { m with fractal_trees := (id, t) :: m.fractal_trees, statistics := { m.statistics with total_trees := m.statistics.total_trees + 1 } }

theorem FNDSManager.addTree_increments_trees (m : FNDSManager) (id : Key) (t : FractalTree) : (m.addTree id t).statistics.total_trees = m.statistics.total_trees + 1 := Eq.refl _
theorem FNDSManager.addTree_preserves_indices (m : FNDSManager) (id : Key) (t : FractalTree) : (m.addTree id t).indices = m.indices := Eq.refl _
theorem FNDSManager.addTree_preserves_cache (m : FNDSManager) (id : Key) (t : FractalTree) : (m.addTree id t).cache = m.cache := Eq.refl _
theorem FNDSManager.addTree_length (m : FNDSManager) (id : Key) (t : FractalTree) : (m.addTree id t).fractal_trees.length = m.fractal_trees.length + 1 := Eq.refl _

def FNDSManager.addIndex (m : FNDSManager) (id : Key) (idx : SelfSimilarIndex) : FNDSManager :=
  { m with indices := (id, idx) :: m.indices, statistics := { m.statistics with total_indices := m.statistics.total_indices + 1 } }

theorem FNDSManager.addIndex_increments_indices (m : FNDSManager) (id : Key) (idx : SelfSimilarIndex) : (m.addIndex id idx).statistics.total_indices = m.statistics.total_indices + 1 := Eq.refl _
theorem FNDSManager.addIndex_preserves_trees (m : FNDSManager) (id : Key) (idx : SelfSimilarIndex) : (m.addIndex id idx).fractal_trees = m.fractal_trees := Eq.refl _
theorem FNDSManager.addIndex_preserves_cache (m : FNDSManager) (id : Key) (idx : SelfSimilarIndex) : (m.addIndex id idx).cache = m.cache := Eq.refl _
theorem FNDSManager.addIndex_length (m : FNDSManager) (id : Key) (idx : SelfSimilarIndex) : (m.addIndex id idx).indices.length = m.indices.length + 1 := Eq.refl _

def FNDSManager.removeTree (m : FNDSManager) (id : Key) : FNDSManager :=
  { m with fractal_trees := remove_key m.fractal_trees id }

theorem FNDSManager.removeTree_length_le (m : FNDSManager) (id : Key) : (m.removeTree id).fractal_trees.length ≤ m.fractal_trees.length :=
  remove_key_length_le m.fractal_trees id

theorem FNDSManager.removeTree_preserves_indices (m : FNDSManager) (id : Key) : (m.removeTree id).indices = m.indices := Eq.refl _
theorem FNDSManager.removeTree_preserves_cache (m : FNDSManager) (id : Key) : (m.removeTree id).cache = m.cache := Eq.refl _

def FNDSManager.removeIndex (m : FNDSManager) (id : Key) : FNDSManager :=
  { m with indices := remove_key m.indices id }

theorem FNDSManager.removeIndex_length_le (m : FNDSManager) (id : Key) : (m.removeIndex id).indices.length ≤ m.indices.length :=
  remove_key_length_le m.indices id

theorem FNDSManager.removeIndex_preserves_trees (m : FNDSManager) (id : Key) : (m.removeIndex id).fractal_trees = m.fractal_trees := Eq.refl _
theorem FNDSManager.removeIndex_preserves_cache (m : FNDSManager) (id : Key) : (m.removeIndex id).cache = m.cache := Eq.refl _

theorem FNDSManager.removeTree_twice_length_le (m : FNDSManager) (id1 id2 : Key) : (m.removeTree id1 |>.removeTree id2).fractal_trees.length ≤ m.fractal_trees.length :=
  Nat.le_trans (remove_key_length_le (m.removeTree id1).fractal_trees id2) (remove_key_length_le m.fractal_trees id1)

theorem FNDSManager.removeIndex_twice_length_le (m : FNDSManager) (id1 id2 : Key) : (m.removeIndex id1 |>.removeIndex id2).indices.length ≤ m.indices.length :=
  Nat.le_trans (remove_key_length_le (m.removeIndex id1).indices id2) (remove_key_length_le m.indices id1)
def find_pattern_in_list (k : Key) : List (Key × List PatternLocation) → List PatternLocation
  | [] => []
  | (k', locs) :: xs =>
    match list_decEq k k' with
    | isTrue _ => locs
    | isFalse _ => find_pattern_in_list k xs

theorem find_pattern_in_list_nil (k : Key) : find_pattern_in_list k [] = [] := Eq.refl []

theorem find_pattern_in_list_update_eq (k : Key) (loc : PatternLocation) : ∀ (l : List (Key × List PatternLocation)),
  find_pattern_in_list k (update_patterns k loc l) = loc :: find_pattern_in_list k l
  | [] =>
    have h_eval : find_pattern_in_list k [(k, [loc])] = (match list_decEq k k with | isTrue _ => [loc] | isFalse _ => find_pattern_in_list k []) := Eq.refl _
    have h_dec : (match list_decEq k k with | isTrue _ => [loc] | isFalse _ => find_pattern_in_list k []) = [loc] :=
      match h : list_decEq k k with
      | isTrue _ => Eq.refl _
      | isFalse h_neq => False.elim (h_neq (Eq.refl k))
    have h_nil : find_pattern_in_list k [] = [] := Eq.refl []
    have h_cons : [loc] = loc :: [] := Eq.refl _
    Eq.trans h_eval (Eq.trans h_dec (Eq.trans h_cons (congrArg (fun x => loc :: x) (Eq.symm h_nil))))
  | (k', locs) :: xs =>
    match h_dec : list_decEq k k' with
    | isTrue h_eq =>
      have h_update : update_patterns k loc ((k', locs) :: xs) = (k, loc :: locs) :: xs :=
        Eq.subst (motive := fun x => (match x with | isTrue _ => (k, loc :: locs) :: xs | isFalse _ => (k', locs) :: update_patterns k loc xs) = (k, loc :: locs) :: xs) (Eq.symm h_dec) (Eq.refl _)
      have h_find1 : find_pattern_in_list k ((k, loc :: locs) :: xs) = (match list_decEq k k with | isTrue _ => loc :: locs | isFalse _ => find_pattern_in_list k xs) := Eq.refl _
      have h_find2 : (match list_decEq k k with | isTrue _ => loc :: locs | isFalse _ => find_pattern_in_list k xs) = loc :: locs :=
        match h_dec2 : list_decEq k k with
        | isTrue _ => Eq.refl _
        | isFalse h_neq => False.elim (h_neq (Eq.refl k))
      have h_find_orig : find_pattern_in_list k ((k', locs) :: xs) = locs :=
        have h_eval : find_pattern_in_list k ((k', locs) :: xs) = (match list_decEq k k' with | isTrue _ => locs | isFalse _ => find_pattern_in_list k xs) := Eq.refl _
        Eq.trans h_eval (Eq.subst (motive := fun x => (match x with | isTrue _ => locs | isFalse _ => find_pattern_in_list k xs) = locs) (Eq.symm h_dec) (Eq.refl _))
      Eq.trans (congrArg (find_pattern_in_list k) h_update) (Eq.trans h_find1 (Eq.trans h_find2 (congrArg (fun x => loc :: x) (Eq.symm h_find_orig))))
    | isFalse h_neq =>
      have h_update : update_patterns k loc ((k', locs) :: xs) = (k', locs) :: update_patterns k loc xs :=
        Eq.subst (motive := fun x => (match x with | isTrue _ => (k, loc :: locs) :: xs | isFalse _ => (k', locs) :: update_patterns k loc xs) = (k', locs) :: update_patterns k loc xs) (Eq.symm h_dec) (Eq.refl _)
      have h_find1 : find_pattern_in_list k ((k', locs) :: update_patterns k loc xs) = (match list_decEq k k' with | isTrue _ => locs | isFalse _ => find_pattern_in_list k (update_patterns k loc xs)) := Eq.refl _
      have h_find2 : (match list_decEq k k' with | isTrue _ => locs | isFalse _ => find_pattern_in_list k (update_patterns k loc xs)) = find_pattern_in_list k (update_patterns k loc xs) :=
        Eq.subst (motive := fun x => (match x with | isTrue _ => locs | isFalse _ => find_pattern_in_list k (update_patterns k loc xs)) = find_pattern_in_list k (update_patterns k loc xs)) (Eq.symm h_dec) (Eq.refl _)
      have h_ih : find_pattern_in_list k (update_patterns k loc xs) = loc :: find_pattern_in_list k xs := find_pattern_in_list_update_eq k loc xs
      have h_find_orig : find_pattern_in_list k ((k', locs) :: xs) = find_pattern_in_list k xs :=
        have h_eval : find_pattern_in_list k ((k', locs) :: xs) = (match list_decEq k k' with | isTrue _ => locs | isFalse _ => find_pattern_in_list k xs) := Eq.refl _
        Eq.trans h_eval (Eq.subst (motive := fun x => (match x with | isTrue _ => locs | isFalse _ => find_pattern_in_list k xs) = find_pattern_in_list k xs) (Eq.symm h_dec) (Eq.refl _))
      Eq.trans (congrArg (find_pattern_in_list k) h_update) (Eq.trans h_find1 (Eq.trans h_find2 (Eq.trans h_ih (congrArg (fun x => loc :: x) (Eq.symm h_find_orig)))))

def SelfSimilarIndex.findPattern (idx : SelfSimilarIndex) (pattern : Key) : List PatternLocation :=
  find_pattern_in_list pattern idx.patterns

theorem SelfSimilarIndex.findPattern_addPattern_eq (idx : SelfSimilarIndex) (pattern : Key) (loc : PatternLocation) :
  (idx.addPattern pattern loc).findPattern pattern = loc :: idx.findPattern pattern :=
  find_pattern_in_list_update_eq pattern loc idx.patterns

def remove_pattern_in_list (k : Key) : List (Key × List PatternLocation) → List (Key × List PatternLocation)
  | [] => []
  | (k', locs) :: xs =>
    match list_decEq k k' with
    | isTrue _ => xs
    | isFalse _ => (k', locs) :: remove_pattern_in_list k xs

theorem remove_pattern_in_list_nil (k : Key) : remove_pattern_in_list k [] = [] := Eq.refl []

theorem remove_pattern_in_list_length_le (k : Key) : ∀ (l : List (Key × List PatternLocation)), (remove_pattern_in_list k l).length ≤ l.length
  | [] => Nat.le_refl 0
  | (k', locs) :: xs =>
    match list_decEq k k' with
    | isTrue _ => Nat.le_step (Nat.le_refl _)
    | isFalse _ => Nat.succ_le_succ (remove_pattern_in_list_length_le k xs)

def SelfSimilarIndex.removePattern (idx : SelfSimilarIndex) (pattern : Key) : SelfSimilarIndex :=
  let new_patterns := remove_pattern_in_list pattern idx.patterns
  let removed_count := (find_pattern_in_list pattern idx.patterns).length
  { idx with patterns := new_patterns,
             total_locations := idx.total_locations - removed_count }

theorem SelfSimilarIndex.removePattern_preserves_dimension_estimate (idx : SelfSimilarIndex) (pattern : Key) :
  (idx.removePattern pattern).dimension_estimate = idx.dimension_estimate := Eq.refl _

theorem SelfSimilarIndex.removePattern_preserves_pattern_count (idx : SelfSimilarIndex) (pattern : Key) :
  (idx.removePattern pattern).pattern_count = idx.pattern_count := Eq.refl _

theorem SelfSimilarIndex.removePattern_total_locations_le (idx : SelfSimilarIndex) (pattern : Key) :
  (idx.removePattern pattern).total_locations ≤ idx.total_locations :=
  Nat.sub_le idx.total_locations ((find_pattern_in_list pattern idx.patterns).length)

def searchNodeInList (k : Key) : List FractalNodeData → Option FractalNodeData
  | [] => none
  | n :: ns =>
    match list_decEq k n.id with
    | isTrue _ => some n
    | isFalse _ => searchNodeInList k ns

theorem searchNodeInList_nil (k : Key) : searchNodeInList k [] = none := Eq.refl none

mutual
def searchNode (k : Key) : FractalLevel → Option FractalNodeData
  | FractalLevel.mk _ _ nodes children =>
    match searchNodeInList k nodes with
    | some n => some n
    | none => searchNodeInChildren k children

def searchNodeInChildren (k : Key) : List FractalLevel → Option FractalNodeData
  | [] => none
  | c :: cs =>
    match searchNode k c with
    | some n => some n
    | none => searchNodeInChildren k cs
end

theorem searchNodeInChildren_nil (k : Key) : searchNodeInChildren k [] = none := Eq.refl none

def FractalTree.search (t : FractalTree) (k : Key) : Option FractalNodeData :=
  searchNode k t.root

theorem FractalTree.search_empty_root (t : FractalTree) (k : Key) (h_nodes : t.root.nodes = []) (h_children : t.root.children = []) :
  t.search k = none :=
  match t.root with
  | FractalLevel.mk l sf nodes children =>
    have h_eval : searchNode k (FractalLevel.mk l sf nodes children) = (match searchNodeInList k nodes with | some n => some n | none => searchNodeInChildren k children) := Eq.refl _
    have h_nodes_subst : searchNodeInList k nodes = none := Eq.subst (motive := fun x => searchNodeInList k x = none) (Eq.symm h_nodes) (Eq.refl none)
    have h_match1 : (match searchNodeInList k nodes with | some n => some n | none => searchNodeInChildren k children) = searchNodeInChildren k children :=
      Eq.subst (motive := fun x => (match x with | some n => some n | none => searchNodeInChildren k children) = searchNodeInChildren k children) (Eq.symm h_nodes_subst) (Eq.refl _)
    have h_children_subst : searchNodeInChildren k children = none := Eq.subst (motive := fun x => searchNodeInChildren k x = none) (Eq.symm h_children) (Eq.refl none)
    Eq.trans h_eval (Eq.trans h_match1 h_children_subst)

def FNDSManager.insertIntoTree (m : FNDSManager) (tree_id : Key) (target_depth : Nat) (node : FractalNodeData) : FNDSManager :=
  let rec update_tree_list (l : List (Key × FractalTree)) : List (Key × FractalTree) :=
    match l with
    | [] => []
    | (k, t) :: xs =>
      match list_decEq tree_id k with
      | isTrue _ => (k, t.insert target_depth node) :: xs
      | isFalse _ => (k, t) :: update_tree_list xs
  let new_trees := update_tree_list m.fractal_trees
  let new_stats := { m.statistics with total_nodes_across_trees := m.statistics.total_nodes_across_trees + 1 }
  { m with fractal_trees := new_trees, statistics := new_stats }

theorem FNDSManager.insertIntoTree_preserves_indices (m : FNDSManager) (tid : Key) (d : Nat) (n : FractalNodeData) :
  (m.insertIntoTree tid d n).indices = m.indices := Eq.refl _

theorem FNDSManager.insertIntoTree_preserves_cache (m : FNDSManager) (tid : Key) (d : Nat) (n : FractalNodeData) :
  (m.insertIntoTree tid d n).cache = m.cache := Eq.refl _

theorem FNDSManager.insertIntoTree_increments_total_nodes (m : FNDSManager) (tid : Key) (d : Nat) (n : FractalNodeData) :
  (m.insertIntoTree tid d n).statistics.total_nodes_across_trees = m.statistics.total_nodes_across_trees + 1 := Eq.refl _

theorem FNDSManager.insertIntoTree_preserves_total_trees (m : FNDSManager) (tid : Key) (d : Nat) (n : FractalNodeData) :
  (m.insertIntoTree tid d n).statistics.total_trees = m.statistics.total_trees := Eq.refl _

theorem FNDSManager.insertIntoTree_preserves_total_indices (m : FNDSManager) (tid : Key) (d : Nat) (n : FractalNodeData) :
  (m.insertIntoTree tid d n).statistics.total_indices = m.statistics.total_indices := Eq.refl _

def FNDSManager.searchInTree (m : FNDSManager) (tree_id : Key) (node_id : Key) : Option FractalNodeData × FNDSManager :=
  let cache_key := tree_id ++ node_id
  let (cached_val, m_after_cache_get) := m.cacheGet cache_key
  match cached_val with
  | some v =>
    let dummy_node := FractalNodeData.init node_id v 0 0
    (some dummy_node, m_after_cache_get)
  | none =>
    let rec find_tree (l : List (Key × FractalTree)) : Option FractalTree :=
      match l with
      | [] => none
      | (k, t) :: xs =>
        match list_decEq tree_id k with
        | isTrue _ => some t
        | isFalse _ => find_tree xs
    match find_tree m.fractal_trees with
    | none => (none, m_after_cache_get)
    | some t =>
      match t.search node_id with
      | none => (none, m_after_cache_get)
      | some n =>
        let m_after_cache_put := m_after_cache_get.cachePut cache_key n.data
        (some n, m_after_cache_put)

theorem FNDSManager.searchInTree_preserves_fractal_trees_length (m : FNDSManager) (tid nid : Key) :
  (m.searchInTree tid nid).snd.fractal_trees.length = m.fractal_trees.length :=
  let cache_key := tid ++ nid
  let (cached_val, m_after_cache_get) := m.cacheGet cache_key
  have h_cache_get_trees : m_after_cache_get.fractal_trees.length = m.fractal_trees.length := Eq.refl _
  match cached_val with
  | some _ => h_cache_get_trees
  | none =>
    let rec find_tree (l : List (Key × FractalTree)) : Option FractalTree :=
      match l with
      | [] => none
      | (k, t) :: xs => match list_decEq tid k with | isTrue _ => some t | isFalse _ => find_tree xs
    match find_tree m.fractal_trees with
    | none => h_cache_get_trees
    | some t =>
      match t.search nid with
      | none => h_cache_get_trees
      | some n =>
        have h_cache_put_trees : (m_after_cache_get.cachePut cache_key n.data).fractal_trees.length = m_after_cache_get.fractal_trees.length := Eq.refl _
        Eq.trans h_cache_put_trees h_cache_get_trees

theorem FNDSManager.searchInTree_preserves_indices_length (m : FNDSManager) (tid nid : Key) :
  (m.searchInTree tid nid).snd.indices.length = m.indices.length :=
  let cache_key := tid ++ nid
  let (cached_val, m_after_cache_get) := m.cacheGet cache_key
  have h_cache_get_indices : m_after_cache_get.indices.length = m.indices.length := Eq.refl _
  match cached_val with
  | some _ => h_cache_get_indices
  | none =>
    let rec find_tree (l : List (Key × FractalTree)) : Option FractalTree :=
      match l with
      | [] => none
      | (k, t) :: xs => match list_decEq tid k with | isTrue _ => some t | isFalse _ => find_tree xs
    match find_tree m.fractal_trees with
    | none => h_cache_get_indices
    | some t =>
      match t.search nid with
      | none => h_cache_get_indices
      | some n =>
        have h_cache_put_indices : (m_after_cache_get.cachePut cache_key n.data).indices.length = m_after_cache_get.indices.length := Eq.refl _
        Eq.trans h_cache_put_indices h_cache_get_indices

def FNDSManager.addPatternToIndex (m : FNDSManager) (index_id : Key) (pattern : Key) (loc : PatternLocation) : FNDSManager :=
  let rec update_index_list (l : List (Key × SelfSimilarIndex)) : List (Key × SelfSimilarIndex) :=
    match l with
    | [] => []
    | (k, idx) :: xs =>
      match list_decEq index_id k with
      | isTrue _ => (k, idx.addPattern pattern loc) :: xs
      | isFalse _ => (k, idx) :: update_index_list xs
  let new_indices := update_index_list m.indices
  let new_stats := { m.statistics with total_patterns_indexed := m.statistics.total_patterns_indexed + 1 }
  { m with indices := new_indices, statistics := new_stats }

theorem FNDSManager.addPatternToIndex_preserves_fractal_trees (m : FNDSManager) (iid p : Key) (loc : PatternLocation) :
  (m.addPatternToIndex iid p loc).fractal_trees = m.fractal_trees := Eq.refl _

theorem FNDSManager.addPatternToIndex_preserves_cache (m : FNDSManager) (iid p : Key) (loc : PatternLocation) :
  (m.addPatternToIndex iid p loc).cache = m.cache := Eq.refl _

theorem FNDSManager.addPatternToIndex_increments_total_patterns (m : FNDSManager) (iid p : Key) (loc : PatternLocation) :
  (m.addPatternToIndex iid p loc).statistics.total_patterns_indexed = m.statistics.total_patterns_indexed + 1 := Eq.refl _

theorem FNDSManager.addPatternToIndex_preserves_total_trees (m : FNDSManager) (iid p : Key) (loc : PatternLocation) :
  (m.addPatternToIndex iid p loc).statistics.total_trees = m.statistics.total_trees := Eq.refl _

theorem FNDSManager.addPatternToIndex_preserves_total_indices (m : FNDSManager) (iid p : Key) (loc : PatternLocation) :
  (m.addPatternToIndex iid p loc).statistics.total_indices = m.statistics.total_indices := Eq.refl _

def FNDSManager.findPatternInIndex (m : FNDSManager) (index_id : Key) (pattern : Key) : List PatternLocation :=
  let rec find_index (l : List (Key × SelfSimilarIndex)) : Option SelfSimilarIndex :=
    match l with
    | [] => none
    | (k, idx) :: xs =>
      match list_decEq index_id k with
      | isTrue _ => some idx
      | isFalse _ => find_index xs
  match find_index m.indices with
  | none => []
  | some idx => idx.findPattern pattern

theorem FNDSManager.findPatternInIndex_empty (c mem : Nat) (iid p : Key) :
  (FNDSManager.init c mem).findPatternInIndex iid p = [] := Eq.refl []

def FNDSManager.removePatternFromIndex (m : FNDSManager) (index_id : Key) (pattern : Key) : FNDSManager :=
  let rec update_index_list (l : List (Key × SelfSimilarIndex)) : List (Key × SelfSimilarIndex) :=
    match l with
    | [] => []
    | (k, idx) :: xs =>
      match list_decEq index_id k with
      | isTrue _ => (k, idx.removePattern pattern) :: xs
      | isFalse _ => (k, idx) :: update_index_list xs
  let new_indices := update_index_list m.indices
  { m with indices := new_indices }

theorem FNDSManager.removePatternFromIndex_preserves_fractal_trees (m : FNDSManager) (iid p : Key) :
  (m.removePatternFromIndex iid p).fractal_trees = m.fractal_trees := Eq.refl _

theorem FNDSManager.removePatternFromIndex_preserves_cache (m : FNDSManager) (iid p : Key) :
  (m.removePatternFromIndex iid p).cache = m.cache := Eq.refl _

theorem FNDSManager.removePatternFromIndex_preserves_statistics (m : FNDSManager) (iid p : Key) :
  (m.removePatternFromIndex iid p).statistics = m.statistics := Eq.refl _

def FNDSManager.deleteFromTree (m : FNDSManager) (tree_id : Key) (node_id : Key) : FNDSManager :=
  let rec update_tree_list (l : List (Key × FractalTree)) : List (Key × FractalTree) :=
    match l with
    | [] => []
    | (k, t) :: xs =>
      match list_decEq tree_id k with
      | isTrue _ => (k, t.delete node_id) :: xs
      | isFalse _ => (k, t) :: update_tree_list xs
  let new_trees := update_tree_list m.fractal_trees
  let new_stats := { m.statistics with total_nodes_across_trees := m.statistics.total_nodes_across_trees - 1 }
  { m with fractal_trees := new_trees, statistics := new_stats }

theorem FNDSManager.deleteFromTree_preserves_indices (m : FNDSManager) (tid nid : Key) :
  (m.deleteFromTree tid nid).indices = m.indices := Eq.refl _

theorem FNDSManager.deleteFromTree_preserves_cache (m : FNDSManager) (tid nid : Key) :
  (m.deleteFromTree tid nid).cache = m.cache := Eq.refl _

theorem FNDSManager.deleteFromTree_decrements_total_nodes (m : FNDSManager) (tid nid : Key) :
  (m.deleteFromTree tid nid).statistics.total_nodes_across_trees = m.statistics.total_nodes_across_trees - 1 := Eq.refl _

theorem FNDSManager.deleteFromTree_preserves_total_trees (m : FNDSManager) (tid nid : Key) :
  (m.deleteFromTree tid nid).statistics.total_trees = m.statistics.total_trees := Eq.refl _

theorem FNDSManager.deleteFromTree_preserves_total_indices (m : FNDSManager) (tid nid : Key) :
  (m.deleteFromTree tid nid).statistics.total_indices = m.statistics.total_indices := Eq.refl _

def FNDSManager.balanceTree (m : FNDSManager) (tree_id : Key) : FNDSManager :=
  let rec update_tree_list (l : List (Key × FractalTree)) : List (Key × FractalTree) :=
    match l with
    | [] => []
    | (k, t) :: xs =>
      match list_decEq tree_id k with
      | isTrue _ => (k, t.balance) :: xs
      | isFalse _ => (k, t) :: update_tree_list xs
  let new_trees := update_tree_list m.fractal_trees
  { m with fractal_trees := new_trees }

theorem FNDSManager.balanceTree_preserves_indices (m : FNDSManager) (tid : Key) :
  (m.balanceTree tid).indices = m.indices := Eq.refl _

theorem FNDSManager.balanceTree_preserves_cache (m : FNDSManager) (tid : Key) :
  (m.balanceTree tid).cache = m.cache := Eq.refl _

theorem FNDSManager.balanceTree_preserves_statistics (m : FNDSManager) (tid : Key) :
  (m.balanceTree tid).statistics = m.statistics := Eq.refl _

def FNDSManager.clearCache (m : FNDSManager) : FNDSManager :=
  { m with cache := LRUCache.init m.cache.capacity m.cache.max_memory }

theorem FNDSManager.clearCache_preserves_fractal_trees (m : FNDSManager) :
  (m.clearCache).fractal_trees = m.fractal_trees := Eq.refl _

theorem FNDSManager.clearCache_preserves_indices (m : FNDSManager) :
  (m.clearCache).indices = m.indices := Eq.refl _

theorem FNDSManager.clearCache_preserves_statistics (m : FNDSManager) :
  (m.clearCache).statistics = m.statistics := Eq.refl _

theorem FNDSManager.clearCache_empties_queue (m : FNDSManager) :
  (m.clearCache).cache.queue = [] := Eq.refl _

theorem FNDSManager.clearCache_empties_map (m : FNDSManager) :
  (m.clearCache).cache.map = [] := Eq.refl _

theorem FNDSManager.clearCache_resets_size (m : FNDSManager) :
  (m.clearCache).cache.current_size = 0 := Eq.refl _

theorem FNDSManager.clearCache_resets_memory (m : FNDSManager) :
  (m.clearCache).cache.current_memory = 0 := Eq.refl _

theorem FNDSManager.clearCache_resets_hits (m : FNDSManager) :
  (m.clearCache).cache.hits = 0 := Eq.refl _

theorem FNDSManager.clearCache_resets_misses (m : FNDSManager) :
  (m.clearCache).cache.misses = 0 := Eq.refl _

theorem FNDSManager.clearCache_resets_evictions (m : FNDSManager) :
  (m.clearCache).cache.evictions = 0 := Eq.refl _

theorem FNDSManager.clearCache_preserves_capacity (m : FNDSManager) :
  (m.clearCache).cache.capacity = m.cache.capacity := Eq.refl _

theorem FNDSManager.clearCache_preserves_max_memory (m : FNDSManager) :
  (m.clearCache).cache.max_memory = m.cache.max_memory := Eq.refl _

def FNDSManager.getTreeCount (m : FNDSManager) : Nat :=
  m.fractal_trees.length

theorem FNDSManager.getTreeCount_init (c mem : Nat) :
  (FNDSManager.init c mem).getTreeCount = 0 := Eq.refl 0

theorem FNDSManager.getTreeCount_addTree (m : FNDSManager) (id : Key) (t : FractalTree) :
  (m.addTree id t).getTreeCount = m.getTreeCount + 1 := Eq.refl _

theorem FNDSManager.getTreeCount_removeTree_le (m : FNDSManager) (id : Key) :
  (m.removeTree id).getTreeCount ≤ m.getTreeCount :=
  remove_key_length_le m.fractal_trees id

def FNDSManager.getIndexCount (m : FNDSManager) : Nat :=
  m.indices.length

theorem FNDSManager.getIndexCount_init (c mem : Nat) :
  (FNDSManager.init c mem).getIndexCount = 0 := Eq.refl 0

theorem FNDSManager.getIndexCount_addIndex (m : FNDSManager) (id : Key) (idx : SelfSimilarIndex) :
  (m.addIndex id idx).getIndexCount = m.getIndexCount + 1 := Eq.refl _

theorem FNDSManager.getIndexCount_removeIndex_le (m : FNDSManager) (id : Key) :
  (m.removeIndex id).getIndexCount ≤ m.getIndexCount :=
  remove_key_length_le m.indices id

def FNDSManager.getCacheHitRatio (m : FNDSManager) : Nat :=
  let total := m.cache.hits + m.cache.misses
  if total = 0 then 0 else m.cache.hits / total

theorem FNDSManager.getCacheHitRatio_init (c mem : Nat) :
  (FNDSManager.init c mem).getCacheHitRatio = 0 := Eq.refl 0

theorem FNDSManager.getCacheHitRatio_clearCache (m : FNDSManager) :
  (m.clearCache).getCacheHitRatio = 0 := Eq.refl 0

def FNDSManager.getCacheSize (m : FNDSManager) : Nat :=
  m.cache.current_size

theorem FNDSManager.getCacheSize_init (c mem : Nat) :
  (FNDSManager.init c mem).getCacheSize = 0 := Eq.refl 0

theorem FNDSManager.getCacheSize_clearCache (m : FNDSManager) :
  (m.clearCache).getCacheSize = 0 := Eq.refl 0

def FNDSManager.getCacheMemoryUsage (m : FNDSManager) : Nat :=
  m.cache.current_memory

theorem FNDSManager.getCacheMemoryUsage_init (c mem : Nat) :
  (FNDSManager.init c mem).getCacheMemoryUsage = 0 := Eq.refl 0

theorem FNDSManager.getCacheMemoryUsage_clearCache (m : FNDSManager) :
  (m.clearCache).getCacheMemoryUsage = 0 := Eq.refl 0

def FNDSManager.getTotalNodesAcrossTrees (m : FNDSManager) : Nat :=
  m.statistics.total_nodes_across_trees

theorem FNDSManager.getTotalNodesAcrossTrees_init (c mem : Nat) :
  (FNDSManager.init c mem).getTotalNodesAcrossTrees = 0 := Eq.refl 0

theorem FNDSManager.getTotalNodesAcrossTrees_insertIntoTree (m : FNDSManager) (tid : Key) (d : Nat) (n : FractalNodeData) :
  (m.insertIntoTree tid d n).getTotalNodesAcrossTrees = m.getTotalNodesAcrossTrees + 1 := Eq.refl _

theorem FNDSManager.getTotalNodesAcrossTrees_deleteFromTree (m : FNDSManager) (tid nid : Key) :
  (m.deleteFromTree tid nid).getTotalNodesAcrossTrees = m.getTotalNodesAcrossTrees - 1 := Eq.refl _

def FNDSManager.getTotalPatternsIndexed (m : FNDSManager) : Nat :=
  m.statistics.total_patterns_indexed

theorem FNDSManager.getTotalPatternsIndexed_init (c mem : Nat) :
  (FNDSManager.init c mem).getTotalPatternsIndexed = 0 := Eq.refl 0

theorem FNDSManager.getTotalPatternsIndexed_addPatternToIndex (m : FNDSManager) (iid p : Key) (loc : PatternLocation) :
  (m.addPatternToIndex iid p loc).getTotalPatternsIndexed = m.getTotalPatternsIndexed + 1 := Eq.refl _

def FNDSManager.getAverageTreeDepth (m : FNDSManager) : Nat :=
  m.statistics.average_tree_depth

theorem FNDSManager.getAverageTreeDepth_init (c mem : Nat) :
  (FNDSManager.init c mem).getAverageTreeDepth = 0 := Eq.refl 0

def FNDSManager.getMemoryUsed (m : FNDSManager) : Nat :=
  m.statistics.memory_used

theorem FNDSManager.getMemoryUsed_init (c mem : Nat) :
  (FNDSManager.init c mem).getMemoryUsed = 0 := Eq.refl 0

def FNDSManager.getTotalTrees (m : FNDSManager) : Nat :=
  m.statistics.total_trees

theorem FNDSManager.getTotalTrees_init (c mem : Nat) :
  (FNDSManager.init c mem).getTotalTrees = 0 := Eq.refl 0

theorem FNDSManager.getTotalTrees_addTree (m : FNDSManager) (id : Key) (t : FractalTree) :
  (m.addTree id t).getTotalTrees = m.getTotalTrees + 1 := Eq.refl _

def FNDSManager.getTotalIndices (m : FNDSManager) : Nat :=
  m.statistics.total_indices

theorem FNDSManager.getTotalIndices_init (c mem : Nat) :
  (FNDSManager.init c mem).getTotalIndices = 0 := Eq.refl 0

theorem FNDSManager.getTotalIndices_addIndex (m : FNDSManager) (id : Key) (idx : SelfSimilarIndex) :
  (m.addIndex id idx).getTotalIndices = m.getTotalIndices + 1 := Eq.refl _

def FNDSManager.getCacheHits (m : FNDSManager) : Nat :=
  m.statistics.cache_hits

theorem FNDSManager.getCacheHits_init (c mem : Nat) :
  (FNDSManager.init c mem).getCacheHits = 0 := Eq.refl 0

def FNDSManager.getCacheMisses (m : FNDSManager) : Nat :=
  m.statistics.cache_misses

theorem FNDSManager.getCacheMisses_init (c mem : Nat) :
  (FNDSManager.init c mem).getCacheMisses = 0 := Eq.refl 0

theorem FNDSManager.cacheGet_miss_increments_misses (m : FNDSManager) (k : Key) (h_miss : m.cache.get k = (none, m.cache.misses + 1)) :
  (m.cacheGet k).snd.statistics.cache_misses = m.statistics.cache_misses + 1 :=
  have h_eval : (m.cacheGet k).snd.statistics = (match (m.cache.get k).fst with | none => m.statistics.recordCacheMiss | some _ => m.statistics.recordCacheHit) := Eq.refl _
  have h_fst : (m.cache.get k).fst = none := congrArg Prod.fst h_miss
  have h_match : (match (m.cache.get k).fst with | none => m.statistics.recordCacheMiss | some _ => m.statistics.recordCacheHit) = m.statistics.recordCacheMiss :=
    Eq.subst (motive := fun x => (match x with | none => m.statistics.recordCacheMiss | some _ => m.statistics.recordCacheHit) = m.statistics.recordCacheMiss) (Eq.symm h_fst) (Eq.refl _)
  have h_record : m.statistics.recordCacheMiss.cache_misses = m.statistics.cache_misses + 1 := Eq.refl _
  Eq.trans (congrArg FNDSStatistics.cache_misses h_eval) (Eq.trans (congrArg FNDSStatistics.cache_misses h_match) h_record)

theorem FNDSManager.cacheGet_hit_increments_hits (m : FNDSManager) (k : Key) (v : Value) (h_hit : m.cache.get k = (some v, m.cache.hits + 1)) :
  (m.cacheGet k).snd.statistics.cache_hits = m.statistics.cache_hits + 1 :=
  have h_eval : (m.cacheGet k).snd.statistics = (match (m.cache.get k).fst with | none => m.statistics.recordCacheMiss | some _ => m.statistics.recordCacheHit) := Eq.refl _
  have h_fst : (m.cache.get k).fst = some v := congrArg Prod.fst h_hit
  have h_match : (match (m.cache.get k).fst with | none => m.statistics.recordCacheMiss | some _ => m.statistics.recordCacheHit) = m.statistics.recordCacheHit :=
    Eq.subst (motive := fun x => (match x with | none => m.statistics.recordCacheMiss | some _ => m.statistics.recordCacheHit) = m.statistics.recordCacheHit) (Eq.symm h_fst) (Eq.refl _)
  have h_record : m.statistics.recordCacheHit.cache_hits = m.statistics.cache_hits + 1 := Eq.refl _
  Eq.trans (congrArg FNDSStatistics.cache_hits h_eval) (Eq.trans (congrArg FNDSStatistics.cache_hits h_match) h_record)

theorem FNDSManager.cacheGet_preserves_fractal_trees (m : FNDSManager) (k : Key) :
  (m.cacheGet k).snd.fractal_trees = m.fractal_trees := Eq.refl _

theorem FNDSManager.cacheGet_preserves_indices (m : FNDSManager) (k : Key) :
  (m.cacheGet k).snd.indices = m.indices := Eq.refl _

theorem FNDSManager.cacheGet_preserves_total_trees (m : FNDSManager) (k : Key) :
  (m.cacheGet k).snd.statistics.total_trees = m.statistics.total_trees :=
  match (m.cache.get k).fst with
  | none => Eq.refl _
  | some _ => Eq.refl _

theorem FNDSManager.cacheGet_preserves_total_indices (m : FNDSManager) (k : Key) :
  (m.cacheGet k).snd.statistics.total_indices = m.statistics.total_indices :=
  match (m.cache.get k).fst with
  | none => Eq.refl _
  | some _ => Eq.refl _

theorem FNDSManager.cacheGet_preserves_total_nodes_across_trees (m : FNDSManager) (k : Key) :
  (m.cacheGet k).snd.statistics.total_nodes_across_trees = m.statistics.total_nodes_across_trees :=
  match (m.cache.get k).fst with
  | none => Eq.refl _
  | some _ => Eq.refl _

theorem FNDSManager.cacheGet_preserves_total_patterns_indexed (m : FNDSManager) (k : Key) :
  (m.cacheGet k).snd.statistics.total_patterns_indexed = m.statistics.total_patterns_indexed :=
  match (m.cache.get k).fst with
  | none => Eq.refl _
  | some _ => Eq.refl _

theorem FNDSManager.cacheGet_preserves_average_tree_depth (m : FNDSManager) (k : Key) :
  (m.cacheGet k).snd.statistics.average_tree_depth = m.statistics.average_tree_depth :=
  match (m.cache.get k).fst with
  | none => Eq.refl _
  | some _ => Eq.refl _

theorem FNDSManager.cacheGet_preserves_memory_used (m : FNDSManager) (k : Key) :
  (m.cacheGet k).snd.statistics.memory_used = m.statistics.memory_used :=
  match (m.cache.get k).fst with
  | none => Eq.refl _
  | some _ => Eq.refl _

theorem FNDSManager.cacheGet_preserves_cache_hit_ratio (m : FNDSManager) (k : Key) :
  (m.cacheGet k).snd.statistics.cache_hit_ratio = m.statistics.cache_hit_ratio :=
  match (m.cache.get k).fst with
  | none => Eq.refl _
  | some _ => Eq.refl _
