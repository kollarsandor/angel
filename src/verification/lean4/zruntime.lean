

import Mathlib

inductive HistoryEntryType
| assign
| transform
| relate
| measure
| entangle

inductive ExecutionAction
| create_variable
| delete_variable
| relational_operation
| entangle_variables
| propagate_information
| fractal_transform
| measure
| quantum_circuit
| relational_expression

inductive RelationalOperationType
| op_and
| op_or
| op_xor
| op_entangle

structure MyComplex where
  re : Float
  im : Float

structure QuantumState where
  alpha_re : Float
  alpha_im : Float
  beta_re : Float
  beta_im : Float

structure MeasurementResult where
  result : Nat
  probability : Float
  collapsed_state : QuantumState

inductive EdgeQuality
| coherent
| entangled
| decoherent
| weak
| strong

inductive LogicGate
| HADAMARD
| PAULI_X
| PAULI_Y
| PAULI_Z
| CNOT
| RELATIONAL_AND
| RELATIONAL_OR
| RELATIONAL_XOR
| FRACTAL_TRANSFORM

structure HistoryEntry where
  entry_type : HistoryEntryType
  value : String
  timestamp : Int

structure ExecutionHistoryEntry where
  action : ExecutionAction
  primary_target : String
  secondary_targets : List String
  operation_type : Option String
  result_value : Option String
  result_int : Option Int
  result_float : Option Float
  timestamp : Int

structure Node where
  id : String
  data : String
  quantum_state : QuantumState

structure Edge where
  src : String
  dst : String
  quality : EdgeQuality
  weight : Float
  correlation : MyComplex
  fractal_dim : Float

structure Graph where
  nodes : List Node
  edges : List Edge

structure ZVariable where
  name : String
  graph : Graph
  history : List HistoryEntry
  quantum_states : List QuantumState
  last_node_id : Option String

structure ZRuntime where
  variables : List (String × ZVariable)
  global_graph : Graph
  execution_history : List ExecutionHistoryEntry

structure VariableState where
  name : String
  value : Option String
  node_count : Nat
  edge_count : Nat
  fractal_dimension : Float
  topology_hash : String
  state_count : Nat
  history_count : Nat

structure SystemState where
  variable_count : Nat
  total_nodes : Nat
  total_edges : Nat
  average_fractal_dimension : Float
  execution_history_length : Nat
  variables : List VariableState

structure GateSpec where
  gate_name : String
  indices : List Nat
  params : Option (List Float)

instance : Inhabited HistoryEntryType where default := .assign
instance : Inhabited ExecutionAction where default := .create_variable
instance : Inhabited RelationalOperationType where default := .op_and
instance : Inhabited EdgeQuality where default := .coherent
instance : Inhabited LogicGate where default := .HADAMARD
instance : Inhabited MyComplex where default := { re := 0.0, im := 0.0 }
instance : Inhabited QuantumState where default := { alpha_re := 1.0, alpha_im := 0.0, beta_re := 0.0, beta_im := 0.0 }
instance : Inhabited MeasurementResult where default := { result := 0, probability := 1.0, collapsed_state := default }
instance : Inhabited HistoryEntry where default := { entry_type := default, value := "", timestamp := 0 }
instance : Inhabited ExecutionHistoryEntry where default := { action := default, primary_target := "", secondary_targets :=[], operation_type := none, result_value := none, result_int := none, result_float := none, timestamp := 0 }
instance : Inhabited Node where default := { id := "", data := "", quantum_state := default }
instance : Inhabited Edge where default := { src := "", dst := "", quality := default, weight := 0.0, correlation := default, fractal_dim := 0.0 }
instance : Inhabited Graph where default := { nodes :=[], edges :=[] }
instance : Inhabited ZVariable where default := { name := "", graph := default, history := [], quantum_states :=[], last_node_id := none }
instance : Inhabited ZRuntime where default := { variables := [], global_graph := default, execution_history :=[] }
instance : Inhabited VariableState where default := { name := "", value := none, node_count := 0, edge_count := 0, fractal_dimension := 0.0, topology_hash := "", state_count := 0, history_count := 0 }
instance : Inhabited SystemState where default := { variable_count := 0, total_nodes := 0, total_edges := 0, average_fractal_dimension := 0.0, execution_history_length := 0, variables :=[] }
instance : Inhabited GateSpec where default := { gate_name := "", indices :=[], params := none }

theorem HistoryEntryType_eq_decidable (a b : HistoryEntryType) : a = b ∨ a ≠ b := by
  cases a
  case assign =>
    cases b
    case assign =>
      apply Or.inl
      rfl
    case transform =>
      apply Or.inr
      intro h
      cases h
    case relate =>
      apply Or.inr
      intro h
      cases h
    case measure =>
      apply Or.inr
      intro h
      cases h
    case entangle =>
      apply Or.inr
      intro h
      cases h
  case transform =>
    cases b
    case assign =>
      apply Or.inr
      intro h
      cases h
    case transform =>
      apply Or.inl
      rfl
    case relate =>
      apply Or.inr
      intro h
      cases h
    case measure =>
      apply Or.inr
      intro h
      cases h
    case entangle =>
      apply Or.inr
      intro h
      cases h
  case relate =>
    cases b
    case assign =>
      apply Or.inr
      intro h
      cases h
    case transform =>
      apply Or.inr
      intro h
      cases h
    case relate =>
      apply Or.inl
      rfl
    case measure =>
      apply Or.inr
      intro h
      cases h
    case entangle =>
      apply Or.inr
      intro h
      cases h
  case measure =>
    cases b
    case assign =>
      apply Or.inr
      intro h
      cases h
    case transform =>
      apply Or.inr
      intro h
      cases h
    case relate =>
      apply Or.inr
      intro h
      cases h
    case measure =>
      apply Or.inl
      rfl
    case entangle =>
      apply Or.inr
      intro h
      cases h
  case entangle =>
    cases b
    case assign =>
      apply Or.inr
      intro h
      cases h
    case transform =>
      apply Or.inr
      intro h
      cases h
    case relate =>
      apply Or.inr
      intro h
      cases h
    case measure =>
      apply Or.inr
      intro h
      cases h
    case entangle =>
      apply Or.inl
      rfl
  done

theorem RelationalOperationType_eq_decidable (a b : RelationalOperationType) : a = b ∨ a ≠ b := by
  cases a
  case op_and =>
    cases b
    case op_and =>
      apply Or.inl
      rfl
    case op_or =>
      apply Or.inr
      intro h
      cases h
    case op_xor =>
      apply Or.inr
      intro h
      cases h
    case op_entangle =>
      apply Or.inr
      intro h
      cases h
  case op_or =>
    cases b
    case op_and =>
      apply Or.inr
      intro h
      cases h
    case op_or =>
      apply Or.inl
      rfl
    case op_xor =>
      apply Or.inr
      intro h
      cases h
    case op_entangle =>
      apply Or.inr
      intro h
      cases h
  case op_xor =>
    cases b
    case op_and =>
      apply Or.inr
      intro h
      cases h
    case op_or =>
      apply Or.inr
      intro h
      cases h
    case op_xor =>
      apply Or.inl
      rfl
    case op_entangle =>
      apply Or.inr
      intro h
      cases h
  case op_entangle =>
    cases b
    case op_and =>
      apply Or.inr
      intro h
      cases h
    case op_or =>
      apply Or.inr
      intro h
      cases h
    case op_xor =>
      apply Or.inr
      intro h
      cases h
    case op_entangle =>
      apply Or.inl
      rfl
  done

theorem EdgeQuality_eq_decidable (a b : EdgeQuality) : a = b ∨ a ≠ b := by
  cases a
  case coherent =>
    cases b
    case coherent =>
      apply Or.inl
      rfl
    case entangled =>
      apply Or.inr
      intro h
      cases h
    case decoherent =>
      apply Or.inr
      intro h
      cases h
    case weak =>
      apply Or.inr
      intro h
      cases h
    case strong =>
      apply Or.inr
      intro h
      cases h
  case entangled =>
    cases b
    case coherent =>
      apply Or.inr
      intro h
      cases h
    case entangled =>
      apply Or.inl
      rfl
    case decoherent =>
      apply Or.inr
      intro h
      cases h
    case weak =>
      apply Or.inr
      intro h
      cases h
    case strong =>
      apply Or.inr
      intro h
      cases h
  case decoherent =>
    cases b
    case coherent =>
      apply Or.inr
      intro h
      cases h
    case entangled =>
      apply Or.inr
      intro h
      cases h
    case decoherent =>
      apply Or.inl
      rfl
    case weak =>
      apply Or.inr
      intro h
      cases h
    case strong =>
      apply Or.inr
      intro h
      cases h
  case weak =>
    cases b
    case coherent =>
      apply Or.inr
      intro h
      cases h
    case entangled =>
      apply Or.inr
      intro h
      cases h
    case decoherent =>
      apply Or.inr
      intro h
      cases h
    case weak =>
      apply Or.inl
      rfl
    case strong =>
      apply Or.inr
      intro h
      cases h
  case strong =>
    cases b
    case coherent =>
      apply Or.inr
      intro h
      cases h
    case entangled =>
      apply Or.inr
      intro h
      cases h
    case decoherent =>
      apply Or.inr
      intro h
      cases h
    case weak =>
      apply Or.inr
      intro h
      cases h
    case strong =>
      apply Or.inl
      rfl
  done

theorem MyComplex_ext (a b : MyComplex) (hre : a.re = b.re) (him : a.im = b.im) : a = b := by
  cases a
  cases b
  dsimp only at hre
  dsimp only at him
  cases hre
  cases him
  rfl
  done

theorem QuantumState_ext (a b : QuantumState) (h1 : a.alpha_re = b.alpha_re) (h2 : a.alpha_im = b.alpha_im) (h3 : a.beta_re = b.beta_re) (h4 : a.beta_im = b.beta_im) : a = b := by
  cases a
  cases b
  dsimp only at h1
  dsimp only at h2
  dsimp only at h3
  dsimp only at h4
  cases h1
  cases h2
  cases h3
  cases h4
  rfl
  done

theorem MeasurementResult_ext (a b : MeasurementResult) (h1 : a.result = b.result) (h2 : a.probability = b.probability) (h3 : a.collapsed_state = b.collapsed_state) : a = b := by
  cases a
  cases b
  dsimp only at h1
  dsimp only at h2
  dsimp only at h3
  cases h1
  cases h2
  cases h3
  rfl
  done

theorem HistoryEntry_ext (a b : HistoryEntry) (h1 : a.entry_type = b.entry_type) (h2 : a.value = b.value) (h3 : a.timestamp = b.timestamp) : a = b := by
  cases a
  cases b
  dsimp only at h1
  dsimp only at h2
  dsimp only at h3
  cases h1
  cases h2
  cases h3
  rfl
  done

theorem ExecutionHistoryEntry_ext (a b : ExecutionHistoryEntry) (h1 : a.action = b.action) (h2 : a.primary_target = b.primary_target) (h3 : a.secondary_targets = b.secondary_targets) (h4 : a.operation_type = b.operation_type) (h5 : a.result_value = b.result_value) (h6 : a.result_int = b.result_int) (h7 : a.result_float = b.result_float) (h8 : a.timestamp = b.timestamp) : a = b := by
  cases a
  cases b
  dsimp only at h1
  dsimp only at h2
  dsimp only at h3
  dsimp only at h4
  dsimp only at h5
  dsimp only at h6
  dsimp only at h7
  dsimp only at h8
  cases h1
  cases h2
  cases h3
  cases h4
  cases h5
  cases h6
  cases h7
  cases h8
  rfl
  done

theorem Node_ext (a b : Node) (h1 : a.id = b.id) (h2 : a.data = b.data) (h3 : a.quantum_state = b.quantum_state) : a = b := by
  cases a
  cases b
  dsimp only at h1
  dsimp only at h2
  dsimp only at h3
  cases h1
  cases h2
  cases h3
  rfl
  done

theorem Edge_ext (a b : Edge) (h1 : a.src = b.src) (h2 : a.dst = b.dst) (h3 : a.quality = b.quality) (h4 : a.weight = b.weight) (h5 : a.correlation = b.correlation) (h6 : a.fractal_dim = b.fractal_dim) : a = b := by
  cases a
  cases b
  dsimp only at h1
  dsimp only at h2
  dsimp only at h3
  dsimp only at h4
  dsimp only at h5
  dsimp only at h6
  cases h1
  cases h2
  cases h3
  cases h4
  cases h5
  cases h6
  rfl
  done

theorem Graph_ext (a b : Graph) (h1 : a.nodes = b.nodes) (h2 : a.edges = b.edges) : a = b := by
  cases a
  cases b
  dsimp only at h1
  dsimp only at h2
  cases h1
  cases h2
  rfl
  done

theorem ZVariable_ext (a b : ZVariable) (h1 : a.name = b.name) (h2 : a.graph = b.graph) (h3 : a.history = b.history) (h4 : a.quantum_states = b.quantum_states) (h5 : a.last_node_id = b.last_node_id) : a = b := by
  cases a
  cases b
  dsimp only at h1
  dsimp only at h2
  dsimp only at h3
  dsimp only at h4
  dsimp only at h5
  cases h1
  cases h2
  cases h3
  cases h4
  cases h5
  rfl
  done

theorem ZRuntime_ext (a b : ZRuntime) (h1 : a.variables = b.variables) (h2 : a.global_graph = b.global_graph) (h3 : a.execution_history = b.execution_history) : a = b := by
  cases a
  cases b
  dsimp only at h1
  dsimp only at h2
  dsimp only at h3
  cases h1
  cases h2
  cases h3
  rfl
  done

theorem VariableState_ext (a b : VariableState) (h1 : a.name = b.name) (h2 : a.value = b.value) (h3 : a.node_count = b.node_count) (h4 : a.edge_count = b.edge_count) (h5 : a.fractal_dimension = b.fractal_dimension) (h6 : a.topology_hash = b.topology_hash) (h7 : a.state_count = b.state_count) (h8 : a.history_count = b.history_count) : a = b := by
  cases a
  cases b
  dsimp only at h1
  dsimp only at h2
  dsimp only at h3
  dsimp only at h4
  dsimp only at h5
  dsimp only at h6
  dsimp only at h7
  dsimp only at h8
  cases h1
  cases h2
  cases h3
  cases h4
  cases h5
  cases h6
  cases h7
  cases h8
  rfl
  done

theorem SystemState_ext (a b : SystemState) (h1 : a.variable_count = b.variable_count) (h2 : a.total_nodes = b.total_nodes) (h3 : a.total_edges = b.total_edges) (h4 : a.average_fractal_dimension = b.average_fractal_dimension) (h5 : a.execution_history_length = b.execution_history_length) (h6 : a.variables = b.variables) : a = b := by
  cases a
  cases b
  dsimp only at h1
  dsimp only at h2
  dsimp only at h3
  dsimp only at h4
  dsimp only at h5
  dsimp only at h6
  cases h1
  cases h2
  cases h3
  cases h4
  cases h5
  cases h6
  rfl
  done

theorem GateSpec_ext (a b : GateSpec) (h1 : a.gate_name = b.gate_name) (h2 : a.indices = b.indices) (h3 : a.params = b.params) : a = b := by
  cases a
  cases b
  dsimp only at h1
  dsimp only at h2
  dsimp only at h3
  cases h1
  cases h2
  cases h3
  rfl
  done

def ZVariable.empty (name : String) : ZVariable :=
  { name := name, graph := { nodes := [], edges := [] }, history := [], quantum_states :=[], last_node_id := none }

def ZRuntime.empty : ZRuntime :=
  { variables := [], global_graph := { nodes := [], edges :=[] }, execution_history :=[] }

def Graph.nodeCount (g : Graph) : Nat :=
  g.nodes.length

def Graph.edgeCount (g : Graph) : Nat :=
  g.edges.length

def ZRuntime.variableCount (r : ZRuntime) : Nat :=
  r.variables.length

def ZVariable.historyCount (v : ZVariable) : Nat :=
  v.history.length

def ZVariable.stateCount (v : ZVariable) : Nat :=
  v.quantum_states.length

def ZRuntime.hasVariable (r : ZRuntime) (name : String) : Bool :=
  r.variables.any (fun p => p.1 == name)

def ZRuntime.getVariable (r : ZRuntime) (name : String) : Option ZVariable :=
  (r.variables.find? (fun p => p.1 == name)).map Prod.snd

def HistoryEntryType.toString : HistoryEntryType → String
| .assign => "assign"
| .transform => "transform"
| .relate => "relate"
| .measure => "measure"
| .entangle => "entangle"

def ExecutionAction.toString : ExecutionAction → String
| .create_variable => "create_variable"
| .delete_variable => "delete_variable"
| .relational_operation => "relational_operation"
| .entangle_variables => "entangle_variables"
| .propagate_information => "propagate_information"
| .fractal_transform => "fractal_transform"
| .measure => "measure"
| .quantum_circuit => "quantum_circuit"
| .relational_expression => "relational_expression"

def RelationalOperationType.toString : RelationalOperationType → String
| .op_and => "and"
| .op_or => "or"
| .op_xor => "xor"
| .op_entangle => "entangle"

def RelationalOperationType.fromString (s : String) : Option RelationalOperationType :=
  if s.toLower == "and" then some .op_and
  else if s.toLower == "or" then some .op_or
  else if s.toLower == "xor" then some .op_xor
  else if s.toLower == "entangle" then some .op_entangle
  else none

def RelationalOperationType.toGate : RelationalOperationType → Option LogicGate
| .op_and => some .RELATIONAL_AND
| .op_or => some .RELATIONAL_OR
| .op_xor => some .RELATIONAL_XOR
| .op_entangle => none

def MyComplex.mul (a b : MyComplex) : MyComplex :=
  { re := a.re * b.re - a.im * b.im, im := a.re * b.im + a.im * b.re }

def MyComplex.conjugate (c : MyComplex) : MyComplex :=
  { re := c.re, im := -c.im }

def MyComplex.magnitude (c : MyComplex) : Float :=
  Float.sqrt (c.re * c.re + c.im * c.im)

def QuantumState.isNormalized (s : QuantumState) : Prop :=
  s.alpha_re^2 + s.alpha_im^2 + s.beta_re^2 + s.beta_im^2 = 1

def ZVariable.assign (v : ZVariable) (value : String) (timestamp : Int) : ZVariable :=
  let newNode : Node := { id := s!"node_{v.graph.nodes.length}", data := value, quantum_state := { alpha_re := 1, alpha_im := 0, beta_re := 0, beta_im := 0 } }
  let newHistory : HistoryEntry := { entry_type := .assign, value := value, timestamp := timestamp }
  { v with graph := { v.graph with nodes := v.graph.nodes ++ [newNode] }, history := v.history ++ [newHistory], quantum_states := v.quantum_states ++ [newNode.quantum_state], last_node_id := some newNode.id }

def ZRuntime.createVariable (r : ZRuntime) (name : String) (initial_value : Option String) (timestamp : Int) : Option ZRuntime :=
  if r.hasVariable name then none
  else
    let newVar := match initial_value with
      | some val => (ZVariable.empty name).assign val timestamp
      | none => ZVariable.empty name
    let newHistory : ExecutionHistoryEntry := { action := .create_variable, primary_target := name, secondary_targets :=[], operation_type := none, result_value := initial_value, result_int := none, result_float := none, timestamp := timestamp }
    some { r with variables := r.variables ++ [(name, newVar)], execution_history := r.execution_history ++ [newHistory] }

def ZRuntime.deleteVariable (r : ZRuntime) (name : String) (timestamp : Int) : ZRuntime × Bool :=
  if r.hasVariable name then
    let newHistory : ExecutionHistoryEntry := { action := .delete_variable, primary_target := name, secondary_targets :=[], operation_type := none, result_value := none, result_int := none, result_float := none, timestamp := timestamp }
    ({ r with variables := r.variables.filter (fun p => p.1 != name), execution_history := r.execution_history ++ [newHistory] }, true)
  else (r, false)

def ZVariable.getValue (v : ZVariable) : Option String :=
  match v.last_node_id with
  | none => none
  | some id => (v.graph.nodes.find? (fun n => n.id == id)).map Node.data

def Graph.addEdge (g : Graph) (e : Edge) : Graph :=
  { g with edges := g.edges ++ [e] }

def Graph.addNode (g : Graph) (n : Node) : Graph :=
  { g with nodes := g.nodes ++ [n] }

def ZRuntime.reset : ZRuntime :=
  ZRuntime.empty

def ZVariable.clearHistory (v : ZVariable) : ZVariable :=
  { v with history :=[] }

def ZRuntime.clearExecutionHistory (r : ZRuntime) : ZRuntime :=
  { r with execution_history :=[] }

def ZVariable.getFractalDimension (v : ZVariable) : Float :=
  let n := v.graph.nodeCount
  let e := v.graph.edgeCount
  if n < 2 ∨ e = 0 then 0 else Float.log (Float.ofNat e) / Float.log (Float.ofNat n)

theorem toString_assign_correct : HistoryEntryType.assign.toString = "assign" := by
  rfl
  done

theorem toString_transform_correct : HistoryEntryType.transform.toString = "transform" := by
  rfl
  done

theorem toString_relate_correct : HistoryEntryType.relate.toString = "relate" := by
  rfl
  done

theorem toString_measure_correct : HistoryEntryType.measure.toString = "measure" := by
  rfl
  done

theorem toString_entangle_correct : HistoryEntryType.entangle.toString = "entangle" := by
  rfl
  done

theorem exec_action_toString_create : ExecutionAction.create_variable.toString = "create_variable" := by
  rfl
  done

theorem exec_action_toString_delete : ExecutionAction.delete_variable.toString = "delete_variable" := by
  rfl
  done

theorem rel_op_toString_and : RelationalOperationType.op_and.toString = "and" := by
  rfl
  done

theorem rel_op_toString_or : RelationalOperationType.op_or.toString = "or" := by
  rfl
  done

theorem rel_op_toString_xor : RelationalOperationType.op_xor.toString = "xor" := by
  rfl
  done

theorem rel_op_toString_entangle : RelationalOperationType.op_entangle.toString = "entangle" := by
  rfl
  done

theorem rel_op_toGate_and : RelationalOperationType.op_and.toGate = some LogicGate.RELATIONAL_AND := by
  rfl
  done

theorem rel_op_toGate_or : RelationalOperationType.op_or.toGate = some LogicGate.RELATIONAL_OR := by
  rfl
  done

theorem rel_op_toGate_xor : RelationalOperationType.op_xor.toGate = some LogicGate.RELATIONAL_XOR := by
  rfl
  done

theorem rel_op_toGate_entangle : RelationalOperationType.op_entangle.toGate = none := by
  rfl
  done

theorem empty_runtime_variable_count : ZRuntime.empty.variableCount = 0 := by
  rfl
  done

theorem empty_runtime_has_no_variables (name : String) : ZRuntime.empty.hasVariable name = false := by
  dsimp only [ZRuntime.empty]
  dsimp only[ZRuntime.hasVariable]
  dsimp only [List.any]
  rfl
  done

theorem empty_variable_node_count (name : String) : (ZVariable.empty name).graph.nodeCount = 0 := by
  rfl
  done

theorem empty_variable_edge_count (name : String) : (ZVariable.empty name).graph.edgeCount = 0 := by
  rfl
  done

theorem empty_variable_history_count (name : String) : (ZVariable.empty name).historyCount = 0 := by
  rfl
  done

theorem empty_variable_state_count (name : String) : (ZVariable.empty name).stateCount = 0 := by
  rfl
  done

theorem empty_variable_no_value (name : String) : (ZVariable.empty name).getValue = none := by
  rfl
  done

theorem assign_increases_node_count (v : ZVariable) (val : String) (t : Int) : (v.assign val t).graph.nodeCount = v.graph.nodeCount + 1 := by
  dsimp only [ZVariable.assign]
  dsimp only [Graph.nodeCount]
  exact List.length_append v.graph.nodes[{ id := s!"node_{v.graph.nodes.length}", data := val, quantum_state := { alpha_re := 1, alpha_im := 0, beta_re := 0, beta_im := 0 } }]
  done

theorem assign_increases_history_count (v : ZVariable) (val : String) (t : Int) : (v.assign val t).historyCount = v.historyCount + 1 := by
  dsimp only [ZVariable.assign]
  dsimp only[ZVariable.historyCount]
  exact List.length_append v.history[{ entry_type := HistoryEntryType.assign, value := val, timestamp := t }]
  done

theorem assign_increases_state_count (v : ZVariable) (val : String) (t : Int) : (v.assign val t).stateCount = v.stateCount + 1 := by
  dsimp only [ZVariable.assign]
  dsimp only [ZVariable.stateCount]
  exact List.length_append v.quantum_states[{ alpha_re := 1, alpha_im := 0, beta_re := 0, beta_im := 0 }]
  done

theorem assign_sets_last_node_id (v : ZVariable) (val : String) (t : Int) : (v.assign val t).last_node_id.isSome := by
  dsimp only [ZVariable.assign]
  dsimp only [Option.isSome]
  rfl
  done

theorem createVariable_fails_if_exists (r : ZRuntime) (name : String) (val : Option String) (t : Int) (h : r.hasVariable name = true) : r.createVariable name val t = none := by
  dsimp only[ZRuntime.createVariable]
  cases h_cond : r.hasVariable name
  case true =>
    rfl
  case false =>
    have h_false : false = true := Eq.trans (Eq.symm h_cond) h
    cases h_false
  done

theorem createVariable_succeeds_if_not_exists (r : ZRuntime) (name : String) (val : Option String) (t : Int) (h : r.hasVariable name = false) : (r.createVariable name val t).isSome = true := by
  dsimp only [ZRuntime.createVariable]
  cases h_cond : r.hasVariable name
  case false =>
    dsimp only [Option.isSome]
    rfl
  case true =>
    have h_false : true = false := Eq.trans (Eq.symm h_cond) h
    cases h_false
  done

theorem createVariable_increases_count (r : ZRuntime) (name : String) (val : Option String) (t : Int) (h : r.hasVariable name = false) : ∃ r', r.createVariable name val t = some r' ∧ r'.variableCount = r.variableCount + 1 := by
  dsimp only[ZRuntime.createVariable]
  cases h_cond : r.hasVariable name
  case false =>
    apply Exists.intro
    apply And.intro
    case left =>
      rfl
    case right =>
      dsimp only [ZRuntime.variableCount]
      exact List.length_append r.variables[(name, match val with | some val_1 => ZVariable.assign (ZVariable.empty name) val_1 t | none => ZVariable.empty name)]
  case true =>
    have h_false : true = false := Eq.trans (Eq.symm h_cond) h
    cases h_false
  done

theorem createVariable_adds_history (r : ZRuntime) (name : String) (val : Option String) (t : Int) (h : r.hasVariable name = false) : ∃ r', r.createVariable name val t = some r' ∧ r'.execution_history.length = r.execution_history.length + 1 := by
  dsimp only [ZRuntime.createVariable]
  cases h_cond : r.hasVariable name
  case false =>
    apply Exists.intro
    apply And.intro
    case left =>
      rfl
    case right =>
      exact List.length_append r.execution_history [{ action := ExecutionAction.create_variable, primary_target := name, secondary_targets := [], operation_type := none, result_value := val, result_int := none, result_float := none, timestamp := t }]
  case true =>
    have h_false : true = false := Eq.trans (Eq.symm h_cond) h
    cases h_false
  done

theorem neq_true_implies_eq_false (a b : String) (h : (a != b) = true) : (a == b) = false := by
  cases h_eq : (a == b)
  case false =>
    rfl
  case true =>
    have h_not : (!true) = true := Eq.ndrec (motive := fun x => (!x) = true) h h_eq
    cases h_not
  done

theorem deleteVariable_removes_from_list (r : ZRuntime) (name : String) (t : Int) (h : r.hasVariable name = true) : (r.deleteVariable name t).1.hasVariable name = false := by
  dsimp only [ZRuntime.deleteVariable]
  cases h_cond : r.hasVariable name
  case false =>
    have h_false : false = true := Eq.trans (Eq.symm h_cond) h
    cases h_false
  case true =>
    dsimp only [ZRuntime.hasVariable]
    have h_ind : ∀ (l : List (String × ZVariable)), List.any (List.filter (fun p => p.1 != name) l) (fun p => p.1 == name) = false := by
      intro l
      induction l
      case nil =>
        rfl
      case cons hd tl ih =>
        dsimp only [List.filter]
        cases h_neq : (hd.1 != name)
        case false =>
          exact ih
        case true =>
          dsimp only [List.any]
          cases h_eq : (hd.1 == name)
          case false =>
            exact ih
          case true =>
            have h_not : (!true) = true := Eq.ndrec (motive := fun x => (!x) = true) h_neq h_eq
            cases h_not
    exact h_ind r.variables
  done

theorem deleteVariable_returns_true_if_exists (r : ZRuntime) (name : String) (t : Int) (h : r.hasVariable name = true) : (r.deleteVariable name t).2 = true := by
  dsimp only [ZRuntime.deleteVariable]
  cases h_cond : r.hasVariable name
  case false =>
    have h_false : false = true := Eq.trans (Eq.symm h_cond) h
    cases h_false
  case true =>
    rfl
  done

theorem deleteVariable_returns_false_if_not_exists (r : ZRuntime) (name : String) (t : Int) (h : r.hasVariable name = false) : (r.deleteVariable name t).2 = false := by
  dsimp only[ZRuntime.deleteVariable]
  cases h_cond : r.hasVariable name
  case false =>
    rfl
  case true =>
    have h_false : true = false := Eq.trans (Eq.symm h_cond) h
    cases h_false
  done

theorem deleteVariable_unchanged_if_not_exists (r : ZRuntime) (name : String) (t : Int) (h : r.hasVariable name = false) : (r.deleteVariable name t).1 = r := by
  dsimp only [ZRuntime.deleteVariable]
  cases h_cond : r.hasVariable name
  case false =>
    rfl
  case true =>
    have h_false : true = false := Eq.trans (Eq.symm h_cond) h
    cases h_false
  done

theorem reset_clears_all_variables : ZRuntime.reset.variableCount = 0 := by
  rfl
  done

theorem reset_clears_execution_history : ZRuntime.reset.execution_history.length = 0 := by
  rfl
  done

theorem clearHistory_empties_history (v : ZVariable) : v.clearHistory.historyCount = 0 := by
  rfl
  done

theorem clearHistory_preserves_graph (v : ZVariable) : v.clearHistory.graph = v.graph := by
  rfl
  done

theorem clearHistory_preserves_name (v : ZVariable) : v.clearHistory.name = v.name := by
  rfl
  done

theorem clearExecutionHistory_empties_history (r : ZRuntime) : (r.clearExecutionHistory).execution_history.length = 0 := by
  rfl
  done

theorem clearExecutionHistory_preserves_variables (r : ZRuntime) : (r.clearExecutionHistory).variables = r.variables := by
  rfl
  done

theorem addNode_increases_count (g : Graph) (n : Node) : (g.addNode n).nodeCount = g.nodeCount + 1 := by
  dsimp only[Graph.addNode]
  dsimp only [Graph.nodeCount]
  exact List.length_append g.nodes [n]
  done

theorem addEdge_increases_count (g : Graph) (e : Edge) : (g.addEdge e).edgeCount = g.edgeCount + 1 := by
  dsimp only [Graph.addEdge]
  dsimp only [Graph.edgeCount]
  exact List.length_append g.edges [e]
  done

theorem addNode_preserves_edges (g : Graph) (n : Node) : (g.addNode n).edges = g.edges := by
  rfl
  done

theorem addEdge_preserves_nodes (g : Graph) (e : Edge) : (g.addEdge e).nodes = g.nodes := by
  rfl
  done

theorem complex_magnitude_nonneg (c : MyComplex) : c.magnitude ≥ 0 := by
  exact Float.sqrt_nonneg _
  done

theorem fromString_and_roundtrip : RelationalOperationType.fromString "and" = some .op_and := by
  rfl
  done

theorem fromString_or_roundtrip : RelationalOperationType.fromString "or" = some .op_or := by
  rfl
  done

theorem fromString_xor_roundtrip : RelationalOperationType.fromString "xor" = some .op_xor := by
  rfl
  done

theorem fromString_entangle_roundtrip : RelationalOperationType.fromString "entangle" = some .op_entangle := by
  rfl
  done

theorem fromString_invalid_none : RelationalOperationType.fromString "invalid" = none := by
  rfl
  done

theorem getVariable_none_if_not_exists (r : ZRuntime) (name : String) (h : r.hasVariable name = false) : r.getVariable name = none := by
  dsimp only [ZRuntime.getVariable]
  dsimp only[ZRuntime.hasVariable] at h
  have h_ind : ∀ (l : List (String × ZVariable)), List.any l (fun p => p.1 == name) = false → Option.map Prod.snd (List.find? (fun p => p.1 == name) l) = none := by
    intro l
    induction l
    case nil =>
      intro h_any
      rfl
    case cons hd tl ih =>
      intro h_any
      dsimp only [List.any] at h_any
      dsimp only [List.find?]
      cases h_eq : (hd.1 == name)
      case true =>
        cases h_any
      case false =>
        exact ih h_any
  exact h_ind r.variables h
  done

theorem getVariable_some_if_exists (r : ZRuntime) (name : String) (h : r.hasVariable name = true) : (r.getVariable name).isSome = true := by
  dsimp only [ZRuntime.getVariable]
  dsimp only [ZRuntime.hasVariable] at h
  have h_ind : ∀ (l : List (String × ZVariable)), List.any l (fun p => p.1 == name) = true → (Option.map Prod.snd (List.find? (fun p => p.1 == name) l)).isSome = true := by
    intro l
    induction l
    case nil =>
      intro h_any
      cases h_any
    case cons hd tl ih =>
      intro h_any
      dsimp only[List.any] at h_any
      dsimp only [List.find?]
      cases h_eq : (hd.1 == name)
      case true =>
        rfl
      case false =>
        exact ih h_any
  exact h_ind r.variables h
  done

theorem assign_preserves_name (v : ZVariable) (val : String) (t : Int) : (v.assign val t).name = v.name := by
  rfl
  done

theorem empty_fractal_dimension (name : String) : (ZVariable.empty name).getFractalDimension = 0 := by
  rfl
  done

def ZRuntime.invariant (r : ZRuntime) : Prop :=
  ∀ (n1 n2 : String × ZVariable), n1 ∈ r.variables → n2 ∈ r.variables → n1.1 = n2.1 → n1 = n2

theorem empty_runtime_invariant : ZRuntime.empty.invariant := by
  dsimp only [ZRuntime.invariant]
  dsimp only[ZRuntime.empty]
  intro n1 n2 hn1 hn2 heq
  cases hn1
  done

theorem mem_append {α : Type} (a : α) (l1 l2 : List α) (h : a ∈ l1 ++ l2) : a ∈ l1 ∨ a ∈ l2 := by
  induction l1
  case nil =>
    apply Or.inr
    exact h
  case cons hd tl ih =>
    cases h
    case head =>
      apply Or.inl
      apply List.Mem.head
    case tail h_tail =>
      have h_or := ih h_tail
      cases h_or
      case inl h_in_tl =>
        apply Or.inl
        apply List.Mem.tail
        exact h_in_tl
      case inr h_in_l2 =>
        apply Or.inr
        exact h_in_l2
  done

theorem mem_singleton {α : Type} (a b : α) (h : a ∈ [b]) : a = b := by
  cases h
  case head =>
    rfl
  case tail h_tail =>
    cases h_tail
  done

theorem mem_implies_hasVariable (l : List (String × ZVariable)) (a : String × ZVariable) (h : a ∈ l) : List.any l (fun p => p.1 == a.1) = true := by
  induction l
  case nil =>
    cases h
  case cons hd tl ih =>
    dsimp only [List.any]
    cases h
    case head =>
      have h_refl : (a.1 == a.1) = true := LawfulBEq.rfl
      have h_subst : ((a.1 == a.1) || List.any tl (fun p => p.1 == a.1)) = (true || List.any tl (fun p => p.1 == a.1)) :=
        Eq.ndrec (motive := fun x => (x || List.any tl (fun p => p.1 == a.1)) = (true || List.any tl (fun p => p.1 == a.1))) rfl h_refl.symm
      exact Eq.trans h_subst rfl
    case tail h_tail =>
      have h_ih := ih h_tail
      cases h_eq : (hd.1 == a.1)
      case true =>
        rfl
      case false =>
        exact h_ih
  done

theorem createVariable_preserves_invariant (r : ZRuntime) (name : String) (val : Option String) (t : Int) (hinv : r.invariant) (hnovar : r.hasVariable name = false) (r' : ZRuntime) (hcreate : r.createVariable name val t = some r') : r'.invariant := by
  dsimp only [ZRuntime.createVariable] at hcreate
  cases h_cond : r.hasVariable name
  case true =>
    have h_false : true = false := Eq.trans (Eq.symm h_cond) hnovar
    cases h_false
  case false =>
    injection hcreate with h_eq
    cases h_eq
    dsimp only [ZRuntime.invariant]
    intro n1 n2 hn1 hn2 heq
    have hn1_or := mem_append n1 r.variables[(name, match val with | some val_1 => ZVariable.assign (ZVariable.empty name) val_1 t | none => ZVariable.empty name)] hn1
    have hn2_or := mem_append n2 r.variables[(name, match val with | some val_1 => ZVariable.assign (ZVariable.empty name) val_1 t | none => ZVariable.empty name)] hn2
    cases hn1_or
    case inl hn1_in_r =>
      cases hn2_or
      case inl hn2_in_r =>
        exact hinv n1 n2 hn1_in_r hn2_in_r heq
      case inr hn2_in_new =>
        have hn2_eq := mem_singleton n2 (name, match val with | some val_1 => ZVariable.assign (ZVariable.empty name) val_1 t | none => ZVariable.empty name) hn2_in_new
        have hn2_name : n2.1 = name := by
          have h_subst : n2.1 = (name, match val with | some val_1 => ZVariable.assign (ZVariable.empty name) val_1 t | none => ZVariable.empty name).1 :=
            Eq.ndrec (motive := fun x => x.1 = (name, match val with | some val_1 => ZVariable.assign (ZVariable.empty name) val_1 t | none => ZVariable.empty name).1) rfl hn2_eq.symm
          exact h_subst
        have hn1_name : n1.1 = name := Eq.trans heq hn2_name
        have h_any := mem_implies_hasVariable r.variables n1 hn1_in_r
        have h_any_name : List.any r.variables (fun p => p.1 == name) = true :=
          Eq.ndrec (motive := fun x => List.any r.variables (fun p => p.1 == x) = true) h_any hn1_name
        have h_false : true = false := Eq.trans (Eq.symm h_any_name) hnovar
        cases h_false
    case inr hn1_in_new =>
      cases hn2_or
      case inl hn2_in_r =>
        have hn1_eq := mem_singleton n1 (name, match val with | some val_1 => ZVariable.assign (ZVariable.empty name) val_1 t | none => ZVariable.empty name) hn1_in_new
        have hn1_name : n1.1 = name := by
          have h_subst : n1.1 = (name, match val with | some val_1 => ZVariable.assign (ZVariable.empty name) val_1 t | none => ZVariable.empty name).1 :=
            Eq.ndrec (motive := fun x => x.1 = (name, match val with | some val_1 => ZVariable.assign (ZVariable.empty name) val_1 t | none => ZVariable.empty name).1) rfl hn1_eq.symm
          exact h_subst
        have hn2_name : n2.1 = name := Eq.trans (Eq.symm heq) hn1_name
        have h_any := mem_implies_hasVariable r.variables n2 hn2_in_r
        have h_any_name : List.any r.variables (fun p => p.1 == name) = true :=
          Eq.ndrec (motive := fun x => List.any r.variables (fun p => p.1 == x) = true) h_any hn2_name
        have h_false : true = false := Eq.trans (Eq.symm h_any_name) hnovar
        cases h_false
      case inr hn2_in_new =>
        have hn1_eq := mem_singleton n1 (name, match val with | some val_1 => ZVariable.assign (ZVariable.empty name) val_1 t | none => ZVariable.empty name) hn1_in_new
        have hn2_eq := mem_singleton n2 (name, match val with | some val_1 => ZVariable.assign (ZVariable.empty name) val_1 t | none => ZVariable.empty name) hn2_in_new
        exact Eq.trans hn1_eq (Eq.symm hn2_eq)
  done

def QuantumState.isValid (s : QuantumState) : Prop :=
  s.alpha_re^2 + s.alpha_im^2 + s.beta_re^2 + s.beta_im^2 > 0

def Graph.isConsistent (g : Graph) : Prop :=
  ∀ e ∈ g.edges, (∃ n ∈ g.nodes, n.id = e.src) ∧ (∃ n ∈ g.nodes, n.id = e.dst)

theorem empty_graph_consistent : ({ nodes := [], edges :=[] } : Graph).isConsistent := by
  dsimp only [Graph.isConsistent]
  intro e he
  cases he
  done

theorem mem_append_left_manual {α : Type} (a : α) (l1 l2 : List α) (h : a ∈ l1) : a ∈ l1 ++ l2 := by
  induction l1
  case nil =>
    cases h
  case cons hd tl ih =>
    cases h
    case head =>
      apply List.Mem.head
    case tail h_tail =>
      apply List.Mem.tail
      exact ih h_tail
  done

theorem addNode_preserves_consistency (g : Graph) (n : Node) (h : g.isConsistent) : (g.addNode n).isConsistent := by
  dsimp only [Graph.isConsistent]
  dsimp only[Graph.addNode]
  intro e he
  have h_cons := h e he
  cases h_cons
  case intro h_src h_dst =>
    apply And.intro
    case left =>
      cases h_src
      case intro n_src h_n_src =>
        cases h_n_src
        case intro h_in_nodes h_id_eq =>
          apply Exists.intro n_src
          apply And.intro
          case left =>
            have h_append := mem_append_left_manual n_src g.nodes [n] h_in_nodes
            exact h_append
          case right =>
            exact h_id_eq
    case right =>
      cases h_dst
      case intro n_dst h_n_dst =>
        cases h_n_dst
        case intro h_in_nodes h_id_eq =>
          apply Exists.intro n_dst
          apply And.intro
          case left =>
            have h_append := mem_append_left_manual n_dst g.nodes [n] h_in_nodes
            exact h_append
          case right =>
            exact h_id_eq
  done

def MeasurementResult.isValid (m : MeasurementResult) : Prop :=
  (m.result = 0 ∨ m.result = 1) ∧ m.probability ≥ 0 ∧ m.probability ≤ 1

theorem history_entry_type_exhaustive (t : HistoryEntryType) : t = .assign ∨ t = .transform ∨ t = .relate ∨ t = .measure ∨ t = .entangle := by
  cases t
  case assign =>
    apply Or.inl
    rfl
  case transform =>
    apply Or.inr
    apply Or.inl
    rfl
  case relate =>
    apply Or.inr
    apply Or.inr
    apply Or.inl
    rfl
  case measure =>
    apply Or.inr
    apply Or.inr
    apply Or.inr
    apply Or.inl
    rfl
  case entangle =>
    apply Or.inr
    apply Or.inr
    apply Or.inr
    apply Or.inr
    rfl
  done

theorem execution_action_exhaustive (a : ExecutionAction) : a = .create_variable ∨ a = .delete_variable ∨ a = .relational_operation ∨ a = .entangle_variables ∨ a = .propagate_information ∨ a = .fractal_transform ∨ a = .measure ∨ a = .quantum_circuit ∨ a = .relational_expression := by
  cases a
  case create_variable =>
    apply Or.inl
    rfl
  case delete_variable =>
    apply Or.inr
    apply Or.inl
    rfl
  case relational_operation =>
    apply Or.inr
    apply Or.inr
    apply Or.inl
    rfl
  case entangle_variables =>
    apply Or.inr
    apply Or.inr
    apply Or.inr
    apply Or.inl
    rfl
  case propagate_information =>
    apply Or.inr
    apply Or.inr
    apply Or.inr
    apply Or.inr
    apply Or.inl
    rfl
  case fractal_transform =>
    apply Or.inr
    apply Or.inr
    apply Or.inr
    apply Or.inr
    apply Or.inr
    apply Or.inl
    rfl
  case measure =>
    apply Or.inr
    apply Or.inr
    apply Or.inr
    apply Or.inr
    apply Or.inr
    apply Or.inr
    apply Or.inl
    rfl
  case quantum_circuit =>
    apply Or.inr
    apply Or.inr
    apply Or.inr
    apply Or.inr
    apply Or.inr
    apply Or.inr
    apply Or.inr
    apply Or.inl
    rfl
  case relational_expression =>
    apply Or.inr
    apply Or.inr
    apply Or.inr
    apply Or.inr
    apply Or.inr
    apply Or.inr
    apply Or.inr
    apply Or.inr
    rfl
  done

theorem relational_op_exhaustive (op : RelationalOperationType) : op = .op_and ∨ op = .op_or ∨ op = .op_xor ∨ op = .op_entangle := by
  cases op
  case op_and =>
    apply Or.inl
    rfl
  case op_or =>
    apply Or.inr
    apply Or.inl
    rfl
  case op_xor =>
    apply Or.inr
    apply Or.inr
    apply Or.inl
    rfl
  case op_entangle =>
    apply Or.inr
    apply Or.inr
    apply Or.inr
    rfl
  done

def ZRuntime.entanglementSymmetric (r : ZRuntime) : Prop :=
  ∀ v1 v2 : ZVariable, ∀ e ∈ v1.graph.edges, e.quality = .entangled → e.dst = v2.name → ∃ e' ∈ v2.graph.edges, e'.quality = .entangled ∧ e'.dst = v1.name

def operationCommutative (op : RelationalOperationType) : Prop :=
  op = .op_and ∨ op = .op_or ∨ op = .op_xor

theorem and_is_commutative : operationCommutative .op_and := by
  dsimp only [operationCommutative]
  apply Or.inl
  rfl
  done

theorem or_is_commutative : operationCommutative .op_or := by
  dsimp only[operationCommutative]
  apply Or.inr
  apply Or.inl
  rfl
  done

theorem xor_is_commutative : operationCommutative .op_xor := by
  dsimp only[operationCommutative]
  apply Or.inr
  apply Or.inr
  rfl
  done

theorem entangle_not_commutative_labeled : ¬operationCommutative .op_entangle := by
  dsimp only [operationCommutative]
  intro h
  cases h
  case inl h1 =>
    cases h1
  case inr h2 =>
    cases h2
    case inl h3 =>
      cases h3
    case inr h4 =>
      cases h4
  done

def ZRuntime.historyMonotonic (r1 r2 : ZRuntime) : Prop :=
  r1.execution_history.length ≤ r2.execution_history.length

theorem createVariable_history_monotonic (r : ZRuntime) (name : String) (val : Option String) (t : Int) (hnovar : r.hasVariable name = false) (r' : ZRuntime) (hcreate : r.createVariable name val t = some r') : r.historyMonotonic r' := by
  dsimp only[ZRuntime.createVariable] at hcreate
  cases h_cond : r.hasVariable name
  case true =>
    have h_false : true = false := Eq.trans (Eq.symm h_cond) hnovar
    cases h_false
  case false =>
    injection hcreate with h_eq
    cases h_eq
    dsimp only [ZRuntime.historyMonotonic]
    have h_len : (r.execution_history ++[{ action := ExecutionAction.create_variable, primary_target := name, secondary_targets := [], operation_type := none, result_value := val, result_int := none, result_float := none, timestamp := t }]).length = r.execution_history.length + 1 :=
      List.length_append r.execution_history[{ action := ExecutionAction.create_variable, primary_target := name, secondary_targets := [], operation_type := none, result_value := val, result_int := none, result_float := none, timestamp := t }]
    have h_le : r.execution_history.length ≤ r.execution_history.length + 1 := Nat.le_add_right r.execution_history.length 1
    have h_goal : r.execution_history.length ≤ (r.execution_history ++[{ action := ExecutionAction.create_variable, primary_target := name, secondary_targets := [], operation_type := none, result_value := val, result_int := none, result_float := none, timestamp := t }]).length :=
      Eq.ndrec (motive := fun x => r.execution_history.length ≤ x) h_le h_len.symm
    exact h_goal
  done

theorem nodeCount_nonneg (g : Graph) : g.nodeCount ≥ 0 := by
  exact Nat.zero_le _
  done

theorem edgeCount_nonneg (g : Graph) : g.edgeCount ≥ 0 := by
  exact Nat.zero_le _
  done

theorem variableCount_nonneg (r : ZRuntime) : r.variableCount ≥ 0 := by
  exact Nat.zero_le _
  done

theorem historyCount_nonneg (v : ZVariable) : v.historyCount ≥ 0 := by
  exact Nat.zero_le _
  done

theorem stateCount_nonneg (v : ZVariable) : v.stateCount ≥ 0 := by
  exact Nat.zero_le _
  done

def SystemInvariant (r : ZRuntime) : Prop :=
  r.invariant ∧ r.global_graph.isConsistent ∧ (∀ p ∈ r.variables, p.2.graph.isConsistent)

theorem empty_system_invariant : SystemInvariant ZRuntime.empty := by
  dsimp only [SystemInvariant]
  apply And.intro
  case left =>
    exact empty_runtime_invariant
  case right =>
    apply And.intro
    case left =>
      exact empty_graph_consistent
    case right =>
      intro p hp
      dsimp only [ZRuntime.empty] at hp
      cases hp
  done