use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::cmp::Reverse;
use core::f64;
use core::num::NonZeroU64;

use hashbrown::hash_map::Entry;
use hashbrown::{HashMap, HashSet};

use crate::constraint::Constraint;
use crate::row::{near_zero, Row, Symbol, SymbolKind};
use crate::semantic::{ConstraintFingerprint, SemanticConstraints};
use crate::strength::Strength;
use crate::{
    AddConstraintError, AddEditVariableError, Expression, RelationalOperator,
    RemoveConstraintError, RemoveEditVariableError, SuggestValueError, Term, Variable,
};

#[derive(Debug, Copy, Clone, thiserror::Error)]
#[error("The solver entered an invalid state. If this occurs please report the issue.")]
pub enum InternalSolverError {
    #[error("The objective is unbounded.")]
    ObjectiveUnbounded,
    #[error("Dual optimize failed.")]
    DualOptimizeFailed,
    #[error("Failed to find leaving row.")]
    FailedToFindLeavingRow,
    #[error("Edit constraint not in system")]
    EditConstraintNotInSystem,
}

#[derive(Copy, Clone)]
struct Tag {
    marker: Symbol,
    other: Symbol,
}

/// 约束身份(ConstraintId).
///
/// 设计目标:
/// - 对外暴露一等的 handle,让 remove/update 不再依赖 `Constraint` 的 Arc 指针身份.
/// - identity 与内存地址解耦,上层可以自由重建同语义的约束对象.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConstraintId(NonZeroU64);

impl ConstraintId {
    /// 获取底层数值(用于日志/诊断; 不建议业务逻辑依赖该数值).
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

impl core::fmt::Display for ConstraintId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.get())
    }
}

#[derive(Clone)]
struct ConstraintEntry {
    constraint: Constraint,
    tag: Tag,
}

/// batch apply 的增量操作.
#[derive(Debug, Clone)]
pub enum ConstraintBatchOp {
    /// 新增约束(由 solver 分配 `ConstraintId`).
    Add(Constraint),
    /// 移除约束(以 `ConstraintId` 为准).
    Remove(ConstraintId),
    /// 更新约束(以 `ConstraintId` 为准).
    ///
    /// 说明:
    /// - Cassowary 不支持“原地 edit 一般约束”,因此这里的 update 语义等价于 remove+add,
    ///   但会尽量保持对外 identity(ConstraintId) 不变.
    Update {
        id: ConstraintId,
        constraint: Constraint,
    },
}

/// batch apply 的返回值.
#[derive(Debug, Default, Clone)]
pub struct ConstraintBatchResult {
    /// 成功 Add 的结果:
    /// - `usize`: 原始 op 的 index(调用方输入序列的下标)。
    /// - `ConstraintId`: solver 分配的 id。
    pub added: Vec<(usize, ConstraintId)>,
    /// batch 内部 Add 失败但被忽略的项(只包含非致命错误,例如 Unsatisfiable/Duplicate).
    ///
    /// - `usize`: 原始 op 的 index(调用方输入序列的下标).
    /// - `AddConstraintError`: 失败原因.
    pub skipped_adds: Vec<(usize, AddConstraintError)>,
}

/// batch apply 的错误.
///
/// 说明:
/// - 该 API 不承诺原子性: 一旦中途失败,可能已经有部分 op 生效。
/// - 调用方若需要“强原子”,推荐在失败后执行 hard reset(重建 solver 并重放最新全集)。
#[derive(Debug, thiserror::Error)]
pub enum ConstraintBatchApplyError {
    #[error("batch remove failed at index={index}: {source}")]
    Remove {
        index: usize,
        #[source]
        source: RemoveConstraintError,
    },

    #[error("batch add failed at index={index}: {source}")]
    Add {
        index: usize,
        #[source]
        source: AddConstraintError,
    },

    #[error("batch update(remove) failed at index={index}: {source}")]
    UpdateRemove {
        index: usize,
        #[source]
        source: RemoveConstraintError,
    },

    #[error("batch update(add) failed at index={index}: {source}")]
    UpdateAdd {
        index: usize,
        #[source]
        source: AddConstraintError,
    },
}

/// 语义 reconcile 的执行结果(增量 diff + batch apply).
///
/// 说明:
/// - 该结构只承诺“map 状态已被更新为 reconcile 后的状态”。
/// - 若发生 `skipped_adds`,表示新增约束被 solver 忽略(例如 Duplicate/Unsatisfiable),此时 map
///   不会插入对应项。
#[derive(Debug, Default, Clone)]
pub struct SemanticReconcileResult {
    /// 本次 reconcile 是否导致 solver/map 发生了可观测变更(成功 remove 或成功 add)。
    pub did_update: bool,
    /// 本轮被移除的语义指纹列表(顺序稳定,按 fingerprint 升序)。
    ///
    /// 说明:
    /// - 该列表反映“语义集合 diff”的 remove 部分,用于日志/回归定位。
    /// - 即便对应 `ConstraintId` 已不在 solver(脏数据),这里仍会记录并从 map 中移除。
    pub removed_fingerprints: Vec<ConstraintFingerprint>,
    /// 本轮新增成功的语义指纹(以及 solver 分配的 `ConstraintId`)。
    ///
    /// 说明:
    /// - 仅包含成功 add 的项。
    /// - 若 add 失败且被忽略,会出现在 `skipped_adds`。
    pub added_fingerprints: Vec<(ConstraintFingerprint, ConstraintId)>,
    /// 本轮修复成功的语义指纹(以及新的 `ConstraintId`)。
    ///
    /// repair 的定义:
    /// - fingerprint 在 `last_observation` 中存在,但对应 id 已不在 solver。
    /// - reconcile 会通过一次 add 修复该脏数据,并用新的 id 覆盖 map。
    pub repaired_fingerprints: Vec<(ConstraintFingerprint, ConstraintId)>,
    /// batch add 里被忽略的项(例如 Duplicate/Unsatisfiable)。
    pub skipped_adds: Vec<(ConstraintFingerprint, AddConstraintError)>,
}

#[derive(Clone)]
struct EditInfo {
    tag: Tag,
    constraint_id: ConstraintId,
    constant: f64,
}

/// A constraint solver using the Cassowary algorithm. For proper usage please see the top level
/// crate documentation.
pub struct Solver {
    constraints: HashMap<ConstraintId, ConstraintEntry>,
    constraint_ids: HashMap<Constraint, ConstraintId>,
    var_data: HashMap<Variable, (f64, Symbol, usize)>,
    var_for_symbol: HashMap<Symbol, Variable>,
    public_changes: Vec<(Variable, f64)>,
    changed: HashSet<Variable>,
    should_clear_changes: bool,
    rows: HashMap<Symbol, Box<Row>>,
    edits: HashMap<Variable, EditInfo>,
    ////////////////////////////////////////////////////////////////////////////////
    // determinism:
    //
    // - 原实现用 `Vec<Symbol>` + `pop()` 当栈来处理 infeasible rows.
    // - 但 infeasible rows 的收集顺序来自 `HashMap` 迭代,天然不稳定,会把 dual-optimize 的 pivot
    //   路径变成 “看运气”.
    // - 这里改为“稳定 worklist”:用小根堆(按 Symbol 的 Ord)取下一项,从而让处理顺序与迭代顺序解耦.
    ////////////////////////////////////////////////////////////////////////////////
    infeasible_rows: BinaryHeap<Reverse<Symbol>>, // never contains external symbols
    objective: Rc<RefCell<Row>>,
    artificial: Option<Rc<RefCell<Row>>>,
    id_tick: usize,
    next_constraint_id: u64,
}

impl Default for Solver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver {
    /// Construct a new solver.
    pub fn new() -> Solver {
        Solver {
            constraints: HashMap::new(),
            constraint_ids: HashMap::new(),
            var_data: HashMap::new(),
            var_for_symbol: HashMap::new(),
            public_changes: Vec::new(),
            changed: HashSet::new(),
            should_clear_changes: false,
            rows: HashMap::new(),
            edits: HashMap::new(),
            infeasible_rows: BinaryHeap::new(),
            objective: Rc::new(RefCell::new(Row::new(0.0))),
            artificial: None,
            id_tick: 1,
            next_constraint_id: 1,
        }
    }

    #[must_use]
    fn alloc_constraint_id(&mut self) -> ConstraintId {
        ////////////////////////////////////////////////////////////////////////////////
        // 说明:
        // - `ConstraintId` 仅用于 identity,不承诺跨进程/跨平台稳定.
        // - 这里用单调递增计数器分配,并使用 NonZeroU64 让 Option<ConstraintId> 更紧凑.
        ////////////////////////////////////////////////////////////////////////////////
        let raw = self.next_constraint_id;
        self.next_constraint_id = self
            .next_constraint_id
            .checked_add(1)
            .expect("ConstraintId overflow");
        ConstraintId(NonZeroU64::new(raw).expect("ConstraintId must be non-zero"))
    }

    pub fn add_constraints<I: IntoIterator<Item = Constraint>>(
        &mut self,
        constraints: I,
    ) -> Result<(), AddConstraintError> {
        for constraint in constraints {
            let _ = self.add_constraint(constraint)?;
        }
        Ok(())
    }

    /// 批量应用一组约束增量操作.
    ///
    /// # 设计目标(最小可用版本)
    ///
    /// - 让调用方一次性提交一批 op,避免上层“靠手工排序调用顺序”来换稳定性。
    /// - solver 内部会对 op 做稳定排序:
    ///   - Remove/Update: 按 `ConstraintId` 升序。
    ///   - Add: 按 `ConstraintFingerprint` 升序(语义稳定)。
    ///
    /// # 错误语义
    ///
    /// - Add 的 `DuplicateConstraint/UnsatisfiableConstraint` 会被记录到
    ///   `skipped_adds`,并继续处理后续 op。
    /// - 其它错误会提前返回 `ConstraintBatchApplyError`。
    /// - 该 API 不承诺原子性: 出错时可能已有部分变更生效;推荐调用方执行 hard reset 重建。
    pub fn apply_constraint_batch(
        &mut self,
        ops: &[ConstraintBatchOp],
    ) -> Result<ConstraintBatchResult, ConstraintBatchApplyError> {
        let mut removes: Vec<(usize, ConstraintId)> = Vec::new();
        let mut updates: Vec<(usize, ConstraintId, Constraint)> = Vec::new();
        let mut adds: Vec<(usize, ConstraintFingerprint, Constraint)> = Vec::new();

        for (index, op) in ops.iter().enumerate() {
            match op {
                ConstraintBatchOp::Add(constraint) => {
                    let fp = ConstraintFingerprint::new(constraint);
                    adds.push((index, fp, constraint.clone()));
                }
                ConstraintBatchOp::Remove(id) => {
                    removes.push((index, *id));
                }
                ConstraintBatchOp::Update { id, constraint } => {
                    updates.push((index, *id, constraint.clone()));
                }
            }
        }

        // 稳定排序: 不依赖输入顺序,也不依赖 HashMap/HashSet 的迭代顺序。
        removes.sort_unstable_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        updates.sort_unstable_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        adds.sort_unstable_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        let mut out = ConstraintBatchResult::default();

        for (index, id) in removes {
            self.remove_constraint(id)
                .map_err(|source| ConstraintBatchApplyError::Remove { index, source })?;
        }

        for (index, id, constraint) in updates {
            self.remove_constraint(id)
                .map_err(|source| ConstraintBatchApplyError::UpdateRemove { index, source })?;
            self.add_constraint_with_id(id, constraint, true)
                .map_err(|source| ConstraintBatchApplyError::UpdateAdd { index, source })?;
        }

        for (index, _fp, constraint) in adds {
            match self.add_constraint(constraint) {
                Ok(id) => {
                    out.added.push((index, id));
                }
                Err(
                    e @ (AddConstraintError::DuplicateConstraint
                    | AddConstraintError::UnsatisfiableConstraint),
                ) => {
                    out.skipped_adds.push((index, e));
                }
                Err(e) => {
                    return Err(ConstraintBatchApplyError::Add { index, source: e });
                }
            }
        }

        Ok(out)
    }

    /// 以 `ConstraintFingerprint` 为 key,对约束集合做 reconcile,并最大化复用旧 `ConstraintId`.
    ///
    /// 输入:
    /// - `last_observation`: 上一轮已成功应用到 solver 的 `(fingerprint -> ConstraintId)`。
    /// - `newest`: 本轮输入全集的语义集合 `(fingerprint -> Constraint)`。
    ///
    /// 行为:
    /// - 对“上一轮存在但本轮不存在”的 fingerprint,执行 remove(若 id 仍在 solver 内)。
    /// - 对“本轮存在但上一轮不存在”的 fingerprint,执行 add(由 solver 分配新 id)。
    /// - 对“fingerprint 相同但对象重建”的情况,不发出任何 op,从而复用旧 id 与 solver 内部状态。
    /// - 若发现 `last_observation` 中的 id 已不在 solver(脏数据),会尝试通过 add 修复。
    ///
    /// 错误语义:
    /// - 该 API 不承诺原子性: 一旦中途失败,可能已部分生效(与 `apply_constraint_batch` 一致)。
    /// - 建议调用方在遇到 `ConstraintBatchApplyError` 后执行 hard reset(重建 solver
    ///   并重放最新全集)。
    pub fn reconcile_semantic_constraints(
        &mut self,
        last_observation: &mut BTreeMap<ConstraintFingerprint, ConstraintId>,
        newest: &SemanticConstraints,
    ) -> Result<SemanticReconcileResult, ConstraintBatchApplyError> {
        ////////////////////////////////////////////////////////////////////////////////
        // 语义集合 diff:
        // - remove: last 有但 newest 无
        // - add: newest 有但 last 无
        // - repair: newest 有且 last 有,但 id 已不在 solver
        //
        // 统一通过 batch apply 来执行,避免调用方手写时序与错误分类。
        ////////////////////////////////////////////////////////////////////////////////

        // remove 列表(用于 map commit + 诊断报告)
        let mut removed_fingerprints: Vec<ConstraintFingerprint> = Vec::new();
        let mut ops: Vec<ConstraintBatchOp> = Vec::new();

        // add op 的 index -> fingerprint 映射(用于把 batch 返回的 `ConstraintId` 写回 map)
        let mut add_index_to_fingerprint: BTreeMap<usize, ConstraintFingerprint> = BTreeMap::new();
        // add op 的 index -> 是否为 repair(用于报告 added vs repaired)
        let mut add_index_is_repair: BTreeMap<usize, bool> = BTreeMap::new();

        // 1) remove: last 有但 newest 无
        for (fp, constraint_id) in last_observation.iter() {
            if newest.contains_key(fp) {
                continue;
            }
            removed_fingerprints.push(fp.clone());
            if self.has_constraint(*constraint_id) {
                ops.push(ConstraintBatchOp::Remove(*constraint_id));
            }
        }

        // 2) add/repair: newest 有但 last 无,或 last 的 id 已不在 solver
        for (fp, constraint) in newest.iter() {
            match last_observation.get(fp) {
                None => {
                    let op_index = ops.len();
                    ops.push(ConstraintBatchOp::Add(constraint.clone()));
                    add_index_to_fingerprint.insert(op_index, fp.clone());
                    add_index_is_repair.insert(op_index, false);
                }
                Some(id) if !self.has_constraint(*id) => {
                    // map 脏数据: fingerprint 存在但 id 不在 solver.
                    let op_index = ops.len();
                    ops.push(ConstraintBatchOp::Add(constraint.clone()));
                    add_index_to_fingerprint.insert(op_index, fp.clone());
                    add_index_is_repair.insert(op_index, true);
                }
                Some(_) => {}
            }
        }

        // 3) 没有任何 solver 侧操作,但可能仍需要更新 map(例如 remove 的 id 已不存在)
        if ops.is_empty() {
            if removed_fingerprints.is_empty() {
                return Ok(SemanticReconcileResult::default());
            }

            for fp in removed_fingerprints.iter() {
                let _ = last_observation.remove(fp);
            }

            return Ok(SemanticReconcileResult {
                did_update: true,
                removed_fingerprints,
                added_fingerprints: Vec::new(),
                repaired_fingerprints: Vec::new(),
                skipped_adds: Vec::new(),
            });
        }

        let batch = self.apply_constraint_batch(ops.as_slice())?;

        // 4) commit:仅在 batch 成功后更新 map
        let mut did_update = false;

        for fp in removed_fingerprints.iter() {
            if last_observation.remove(fp).is_some() {
                did_update = true;
            }
        }

        let mut added_fingerprints: Vec<(ConstraintFingerprint, ConstraintId)> = Vec::new();
        let mut repaired_fingerprints: Vec<(ConstraintFingerprint, ConstraintId)> = Vec::new();

        for (op_index, id) in batch.added {
            let Some(fp) = add_index_to_fingerprint.get(&op_index) else {
                // 防御式:理论上不应发生,但这里不 panic,避免把库函数升级成崩溃点。
                continue;
            };

            // 记录“变化原因”(added vs repaired),便于下游日志/回归定位。
            let is_repair = add_index_is_repair.get(&op_index).copied().unwrap_or(false);
            if is_repair {
                repaired_fingerprints.push((fp.clone(), id));
            } else {
                added_fingerprints.push((fp.clone(), id));
            }

            let previous = last_observation.insert(fp.clone(), id);
            if previous.is_none() {
                did_update = true;
            } else {
                // repair 覆盖旧 id 也算更新
                did_update = true;
            }
        }

        let mut skipped_adds: Vec<(ConstraintFingerprint, AddConstraintError)> = Vec::new();
        for (op_index, err) in batch.skipped_adds {
            let Some(fp) = add_index_to_fingerprint.get(&op_index) else {
                continue;
            };
            skipped_adds.push((fp.clone(), err));
        }

        Ok(SemanticReconcileResult {
            did_update,
            removed_fingerprints,
            added_fingerprints,
            repaired_fingerprints,
            skipped_adds,
        })
    }

    /// Add a constraint to the solver.
    pub fn add_constraint(
        &mut self,
        constraint: Constraint,
    ) -> Result<ConstraintId, AddConstraintError> {
        let id = self.alloc_constraint_id();
        self.add_constraint_with_id(id, constraint, true)?;
        Ok(id)
    }

    fn add_constraint_with_id(
        &mut self,
        id: ConstraintId,
        constraint: Constraint,
        should_optimize: bool,
    ) -> Result<(), AddConstraintError> {
        ////////////////////////////////////////////////////////////////////////////////
        // NOTE:
        // - 为保持旧行为:同一个 Constraint(同 Arc 指针身份)重复添加会被拒绝.
        // - 语义重复(同 fingerprint 但不同对象)应由上层 reconcile/去重策略处理.
        ////////////////////////////////////////////////////////////////////////////////
        if self.constraint_ids.contains_key(&constraint) || self.constraints.contains_key(&id) {
            return Err(AddConstraintError::DuplicateConstraint);
        }

        // Creating a row causes symbols to reserved for the variables in the constraint. If this
        // method exits with an exception, then its possible those variables will linger in the var
        // map. Since its likely that those variables will be used in other constraints and since
        // exceptional conditions are uncommon, i'm not too worried about aggressive cleanup of the
        // var map.
        let (mut row, tag) = self.create_row(&constraint);
        let mut subject = Solver::choose_subject(&row, &tag);

        // If choose_subject could find a valid entering symbol, one last option is available if the
        // entire row is composed of dummy variables. If the constant of the row is zero, then this
        // represents redundant constraints and the new dummy marker can enter the basis. If the
        // constant is non-zero, then it represents an unsatisfiable constraint.
        if subject.kind() == SymbolKind::Invalid && Solver::all_dummies(&row) {
            if !near_zero(row.constant) {
                return Err(AddConstraintError::UnsatisfiableConstraint);
            } else {
                subject = tag.marker;
            }
        }

        // If an entering symbol still isn't found, then the row must be added using an artificial
        // variable. If that fails, then the row represents an unsatisfiable constraint.
        if subject.kind() == SymbolKind::Invalid {
            let satisfiable = self.add_with_artificial_variable(&row)?;
            if !satisfiable {
                return Err(AddConstraintError::UnsatisfiableConstraint);
            }
        } else {
            row.solve_for_symbol(subject);
            self.substitute(subject, &row);
            if subject.kind() == SymbolKind::External && row.constant != 0.0 {
                let v = self.var_for_symbol[&subject];
                self.var_changed(v);
            }
            self.rows.insert(subject, row);
        }

        // 插入约束存储(此时 tableau 已经被修改,因此即使后续 optimize 报错,也认为约束已进入系统).
        self.constraint_ids.insert(constraint.clone(), id);
        self.constraints
            .insert(id, ConstraintEntry { constraint, tag });

        // Optimizing after each constraint is added performs less aggregate work due to a smaller
        // average system size. It also ensures the solver remains in a consistent state.
        if should_optimize {
            let objective = self.objective.clone();
            self.optimize(&objective)?;
        }
        Ok(())
    }

    /// Remove a constraint from the solver.
    pub fn remove_constraint(
        &mut self,
        constraint_id: ConstraintId,
    ) -> Result<(), RemoveConstraintError> {
        let entry = self
            .constraints
            .remove(&constraint_id)
            .ok_or(RemoveConstraintError::UnknownConstraint)?;
        let _ = self.constraint_ids.remove(&entry.constraint);
        self.remove_constraint_with_tag(&entry.constraint, &entry.tag, true)
    }

    fn remove_constraint_with_tag(
        &mut self,
        constraint: &Constraint,
        tag: &Tag,
        should_optimize: bool,
    ) -> Result<(), RemoveConstraintError> {
        // Remove the error effects from the objective function
        // *before* pivoting, or substitutions into the objective
        // will lead to incorrect solver results.
        self.remove_constraint_effects(constraint, tag);

        // If the marker is basic, simply drop the row. Otherwise,
        // pivot the marker into the basis and then drop the row.
        if self.rows.remove(&tag.marker).is_none() {
            let (leaving, mut row) = self.get_marker_leaving_row(tag.marker).ok_or(
                RemoveConstraintError::InternalSolverError(
                    InternalSolverError::FailedToFindLeavingRow,
                ),
            )?;
            row.solve_for_symbols(leaving, tag.marker);
            self.substitute(tag.marker, &row);
        }

        // Optimizing after each constraint is removed ensures that the
        // solver remains consistent. It makes the solver api easier to
        // use at a small tradeoff for speed.
        if should_optimize {
            let objective = self.objective.clone();
            self.optimize(&objective)?;
        }

        // Check for and decrease the reference count for variables referenced by the constraint
        // If the reference count is zero remove the variable from the variable map
        for term in &constraint.expr().terms {
            if !near_zero(term.coefficient) {
                let mut should_remove = false;
                if let Some(&mut (_, _, ref mut count)) = self.var_data.get_mut(&term.variable) {
                    *count -= 1;
                    should_remove = *count == 0;
                }
                if should_remove {
                    self.var_for_symbol.remove(&self.var_data[&term.variable].1);
                    self.var_data.remove(&term.variable);
                }
            }
        }
        Ok(())
    }

    /// Test whether a constraint has been added to the solver.
    pub fn has_constraint(&self, constraint_id: ConstraintId) -> bool {
        self.constraints.contains_key(&constraint_id)
    }

    /// Add an edit variable to the solver.
    ///
    /// This method should be called before the `suggest_value` method is
    /// used to supply a suggested value for the given edit variable.
    pub fn add_edit_variable(
        &mut self,
        v: Variable,
        strength: Strength,
    ) -> Result<(), AddEditVariableError> {
        if self.edits.contains_key(&v) {
            return Err(AddEditVariableError::DuplicateEditVariable);
        }
        if strength == Strength::REQUIRED {
            return Err(AddEditVariableError::BadRequiredStrength);
        }
        let cn = Constraint::new(
            Expression::from_term(Term::new(v, 1.0)),
            RelationalOperator::Equal,
            strength,
        );
        ////////////////////////////////////////////////////////////////////////////////
        // 防御式：add_constraint 可能返回 InternalSolverError（例如 ObjectiveUnbounded）。
        //
        // 旧实现这里直接 unwrap，会把“内部错误”升级成 panic，导致 GUI 场景直接崩溃。
        // 上层（例如 emg_layout）有 hard reset 兜底逻辑，应该让错误可传播而不是 panic。
        ////////////////////////////////////////////////////////////////////////////////
        let constraint_id = match self.add_constraint(cn.clone()) {
            Ok(id) => id,
            Err(AddConstraintError::DuplicateConstraint) => {
                // 理论上不应发生：我们已经检查过 edits.contains_key(&v)。
                return Err(AddEditVariableError::DuplicateEditVariable);
            }
            Err(AddConstraintError::UnsatisfiableConstraint) => {
                // edit variable 禁止 REQUIRED 强度，按理不会触发。
                return Err(AddEditVariableError::UnsatisfiableConstraint);
            }
            Err(AddConstraintError::InternalSolverError(inner)) => {
                return Err(AddEditVariableError::InternalSolverError(inner));
            }
        };
        let tag = self
            .constraints
            .get(&constraint_id)
            .expect("constraint_id should exist after successful add_constraint")
            .tag;
        self.edits.insert(
            v,
            EditInfo {
                tag,
                constraint_id,
                constant: 0.0,
            },
        );
        Ok(())
    }

    /// Remove an edit variable from the solver.
    pub fn remove_edit_variable(&mut self, v: Variable) -> Result<(), RemoveEditVariableError> {
        if let Some(constraint_id) = self.edits.remove(&v).map(|e| e.constraint_id) {
            self.remove_constraint(constraint_id).map_err(|e| match e {
                RemoveConstraintError::UnknownConstraint => {
                    RemoveEditVariableError::InternalSolverError(
                        InternalSolverError::EditConstraintNotInSystem,
                    )
                }
                RemoveConstraintError::InternalSolverError(s) => {
                    RemoveEditVariableError::InternalSolverError(s)
                }
            })?;
            Ok(())
        } else {
            Err(RemoveEditVariableError::UnknownEditVariable)
        }
    }

    /// Test whether an edit variable has been added to the solver.
    pub fn has_edit_variable(&self, v: &Variable) -> bool {
        self.edits.contains_key(v)
    }

    /// Suggest a value for the given edit variable.
    ///
    /// This method should be used after an edit variable has been added to
    /// the solver in order to suggest the value for that variable.
    pub fn suggest_value(
        &mut self,
        variable: Variable,
        value: f64,
    ) -> Result<(), SuggestValueError> {
        let (info_tag_marker, info_tag_other, delta) = {
            let info = self
                .edits
                .get_mut(&variable)
                .ok_or(SuggestValueError::UnknownEditVariable)?;
            let delta = value - info.constant;
            info.constant = value;
            (info.tag.marker, info.tag.other, delta)
        };
        // tag.marker and tag.other are never external symbols

        // The nice version of the following code runs into non-lexical borrow issues.
        // Ideally the `if row...` code would be in the body of the if. Pretend that it is.
        {
            let infeasible_rows = &mut self.infeasible_rows;
            if self
                .rows
                .get_mut(&info_tag_marker)
                .map(|row| {
                    if row.add(-delta) < 0.0 {
                        infeasible_rows.push(Reverse(info_tag_marker));
                    }
                })
                .is_some()
                || self
                    .rows
                    .get_mut(&info_tag_other)
                    .map(|row| {
                        if row.add(delta) < 0.0 {
                            infeasible_rows.push(Reverse(info_tag_other));
                        }
                    })
                    .is_some()
            {
            } else {
                for (symbol, row) in &mut self.rows {
                    let coeff = row.coefficient_for(info_tag_marker);
                    let diff = delta * coeff;
                    if diff != 0.0 && symbol.kind() == SymbolKind::External {
                        let v = self.var_for_symbol[symbol];
                        // inline var_changed - borrow checker workaround
                        if self.should_clear_changes {
                            self.changed.clear();
                            self.should_clear_changes = false;
                        }
                        self.changed.insert(v);
                    }
                    if coeff != 0.0 && row.add(diff) < 0.0 && symbol.kind() != SymbolKind::External
                    {
                        infeasible_rows.push(Reverse(*symbol));
                    }
                }
            }
        }
        self.dual_optimize()?;
        Ok(())
    }

    fn var_changed(&mut self, v: Variable) {
        if self.should_clear_changes {
            self.changed.clear();
            self.should_clear_changes = false;
        }
        self.changed.insert(v);
    }

    /// Fetches all changes to the values of variables since the last call to this function.
    ///
    /// The list of changes returned is not in a specific order. Each change comprises the variable
    /// changed and the new value of that variable.
    pub fn fetch_changes(&mut self) -> &[(Variable, f64)] {
        if self.should_clear_changes {
            self.changed.clear();
            self.should_clear_changes = false;
        } else {
            self.should_clear_changes = true;
        }
        self.public_changes.clear();
        for &v in &self.changed {
            if let Some(var_data) = self.var_data.get_mut(&v) {
                let new_value = self
                    .rows
                    .get(&var_data.1)
                    .map(|r| r.constant)
                    .unwrap_or(0.0);
                let old_value = var_data.0;
                if old_value != new_value {
                    self.public_changes.push((v, new_value));
                    var_data.0 = new_value;
                }
            }
        }
        &self.public_changes
    }

    /// Reset the solver to the empty starting condition.
    ///
    /// This method resets the internal solver state to the empty starting
    /// condition, as if no constraints or edit variables have been added.
    /// This can be faster than deleting the solver and creating a new one
    /// when the entire system must change, since it can avoid unnecessary
    /// heap (de)allocations.
    pub fn reset(&mut self) {
        self.rows.clear();
        self.constraints.clear();
        self.constraint_ids.clear();
        self.var_data.clear();
        self.var_for_symbol.clear();
        self.changed.clear();
        self.should_clear_changes = false;
        self.edits.clear();
        self.infeasible_rows.clear();
        *self.objective.borrow_mut() = Row::new(0.0);
        self.artificial = None;
        self.id_tick = 1;
        self.next_constraint_id = 1;
    }

    /// Get the symbol for the given variable.
    ///
    /// If a symbol does not exist for the variable, one will be created.
    fn get_var_symbol(&mut self, v: Variable) -> Symbol {
        let id_tick = &mut self.id_tick;
        let var_for_symbol = &mut self.var_for_symbol;
        let value = self.var_data.entry(v).or_insert_with(|| {
            let s = Symbol::new(*id_tick, SymbolKind::External);
            var_for_symbol.insert(s, v);
            *id_tick += 1;
            (f64::NAN, s, 0)
        });
        value.2 += 1;
        value.1
    }

    /// Create a new Row object for the given constraint.
    ///
    /// The terms in the constraint will be converted to cells in the row. Any term in the
    /// constraint with a coefficient of zero is ignored. This method uses the `get_var_symbol`
    /// method to get the symbol for the variables added to the row. If the symbol for a given cell
    /// variable is basic, the cell variable will be substituted with the basic row.
    ///
    /// The necessary slack and error variables will be added to the row. If the constant for the
    /// row is negative, the sign for the row will be inverted so the constant becomes positive.
    ///
    /// The tag will be updated with the marker and error symbols to use for tracking the movement
    /// of the constraint in the tableau.
    fn create_row(&mut self, constraint: &Constraint) -> (Box<Row>, Tag) {
        let expr = constraint.expr();
        let mut row = Row::new(expr.constant);

        // Substitute the current basic variables into the row.
        for term in &expr.terms {
            if !near_zero(term.coefficient) {
                let symbol = self.get_var_symbol(term.variable);
                if let Some(other_row) = self.rows.get(&symbol) {
                    row.insert_row(other_row, term.coefficient);
                } else {
                    row.insert_symbol(symbol, term.coefficient);
                }
            }
        }

        let mut objective = self.objective.borrow_mut();

        // Add the necessary slack, error, and dummy variables.
        let tag = match constraint.op() {
            RelationalOperator::GreaterOrEqual | RelationalOperator::LessOrEqual => {
                let coeff = if constraint.op() == RelationalOperator::LessOrEqual {
                    1.0
                } else {
                    -1.0
                };
                let slack = Symbol::new(self.id_tick, SymbolKind::Slack);
                self.id_tick += 1;
                row.insert_symbol(slack, coeff);
                if constraint.strength() < Strength::REQUIRED {
                    let error = Symbol::new(self.id_tick, SymbolKind::Error);
                    self.id_tick += 1;
                    row.insert_symbol(error, -coeff);
                    objective.insert_symbol(error, constraint.strength().value());
                    Tag {
                        marker: slack,
                        other: error,
                    }
                } else {
                    Tag {
                        marker: slack,
                        other: Symbol::invalid(),
                    }
                }
            }
            RelationalOperator::Equal => {
                if constraint.strength() < Strength::REQUIRED {
                    let errplus = Symbol::new(self.id_tick, SymbolKind::Error);
                    self.id_tick += 1;
                    let errminus = Symbol::new(self.id_tick, SymbolKind::Error);
                    self.id_tick += 1;
                    row.insert_symbol(errplus, -1.0); // v = eplus - eminus
                    row.insert_symbol(errminus, 1.0); // v - eplus + eminus = 0
                    objective.insert_symbol(errplus, constraint.strength().value());
                    objective.insert_symbol(errminus, constraint.strength().value());
                    Tag {
                        marker: errplus,
                        other: errminus,
                    }
                } else {
                    let dummy = Symbol::new(self.id_tick, SymbolKind::Dummy);
                    self.id_tick += 1;
                    row.insert_symbol(dummy, 1.0);
                    Tag {
                        marker: dummy,
                        other: Symbol::invalid(),
                    }
                }
            }
        };

        // Ensure the row has a positive constant.
        if row.constant < 0.0 {
            row.reverse_sign();
        }
        (Box::new(row), tag)
    }

    /// Choose the subject for solving for the row.
    ///
    /// This method will choose the best subject for using as the solve
    /// target for the row. An invalid symbol will be returned if there
    /// is no valid target.
    ///
    /// The symbols are chosen according to the following precedence:
    ///
    /// 1) An external variable symbol (chosen deterministically).
    /// 2) A negative slack or error tag variable.
    ///
    /// If a subject cannot be found, an invalid symbol will be returned.
    fn choose_subject(row: &Row, tag: &Tag) -> Symbol {
        ////////////////////////////////////////////////////////////////////////////////
        // determinism:
        //
        // - `row.cells` 是 HashMap,迭代顺序不稳定.
        // - 旧实现“遇到第一个 External 就返回”,会导致 pivot 路径依赖 HashMap 迭代顺序.
        // - 这里改为:在所有 External 候选里选择 Ord 最小的那个,保证选择稳定.
        ////////////////////////////////////////////////////////////////////////////////
        let mut best_external: Option<Symbol> = None;
        for s in row.cells.keys() {
            if s.kind() != SymbolKind::External {
                continue;
            }
            best_external = Some(best_external.map_or(*s, |best| best.min(*s)));
        }
        if let Some(external) = best_external {
            return external;
        }
        if (tag.marker.kind() == SymbolKind::Slack || tag.marker.kind() == SymbolKind::Error)
            && row.coefficient_for(tag.marker) < 0.0
        {
            return tag.marker;
        }
        if (tag.other.kind() == SymbolKind::Slack || tag.other.kind() == SymbolKind::Error)
            && row.coefficient_for(tag.other) < 0.0
        {
            return tag.other;
        }
        Symbol::invalid()
    }

    /// Add the row to the tableau using an artificial variable.
    ///
    /// This will return false if the constraint cannot be satisfied.
    fn add_with_artificial_variable(&mut self, row: &Row) -> Result<bool, InternalSolverError> {
        // Create and add the artificial variable to the tableau
        let art = Symbol::new(self.id_tick, SymbolKind::Slack);
        self.id_tick += 1;
        self.rows.insert(art, Box::new(row.clone()));
        self.artificial = Some(Rc::new(RefCell::new(row.clone())));

        // Optimize the artificial objective. This is successful
        // only if the artificial objective is optimized to zero.
        let artificial = self.artificial.as_ref().unwrap().clone();
        self.optimize(&artificial)?;
        let success = near_zero(artificial.borrow().constant);
        self.artificial = None;

        // If the artificial variable is basic, pivot the row so that
        // it becomes basic. If the row is constant, exit early.
        if let Some(mut row) = self.rows.remove(&art) {
            if row.cells.is_empty() {
                return Ok(success);
            }
            let entering = Solver::any_pivotable_symbol(&row); // never External
            if entering.kind() == SymbolKind::Invalid {
                return Ok(false); // unsatisfiable (will this ever happen?)
            }
            row.solve_for_symbols(art, entering);
            self.substitute(entering, &row);
            self.rows.insert(entering, row);
        }

        // Remove the artificial row from the tableau
        for row in self.rows.values_mut() {
            row.remove(art);
        }
        self.objective.borrow_mut().remove(art);
        Ok(success)
    }

    /// Substitute the parametric symbol with the given row.
    ///
    /// This method will substitute all instances of the parametric symbol
    /// in the tableau and the objective function with the given row.
    fn substitute(&mut self, symbol: Symbol, row: &Row) {
        for (&other_symbol, other_row) in &mut self.rows {
            let constant_changed = other_row.substitute(symbol, row);
            if other_symbol.kind() == SymbolKind::External && constant_changed {
                let v = self.var_for_symbol[&other_symbol];
                // inline var_changed
                if self.should_clear_changes {
                    self.changed.clear();
                    self.should_clear_changes = false;
                }
                self.changed.insert(v);
            }
            if other_symbol.kind() != SymbolKind::External && other_row.constant < 0.0 {
                self.infeasible_rows.push(Reverse(other_symbol));
            }
        }
        self.objective.borrow_mut().substitute(symbol, row);
        if let Some(artificial) = self.artificial.as_ref() {
            artificial.borrow_mut().substitute(symbol, row);
        }
    }

    /// Optimize the system for the given objective function.
    ///
    /// This method performs iterations of Phase 2 of the simplex method
    /// until the objective function reaches a minimum.
    fn optimize(&mut self, objective: &RefCell<Row>) -> Result<(), InternalSolverError> {
        loop {
            let entering = Solver::get_entering_symbol(&objective.borrow());
            if entering.kind() == SymbolKind::Invalid {
                return Ok(());
            }
            let (leaving, mut row) = self
                .get_leaving_row(entering)
                .ok_or(InternalSolverError::ObjectiveUnbounded)?;
            // pivot the entering symbol into the basis
            row.solve_for_symbols(leaving, entering);
            self.substitute(entering, &row);
            if entering.kind() == SymbolKind::External && row.constant != 0.0 {
                let v = self.var_for_symbol[&entering];
                self.var_changed(v);
            }
            self.rows.insert(entering, row);
        }
    }

    /// Optimize the system using the dual of the simplex method.
    ///
    /// The current state of the system should be such that the objective
    /// function is optimal, but not feasible. This method will perform
    /// an iteration of the dual simplex method to make the solution both
    /// optimal and feasible.
    fn dual_optimize(&mut self) -> Result<(), InternalSolverError> {
        while let Some(Reverse(leaving)) = self.infeasible_rows.pop() {
            let row = if let Entry::Occupied(entry) = self.rows.entry(leaving) {
                if entry.get().constant < 0.0 {
                    Some(entry.remove())
                } else {
                    None
                }
            } else {
                None
            };
            if let Some(mut row) = row {
                let entering = self.get_dual_entering_symbol(&row);
                if entering.kind() == SymbolKind::Invalid {
                    return Err(InternalSolverError::DualOptimizeFailed);
                }
                // pivot the entering symbol into the basis
                row.solve_for_symbols(leaving, entering);
                self.substitute(entering, &row);
                if entering.kind() == SymbolKind::External && row.constant != 0.0 {
                    let v = self.var_for_symbol[&entering];
                    self.var_changed(v);
                }
                self.rows.insert(entering, row);
            }
        }
        Ok(())
    }

    /// Compute the entering variable for a pivot operation.
    ///
    /// This method will return a symbol in the objective function which
    /// is non-dummy and has a coefficient less than zero. If no symbol meets
    /// the criteria, it means the objective function is at a minimum, and an
    /// invalid symbol is returned.
    /// Could return an External symbol
    fn get_entering_symbol(objective: &Row) -> Symbol {
        ////////////////////////////////////////////////////////////////////////////////
        // determinism:
        //
        // - `objective.cells` 是 HashMap,迭代顺序不稳定.
        // - 旧实现“遇到第一个 value<0 就返回”,会导致 pivot 路径依赖迭代顺序.
        // - 这里改为:
        //   1) 选择系数最负(值最小)的候选作为 entering.
        //   2) 若系数相等,按 Symbol 的 Ord 做 tie-break.
        ////////////////////////////////////////////////////////////////////////////////
        let mut entering: Option<Symbol> = None;
        let mut best_value: f64 = 0.0;

        for (symbol, value) in &objective.cells {
            if symbol.kind() == SymbolKind::Dummy {
                continue;
            }
            if value.is_nan() || *value >= 0.0 {
                continue;
            }

            match entering {
                None => {
                    entering = Some(*symbol);
                    best_value = *value;
                }
                Some(current) => {
                    if *value < best_value || (*value == best_value && *symbol < current) {
                        entering = Some(*symbol);
                        best_value = *value;
                    }
                }
            }
        }

        entering.unwrap_or_else(Symbol::invalid)
    }

    /// Compute the entering symbol for the dual optimize operation.
    ///
    /// This method will return the symbol in the row which has a positive
    /// coefficient and yields the minimum ratio for its respective symbol
    /// in the objective function. The provided row *must* be infeasible.
    /// If no symbol is found which meats the criteria, an invalid symbol
    /// is returned.
    /// Could return an External symbol
    fn get_dual_entering_symbol(&self, row: &Row) -> Symbol {
        ////////////////////////////////////////////////////////////////////////////////
        // determinism:
        //
        // - `row.cells` 是 HashMap,迭代顺序不稳定.
        // - 旧实现当 ratio 相等时会“谁先被遍历到谁赢”,导致 entering 选择漂移.
        // - 这里改为:ratio 最小优先,ratio 相等时按 Symbol 的 Ord 做 tie-break.
        ////////////////////////////////////////////////////////////////////////////////
        let mut entering: Option<Symbol> = None;
        let mut ratio = f64::INFINITY;
        let objective = self.objective.borrow();
        for (symbol, value) in &row.cells {
            if *value > 0.0 && symbol.kind() != SymbolKind::Dummy {
                let coeff = objective.coefficient_for(*symbol);
                let r = coeff / *value;
                if r.is_nan() {
                    continue;
                }

                match entering {
                    None => {
                        ratio = r;
                        entering = Some(*symbol);
                    }
                    Some(current) => {
                        if r < ratio || (r == ratio && *symbol < current) {
                            ratio = r;
                            entering = Some(*symbol);
                        }
                    }
                }
            }
        }
        entering.unwrap_or_else(Symbol::invalid)
    }

    /// Get a Slack or Error symbol in the row.
    ///
    /// If no such symbol is present, and Invalid symbol will be returned.
    /// Never returns an External symbol
    fn any_pivotable_symbol(row: &Row) -> Symbol {
        ////////////////////////////////////////////////////////////////////////////////
        // determinism:
        //
        // - `row.cells` 是 HashMap,迭代顺序不稳定.
        // - 旧实现“遇到第一个 Slack/Error 就返回”,会导致 pivot 路径漂移.
        // - 这里改为:选择 Ord 最小的 Slack/Error.
        ////////////////////////////////////////////////////////////////////////////////
        let mut best: Option<Symbol> = None;
        for symbol in row.cells.keys() {
            if symbol.kind() != SymbolKind::Slack && symbol.kind() != SymbolKind::Error {
                continue;
            }
            best = Some(best.map_or(*symbol, |b| b.min(*symbol)));
        }
        best.unwrap_or_else(Symbol::invalid)
    }

    /// Compute the row which holds the exit symbol for a pivot.
    ///
    /// This method will return an iterator to the row in the row map
    /// which holds the exit symbol. If no appropriate exit symbol is
    /// found, the end() iterator will be returned. This indicates that
    /// the objective function is unbounded.
    /// Never returns a row for an External symbol
    fn get_leaving_row(&mut self, entering: Symbol) -> Option<(Symbol, Box<Row>)> {
        ////////////////////////////////////////////////////////////////////////////////
        // determinism:
        //
        // - `self.rows` 是 HashMap,迭代顺序不稳定.
        // - 旧实现当 ratio 相等时会“谁先被遍历到谁赢”,导致 leaving 选择漂移.
        // - 这里改为:ratio 最小优先,ratio 相等时按 Symbol 的 Ord 做 tie-break.
        ////////////////////////////////////////////////////////////////////////////////
        let mut ratio = f64::INFINITY;
        let mut found: Option<Symbol> = None;
        for (symbol, row) in &self.rows {
            if symbol.kind() != SymbolKind::External {
                let temp = row.coefficient_for(entering);
                if temp < 0.0 {
                    let temp_ratio = -row.constant / temp;
                    if temp_ratio.is_nan() {
                        continue;
                    }

                    match found {
                        None => {
                            ratio = temp_ratio;
                            found = Some(*symbol);
                        }
                        Some(current) => {
                            if temp_ratio < ratio || (temp_ratio == ratio && *symbol < current) {
                                ratio = temp_ratio;
                                found = Some(*symbol);
                            }
                        }
                    }
                }
            }
        }
        found.map(|s| (s, self.rows.remove(&s).unwrap()))
    }

    /// Compute the leaving row for a marker variable.
    ///
    /// This method will return an iterator to the row in the row map
    /// which holds the given marker variable. The row will be chosen
    /// according to the following precedence:
    ///
    /// 1) The row with a restricted basic varible and a negative coefficient for the marker with
    ///    the smallest ratio of -constant / coefficient.
    ///
    /// 2) The row with a restricted basic variable and the smallest ratio of constant /
    ///    coefficient.
    ///
    /// 3) A deterministically chosen unrestricted row which contains the marker.
    ///
    /// If the marker does not exist in any row, the row map end() iterator
    /// will be returned. This indicates an internal solver error since
    /// the marker *should* exist somewhere in the tableau.
    fn get_marker_leaving_row(&mut self, marker: Symbol) -> Option<(Symbol, Box<Row>)> {
        let mut r1 = f64::INFINITY;
        let mut r2 = r1;
        let mut first: Option<Symbol> = None;
        let mut second: Option<Symbol> = None;
        let mut third: Option<Symbol> = None;
        for (symbol, row) in &self.rows {
            let c = row.coefficient_for(marker);
            if c == 0.0 {
                continue;
            }
            if symbol.kind() == SymbolKind::External {
                // determinism: external fallback 也必须稳定,不能依赖 HashMap 迭代顺序.
                third = Some(third.map_or(*symbol, |t| t.min(*symbol)));
            } else if c < 0.0 {
                let r = -row.constant / c;
                if r.is_nan() {
                    continue;
                }
                match first {
                    None => {
                        r1 = r;
                        first = Some(*symbol);
                    }
                    Some(current) => {
                        if r < r1 || (r == r1 && *symbol < current) {
                            r1 = r;
                            first = Some(*symbol);
                        }
                    }
                }
            } else {
                let r = row.constant / c;
                if r.is_nan() {
                    continue;
                }
                match second {
                    None => {
                        r2 = r;
                        second = Some(*symbol);
                    }
                    Some(current) => {
                        if r < r2 || (r == r2 && *symbol < current) {
                            r2 = r;
                            second = Some(*symbol);
                        }
                    }
                }
            }
        }
        first.or(second).or(third).and_then(|s| {
            if s.kind() == SymbolKind::External && self.rows[&s].constant != 0.0 {
                let v = self.var_for_symbol[&s];
                self.var_changed(v);
            }
            self.rows.remove(&s).map(|r| (s, r))
        })
    }

    /// Remove the effects of a constraint on the objective function.
    fn remove_constraint_effects(&mut self, constraint: &Constraint, tag: &Tag) {
        if tag.marker.kind() == SymbolKind::Error {
            self.remove_marker_effects(tag.marker, constraint.strength().value());
        }
        if tag.other.kind() == SymbolKind::Error {
            self.remove_marker_effects(tag.other, constraint.strength().value());
        }
    }

    /// Remove the effects of an error marker on the objective function.
    fn remove_marker_effects(&mut self, marker: Symbol, strength: f64) {
        if let Some(row) = self.rows.get(&marker) {
            self.objective.borrow_mut().insert_row(row, -strength);
        } else {
            self.objective.borrow_mut().insert_symbol(marker, -strength);
        }
    }

    /// Test whether a row is composed of all dummy variables.
    fn all_dummies(row: &Row) -> bool {
        for symbol in row.cells.keys() {
            if symbol.kind() != SymbolKind::Dummy {
                return false;
            }
        }
        true
    }

    /// Get the stored value for a variable.
    ///
    /// Normally values should be retrieved and updated using `fetch_changes`, but this method can
    /// be used for debugging or testing.
    pub fn get_value(&self, v: Variable) -> f64 {
        self.var_data
            .get(&v)
            .and_then(|s| self.rows.get(&s.1).map(|r| r.constant))
            .unwrap_or(0.0)
    }
}
