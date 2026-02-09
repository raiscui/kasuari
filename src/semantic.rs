use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::row::near_zero;
use crate::{Constraint, RelationalOperator, Variable};

/// 约束语义指纹(ConstraintFingerprint).
///
/// # 设计动机
///
/// kasuari 的 `Constraint` 目前仍然是“指针身份”(Arc pointer identity):
/// - `Eq/Hash/Ord` 都基于内部 Arc 地址.
/// - 这对增量 remove/update 来说很高效,但对上层“频繁重建同语义约束对象”的场景非常不友好.
///
/// 因此我们需要一个“只看语义”的稳定 key:
/// - 同语义(terms 顺序不同但内容一致)的约束,必须得到相同 fingerprint.
/// - 该 fingerprint 可以用于 stable diff/sort/诊断,以及 reconcile 时复用旧 Constraint 对象.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConstraintFingerprint {
    op: RelationalOperator,
    strength_bits: u64,
    constant_bits: u64,
    /// terms 的顺序与表达式构造方式不保证等价表达式一致:
    /// - terms 先按 Variable 归并求和(消除重复项).
    /// - 系数 near-zero(EPS=1e-8) 视为 0 并删除该 term(与 solver 行为对齐).
    /// - 最终按 (var_id, coeff_bits) 排序,消除构造顺序差异.
    /// - coeff/constant 用 canonical bits 进入全序:
    ///   - ±0 统一为 +0
    ///   - NaN 统一为 canonical NaN bits(避免 payload 差异导致 fingerprint 不稳定)
    terms: Vec<(Variable, u64)>,
}

impl ConstraintFingerprint {
    /// 从约束语义构造 fingerprint.
    ///
    /// 说明:
    /// - 该函数刻意保持与 `emg_layout` 旧实现(`ConstraintStableKey`)一致, 以避免迁移时产生多余
    ///   remove+add 进而打散 solver basis.
    #[must_use]
    pub fn new(constraint: &Constraint) -> Self {
        ////////////////////////////////////////////////////////////////////////////////
        // 重要说明(稳定性优先):
        //
        // `ConstraintFingerprint` 的职责是 stable diff/sort/reconcile 的“语义 key”。
        // 因此这里的规范化必须同时满足:
        // - 不受 term 构造顺序影响(确定性)
        // - 与 solver 的“近似为 0”口径对齐(避免噪声系数导致误判 diff)
        // - 在 NaN/±0 等边界上可预测(避免跨运行漂移)
        ////////////////////////////////////////////////////////////////////////////////

        let expr = constraint.expr();

        // 1) terms:按 Variable 归并求和(去掉重复项)
        let mut merged: BTreeMap<Variable, f64> = BTreeMap::new();
        for t in &expr.terms {
            let entry = merged.entry(t.variable).or_insert(0.0);
            *entry += t.coefficient;
        }

        // 2) terms:删除 near-zero 系数,并把系数转成 canonical bits
        let mut terms: Vec<(Variable, u64)> = Vec::new();
        for (var, coefficient) in merged {
            if near_zero(coefficient) {
                // 与 solver 对齐:极小系数视为 0,不进入 fingerprint,避免误判 diff。
                continue;
            }
            terms.push((var, canon_f64_bits_for_fingerprint(coefficient)));
        }

        // 3) 语义稳定:按 (var_id, coeff_bits) 排序,避免 term 插入顺序影响 fingerprint
        terms.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        Self {
            op: constraint.op(),
            strength_bits: canon_f64_bits_for_fingerprint(constraint.strength().value()),
            constant_bits: canon_f64_bits_for_fingerprint(expr.constant),
            terms,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// f64 -> canonical bits(用于 ConstraintFingerprint)
//
// 设计目标:
// - 不追求跨平台 bit-level 一致,但要避免“同平台/同工具链内”的非必要漂移。
// - 把 solver 侧的 near_zero 口径(EPS=1e-8)纳入 fingerprint,减少噪声 diff。
////////////////////////////////////////////////////////////////////////////////
fn canon_f64_bits_for_fingerprint(value: f64) -> u64 {
    // 统一 NaN(无论 payload/符号位),避免 fingerprint 受 NaN bits 波动影响。
    const CANONICAL_NAN_BITS: u64 = 0x7ff8_0000_0000_0000;
    if value.is_nan() {
        return CANONICAL_NAN_BITS;
    }

    // 统一 ±0 以及 near-zero:对齐 solver 的 near_zero 口径。
    if near_zero(value) {
        return 0.0f64.to_bits();
    }

    value.to_bits()
}

/// 语义去重策略.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DuplicateFingerprintMode {
    /// 默认策略:去重并继续(重复约束不会用于“加权”).
    Ignore,
    /// 诊断策略:发现重复 fingerprint 直接返回错误.
    Error,
}

/// 诊断模式下的重复 fingerprint 错误.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("duplicate constraint fingerprint: count={count}")]
pub struct DuplicateFingerprintError {
    pub fingerprint: ConstraintFingerprint,
    pub count: usize,
}

/// 语义约束集合.
///
/// - key: `ConstraintFingerprint`(语义)
/// - value: `Constraint`(对象实例,用于 pointer-identity remove)
///
/// 选择 `BTreeMap` 的原因:
/// - 迭代顺序稳定,便于 stable sort/diff.
/// - 作为 reconcile 的基础数据结构,可以避免 HashMap 迭代顺序引入的额外漂移.
pub type SemanticConstraints = BTreeMap<ConstraintFingerprint, Constraint>;

/// 将一批约束转换为“语义集合”(默认去重,保留第一条).
#[must_use]
pub fn semantic_constraints<'a, I>(constraints: I) -> SemanticConstraints
where
    I: IntoIterator<Item = &'a Constraint>,
{
    // Ignore 模式不会失败,因此 unwrap 是安全的.
    semantic_constraints_with_mode(constraints, DuplicateFingerprintMode::Ignore)
        .expect("DuplicateFingerprintMode::Ignore should never error")
}

/// 将一批约束转换为“语义集合”,并按 mode 处理重复 fingerprint.
pub fn semantic_constraints_with_mode<'a, I>(
    constraints: I,
    mode: DuplicateFingerprintMode,
) -> Result<SemanticConstraints, DuplicateFingerprintError>
where
    I: IntoIterator<Item = &'a Constraint>,
{
    match mode {
        DuplicateFingerprintMode::Ignore => {
            let mut out: SemanticConstraints = SemanticConstraints::new();
            for c in constraints {
                let fp = ConstraintFingerprint::new(c);
                // 保留第一条,避免“重复输入”意外产生加权效果.
                out.entry(fp).or_insert_with(|| c.clone());
            }
            Ok(out)
        }
        DuplicateFingerprintMode::Error => {
            let mut out: SemanticConstraints = SemanticConstraints::new();
            let mut counts: BTreeMap<ConstraintFingerprint, usize> = BTreeMap::new();

            for c in constraints {
                let fp = ConstraintFingerprint::new(c);
                *counts.entry(fp.clone()).or_insert(0) += 1;
                out.entry(fp).or_insert_with(|| c.clone());
            }

            if let Some((fingerprint, count)) = counts.iter().find(|(_, c)| **c > 1) {
                return Err(DuplicateFingerprintError {
                    fingerprint: fingerprint.clone(),
                    count: *count,
                });
            }

            Ok(out)
        }
    }
}

/// 语义 merge(复用 previous 的旧对象).
///
/// 背景:
/// - 现在 kasuari 的 remove 仍然依赖 pointer identity.
/// - 上层在 rebuild/hard reset 时会重新构造同语义的新 Constraint 对象(指针不同).
/// - merge 的职责是:对同 fingerprint 的约束,优先复用 previous 的旧对象,从而保持 identity 稳定.
#[must_use]
pub fn merge_semantic_constraints(
    previous: &SemanticConstraints,
    newest: &SemanticConstraints,
) -> SemanticConstraints {
    let mut merged: SemanticConstraints = SemanticConstraints::new();

    for (fp, new_c) in newest.iter() {
        if let Some(old_c) = previous.get(fp) {
            merged.insert(fp.clone(), old_c.clone());
        } else {
            merged.insert(fp.clone(), new_c.clone());
        }
    }

    merged
}
