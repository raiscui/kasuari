use kasuari::WeightedRelation::*;
use kasuari::{
    merge_semantic_constraints, semantic_constraints, semantic_constraints_with_mode, Constraint,
    ConstraintFingerprint, DuplicateFingerprintMode, Expression, RelationalOperator, Strength,
    Term, Variable,
};

#[test]
fn constraint_fingerprint_should_be_semantic_only() {
    let x = Variable::new();

    // 同语义,不同对象(指针身份不同)
    let c0_a = x | EQ(Strength::MEDIUM) | 0.0;
    let c0_b = x | EQ(Strength::MEDIUM) | 0.0;
    assert_ne!(c0_a, c0_b, "Constraint Eq should be pointer identity");

    assert_eq!(
        ConstraintFingerprint::new(&c0_a),
        ConstraintFingerprint::new(&c0_b),
        "ConstraintFingerprint should ignore pointer identity"
    );
}

#[test]
fn semantic_constraints_should_dedup_by_fingerprint_keep_first() {
    let x = Variable::new();

    let first = x | EQ(Strength::MEDIUM) | 0.0;
    let second = x | EQ(Strength::MEDIUM) | 0.0;
    assert_ne!(first, second);

    let set = semantic_constraints([&first, &second]);
    assert_eq!(set.len(), 1);

    let fp = ConstraintFingerprint::new(&first);
    let kept = set.get(&fp).expect("should keep the fingerprint entry");
    assert_eq!(kept, &first, "should keep first constraint object");
    assert_ne!(kept, &second, "should not replace with later duplicate");
}

#[test]
fn semantic_constraints_error_mode_should_report_duplicate() {
    let x = Variable::new();

    let c0_a = x | EQ(Strength::MEDIUM) | 0.0;
    let c0_b = x | EQ(Strength::MEDIUM) | 0.0;
    let fp = ConstraintFingerprint::new(&c0_a);

    let err = semantic_constraints_with_mode([&c0_a, &c0_b], DuplicateFingerprintMode::Error)
        .expect_err("should error on duplicate fingerprints");

    assert_eq!(err.fingerprint, fp);
    assert_eq!(err.count, 2);
}

#[test]
fn merge_semantic_constraints_should_reuse_previous_object() {
    let x = Variable::new();

    let old = x | EQ(Strength::MEDIUM) | 0.0;
    let new = x | EQ(Strength::MEDIUM) | 0.0;
    assert_ne!(old, new);

    let previous = semantic_constraints([&old]);
    let newest = semantic_constraints([&new]);

    let merged = merge_semantic_constraints(&previous, &newest);

    let fp = ConstraintFingerprint::new(&old);
    let merged_c = merged.get(&fp).expect("merged should contain fingerprint");

    assert_eq!(
        merged_c, &old,
        "merged should reuse previous constraint object"
    );
    assert_ne!(merged_c, &new, "merged should not use newly created object");
}

#[test]
fn constraint_fingerprint_should_normalize_zero_and_near_zero_terms() {
    ////////////////////////////////////////////////////////////////////////////////
    // 回归测试(float 语义边界):
    //
    // - terms 里出现 ±0 / near-zero 系数时,应当与 solver 的 near_zero(EPS=1e-8)口径一致:
    //   - 这些 term 对求解等价于“不存在”
    //   - fingerprint 必须忽略它们,避免被误判为 diff 进而触发 remove+add
    ////////////////////////////////////////////////////////////////////////////////
    let x = Variable::new();
    let y = Variable::new();

    // baseline: e = 1*x + 0
    let baseline = Constraint::new(
        Expression::new(vec![Term::new(x, 1.0)], 0.0),
        RelationalOperator::Equal,
        Strength::MEDIUM,
    );

    // case1: e = 1*x + (-0)*y + (-0)
    let with_negative_zero = Constraint::new(
        Expression::new(vec![Term::new(x, 1.0), Term::new(y, -0.0)], -0.0),
        RelationalOperator::Equal,
        Strength::MEDIUM,
    );

    // case2: e = 1*x + (1e-9)*y + 0
    // 说明: 1e-9 < EPS(1e-8), solver 侧会当作 0,因此 fingerprint 也必须视为“无该 term”。
    let with_near_zero = Constraint::new(
        Expression::new(vec![Term::new(x, 1.0), Term::new(y, 1e-9)], 0.0),
        RelationalOperator::Equal,
        Strength::MEDIUM,
    );

    let fp_baseline = ConstraintFingerprint::new(&baseline);
    assert_eq!(
        fp_baseline,
        ConstraintFingerprint::new(&with_negative_zero),
        "negative zero term/constant should be normalized"
    );
    assert_eq!(
        fp_baseline,
        ConstraintFingerprint::new(&with_near_zero),
        "near-zero coefficient term should be dropped"
    );
}

#[test]
fn constraint_fingerprint_should_merge_duplicate_terms_and_drop_near_zero_sum() {
    ////////////////////////////////////////////////////////////////////////////////
    // 回归测试(terms 归并规则):
    //
    // Expression 的 Add/Sub 会直接 append terms,因此同一变量可能出现重复项。
    // solver 内部会在 Row::insert_symbol 时“按 Symbol 归并求和 + near-zero 删除”。
    // fingerprint 也必须做同样的归并,否则上层 stable diff 会被无意义的构造差异污染。
    ////////////////////////////////////////////////////////////////////////////////
    let x = Variable::new();
    let y = Variable::new();

    // 1) x + x  与  2*x 语义等价
    let expr_dup = Expression::new(vec![Term::new(x, 1.0), Term::new(x, 1.0)], 0.0);
    let expr_merged = Expression::new(vec![Term::new(x, 2.0)], 0.0);
    let c_dup = Constraint::new(expr_dup, RelationalOperator::Equal, Strength::MEDIUM);
    let c_merged = Constraint::new(expr_merged, RelationalOperator::Equal, Strength::MEDIUM);
    assert_eq!(
        ConstraintFingerprint::new(&c_dup),
        ConstraintFingerprint::new(&c_merged),
        "duplicate terms should be merged by summing coefficients"
    );

    // 2) x + (-x) + y  与  y 语义等价(抵消后系数为 0)
    let expr_cancel = Expression::new(
        vec![Term::new(x, 1.0), Term::new(x, -1.0), Term::new(y, 1.0)],
        0.0,
    );
    let expr_y_only = Expression::new(vec![Term::new(y, 1.0)], 0.0);
    let c_cancel = Constraint::new(expr_cancel, RelationalOperator::Equal, Strength::MEDIUM);
    let c_y_only = Constraint::new(expr_y_only, RelationalOperator::Equal, Strength::MEDIUM);
    assert_eq!(
        ConstraintFingerprint::new(&c_cancel),
        ConstraintFingerprint::new(&c_y_only),
        "merged coefficient should drop near-zero sum"
    );
}

#[test]
fn constraint_fingerprint_should_canonicalize_nan_bits() {
    ////////////////////////////////////////////////////////////////////////////////
    // 回归测试(NaN bits 规范化):
    //
    // - NaN 有多个 payload 表示(不同 bits 但都满足 is_nan())。
    // - fingerprint 若直接用 to_bits,可能因为 payload 差异导致“同一 NaN 语义”不稳定。
    // - 这里要求:任何 NaN 都映射到 canonical NaN bits,从而保证 fingerprint 可预测。
    ////////////////////////////////////////////////////////////////////////////////
    let x = Variable::new();

    // 两个不同 payload 的 NaN(都为 is_nan())
    let nan_a = f64::from_bits(0x7ff8_0000_0000_0001);
    let nan_b = f64::from_bits(0x7ff8_0000_0000_0002);
    assert!(nan_a.is_nan());
    assert!(nan_b.is_nan());

    // case1: NaN 作为常量
    let c1 = Constraint::new(
        Expression::new(vec![Term::new(x, 1.0)], nan_a),
        RelationalOperator::Equal,
        Strength::MEDIUM,
    );
    let c2 = Constraint::new(
        Expression::new(vec![Term::new(x, 1.0)], nan_b),
        RelationalOperator::Equal,
        Strength::MEDIUM,
    );
    assert_eq!(
        ConstraintFingerprint::new(&c1),
        ConstraintFingerprint::new(&c2),
        "NaN constant bits should be canonicalized"
    );

    // case2: NaN 作为系数
    let c3 = Constraint::new(
        Expression::new(vec![Term::new(x, nan_a)], 0.0),
        RelationalOperator::Equal,
        Strength::MEDIUM,
    );
    let c4 = Constraint::new(
        Expression::new(vec![Term::new(x, nan_b)], 0.0),
        RelationalOperator::Equal,
        Strength::MEDIUM,
    );
    assert_eq!(
        ConstraintFingerprint::new(&c3),
        ConstraintFingerprint::new(&c4),
        "NaN coefficient bits should be canonicalized"
    );
}
