use kasuari::WeightedRelation::*;
use kasuari::{Solver, Strength, Variable};

/// 这个测试的目标不是验证“哪一个解更正确”,
/// 而是验证:当存在多个同样有效的解时,solver 的选择路径是确定性的。
///
/// 这样可以避免 UI 布局在 underconstrained/等强度冲突场景里出现 jitter/offset。
#[test]
fn ambiguous_weak_constraints_should_be_deterministic_across_solver_instances() {
    // 固定变量,避免 Variable::new() 的全局计数器影响不同 run 之间的输入形态。
    let window_width = Variable::new();
    let box1_left = Variable::new();
    let box1_right = Variable::new();
    let box2_left = Variable::new();
    let box2_right = Variable::new();

    let solve = || -> (f64, f64, f64, f64) {
        let mut solver = Solver::new();

        solver
            .add_constraints([
                window_width | GE(Strength::REQUIRED) | 0.0, // window_width >= 0
                box1_left | EQ(Strength::REQUIRED) | 0.0,    // box1.left == 0
                box2_right | EQ(Strength::REQUIRED) | window_width, // box2.right == window_width
                box2_left | GE(Strength::REQUIRED) | box1_right, // no overlap
                box1_left | LE(Strength::REQUIRED) | box1_right, // box1 width >= 0
                box2_left | LE(Strength::REQUIRED) | box2_right, // box2 width >= 0
                // preferred widths (weak): 两条都满足时更好,冲突时可能会被违反。
                box1_right - box1_left | EQ(Strength::WEAK) | 50.0,
                box2_right - box2_left | EQ(Strength::WEAK) | 100.0,
            ])
            .unwrap();

        solver
            .add_edit_variable(window_width, Strength::STRONG)
            .unwrap();
        solver.suggest_value(window_width, 75.0).unwrap();

        // 触发一次变化收集,让测试口径更接近真实使用方式。
        let _ = solver.fetch_changes();

        (
            solver.get_value(box1_right),
            solver.get_value(box2_left),
            solver.get_value(box2_right),
            solver.get_value(window_width),
        )
    };

    let baseline = solve();

    // 多跑几次,用于捕捉 HashMap 迭代顺序或 tie-break 漂移导致的“不一致解”。
    for _ in 0..64 {
        let got = solve();
        assert_eq!(
            got, baseline,
            "solver should be deterministic: baseline={baseline:?} got={got:?}"
        );
    }
}

/// 欠约束系统:只约束相对关系,保留整体平移自由度。
/// determinism contract 要求:同平台/同工具链/同输入序列下,跨 Solver 实例解一致。
#[test]
fn underconstrained_chain_should_be_deterministic_across_solver_instances() {
    let a = Variable::new();
    let b = Variable::new();
    let c = Variable::new();

    let solve = || -> (f64, f64, f64) {
        let mut solver = Solver::new();

        solver
            .add_constraints([
                b | EQ(Strength::REQUIRED) | a + 8.0,
                c | EQ(Strength::REQUIRED) | b + 8.0,
            ])
            .unwrap();

        let _ = solver.fetch_changes();

        (
            solver.get_value(a),
            solver.get_value(b),
            solver.get_value(c),
        )
    };

    let baseline = solve();

    for _ in 0..64 {
        let got = solve();
        assert_eq!(
            got, baseline,
            "solver should be deterministic: baseline={baseline:?} got={got:?}"
        );

        // 同时验证相对关系始终成立(防止出现“确定但错了”的回归)。
        assert!(((got.1 - got.0) - 8.0).abs() < 1e-9);
        assert!(((got.2 - got.1) - 8.0).abs() < 1e-9);
    }
}
