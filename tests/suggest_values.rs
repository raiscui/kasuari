use kasuari::WeightedRelation::*;
use kasuari::{Solver, Strength, SuggestValueError, Variable};

#[test]
fn suggest_values_should_match_sequential_suggest_value() {
    let x = Variable::new();
    let y = Variable::new();

    let build_solver = || {
        let mut solver = Solver::new();
        solver
            .add_constraints([
                // 让解唯一且与 suggest 值兼容,避免“多个同样有效解”导致比较不稳定。
                x + y | EQ(Strength::REQUIRED) | 10.0,
                x | GE(Strength::REQUIRED) | 0.0,
                y | GE(Strength::REQUIRED) | 0.0,
            ])
            .unwrap();
        solver.add_edit_variable(x, Strength::STRONG).unwrap();
        solver.add_edit_variable(y, Strength::STRONG).unwrap();
        solver
    };

    // baseline: 逐个 suggest_value(每次都会 dual_optimize)
    let mut solver_seq = build_solver();
    solver_seq.suggest_value(x, 3.0).unwrap();
    solver_seq.suggest_value(y, 7.0).unwrap();
    let seq_1 = (solver_seq.get_value(x), solver_seq.get_value(y));

    // batch: suggest_values(只 dual_optimize 一次)
    let mut solver_batch = build_solver();
    solver_batch.suggest_values(&[(x, 3.0), (y, 7.0)]).unwrap();
    let batch_1 = (solver_batch.get_value(x), solver_batch.get_value(y));

    assert!(
        (seq_1.0 - batch_1.0).abs() < 1e-9 && (seq_1.1 - batch_1.1).abs() < 1e-9,
        "batch suggest should match sequential: seq={seq_1:?} batch={batch_1:?}"
    );

    // 再做一轮更新,用于捕捉“多次调用后状态漂移”的回归。
    solver_seq.suggest_value(x, 4.0).unwrap();
    solver_seq.suggest_value(y, 6.0).unwrap();
    let seq_2 = (solver_seq.get_value(x), solver_seq.get_value(y));

    solver_batch.suggest_values(&[(x, 4.0), (y, 6.0)]).unwrap();
    let batch_2 = (solver_batch.get_value(x), solver_batch.get_value(y));

    assert!(
        (seq_2.0 - batch_2.0).abs() < 1e-9 && (seq_2.1 - batch_2.1).abs() < 1e-9,
        "batch suggest should match sequential: seq={seq_2:?} batch={batch_2:?}"
    );
}

#[test]
fn suggest_values_should_be_atomic_on_unknown_edit_variable() {
    let x = Variable::new();
    let y = Variable::new();

    let mut solver = Solver::new();
    solver
        .add_constraints([x | GE(Strength::REQUIRED) | 0.0])
        .unwrap();
    solver.add_edit_variable(x, Strength::STRONG).unwrap();

    let before = solver.get_value(x);

    let err = solver
        .suggest_values(&[(x, 10.0), (y, 5.0)])
        .expect_err("y is not an edit variable, so suggest_values should fail");
    assert!(
        matches!(err, SuggestValueError::UnknownEditVariable),
        "unexpected error: {err:?}"
    );

    // 保证 \"atomic\": 失败时不修改 solver 状态。
    let after = solver.get_value(x);
    assert!(
        (before - after).abs() < 1e-9,
        "solver should not be mutated on error: before={before} after={after}"
    );
}
