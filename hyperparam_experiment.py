import argparse
import csv
import itertools
import json
import math
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from recbole_custom.quick_start import run_recbole

def _parse_seeds(seed_text: str):
    seeds = []
    for item in seed_text.split(","):
        item = item.strip()
        if not item:
            continue
        seeds.append(int(item))
    if not seeds:
        raise ValueError("至少需要提供一个 seed，例如 --seeds 2023,2024")
    return seeds


def _parse_target_params(target_text: str):
    if target_text is None:
        return None
    items = [x.strip() for x in target_text.split(",") if x.strip()]
    return items if items else None


def _load_search_space(path: str):
    search_path = Path(path)
    with open(search_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    if search_path.suffix.lower() == ".json":
        data = json.loads(raw_text)
    else:
        try:
            import yaml
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "未安装 PyYAML。请改用 .json 搜索空间文件，或安装依赖：pip install pyyaml"
            ) from e
        data = yaml.safe_load(raw_text)
    if not isinstance(data, dict) or not data:
        raise ValueError("搜索空间文件必须是非空字典")
    return data


def _build_config_files(dataset: str, model: str):
    config_files = ["./configs/general_full.yaml", f"./configs/dataset/{dataset}.yaml"]
    model_config_path = Path(f"./configs/model/{model}/{dataset}.yaml")
    if model_config_path.exists():
        config_files.append(str(model_config_path))
    return config_files


def _generate_trials(search_space: dict):
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _normalize_search_plan(raw_space: dict):
    """
    兼容两种搜索空间格式：
    1) 旧格式（全量组合）:
       learning_rate: [0.0005, 0.001]
       n_ui_layers: [1, 2]
    2) 新格式（分别实验 / OFAT）:
       mode: one_factor
       base_params: {learning_rate: 0.001, ...}
       separate:
         learning_rate: [0.0005, 0.001, 0.002]
         ...
    """
    if "separate" in raw_space:
        mode = raw_space.get("mode", "one_factor")
        if mode != "one_factor":
            raise ValueError("当使用 `separate` 字段时，仅支持 mode=one_factor")
        base_params = raw_space.get("base_params", {})
        separate = raw_space["separate"]
        if not isinstance(base_params, dict):
            raise ValueError("`base_params` 必须是字典")
        if not isinstance(separate, dict) or not separate:
            raise ValueError("`separate` 必须是非空字典，格式为 参数名 -> 候选值列表")
        for key, values in separate.items():
            if not isinstance(values, list) or len(values) == 0:
                raise ValueError(f"`separate.{key}` 的候选值必须是非空列表")
        return {"mode": "one_factor", "base_params": base_params, "search_space": separate}

    # 兼容旧格式，默认做笛卡尔积
    for key, values in raw_space.items():
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"参数 `{key}` 的候选值必须是非空列表")
    return {"mode": "grid", "base_params": {}, "search_space": raw_space}


def _generate_one_factor_trials(search_space: dict, base_params: dict):
    for param_name, values in search_space.items():
        for value in values:
            trial = dict(base_params)
            trial[param_name] = value
            yield trial, param_name, value


def _filter_search_space(search_space: dict, target_params):
    if not target_params:
        return search_space
    missed = [p for p in target_params if p not in search_space]
    if missed:
        raise ValueError(f"以下目标超参数未在搜索空间中定义: {missed}")
    return {k: search_space[k] for k in target_params}


def _to_float(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_stat_std(numbers):
    if len(numbers) <= 1:
        return 0.0
    return stdev(numbers)


def _effect_strength(results, param_name, metric_name):
    groups = defaultdict(list)
    for row in results:
        key = row["params"][param_name]
        groups[str(key)].append(row[metric_name])

    all_scores = [row[metric_name] for row in results]
    global_mean = mean(all_scores)

    between_ss = 0.0
    for score_list in groups.values():
        if not score_list:
            continue
        m = mean(score_list)
        between_ss += len(score_list) * ((m - global_mean) ** 2)

    total_ss = sum((x - global_mean) ** 2 for x in all_scores)
    if math.isclose(total_ss, 0.0):
        return 0.0
    return between_ss / total_ss


def _write_csv(path: Path, rows):
    headers = [
        "trial_id",
        "seed",
        "varied_param",
        "best_valid_score",
        "valid_metric_bigger",
        "params_json",
        "best_valid_result_json",
        "test_result_json",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "trial_id": row["trial_id"],
                    "seed": row["seed"],
                    "varied_param": row.get("varied_param", ""),
                    "best_valid_score": row["best_valid_score"],
                    "valid_metric_bigger": row["valid_metric_bigger"],
                    "params_json": json.dumps(row["params"], ensure_ascii=False),
                    "best_valid_result_json": json.dumps(row["best_valid_result"], ensure_ascii=False),
                    "test_result_json": json.dumps(row["test_result"], ensure_ascii=False),
                }
            )


def _write_markdown_report(path: Path, summary: dict):
    lines = []
    lines.append("# 超参数实验分析报告")
    lines.append("")
    lines.append(f"- 生成时间: {summary['timestamp']}")
    lines.append(f"- 模型: `{summary['model']}`")
    lines.append(f"- 数据集: `{summary['dataset']}`")
    lines.append(f"- 实验次数: `{summary['num_runs']}`")
    lines.append(f"- 指标: `{summary['metric_name']}`")
    lines.append("")

    best = summary["best_run"]
    lines.append("## 最优结果")
    lines.append("")
    lines.append(f"- trial_id: `{best['trial_id']}`")
    lines.append(f"- seed: `{best['seed']}`")
    lines.append(f"- best_valid_score: `{best['best_valid_score']:.6f}`")
    lines.append(f"- params: `{json.dumps(best['params'], ensure_ascii=False)}`")
    lines.append("")

    lines.append("## 超参数影响度（越高表示对指标变化贡献越大）")
    lines.append("")
    for item in summary["effects"]:
        lines.append(f"- `{item['param']}`: `{item['effect_strength']:.4f}`")
    lines.append("")

    lines.append("## 参数取值统计")
    lines.append("")
    for param_name, value_stats in summary["value_stats"].items():
        lines.append(f"### {param_name}")
        for vs in value_stats:
            lines.append(
                f"- 值 `{vs['value']}`: count={vs['count']}, mean={vs['mean']:.6f}, std={vs['std']:.6f}"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _build_progress_map(varied_params, seeds):
    per_param_total = defaultdict(int)
    for p in varied_params:
        per_param_total[p] += len(seeds)
    return {
        "total_runs": len(varied_params) * len(seeds),
        "completed_runs": 0,
        "per_param_total": dict(per_param_total),
        "per_param_completed": {k: 0 for k in per_param_total},
    }


def _print_progress(progress, run_start_time):
    completed = progress["completed_runs"]
    total = progress["total_runs"]
    elapsed = time.time() - run_start_time
    avg = elapsed / completed if completed else 0.0
    eta = avg * (total - completed)
    percent = (completed / total * 100.0) if total else 100.0

    print(
        f"[PROGRESS] {completed}/{total} ({percent:.2f}%) | "
        f"elapsed={elapsed:.1f}s | eta={eta:.1f}s"
    )
    for p in progress["per_param_total"]:
        done = progress["per_param_completed"][p]
        need = progress["per_param_total"][p]
        print(f"  - {p}: {done}/{need}")


def main():
    parser = argparse.ArgumentParser(description="运行 MMHCL/RecBole 风格超参数实验并自动生成分析报告")
    parser.add_argument("--model", type=str, default="MMHyperHawkes")
    parser.add_argument("--dataset", type=str, default="magazine_subscriptions")
    parser.add_argument(
        "--search-space",
        type=str,
        default="configs/hyperparam/MMHyperHawkes_magazine_subscriptions.json",
        help="搜索空间文件（推荐 JSON；也支持 YAML）",
    )
    parser.add_argument("--metric", type=str, default="best_valid_score", help="用于分析的指标字段")
    parser.add_argument("--seeds", type=str, default="2023", help="多个 seed 用逗号分隔")
    parser.add_argument("--max-trials", type=int, default=0, help="限制超参数组合数，0 表示不限制")
    parser.add_argument("--show-progress", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="outputs/hyperparam_analysis")
    parser.add_argument("--dry-run", action="store_true", help="只生成实验计划，不实际训练")
    parser.add_argument(
        "--target-params",
        type=str,
        default=None,
        help="仅实验指定超参数，多个用逗号分隔；不传则默认实验全部",
    )

    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    target_params = _parse_target_params(args.target_params)
    raw_space = _load_search_space(args.search_space)
    plan = _normalize_search_plan(raw_space)
    search_space = _filter_search_space(plan["search_space"], target_params)
    base_params = plan["base_params"]
    config_files = _build_config_files(args.dataset, args.model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    trial_id = 0
    if plan["mode"] == "one_factor":
        trial_records = list(_generate_one_factor_trials(search_space, base_params))
        trial_combos = [x[0] for x in trial_records]
        varied_params = [x[1] for x in trial_records]
    else:
        trial_combos = list(_generate_trials(search_space))
        varied_params = [",".join(search_space.keys())] * len(trial_combos)

    if args.max_trials > 0:
        trial_combos = trial_combos[: args.max_trials]
        varied_params = varied_params[: args.max_trials]

    print(f"[INFO] 模式: {plan['mode']}, 组合数: {len(trial_combos)}, seeds: {seeds}")
    if target_params:
        print(f"[INFO] 指定实验超参数: {target_params}")
    if plan["mode"] == "one_factor":
        print(f"[INFO] 基线参数: {base_params}")

    if args.dry_run:
        preview_path = output_dir / "trial_plan.json"
        payload = []
        for idx, combo in enumerate(trial_combos, start=1):
            payload.append(
                {
                    "trial_id": idx,
                    "varied_param": varied_params[idx - 1],
                    "params": combo,
                    "seeds": seeds,
                }
            )
        preview_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[DRY-RUN] 已生成实验计划: {preview_path}")
        return

    progress = _build_progress_map(varied_params, seeds)
    progress_path = output_dir / "progress.json"
    run_start_time = time.time()

    for combo, varied_param in zip(trial_combos, varied_params):
        for seed in seeds:
            trial_id += 1
            config_dict = dict(combo)
            config_dict["seed"] = seed
            config_dict["show_progress"] = args.show_progress

            print(
                f"\n[RUN] trial={trial_id}/{progress['total_runs']}, "
                f"varied_param={varied_param}, params={combo}, seed={seed}"
            )
            result = run_recbole(
                model=args.model,
                dataset=args.dataset,
                config_file_list=config_files,
                config_dict=config_dict,
            )

            row = {
                "trial_id": trial_id,
                "seed": seed,
                "varied_param": varied_param,
                "params": combo,
                "best_valid_score": _to_float(result.get("best_valid_score")),
                "valid_metric_bigger": bool(result.get("valid_score_bigger", True)),
                "best_valid_result": result.get("best_valid_result", {}),
                "test_result": result.get("test_result", {}),
            }
            runs.append(row)

            progress["completed_runs"] += 1
            progress["per_param_completed"][varied_param] += 1
            progress_path.write_text(
                json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            _print_progress(progress, run_start_time)

    if not runs:
        raise RuntimeError("未产生任何实验结果，请检查搜索空间或参数设置")

    metric_name = args.metric
    if metric_name != "best_valid_score":
        for row in runs:
            metric_val = row["best_valid_result"].get(metric_name)
            if metric_val is None:
                metric_val = row["test_result"].get(metric_name)
            row[metric_name] = _to_float(metric_val)
    else:
        for row in runs:
            row[metric_name] = row["best_valid_score"]

    valid_scores = [r[metric_name] for r in runs if not math.isnan(r[metric_name])]
    if not valid_scores:
        raise RuntimeError(f"指标 {metric_name} 无有效数值，请确认名字是否正确")

    valid_metric_bigger = runs[0]["valid_metric_bigger"]
    if valid_metric_bigger:
        best_run = max(runs, key=lambda x: x[metric_name])
    else:
        best_run = min(runs, key=lambda x: x[metric_name])

    effects = []
    for p in search_space.keys():
        effects.append(
            {
                "param": p,
                "effect_strength": _effect_strength(runs, p, metric_name),
            }
        )
    effects.sort(key=lambda x: x["effect_strength"], reverse=True)

    value_stats = {}
    for p in search_space.keys():
        groups = defaultdict(list)
        for row in runs:
            groups[str(row["params"][p])].append(row[metric_name])

        stats_list = []
        for value, score_list in groups.items():
            stats_list.append(
                {
                    "value": value,
                    "count": len(score_list),
                    "mean": mean(score_list),
                    "std": _safe_stat_std(score_list),
                }
            )
        stats_list.sort(
            key=lambda x: x["mean"], reverse=bool(valid_metric_bigger)
        )
        value_stats[p] = stats_list

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    summary = {
        "timestamp": timestamp,
        "model": args.model,
        "dataset": args.dataset,
        "mode": plan["mode"],
        "target_params": target_params if target_params else list(search_space.keys()),
        "base_params": base_params,
        "num_runs": len(runs),
        "metric_name": metric_name,
        "valid_metric_bigger": valid_metric_bigger,
        "best_run": {
            "trial_id": best_run["trial_id"],
            "seed": best_run["seed"],
            "best_valid_score": best_run[metric_name],
            "params": best_run["params"],
        },
        "effects": effects,
        "value_stats": value_stats,
    }

    csv_path = output_dir / "trials.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"

    _write_csv(csv_path, runs)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown_report(report_path, summary)

    print("\n[DONE] 超参数实验完成")
    print(f"- trials: {csv_path}")
    print(f"- summary: {summary_path}")
    print(f"- report: {report_path}")


if __name__ == "__main__":
    main()