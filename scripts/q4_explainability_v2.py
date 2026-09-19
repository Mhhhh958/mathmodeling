# AI ASSISTANCE NOTICE
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：ChatGPT；版本/型号：GPT-5.6 Sol（ChatGPT 2026-08-06更新版）；
# 开发机构/公司：OpenAI；版本发布日期：2026-08-06。
# 人工智能仅用于代码检查、调试建议与说明整理；最终算法、参数与结果由参赛队审查并由冻结复现链验证。
import matplotlib.pyplot as plt

import q4_explainability as base


def make_stability_figure_fixed(stab):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    data = [
        stab["importance_spearman_original_vs_perturbed"].to_numpy(float),
        stab["random_permutation_spearman_control"].to_numpy(float),
    ]
    ax.boxplot(data, tick_labels=["0.5% source-SD perturbation", "random permutation control"], showmeans=True)
    ax.set_ylabel("Spearman correlation of 28-feature explanation")
    ax.set_title("Explanation stability under small input perturbations")
    ax.axhline(0, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(base.FIG / "explanation_stability.png", dpi=180)
    plt.close(fig)


base.make_stability_figure = make_stability_figure_fixed

if __name__ == "__main__":
    base.main()
