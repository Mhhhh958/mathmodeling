# AI ASSISTANCE NOTICE
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：ChatGPT；版本/型号：GPT-5.6 Sol（ChatGPT 2026-08-06更新版）；
# 开发机构/公司：OpenAI；版本发布日期：2026-08-06。
# 人工智能仅用于代码检查、调试建议与说明整理；最终算法、参数与结果由参赛队审查并由冻结复现链验证。
import q_joint_validation_step12 as q

_original_load = q.joblib.load

def _load_pipeline(path):
    obj = _original_load(path)
    if isinstance(obj, dict) and "pipeline" in obj:
        return obj["pipeline"]
    return obj

q.joblib.load = _load_pipeline

if __name__ == "__main__":
    q.main()
