# 积信通 · 科技企业可解释授信辅助引擎（云部署版）

基于创新积分2.0指标体系的科创授信决策辅助工具（第十七届工行杯参赛原型）。

## 功能
- 输入企业名称 → 自动匹配：
  - A股/新三板企业：双模型（Logistic+XGBoost）信用评分 + SHAP可解释图 + 风险等级 + 建议额度 + 工行产品匹配
  - 珠海本地企业：资质+专利+覆盖率三维画像 + 材料补全建议
- 珠海科创企业总库浏览（1594家）

## 本地运行
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 云部署（Streamlit Community Cloud，免费）
1. 把本目录上传为 GitHub 仓库（建议命名 `jixintong`）
2. 打开 https://share.streamlit.io 用 GitHub 账号登录
3. 点击 "New app" → 选择仓库 → Branch 选 main → Main file 填 `app.py`
4. 点击 Deploy，等待 3-5 分钟构建完成
5. 部署完成后会获得一个永久公网链接，任何人可访问

## 文件结构
```
app.py                    # 主程序
requirements.txt          # 依赖
models/                   # 训练好的模型（Logistic/XGBoost/Scaler）
data/                     # 数据文件（训练集3559家、珠海总库1594家、专利106家）
```

## 数据来源
- 国家知识产权局专利检索系统（专利数据）
- 珠海市科技创新局/工信局公示（企业名单）
- 东方财富/新浪财经/全国股转系统（财务数据）

## 说明
本工具为参赛原型，输出仅供参考，不构成实际授信决策。
