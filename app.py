# -*- coding: utf-8 -*-
"""
积信通 - 科技企业可解释授信辅助引擎（Streamlit原型）v2.0
运行: streamlit run app.py
v2.0 改动：商务灰白配色、模型分歧预警、Tab报告、SHAP口径修正、代理评分分层、专利三态、产品预匹配
"""
import os
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm

# ---------- 中文字体保障 ----------
def _ensure_cn_font():
    try:
        _names = {f.name for f in _fm.fontManager.ttflist}
        if any(k in n for n in _names for k in ('YaHei', 'SimHei', 'SimSun', 'Noto Sans CJK', 'WenQuanYi', 'Source Han')):
            return
        _base = os.path.dirname(os.path.abspath(__file__))
        _local = os.path.join(_base, 'simhei.ttf')
        if os.path.exists(_local) and os.path.getsize(_local) > 1000000:
            _fm.fontManager.addfont(_local)
            try:
                _cache = os.path.join(os.path.expanduser('~'), '.cache', 'matplotlib')
                if os.path.isdir(_cache):
                    for _f in os.listdir(_cache):
                        if _f.startswith('fontlist'):
                            os.remove(os.path.join(_cache, _f))
                _fm._load_fontmanager(try_read_cache=False)
            except Exception:
                pass
    except Exception as _e:
        print('中文字体加载失败:', _e)
_ensure_cn_font()
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei', 'Microsoft YaHei', 'SimSun', 'Noto Sans CJK SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
import streamlit as st
import shap
import plotly.graph_objects as go

# ---------- 路径配置 ----------
BASE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(BASE, 'data')):
    DATA = os.path.join(BASE, 'data')
else:
    DATA = os.path.dirname(BASE)
MODEL = os.path.join(BASE, 'models')

st.set_page_config(page_title='积信通·科创授信辅助引擎', page_icon='🏦', layout='wide')

# ---------- 商务灰白配色 + 少量工行红 ----------
st.markdown("""
<style>
:root {
  --icbc-red: #C7000B;
  --icbc-red-light: #FBEAEA;
  --text-main: #1F2329;
  --text-sub: #646A73;
  --border: #E5E6EB;
  --bg-side: #F7F8FA;
  --bg-card: #FFFFFF;
}
.stApp { background: #FFFFFF; }
body, p, span, div, label { color: var(--text-main) !important; }
h1, h2, h3 { color: var(--text-main); font-weight: 700; }
.stCaption, small { color: var(--text-sub) !important; font-size: 13px !important; }
div[data-testid="stMarkdownContainer"] p { color: var(--text-main); font-size: 15px; line-height: 1.7; }
div[data-testid="stMarkdownContainer"] strong { color: var(--text-main); }
/* metric卡片：白底灰边，数值默认深灰，仅高风险用红 */
div[data-testid="stMetric"] {
  background: #FFFFFF;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
}
div[data-testid="stMetric"] label { color: var(--text-sub) !important; font-size: 13px !important; font-weight: 500; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: var(--text-main) !important; font-weight: 700 !important; font-size: 26px !important; }
div[data-testid="stMetric"] div[data-testid="stMetricDelta"] { color: var(--text-sub) !important; font-size: 12px !important; }
div.stButton > button {
  border-radius: 6px;
  border: 1px solid var(--border);
  background: #FFFFFF;
  color: var(--text-main) !important;
  font-weight: 500;
  font-size: 14px;
}
div.stButton > button:hover {
  background: #F2F3F5 !important;
  color: var(--text-main) !important;
  border-color: #C9CDD4 !important;
}
section[data-testid="stSidebar"] {
  background: var(--bg-side);
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: var(--text-main); }
section[data-testid="stSidebar"] .stRadio label { color: var(--text-main); }
div[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: 8px;
  background: #FFFFFF;
}
div[data-testid="stExpander"] summary {
  background: #F7F8FA !important;
  border-bottom: 1px solid var(--border);
  color: var(--text-main) !important;
}
div[data-testid="stExpander"] summary:hover {
  background: #F0F1F3 !important;
}
input, textarea, .stTextInput input, .stTextArea textarea,
.stTextInput > div, .stTextArea > div,
div[data-baseweb="input"], div[data-baseweb="textarea"] {
  background-color: #FFFFFF !important;
  color: var(--text-main) !important;
  border: 1px solid var(--border) !important;
  font-size: 15px !important;
  box-shadow: none !important;
}
input:focus, textarea:focus,
.stTextInput > div:focus-within, .stTextArea > div:focus-within,
div[data-baseweb="input"]:focus-within, div[data-baseweb="textarea"]:focus-within {
  border-color: #8BC8EA !important;
  box-shadow: none !important;
}
/* selectbox浅色 */
.stSelectbox > div > div {
  background-color: #FFFFFF !important;
  color: var(--text-main) !important;
}
div[role="radiogroup"] label, div[role="checkbox"] label { color: var(--text-main) !important; font-size: 14px; }
/* radio圆点：未选中灰色小圆，选中红色大圆，通过大小+颜色区分 */
div[role="radiogroup"] label svg {
  width: 14px !important; height: 14px !important;
  color: #C9CDD4 !important;
}
div[role="radiogroup"] label:has(input:checked) svg {
  width: 18px !important; height: 18px !important;
  color: var(--icbc-red) !important;
}
div[role="radiogroup"] label:has(input:checked) { color: var(--icbc-red) !important; }
header[data-testid="stHeader"] { background: #FFFFFF !important; }
section.main, div.main { background: #FFFFFF !important; }
a { color: var(--icbc-red) !important; }
div[role="radiogroup"] label svg, div[role="checkbox"] label svg { color: var(--icbc-red) !important; }
.stSelectbox > div > div { border: 1px solid var(--border) !important; }
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #F2F3F5; }
::-webkit-scrollbar-thumb { background: #C9CDD4; border-radius: 4px; }
hr { border-color: var(--border); }
div[data-testid="stTitle"], div[data-testid="stTitle"] h1 {
  background: transparent !important; border: none !important; padding: 0 !important;
}
div[data-testid="stCaption"] { background: transparent !important; border: none !important; padding: 0 !important; }
section[data-testid="stSidebar"] div[data-testid="stRadio"] label,
section[data-testid="stSidebar"] div[data-testid="stRadio"] label > div,
section[data-testid="stSidebar"] div[data-testid="stRadio"] label > div > div,
section[data-testid="stSidebar"] div[data-testid="stRadio"] label > div > div > div {
  background: transparent !important; box-shadow: none !important; border: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) > div > div,
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) > div > div > div {
  color: var(--icbc-red) !important; font-weight: 700;
}
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 2px solid var(--border); }
.stTabs [data-baseweb="tab"] {
  background: transparent; color: var(--text-sub); font-weight: 500;
  border-radius: 6px 6px 0 0; padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
  color: var(--icbc-red) !important; font-weight: 700;
  border-bottom: 2px solid var(--icbc-red);
}
div.warn-box {
  background: #FFF7E6; border-left: 4px solid #FA8C16;
  padding: 12px 16px; border-radius: 4px; margin: 8px 0;
}
div.warn-box p { color: #874D00 !important; margin: 0; }
/* dataframe表格：跟随主题，不加额外边框 */
/* checkbox保持默认样式，只改选中文字色 */
div[role="checkbox"] label:has(input:checked) { color: var(--icbc-red) !important; }
</style>
""", unsafe_allow_html=True)

# ---------- 加载资源 ----------
@st.cache_resource
def load_models():
    lr = joblib.load(os.path.join(MODEL, 'logistic.pkl'))
    xgb_m = joblib.load(os.path.join(MODEL, 'xgboost.pkl'))
    scaler = joblib.load(os.path.join(MODEL, 'scaler.pkl'))
    with open(os.path.join(MODEL, 'model_meta.json'), 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return lr, xgb_m, scaler, meta

@st.cache_data
def load_data():
    train = pd.read_excel(os.path.join(DATA, '合并训练数据集_A股+新三板.xlsx'))
    zhuhai = pd.read_excel(os.path.join(DATA, '珠海科创企业总库.xlsx'))
    patents = pd.read_excel(os.path.join(DATA, '创新百强企业专利数据_106家.xlsx'))
    if '专利申请总量' not in zhuhai.columns:
        zhuhai = zhuhai.merge(patents[['企业名称', '专利申请总量']], on='企业名称', how='left')
    return train, zhuhai, patents

@st.cache_resource
def load_explainers():
    lr, xgb_m, scaler, meta = load_models()
    try:
        explainer_xgb = shap.TreeExplainer(xgb_m)
        train_raw = pd.read_excel(os.path.join(DATA, '合并训练数据集_A股+新三板.xlsx'))
        bg = train_raw[meta['features']].fillna(train_raw[meta['features']].median())
        bg_s = scaler.transform(bg)
        explainer_lr = shap.LinearExplainer(lr, bg_s)
        return explainer_xgb, explainer_lr
    except Exception as e:
        st.warning(f'SHAP解释器加载受限：{e}')
        return None, None

lr, xgb_m, scaler, meta = load_models()
train, zhuhai, patents = load_data()
explainer_xgb, explainer_lr = load_explainers()
FEATURES = meta['features']
medians = train[FEATURES].median()

# ---------- 模型元信息 ----------
MODEL_META = {
    '版本': 'v1.2',
    '训练样本': f"{int(meta.get('n_samples', 3559)):,}家（A股294+新三板3265）",
    '训练时间': '2026-09',
    '特征数量': f"{len(FEATURES)}项财务+研发特征",
    '标签定义': '近2年是否出现亏损/ST/退市风险警示（高风险=1）',
    'OOT测试期': '2025年度',
    '融合AUC': f"{meta.get('auc_fusion', 0.8614):.4f}",
    '准确率': f"{meta.get('accuracy', 0.817)*100:.1f}%",
    'KS': '0.58（待最终验证集确认）',
    'PR-AUC': '0.74（待最终验证集确认）',
}

# ---------- 工具函数 ----------
def to_score(p_risk):
    return round((1 - p_risk) * 100, 1)

def risk_level(score):
    if score >= 80: return 'A（低风险）'
    elif score >= 60: return 'B（中低风险）'
    elif score >= 40: return 'C（中风险）'
    else: return 'D（高风险）'

def credit_action(score):
    if score >= 80:
        return '建议进入常规授信评估，可适用绿色通道与利率优惠', 'A'
    elif score >= 60:
        return '建议进入常规授信评估，补充1-2项材料后由行内测算额度', 'B'
    elif score >= 40:
        return '建议降额评估，增加担保或引入知识产权质押，额度由行内测算', 'C'
    else:
        return '建议暂缓授信，提示具体风险点，补充材料后可重新评估', 'D'

DIVERGENCE_THRESHOLD = 0.50

def product_prematch(score, info=None):
    results = []
    kc_match = '高' if score >= 75 else ('中' if score >= 60 else '低')
    kc_satisfied = ['模型辅助评分达到准入参考区间'] if score >= 60 else []
    kc_pending = ['营业收入与现金流核验', '负债结构分析', '融资需求合理性']
    if info and info.get('rank', 0) >= 1:
        kc_satisfied.append('具备高新技术企业等科创资质')
    results.append({'产品': '科创贷（信用贷款）', '匹配度': kc_match, '已满足': kc_satisfied, '待核验': kc_pending})
    if info and info.get('pat_grade') in ('高', '中'):
        results.append({'产品': '知识产权质押贷', '匹配度': '高',
                        '已满足': ['持有有效专利', '专利分级达到中/高'],
                        '待核验': ['专利估值', '质押登记可行性', '衔接珠海4000万风险补偿池']})
    if info and info.get('sme') == '是':
        results.append({'产品': '人才贷', '匹配度': '中',
                        '已满足': ['专精特新中小企业资质'],
                        '待核验': ['核心团队稳定性', '人才结构']})
    return results

def coverage_analysis(zhuhai_row):
    covered = []
    typ = str(zhuhai_row.get('企业类型', ''))
    sme = zhuhai_row.get('专精特新中小企业', '')
    giant = zhuhai_row.get('专精特新小巨人', '')
    pat = zhuhai_row.get('专利申请总量', 0)
    if '高新技术企业' in typ or '高企' in typ:
        covered += ['研发投入强度（高企认定材料）', '高新技术产品收入占比（高企认定材料）']
    if '创新型' in typ:
        covered.append('研发人员占比（创新型中小企业自评材料）')
    if pd.notna(sme) and sme:
        covered.append('研发费用加计扣除（专精特新申报材料）')
    if pd.notna(giant) and giant:
        covered.append('利润率（小巨人财务材料）')
    if pd.notna(pat) and float(pat) > 0:
        covered.append('每百人研发人员知识产权数量（专利数据）')
    if '百强' in typ or '高成长' in typ:
        covered += ['入选高成长企业（加分项）', '入选创新百强（加分项）']
    if '独角兽' in typ or '瞪羚' in typ:
        covered.append('获得省级以上科技奖励（加分项）')
    covered = list(dict.fromkeys(covered))
    ratio = round(len(covered) / 12 * 100, 1)
    missing = ['技术合同成交额', '营收增长率', '资产负债率', '重点研发计划参与情况', '发明专利质量']
    missing = [m for m in missing if not any(m in c for c in covered)]
    return ratio, covered, missing[:5]

_FEATURE_UNITS = {
    '营业总收入': '万元', '销售毛利率': '%', '研发费用': '万元', '营收增长率': '%',
    '研发人员数': '人', '员工人数': '人', '资产负债率': '%', '研发费用占比': '%',
    '营收对数': 'ln(万元)', '研发费用对数': 'ln(万元)', '研发人员占比': '%',
}

def build_report(feature_row, show_shap=True):
    X = pd.DataFrame([feature_row], columns=FEATURES)[FEATURES]
    X = X.fillna(medians)
    X_s = scaler.transform(X)
    p_lr = lr.predict_proba(X_s)[:, 1][0]
    p_xgb = xgb_m.predict_proba(X)[:, 1][0]
    divergence = abs(p_lr - p_xgb)
    is_divergent = divergence > DIVERGENCE_THRESHOLD

    if is_divergent:
        p_fusion = None; score = None; level = None; grade = None
        action = '两模型风险判断差异较大，本次不自动生成评分，建议人工复核关键财务指标'
    else:
        p_fusion = 0.4 * p_lr + 0.6 * p_xgb
        score = to_score(p_fusion)
        level = risk_level(score)
        action, grade = credit_action(score)

    shap_fig = None
    shap_summary = None
    if show_shap and explainer_xgb and explainer_lr and not is_divergent:
        try:
            sv_xgb = explainer_xgb.shap_values(X)
            sv_lr = explainer_lr.shap_values(X_s)
            sv_fusion = 0.4 * np.array(sv_lr) + 0.6 * np.array(sv_xgb)
            sv_fusion = sv_fusion.reshape(-1)
            base = float(explainer_xgb.expected_value)
            pos_idx = np.argsort(-sv_fusion)[:3]
            neg_idx = np.argsort(sv_fusion)[:3]
            favorable = [FEATURES[i] for i in neg_idx if sv_fusion[i] < 0]
            concern = [FEATURES[i] for i in pos_idx if sv_fusion[i] > 0]
            shap_summary = {'有利因素': favorable, '关注因素': concern, 'base': base}
            fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(FEATURES))))
            order = np.argsort(-np.abs(sv_fusion))[:8]
            labels = [FEATURES[i] + (f"({_FEATURE_UNITS[FEATURES[i]]})" if FEATURES[i] in _FEATURE_UNITS else "") for i in order][::-1]
            vals = [sv_fusion[i] for i in order][::-1]
            colors = ['#E5533C' if v > 0 else '#3C9AE5' for v in vals]
            ax.barh(labels, vals, color=colors)
            ax.axvline(0, color='#999', lw=0.8)
            ax.set_xlabel('SHAP值（对风险概率的贡献方向，log-odds空间）')
            ax.set_title('特征对风险判断的贡献方向（TreeSHAP+LinearSHAP融合）', fontsize=12)
            ax.tick_params(labelsize=10)
            plt.tight_layout()
            shap_fig = fig
        except Exception:
            shap_fig = None; shap_summary = None

    return {
        'p_lr': p_lr, 'p_xgb': p_xgb, 'divergence': divergence, 'is_divergent': is_divergent,
        'p_fusion': p_fusion, 'score': score, 'level': level,
        'action': action, 'grade': grade,
        'shap_fig': shap_fig, 'shap_summary': shap_summary,
        'feature_values': X.iloc[0].to_dict(),
    }

def patent_status(row):
    try:
        valid = row.get('有效专利数')
        if pd.notna(valid) and float(valid) > 0:
            return '确认有', f"{int(float(valid))}件有效专利"
    except (ValueError, TypeError):
        pass
    try:
        total = row.get('专利申请总量')
        src = str(row.get('数据来源', ''))
        if pd.notna(total) and float(total) == 0 and src and src not in ('', 'nan'):
            return '确认无', '经公开渠道核查未发现专利申请记录'
    except (ValueError, TypeError):
        pass
    return '暂未匹配', '建议在国家知识产权局专利检索系统复核（pss-system.cponline.cnipa.gov.cn）'

def zhuhai_info(row):
    typ = str(row.get('企业类型', ''))
    sme = '是' if pd.notna(row.get('专精特新中小企业')) and row.get('专精特新中小企业') else '否'
    giant = '是' if pd.notna(row.get('专精特新小巨人')) and row.get('专精特新小巨人') else '否'
    pat = row.get('专利申请总量', 0)
    pat_lvl = row.get('专利数量分级', '')
    if pd.isna(pat):
        pat, pat_lvl = 0, '无数据'
    ratio, covered, missing = coverage_analysis(row)

    rank = 0
    if '高新技术企业' in typ: rank = max(rank, 1)
    if '创新型' in typ: rank = max(rank, 2)
    if sme == '是': rank = max(rank, 3)
    if giant == '是': rank = max(rank, 4)
    rank_names = {0: '未明确科创资质', 1: '高新技术企业', 2: '创新型中小企业', 3: '专精特新中小企业', 4: '专精特新小巨人'}
    rank_name = rank_names[rank]

    try:
        pat_n = float(pat)
        if pat_n >= 100: pat_grade = '高'
        elif pat_n >= 20: pat_grade = '中'
        elif pat_n > 0: pat_grade = '低'
        else: pat_grade = '无'
    except (ValueError, TypeError):
        pat_grade = '无'

    if ratio < 50:
        coverage_tier = '证据不足'
        proxy_display = '暂不评分'
        proxy_conclusion = f'当前指标覆盖率仅{ratio}%（覆盖{len(covered)}/12项），证据不足，暂不给出创新能力评分，建议补充核心材料后再评估。'
    elif ratio < 75:
        coverage_tier = '初步评估'
        base_score = ratio + (5 if rank >= 3 else 0) + (5 if pat_grade == '高' else (3 if pat_grade == '中' else 0))
        proxy_display = f"{round(min(base_score, 90), 1)}（初步）"
        proxy_conclusion = f'覆盖率{ratio}%，可给出初步创新画像，完整评分需补充材料后确认。'
    else:
        coverage_tier = '可评分'
        base_score = ratio + (5 if rank >= 3 else 0) + (5 if pat_grade == '高' else (3 if pat_grade == '中' else 0))
        proxy_display = f"{round(min(base_score, 90), 1)}"
        proxy_conclusion = f'覆盖率{ratio}%，材料较完整，可给出创新能力参考评分。'

    pat_status, pat_desc = patent_status(row)

    try:
        valid_pat = float(row.get('有效专利数'))
        if pd.isna(valid_pat): valid_pat = None
    except (ValueError, TypeError):
        valid_pat = None
    tax = str(row.get('纳税信用等级', '')).strip()
    if tax in ('', 'nan') or '未公开' in tax: tax = '未公开'
    try:
        years = row.get('成立年限')
        if pd.isna(years): years = None
    except Exception:
        years = None

    return {
        'rank_name': rank_name, 'rank': rank, 'sme': sme, 'giant': giant,
        'pat': pat_n if isinstance(pat, (int, float)) else 0,
        'pat_lvl': pat_lvl, 'pat_grade': pat_grade,
        'pat_status': pat_status, 'pat_desc': pat_desc,
        'coverage': ratio, 'covered': covered, 'missing': missing,
        'coverage_tier': coverage_tier, 'proxy_display': proxy_display,
        'proxy_conclusion': proxy_conclusion,
        'valid_pat': valid_pat, 'tax': tax, 'years': years,
    }

# ---------- 界面 ----------
st.title('积信通 · 科技企业可解释授信辅助引擎')
st.caption('基于创新积分2.0指标体系的科创授信决策辅助工具（工行杯参赛原型 · 模型辅助分，非工行内部评分）')

try:
    _overlap = len(set(train['企业名称'].astype(str).str.strip()) & set(zhuhai['企业名称'].astype(str).str.strip()))
except Exception:
    _overlap = 0
try:
    _total_pat = int(patents['专利申请总量'].sum(skipna=True)) if '专利申请总量' in patents.columns else None
except Exception:
    _total_pat = None
st.markdown('#### 📊 可查询范围')
_c1, _c2, _c3, _c4 = st.columns(4)
_c1.metric('模型评分企业', f"{int(meta['n_samples']):,}家", delta='A股294 + 新三板3265')
_c2.metric('珠海本地企业', f"{len(zhuhai):,}家", delta='三维画像评估')
_c3.metric('专利覆盖', (f"{_total_pat:,}件" if _total_pat else '—'), delta='创新百强106家')
_c4.metric('去重合计可查', f"{int(meta['n_samples']) + len(zhuhai) - _overlap:,}家", delta='含两类重叠企业')
st.markdown(
    '**查询说明**：输入企业名称后自动匹配——'
    '① **A股/新三板企业**输出完整模型报告（模型辅助评分、风险等级、授信动作建议、SHAP风险解释、工行产品预匹配）；'
    '② **珠海本地企业**财务数据不公开，输出"资质＋专利＋覆盖率"三维画像与材料补充建议。'
)
st.markdown('**覆盖名单**：高新技术企业、创新型中小企业、专精特新中小企业、专精特新小巨人、创新百强、科技型中小企业等8类。**示例查询**：立讯精密、深科技、珠海格力大金机电、格力钛新能源。')
st.divider()

with st.sidebar:
    st.header('⚙️ 查询设置')
    mode = st.radio('查询模式', ['企业名称查询', '珠海企业浏览'], index=0)
    st.divider()
    st.divider()
    with st.expander('模型信息', expanded=True):
        st.caption("版本 v1.2 ｜ 2026-09")
        st.caption("样本 3,559家（A股294+新三板3265）")
        st.caption("特征 11项财务+研发特征")
        st.caption("标签：近2年是否出现亏损/ST/退市风险警示（高风险=1）")
        st.caption("OOT：2026年度")
        st.caption("AUC 0.8614 ｜ 准确率 81.7%")
        st.caption("KS 0.58（待最终验证集确认）")
        st.caption("PR-AUC 0.74（待最终验证集确认）")
    st.divider()
    st.caption('注：本工具输出为模型参考值，不构成最终授信决策。')

if mode == '企业名称查询':
    if 'sel_company' not in st.session_state:
        st.session_state.sel_company = ''
    st.markdown('<h3 style="color:var(--text-main);font-size:16px;margin:8px 0;">🎯 演示案例（答辩直接点击）</h3>', unsafe_allow_html=True)
    _dc1, _dc2, _dc3 = st.columns(3)
    if _dc1.button('🌟 案例A：立讯精密（A股龙头）', use_container_width=True):
        st.session_state.sel_company = '立讯精密'; st.rerun()
    if _dc2.button('📊 案例B：深科技（A股）', use_container_width=True):
        st.session_state.sel_company = '深科技'; st.rerun()
    if _dc3.button('🏙️ 案例C：格力大金机电（珠海本地）', use_container_width=True):
        st.session_state.sel_company = '珠海格力大金机电设备有限公司'; st.rerun()
    q = st.text_input('请输入企业名称（支持模糊匹配）', value=st.session_state.sel_company, placeholder='如：立讯精密 或 格力')
    q = q.strip()
    if q:
        _all_names = pd.concat([train['企业名称'], zhuhai['企业名称']]).astype(str).str.strip()
        _exact = _all_names[_all_names == q].unique()
        if len(_exact) == 1:
            pass  # 精确匹配唯一一家，不弹推荐
        else:
            _sug = _all_names[_all_names.str.contains(q, na=False)].drop_duplicates().head(12).tolist()
            if _sug:
                st.caption(f'🔍 匹配到 {len(_sug)} 家企业，点击直接查询：')
                _cols = st.columns(3)
                for i, _s in enumerate(_sug):
                    if _cols[i % 3].button(_s, key=f'sug_{i}', use_container_width=True):
                        st.session_state.sel_company = _s; st.rerun()
            else:
                st.warning(f'未找到与「{q}」匹配的企业。')
    if q:
        train_hits = train[train['企业名称'].str.contains(q, na=False)]
        zhuhai_hits = zhuhai[zhuhai['企业名称'].str.contains(q, na=False)]
        if len(train_hits) == 0 and len(zhuhai_hits) == 0:
            st.warning(f'未找到与「{q}」匹配的企业。')
        else:
            for _, row in train_hits.head(3).iterrows():
                name = row['企业名称']
                rep = build_report(row)
                st.subheader(f'📋 {name}（{row.get("数据来源", "")}）')
                tab1, tab2, tab3, tab4 = st.tabs(['审批摘要', '特征证据', 'SHAP风险解释', '数据与合规'])
                with tab1:
                    if rep['is_divergent']:
                        st.markdown(
                            f'<div class="warn-box"><p>⚠️ <strong>模型分歧预警</strong>：Logistic判断高风险概率 {rep["p_lr"]*100:.1f}%，'
                            f'XGBoost判断高风险概率 {rep["p_xgb"]*100:.1f}%，两模型差异 {rep["divergence"]*100:.1f}个百分点（超过50%阈值）。'
                            f'本次不自动生成融合评分，建议人工复核关键财务指标（营收、毛利率、研发费用、资产负债率）。</p></div>',
                            unsafe_allow_html=True)
                        st.markdown(f'**模型构成**：Logistic {rep["p_lr"]*100:.1f}%风险 × 0.4 ＋ XGBoost {rep["p_xgb"]*100:.1f}%风险 × 0.6')
                    else:
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric('模型辅助评分', rep['score'], delta='0-100分（非工行内部评分）')
                        c2.metric('风险等级', rep['level'])
                        c3.metric('高风险概率', f"{rep['p_fusion']*100:.1f}%")
                        c4.metric('授信动作', rep['grade'] + '级')
                        _gcolor = '#2E9E5B' if rep['score'] >= 80 else ('#8BC8EA' if rep['score'] >= 60 else ('#FAAD14' if rep['score'] >= 40 else '#EA6668'))
                        _gauge = go.Figure(go.Indicator(
                            mode='gauge+number', value=rep['score'],
                            number={'font': {'size': 36, 'color': _gcolor}},
                            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': _gcolor, 'thickness': 0.3},
                                   'steps': [{'range': [0, 40], 'color': '#FBE3E3'}, {'range': [40, 60], 'color': '#FEF3CD'},
                                             {'range': [60, 80], 'color': '#DCEFFB'}, {'range': [80, 100], 'color': '#D9F2E0'}]},
                            title={'text': '模型辅助评分仪表盘', 'font': {'size': 13}}))
                        _gauge.update_layout(
                            height=220, margin=dict(l=20, r=20, t=40, b=10),
                            paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
                            font=dict(color='#1F2329', size=12)
                        )
                        st.plotly_chart(_gauge, use_container_width=True)
                        st.markdown(f'**授信动作建议**：{rep["action"]}')
                        st.markdown(f'**模型构成**：Logistic {rep["p_lr"]*100:.1f}%风险 × 0.4 ＋ XGBoost {rep["p_xgb"]*100:.1f}%风险 × 0.6')
                        st.markdown('**工行产品预匹配**')
                        for pm in product_prematch(rep['score']):
                            st.markdown(f'- **《{pm["产品"]}》** 匹配度：{pm["匹配度"]}')
                            if pm['已满足']:
                                st.markdown(f'  - 已满足：{"、".join(pm["已满足"])}')
                            st.markdown(f'  - 待核验：{"、".join(pm["待核验"])}')
                with tab2:
                    st.markdown('**模型输入特征明细（11项）**')
                    fv = rep['feature_values']
                    rows = []
                    for f in FEATURES:
                        v = fv.get(f, '')
                        unit = _FEATURE_UNITS.get(f, '')
                        rows.append({'特征': f, '取值': f"{v:.2f}" if isinstance(v, (int, float)) else str(v), '单位': unit})
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    st.caption('以上特征为模型实际输入值，缺失值以训练集中位数填充。')
                with tab3:
                    if rep['shap_summary']:
                        s = rep['shap_summary']
                        st.markdown('**机器判断摘要**')
                        if s['有利因素']:
                            st.markdown(f'- ✅ **有利因素（降低风险判断）**：{"、".join(s["有利因素"])}')
                        if s['关注因素']:
                            st.markdown(f'- ⚠️ **关注因素（提高风险判断）**：{"、".join(s["关注因素"])}')
                        st.markdown(f'*基线值（全体样本平均风险判断）：{s["base"]:.2f}（log-odds空间）*')
                        st.divider()
                    if rep['shap_fig']:
                        st.pyplot(rep['shap_fig'])
                        st.caption('说明：SHAP值为正表示该特征推高了模型的风险判断，为负表示降低了风险判断；单位为log-odds，非评分点数。')
                    elif rep['is_divergent']:
                        st.warning('因两模型风险判断差异过大（见"审批摘要"页模型分歧预警），本次不自动生成SHAP解释，建议人工复核关键财务指标后再查看。')
                    else:
                        st.info('SHAP解释暂不可用：解释器加载失败或计算异常，不影响评分结果。')
                with tab4:
                    st.markdown('**数据来源**：企业财务指标来自东方财富/新浪财经/全国股转系统公开年报；模型训练标签为近2年亏损/ST/退市风险警示。')
                    st.markdown('**合规声明**：本工具所有数据均来自公开渠道，不涉及工商银行内部客户数据；输出为模型参考值，不构成最终授信决策，实际审批以银行尽调为准。')
                    st.markdown('**模型适用性**：本模型适用于A股及新三板科技企业的风险初筛，不适用于金融、房地产等非科技行业，亦不替代银行KYC、反洗钱、终审决策等强监管环节。')
                st.divider()

            for _, row in zhuhai_hits.head(3).iterrows():
                name = row['企业名称']
                info = zhuhai_info(row)
                st.subheader(f'🏙️ {name}（珠海本地企业）')
                zt1, zt2, zt3 = st.tabs(['画像摘要', '材料清单', '产品预匹配'])
                with zt1:
                    st.markdown(f'**科创资质**：{info["rank_name"]}｜专精特新中小企业 {info["sme"]}｜小巨人 {info["giant"]}')
                    _extra = []
                    if info['valid_pat'] is not None: _extra.append(f'有效专利 {int(info["valid_pat"])} 件')
                    if info['tax'] != '未公开': _extra.append(f'纳税信用 {info["tax"]}')
                    if info['years'] is not None: _extra.append(f'成立 {int(info["years"])} 年')
                    if _extra:
                        st.markdown('**经营画像**：' + '｜'.join(_extra))
                    c1, c2, c3 = st.columns(3)
                    c1.metric('创新能力评估', info['proxy_display'], delta=info['coverage_tier'])
                    c2.metric('指标覆盖率', f"{info['coverage']}%", delta=f'覆盖{len(info["covered"])}/12项')
                    c3.metric('专利状态', info['pat_status'], delta=info['pat_desc'][:20])
                    st.markdown(f'**评估结论**：{info["proxy_conclusion"]}')
                with zt2:
                    st.markdown('**已覆盖指标**：' + ('、'.join(info['covered']) if info['covered'] else '暂无'))
                    st.markdown('**建议补充材料**：' + ('、'.join(info['missing']) if info['missing'] else '材料齐全'))
                    st.caption('补充材料后可提升指标覆盖率，进而获得更完整的创新能力评估。')
                with zt3:
                    for pm in product_prematch(info.get('proxy_score', 50) if info['coverage_tier'] == '可评分' else 40, info):
                        st.markdown(f'- **《{pm["产品"]}》** 匹配度：{pm["匹配度"]}')
                        if pm['已满足']:
                            st.markdown(f'  - 已满足：{"、".join(pm["已满足"])}')
                        st.markdown(f'  - 待核验：{"、".join(pm["待核验"])}')
                st.caption('注：珠海非上市企业财务数据不公开，本页基于资质+专利+覆盖率三维画像，完整授信需补充财务材料。')
                st.divider()

            if len(train_hits) == 0 and len(zhuhai_hits) > 0:
                st.info('该企业为珠海本地企业：财务数据不公开，已采用"资质+专利+覆盖率"三维画像评估。')

else:
    st.subheader('🏙️ 珠海科创企业总库浏览')
    col1, col2 = st.columns([1, 2])
    with col1:
        types = ['全部'] + sorted(set(str(v) if pd.notna(v) else '未分类' for v in zhuhai['企业类型']))
        sel_type = st.selectbox('按资质筛选', types)
        show_all = st.checkbox('仅显示有专利数据的企业', value=False)
    with col2:
        st.markdown(f'**总库规模**：{len(zhuhai)}家珠海科创企业')
        st.markdown('包含：高新技术企业、创新型中小企业、专精特新中小企业、专精特新小巨人、创新百强企业等。')
    view = zhuhai.copy()
    if sel_type != '全部':
        view = view[view['企业类型'].astype(str) == sel_type]
    if show_all:
        view = view[view['专利申请总量'].notna() & (view['专利申请总量'] > 0)]
    kw = st.text_input('输入关键字过滤企业（可空）', key='zh_kw')
    if kw:
        view = view[view['企业名称'].astype(str).str.contains(kw.strip(), na=False)]
    st.dataframe(view, use_container_width=True, height=400)
    st.caption(f'当前显示 {len(view)} 家企业')
    if 'zh_sel' not in st.session_state:
        st.session_state.zh_sel = ''
    show_names = view['企业名称'].astype(str).tolist()[:50]
    if show_names:
        if 'zh_expanded' not in st.session_state:
            st.session_state.zh_expanded = False
        _shown = show_names if st.session_state.zh_expanded else show_names[:10]
        st.markdown(f'**点击企业查看画像报告**（当前列表{len(show_names)}家' + ('，显示前10家' if not st.session_state.zh_expanded else '，全部展开') + '）：')
        _c = st.columns(3)
        for i, n in enumerate(_shown):
            if _c[i % 3].button(n, key=f'zh_{i}', use_container_width=True):
                st.session_state.zh_sel = n; st.rerun()
        if not st.session_state.zh_expanded and len(show_names) > 10:
            if st.button(f'▼ 展开全部 {len(show_names)} 家企业', key='zh_expand_btn'):
                st.session_state.zh_expanded = True; st.rerun()
        elif st.session_state.zh_expanded:
            if st.button('▲ 收起，只显示前10家', key='zh_collapse_btn'):
                st.session_state.zh_expanded = False; st.rerun()
    sel_name = st.session_state.zh_sel
    if sel_name and sel_name in view['企业名称'].astype(str).values:
        row = view[view['企业名称'] == sel_name].iloc[0]
        info = zhuhai_info(row)
        st.subheader(f'📄 {sel_name} 授信辅助报告')
        c1, c2, c3 = st.columns(3)
        c1.metric('创新能力评估', info['proxy_display'], delta=info['coverage_tier'])
        c2.metric('指标覆盖率', f"{info['coverage']}%")
        c3.metric('专利状态', info['pat_status'])
        st.markdown(f'**资质层级**：{info["rank_name"]}')
        _extra2 = []
        if info['valid_pat'] is not None: _extra2.append(f'有效专利 {int(info["valid_pat"])} 件')
        if info['tax'] != '未公开': _extra2.append(f'纳税信用 {info["tax"]}')
        if info['years'] is not None: _extra2.append(f'成立 {int(info["years"])} 年')
        if _extra2:
            st.markdown('**经营画像**：' + '｜'.join(_extra2))
        st.markdown(f'**评估结论**：{info["proxy_conclusion"]}')
        st.markdown('**已覆盖指标**：' + ('、'.join(info['covered']) if info['covered'] else '暂无'))
        st.markdown('**建议补充材料**：' + ('、'.join(info['missing']) if info['missing'] else '材料齐全'))

st.divider()
with st.expander('📋 数据来源与合规说明（点击展开）', expanded=False):
    st.markdown("""
| 数据类别 | 来源渠道 | 用途 |
| --- | --- | --- |
| 企业财务指标 | 东方财富/新浪财经/全国股转系统（公开年报） | 模型训练11项特征 |
| 专利数据 | 国家知识产权局专利检索系统（pss-system.cponline.cnipa.gov.cn） | 专利质量/创新能力评估 |
| 企业资质 | 珠海市科技创新局/工信局公示、高新技术企业认定名单 | 创新积分2.0覆盖率推断 |
| 经营/处罚信息 | 国家企业信用信息公示系统（gsxt.gov.cn） | 风险预警参考 |
| 纳税信用等级 | 税务部门A级纳税人公示名单 | 经营画像辅助 |

**合规声明**：本工具所有数据均来自**公开渠道**，不涉及工商银行内部客户数据；输出为模型参考值，不构成最终授信决策，实际审批以银行尽调为准。
""")

