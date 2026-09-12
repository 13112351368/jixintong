# -*- coding: utf-8 -*-
"""
积信通 - 科技企业可解释授信辅助引擎（Streamlit原型）
运行: streamlit run app.py
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
# ---------- 中文字体保障（本地Windows有中文字体则跳过；云部署优先读取随仓库部署的simhei.ttf，失败再在线下载） ----------
def _ensure_cn_font():
    try:
        _names = {f.name for f in _fm.fontManager.ttflist}
        if any(k in n for n in _names for k in ('YaHei', 'SimHei', 'SimSun', 'Noto Sans CJK', 'WenQuanYi', 'Source Han')):
            return
        _base = os.path.dirname(os.path.abspath(__file__))
        _font_path = None
        # 1) 优先使用随仓库部署的 simhei.ttf（不依赖外部网络，最稳）
        _local = os.path.join(_base, 'simhei.ttf')
        if os.path.exists(_local) and os.path.getsize(_local) > 1000000:
            _font_path = _local
        else:
            # 2) 在线下载文泉驿正黑（备用）
            import urllib.request
            _dest = os.path.join(_base, 'wqy-zenhei.ttc')
            if not os.path.exists(_dest) or os.path.getsize(_dest) < 1000000:
                for _url in (
                    'https://cdn.jsdelivr.net/gh/anthonyfok/fonts-wqy-zenhei@master/wqy-zenhei.ttc',
                    'https://raw.githubusercontent.com/anthonyfok/fonts-wqy-zenhei/master/wqy-zenhei.ttc',
                ):
                    try:
                        urllib.request.urlretrieve(_url, _dest)
                        if os.path.getsize(_dest) > 1000000:
                            break
                    except Exception:
                        continue
            if os.path.exists(_dest) and os.path.getsize(_dest) > 1000000:
                _font_path = _dest
        if _font_path:
            _fm.fontManager.addfont(_font_path)
            # 3) 清除matplotlib字体缓存，确保新字体立即生效
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
        print('中文字体加载失败(仅影响图内中文):', _e)
_ensure_cn_font()
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei', 'Microsoft YaHei', 'SimSun', 'Noto Sans CJK SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
import streamlit as st
import shap
import plotly.graph_objects as go

# ---------- 路径配置 ----------
BASE = os.path.dirname(os.path.abspath(__file__))
# 优先使用 data/ 子目录（云部署结构），否则用上级目录（本地结构）
if os.path.exists(os.path.join(BASE, 'data')):
    DATA = os.path.join(BASE, 'data')
else:
    DATA = os.path.dirname(BASE)
MODEL = os.path.join(BASE, 'models')

st.set_page_config(page_title='积信通·科创授信辅助引擎', page_icon='🏦', layout='wide')

# ---------- 商务蓝主题 CSS（高对比度版） ----------
st.markdown("""
<style>
:root {
  --biz-blue: #1F4E79;
  --biz-blue-mid: #2E75B6;
  --biz-blue-light: #EAF2FA;
}
.stApp { background: #FFFFFF; }
/* 全局文字 */
body, p, span, div, label { color: #1A2B3C !important; }
h1, h2, h3 { color: #1F4E79; font-weight: 700; }
.stCaption, small { color: #3A4A5C !important; font-size: 13px !important; }
div[data-testid="stMarkdownContainer"] p { color: #1A2B3C; font-size: 15px; line-height: 1.7; }
div[data-testid="stMarkdownContainer"] strong { color: #1F4E79; }
/* metric卡片 */
div[data-testid="stMetric"] {
  background: linear-gradient(180deg, #F4F9FE 0%, #FFFFFF 100%);
  border: 2px solid #1F4E79;
  border-radius: 12px;
  padding: 14px 16px;
}
div[data-testid="stMetric"] label { color: #1A2B3C !important; font-size: 14px !important; font-weight: 600; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #1F4E79 !important; font-weight: 800 !important; font-size: 28px !important; }
div[data-testid="stMetric"] div[data-testid="stMetricDelta"] { color: #3A4A5C !important; font-size: 12px !important; }
/* 按钮 */
div.stButton > button {
  border-radius: 8px;
  border: 2px solid #1F4E79;
  background: #FFFFFF;
  color: #1F4E79 !important;
  font-weight: 700;
  font-size: 14px;
}
div.stButton > button:hover {
  background: #1F4E79 !important;
  color: #FFFFFF !important;
}
/* 侧边栏：淡蓝马卡龙 */
section[data-testid="stSidebar"] {
  background: #EAF2FA;
  border-right: 2px solid #B4CCE3;
}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #1F4E79; }
section[data-testid="stSidebar"] .stRadio label { color: #1A2B3C; }
section[data-testid="stSidebar"] .stMarkdownContainer p { color: #1A2B3C; }
/* expander */
div[data-testid="stExpander"] {
  border: 2px solid #1F4E79;
  border-radius: 10px;
  background: #FFFFFF;
}
/* input框 */
input { font-size: 16px !important; }
/* radio/checkbox */
div[role="radiogroup"] label, div[role="checkbox"] label { color: #1A2B3C !important; font-size: 14px; }
/* 顶部条 */
header[data-testid="stHeader"] { background: #FFFFFF !important; }
.stApp > header { background: #FFFFFF !important; }
/* 输入框 */
input, textarea, .stTextInput input, .stTextArea textarea {
  background-color: #FFFFFF !important;
  color: #1A2B3C !important;
  border: 2px solid #1F4E79 !important;
  font-size: 16px !important;
}
/* radio选中点 */
div[role="radiogroup"] label div:first-child { background-color: #1F4E79 !important; }
div[data-testid="stSidebar"] div[role="radiogroup"] label { color: #1A2B3C; }
section.main, div.main { background: #FFFFFF !important; }
/* 链接 */
a { color: #2E75B6 !important; }
div[role="radiogroup"] label svg, div[role="checkbox"] label svg { color: #1F4E79 !important; }
.stSelectbox > div > div { border: 2px solid #1F4E79 !important; }
.stSelectbox > div > div:hover { border-color: #2E75B6 !important; }
/* 滚动条 */
::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-track { background: #F4F9FE; }
::-webkit-scrollbar-thumb { background: #1F4E79; border-radius: 5px; }
/* divider */
hr { border-color: #B4CCE3; }
/* title/caption 不要代码块样式 */
div[data-testid="stTitle"], div[data-testid="stTitle"] h1 {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}
div[data-testid="stCaption"] {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}
/* radio选项：选中项完全透明，跟侧边栏融为一体，只加粗变色 */
section[data-testid="stSidebar"] div[data-testid="stRadio"] label,
section[data-testid="stSidebar"] div[data-testid="stRadio"] label > div,
section[data-testid="stSidebar"] div[data-testid="stRadio"] label > div > div,
section[data-testid="stSidebar"] div[data-testid="stRadio"] label > div > div > div {
  background: transparent !important;
  background-color: transparent !important;
  box-shadow: none !important;
  border: none !important;
  outline: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) > div > div,
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) > div > div > div {
  color: #1F4E79 !important;
  font-weight: 700;
}
</style>
""", unsafe_allow_html=True)



# ---------- 加载资源（缓存） ----------
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
    # 合并专利数据到珠海总库
    if '专利申请总量' not in zhuhai.columns:
        zhuhai = zhuhai.merge(patents[['企业名称', '专利申请总量']], on='企业名称', how='left')
    return train, zhuhai, patents

@st.cache_resource
def load_explainers():
    lr, xgb_m, scaler, meta = load_models()
    try:
        explainer_xgb = shap.TreeExplainer(xgb_m)
        # Logistic解释器：需要标准化后的背景数据
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

# ---------- 工具函数 ----------
def to_score(p_risk):
    """风险概率 → 0-100信用评分（概率越低分越高）"""
    return round((1 - p_risk) * 100, 1)

def risk_level(score):
    if score >= 80:
        return 'A（低风险）'
    elif score >= 60:
        return 'B（中低风险）'
    elif score >= 40:
        return 'C（中风险）'
    else:
        return 'D（高风险）'

def amount_suggestion(score):
    if score >= 80:
        return '800-1000万元', '建议正常授信，可适用利率优惠'
    elif score >= 60:
        return '300-600万元', '建议授信，可补充1-2项材料后提额'
    elif score >= 40:
        return '100-300万元', '建议降额授信，增加担保或引入知识产权质押'
    else:
        return '暂缓授信', '提示具体风险点，补充材料后可重新评估'

def product_match(score, zhuhai_info=None):
    """工行产品智能匹配"""
    products = []
    if score >= 60:
        products.append(('科创贷', '信用贷款，匹配度' + ('高' if score >= 75 else '中')))
    if zhuhai_info and zhuhai_info.get('专利分级') in ('高', '中'):
        products.append(('知识产权质押贷', '以专利质押增信，匹配度高'))
    if zhuhai_info and zhuhai_info.get('专精特新'):
        products.append(('人才贷', '核心团队稳定，可评估人才维度授信'))
    if not products:
        products.append(('科创贷', '需补充材料后评估匹配度'))
    return products

def coverage_analysis(zhuhai_row):
    """
    基于企业可观测资质推断创新积分2.0指标覆盖率（12项：9定量+3加分）
    返回: (覆盖率, 已覆盖项, 缺失项)
    """
    covered = []
    typ = str(zhuhai_row.get('企业类型', ''))
    sme = zhuhai_row.get('专精特新中小企业', '')
    giant = zhuhai_row.get('专精特新小巨人', '')
    pat = zhuhai_row.get('专利申请总量', 0)
    pat_lvl = zhuhai_row.get('专利数量分级', '')

    # 可依据公开信息推断覆盖的指标
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
        covered.append('入选高成长企业（加分项）')
        covered.append('入选创新百强（加分项）')
    if '独角兽' in typ or '瞪羚' in typ:
        covered.append('获得省级以上科技奖励（加分项）')

    # 去重
    covered = list(dict.fromkeys(covered))
    total = 12
    ratio = round(len(covered) / total * 100, 1)

    missing = [
        '技术合同成交额', '营收增长率', '资产负债率',
        '重点研发计划参与情况', '发明专利质量'
    ]
    missing = [m for m in missing if not any(m in c for c in covered)]
    return ratio, covered, missing[:5]

# 特征显示单位（SHAP图标注用）
_FEATURE_UNITS = {
    '营业总收入': '万元',
    '销售毛利率': '%',
    '研发费用': '万元',
    '营收增长率': '%',
    '研发人员数': '人',
    '员工人数': '人',
    '资产负债率': '%',
    '研发费用占比': '%',
    '营收对数': 'ln(万元)',
    '研发费用对数': 'ln(万元)',
    '研发人员占比': '%',
}

def build_report(feature_row, show_shap=True):
    """训练集企业：双模型评分+SHAP"""
    global medians
    X = pd.DataFrame([feature_row], columns=FEATURES)[FEATURES]
    X = X.fillna(medians)

    X_s = scaler.transform(X)
    p_lr = lr.predict_proba(X_s)[:, 1][0]
    p_xgb = xgb_m.predict_proba(X)[:, 1][0]
    p_fusion = 0.4 * p_lr + 0.6 * p_xgb
    score = to_score(p_fusion)
    level = risk_level(score)
    amount, advice = amount_suggestion(score)

    shap_fig = None
    if show_shap and explainer_xgb and explainer_lr:
        try:
            sv_xgb = explainer_xgb.shap_values(X)
            sv_lr = explainer_lr.shap_values(X_s)
            sv_fusion = 0.4 * np.array(sv_lr) + 0.6 * np.array(sv_xgb)
            sv_fusion = sv_fusion.reshape(-1)
            # 瀑布图
            base = float(explainer_xgb.expected_value)
            fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(FEATURES))))
            order = np.argsort(-np.abs(sv_fusion))[:8]
            labels = [FEATURES[i] + (f"({_FEATURE_UNITS[FEATURES[i]]})" if FEATURES[i] in _FEATURE_UNITS else "") for i in order][::-1]
            vals = [sv_fusion[i] for i in order][::-1]
            colors = ['#E5533C' if v > 0 else '#3C9AE5' for v in vals]
            ax.barh(labels, vals, color=colors)
            ax.axvline(0, color='#999', lw=0.8)
            ax.set_xlabel('SHAP值（评分点数贡献）')
            ax.set_title(f'综合评分特征贡献分解（基线 {base:.1f} 分）', fontsize=12)
            ax.tick_params(labelsize=10)
            plt.tight_layout()
            shap_fig = fig
        except Exception as e:
            shap_fig = None

    return {
        'p_lr': p_lr, 'p_xgb': p_xgb, 'p_fusion': p_fusion,
        'score': score, 'level': level, 'amount': amount, 'advice': advice,
        'shap_fig': shap_fig,
    }

def zhuhai_info(row):
    """珠海企业：三维画像"""
    typ = str(row.get('企业类型', ''))
    sme = '是' if pd.notna(row.get('专精特新中小企业')) and row.get('专精特新中小企业') else '否'
    giant = '是' if pd.notna(row.get('专精特新小巨人')) and row.get('专精特新小巨人') else '否'
    pat = row.get('专利申请总量', 0)
    pat_lvl = row.get('专利数量分级', '')
    if pd.isna(pat):
        pat, pat_lvl = 0, '无数据'
    ratio, covered, missing = coverage_analysis(row)

    # 资质层级（高企→创新型→专精特新→小巨人）
    rank = 0
    if '高新技术企业' in typ:
        rank = max(rank, 1)
    if '创新型' in typ:
        rank = max(rank, 2)
    if sme == '是':
        rank = max(rank, 3)
    if giant == '是':
        rank = max(rank, 4)
    rank_names = {0: '未明确科创资质', 1: '高新技术企业', 2: '创新型中小企业', 3: '专精特新中小企业', 4: '专精特新小巨人'}
    rank_name = rank_names[rank]

    # 专利分级
    try:
        pat_n = float(pat)
        if pat_n >= 100:
            pat_grade = '高'
        elif pat_n >= 20:
            pat_grade = '中'
        elif pat_n > 0:
            pat_grade = '低'
        else:
            pat_grade = '无'
    except (ValueError, TypeError):
        pat_grade = '无'

    # 代理创新能力评分（覆盖率为主，专利/资质加成）
    base_score = ratio
    if rank >= 3:
        base_score += 5
    if pat_grade == '高':
        base_score += 5
    elif pat_grade == '中':
        base_score += 3
    proxy_score = round(min(base_score, 90), 1)

    # 新字段：有效专利 / 纳税信用 / 成立年限
    try:
        valid_pat = float(row.get('有效专利数'))
        if pd.isna(valid_pat):
            valid_pat = None
    except (ValueError, TypeError):
        valid_pat = None
    tax = str(row.get('纳税信用等级', '')).strip()
    if tax in ('', 'nan') or '未公开' in tax:
        tax = '未公开'
    try:
        years = row.get('成立年限')
        if pd.isna(years):
            years = None
    except Exception:
        years = None

    return {
        'rank_name': rank_name, 'rank': rank,
        'sme': sme, 'giant': giant,
        'pat': pat_n if isinstance(pat, (int, float)) else 0,
        'pat_lvl': pat_lvl, 'pat_grade': pat_grade,
        'coverage': ratio, 'covered': covered, 'missing': missing,
        'proxy_score': proxy_score,
        'valid_pat': valid_pat, 'tax': tax, 'years': years,
    }

# ---------- 界面 ----------
st.title('积信通 · 科技企业可解释授信辅助引擎')
st.caption('基于创新积分2.0指标体系的科创授信决策辅助工具（工行杯参赛原型）')

# ---------- 可查询范围 ----------
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
    '① **A股/新三板企业**（3559家）输出完整模型报告（信用评分、风险等级、授信额度、SHAP特征解释、工行产品匹配）；'
    '② **珠海本地企业**（1594家）财务数据不公开，输出"资质＋专利＋覆盖率"三维画像与材料补充建议。'
)
st.markdown(
    '**覆盖名单**：高新技术企业、创新型中小企业、专精特新中小企业、专精特新小巨人、创新百强、科技型中小企业等8类。'
    '**示例查询**：立讯精密、深科技、珠海格力大金机电、格力钛新能源。'
)
st.divider()

with st.sidebar:
    st.header('⚙️ 查询设置')
    mode = st.radio('查询模式', ['企业名称查询', '珠海企业浏览'], index=0)
    st.divider()
    st.markdown(f'**模型状态**\n\n- 训练样本：{meta["n_samples"]}家\n- 融合AUC：{meta["auc_fusion"]:.4f}\n- 准确率：{meta["accuracy"]*100:.1f}%')

if mode == '企业名称查询':
    if 'sel_company' not in st.session_state:
        st.session_state.sel_company = ''
    # 预设演示案例（答辩一键展示）
    st.markdown('**🎯 演示案例（答辩直接点击）**')
    _dc1, _dc2, _dc3 = st.columns(3)
    if _dc1.button('🌟 案例A：立讯精密（A股龙头·高分报告）', width='stretch'):
        st.session_state.sel_company = '立讯精密'
        st.rerun()
    if _dc2.button('📊 案例B：深科技（A股·风险维度展示）', width='stretch'):
        st.session_state.sel_company = '深科技'
        st.rerun()
    if _dc3.button('🏙️ 案例C：格力大金机电（珠海本地三维画像）', width='stretch'):
        st.session_state.sel_company = '格力大金'
        st.rerun()
    q = st.text_input('请输入企业名称（支持模糊匹配）', value=st.session_state.sel_company,
                      placeholder='如：立讯精密 或 格力')
    q = q.strip()
    # 自动联想：输入时实时弹出匹配企业建议，点击即查询
    if q:
        _all_names = pd.concat([train['企业名称'], zhuhai['企业名称']]).astype(str).str.strip()
        _sug = _all_names[_all_names.str.contains(q, na=False)].drop_duplicates().head(12).tolist()
        if _sug:
            st.caption(f'🔍 匹配到 {len(_sug)} 家企业，点击直接查询：')
            _cols = st.columns(3)
            for i, _s in enumerate(_sug):
                if _cols[i % 3].button(_s, key=f'sug_{i}', width='stretch'):
                    st.session_state.sel_company = _s
                    st.rerun()
        else:
            st.warning(f'未找到与「{q}」匹配的企业。可试：立讯精密、深科技、格力。')
    if q:
        # 1. 训练集匹配
        train_hits = train[train['企业名称'].str.contains(q, na=False)]
        # 2. 珠海总库匹配
        zhuhai_hits = zhuhai[zhuhai['企业名称'].str.contains(q, na=False)]

        if len(train_hits) == 0 and len(zhuhai_hits) == 0:
            st.warning(f'未找到与「{q}」匹配的企业。可尝试：立讯精密、深科技、格力，或输入珠海企业全称。')
        else:
            # 展示训练集企业（完整模型报告）
            for _, row in train_hits.head(3).iterrows():
                name = row['企业名称']
                st.subheader(f'📋 {name}（{row.get("数据来源", "")}）')
                rep = build_report(row)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric('综合信用评分', rep['score'], delta='0-100分')
                c2.metric('风险等级', rep['level'])
                c3.metric('建议额度', rep['amount'])
                c4.metric('高风险概率', f"{rep['p_fusion']*100:.1f}%")
                # 风险仪表盘（Plotly gauge）
                _gcolor = '#2E9E5B' if rep['score'] >= 80 else ('#8BC8EA' if rep['score'] >= 60 else ('#FAAD14' if rep['score'] >= 40 else '#EA6668'))
                _gauge = go.Figure(go.Indicator(
                    mode='gauge+number',
                    value=rep['score'],
                    number={'font': {'size': 36, 'color': _gcolor}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#888'},
                        'bar': {'color': _gcolor, 'thickness': 0.3},
                        'bgcolor': 'white',
                        'steps': [
                            {'range': [0, 40], 'color': '#FBE3E3'},
                            {'range': [40, 60], 'color': '#FEF3CD'},
                            {'range': [60, 80], 'color': '#DCEFFB'},
                            {'range': [80, 100], 'color': '#D9F2E0'},
                        ],
                        'threshold': {'line': {'color': '#333', 'width': 2}, 'thickness': 0.75, 'value': rep['score']},
                    },
                    title={'text': '信用评分仪表盘', 'font': {'size': 13}},
                ))
                _gauge.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10))
                st.plotly_chart(_gauge, width='stretch')
                st.markdown(f'**授信建议**：{rep["advice"]}')
                st.markdown(f'**模型构成**：Logistic {rep["p_lr"]*100:.1f}%风险 × 0.4 ＋ XGBoost {rep["p_xgb"]*100:.1f}%风险 × 0.6')
                if rep['shap_fig']:
                    st.pyplot(rep['shap_fig'])
                prods = product_match(rep['score'])
                st.markdown('**工行产品匹配**：' + '；'.join(f'《{p[0]}》{p[1]}' for p in prods))
                st.divider()

            # 展示珠海企业（三维画像）
            for _, row in zhuhai_hits.head(3).iterrows():
                name = row['企业名称']
                info = zhuhai_info(row)
                st.subheader(f'🏙️ {name}（珠海本地企业）')
                st.markdown(f'**科创画像**：{info["rank_name"]}｜专精特新中小企业 {info["sme"]}｜小巨人 {info["giant"]}')
                _extra = []
                if info['valid_pat'] is not None:
                    _extra.append(f'有效专利 {int(info["valid_pat"])} 件')
                if info['tax'] and info['tax'] != '未公开':
                    _extra.append(f'纳税信用 {info["tax"]}')
                if info['years'] is not None:
                    _extra.append(f'成立 {int(info["years"])} 年')
                if _extra:
                    st.markdown('**经营画像**：' + '｜'.join(_extra))
                c1, c2, c3 = st.columns(3)
                c1.metric('代理创新能力评分', info['proxy_score'])
                c2.metric('指标覆盖率', f"{info['coverage']}%", delta=f'覆盖{len(info["covered"])}/12项')
                c3.metric('专利总量', int(info['pat']), delta=f'分级：{info["pat_grade"]}')
                st.markdown('**已覆盖指标**：' + '、'.join(info['covered']))
                st.markdown('**建议补充材料**：' + '、'.join(info['missing']) if info['missing'] else '材料齐全')
                prods = product_match(info['proxy_score'], {'专利分级': info['pat_grade'], '专精特新': info['sme'] == '是'})
                st.markdown('**工行产品匹配**：' + '；'.join(f'《{p[0]}》{p[1]}' for p in prods))
                st.caption('注：珠海非上市企业财务数据不公开，评分基于资质+专利+覆盖率三维画像，完整授信需补充财务材料。')
                st.divider()

            if len(train_hits) == 0 and len(zhuhai_hits) > 0:
                st.info('该企业为珠海本地企业：财务数据不公开，已采用"资质+专利+覆盖率"三维画像评估，符合银行真实审批场景。')

else:
    # 珠海企业浏览模式
    st.subheader('🏙️ 珠海科创企业总库浏览')
    col1, col2 = st.columns([1, 2])
    with col1:
        # 按资质筛选
        types = ['全部'] + sorted(zhuhai['企业类型'].astype(str).unique().tolist())
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

    # 关键字过滤（替代大下拉，避免上千选项导致卡顿）
    kw = st.text_input('输入关键字过滤企业（可空）', key='zh_kw')
    if kw:
        view = view[view['企业名称'].astype(str).str.contains(kw.strip(), na=False)]

    st.dataframe(view, width='stretch', height=400)
    st.caption(f'当前显示 {len(view)} 家企业')

    # 企业选择：显示前50家可点击按钮（替代大selectbox，性能稳定）
    if 'zh_sel' not in st.session_state:
        st.session_state.zh_sel = ''
    show_names = view['企业名称'].astype(str).tolist()[:50]
    if show_names:
        st.markdown('**点击企业查看画像报告**（当前列表前50家）：')
        _c = st.columns(3)
        for i, n in enumerate(show_names):
            if _c[i % 3].button(n, key=f'zh_{i}', width='stretch'):
                st.session_state.zh_sel = n
                st.rerun()
    sel_name = st.session_state.zh_sel
    if sel_name and sel_name in view['企业名称'].astype(str).values:
        row = view[view['企业名称'] == sel_name].iloc[0]
        info = zhuhai_info(row)
        st.subheader(f'📄 {sel_name} 授信辅助报告')
        c1, c2, c3 = st.columns(3)
        c1.metric('代理创新能力评分', info['proxy_score'])
        c2.metric('指标覆盖率', f"{info['coverage']}%")
        c3.metric('专利总量', int(info['pat']))
        st.markdown(f'**资质层级**：{info["rank_name"]}')
        _extra2 = []
        if info['valid_pat'] is not None:
            _extra2.append(f'有效专利 {int(info["valid_pat"])} 件')
        if info['tax'] and info['tax'] != '未公开':
            _extra2.append(f'纳税信用 {info["tax"]}')
        if info['years'] is not None:
            _extra2.append(f'成立 {int(info["years"])} 年')
        if _extra2:
            st.markdown('**经营画像**：' + '｜'.join(_extra2))
        st.markdown('**已覆盖指标**：' + '、'.join(info['covered']))
        st.markdown('**建议补充材料**：' + ('、'.join(info['missing']) if info['missing'] else '材料齐全'))
        st.markdown('**授信结论**：' + ('建议纳入创新积分贷初步评估，补充材料后完整评估。' if info['proxy_score'] >= 50 else '建议补充财务及研发材料后再评估，当前材料完整度不足。'))

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
