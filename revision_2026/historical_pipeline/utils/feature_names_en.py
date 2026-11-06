"""
Chinese → English feature name translation dictionary.

Usage:
    from utils.feature_names_en import FEAT_EN, en
    english_name = en('年龄')  # → 'Age'
"""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
FEAT_EN = {'年龄': 'Age', '载脂蛋白B': 'Apolipoprotein B', '低密度脂蛋白胆固醇': 'LDL Cholesterol', '总胆固醇': 'Total Cholesterol', '胆碱酯酶': 'Cholinesterase', '白蛋白': 'Albumin', '总胆汁酸': 'Total Bile Acids', '前白蛋白': 'Prealbumin', 'D-二聚体': 'D-Dimer', '细胞角蛋白19片段': 'CYFRA 21-1', '胱抑素C': 'Cystatin C', '甘油三脂': 'Triglycerides', '腺苷脱氨酶': 'Adenosine Deaminase', '球蛋白': 'Globulin', '血小板压积': 'Plateletcrit', '鳞状细胞癌相关抗原': 'SCC Antigen', '直接胆红素': 'Direct Bilirubin', '医生数(人)': 'No. of Physicians', '淀粉酶': 'Amylase', '医院、卫生院床位数(张)': 'Hospital Beds', '葡萄糖': 'Glucose', '性别': 'Sex', '高密度脂蛋白胆固醇': 'HDL Cholesterol', '镁': 'Magnesium', 'α-羟丁酸脱氢酶': 'α-HBDH', '丙氨酸氨基转移酶': 'ALT', '人均地区生产总值(元)': 'GDP per Capita', '医院、卫生院数(个)': 'No. of Hospitals', '乙肝核心抗体': 'HBcAb', '乳酸脱氢酶': 'LDH', '磷': 'Phosphorus', '天门冬氨酸氨基转移酶': 'AST', '尿酸': 'Uric Acid', '甲胎蛋白': 'Alpha-Fetoprotein', '总蛋白': 'Total Protein', '糖化血红蛋白': 'HbA1c', '肌酐': 'Creatinine', '肌酸激酶': 'Creatine Kinase', '职工平均工资(元)': 'Avg. Employee Wage', '钾': 'Potassium', '钠': 'Sodium', '载脂蛋白AⅠ': 'Apolipoprotein AI', '铁蛋白': 'Ferritin'}

def en(name: str) -> str:
    """Translate a Chinese feature name to English, or return as-is if not found."""
    return FEAT_EN.get(name, name)
