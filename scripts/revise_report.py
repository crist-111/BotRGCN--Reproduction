from docx import Document
from pathlib import Path

src=Path(r'E:/xjtu/os/科研实习/botrgcn-reproduction/复现报告.docx')
out=Path(r'E:/xjtu/os/科研实习/BotRGCN_report_revised.docx')
d=Document(src)

def reset_table(t, rows):
    for row in list(t.rows):
        t._tbl.remove(row._tr)
    for row in rows:
        cells=t.add_row().cells
        for c,v in zip(cells,row):
            c.text=v

reset_table(d.tables[3], [
 ['设置','Accuracy','F1','MCC'],
 ['完整特征','0.8155 ± 0.0010','0.8528 ± 0.0012','0.6611 ± 0.0040'],
 ['去除 description','0.8163 ± 0.0022','0.8536 ± 0.0018','0.6633 ± 0.0054'],
 ['去除 tweets','0.8177 ± 0.0013','0.8544 ± 0.0006','0.6652 ± 0.0021'],
 ['去除数值属性','0.8157 ± 0.0009','0.8533 ± 0.0009','0.6627 ± 0.0033'],
 ['去除类别属性','0.6915 ± 0.0400','0.7549 ± 0.0142','0.3885 ± 0.0669'],
 ['仅保留属性','0.6205 ± 0.0688','0.6842 ± 0.0155','0.2137 ± 0.1851']])
reset_table(d.tables[4], [
 ['关系设置','Accuracy','F1','MCC'],
 ['all','0.8163 ± 0.0005','0.8536 ± 0.0005','0.6633 ± 0.0016'],
 ['following','0.7681 ± 0.0788','0.8169 ± 0.0605','0.5590 ± 0.1733'],
 ['follower','0.8163 ± 0.0022','0.8541 ± 0.0017','0.6654 ± 0.0049']])

for p in d.paragraphs:
    if 'F1 �� MCC' in p.text:
        p.text='修正后的主实验采用论文明确的二元交叉熵损失、单 logit 输出和 L2 正则，使用完整特征与两种关系。三次随机种子的平均 F1 为 0.8537 ± 0.0012，MCC 为 0.6633 ± 0.0042。'
    if 'following �� all' in p.text:
        p.text='修正后的关系消融显示，all 与 follower 的 F1 接近，而 following 的 seed=44 出现明显低值，导致方差较大。因此当前实验不能证明 all 一定优于单一关系，报告保留每次运行差异。'
    if 'follower ����ʹ��ʱ���������½�' in p.text:
        p.text='修正后的关系消融显示，all 与 follower 的平均 F1 接近，而 following 的 seed=44 出现明显低值，导致 following 方差较大。因此当前实验不能证明 all 一定优于单一关系，报告保留每次运行差异。'
    if 'Python3.3' in p.text:
        p.text=p.text.replace('Python3.3','Python 3.13')
    if 'BotRCGN' in p.text:
        p.text=p.text.replace('BotRCGN','BotRGCN')

d.add_heading('六 主实验结果与论文对照',1)
d.add_paragraph('本项目正式 BotRGCN 主实验使用 229,580 个节点、227,979 条边、17 维结构化属性、768 维 description 表示和 768 维 tweet 平均表示。模型按照论文结构将四类输入分别投影到 D/4 维，经两层 R-GCN 和 R-GCN 后 MLP 完成节点分类。')
t=d.add_table(rows=1, cols=4)
for c,v in zip(t.rows[0].cells,['指标','论文','本复现','差值']): c.text=v
for row in [['Accuracy','0.8462','0.8166 ± 0.0009','-0.0296'],['F1','0.8707','0.8537 ± 0.0012','-0.0170'],['MCC','0.7021','0.6633 ± 0.0042','-0.0388']]:
    cells=t.add_row().cells
    for c,v in zip(cells,row): c.text=v
d.add_paragraph('本复现结果低于论文报告值，但三次运行的 F1 标准差仅为 0.0012，说明当前实现具有稳定重复性。论文正文未完整公开隐藏维度、学习率、训练轮数、早停和 RoBERTa 处理细节，因此这些因素可能造成差异，不能在报告中声称已完全恢复作者训练配置。')

d.add_heading('七 基线实验结果',1)
d.add_paragraph('在相同多模态输入和 BCE 训练协议下，MLP、GCN、GAT 的三次种子结果如下。')
t=d.add_table(rows=1, cols=4)
for c,v in zip(t.rows[0].cells,['模型','Accuracy','F1','MCC']): c.text=v
for row in [['MLP','0.8169 ± 0.0005','0.8550 ± 0.0003','0.6688 ± 0.0010'],['GCN','0.5900 ± 0.0301','0.7139 ± 0.0092','0.1831 ± 0.0743'],['GAT','0.5424 ± 0.0024','0.7026 ± 0.0009','0.0205 ± 0.0356'],['BotRGCN','0.8166 ± 0.0009','0.8537 ± 0.0012','0.6633 ± 0.0042']]:
    cells=t.add_row().cells
    for c,v in zip(cells,row): c.text=v
d.add_paragraph('MLP 略高于 BotRGCN 是本次复现的实际结果，应如实保留。GCN 和 GAT 的结果明显较低，但这些基线是统一特征框架下的实现，不等同于作者未公开的原始基线代码。')

d.add_heading('八 我的收获',1)
for s in ['掌握了从原始 JSON 数据构建大规模异构用户图的方法，包括节点 ID 对齐、关系类型编码、边索引生成和 train/dev/test mask 构造。','理解了论文中的多模态用户表示：RoBERTa 文本语义特征、数值属性、类别属性分别编码后再融合。','掌握了 RoBERTa 全量文本编码的分块、断点续跑、特征合并和有限值检查，解决了 support 数据规模大、文本为空和 GPU 显存受限等工程问题。','理解了 R-GCN 如何使用 relation-specific 参数区分 following 和 follower，并完成了关系消融和基线对比。','建立了科研复现意识：区分论文明确内容、合理实现选择和自行设定参数；保留异常结果，不为了符合论文结论而修改实验数据。','通过主实验与论文结果对照，认识到复现不仅是跑通代码，还包括数据一致性、指标定义、随机种子稳定性和差异分析。']:
    d.add_paragraph('• '+s)
d.add_heading('九 是否需要改进',1)
d.add_paragraph('该报告已经能够作为复现项目阶段性成果提交。若希望进一步提升为更完整的论文复现，建议补充作者官方代码对照、至少五个随机种子、完整超参数搜索记录、训练曲线和错误案例分析；这些属于增强项，不影响当前报告对已完成工作的真实陈述。')
d.save(out)
print(out)
