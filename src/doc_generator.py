# src/doc_generator.py
import os
import re
from typing import List, Dict, Any, Tuple
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

class DocumentGenerator:
    """
    Level 3: 重型结构化文档与长篇报告自动生成引擎
    支持标准《技术规范书》与《可行性研究报告》一键结构化起草与 Word (.docx) 导出
    """
    def __init__(self, output_dir: str = 'generated_docs'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_tech_spec(self, title: str, requirements: str, references: List[Dict[str, Any]] = None) -> Tuple[str, str]:
        """
        生成完整的《企业级系统技术规范书 (Technical Specification)》
        """
        title = title.strip() or "核心系统高可用架构技术规范书"
        req_text = requirements.strip() or "支持高并发、多可用区容灾、多租户隔离与自动化故障转移。"

        md_content = f"""# {title}

## 1. 业务背景与系统目标
随着业务规模扩张，系统承载的企业级流量呈现高并发、跨地域分布特征。本规范旨在明确系统在架构设计、高可用容灾、网络分区容错以及敏捷交付方面的统一技术标准，保障 SLA 达到 99.99% 以上。

- **核心目标**: 消除单点故障，实现跨 AZ / 跨 Region 的自动化故障隔离与容灾接管；
- **业务诉求**: {req_text}
- **适用范围**: 覆盖敏捷研发团队、SRE 基础架构团队与自动化测试流水线。

## 2. 总体系统架构设计
系统采用多层次微服务与分布式集群治理模型，核心分为：接入流量网关层、分布式业务逻辑层、数据持久与状态共识层。

- **流量接入与调度**:
  - 全局智能 DNS 解析配合四层/七层双活负载均衡 (LB)，支持按权重引流与健康心跳探活；
  - 接入层集成分布式限流与熔断器 (基于令牌桶算法)，针对突发洪峰流量执行软驱逐策略。
- **计算集群自治**:
  - 计算节点容器化纳管，遵循无状态设计原则，支持 HPA 秒级弹性扩容；
  - 核心服务保障多可用区 (Multi-AZ) 均衡拓扑分布，单一 AZ 故障不影响全局吞吐能力。

## 3. 高可用容灾与数据一致性方案
本规范严格定义在各种物理与网络拓扑异常下的容灾兜底准则：

- **多可用区双活 (Active-Active)**:
  - 数据库与消息中间件采用多数派共识协议 (Raft/Paxos) 跨 AZ 部署，租约超时与选举机制严格对齐跨机房网络延迟上限；
  - 网络分区发生时，主动触发隔离防护策略，避免出现脑裂 (Split-Brain) 并自动切流至健康机房。
- **数据备份与 RPO/RTO 关键指标**:
  - **RPO (恢复点目标)**: 核心交易数据 RPO ≤ 0（零数据丢失），非核心业务 RPO < 1 分钟；
  - **RTO (恢复时间目标)**: AZ 级自动倒换 RTO < 30 秒，Region 级跨地域灾备切换 RTO < 15 分钟；
  - 定期执行全量快照与连续 binlog/wal 日志异地冷备，并具备自动化恢复演练验证能力。

## 4. 接口规范、安全与权限治理
- **API 通信标准**: 统一采用 RESTful / gRPC 协议，请求头强制携带链路追踪标识 `Trace-Id` 与企业签名 Token；
- **访问控制与鉴权**:
  - 严格遵循最小权限原则 (PoLP) 与基于角色的访问控制 (RBAC)；
  - 跨系统调用强制执行双向 mTLS 加密传输，敏感凭据由统一 Vault 密钥管理器动态派发，严禁明文落地；
- **发布与防重**: 关键变更接口必须支持全局唯一幂等键 (Idempotency-Key)。

## 5. 质量保障、测试验证与上线 Checklist
在正式推进生产上线前，研发团队与 SRE 必须联合签署以下检查清单：

- [x] **单元与集成测试**: 核心业务代码测试覆盖率不低于 85%，高危边界条件覆盖率 100%；
- [x] **CI/CD 自动化流水线**: 代码合入必须通过至少两位高级架构师 Review 批准，且自动化回归无挂起失败；
- [x] **故障演练 (Chaos Engineering)**: 在 Staging 环境模拟模拟高延迟与 AZ 断网注入，验证服务自我愈合能力；
- [x] **监控报警大盘**: 关键 SLA 指标覆盖率 100%，包括 P99 响应延迟、CPU/内存水位、错误率报警及回滚阈值。
"""
        file_path = self.export_to_docx(
            title=title,
            doc_type="企业级技术规范书 (Tech Spec)",
            markdown_content=md_content,
            references=references
        )
        return file_path, md_content

    def generate_feasibility_report(self, title: str, requirements: str, references: List[Dict[str, Any]] = None) -> Tuple[str, str]:
        """
        生成完整的《重大项目技术可行性研究报告 (Feasibility Study Report)》
        """
        title = title.strip() or "重大业务系统升级技术可行性研究报告"
        req_text = requirements.strip() or "评估系统在云原生容器化重构、算力成本优化与生产稳定性方面的实施可行性。"

        md_content = f"""# {title}

## 1. 项目立项背景与必要性分析
当前企业内部多个核心服务随着数据吞吐量增长，面临老旧单体架构维护难度大、版本发布周期长、跨团队协作摩擦多等挑战。为支持未来 3 年业务高速发展，启动本次技术重构升级方案的可行性评估。

- **核心评估诉求**: {req_text}
- **必要性结论**: 现有系统在当前架构下的扩容边际成本呈指数级上升，推进现代化微服务化与容器化重构是保障企业业务连续性与交付敏捷度的必然路径。

## 2. 方案对比与技术选型论证
报告对业界主流的三种实施技术路线进行了横向打分与风险评估：

| 评估维度 | 方案 A：原地微调与配置加固 | 方案 B：云原生容器化与解耦重构 (推荐) | 方案 C：彻底重写全新底座 |
| :--- | :--- | :--- | :--- |
| **改造成本** | 低（预计 2 周） | 中等（预计 6-8 周） | 极高（预计 6 个月以上） |
| **技术债务消除** | 无法根除，后续仍有瓶颈 | 显著重构模块，彻底解耦核心状态 | 彻底消除，但引入极高交付风险 |
| **业务平滑过渡** | 业务无感，但天花板明显 | 支持多阶段灰度切流、双写对账 | 割接风险巨大，容易产生跨系统断层 |
| **综合可行性结论** | 仅适合短期过渡 | **综合 ROI 最高，风险完全可控 (采纳)** | 投入产出比过低，暂不予采纳 |

## 3. 跨系统依赖与集成风险评估
- **上游数据源协同**: 需同 Jira、Confluence、GitHub CI 等平台打通实时 Webhook 状态联动；
- **跨团队排期冲突**: 涉及基础架构科、敏捷研发科与测试科的多方资源统筹，建议设立联合专项小组；
- **回滚与兜底预案**: 制定精细的双写灰度与一键切回方案，在 Staging 环境模拟切流演练成功后方可投产。

## 4. 实施阶段计划与里程碑 (Milestones)
- **阶段一 (M1 - 第 1~2 周)**: 细化详细设计方案，完成核心 POC 原型验证与压测验证；
- **阶段二 (M2 - 第 3~5 周)**: 完成服务拆分、数据模型迁移工具开发与单元测试覆盖；
- **阶段三 (M3 - 第 6~7 周)**: 灰度部署于预发环境，开展为期两周的真实业务影子流量对比验证；
- **阶段四 (M4 - 第 8 周)**: 生产窗口分批次放量，建立 7x24 小时实时监控指挥台，项目正式交付投产。

## 5. 综合可行性结论与专家审批意见
经多科室联合会诊与综合评定，本项目在技术成熟度、团队交付能力、成本收益比 (ROI) 及系统安全性方面均具备充分的可行性，方案成熟完备，建议准予正式立项推进。
"""
        file_path = self.export_to_docx(
            title=title,
            doc_type="重大项目可行性研究报告 (Feasibility Report)",
            markdown_content=md_content,
            references=references
        )
        return file_path, md_content

    def export_to_docx(self, title: str, doc_type: str, markdown_content: str, references: List[Dict[str, Any]] = None) -> str:
        doc = Document()
        for section in doc.sections:
            section.top_margin = Inches(1.0)
            section.bottom_margin = Inches(1.0)
            section.left_margin = Inches(1.0)
            section.right_margin = Inches(1.0)

        p_title = doc.add_paragraph()
        p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_title = p_title.add_run(title)
        r_title.font.name = 'Microsoft YaHei'
        r_title.font.size = Pt(22)
        r_title.font.bold = True
        r_title.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)

        p_sub = doc.add_paragraph()
        p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_sub = p_sub.add_run(f'[{doc_type}]  |  企业智能知识库多科室联合会诊中心自动输出')
        r_sub.font.name = 'Microsoft YaHei'
        r_sub.font.size = Pt(11)
        r_sub.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

        doc.add_paragraph()

        lines = markdown_content.splitlines()
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if s.startswith('# '):
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(12)
                p.paragraph_format.space_after = Pt(4)
                r = p.add_run(s[2:])
                r.font.name = 'Microsoft YaHei'
                r.font.size = Pt(16)
                r.font.bold = True
                r.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
            elif s.startswith('## '):
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(10)
                p.paragraph_format.space_after = Pt(3)
                r = p.add_run(s[3:])
                r.font.name = 'Microsoft YaHei'
                r.font.size = Pt(13)
                r.font.bold = True
                r.font.color.rgb = RGBColor(0x25, 0x63, 0xEB)
            elif s.startswith('### '):
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(6)
                p.paragraph_format.space_after = Pt(2)
                r = p.add_run(s[4:])
                r.font.name = 'Microsoft YaHei'
                r.font.size = Pt(11.5)
                r.font.bold = True
                r.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
            elif s.startswith('- ') or s.startswith('* '):
                p = doc.add_paragraph(style='List Bullet')
                p.paragraph_format.space_after = Pt(2)
                r = p.add_run(s[2:])
                r.font.name = 'Microsoft YaHei'
                r.font.size = Pt(10.5)
                r.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
            else:
                p = doc.add_paragraph()
                p.paragraph_format.space_after = Pt(4)
                p.paragraph_format.line_spacing = 1.25
                r = p.add_run(s)
                r.font.name = 'Microsoft YaHei'
                r.font.size = Pt(10.5)
                r.font.color.rgb = RGBColor(0x33, 0x41, 0x55)

        if references:
            doc.add_paragraph()
            p_ref = doc.add_paragraph()
            p_ref.paragraph_format.space_before = Pt(14)
            r_ref = p_ref.add_run('附录：多源知识库支撑事实与溯源')
            r_ref.font.name = 'Microsoft YaHei'
            r_ref.font.size = Pt(14)
            r_ref.font.bold = True
            r_ref.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

            for idx, ref in enumerate(references, 1):
                p_item = doc.add_paragraph()
                p_item.paragraph_format.space_after = Pt(2)
                src = ref.get('source_type', 'DOC').upper()
                tit = ref.get('title', '无标题')
                uid = ref.get('uuid', 'N/A')
                r_item = p_item.add_run(f'[{idx}] [{src}] {tit} (ID: {uid})')
                r_item.font.name = 'Microsoft YaHei'
                r_item.font.size = Pt(9.5)
                r_item.font.bold = True
                r_item.font.color.rgb = RGBColor(0x25, 0x63, 0xEB)

                p_snip = doc.add_paragraph()
                p_snip.paragraph_format.left_indent = Inches(0.25)
                p_snip.paragraph_format.space_after = Pt(4)
                r_snip = p_snip.add_run(ref.get('text', '')[:200] + '...')
                r_snip.font.name = 'Microsoft YaHei'
                r_snip.font.size = Pt(9)
                r_snip.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

        clean = re.sub(r'[^a-zA-Z0-9_一-龥]', '_', title)[:30] or 'document'
        path = os.path.join(self.output_dir, f'{clean}.docx')
        doc.save(path)
        return path
