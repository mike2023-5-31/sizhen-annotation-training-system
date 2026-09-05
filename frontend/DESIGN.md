# 四诊标注训练平台 — 前端设计系统（DESIGN.md）

> 本文件是本前端（Vue 3 + Element Plus）所有视觉决策的唯一权威来源。任何组件、页面、
> 样式在落码前必须先读本文件；所有颜色 / 字号 / 间距必须引用本文件 token，
> 禁止裸 hex、任意 px 与绕过系统的临时样式。
>
> **当前阶段**：本文件为「高级柔和风」设计基础（token + 应用外壳）。7 个功能视图
> （DatasetManage / Annotate / Train / Evaluate / Infer / Onnx）仍使用 Element Plus 默认样式，
> 将在后续任务中按本文件第 5 节逐页重构。
>
> 最后更新：2026-09-05

---

## 0. Research Log

- **Embedded refs**：短名单 [`supabase.md`（翠绿强调色，最接近青玉 hue）、`stripe.md`（紫渐变）、`linear.app.md`（紫，极简）]
  → 选定 **Layer A `soft-skill.md`** + **Layer B `supabase.md`**。理由：任务要求单一青玉绿强调色，supabase 是
  70 个 Layer B 中唯一以「绿」为品牌身份的系统（`#3ecf8e` 家族），其「HSL+alpha 半透明分层」「绿仅作身份标记、
  不作装饰」「胶囊 CTA（999px）vs 小圆角次级件」「字重克制（400/500 为主）」可直接迁移到浅色医疗工具；
  stripe/linear 均为紫色 hue，与青玉冲突。
- **补充参考**：`design/README.md`（路由与两个 Gate）、`redesign-skill.md`（audit-first 清单）、
  `design-system-architecture.md`（DESIGN.md 八段结构）、`_INDEX.md`（Layer B 短名单）。
- **Lazyweb**：跳过 — 现有代码库审计（Element Plus 默认风格、6 视图、横排 el-menu）即为真实产品屏幕来源。
- **Imagen drafts**：跳过 — 本任务是「既有应用的壳层重构」，非营销页；方向由 Layer A+B 合成锁定。
- **跳过的 lane**：lazyweb / imagen（见上）。

### 锁定方向（1-2 句）

一间安静的诊室：暖白宣纸质感的分层表面 + 高度弥散的青玉色调阴影，唯一的玻璃材质出现在悬浮导航胶囊上。
**签名**：顶部「浮岛式」玻璃导航胶囊 —— 青玉只出现在有语义的地方（当前页、主操作、焦点环），绝不装饰。

---

## 1. Atmosphere & Identity

**氛围**：安静、克制、专业。像一间光线柔和的诊室——表面之间用色阶与光晕分层，
而不是用硬边框吵架。数据密集时可以密，空白处必须透气。

**签名材质**：玻璃导航胶囊（浮岛）。这是全应用**唯一**允许使用 `backdrop-filter` 的元素
（sticky 定位，符合性能护栏）。玻璃配方是多层的，不是单层 blur（见第 7 节）。

**色彩故事**：全系统只有**一个**强调色——青玉绿（jade），中性色全部带轻微绿色暖调
（单一灰色家族，禁止冷灰暖灰混用）。青玉是「身份标记」：只用于主按钮、当前导航态、
焦点环、选中态、成功态，绝不出现在大面积背景上。

---

## 2. Color

### 2.1 中性色（暖纸白阶，全部带轻微青玉暖调）

| 角色 | Token | 值 | 用途 |
|------|-------|-----|------|
| Canvas | `--ts-bg-canvas` | `#f4f6f2` | 页面背景（暖纸白） |
| Surface/base | `--ts-surface-base` | `#ffffff` | 卡片、面板 |
| Surface/raised | `--ts-surface-raised` | `#fbfdfb` | 悬停抬升面 |
| Surface/sunken | `--ts-surface-sunken` | `#eef1eb` | 输入井、内凹背景 |
| Text/primary | `--ts-text-primary` | `#22302a` | 标题、正文（深青墨，非纯黑） |
| Text/secondary | `--ts-text-secondary` | `#59665f` | 次级说明 |
| Text/tertiary | `--ts-text-tertiary` | `#8d9a92` | 弱化、元信息 |
| Text/disabled | `--ts-text-disabled` | `#b7c0ba` | 禁用 |
| Border/default | `--ts-border-default` | `#dde3da` | 常规描边、分割线 |
| Border/subtle | `--ts-border-subtle` | `#eaeeda` | 软分隔 |
| Border/strong | `--ts-border-strong` | `#cdd6cb` | 交互态描边 |

### 2.2 青玉强调色（多档感知渐变，非同一 hex 调透明度）

| Token | 值 | 用途 |
|------|-----|------|
| `--ts-jade-50` | `#eef6f0` | 极淡底 |
| `--ts-jade-100` | `#dcf0e3` | 导航激活胶囊底、选中底 |
| `--ts-jade-200` | `#b8e0c7` | 浅描边 |
| `--ts-jade-300` | `#8ecba5` | 浅图标 |
| `--ts-jade-400` | `#5fb27f` | 次级强调 |
| `--ts-jade-500` | **`#3d9260`** | **主强调色（青玉）** |
| `--ts-jade-600` | `#2f7a4f` | hover |
| `--ts-jade-700` | `#256240` | active |
| `--ts-jade-800` | `#1d4d32` | 激活文字、深强调 |
| `--ts-jade-900` | `#163c27` | 最深（极少用） |
| `--ts-focus-ring` | `hsla(150, 42%, 41%, 0.35)` | 焦点环 |

### 2.3 语义色（全部降饱和，饱和 < 80%）

| 角色 | Token | 值 | 用途 |
|------|-------|-----|------|
| Success | `--ts-success` | `#3d9260`（同 jade-500） | 成功 = 青玉，同族 |
| Warning | `--ts-warning` | `#b07d2e` | 警告（哑琥珀） |
| Danger | `--ts-danger` | `#c25454` | 错误（哑红） |
| Info | `--ts-info` | `#5c7f96` | 信息（哑蓝灰） |

> **EP 光阶派生（Element Plus 变量映射）**：`--el-color-{success,warning,danger,error,info}-light-{3,5,7,8,9}`
> 与 `-dark-2` 为上述 4 个语义主色按 EP 惯例派生的淡/深阶（success 直接复用青玉 ramp 档位），
> 全部声明在 `theme.css`，视作语义色的合法派生，不另立新色。

### 2.4 大气层 token（仅壳层背景用，App.vue）

| Token | 值 | 用途 |
|------|-----|------|
| `--ts-atmo-jade` | `hsla(150, 45%, 60%, 0.10)` | 左上青玉径向光 |
| `--ts-atmo-mist` | `hsla(95, 35%, 65%, 0.08)` | 右上薄雾径向光 |
| `--ts-atmo-warm` | `hsla(40, 30%, 80%, 0.14)` | 底部暖光 |
| `--ts-hover-tint` | `hsla(150, 25%, 50%, 0.08)` | 通用悬停底色（导航/菜单项） |

### 2.5 规则

- 表面层级优先靠**色阶**与**色调阴影**制造深度，少用描边。
- 强调色**只**用于交互元素（主按钮 / 激活态 / 焦点 / 选中 / 成功），禁止装饰性使用。
- 禁止引入本表之外的颜色；确需新颜色 → 先扩展本表。
- 禁止在浅色页面中插入任何深色区块（redesign-skill 红线）。

---

## 3. Typography

### 3.1 字体栈（中文优先）

- **Primary**：`"PingFang SC", "HarmonyOS Sans SC", "Microsoft YaHei", "Noto Sans SC", system-ui, sans-serif`
  （Mac → PingFang；Windows → Microsoft YaHei；Linux → Noto Sans SC）
- **Mono（指标/数值）**：`"JetBrains Mono", "Cascadia Mono", Consolas, monospace`
- 数据表格数值一律 `font-variant-numeric: tabular-nums`。

### 3.2 字阶（数据工具密度，非营销密度）

| Level | Size | Weight | Line-height | Tracking | 用途 |
|-------|------|--------|-------------|----------|------|
| Display | 28px / 1.75rem | 600 | 1.25 | -0.01em | 页面标题 |
| H1 | 22px / 1.375rem | 600 | 1.3 | 0 | 区块标题 |
| H2 | 18px / 1.125rem | 600 | 1.35 | 0 | 小节标题 |
| H3 | 16px / 1rem | 600 | 1.4 | 0 | 卡片标题 |
| Body | 14px / 0.875rem | 400 | 1.6 | 0 | 默认正文（EP 基准） |
| Body/sm | 13px / 0.8125rem | 400 | 1.5 | 0 | 次级信息 |
| Caption | 12px / 0.75rem | 500 | 1.4 | 0.02em | 标签、元数据 |
| Overline | 11px / 0.6875rem | 600 | 1.3 | 0.08em | 小节眉标（大写/中文则字距放宽） |

### 3.3 规则

- 字重克制（源自 supabase）：正文 400，导航/按钮 500，标题 600。**不使用 700+**（中文字重粗体糊）。
- 正文不小于 14px；12–13px 仅用于 Caption / Body-sm。
- 多行标题用 `text-wrap: balance`，段落避免孤行用 `text-wrap: pretty`。
- 最多 2 个字体族（Primary + Mono），无第三族。

---

## 4. Spacing & Layout

### 4.1 基础单位：4px

| Token | 值 | 用途 |
|-------|-----|------|
| `--ts-space-1` | 4px | 图标与文字紧贴 |
| `--ts-space-2` | 8px | 列表项、行内组 |
| `--ts-space-3` | 12px | 表单域默认 padding |
| `--ts-space-4` | 16px | 标准内距 |
| `--ts-space-5` | 20px | 舒适内距 |
| `--ts-space-6` | 24px | 卡片默认内距、页面留白 |
| `--ts-space-8` | 32px | 卡片组间距 |
| `--ts-space-10` | 40px | 页内区块间距 |
| `--ts-space-12` | 48px | 大区块分隔 |
| `--ts-space-16` | 64px | 页级纵向节奏 |

### 4.2 Grid & 断点

- 内容最大宽度：**1440px**，居中（`.app-page` 容器）。
- 断点：sm 640 / md 768 / lg 1024 / xl 1280 / 2xl 1536。导航在 <1024 时允许横向滚动。
- 浏览器机制（`auto` / `%` / `clamp()` / `min-content` 等）保持原生，不强行 token 化。

---

## 5. Components

> 本阶段通过 **Element Plus CSS 变量覆写**（`theme.css`）实现基础件风格；7 个视图页尚未重构。
> 后续任务按本节状态表逐页落地。导航为自定义组件（见 5.7）。

### 5.1 Button（EP `el-button`）

- **变体**：primary（胶囊 999px，仅主 CTA）、default/plain（8px 圆角）、text、danger。
- **Primary**：底 `--ts-jade-500`，hover `--ts-jade-600`，active `--ts-jade-700`，文字 `#fff`。
- **状态**：default / hover（加深一档）/ active（`transform: scale(0.98)` + 再深一档）/
  focus-visible（`--ts-focus-ring` 焦点环）/ disabled（底 `--ts-jade-50`，文字 `--ts-text-disabled`）/
  loading（EP 内置 spinner，不换）。过渡 `--ts-motion-micro`。
- **A11y**：键盘可达（原生 button），焦点环可见。

### 5.2 Card（EP `el-card`）

- 圆角 `--ts-radius-md`（12px）；描边 `--ts-border-subtle`；阴影 `--ts-shadow-default`（青玉色调）。
- hover 抬升：`--ts-shadow-raised`，`transform: translateY(-1px)`（仅在有交互语义时）。
- 内距 `--ts-space-6`；头部标题 H3 字阶。

### 5.3 Input / Form（EP `el-input` / `el-select` / `el-form`）

- 圆角 `--ts-radius-sm`（8px）；底 `--ts-surface-base`；描边 `--ts-border-default`。
- hover 描边 `--ts-border-strong`；focus 描边 `--ts-jade-500` + 焦点环；error 描边 `--ts-danger`。
- 文本颜色走第 2.1 节 token；placeholder 用 `--ts-text-tertiary`。

### 5.4 Table（EP `el-table`）

- 表头底 `--ts-surface-sunken`，文字 `--ts-text-secondary` 500；行描边 `--ts-border-subtle`。
- 行 hover 底 `--ts-jade-50`；数值列 `tabular-nums`。
- 空态用 `el-empty`（重描色为 `--ts-text-tertiary`）。

### 5.5 Dialog / 浮层（EP `el-dialog` / `el-message` / popover）

- Dialog 圆角 `--ts-radius-lg`（16px），阴影 `--ts-shadow-prominent`，底 `--ts-surface-base`。
- Message：圆角 `--ts-radius-sm`，阴影 `--ts-shadow-default`；语义色用第 2.3 节。

### 5.6 Tag / Alert（EP `el-tag` / `el-alert`）

- Tag 圆角 `--ts-radius-xs`（6px，避免「药丸徽章」通用感）；语义色降饱和版（light-9 底 + dark 文字）。
- Alert 圆角 `--ts-radius-sm`；描边同语义色 30% alpha。

### 5.7 AppNav（自定义，`App.vue`）

- **结构**：`<header>`（sticky）> `<nav>`（玻璃胶囊）> 品牌 SVG + wordmark + 6 个 `RouterLink`。
- **玻璃配方（多层，非单 blur）**：见第 7.1 节。
- **状态**：
  - default：文字 `--ts-text-secondary` 500，透明底。
  - hover：底 `hsla(150, 25%, 50%, 0.08)`，文字 `--ts-text-primary`。
  - active：青玉胶囊（底 `--ts-jade-100`，文字 `--ts-jade-800` 600）。
  - focus-visible：`--ts-focus-ring`。active 态通过 `router-link-exact-active` 自动标记
    （`aria-current="page"`）。
- **Motion**：色/底过渡 `--ts-motion-micro`；激活胶囊出现 = `opacity` + `scale` 从 0.96 入（GPU 合成）。
- **布局**：`position: sticky; top: 16px`，胶囊居中 `max-width: fit-content`；<1024 横向滚动，隐藏滚动条。

---

## 6. Motion & Interaction

### 6.1 Timing

| 类型 | 时长 | 缓动 | 用途 |
|------|------|------|------|
| Micro | 120ms | `cubic-bezier(0, 0, 0.2, 1)` | 按钮按压、开关 |
| Standard | 220ms | `cubic-bezier(0.4, 0, 0.2, 1)` | 面板开合、tab 切换、导航 hover |
| Emphasis | 480ms | `cubic-bezier(0.16, 1, 0.3, 1)` | 激活胶囊入场、路由切换 |

Token：`--ts-motion-micro` / `--ts-motion-standard` / `--ts-motion-emphasis`。

### 6.2 规则

- **只动画 `transform` 与 `opacity`**（GPU 合成），禁止动画 `top/left/width/height`、禁止 `ease-in-out/linear`。
- 每个可交互元素必须有 hover + active + focus-visible 三态。
- `backdrop-filter` 仅用于 sticky 导航（软技能性能护栏）。
- `prefers-reduced-motion: reduce` → 关闭全部非必要动画与过渡。
- **禁止装饰性动画**（无信息目的的悬停微动、无意义入场）。
- 滚动触发动画用 `IntersectionObserver`，禁止 `scroll` 监听器（后续视图重构时遵守）。

---

## 7. Depth & Surface

### 7.0 策略：**Mixed**（色调阴影 + 发丝描边；玻璃仅导航）

### 7.1 玻璃配方（多层，来自 soft-skill「Doppelrand / glass」+ supabase 半透明分层）

```css
/* 导航胶囊 — 唯一玻璃表面 */
background: hsla(165, 24%, 99%, 0.78);              /* 暖白半透明底 */
backdrop-filter: blur(16px) saturate(180%);          /* 模糊 + 增色 */
border: 1px solid hsla(150, 22%, 45%, 0.16);         /* 青玉发丝描边（棱） */
box-shadow:
  inset 0 1px 0 hsla(0, 0%, 100%, 0.75),             /* 内高光（顶部折射） */
  0 8px 28px hsla(150, 28%, 20%, 0.10);              /* 青玉色调弥散阴影 */
```

### 7.2 阴影阶（全部色调化，禁纯黑阴影）

| Level | Token | 值 |
|-------|-------|-----|
| Subtle | `--ts-shadow-subtle` | `0 1px 2px hsla(150, 20%, 28%, 0.05), 0 1px 1px hsla(150, 20%, 28%, 0.04)` |
| Default | `--ts-shadow-default` | `0 2px 8px hsla(150, 22%, 24%, 0.06), 0 1px 2px hsla(150, 22%, 24%, 0.04)` |
| Raised | `--ts-shadow-raised` | `0 6px 16px hsla(150, 24%, 22%, 0.08), 0 2px 4px hsla(150, 24%, 22%, 0.05)` |
| Prominent | `--ts-shadow-prominent` | `0 12px 32px hsla(150, 26%, 20%, 0.12), 0 2px 8px hsla(150, 26%, 20%, 0.06)` |

### 7.3 圆角

| Token | 值 | 用途 |
|-------|-----|------|
| `--ts-radius-xs` | 6px | 内层小件、tag |
| `--ts-radius-sm` | 8px | 输入、次级按钮 |
| `--ts-radius-md` | 12px | 卡片 |
| `--ts-radius-lg` | 16px | Dialog、大容器 |
| `--ts-radius-pill` | 999px | 主 CTA、导航激活胶囊 |

规则：内层元素圆角 ≤ 外层容器圆角（同心圆曲率，Doppelrand）。

### 7.4 大气层（页面背景，App.vue）

- 固定、`pointer-events: none` 的径向渐变两处（左上青玉 0.10 alpha、右下暖中性 0.08–0.14 alpha），
  叠加 0.02 alpha 的 SVG 噪点纹理（fixed overlay，符合护栏）。z-index：大气 0 < 内容 1 < 噪点 2 < 导航 100 < EP 浮层（系统级）。

---

## 8. Accessibility Constraints & Accepted Debt

### 8.1 约束

- 目标：WCAG 2.2 AA。正文对比 ≥ 4.5:1，大字号 ≥ 3:1（`--ts-text-primary` on `--ts-surface-base` ≈ 12:1）。
- 所有交互元素 `:focus-visible` 可见焦点环（`--ts-focus-ring`），全键盘可达。
- `prefers-reduced-motion` 尊重（第 6 节）。
- 语义化标签：`<nav>` / `<main>` / `<header>`；图标 SVG 带 `aria-hidden`，不依赖颜色传意
  （激活态同时用字重 + `aria-current`）。

### 8.2 Accepted Debt

| 项 | 位置 | 接受原因 | 退出条件 |
|----|------|----------|----------|
| 无 webfont（CJK 全量字重 ≥2MB，离线环境不引 CDN） | 全应用 | 系统 CJK 栈（YaHei/PingFang/Noto）在目标平台渲染质量达标 | 若部署环境明确 → 引入子集化可变字重 |
| 无深色模式 | 全应用 | 医学标注以浅色影像为主，本期锁定浅色 | 用户提出 → 新增 dark token 组 |
| 7 视图仍为 EP 默认样式 | views/* | 本任务范围=基础+外壳，重构是后续任务 | 按第 5 节逐页落地 |
| 噪声纹理仅壳层启用（App.vue 大气层） | App.vue | 视图重构时统一迁移到全局层 | 视图重构任务 |
| 导航 <1024 横向滚动（不做汉堡折叠） | App.vue | 6 项 CJK 标签 + 桌面优先工具，避免过早移动端复杂度 | 移动端需求明确时 |
