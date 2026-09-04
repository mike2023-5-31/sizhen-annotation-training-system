# 四诊标注 XML Schema（T3）

> 全系统标注 XML 的唯一真相源：标注写入（T10）与预标注生成（T17）共用
> `backend/app/services/xml_io.py` 读写，标签体系见 `backend/config/labels.yaml`。
> 具体样例见 `docs/annotation-schema.sample.xml`。

## 结构

```xml
<annotation>
  <folder>dataset_name</folder>
  <filename>image.jpg</filename>
  <path>relative/path/image.jpg</path>
  <size><width>640</width><height>480</height><depth>3</depth></size>
  <quality>合格</quality>
  <labels>
    <label><category>舌</category><name>舌苔黄</name><confidence>1.0</confidence></label>
  </labels>
</annotation>
```

## 字段表

| 路径 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `annotation` | 元素 | 是 | 根元素，每张图片一个 XML |
| `folder` | 文本 | 否 | 数据集名称，缺省 `''` |
| `filename` | 文本 | 否 | 图片文件名（如 `image.jpg`） |
| `path` | 文本 | 否 | 图片相对路径（如 `relative/path/image.jpg`） |
| `size` | 元素 | 否 | 图片尺寸；缺失时可省略整个元素 |
| `size/width` | 整数 | 否 | 宽（像素） |
| `size/height` | 整数 | 否 | 高（像素） |
| `size/depth` | 整数 | 否 | 通道数（通常 3） |
| `quality` | 文本 | 是 | 质量标记，取值 ∈ {`合格`, `不合格`, `''`}；缺失按 `''` |
| `labels` | 元素 | 是 | 标签容器，可含 0–4 个 `<label>` |
| `labels/label` | 元素 | 否 | 单条标签，**每 category 至多一个** |
| `label/category` | 文本 | 是 | 四诊类别：舌 / 面 / 脉 / 声音 |
| `label/name` | 文本 | 是 | 子标签名（见 labels.yaml，每类首项为「正常/无」） |
| `label/confidence` | 浮点 | 否 | 人工标注 = `1.0`；预标注 = [0, 1] 概率；缺失默认 `1.0` |

## 规则

- **一图 0–4 个 label**：四类并列（舌/面/脉/声音），每 category 至多一个 `<label>`。
- **标签与质量以 XML 为准**：DB 仅镜像 `quality` 供快速过滤，不做标签双源。
- **无 bbox/分割语义**：本系统是分类系统，XML 不含 `bndbox`/掩码。
- **原子写**：`write_xml` 用「同目录临时文件 + `os.replace`」，崩溃/并发不产生半截文件。
- **确定性序列化**：`write_xml(path, read_xml(path))` 产物逐字节一致（UTF-8，中文原样保留）。
