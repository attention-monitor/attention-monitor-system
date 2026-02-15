const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
        ShadingType, PageNumber, PageBreak, LevelFormat } = require('docx');

// ============ 配色 ============
const BLUE = "2B579A";
const LIGHT_BLUE = "D6E4F0";
const LIGHT_GRAY = "F2F2F2";
const DARK_GRAY = "333333";
const MEDIUM_GRAY = "666666";
const ACCENT = "4472C4";

// ============ 辅助函数 ============
const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };

function headerCell(text, width) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: BLUE, type: ShadingType.CLEAR },
    margins: cellMargins,
    verticalAlign: "center",
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text, bold: true, font: "Arial", size: 20, color: "FFFFFF" })] })]
  });
}

function bodyCell(text, width, opts = {}) {
  const runs = [];
  if (opts.bold) {
    runs.push(new TextRun({ text, bold: true, font: "Arial", size: 19, color: opts.color || DARK_GRAY }));
  } else {
    runs.push(new TextRun({ text, font: "Arial", size: 19, color: opts.color || DARK_GRAY }));
  }
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: opts.shading ? { fill: opts.shading, type: ShadingType.CLEAR } : undefined,
    margins: cellMargins,
    children: [new Paragraph({ children: runs })]
  });
}

function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 200 },
    children: [new TextRun({ text, bold: true, font: "Arial", size: 32, color: BLUE })]
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 160 },
    children: [new TextRun({ text, bold: true, font: "Arial", size: 26, color: ACCENT })]
  });
}

function heading3(text) {
  return new Paragraph({
    spacing: { before: 200, after: 120 },
    children: [new TextRun({ text, bold: true, font: "Arial", size: 22, color: DARK_GRAY })]
  });
}

function para(text, opts = {}) {
  return new Paragraph({
    spacing: { before: opts.spaceBefore || 80, after: opts.spaceAfter || 80, line: 320 },
    children: [new TextRun({ text, font: "Arial", size: 20, color: opts.color || DARK_GRAY, bold: opts.bold || false })]
  });
}

function richPara(runs, opts = {}) {
  return new Paragraph({
    spacing: { before: opts.spaceBefore || 80, after: opts.spaceAfter || 80, line: 320 },
    children: runs.map(r => new TextRun({ font: "Arial", size: 20, color: DARK_GRAY, ...r }))
  });
}

function bulletItem(text, ref = "bullets") {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    spacing: { before: 40, after: 40, line: 300 },
    children: [new TextRun({ text, font: "Arial", size: 20, color: DARK_GRAY })]
  });
}

function codeBlock(text) {
  return new Paragraph({
    spacing: { before: 60, after: 60 },
    indent: { left: 360 },
    children: [new TextRun({ text, font: "Courier New", size: 18, color: "2D2D2D" })]
  });
}

function divider() {
  return new Paragraph({
    spacing: { before: 200, after: 200 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" } },
    children: [new TextRun({ text: "", size: 4 })]
  });
}

function tipBox(title, text) {
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [9360],
    rows: [new TableRow({
      children: [new TableCell({
        borders: { top: { style: BorderStyle.SINGLE, size: 6, color: ACCENT }, bottom: border, left: border, right: border },
        width: { size: 9360, type: WidthType.DXA },
        shading: { fill: "EBF0FA", type: ShadingType.CLEAR },
        margins: { top: 100, bottom: 100, left: 160, right: 160 },
        children: [
          new Paragraph({ children: [new TextRun({ text: title, bold: true, font: "Arial", size: 20, color: ACCENT })] }),
          new Paragraph({ spacing: { before: 60 }, children: [new TextRun({ text, font: "Arial", size: 19, color: DARK_GRAY })] })
        ]
      })]
    })]
  });
}

// ============ 文档内容 ============
const children = [];

// === 封面 ===
children.push(new Paragraph({ spacing: { before: 2400 }, children: [] }));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 200 },
  children: [new TextRun({ text: "ADHD \u6CE8\u610F\u529B\u76D1\u6D4B\u7CFB\u7EDF", bold: true, font: "Arial", size: 48, color: BLUE })]
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 120 },
  children: [new TextRun({ text: "\u4EE3\u7801\u5BFC\u89C8\u624B\u518C", bold: true, font: "Arial", size: 36, color: ACCENT })]
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 600 },
  children: [new TextRun({ text: "\u2014\u2014 \u5E2E\u4F60\u81EA\u4FE1\u5730\u8BB2\u6E05\u695A\u6BCF\u4E00\u4E2A\u6A21\u5757", font: "Arial", size: 24, color: MEDIUM_GRAY })]
}));

children.push(new Table({
  width: { size: 5000, type: WidthType.DXA },
  columnWidths: [2000, 3000],
  rows: [
    new TableRow({ children: [
      new TableCell({ borders: { top: border, bottom: border, left: border, right: border }, width: { size: 2000, type: WidthType.DXA }, margins: cellMargins, children: [new Paragraph({ children: [new TextRun({ text: "\u6587\u4EF6", font: "Arial", size: 20, color: MEDIUM_GRAY })] })] }),
      new TableCell({ borders: { top: border, bottom: border, left: border, right: border }, width: { size: 3000, type: WidthType.DXA }, margins: cellMargins, children: [new Paragraph({ children: [new TextRun({ text: "ui.py (\u7EA6 5050 \u884C)", font: "Arial", size: 20, color: DARK_GRAY })] })] })
    ]}),
    new TableRow({ children: [
      new TableCell({ borders: { top: border, bottom: border, left: border, right: border }, width: { size: 2000, type: WidthType.DXA }, margins: cellMargins, children: [new Paragraph({ children: [new TextRun({ text: "\u7248\u672C", font: "Arial", size: 20, color: MEDIUM_GRAY })] })] }),
      new TableCell({ borders: { top: border, bottom: border, left: border, right: border }, width: { size: 3000, type: WidthType.DXA }, margins: cellMargins, children: [new Paragraph({ children: [new TextRun({ text: "v4.0", font: "Arial", size: 20, color: DARK_GRAY })] })] })
    ]}),
    new TableRow({ children: [
      new TableCell({ borders: { top: border, bottom: border, left: border, right: border }, width: { size: 2000, type: WidthType.DXA }, margins: cellMargins, children: [new Paragraph({ children: [new TextRun({ text: "\u6838\u5FC3\u7C7B", font: "Arial", size: 20, color: MEDIUM_GRAY })] })] }),
      new TableCell({ borders: { top: border, bottom: border, left: border, right: border }, width: { size: 3000, type: WidthType.DXA }, margins: cellMargins, children: [new Paragraph({ children: [new TextRun({ text: "9 \u4E2A", font: "Arial", size: 20, color: DARK_GRAY })] })] })
    ]})
  ]
}));

children.push(new Paragraph({ children: [new PageBreak()] }));

// === 第一章：系统全景 ===
children.push(heading1("\u7B2C\u4E00\u7AE0  \u7CFB\u7EDF\u5168\u666F\u2014\u201430\u79D2\u8BF4\u6E05\u695A\u4F60\u7684\u7CFB\u7EDF"));

children.push(para("\u5982\u679C\u522B\u4EBA\u95EE\u4F60\u201C\u4F60\u7684\u7CFB\u7EDF\u662F\u505A\u4EC0\u4E48\u7684\u201D\uFF0C\u4F60\u53EF\u4EE5\u8FD9\u6837\u56DE\u7B54\uFF1A"));
children.push(tipBox("\u4E00\u53E5\u8BDD\u7248\u672C", "\u8FD9\u662F\u4E00\u4E2A\u57FA\u4E8E\u6444\u50CF\u5934\u7684\u5B9E\u65F6\u6CE8\u610F\u529B\u76D1\u6D4B\u7CFB\u7EDF\uFF0C\u4E13\u95E8\u9488\u5BF9 ADHD \u513F\u7AE5\u5728\u7EBF\u5B66\u4E60\u573A\u666F\u8BBE\u8BA1\u3002\u5B83\u901A\u8FC7\u5206\u6790\u5B66\u751F\u7684\u773C\u775B\u72B6\u6001\u3001\u5934\u90E8\u59FF\u6001\u3001\u89C6\u7EBF\u65B9\u5411\u548C\u9762\u90E8\u60C5\u7EEA\uFF0C\u8BA1\u7B97\u51FA\u4E00\u4E2A 0-100 \u5206\u7684\u6CE8\u610F\u529B\u5F97\u5206\uFF0C\u5E76\u5728\u5B66\u751F\u8D70\u795E\u65F6\u7528\u8BED\u97F3\u63D0\u9192\u3002"));

children.push(para(""));
children.push(heading2("1.1 \u6570\u636E\u6D41\u7BA1\u9053\uFF08\u4F60\u5FC5\u987B\u80FD\u753B\u51FA\u6765\u7684\u56FE\uFF09"));
children.push(para("\u6574\u4E2A\u7CFB\u7EDF\u7684\u6570\u636E\u6D41\u52A8\u53EF\u4EE5\u7528\u4E00\u5F20\u56FE\u6982\u62EC\uFF1A", { spaceBefore: 120 }));

// 数据流表格
children.push(new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [1800, 1200, 2200, 1200, 2960],
  rows: [
    new TableRow({ children: [
      headerCell("\u8F93\u5165", 1800),
      headerCell("\u2192", 1200),
      headerCell("\u5904\u7406\u5F15\u64CE", 2200),
      headerCell("\u2192", 1200),
      headerCell("\u8F93\u51FA", 2960),
    ]}),
    new TableRow({ children: [
      bodyCell("\u6444\u50CF\u5934\u753B\u9762\n(640\u00D7480)", 1800),
      bodyCell("\u6BCF\u4E00\u5E27", 1200, { color: MEDIUM_GRAY }),
      bodyCell("AttentionAnalyzer\n(\u773C\u775B/\u5934\u90E8/\u89C6\u7EBF)", 2200, { bold: true }),
      bodyCell("\u72B6\u6001\u6570\u636E", 1200, { color: MEDIUM_GRAY }),
      bodyCell("OptimizedAttention\nScoringSystem\n\u2192 0-100\u5206", 2960, { bold: true }),
    ]}),
    new TableRow({ children: [
      bodyCell("", 1800),
      bodyCell("\u6BCF\u4E00\u5E27", 1200, { color: MEDIUM_GRAY }),
      bodyCell("EmotionAnalyzer\n(7\u7C7B\u60C5\u7EEA\u8BC6\u522B)", 2200, { bold: true }),
      bodyCell("\u60C5\u7EEA\u7F6E\u4FE1\u5EA6", 1200, { color: MEDIUM_GRAY }),
      bodyCell("\u60C5\u7EEA\u8C03\u6574\u9879\n\u2192 \u52A0\u51CF\u5206", 2960, { bold: true }),
    ]}),
    new TableRow({ children: [
      bodyCell("", 1800),
      bodyCell("", 1200),
      bodyCell("", 2200),
      bodyCell("\u6700\u7EC8\u5206\u6570", 1200, { color: MEDIUM_GRAY }),
      bodyCell("VoiceReminderSystem\n\u2192 \u8BED\u97F3\u63D0\u9192\nRealTimeCharts\n\u2192 \u5B9E\u65F6\u56FE\u8868", 2960, { bold: true }),
    ]})
  ]
}));

children.push(para(""));
children.push(heading2("1.2 \u4E5D\u4E2A\u7C7B\u7684\u5168\u666F\u5730\u56FE"));
children.push(para("\u4EE5\u4E0B\u662F ui.py \u4E2D\u6240\u6709\u7C7B\u7684\u529F\u80FD\u4E00\u89C8\u8868\u3002\u5EFA\u8BAE\u4F60\u5148\u8BB0\u4F4F\u524D 4 \u4E2A\u6838\u5FC3\u7C7B\uFF0C\u5176\u4F59\u7684\u4E86\u89E3\u5373\u53EF\u3002", { spaceBefore: 120 }));

children.push(new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [600, 2600, 3360, 1200, 1600],
  rows: [
    new TableRow({ children: [
      headerCell("#", 600),
      headerCell("\u7C7B\u540D", 2600),
      headerCell("\u529F\u80FD\u4E00\u53E5\u8BDD\u8BF4\u6E05\u695A", 3360),
      headerCell("\u884C\u53F7", 1200),
      headerCell("\u91CD\u8981\u5EA6", 1600),
    ]}),
    new TableRow({ children: [
      bodyCell("1", 600, { shading: LIGHT_BLUE }), bodyCell("AttentionAnalyzer", 2600, { bold: true, shading: LIGHT_BLUE }),
      bodyCell("\u7528 MediaPipe \u68C0\u6D4B\u773C\u775B/\u5934\u90E8/\u89C6\u7EBF\uFF0C\u5224\u65AD\u201C\u4E13\u6CE8\u201D\u6216\u201C\u8D70\u795E\u201D", 3360, { shading: LIGHT_BLUE }),
      bodyCell("92", 1200, { shading: LIGHT_BLUE }), bodyCell("\u2B50\u2B50\u2B50 \u6838\u5FC3", 1600, { shading: LIGHT_BLUE }),
    ]}),
    new TableRow({ children: [
      bodyCell("2", 600), bodyCell("EmotionAnalyzer", 2600, { bold: true }),
      bodyCell("\u7528\u6DF1\u5EA6\u5B66\u4E60\u6A21\u578B\u8BC6\u522B 7 \u7C7B\u60C5\u7EEA", 3360),
      bodyCell("422", 1200), bodyCell("\u2B50\u2B50\u2B50 \u6838\u5FC3", 1600),
    ]}),
    new TableRow({ children: [
      bodyCell("3", 600, { shading: LIGHT_GRAY }), bodyCell("OptimizedAttention\nScoringSystem", 2600, { bold: true, shading: LIGHT_GRAY }),
      bodyCell("\u628A\u591A\u7EF4\u6570\u636E\u878D\u5408\u6210 0-100 \u5206\u7684\u6CE8\u610F\u529B\u5F97\u5206", 3360, { shading: LIGHT_GRAY }),
      bodyCell("3387", 1200, { shading: LIGHT_GRAY }), bodyCell("\u2B50\u2B50\u2B50 \u6838\u5FC3", 1600, { shading: LIGHT_GRAY }),
    ]}),
    new TableRow({ children: [
      bodyCell("4", 600), bodyCell("VoiceReminderSystem", 2600, { bold: true }),
      bodyCell("\u5B66\u751F\u8D70\u795E\u65F6\u7528\u8BED\u97F3\u63D0\u9192\uFF08pyttsx3\uFF09", 3360),
      bodyCell("674", 1200), bodyCell("\u2B50\u2B50 \u91CD\u8981", 1600),
    ]}),
    new TableRow({ children: [
      bodyCell("5", 600, { shading: LIGHT_GRAY }), bodyCell("ADHDDetectionSystem", 2600, { bold: true, shading: LIGHT_GRAY }),
      bodyCell("\u4E3B\u754C\u9762\uFF08PyQt5\uFF09\uFF0C\u7EC4\u88C5\u6240\u6709\u6A21\u5757", 3360, { shading: LIGHT_GRAY }),
      bodyCell("803", 1200, { shading: LIGHT_GRAY }), bodyCell("\u2B50\u2B50 \u91CD\u8981", 1600, { shading: LIGHT_GRAY }),
    ]}),
    new TableRow({ children: [
      bodyCell("6", 600), bodyCell("FacialModeling", 2600, { bold: true }),
      bodyCell("\u9762\u90E8\u5EFA\u6A21\uFF0C\u8BB0\u5F55\u7528\u6237\u4E2A\u4F53\u5316\u7279\u5F81\u57FA\u7EBF", 3360),
      bodyCell("3184", 1200), bodyCell("\u2B50 \u8F85\u52A9", 1600),
    ]}),
    new TableRow({ children: [
      bodyCell("7", 600, { shading: LIGHT_GRAY }), bodyCell("CalibrationSystem", 2600, { bold: true, shading: LIGHT_GRAY }),
      bodyCell("5 \u70B9\u6821\u51C6\u6D41\u7A0B\uFF0C\u8BA9\u7CFB\u7EDF\u9002\u914D\u4E0D\u540C\u7528\u6237", 3360, { shading: LIGHT_GRAY }),
      bodyCell("4346", 1200, { shading: LIGHT_GRAY }), bodyCell("\u2B50 \u8F85\u52A9", 1600, { shading: LIGHT_GRAY }),
    ]}),
    new TableRow({ children: [
      bodyCell("8", 600), bodyCell("RealTimeCharts", 2600, { bold: true }),
      bodyCell("\u7528 PyQt5 \u7ED8\u5236\u5B9E\u65F6\u56FE\u8868\uFF08\u66F2\u7EBF\u56FE\u3001\u96F7\u8FBE\u56FE\uFF09", 3360),
      bodyCell("4584", 1200), bodyCell("\u2B50 \u8F85\u52A9", 1600),
    ]}),
    new TableRow({ children: [
      bodyCell("9", 600, { shading: LIGHT_GRAY }), bodyCell("AttentionConfig", 2600, { bold: true, shading: LIGHT_GRAY }),
      bodyCell("\u5B58\u50A8\u9608\u503C\u53C2\u6570\uFF08EAR\u3001\u504F\u8F6C\u89D2\u5EA6\u7B49\uFF09", 3360, { shading: LIGHT_GRAY }),
      bodyCell("50", 1200, { shading: LIGHT_GRAY }), bodyCell("\u914D\u7F6E", 1600, { shading: LIGHT_GRAY }),
    ]})
  ]
}));

children.push(new Paragraph({ children: [new PageBreak()] }));

// === 第二章：三个模型文件 ===
children.push(heading1("\u7B2C\u4E8C\u7AE0  \u4E09\u4E2A\u6A21\u578B\u6587\u4EF6\u2014\u2014\u522B\u4EBA\u95EE\u8D77\u4F60\u5F97\u77E5\u9053"));

children.push(para("\u4F60\u7684 Models/ \u6587\u4EF6\u5939\u91CC\u6709\u4E09\u4E2A\u6587\u4EF6\uFF0C\u6BCF\u4E2A\u90FD\u6709\u660E\u786E\u7684\u7528\u9014\uFF1A", { spaceBefore: 120 }));

children.push(new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [2800, 1000, 2400, 3160],
  rows: [
    new TableRow({ children: [
      headerCell("\u6587\u4EF6\u540D", 2800),
      headerCell("\u5927\u5C0F", 1000),
      headerCell("\u7528\u9014", 2400),
      headerCell("\u8C03\u7528\u4F4D\u7F6E", 3160),
    ]}),
    new TableRow({ children: [
      bodyCell("video.h5", 2800, { bold: true }),
      bodyCell("16 MB", 1000),
      bodyCell("\u60C5\u7EEA\u8BC6\u522B\u6A21\u578B\uFF08Xception \u7F51\u7EDC\uFF09\uFF0C\u8F93\u51FA 7 \u7C7B\u60C5\u7EEA\u6982\u7387", 2400),
      bodyCell("EmotionAnalyzer.__init__()\n\u7B2C 456 \u884C\nload_model('Models/\nEmotionXCeption/video.h5')", 3160),
    ]}),
    new TableRow({ children: [
      bodyCell("haarcascade_frontalface\n_default.xml", 2800, { bold: true, shading: LIGHT_GRAY }),
      bodyCell("0.9 MB", 1000, { shading: LIGHT_GRAY }),
      bodyCell("OpenCV \u4EBA\u8138\u68C0\u6D4B\u7EA7\u8054\u5206\u7C7B\u5668\uFF0C\u5B9A\u4F4D\u753B\u9762\u4E2D\u7684\u4EBA\u8138\u533A\u57DF", 2400, { shading: LIGHT_GRAY }),
      bodyCell("(\u901A\u8FC7 dlib \u95F4\u63A5\u4F7F\u7528\uFF0C\u7528\u4E8E\u60C5\u7EEA\u5206\u6790\u7684\u4EBA\u8138\u68C0\u6D4B)", 3160, { shading: LIGHT_GRAY }),
    ]}),
    new TableRow({ children: [
      bodyCell("face_landmarks.dat", 2800, { bold: true }),
      bodyCell("99 MB", 1000),
      bodyCell("dlib \u7684 68 \u70B9\u9762\u90E8\u7279\u5F81\u70B9\u5B9A\u4F4D\u6A21\u578B", 2400),
      bodyCell("EmotionAnalyzer.__init__()\n\u7B2C 465 \u884C\ndlib.shape_predictor(\n\"Models/Landmarks/\nface_landmarks.dat\")", 3160),
    ]})
  ]
}));

children.push(para(""));
children.push(tipBox("\u9762\u8BD5/\u6F14\u793A\u65F6\u7684\u5173\u952E\u8BF4\u6CD5", "\u201C\u6CE8\u610F\u529B\u68C0\u6D4B\u7528\u7684\u662F MediaPipe FaceMesh\uFF08478\u4E2A\u9762\u90E8\u7279\u5F81\u70B9\uFF09\uFF0C\u60C5\u7EEA\u8BC6\u522B\u7528\u7684\u662F\u6539\u8FDB\u7684 Xception \u6DF1\u5EA6\u5B66\u4E60\u7F51\u7EDC\u3002\u4E24\u8005\u72EC\u7ACB\u8FD0\u884C\uFF0C\u6700\u540E\u5728\u8BC4\u5206\u5C42\u878D\u5408\u3002\u201D"));

children.push(new Paragraph({ children: [new PageBreak()] }));

// === 第三章：注意力检测管道 ===
children.push(heading1("\u7B2C\u4E09\u7AE0  \u6CE8\u610F\u529B\u68C0\u6D4B\u7BA1\u9053\u8BE6\u89E3"));

children.push(heading2("3.1 AttentionAnalyzer \u2014\u2014 \u201C\u770B\u201D\u5B66\u751F\u7684\u773C\u775B"));
children.push(para("\u8FD9\u4E2A\u7C7B\u8D1F\u8D23\u4ECE\u6BCF\u4E00\u5E27\u753B\u9762\u4E2D\u63D0\u53D6\u56DB\u7EC4\u5173\u952E\u6570\u636E\uFF1A", { spaceBefore: 120 }));

children.push(new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [1800, 3000, 2200, 2360],
  rows: [
    new TableRow({ children: [
      headerCell("\u68C0\u6D4B\u7EF4\u5EA6", 1800),
      headerCell("\u5B9E\u73B0\u65B9\u6CD5", 3000),
      headerCell("\u5173\u952E\u51FD\u6570", 2200),
      headerCell("\u8F93\u51FA\u503C", 2360),
    ]}),
    new TableRow({ children: [
      bodyCell("\u773C\u775B\u72B6\u6001\n(EAR)", 1800, { bold: true }),
      bodyCell("\u8BA1\u7B97\u773C\u775B\u7EB5\u6A2A\u6BD4\nEAR = (A+B) / (2C)\n6\u4E2A\u7279\u5F81\u70B9\u7684\u8DDD\u79BB\u6BD4\u503C", 3000),
      bodyCell("eye_aspect_ratio()\n\u7B2C 134 \u884C", 2200),
      bodyCell("0-0.4 \u7684\u6D6E\u70B9\u6570\n< 0.21 = \u95ED\u773C", 2360),
    ]}),
    new TableRow({ children: [
      bodyCell("\u5934\u90E8\u59FF\u6001\n(pitch/yaw/roll)", 1800, { bold: true, shading: LIGHT_GRAY }),
      bodyCell("\u7528 solvePnP \u7B97\u6CD5\n\u5C06 2D \u4EBA\u8138\u70B9\u6620\u5C04\u5230 3D \u7A7A\u95F4\n\u6C42\u89E3\u6B27\u62C9\u89D2", 3000, { shading: LIGHT_GRAY }),
      bodyCell("head_pose()\n\u7B2C 149 \u884C", 2200, { shading: LIGHT_GRAY }),
      bodyCell("pitch: \u62AC\u5934/\u4F4E\u5934\nyaw: \u5DE6\u53F3\u8F6C\u5934\nroll: \u6B6A\u5934\n(\u5355\u4F4D\uFF1A\u5EA6)", 2360, { shading: LIGHT_GRAY }),
    ]}),
    new TableRow({ children: [
      bodyCell("\u89C6\u7EBF\u65B9\u5411\n(gaze)", 1800, { bold: true }),
      bodyCell("\u8BA1\u7B97\u8679\u819C\u4E2D\u5FC3\u76F8\u5BF9\u4E8E\n\u773C\u7403\u4E2D\u5FC3\u7684\u504F\u79FB\u91CF\n\u5DE6\u53F3\u773C\u5E73\u5747", 3000),
      bodyCell("iris_center()\ngaze_vector()\n\u7B2C 190-217 \u884C", 2200),
      bodyCell("gaze_x, gaze_y\n-1.5 \u5230 1.5\n> 0.35 = \u89C6\u7EBF\u504F\u79FB", 2360),
    ]}),
    new TableRow({ children: [
      bodyCell("\u7736\u773C\u8BA1\u6570", 1800, { bold: true, shading: LIGHT_GRAY }),
      bodyCell("\u8FDE\u7EED\u5E27 EAR < \u9608\u503C\n\u2192 \u8BA1\u4E3A\u4E00\u6B21\u7736\u773C", 3000, { shading: LIGHT_GRAY }),
      bodyCell("analyze_frame()\n\u7B2C 260-266 \u884C", 2200, { shading: LIGHT_GRAY }),
      bodyCell("\u7D2F\u8BA1\u6B21\u6570\n\u7528\u4E8E\u75B2\u52B3\u5224\u65AD", 2360, { shading: LIGHT_GRAY }),
    ]})
  ]
}));

children.push(para(""));
children.push(heading3("\u6838\u5FC3\u51FD\u6570 analyze_frame() \u7684\u6267\u884C\u6D41\u7A0B\uFF08\u7B2C 234 \u884C\uFF09"));
children.push(richPara([
  { text: "\u6BCF\u6B21\u6444\u50CF\u5934\u6355\u83B7\u4E00\u5E27\u753B\u9762\uFF0C\u90FD\u4F1A\u8C03\u7528\u8FD9\u4E2A\u51FD\u6570\u3002\u5B83\u7684\u6267\u884C\u987A\u5E8F\u662F\uFF1A" }
]));
children.push(richPara([{ text: "\u2460 BGR \u2192 RGB \u8F6C\u6362", bold: true }, { text: " \u2192 " }, { text: "\u2461 MediaPipe \u68C0\u6D4B 478 \u4E2A\u9762\u90E8\u70B9", bold: true }, { text: " \u2192 " }, { text: "\u2462 \u8BA1\u7B97\u5DE6\u53F3\u773C EAR", bold: true }, { text: " \u2192 " }, { text: "\u2463 solvePnP \u6C42\u5934\u90E8\u59FF\u6001", bold: true }, { text: " \u2192 " }, { text: "\u2464 \u8679\u819C\u5B9A\u4F4D + \u89C6\u7EBF\u5411\u91CF", bold: true }, { text: " \u2192 " }, { text: "\u2465 \u7EFC\u5408\u5224\u65AD\u6CE8\u610F\u529B\u6807\u7B7E", bold: true }]));

children.push(para(""));
children.push(heading3("\u6CE8\u610F\u529B\u6807\u7B7E\u5224\u65AD\u903B\u8F91\uFF08\u7B2C 219 \u884C\uFF09"));
children.push(para("\u7CFB\u7EDF\u7528\u4E09\u4E2A\u6761\u4EF6\u7684\u201C\u4E0E\u201D\u5173\u7CFB\u6765\u5224\u65AD\u5B66\u751F\u662F\u5426\u4E13\u6CE8\uFF1A"));

children.push(new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [2200, 3200, 1800, 2160],
  rows: [
    new TableRow({ children: [
      headerCell("\u6761\u4EF6", 2200), headerCell("\u4EE3\u7801\u5B9E\u73B0", 3200), headerCell("\u9608\u503C", 1800), headerCell("\u5224\u65AD\u7ED3\u679C", 2160),
    ]}),
    new TableRow({ children: [
      bodyCell("\u773C\u775B\u7761\u5F00", 2200, { bold: true }),
      bodyCell("EAR\u5DE6 > 0.21 \u4E14 EAR\u53F3 > 0.21", 3200),
      bodyCell("ear_thresh = 0.21", 1800),
      bodyCell("\u5426 \u2192 \u201C\u773C\u775B\u95ED\u5408\u201D", 2160),
    ]}),
    new TableRow({ children: [
      bodyCell("\u9762\u671D\u524D\u65B9", 2200, { bold: true, shading: LIGHT_GRAY }),
      bodyCell("|yaw| < 20\u00B0 \u4E14 |pitch| < 20\u00B0", 3200, { shading: LIGHT_GRAY }),
      bodyCell("yaw/pitch_thresh\n= 20.0\u00B0", 1800, { shading: LIGHT_GRAY }),
      bodyCell("\u5426 \u2192 \u201C\u89C6\u7EBF\u504F\u79BB\u201D", 2160, { shading: LIGHT_GRAY }),
    ]}),
    new TableRow({ children: [
      bodyCell("\u89C6\u7EBF\u5C45\u4E2D", 2200, { bold: true }),
      bodyCell("|gaze_x| < 0.35 \u4E14 |gaze_y| < 0.35", 3200),
      bodyCell("gaze_off_center\n= 0.35", 1800),
      bodyCell("\u5426 \u2192 \u201C\u89C6\u7EBF\u504F\u79FB\u201D", 2160),
    ]}),
    new TableRow({ children: [
      bodyCell("\u4EE5\u4E0A\u5168\u90E8\u6EE1\u8DB3", 2200, { bold: true, shading: LIGHT_BLUE }),
      bodyCell("", 3200, { shading: LIGHT_BLUE }),
      bodyCell("", 1800, { shading: LIGHT_BLUE }),
      bodyCell("\u2192 \u201C\u4E13\u6CE8\u201D", 2160, { shading: LIGHT_BLUE }),
    ]})
  ]
}));

children.push(new Paragraph({ children: [new PageBreak()] }));

// === 第四章：评分公式 ===
children.push(heading1("\u7B2C\u56DB\u7AE0  \u8BC4\u5206\u516C\u5F0F\u2014\u2014\u8BBA\u6587\u516C\u5F0F 14 \u5728\u4EE3\u7801\u4E2D\u600E\u4E48\u5FEB\u901F\u8BB2\u6E05\u695A"));

children.push(heading2("4.1 \u516C\u5F0F\u603B\u89C8"));
children.push(richPara([
  { text: "\u8BBA\u6587\u516C\u5F0F\uFF1A", bold: true },
  { text: "S_att(t) = S_base \u2212 \u03A3\u03B4(i,t) \u2212 \u03BB\u03A3\u03B7_j\u00B7\u0177_j(t)" }
]));
children.push(para("\u7528\u4EBA\u8BDD\u8BF4\uFF1A\u6CE8\u610F\u529B\u5F97\u5206 = \u6EE1\u5206\u57FA\u7EBF \u2212 \u884C\u4E3A\u504F\u79BB\u6263\u5206 \u2212 \u60C5\u7EEA\u5F71\u54CD\u8C03\u6574", { spaceBefore: 120 }));

children.push(para(""));
children.push(heading2("4.2 \u4EE3\u7801\u7684\u201C\u53CD\u5411\u201D\u5B9E\u73B0"));
children.push(tipBox("\u5173\u952E\u7406\u89E3", "\u8BBA\u6587\u662F\u201C\u4ECE 100 \u5206\u5F00\u59CB\u6263\u5206\u201D\uFF0C\u4EE3\u7801\u662F\u201C\u4ECE 0 \u5206\u5F00\u59CB\u6323\u5206\u201D\u3002\u6570\u5B66\u4E0A\u5B8C\u5168\u7B49\u4EF7\uFF0C\u53EA\u662F\u65B9\u5411\u53CD\u4E86\u3002"));
children.push(para("\u4EE3\u7801\u628A\u201C\u884C\u4E3A\u504F\u79BB\u201D\u5206\u89E3\u6210 6 \u4E2A\u5B50\u8BC4\u5206\u7EF4\u5EA6\uFF0C\u52A0\u8D77\u6765\u51D1\u6EE1 100 \u5206\uFF1A", { spaceBefore: 120 }));

children.push(new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [500, 2000, 800, 3600, 2460],
  rows: [
    new TableRow({ children: [
      headerCell("#", 500), headerCell("\u8BC4\u5206\u7EF4\u5EA6", 2000), headerCell("\u6EE1\u5206", 800), headerCell("\u5173\u952E\u903B\u8F91", 3600), headerCell("\u5BF9\u5E94\u8BBA\u6587", 2460),
    ]}),
    new TableRow({ children: [
      bodyCell("1", 500), bodyCell("\u773C\u775B\u7741\u5F00\u5EA6", 2000, { bold: true }), bodyCell("25\u5206", 800),
      bodyCell("EAR\u5747\u503C\u8D8A\u9AD8\u5F97\u5206\u8D8A\u591A\uFF1B\u5DE6\u53F3\u773C\u4E0D\u5BF9\u79F0\u6263\u5206\uFF1B30\u5E27\u7A33\u5B9A\u6027\u5956\u52B1", 3600),
      bodyCell("Section 2.3.1\nEAR \u6307\u6807", 2460),
    ]}),
    new TableRow({ children: [
      bodyCell("2", 500, { shading: LIGHT_GRAY }), bodyCell("\u89C6\u7EBF\u7A33\u5B9A\u6027", 2000, { bold: true, shading: LIGHT_GRAY }), bodyCell("20\u5206", 800, { shading: LIGHT_GRAY }),
      bodyCell("\u89C6\u7EBF\u504F\u79FB\u5E45\u5EA6\u8D8A\u5C0F\u5F97\u5206\u8D8A\u591A\uFF1B\u60E9\u7F5A\u5FEB\u901F\u626B\u89C6\uFF1B60\u5E27\u7A33\u5B9A\u5956\u52B1", 3600, { shading: LIGHT_GRAY }),
      bodyCell("Section 2.3\n\u89C6\u7EBF\u8FFD\u8E2A", 2460, { shading: LIGHT_GRAY }),
    ]}),
    new TableRow({ children: [
      bodyCell("3", 500), bodyCell("\u5934\u90E8\u7A33\u5B9A\u6027", 2000, { bold: true }), bodyCell("15\u5206", 800),
      bodyCell("\u5934\u90E8\u504F\u8F6C\u89D2\u5EA6\u8D8A\u5C0F\u5F97\u5206\u8D8A\u591A\uFF1B\u60E9\u7F5A\u5FEB\u901F\u8F6C\u5934", 3600),
      bodyCell("Section 2.3\nSolvePnP", 2460),
    ]}),
    new TableRow({ children: [
      bodyCell("4", 500, { shading: LIGHT_GRAY }), bodyCell("\u6301\u7EED\u4E13\u6CE8\u65F6\u957F", 2000, { bold: true, shading: LIGHT_GRAY }), bodyCell("20\u5206", 800, { shading: LIGHT_GRAY }),
      bodyCell("\u4E13\u6CE8 \u226510\u79D2=18\u5206\uFF1B\u22655\u79D2=14\u5206\uFF1B\u22652\u79D2=9\u5206\uFF1B<2\u79D2=4\u5206", 3600, { shading: LIGHT_GRAY }),
      bodyCell("ADHD \u6838\u5FC3\u6307\u6807\n\u6301\u7EED\u6CE8\u610F\u529B\u7F3A\u9677", 2460, { shading: LIGHT_GRAY }),
    ]}),
    new TableRow({ children: [
      bodyCell("5", 500), bodyCell("\u7736\u773C\u6A21\u5F0F", 2000, { bold: true }), bodyCell("10\u5206", 800),
      bodyCell("\u6B63\u5E38\u8303\u56F4 10-30\u6B21/\u5206\u949F\uFF1B\u8FC7\u5FEB\u6216\u8FC7\u6162\u6263\u5206\uFF1B\u8FDE\u7EED\u5FEB\u901F\u7736\u773C\u7C07\u60E9\u7F5A", 3600),
      bodyCell("Section 2.3.1\n\u7736\u773C\u5F02\u5E38\u68C0\u6D4B", 2460),
    ]}),
    new TableRow({ children: [
      bodyCell("6", 500, { shading: LIGHT_GRAY }), bodyCell("\u8FD0\u52A8\u8E81\u52A8\u5EA6", 2000, { bold: true, shading: LIGHT_GRAY }), bodyCell("10\u5206", 800, { shading: LIGHT_GRAY }),
      bodyCell("\u68C0\u6D4B\u5FAE\u5C0F\u5934\u90E8\u52A8\u4F5C\u9891\u7387\uFF1B\u9891\u7E41\u5FAE\u52A8 = ADHD\u591A\u52A8\u7279\u5F81", 3600, { shading: LIGHT_GRAY }),
      bodyCell("Section 2.1\nbody micromotion", 2460, { shading: LIGHT_GRAY }),
    ]})
  ]
}));

children.push(para(""));
children.push(heading2("4.3 \u60C5\u7EEA\u8C03\u6574\u9879\uFF08\u516C\u5F0F\u7684\u7B2C\u4E09\u9879\uFF09"));
children.push(para("\u60C5\u7EEA\u8BC6\u522B\u7ED3\u679C\u4F1A\u5BF9\u6CE8\u610F\u529B\u5F97\u5206\u8FDB\u884C\u5FAE\u8C03\uFF1A", { spaceBefore: 120 }));

children.push(new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [1600, 1600, 3200, 2960],
  rows: [
    new TableRow({ children: [
      headerCell("\u60C5\u7EEA", 1600), headerCell("\u8C03\u6574\u5206\u6570", 1600), headerCell("\u8BF4\u660E", 3200), headerCell("\u8BA1\u7B97\u65B9\u5F0F", 2960),
    ]}),
    new TableRow({ children: [
      bodyCell("\u5FEB\u4E50", 1600, { bold: true }), bodyCell("+6", 1600, { color: "2E7D32" }),
      bodyCell("\u79EF\u6781\u60C5\u7EEA\u6709\u52A9\u4E8E\u6CE8\u610F\u529B\u7EF4\u6301", 3200),
      bodyCell("\u8C03\u6574\u5206 = \u7CFB\u6570 \u00D7 \u7F6E\u4FE1\u5EA6\n\u4F8B: +6 \u00D7 0.9 = +5.4\u5206", 2960),
    ]}),
    new TableRow({ children: [
      bodyCell("\u60CA\u8BB6", 1600, { bold: true, shading: LIGHT_GRAY }), bodyCell("+3", 1600, { color: "2E7D32", shading: LIGHT_GRAY }),
      bodyCell("\u77ED\u6682\u63D0\u9AD8\u6CE8\u610F\u529B", 3200, { shading: LIGHT_GRAY }),
      bodyCell("", 2960, { shading: LIGHT_GRAY }),
    ]}),
    new TableRow({ children: [
      bodyCell("\u4E2D\u6027", 1600, { bold: true }), bodyCell("0", 1600),
      bodyCell("\u6700\u5229\u4E8E\u4E13\u6CE8\u7684\u72B6\u6001\uFF0C\u4E0D\u8C03\u6574", 3200), bodyCell("", 2960),
    ]}),
    new TableRow({ children: [
      bodyCell("\u538C\u6076", 1600, { bold: true, shading: LIGHT_GRAY }), bodyCell("-4", 1600, { color: "C62828", shading: LIGHT_GRAY }),
      bodyCell("\u8F7B\u5EA6\u8D1F\u9762\u5F71\u54CD", 3200, { shading: LIGHT_GRAY }), bodyCell("", 2960, { shading: LIGHT_GRAY }),
    ]}),
    new TableRow({ children: [
      bodyCell("\u60B2\u4F24", 1600, { bold: true }), bodyCell("-6", 1600, { color: "C62828" }),
      bodyCell("\u5F71\u54CD\u6CE8\u610F\u529B\u7EF4\u6301", 3200), bodyCell("", 2960),
    ]}),
    new TableRow({ children: [
      bodyCell("\u6050\u60E7", 1600, { bold: true, shading: LIGHT_GRAY }), bodyCell("-8", 1600, { color: "C62828", shading: LIGHT_GRAY }),
      bodyCell("\u6CE8\u610F\u529B\u5206\u6563", 3200, { shading: LIGHT_GRAY }), bodyCell("", 2960, { shading: LIGHT_GRAY }),
    ]}),
    new TableRow({ children: [
      bodyCell("\u751F\u6C14", 1600, { bold: true }), bodyCell("-10", 1600, { color: "C62828" }),
      bodyCell("\u6700\u4E25\u91CD\u7684\u8D1F\u9762\u5F71\u54CD", 3200), bodyCell("", 2960),
    ]})
  ]
}));

children.push(new Paragraph({ children: [new PageBreak()] }));

// === 第五章：语音提醒 ===
children.push(heading1("\u7B2C\u4E94\u7AE0  \u8BED\u97F3\u63D0\u9192\u7CFB\u7EDF"));

children.push(para("VoiceReminderSystem \u662F\u7528\u6237\u611F\u77E5\u6700\u5F3A\u7684\u529F\u80FD\u2014\u2014\u5B66\u751F\u8D70\u795E\u65F6\uFF0C\u7CFB\u7EDF\u4F1A\u81EA\u52A8\u8BED\u97F3\u63D0\u9192\u3002", { spaceBefore: 120 }));

children.push(new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [2000, 3600, 3760],
  rows: [
    new TableRow({ children: [
      headerCell("\u53C2\u6570", 2000), headerCell("\u8BBE\u7F6E", 3600), headerCell("\u8BF4\u660E", 3760),
    ]}),
    new TableRow({ children: [
      bodyCell("\u63D0\u9192\u51B7\u5374\u65F6\u95F4", 2000, { bold: true }),
      bodyCell("15 \u79D2", 3600),
      bodyCell("\u4E24\u6B21\u63D0\u9192\u4E4B\u95F4\u81F3\u5C11\u95F4\u9694 15 \u79D2\uFF0C\u907F\u514D\u53CD\u590D\u9A9A\u6270", 3760),
    ]}),
    new TableRow({ children: [
      bodyCell("\u8BED\u97F3\u901F\u7387", 2000, { bold: true, shading: LIGHT_GRAY }),
      bodyCell("150 \u5B57/\u5206\u949F", 3600, { shading: LIGHT_GRAY }),
      bodyCell("\u9002\u5408\u513F\u7AE5\u7684\u8BED\u901F", 3760, { shading: LIGHT_GRAY }),
    ]}),
    new TableRow({ children: [
      bodyCell("\u89E6\u53D1\u65B9\u5F0F", 2000, { bold: true }),
      bodyCell("\u6CE8\u610F\u529B\u5206\u6570\u4F4E\u4E8E\u9608\u503C\u65F6\u89E6\u53D1", 3600),
      bodyCell("\u901A\u8FC7\u8BED\u97F3\u961F\u5217 + \u72EC\u7ACB\u7EBF\u7A0B\u5904\u7406\uFF0C\u4E0D\u5361\u4F4F\u4E3B\u754C\u9762", 3760),
    ]})
  ]
}));

children.push(para(""));
children.push(tipBox("\u6F14\u793A\u65F6\u7684\u8BF4\u6CD5", "\u201C\u6211\u4EEC\u7684\u8BED\u97F3\u53CD\u9988\u7CFB\u7EDF\u662F\u5F02\u6B65\u8BBE\u8BA1\u7684\u2014\u2014\u5B83\u5728\u72EC\u7ACB\u7EBF\u7A0B\u91CC\u8FD0\u884C\uFF0C\u4E0D\u4F1A\u56E0\u4E3A\u8BED\u97F3\u64AD\u62A5\u800C\u5361\u4F4F\u89C6\u9891\u5206\u6790\u3002\u800C\u4E14\u6709 15 \u79D2\u7684\u51B7\u5374\u65F6\u95F4\uFF0C\u4E0D\u4F1A\u8BA9\u5B69\u5B50\u89C9\u5F97\u88AB\u201C\u8F70\u70B8\u201D\u3002\u201D"));

children.push(new Paragraph({ children: [new PageBreak()] }));

// === 第六章：论文对应 ===
children.push(heading1("\u7B2C\u516D\u7AE0  \u8BBA\u6587\u4E0E\u4EE3\u7801\u7684\u5BF9\u5E94\u5173\u7CFB\u901F\u67E5\u8868"));

children.push(para("\u8FD9\u662F\u4F60\u6700\u9700\u8981\u719F\u60C9\u7684\u5BF9\u5E94\u5173\u7CFB\u3002\u5EFA\u8BAE\u6253\u5370\u51FA\u6765\u653E\u5728\u624B\u8FB9\uFF1A", { spaceBefore: 120 }));

children.push(new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [2400, 2400, 2400, 2160],
  rows: [
    new TableRow({ children: [
      headerCell("\u8BBA\u6587\u67B6\u6784\u5C42", 2400), headerCell("\u8BBA\u6587\u7AE0\u8282", 2400), headerCell("\u4EE3\u7801\u7C7B", 2400), headerCell("\u6838\u5FC3\u51FD\u6570", 2160),
    ]}),
    new TableRow({ children: [
      bodyCell("\u5E27\u63D0\u53D6\u4E0E\u566A\u58F0\u6291\u5236", 2400, { bold: true }),
      bodyCell("Section 2 \u5F00\u5934", 2400),
      bodyCell("ADHDDetectionSystem", 2400),
      bodyCell("update_frame()\nstart_camera()", 2160),
    ]}),
    new TableRow({ children: [
      bodyCell("\u6DF1\u5EA6\u884C\u4E3A\u6316\u6398\u5F15\u64CE", 2400, { bold: true, shading: LIGHT_GRAY }),
      bodyCell("Section 2.1", 2400, { shading: LIGHT_GRAY }),
      bodyCell("FacialModeling\n+ ADHD\u7279\u5F81\u68C0\u6D4B", 2400, { shading: LIGHT_GRAY }),
      bodyCell("extract_face_features()\ndetect_adhd_features()", 2160, { shading: LIGHT_GRAY }),
    ]}),
    new TableRow({ children: [
      bodyCell("DAF-Xception\n\u60C5\u611F\u8BC6\u522B\u7F51\u7EDC", 2400, { bold: true }),
      bodyCell("Section 2.2", 2400),
      bodyCell("EmotionAnalyzer", 2400),
      bodyCell("predict_emotion()\nanalyze_frame()", 2160),
    ]}),
    new TableRow({ children: [
      bodyCell("STBP-AS\n\u591A\u6A21\u6001\u8FFD\u8E2A\u5668", 2400, { bold: true, shading: LIGHT_GRAY }),
      bodyCell("Section 2.3", 2400, { shading: LIGHT_GRAY }),
      bodyCell("AttentionAnalyzer", 2400, { shading: LIGHT_GRAY }),
      bodyCell("eye_aspect_ratio()\nhead_pose()\ngaze_vector()", 2160, { shading: LIGHT_GRAY }),
    ]}),
    new TableRow({ children: [
      bodyCell("\u6CE8\u610F\u529B\u8BC4\u5206\u4E0E\u878D\u5408", 2400, { bold: true }),
      bodyCell("Section 2.3.2\n\u516C\u5F0F 13-14", 2400),
      bodyCell("OptimizedAttention\nScoringSystem", 2400),
      bodyCell("calculate_attention\n_score()\n6 \u4E2A\u5B50\u8BC4\u5206\u51FD\u6570", 2160),
    ]}),
    new TableRow({ children: [
      bodyCell("\u81EA\u9002\u5E94\u5B66\u4E60\u652F\u6301", 2400, { bold: true, shading: LIGHT_GRAY }),
      bodyCell("Section 2.4", 2400, { shading: LIGHT_GRAY }),
      bodyCell("VoiceReminderSystem\n+ CalibrationSystem\n+ RealTimeCharts", 2400, { shading: LIGHT_GRAY }),
      bodyCell("speak()\ncalibrate()\ndraw_attention_chart()", 2160, { shading: LIGHT_GRAY }),
    ]})
  ]
}));

children.push(new Paragraph({ children: [new PageBreak()] }));

// === 第七章：常见问题 ===
children.push(heading1("\u7B2C\u4E03\u7AE0  \u5E38\u89C1\u95EE\u9898\u9884\u6F14\u2014\u2014\u522B\u4EBA\u4F1A\u95EE\u4F60\u4EC0\u4E48"));

children.push(para("\u4EE5\u4E0B\u662F\u4F60\u5728\u6F14\u793A\u3001\u6C9F\u901A\u3001\u9762\u8BD5\u65F6\u6700\u53EF\u80FD\u88AB\u95EE\u5230\u7684\u95EE\u9898\u548C\u5EFA\u8BAE\u56DE\u7B54\uFF1A", { spaceBefore: 120 }));

const qaData = [
  { q: "Q1: \u201C\u4F60\u7684\u7CFB\u7EDF\u548C\u666E\u901A\u7684\u6CE8\u610F\u529B\u68C0\u6D4B\u6709\u4EC0\u4E48\u533A\u522B\uFF1F\u201D",
    a: "\u201C\u6700\u5927\u7684\u533A\u522B\u662F\u6211\u4EEC\u7684\u8BC4\u5206\u6743\u91CD\u662F\u57FA\u4E8E ADHD \u795E\u7ECF\u75C5\u7406\u5B66\u7814\u7A76\u7684\u3002\u6BD4\u5982\u6301\u7EED\u4E13\u6CE8\u65F6\u957F\u5360\u4E86 20% \u7684\u6743\u91CD\uFF0C\u56E0\u4E3A\u6CE8\u610F\u529B\u7EF4\u6301\u56F0\u96BE\u662F ADHD \u7684\u6838\u5FC3\u75C7\u72B6\u3002\u53E6\u5916\u6211\u4EEC\u4E13\u95E8\u589E\u52A0\u4E86\u8FD0\u52A8\u8E81\u52A8\u5EA6\u68C0\u6D4B\uFF0C\u8FD9\u662F\u9488\u5BF9 ADHD \u591A\u52A8\u7279\u5F81\u8BBE\u8BA1\u7684\u3002\u201D" },
  { q: "Q2: \u201C\u4F60\u4EEC\u7684\u60C5\u7EEA\u8BC6\u522B\u51C6\u786E\u7387\u600E\u4E48\u6837\uFF1F\u201D",
    a: "\u201C\u6211\u4EEC\u7528\u7684\u662F\u6539\u8FDB\u7684 Xception \u7F51\u7EDC\uFF0C\u5728 CK+ \u6570\u636E\u96C6\u4E0A\u7684\u8868\u73B0\u4F18\u4E8E\u6807\u51C6 Xception\u3002\u5177\u4F53\u6765\u8BF4\uFF0C\u60C5\u7EEA\u8BC6\u522B\u662F\u8F85\u52A9\u6307\u6807\u800C\u4E0D\u662F\u4E3B\u6307\u6807\u2014\u2014\u5B83\u53EA\u5360\u8BC4\u5206\u8C03\u6574\u7684\u4E00\u5C0F\u90E8\u5206\uFF0C\u4E3B\u8981\u5F97\u5206\u8FD8\u662F\u6765\u81EA\u773C\u52A8\u548C\u5934\u90E8\u59FF\u6001\u68C0\u6D4B\u3002\u201D" },
  { q: "Q3: \u201C\u5B9E\u65F6\u6027\u600E\u4E48\u6837\uFF1F\u4F1A\u4E0D\u4F1A\u5361\uFF1F\u201D",
    a: "\u201C\u6211\u4EEC\u7684\u89C6\u9891\u5904\u7406\u5E27\u7387\u8BBE\u7F6E\u4E3A 15fps\uFF0C\u8DB3\u591F\u6355\u6349\u6CE8\u610F\u529B\u53D8\u5316\u3002\u8BED\u97F3\u63D0\u9192\u5728\u72EC\u7ACB\u7EBF\u7A0B\u4E2D\u8FD0\u884C\uFF0C\u4E0D\u4F1A\u5361\u4F4F\u4E3B\u754C\u9762\u3002\u60C5\u7EEA\u8BC6\u522B\u56E0\u4E3A\u7528\u4E86\u6DF1\u5EA6\u5B66\u4E60\u6A21\u578B\u4F1A\u7A0D\u5FAE\u6162\u4E00\u70B9\uFF0C\u4F46\u4E0D\u5F71\u54CD\u6574\u4F53\u4F53\u9A8C\u3002\u201D" },
  { q: "Q4: \u201C\u4E3A\u4EC0\u4E48\u7528 MediaPipe \u800C\u4E0D\u662F\u5176\u4ED6\u65B9\u6848\uFF1F\u201D",
    a: "\u201CMediaPipe FaceMesh \u80FD\u63D0\u4F9B 478 \u4E2A\u9762\u90E8\u7279\u5F81\u70B9\uFF0C\u5305\u62EC\u8679\u819C\u5B9A\u4F4D\uFF0C\u800C\u4E14\u5B83\u662F\u8F7B\u91CF\u7EA7\u7684\u2014\u2014\u4E0D\u9700\u8981 GPU \u5C31\u80FD\u5B9E\u65F6\u8FD0\u884C\u3002\u8FD9\u5BF9\u4E8E\u5BB6\u5EAD\u573A\u666F\u5F88\u91CD\u8981\uFF0C\u56E0\u4E3A\u5B69\u5B50\u7684\u7535\u8111\u53EF\u80FD\u914D\u7F6E\u4E0D\u9AD8\u3002\u201D" },
  { q: "Q5: \u201C\u8FD9\u4E2A\u7CFB\u7EDF\u80FD\u5904\u7406\u591A\u4E2A\u5B66\u751F\u5417\uFF1F\u201D",
    a: "\u201C\u76EE\u524D\u8BBE\u8BA1\u7684\u662F\u5355\u5B66\u751F\u6A21\u5F0F\u2014\u2014\u4E00\u4E2A\u6444\u50CF\u5934\u76D1\u6D4B\u4E00\u4E2A\u5B66\u751F\u3002\u8FD9\u5176\u5B9E\u66F4\u7B26\u5408 ADHD \u7684\u4F7F\u7528\u573A\u666F\uFF0C\u56E0\u4E3A\u8FD9\u4E9B\u5B69\u5B50\u901A\u5E38\u9700\u8981\u4E00\u5BF9\u4E00\u6216\u5C0F\u73ED\u6559\u5B66\u3002\u672A\u6765\u53EF\u4EE5\u6269\u5C55\u4E3A\u591A\u5B66\u751F\u7248\u672C\u3002\u201D" },
];

for (const qa of qaData) {
  children.push(heading3(qa.q));
  children.push(para(qa.a, { spaceBefore: 60 }));
  children.push(para(""));
}

children.push(new Paragraph({ children: [new PageBreak()] }));

// === 第八章：动手实验 ===
children.push(heading1("\u7B2C\u516B\u7AE0  \u52A8\u624B\u5B9E\u9A8C\u2014\u2014\u7528\u8EAB\u4F53\u611F\u53D7\u4EE3\u7801"));

children.push(para("\u8BFB\u4EE3\u7801\u4E0D\u5982\u73A9\u4EE3\u7801\u3002\u4EE5\u4E0B\u662F 5 \u4E2A\u4F60\u53EF\u4EE5\u5728\u81EA\u5DF1\u7535\u8111\u4E0A\u505A\u7684\u5B9E\u9A8C\uFF0C\u6BCF\u4E2A\u5B9E\u9A8C\u90FD\u80FD\u5E2E\u4F60\u52A0\u6DF1\u5BF9\u67D0\u4E2A\u6A21\u5757\u7684\u7406\u89E3\uFF1A", { spaceBefore: 120 }));

const experiments = [
  { title: "\u5B9E\u9A8C 1\uFF1A\u89C2\u5BDF\u773C\u775B\u72B6\u6001\u5206\u6570\u53D8\u5316",
    action: "\u542F\u52A8\u7CFB\u7EDF\u540E\uFF0C\u5FEB\u901F\u7736\u773C 10 \u6B21\uFF0C\u7136\u540E\u95ED\u773C 3 \u79D2",
    observe: "\u89C2\u5BDF EAR \u503C\u548C\u6CE8\u610F\u529B\u5206\u6570\u7684\u53D8\u5316\uFF0C\u6CE8\u610F\u95ED\u773C\u65F6\u5206\u6570\u4F1A\u5927\u5E45\u4E0B\u964D\uFF08\u773C\u775B\u7EF4\u5EA6\u76F4\u63A5 0 \u5206\uFF09",
    learn: "\u7406\u89E3 eye_aspect_ratio() \u548C\u9608\u503C 0.21 \u7684\u542B\u4E49" },
  { title: "\u5B9E\u9A8C 2\uFF1A\u6D4B\u8BD5\u5934\u90E8\u59FF\u6001\u68C0\u6D4B",
    action: "\u5148\u6B63\u5BF9\u6444\u50CF\u5934 10 \u79D2\uFF0C\u7136\u540E\u6162\u6162\u5411\u5DE6\u8F6C\u5934",
    observe: "\u89C2\u5BDF yaw \u503C\u7684\u53D8\u5316\uFF0C\u5F53 |yaw| > 20\u00B0 \u65F6\u7CFB\u7EDF\u5224\u65AD\u4E3A\u201C\u89C6\u7EBF\u504F\u79BB\u201D",
    learn: "\u7406\u89E3 head_pose() \u548C solvePnP \u7684\u5B9E\u9645\u6548\u679C" },
  { title: "\u5B9E\u9A8C 3\uFF1A\u89E6\u53D1\u8BED\u97F3\u63D0\u9192",
    action: "\u6301\u7EED\u770B\u5411\u65C1\u8FB9\u8D85\u8FC7 15 \u79D2",
    observe: "\u7CFB\u7EDF\u4F1A\u89E6\u53D1\u8BED\u97F3\u63D0\u9192\uFF0C\u4E14\u4E0D\u4F1A\u5728 15 \u79D2\u5185\u91CD\u590D\u63D0\u9192",
    learn: "\u7406\u89E3 VoiceReminderSystem \u7684\u51B7\u5374\u673A\u5236" },
  { title: "\u5B9E\u9A8C 4\uFF1A\u89C2\u5BDF\u60C5\u7EEA\u8C03\u6574\u6548\u679C",
    action: "\u5148\u4FDD\u6301\u5FAE\u7B11\u770B\u5C4F\u5E55\uFF0C\u7136\u540E\u505A\u51FA\u751F\u6C14\u7684\u8868\u60C5",
    observe: "\u89C2\u5BDF\u60C5\u7EEA\u6807\u7B7E\u53D8\u5316\u548C\u6CE8\u610F\u529B\u5206\u6570\u7684\u5FAE\u8C03\uFF08\u5FEB\u4E50\u65F6+\u5206\uFF0C\u751F\u6C14\u65F6-\u5206\uFF09",
    learn: "\u7406\u89E3 EmotionAnalyzer \u548C emotion_adjustment \u7684\u5DE5\u4F5C\u539F\u7406" },
  { title: "\u5B9E\u9A8C 5\uFF1A\u6D4B\u8BD5\u6301\u7EED\u4E13\u6CE8\u65F6\u957F\u8BC4\u5206",
    action: "\u76EE\u4E0D\u8F6C\u775B\u76EF\u7740\u5C4F\u5E55 15 \u79D2\uFF0C\u89C2\u5BDF\u5206\u6570\u7A33\u6B65\u4E0A\u5347",
    observe: "\u5206\u6570\u4F1A\u968F\u7740\u4E13\u6CE8\u65F6\u957F\u7684\u589E\u52A0\u800C\u9010\u6B65\u63D0\u9AD8\uFF082\u79D2\u21925\u79D2\u219210\u79D2\u7684\u8DF3\u53D8\u70B9\uFF09",
    learn: "\u7406\u89E3 duration_score \u4E3A\u4EC0\u4E48\u662F ADHD \u6838\u5FC3\u6307\u6807" }
];

for (let i = 0; i < experiments.length; i++) {
  const exp = experiments[i];
  children.push(heading3(exp.title));
  children.push(richPara([{ text: "\u64CD\u4F5C\uFF1A", bold: true }, { text: exp.action }]));
  children.push(richPara([{ text: "\u89C2\u5BDF\uFF1A", bold: true }, { text: exp.observe }]));
  children.push(richPara([{ text: "\u5B66\u5230\uFF1A", bold: true }, { text: exp.learn }]));
  if (i < experiments.length - 1) children.push(para(""));
}

// ============ 构建文档 ============
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: BLUE },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: ACCENT },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1 } },
    ]
  },
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "ADHD \u6CE8\u610F\u529B\u76D1\u6D4B\u7CFB\u7EDF \u00B7 \u4EE3\u7801\u5BFC\u89C8\u624B\u518C", font: "Arial", size: 16, color: "999999" })]
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "\u2014 ", font: "Arial", size: 16, color: "999999" }),
                     new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 16, color: "999999" }),
                     new TextRun({ text: " \u2014", font: "Arial", size: 16, color: "999999" })]
        })]
      })
    },
    children
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/sessions/intelligent-elegant-dirac/mnt/系统代码_claude/ADHD系统代码导览手册.docx", buffer);
  console.log("Document created successfully!");
});
