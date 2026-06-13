/**
 * v2.0.72 (#115 #120 root-cause workaround) — NLU intent extractor.
 *
 * Cascade upstream's `SendUserCascadeMessage` proto has no OpenAI
 * `tools[]` field. The proxy injects tool definitions into the system
 * prompt (additional_instructions_section), but GPT / GLM / Kimi
 * weren't trained on prompt-level tool-calling protocols — they see the
 * `<tool_call>{"name":...}</tool_call>` instructions, decide to call
 * the tool, but emit it as natural-language NARRATION instead of the
 * exact markup we asked for. v2.0.71 fabricate detection just flagged
 * these as failures; v2.0.72 actually RECOVERS the call.
 *
 * Real probe captures (from scripts/probes/v2071-glm-kimi-tool-probe):
 *
 *   GLM-4.7  → "I should call the shell_exec function with the command
 *               'echo HELLO_FROM_PROBE'."
 *   GLM-5.1  → "I'll run the shell command as requested."  (no args!)
 *   GPT-5.5  → "PROBE_V0270_1777751588"  (pure fabricated output)
 *
 * The first one carries enough signal to reconstruct the call; the
 * second has the intent but no args; the third is hopeless. Layered
 * extraction:
 *
 *   Layer 1 (highest confidence) — explicit invocation syntax:
 *     "Let me run shell_command(command='echo HELLO')"
 *     "function_call: shell_exec(\"echo HELLO\")"
 *
 *   Layer 2 — backtick-quoted name + value:
 *     "I'll call `shell_exec` with command `echo HELLO`"
 *     "use the `Read` function with file_path `/etc/hosts`"
 *
 *   Layer 3 — natural narrative (model "thinking out loud"):
 *     "I should call the shell_exec function with the command 'echo HI'"
 *     "Let me invoke the Read tool to read /etc/hosts"
 *
 * Each layer requires the extracted name to match a caller-declared
 * tool. Layer 3 also requires the user prompt to plausibly want a
 * tool call (shell-style verbs in the most recent user message).
 *
 * Conservative by design: false-positive tool_calls drive agent loops
 * to execute things the model didn't actually decide on. When in
 * doubt, return [].
 */

import { log } from '../config.js';

/**
 * @typedef {Object} ExtractedToolCall
 * @property {string} name        OpenAI tool name (matches caller's tools[])
 * @property {string} argumentsJson  JSON-stringified args
 * @property {'explicit-syntax'|'backtick-quoted'|'narrative'} layer
 * @property {number} confidence  0..1
 */

/**
 * Build a Set of declared tool names + a name → primaryParamName map
 * for inference of single-arg shorthands ("with command 'echo X'" →
 * arguments.command = 'echo X').
 */
function indexTools(tools) {
  const names = new Set();
  const primaryParam = new Map(); // tool name → inferable single string param
  if (!Array.isArray(tools)) return { names, primaryParam };
  for (const t of tools) {
    if (t?.type !== 'function') continue;
    const name = t.function?.name;
    if (!name || typeof name !== 'string') continue;
    names.add(name);
    const params = t.function?.parameters;
    if (params?.type === 'object' && params.properties) {
      const required = Array.isArray(params.required) ? params.required : [];
      const requiredStringKeys = required.filter(r => params.properties[r]?.type === 'string');
      const stringKeys = Object.keys(params.properties || {})
        .filter(k => params.properties[k]?.type === 'string');
      let primary = null;
      // Layer 2/3 shorthand recovery can only safely fill one argument.
      // Multi-field tools such as Claude Code's Edit need file_path,
      // old_string and new_string together; inferring only file_path from
      // narrative prose creates invalid tool_use blocks and stalls agents.
      if (required.length === 1 && requiredStringKeys.length === 1) {
        primary = requiredStringKeys[0];
      } else if (required.length === 0 && stringKeys.length === 1) {
        primary = stringKeys[0];
      }
      if (primary) primaryParam.set(name, primary);
    }
  }
  return { names, primaryParam };
}

// Regex utilities — escape user-controlled tool name for regex insertion.
function escapeRe(s) {
  return String(s).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// v2.0.78 (#120 follow-up + audit H-2): values extracted from narrative
// can easily be a generic noun phrase ("a shell command", "the file",
// "your input") or a literal placeholder keyword ("command",
// "argument"). Both produce garbage tool_calls — the agent loop will
// then try to execute `command` as a literal command, fail, and recurse.
// Reject these uniformly across all three layers.
const PLACEHOLDER_KEYWORDS = new Set([
  'command', 'argument', 'arguments', 'param', 'parameter',
  'parameters', 'input', 'value', 'file_path', 'filepath', 'path',
  'query', 'string', 'text', 'name', 'arg', 'output',
  // v2.0.81 (#125 — GLM-5.1 Chinese narrate): models echo Chinese
  // param-name keywords as the value too. "调用 shell_exec 命令 '命令'"
  // would otherwise produce a real tool_call with command='命令'.
  '命令', '参数', '文件', '路径', '输入', '值', '字符串', '文本', '名称', '查询', '输出',
]);
const ARTICLE_PREFIX_RE = /^(?:a|an|the|this|that|these|those|your|my|our|some|any|each|every)\s+/i;
// Chinese article-led / vague phrase prefixes — "某个命令" / "一个命令"
// / "某种参数" — same idea as ARTICLE_PREFIX_RE but for CJK.
const CN_VAGUE_PREFIX_RE = /^(?:某个?|一个|这个|那个|某种|什么|任何|每个|所有的?)/;

function looksLikePlaceholderValue(value) {
  if (typeof value !== 'string' || !value.trim()) return true;
  const v = value.trim();
  // Strip trailing punctuation (`.`, `,`, `;`, `:`, `。`, `，`) before comparison.
  const stripped = v.replace(/[.,;:!?。，；：！？]+$/, '');
  if (PLACEHOLDER_KEYWORDS.has(stripped.toLowerCase())) return true;
  // Article-led phrase ("a shell command", "the file") — model
  // narrating about the call rather than supplying the call value.
  if (ARTICLE_PREFIX_RE.test(stripped)) return true;
  // Chinese vague prefix — "某个命令", "一个文件", "这个参数"
  if (CN_VAGUE_PREFIX_RE.test(stripped)) return true;
  return false;
}

/**
 * Layer 1: explicit invocation syntax.
 *
 *   shell_command(command="echo X")
 *   shell_exec("echo X")
 *   function_call: name=shell_exec args={"command":"echo X"}
 */
function extractLayer1(text, names) {
  const out = [];
  // function_name(arg=value) or function_name("value")
  const reExplicit = /\b([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*(?:([A-Za-z_][A-Za-z0-9_]*)\s*=\s*)?["'`]([^"'`)]{1,2000})["'`]\s*\)/g;
  let m;
  while ((m = reExplicit.exec(text)) !== null) {
    const [, fn, paramName, value] = m;
    if (!names.has(fn)) continue;
    if (looksLikePlaceholderValue(value)) continue;
    const args = paramName ? { [paramName]: value } : { _value: value };
    out.push({
      name: fn,
      argumentsJson: JSON.stringify(args),
      layer: 'explicit-syntax',
      confidence: paramName ? 0.95 : 0.85,
    });
  }
  // function_call: name=X args={...}
  const reFc = /function[_\s]?call\s*[:=][^{]*?\bname\s*[:=]\s*["'`]?([A-Za-z_][A-Za-z0-9_]*)["'`]?[^{]*?(\{[\s\S]{1,2000}?\})/g;
  while ((m = reFc.exec(text)) !== null) {
    const [, fn, argsBlob] = m;
    if (!names.has(fn)) continue;
    let args = {};
    try { args = JSON.parse(argsBlob); } catch {}
    out.push({
      name: fn,
      argumentsJson: JSON.stringify(args),
      layer: 'explicit-syntax',
      confidence: 0.9,
    });
  }
  return out;
}

/**
 * Layer 2: backtick-quoted name + later backtick-quoted value.
 *
 *   "I'll call `shell_exec` with command `echo HELLO`"
 *   "use the `Read` function with file_path `/etc/hosts`"
 */
function extractLayer2(text, names, primaryParam) {
  const out = [];
  for (const fn of names) {
    const fnRe = new RegExp(`\\\`${escapeRe(fn)}\\\``, 'g');
    let m;
    while ((m = fnRe.exec(text)) !== null) {
      // Look for next backtick-quoted token within 200 chars
      const tail = text.slice(m.index + m[0].length, m.index + m[0].length + 200);
      // Capture optional "with PARAM `value`" or just "`value`"
      const argRe = /(?:with\s+)?(?:the\s+)?(?:argument|param|parameter|input|command|file[_-]?path|path|query)?\s*[:=]?\s*`([^`]{1,1000})`/i;
      const a = tail.match(argRe);
      if (!a) continue;
      const value = a[1];
      if (looksLikePlaceholderValue(value)) continue;
      const param = primaryParam.get(fn);
      if (!param) continue;
      out.push({
        name: fn,
        argumentsJson: JSON.stringify({ [param]: value }),
        layer: 'backtick-quoted',
        confidence: 0.8,
      });
    }
  }
  return out;
}

/**
 * Layer 3: natural narrative.
 *
 *   "I should call the shell_exec function with the command 'echo HI'"
 *   "Let me invoke the Read tool to read /etc/hosts"
 *   "I'll run shell_command with command echo HELLO"
 */
function extractLayer3(text, names, primaryParam) {
  const out = [];
  // v2.0.81 (#125 DuZunTianXia): GLM-5.1 narrate in Chinese — log
  // showed "让我用 Bash 来列出..." / "用户想查看..." / "我会调用 X
  // 工具" — none of which the English-only verb regex picked up.
  // Add Chinese verbs alongside English so the name pattern matches
  // either language (or mixed). The primary tool-name match still
  // requires the literal tool name (e.g. `Bash`, `shell_exec`) since
  // those are emitted in the original alphabet by every model.
  const verbs = '(?:call|invoke|run|use|execute|exec|trigger|fire'
    + '|调用|使用|运行|执行|触发|启动|让我用|让我使用|我会用|我将用|通过|借助|采用)';
  const articles = '(?:the\\s+)?';
  // Suffix matches ONLY tool/function meta-words (not arg labels like
  // "command" / "命令") so the latter stay in the tail and feed the
  // argPatterns. Pre-v2.0.81 it included "command" / "命令" which
  // greedily consumed the very keyword that argPattern 2/4 needs.
  const suffix = '(?:\\s+(?:function|tool|method|函数|工具|方法))?';
  for (const fn of names) {
    // Pattern: "<verb> [the] [function|tool] <fn> [function|tool]"
    // \b doesn't match between Chinese and Latin, so we drop the
    // leading word boundary and rely on the verb list itself.
    const namePatterns = [
      new RegExp(
        `${verbs}\\s*${articles}(?:function|tool|method|函数|工具|方法)?\\s*\\\`?${escapeRe(fn)}\\\`?${suffix}`,
        'gi',
      ),
      new RegExp(`(?:^|\\n)\\s*(?:\\d+[.)]\\s*|[-*]\\s*)?\\\`?${escapeRe(fn)}\\\`?\\s*[:：-]`, 'gi'),
    ];
    for (const namePat of namePatterns) {
      let m;
      while ((m = namePat.exec(text)) !== null) {
        // Hunt for value within next 300 chars
        const tail = text.slice(m.index + m[0].length, m.index + m[0].length + 300);
        // ordered by specificity:
        const argPatterns = [
          // Backtick-delimited command/path/query; allow quotes inside the value.
          /(?:^|[\s:：，,。.;；])(?:to\s+)?(?:run|execute|exec|call|invoke|use|运行|执行|调用|使用|命令(?:为)?|command(?:\s+is)?)\s*[:：]?\s*`([^`]{1,1000})`/i,
          // Plan-line value after "Bash:" / "Read:".
          /^[\s:：—-]*`([^`]{1,1000})`/,
          // 执行 `printf "OK\n"` / run `printf "OK\n"` — common in Claude Code
          // thinking text when the model narrates a tool intent instead of
          // emitting the protocol block.
          /(?:^|[\s:：，,。.;；])(?:to\s+)?(?:run|execute|exec|call|invoke|use|运行|执行|调用|使用|命令(?:为)?|command(?:\s+is)?)\s*[:：]?\s*["'`「『]([^"'`\n「」『』]{1,500})["'`」』]/i,
          // 调用 Bash 工具执行：printf "tool-ok\n" — command itself may contain
          // quotes, so capture the rest of the current plan line.
          /(?:^|[\s:：，,。.;；])(?:to\s+)?(?:run|execute|exec|运行|执行|命令(?:为)?|command(?:\s+is)?)\s*[:：]\s*([^\n。；;]{1,500})/i,
          // with the command 'echo X' / with command "echo X" / with command `echo X`
          /\bwith\s+(?:the\s+)?(?:command|argument|param(?:eter)?|input|file[_-]?path|path|query)\s+["'`]([^"'`\n]{1,500})["'`]/i,
          // bare keyword + value (no "with"): command 'echo X' / argument "X"
          /(?:^|\s)(?:command|argument|param(?:eter)?|input|file[_-]?path|path|query)\s+["'`]([^"'`\n]{1,500})["'`]/i,
          // 中文：用命令 'X' / 传入 'X' / 参数 'X' / 命令 'X' / 路径 'X'
          /(?:用|使用|传入|输入|参数(?:为)?|命令(?:为)?|路径(?:为)?|文件(?:为)?|查询(?:为)?)\s*[:：]?\s*["'`「『]([^"'`\n「」『』]{1,500})["'`」』]/,
          // with 'echo X' (no param keyword)
          /\bwith\s+["'`]([^"'`\n]{1,500})["'`]/i,
          // to read /etc/hosts (positional after action verb)
          /\bto\s+(?:read|run|execute|view|search|find|cat|ls)\s+([\S][^\n]{0,200})/i,
          // Bash: printf "OK\n" / shell_exec - echo OK (single-line plan form)
          /^[\s:：—-]+([^\n。；;]{1,500})/,
          // : 'echo X' / = 'echo X'
          /[:=]\s*["'`]([^"'`\n]{1,500})["'`]/,
          // last resort: very first quoted string in the tail
          /^[\s,，。.]*["'`「『]([^"'`\n「」『』]{1,500})["'`」』]/,
        ];
        let value = null;
        for (const pat of argPatterns) {
          const a = tail.match(pat);
          if (a && a[1]) { value = a[1].trim(); break; }
        }
        if (!value) continue;
        const param = primaryParam.get(fn);
        if (!param) continue;
        value = normalizeNarrativeValue(param, value);
        if (!value) continue;
        // v2.0.76 + v2.0.78 (audit H-2): reject placeholder keywords
        // (`command` / `argument` / ...) AND article-led prose phrases
        // (`a shell command` / `the file` / `your input`). GLM-4.7
        // narrative reproducer "to run a shell command" was capturing
        // "a shell command." as the value pre-v2.0.78 even with the
        // single-word filter in place.
        if (looksLikePlaceholderValue(value)) continue;
        out.push({
          name: fn,
          argumentsJson: JSON.stringify({ [param]: value }),
          layer: 'narrative',
          confidence: 0.65,
        });
      }
    }
  }
  return out;
}

function normalizeNarrativeValue(param, value) {
  if (typeof value !== 'string') return null;
  let v = value.trim();
  if (!v) return null;
  if (param === 'command') {
    v = v.replace(/^(?:Bash\s+)?command\s*[:：]\s*/i, '').trim();
    v = v.replace(/^Bash\s*[:：]\s*/i, '').trim();
    // Do not re-run historical steps described as already completed.
    if (/(?:已(?:经)?完成|完成|done|completed)/i.test(v)) return null;
    // Keep the executable command, not the explanatory sentence after it.
    v = v.replace(/\s+(?:[-—–]\s*)?(?:已(?:经)?完成|完成|done|completed).*$/i, '').trim();
    // Reject Chinese prose that is not a shell command.
    if (/^(?:创建|新建|读取|修改|编辑|使用|调用|执行|运行|列出|查看|搜索|检查|确认|等待|把|将|来|去|为了|然后|再|先)/.test(v)) return null;
    // Chinese prose such as "来列出当前工作目录下的文件" is not a shell
    // command. Keep real shell commands that merely contain CJK text.
    if (/[\p{Script=Han}]/u.test(v) && !shellCommandLooksConcrete(v)) return null;
  }
  if (param === 'file_path') {
    v = v.replace(/[.,;:!?。，；：！？]+$/, '').trim();
    // Avoid promoting prose such as "or Bash to confirm generated.txt".
    if (/^(?:or|and|或|和)\b/i.test(v)) return null;
    if (/\s/.test(v)) return null;
  }
  return v || null;
}

function shellCommandLooksConcrete(value) {
  return /^(?:printf|echo|cat|ls|pwd|grep|find|sed|awk|python3?|node|npm|pnpm|yarn|curl|mkdir|touch|cp|mv|rm|git|docker|bash|sh)\b/.test(value)
    || /(?:^|\s)(?:>|>>|\|\||&&|\|)(?:\s|$)/.test(value);
}

function preferredName(names, candidates) {
  for (const name of candidates) {
    if (names.has(name)) return name;
  }
  return null;
}

/**
 * 判断当前步骤匹配片段是否只是回顾已经完成的历史步骤。
 */
function looksLikeCompletedStepReference(snippet) {
  const s = String(snippet || '');
  return /(?:已经|已|刚刚|previously|already)[^\n。；;]{0,30}(?:完成|执行|运行|调用|处理|读取|编辑|做|complete|completed|ran|called|read|edited)/i.test(s)
    || /(?:完成|执行|运行|调用|处理|读取|编辑|做|尝试执行)了[^\n。；;]{0,20}(?:步骤|step)\s*\d+/i.test(s);
}

function detectActivePlanStepRange(text) {
  const patterns = [
    /(?:先|首先|当前|现在|本轮|只|仅|first|now|only)[^\n。；;]{0,50}(?:执行|运行|调用|处理|做|开始|execute|run|call|do)[^\n。；;]{0,30}(?:步骤|step(?:s)?)\s*(\d+)\s*(?:[-–—~至到]|to|through)\s*(\d+)/gi,
  ];
  for (const pattern of patterns) {
    let m;
    while ((m = pattern.exec(text)) !== null) {
      if (looksLikeCompletedStepReference(m[0])) continue;
      const a = Number.parseInt(m[1], 10);
      const b = Number.parseInt(m[2], 10);
      if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
      return { start: Math.min(a, b), end: Math.max(a, b) };
    }
  }
  const singleRe = /(?:接下来|下一步|然后|再|先|首先|当前|现在|本轮|只|仅|next|then|first|now|only)[^\n。；;]{0,50}(?:执行|运行|调用|处理|做|读取|编辑|execute|run|call|do|read|edit)[^\n。；;]{0,30}(?:步骤|step)\s*(\d+)(?!\s*(?:[-–—~至到]|to|through)\s*\d+)/gi;
  let single;
  while ((single = singleRe.exec(text)) !== null) {
    if (looksLikeCompletedStepReference(single[0])) continue;
    const stepNo = Number.parseInt(single[1], 10);
    if (Number.isFinite(stepNo)) return { start: stepNo, end: stepNo };
  }
  return null;
}

function filterPlanTextToActiveStepRange(text) {
  const range = detectActivePlanStepRange(text);
  if (!range) return text;
  const kept = [];
  for (const line of text.split(/\n/)) {
    const stepNos = [
      ...[...line.matchAll(/^\s*(?:[-*]\s*)?(?:(?:步骤|step)\s*)?(\d+)[.)：:]/ig)].map(m => m[1]),
      ...[...line.matchAll(/(?:^|[\s，,。；;])(?:接下来|下一步|然后|再|先|首先|当前|现在|本轮|只|仅|next|then|first|now|only)[^\n。；;]{0,60}(?:步骤|step)\s*(\d+)\s*[.)：:]/ig)].map(m => m[1]),
    ].map(n => Number.parseInt(n, 10)).filter(Number.isFinite);
    if (stepNos.some(stepNo => stepNo >= range.start && stepNo <= range.end)) kept.push(line);
  }
  return kept.join('\n');
}

function extractClaudeCodePlanLines(text, names) {
  const out = [];
  const add = (index, call) => out.push({ ...call, _pos: Number.isFinite(index) ? index : text.length });
  const planText = filterPlanTextToActiveStepRange(text);
  const filePath = '((?:\\.{1,2}/|/)?[A-Za-z0-9_./~@%+=:,\\\\-]+(?:\\.[A-Za-z0-9_\\\\-]+)?)';
  const quotedValue = '["\'`「『]([^"\'`\\n「」『』]{1,1000})["\'`」』]';

  /**
   * 归一化计划行中抽取出的路径或内容，避免把句末标点带入工具参数。
   */
  const cleanPlanValue = (value) => String(value || '')
    .trim()
    .replace(/[.,;:!?。，；：！？]+$/g, '')
    .trim();

  /**
   * 截取命中项所在的计划行，用于判断该行是否只是历史回顾。
   */
  const lineAt = (index) => {
    const start = Math.max(0, planText.lastIndexOf('\n', Math.max(0, index - 1)) + 1);
    const end = planText.indexOf('\n', index);
    return planText.slice(start, end === -1 ? planText.length : end);
  };

  /**
   * 跳过“已完成/已读取/已执行”的历史步骤；“还未完成/需要执行”仍保留。
   */
  const lineLooksAlreadyDone = (line) => {
    const s = String(line || '');
    if (/(?:还未|尚未|未完成|未执行|未读取|未编辑|未写入|还没|没做|需要(?:执行|编辑|写入|创建|读取)|待执行|not yet|needs?|todo)/i.test(s)) {
      return false;
    }
    return /(?:已(?:经)?(?:完成|读取|执行|运行|编辑|写入|创建)|完成(?:，|。|\)|）|$)|done|completed|already (?:done|completed|ran|read|edited|written))/i.test(s);
  };

  /**
   * 清理内容值末尾的计划状态短语，例如 “ - 还未完成”。
   */
  const cleanContentValue = (value) => {
    let v = cleanPlanValue(value)
      .replace(/\s+[-—–]\s*(?:已(?:经)?完成|还未完成|尚未完成|还没做|需要执行|not yet|done|completed).*$/i, '')
      .trim();
    const quotePairs = [['"', '"'], ["'", "'"], ['`', '`'], ['「', '」'], ['『', '』']];
    for (const [left, right] of quotePairs) {
      if (v.startsWith(left) && v.endsWith(right) && v.length >= left.length + right.length) {
        v = v.slice(left.length, -right.length).trim();
        break;
      }
    }
    return v;
  };

  const bashName = preferredName(names, ['Bash', 'shell_exec', 'shell_command']);
  if (bashName) {
    const lineRe = /(?:^|\n)\s*(?:[-*]\s*|\d+[.)]\s*|步骤\s*\d+\s*[:：]\s*)(?:Bash\s*[:：-]\s*)?([^\n。；;]{1,500})/gi;
    let m;
    while ((m = lineRe.exec(planText)) !== null) {
      if (lineLooksAlreadyDone(lineAt(m.index))) continue;
      const command = normalizeNarrativeValue('command', m[1]);
      if (!command || !shellCommandLooksConcrete(command)) continue;
      add(m.index, {
        name: bashName,
        argumentsJson: JSON.stringify({ command }),
        layer: 'narrative',
        confidence: 0.65,
      });
    }
    const bashToolRe = new RegExp(
      `(?:^|\\n)\\s*(?:[-*]\\s*|\\d+[.)]\\s*|步骤\\s*\\d+\\s*[:：]\\s*)?` +
      `(?:Use\\s+)?Bash\\s*(?:tool|工具)?[^\\n。；;]{0,80}?` +
      `(?:execute|run|exec|执行|运行)\\s*[:：]?\\s*([^\\n。；;]{1,500})`,
      'gi',
    );
    while ((m = bashToolRe.exec(planText)) !== null) {
      if (lineLooksAlreadyDone(lineAt(m.index))) continue;
      const command = normalizeNarrativeValue('command', cleanPlanValue(m[1]));
      if (!command || !shellCommandLooksConcrete(command)) continue;
      add(m.index, {
        name: bashName,
        argumentsJson: JSON.stringify({ command }),
        layer: 'narrative',
        confidence: 0.65,
      });
    }
  }

  const editName = preferredName(names, ['Edit']);
  if (editName) {
    const editPatterns = [
      /(?:Edit\s*(?:工具)?|使用\s*Edit\s*(?:工具)?|调用\s*Edit\s*(?:工具)?)[^\n。；;]{0,80}?file_path\s*[:=]\s*([A-Za-z0-9_./-]+)[,\s，]+old_string\s*[:=]\s*["'`「『]?([^"'`「」『』\s，,。；;]+)["'`」』]?[,\s，]+new_string\s*[:=]\s*["'`「『]?([^"'`「」『』\s，,。；;]+)["'`」』]?/gi,
      /(?:把|将)\s*([A-Za-z0-9_./-]+)\s*(?:中|中的|里|里的|内|内的)\s*["'`「『]?([^"'`「」『』\s，。；;]+)["'`」』]?\s*(?:修改为|改为|替换为|变成|=>|->|为)\s*["'`「『]?([^"'`「」『』\s，。；;]+)["'`」』]?/gi,
      /(?:Edit\s*(?:工具)?|使用\s*Edit\s*(?:工具)?|调用\s*Edit\s*(?:工具)?)[^\n。；;]{0,80}?\s*([A-Za-z0-9_./-]+)\s*(?:中|中的|里|里的|内|内的)\s*["'`「『]?([^"'`「」『』\s，。；;]+)["'`」』]?\s*(?:修改为|改为|替换为|变成|=>|->|为)\s*["'`「『]?([^"'`「」『』\s，。；;]+)["'`」』]?/gi,
      /(?:Edit\s*(?:工具|tool)?|使用\s*Edit\s*(?:工具)?|调用\s*Edit\s*(?:工具)?)[^\n。；;]{0,80}?(?:把|将)\s*([A-Za-z0-9_./-]+)\s*(?:中|中的|里|里的|内|内的)\s*(?:唯一(?:的)?\s*)?["'`「『]?([^"'`「」『』\s，。；;]+)["'`」』]?\s*(?:修改为|改为|改成|替换为|变成|=>|->|为)\s*["'`「『]?([^"'`「」『』\s，。；;]+)["'`」』]?/gi,
    ];
    for (const editRe of editPatterns) {
      let m;
      while ((m = editRe.exec(planText)) !== null) {
        if (lineLooksAlreadyDone(lineAt(m.index))) continue;
        add(m.index, {
          name: editName,
          argumentsJson: JSON.stringify({ file_path: normalizeNarrativeValue('file_path', cleanPlanValue(m[1])), old_string: cleanPlanValue(m[2]), new_string: cleanPlanValue(m[3]) }),
          layer: 'narrative',
          confidence: 0.65,
        });
      }
    }
    const englishEditPatterns = [
      new RegExp(`(?:Edit\\s*(?:tool)?|Use\\s+Edit\\s*(?:tool)?)[^\\n。；;]{0,80}?(?:change|replace)[^\\n。；;]{0,80}?${quotedValue}\\s+(?:in|inside|within)\\s+${filePath}\\s+(?:to|with)\\s+${quotedValue}`, 'gi'),
      new RegExp(`(?:Edit\\s*(?:tool)?|Use\\s+Edit\\s*(?:tool)?)[^\\n。；;]{0,80}?(?:replace|change)[^\\n。；;]{0,80}?${quotedValue}\\s+(?:with|to)\\s+${quotedValue}\\s+(?:in|inside|within)\\s+${filePath}`, 'gi'),
    ];
    let m;
    while ((m = englishEditPatterns[0].exec(planText)) !== null) {
      if (lineLooksAlreadyDone(lineAt(m.index))) continue;
      add(m.index, {
        name: editName,
        argumentsJson: JSON.stringify({ file_path: normalizeNarrativeValue('file_path', cleanPlanValue(m[2])), old_string: cleanPlanValue(m[1]), new_string: cleanPlanValue(m[3]) }),
        layer: 'narrative',
        confidence: 0.65,
      });
    }
    while ((m = englishEditPatterns[1].exec(planText)) !== null) {
      if (lineLooksAlreadyDone(lineAt(m.index))) continue;
      add(m.index, {
        name: editName,
        argumentsJson: JSON.stringify({ file_path: normalizeNarrativeValue('file_path', cleanPlanValue(m[3])), old_string: cleanPlanValue(m[1]), new_string: cleanPlanValue(m[2]) }),
        layer: 'narrative',
        confidence: 0.65,
      });
    }
  }

  const writeName = preferredName(names, ['Write']);
  if (writeName) {
    const writePatterns = [
      new RegExp(`(?:Write\\s*(?:工具|tool)?|使用\\s*Write\\s*(?:工具)?|调用\\s*Write\\s*(?:工具)?)[^\\n。；;]{0,80}?file_path\\s*[:=]\\s*${filePath}[^\\n。；;]{0,120}?(?:content|内容)\\s*[:=]\\s*${quotedValue}`, 'gi'),
      new RegExp(`(?:Write\\s*(?:工具|tool)?|使用\\s*Write\\s*(?:工具)?|调用\\s*Write\\s*(?:工具)?)[^\\n。；;]{0,80}?(?:create|write|生成|创建|新建|写入)[^\\n。；;]{0,80}?${filePath}[^\\n。；;]{0,120}?(?:content|内容)(?:\\s*(?:must\\s+be|is|为|是|写(?:为)?|包含))?\\s*[:：]?\\s*${quotedValue}`, 'gi'),
      new RegExp(`(?:创建|新建|生成|写入)[^\\n。；;]{0,60}?${filePath}[^\\n。；;]{0,120}?(?:内容)(?:\\s*(?:为|是|写(?:为)?|包含))?\\s*[:：]?\\s*${quotedValue}`, 'gi'),
      new RegExp(`(?:Write\\s*(?:工具|tool)?|使用\\s*Write\\s*(?:工具)?|调用\\s*Write\\s*(?:工具)?)[^\\n。；;]{0,80}?(?:create|write|生成|创建|新建|写入)[^\\n。；;]{0,80}?${filePath}[^\\n。；;]{0,120}?(?:content|内容)(?:\\s*(?:must\\s+be|is|为|是|写(?:为)?|包含|只写一行))?\\s*[:：]?\\s*([^\\n。；;，,]{1,500})`, 'gi'),
      new RegExp(`(?:创建|新建|生成|写入)[^\\n。；;]{0,60}?${filePath}[^\\n。；;]{0,120}?(?:内容)(?:\\s*(?:为|是|写(?:为)?|包含|只写一行))?\\s*[:：]?\\s*([^\\n。；;，,]{1,500})`, 'gi'),
    ];
    for (const writeRe of writePatterns) {
      let m;
      while ((m = writeRe.exec(planText)) !== null) {
        if (lineLooksAlreadyDone(lineAt(m.index))) continue;
        add(m.index, {
          name: writeName,
          argumentsJson: JSON.stringify({ file_path: normalizeNarrativeValue('file_path', cleanPlanValue(m[1])), content: cleanContentValue(m[2]) }),
          layer: 'narrative',
          confidence: 0.65,
        });
      }
    }
  }

  const readName = preferredName(names, ['Read']);
  if (readName) {
    const readRe = /(?:Read\s*(?:工具)?|使用\s*Read\s*(?:工具)?|调用\s*Read\s*(?:工具)?|读取)\s*(?:file_path\s*[:=]\s*)?([A-Za-z0-9_./-]+\.[A-Za-z0-9_-]+)(?:\s*(?:和|,|，|and)\s*(?:file_path\s*[:=]\s*)?([A-Za-z0-9_./-]+\.[A-Za-z0-9_-]+))?/gi;
    let m;
    while ((m = readRe.exec(planText)) !== null) {
      if (lineLooksAlreadyDone(lineAt(m.index))) continue;
      for (const filePath of [m[1], m[2]].filter(Boolean)) {
        add(m.index, {
          name: readName,
          argumentsJson: JSON.stringify({ file_path: normalizeNarrativeValue('file_path', cleanPlanValue(filePath)) }),
          layer: 'narrative',
          confidence: 0.65,
        });
      }
    }
    const englishReadRe = new RegExp(
      `(?:^|\\n)\\s*(?:[-*]\\s*|\\d+[.)]\\s*|步骤\\s*\\d+\\s*[:：]\\s*)?` +
      `(?:Use\\s+)?Read\\s*(?:tool|工具)?\\s*(?:to\\s+)?read\\s+${filePath}`,
      'gi',
    );
    while ((m = englishReadRe.exec(planText)) !== null) {
      if (lineLooksAlreadyDone(lineAt(m.index))) continue;
      add(m.index, {
        name: readName,
        argumentsJson: JSON.stringify({ file_path: normalizeNarrativeValue('file_path', cleanPlanValue(m[1])) }),
        layer: 'narrative',
        confidence: 0.65,
      });
    }
  }
  return out
    .sort((a, b) => a._pos - b._pos)
    .map(({ _pos, ...tc }) => tc);
}

/**
 * Detect whether the user prompt asked for an action a function could
 * perform. Layer 3 (narrative) only fires when this is true to avoid
 * false-positive tool_call extraction from casual chat.
 */
function userPromptLooksActionable(lastUserText) {
  if (!lastUserText) return false;
  // v2.0.81 (#125): widen to Chinese verbs/nouns so GLM-5.1 / Kimi
  // running with a Chinese system prompt + Chinese user turn still
  // routes through Layer 3.
  if (/\b(?:run|exec|execute|cat|ls|echo|grep|find|read|search|list|invoke|call|fetch|get|fix|edit|write|patch)\b/i.test(lastUserText)) return true;
  if (/\b(?:shell|bash|terminal|command|tool|function|file|path)\b/i.test(lastUserText)) return true;
  if (/(?:运行|执行|读取|查看|列出|查找|搜索|获取|修改|编辑|写入|修复|分析|调用|使用|拉取|下载|找到|看一下|看看|检查)/.test(lastUserText)) return true;
  if (/(?:文件|目录|路径|命令|工具|函数|参数|项目|代码|配置)/.test(lastUserText)) return true;
  return false;
}

/**
 * Detect whether the model's narrative looks like it INTENDED to call
 * a tool but never produced a usable extraction. Used to gate the
 * retry-with-correction loop in chat.js — we only burn an extra
 * cascade round-trip when there's clear tool intent we couldn't
 * recover.
 *
 * Returns one of:
 *   - the matched declared tool name (when the model named it inline)
 *   - the FIRST declared tool name (when the narrative shows clear
 *     action intent + user actionable prompt + an action verb,
 *     even if the model didn't name a specific tool — GLM-5.1 will
 *     say "Let me list the files" without saying "Bash")
 *   - null when there's no usable signal
 *
 * v2.0.82 (#125 — proper translator layer beyond NLU).
 */
export function detectToolIntentInNarrative(text, tools, opts = {}) {
  if (typeof text !== 'string' || !text.trim()) return null;
  // ReDoS/CPU bound (audit #3): this scans the full text once per declared
  // tool name, so a pathologically large model output could drive the
  // per-name regex loop into a polynomial blow-up. Cap the scanned length;
  // NLU recovery is best-effort so operating on a prefix is acceptable.
  if (text.length > 200_000) text = text.slice(0, 200_000);
  if (!Array.isArray(tools) || !tools.length) return null;
  const lastUserText = opts.lastUserText || '';
  if (!userPromptLooksActionable(lastUserText)) return null;
  const { names } = indexTools(tools);
  if (!names.size) return null;
  // Verb forms (English + Chinese) that signal "I'm about to call X".
  const verbPattern = /\b(?:call|invoke|run|use|execute|exec|trigger|fire|going to|will|let me|i'?ll|i'?m going|need to|should)\b|(?:调用|使用|运行|执行|触发|启动|让我|我会|我将|准备|打算|想要|需要|应该)/i;
  if (!verbPattern.test(text)) return null;
  // Action keywords (file ops, search, read, etc.) — these stand in
  // for "the model is talking about USING tools generically".
  const actionVerbPattern = /\b(?:list|show|read|cat|grep|find|search|view|fetch|get|create|write|edit|run|execute|check|inspect|examine|analyz|browse|explore)\b|(?:列出|展示|读取|查看|查找|搜索|获取|拉取|下载|创建|写入|编辑|运行|执行|检查|检视|分析|浏览|探索|看一下|看看)/i;
  // Pass 1: specific tool name in narrative (most precise).
  for (const fn of names) {
    const fnRe = new RegExp(`\\b${escapeRe(fn)}\\b|\\\`${escapeRe(fn)}\\\``);
    if (fnRe.test(text)) return fn;
  }
  // Pass 2: action keyword present (model said "let me list..." but
  // didn't name the tool). Return the first declared tool — caller's
  // correction prompt will name it explicitly so the retry knows
  // which tool to emit.
  if (actionVerbPattern.test(text)) return [...names][0];
  return null;
}

/**
 * Top-level extractor. Returns a deduped, confidence-sorted list of
 * extracted tool_calls. Empty array when nothing is recoverable.
 *
 * Set the `WINDSURFAPI_NLU_RECOVERY=0` env to turn off entirely
 * (default ON).
 */
export function extractIntentFromNarrative(text, tools, opts = {}) {
  if (process.env.WINDSURFAPI_NLU_RECOVERY === '0') return [];
  if (typeof text !== 'string' || !text.trim()) return [];
  // ReDoS/CPU bound (audit #3, see detectToolIntentInNarrative): cap the
  // scanned length before the per-tool-name regex layers run.
  if (text.length > 200_000) text = text.slice(0, 200_000);
  if (!Array.isArray(tools) || !tools.length) return [];
  const lastUserText = opts.lastUserText || '';
  const minConfidence = typeof opts.minConfidence === 'number' ? opts.minConfidence : 0.65;
  // v2.0.78 (audit H-4): structural markers MAY indicate a malformed
  // protocol attempt — Layer 3 narrative around it tends to be
  // descriptive prose, not args. v2.0.79 narrowed the gate after
  // GLM-4.7 e2e probe regressed: GLM emits `markers=bare_json`
  // (because thinking text contains JSON-shaped fragments) AND a
  // legitimate narrate; Layer 3 is exactly what catches the narrate.
  // Now we only skip Layer 3 for `xml_tag` (Claude's tool_use shape)
  // — that's where parser-failure → Layer 3 most often produces
  // false positives. fenced_json / bare_json / openai_native still
  // allow Layer 3 because models emitting those shapes (GLM, Kimi,
  // some GPT) also reliably narrate the call in surrounding prose.
  const markers = Array.isArray(opts.markers) ? opts.markers : [];
  const skipLayer3 = markers.includes('xml_tag') && !markers.includes('natural_lang');

  const { names, primaryParam } = indexTools(tools);
  if (!names.size) return [];

  const actionableNarrative = !skipLayer3 && userPromptLooksActionable(lastUserText);
  const scopedNarrativeText = actionableNarrative ? filterPlanTextToActiveStepRange(text) : text;
  const all = [
    ...extractLayer1(text, names),
    ...extractLayer2(text, names, primaryParam),
    ...(actionableNarrative ? extractClaudeCodePlanLines(text, names) : []),
    ...(actionableNarrative ? extractLayer3(scopedNarrativeText, names, primaryParam) : []),
  ];
  if (!all.length) return [];

  // Dedupe by (name, argumentsJson). Keep the highest-confidence pick.
  const byKey = new Map();
  for (const tc of all) {
    if (tc.confidence < minConfidence) continue;
    const key = `${tc.name}::${tc.argumentsJson}`;
    const existing = byKey.get(key);
    if (!existing || tc.confidence > existing.confidence) byKey.set(key, tc);
  }
  const recovered = [...byKey.values()].sort((a, b) => b.confidence - a.confidence);
  if (recovered.length) {
    log.info(`NLU recovery: extracted ${recovered.length} tool_call(s) from narrative — ${recovered.map(t => `${t.name}@${t.layer}/${t.confidence.toFixed(2)}`).join(', ')}${skipLayer3 ? ' (layer3-skipped: structural markers seen)' : ''}`);
  }
  return recovered;
}
